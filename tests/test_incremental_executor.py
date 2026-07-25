#!/usr/bin/env python3
"""Tests for the incremental decode protocol in the offline executor (K2).

Pure CPU: every engine component around ``OfflineExecutor`` is a small fake,
and the strategy is a synthetic ``incremental=True`` implementation that emits
one token per batched decoder step.  This pins the executor-side contract —
bounded steps per tick, the pending pool lifecycle, admission gating, abort —
independent of any real model (Whisper etc. plug into the same seam).
"""

import time
from typing import Dict, List

import pytest
import torch

from oasr.engine.decode.base import DecodeStrategy
from oasr.engine.executor.offline import OfflineExecutor
from oasr.engine.generation import StepBudget
from oasr.engine.request import Request, RequestOutput, RequestState


class FakeIncrementalStrategy(DecodeStrategy):
    """Emits one token per advance step; request ``r`` finishes after
    ``target_lens[r]`` tokens.  Counts batched steps for budget assertions."""

    decode_type = "aed"
    consumes = "hidden"
    incremental = True

    def __init__(self, target_lens: Dict[str, int]):
        self.target_lens = target_lens
        self.states: Dict[str, List[int]] = {}
        self.freed: List[str] = []
        self.steps_per_tick: List[int] = []
        #: Simulated per-batched-step cost, for wall-clock budget assertions.
        self.step_delay_s: float = 0.0

    # -- incremental protocol ------------------------------------------
    def begin_offline(self, requests, enc_out, enc_lengths):
        for req in requests:
            self.states[req.request_id] = []

    def advance(self, budget: StepBudget) -> List[RequestOutput]:
        outputs: List[RequestOutput] = []
        steps = 0
        while self.states and budget.take():
            steps += 1
            if self.step_delay_s:
                time.sleep(self.step_delay_s)
            for rid in list(self.states):
                self.states[rid].append(len(self.states[rid]) + 1)
                if len(self.states[rid]) >= self.target_lens[rid]:
                    toks = self.states.pop(rid)
                    outputs.append(
                        RequestOutput(
                            request_id=rid,
                            text=" ".join(map(str, toks)),
                            tokens=[toks],
                            finished=True,
                        )
                    )
        self.steps_per_tick.append(steps)
        return outputs

    def has_pending(self) -> bool:
        return bool(self.states)

    # -- unused abstract surface ----------------------------------------
    def decode_offline(self, enc_out, enc_lengths):
        raise AssertionError("incremental strategy must not be one-shot decoded")

    def decode_streaming_batch(self, requests, enc_out_map):
        raise NotImplementedError

    def decode_streaming_chunk(self, request, enc_out):
        raise NotImplementedError

    def finalize(self, request):
        raise NotImplementedError

    def free_session(self, request):
        self.states.pop(request.request_id, None)
        self.freed.append(request.request_id)


class FakeScheduler:
    def __init__(self):
        self.waiting: List[Request] = []

    def add_request(self, req):
        self.waiting.append(req)

    def schedule_offline(self):
        batch, self.waiting = self.waiting, []
        return batch

    def split_offline_batch(self, batch):
        return [batch], None

    @property
    def num_waiting_offline(self):
        return len(self.waiting)

    def abort_request(self, request_id):
        self.waiting = [r for r in self.waiting if r.request_id != request_id]

    def find_request(self, request_id):
        return next((r for r in self.waiting if r.request_id == request_id), None)


class FakeInputProcessor:
    def prepare_offline(self, req):
        pass

    def collate(self, chunk):
        B = len(chunk)
        return torch.zeros(B, 4, 8), torch.full((B,), 4, dtype=torch.int32)


class FakeModelRunner:
    def encode_offline(self, features, lengths):
        return torch.zeros(features.shape[0], 2, 8), lengths

    def forward_offline(self, features, lengths):
        raise AssertionError("consumes='hidden' must route to encode_offline")


class FakeOutputProcessor:
    def __init__(self, strategy):
        self.strategy = strategy

    def decode_offline(self, enc_out, enc_lengths):
        return self.strategy.decode_offline(enc_out, enc_lengths)

    def fill_nbest_texts(self, request, output):
        return None


def _make_executor(target_lens, *, steps_per_tick=4, slots=8):
    strat = FakeIncrementalStrategy(target_lens)
    ex = OfflineExecutor(
        scheduler=FakeScheduler(),
        input_processor=FakeInputProcessor(),
        model_runner=FakeModelRunner(),
        output_processor=FakeOutputProcessor(strat),
        device=torch.device("cpu"),
        decode_steps_per_tick=steps_per_tick,
        max_decode_slots=slots,
    )
    return ex, strat


def _admit(ex, rid):
    req = Request(audio=torch.zeros(64), request_id=rid, streaming=False)
    ex.admit(req)
    return req


class TestIncrementalLifecycle:
    def test_prefill_parks_then_advance_finishes(self):
        ex, strat = _make_executor({"a": 3}, steps_per_tick=8)
        req = _admit(ex, "a")
        outs = ex.step()  # tick 1: prefill only (advance had nothing pending)
        assert outs == []
        assert req.state == RequestState.RUNNING
        assert ex.num_running() == 1 and ex.has_pending()

        outs = ex.step()  # tick 2: 3 steps finish it (budget 8)
        assert [o.request_id for o in outs if o.finished] == ["a"]
        assert outs[0].tokens == [[1, 2, 3]]
        assert req.state == RequestState.FINISHED
        assert ex.num_running() == 0 and not ex.has_pending()

    def test_budget_bounds_steps_per_tick(self):
        ex, strat = _make_executor({"a": 10}, steps_per_tick=4)
        _admit(ex, "a")
        ex.step()  # prefill
        for _ in range(2):
            assert ex.step() == []  # 4 + 4 steps, not finished yet
        outs = ex.step()  # 2 remaining steps
        assert [o.request_id for o in outs] == ["a"]
        assert strat.steps_per_tick == [4, 4, 2]

    def test_continuous_batching_across_requests(self):
        """A request admitted later joins the advance loop mid-flight.

        It joins on the *next* tick, not the one that spent its decode budget:
        prefill is unbudgeted, so stacking it on top of a full budget would make
        the real tick bound ``budget + prefill``.  See
        :class:`TestTickBudget.test_admission_deferred_when_budget_spent`.
        """
        ex, strat = _make_executor({"a": 8, "b": 2}, steps_per_tick=2)
        _admit(ex, "a")
        ex.step()  # prefill a
        ex.step()  # a: 2/8 — budget spent
        _admit(ex, "b")
        outs = ex.step()  # a: 4/8; budget spent again, so b's prefill waits
        assert outs == [] and ex.num_running() == 1

        # b joins on a later tick and then advances in the same batched loop as a.
        finals = []
        for _ in range(20):
            if not ex.has_pending():
                break
            finals.extend(o for o in ex.step() if o.finished)
        assert sorted(o.request_id for o in finals) == ["a", "b"]
        assert ex.num_running() == 0

    def test_run_drives_to_completion(self):
        """engine.run()-style loop: step until has_pending() clears."""
        ex, strat = _make_executor({"a": 5, "b": 9}, steps_per_tick=3)
        ra, rb = _admit(ex, "a"), _admit(ex, "b")
        finals = []
        for _ in range(50):
            if not ex.has_pending():
                break
            finals.extend(o for o in ex.step() if o.finished)
        assert sorted(o.request_id for o in finals) == ["a", "b"]
        assert ra.state == rb.state == RequestState.FINISHED
        assert {o.request_id: o.tokens[0] for o in finals}["b"] == list(range(1, 10))


class TestAdmissionGating:
    def test_full_slots_pause_admission(self):
        ex, strat = _make_executor({"a": 20, "b": 1}, steps_per_tick=1, slots=1)
        _admit(ex, "a")
        ex.step()  # prefill a → pool full
        _admit(ex, "b")
        outs = ex.step()  # advance a only; b must NOT be admitted (slots=1)
        assert outs == []
        assert ex.num_running() == 1 and ex.num_waiting() == 1
        assert "b" not in strat.states

    def test_slot_frees_on_finish(self):
        ex, strat = _make_executor({"a": 2, "b": 1}, steps_per_tick=4, slots=1)
        _admit(ex, "a")
        ex.step()
        _admit(ex, "b")
        outs = ex.step()  # a finishes → slot frees → b prefills same tick
        assert [o.request_id for o in outs] == ["a"]
        assert "b" in strat.states
        outs = ex.step()
        assert [o.request_id for o in outs] == ["b"]


class TestAbort:
    def test_abort_pending_frees_strategy_state(self):
        ex, strat = _make_executor({"a": 50}, steps_per_tick=1)
        req = _admit(ex, "a")
        ex.step()  # prefill
        assert ex.find_request("a") is req
        ex.abort("a")
        assert strat.freed == ["a"]
        assert req.state == RequestState.FINISHED
        assert not ex.has_pending() and ex.num_running() == 0

    def test_abort_waiting_untouched_by_pool(self):
        ex, strat = _make_executor({"a": 5})
        _admit(ex, "a")
        ex.abort("a")  # still in the scheduler queue
        assert strat.freed == []
        assert not ex.has_pending()


class TestShutdown:
    def test_shutdown_releases_pending_decode_state(self):
        """Teardown must free parked AR sessions, not leave them to the GC."""
        ex, strat = _make_executor({"a": 50, "b": 50}, steps_per_tick=1)
        ra, rb = _admit(ex, "a"), _admit(ex, "b")
        ex.step()  # prefill both; neither finishes
        assert ex.num_running() == 2

        ex.shutdown()

        assert sorted(strat.freed) == ["a", "b"]
        assert ra.state == RequestState.FINISHED and rb.state == RequestState.FINISHED
        assert ex.num_running() == 0

    def test_shutdown_is_a_noop_without_pending(self):
        ex, strat = _make_executor({"a": 1})
        ex.shutdown()
        assert strat.freed == []


class TestOneShotUnaffected:
    def test_one_shot_strategy_never_parks(self):
        class OneShot(DecodeStrategy):
            decode_type = "ctc"
            consumes = "log_probs"

            def decode_offline(self, enc_out, enc_lengths):
                return [
                    RequestOutput(request_id="", text="x", tokens=[], finished=True)
                    for _ in range(enc_out.shape[0])
                ]

            def decode_streaming_batch(self, requests, enc_out_map):
                raise NotImplementedError

            def decode_streaming_chunk(self, request, enc_out):
                raise NotImplementedError

            def finalize(self, request):
                raise NotImplementedError

        class LogProbRunner(FakeModelRunner):
            def forward_offline(self, features, lengths):
                return torch.zeros(features.shape[0], 2, 5), lengths

        ex = OfflineExecutor(
            scheduler=FakeScheduler(),
            input_processor=FakeInputProcessor(),
            model_runner=LogProbRunner(),
            output_processor=FakeOutputProcessor(OneShot()),
            device=torch.device("cpu"),
        )
        req = _admit(ex, "a")
        outs = ex.step()
        assert len(outs) == 1 and outs[0].finished and outs[0].request_id == "a"
        assert req.state == RequestState.FINISHED
        assert ex.num_running() == 0 and not ex.has_pending()


class TestStepBudget:
    def test_take_semantics(self):
        b = StepBudget(max_steps=2)
        assert b.take() and b.take() and not b.take()
        assert b.exhausted() and b.remaining == 0
        assert b.used == 2

    def test_no_deadline_by_default(self):
        b = StepBudget(max_steps=4)
        assert b.deadline_s is None
        assert not b.out_of_time()

    def test_deadline_stops_further_steps(self):
        """The wall-clock limit binds even when steps remain.

        A step count alone does not bound tick *time* — one decoder step is
        ~1.5 ms on whisper-tiny and ~18 ms on a 7B, so a fixed 32-step tick spans
        ~50 ms to ~580 ms across models.
        """
        b = StepBudget.for_tick(max_steps=1000, max_tick_ms=5.0)
        assert b.take()  # first step is always granted
        time.sleep(0.02)  # blow through the 5 ms deadline
        assert b.out_of_time()
        assert not b.take()
        assert b.exhausted()
        assert b.remaining > 0  # steps left; time is what ran out

    def test_first_step_always_granted(self):
        """Progress beats holding a deadline a single step cannot fit inside."""
        b = StepBudget.for_tick(max_steps=8, max_tick_ms=0.0001)
        time.sleep(0.005)
        assert b.take()
        assert not b.take()

    def test_for_tick_without_deadline(self):
        b = StepBudget.for_tick(max_steps=3, max_tick_ms=0.0)
        assert b.deadline_s is None
        assert b.take() and b.take() and b.take() and not b.take()


class TestTickBudget:
    """Tick-level composition of the budget with admission (C1 + C4)."""

    def test_admission_deferred_when_budget_spent(self):
        ex, strat = _make_executor({"a": 100}, steps_per_tick=2, slots=8)
        _admit(ex, "a")
        ex.step()  # prefill a
        _admit(ex, "b")
        ex.step()  # spends both steps on a → b's prefill is deferred
        assert ex.num_running() == 1

    def test_admission_forced_after_repeated_skips(self):
        """A saturated decode pool must not starve admission indefinitely."""
        from oasr.engine.executor.offline import _MAX_SKIPPED_ADMITS

        ex, strat = _make_executor({"a": 10_000}, steps_per_tick=1, slots=8)
        _admit(ex, "a")
        ex.step()  # prefill a
        _admit(ex, "b")
        for _ in range(_MAX_SKIPPED_ADMITS):
            ex.step()
            assert ex.num_running() == 1, "b should still be deferred"
        ex.step()  # the forced-admission tick
        assert ex.num_running() == 2

    def test_wall_clock_budget_bounds_a_tick(self):
        """With a slow strategy the tick stops on time, not on the step count."""
        ex, strat = _make_executor({"a": 10_000}, steps_per_tick=1000, slots=8)
        strat.step_delay_s = 0.004  # ~4 ms per batched step
        ex._max_tick_ms = 20.0  # noqa: SLF001 - exercising the executor's budget
        _admit(ex, "a")
        ex.step()  # prefill
        t0 = time.perf_counter()
        ex.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Deadline stops *starting* steps, so the bound is deadline + one step.
        assert elapsed_ms < 20.0 + 8.0, f"tick ran {elapsed_ms:.1f}ms"
        assert strat.steps_per_tick[-1] < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
