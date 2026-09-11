#!/usr/bin/env python3
"""``oasr/engine/executor/`` -- the offline and streaming executors.

Failure isolation used to be its own file with its own stack of fake
collaborators, testing the same two executors from the other side: one bad
request must not take down its peers. Two files, two sets of fakes, one pair
of classes.

The ``CacheConfig`` pool invariant that lived there moved to
``test_memory.py``, next to the sizing arithmetic that derives the pool.

Pure CPU, no checkpoints — every engine component is a small fake.

* ``TestIncrementalLifecycle`` / ``TestAdmissionGating`` / ``TestAbort`` /
  ``TestShutdown`` / ``TestStepBudget`` / ``TestTickBudget`` pin the
  **executor** side: bounded work per tick (step cap *and* wall-clock deadline),
  the pending-pool lifecycle, admission gating and deferral, abort, teardown.
* ``TestIncrementalArBase`` pins the **strategy** side — the shared
  :class:`~oasr.engine.decode.incremental.IncrementalArStrategy` that AED and the
  speech-LLM sit on, driven through a fake decoder so a third AR family can be
  written against a tested contract (two hooks) rather than by copying a sibling.
"""

import time
from collections import deque
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from oasr.engine.decode.base import DecodeStrategy
from oasr.engine.executor.offline import _MAX_SKIPPED_ADMITS, OfflineExecutor
from oasr.engine.executor.streaming import StreamingExecutor
from oasr.engine.generation import StepBudget
from oasr.engine.metrics import build_metrics
from oasr.engine.request import Request, RequestOutput, RequestState


def _fake_detok(render=lambda ids: " ".join(map(str, ids))):
    """A stand-in Detokenizer covering the *whole* contract, not just decode.

    ``IncrementalArStrategy`` decodes partials incrementally (T3), so a fake
    carrying only ``detokenize`` no longer satisfies the surface it is
    substituted for.  Implementing both here — over the same ``render`` — keeps
    the fake honest instead of narrowing the production path to whatever the
    fake happens to provide.
    """

    def incremental(new_ids, state):
        ids = state.setdefault("ids", [])
        ids.extend(int(i) for i in new_ids)
        full = render(ids)
        prev = state.get("text", "")
        state["text"] = full
        return full[len(prev) :] if full.startswith(prev) else full

    return SimpleNamespace(
        detokenize=render,
        new_state=lambda: {"ids": [], "text": ""},
        detokenize_incremental=incremental,
    )


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

    def schedule_offline(self, limit=None):
        cap = len(self.waiting) if limit is None else max(0, int(limit))
        batch, self.waiting = self.waiting[:cap], self.waiting[cap:]
        return batch

    def split_offline_batch(self, batch):
        return [batch], None

    @property
    def num_waiting_offline(self):
        return len(self.waiting)

    def oldest_offline_wait(self):
        if not self.waiting:
            return None
        return max(r.waited_for for r in self.waiting)

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


def _make_executor(
    target_lens, *, steps_per_tick=4, slots=8, admit_window_ms=0.0, max_batch_size=32
):
    strat = FakeIncrementalStrategy(target_lens)
    ex = OfflineExecutor(
        scheduler=FakeScheduler(),
        input_processor=FakeInputProcessor(),
        model_runner=FakeModelRunner(),
        output_processor=FakeOutputProcessor(strat),
        device=torch.device("cpu"),
        decode_steps_per_tick=steps_per_tick,
        max_decode_slots=slots,
        decode_admit_window_ms=admit_window_ms,
        max_batch_size=max_batch_size,
    )
    return ex, strat


class OneShotStrategy(DecodeStrategy):
    """Frame-synchronous stand-in: decodes a whole batch in one tick, never parks."""

    decode_type = "ctc"
    consumes = "log_probs"

    def decode_offline(self, enc_out, enc_lengths):
        return [
            RequestOutput(request_id="", text="x", tokens=[[1]], finished=True)
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


def _stub_ctc_model():
    """Minimal object satisfying CAPABILITIES["ctc"] (head + forward_offline)."""
    return SimpleNamespace(head=lambda *a: None, forward_offline=lambda *a: None)


def _make_one_shot_executor(**kwargs):
    """Executor driving a frame-synchronous strategy (no pending pool)."""
    strat = OneShotStrategy(
        SimpleNamespace(max_new_tokens=8),
        _fake_detok(lambda ids: ""),
        _stub_ctc_model(),
    )
    ex = OfflineExecutor(
        scheduler=FakeScheduler(),
        input_processor=FakeInputProcessor(),
        model_runner=LogProbRunner(),
        output_processor=FakeOutputProcessor(strat),
        device=torch.device("cpu"),
        **kwargs,
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
        ex, _ = _make_one_shot_executor()
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


# ---------------------------------------------------------------------------
# Shared incremental-AR extension seam
# ---------------------------------------------------------------------------


class _FakeArDecoder:
    """Minimal batched incremental decoder surface (prefill / step / select).

    Emits token ``id = row_seed + position`` so per-row token streams are
    distinguishable and row/state alignment is checkable after compaction.
    """

    def __init__(self, vocab: int = 32):
        self.vocab = vocab
        self.select_calls: list[list[int]] = []

    def _logits(self, seeds: torch.Tensor, pos: int) -> torch.Tensor:
        out = torch.full((seeds.numel(), self.vocab), -1e4)
        for row, seed in enumerate(seeds.tolist()):
            out[row, (int(seed) + pos) % self.vocab] = 10.0
        return out

    def prefill(self, enc_out, valid_or_prompt, capacity=None):
        del capacity
        B = enc_out.size(0)
        seeds = torch.arange(1, B + 1)
        return self._logits(seeds, 0), {"seeds": seeds, "pos": 1}

    def step(self, tokens, state):
        # Logits for the *current* position, then advance — so row ``seed``
        # emits seed, seed+1, seed+2, ... with no gap.
        logits = self._logits(state["seeds"], state["pos"])
        return logits, {"seeds": state["seeds"], "pos": state["pos"] + 1}

    def select(self, state, keep):
        self.select_calls.append(keep.tolist())
        return {"seeds": state["seeds"].index_select(0, keep.cpu()), "pos": state["pos"]}


def _ar_strategy(*, partials: bool, eos_token: int = -1, max_new_tokens: int = 4):
    """A minimal IncrementalArStrategy subclass over the fake decoder."""
    from oasr.engine.decode.incremental import IncrementalArStrategy, Prefill

    class _Strategy(IncrementalArStrategy):
        decode_type = "fake_ar"
        emit_partials = partials

        def _prefill(self, requests, enc_out, enc_lengths):
            logits, state = self._decoder().prefill(enc_out, None)
            return Prefill(
                state=state,
                logits=logits,
                max_new=[self._row_cap(r, 999) for r in requests],
            )

        def _is_eos(self, token):
            return token == eos_token

    decoder = _FakeArDecoder()
    model = SimpleNamespace(decoder=decoder)
    detok = _fake_detok()
    cfg = SimpleNamespace(max_new_tokens=max_new_tokens)
    return _Strategy(cfg, detok, model), decoder


class TestIncrementalArBase:
    """The shared base must be usable from the two required hooks alone."""

    def _requests(self, n):
        return [Request(audio=None, request_id=f"r{i}", streaming=False) for i in range(n)]

    def _drive(self, strat, reqs, steps_per_tick=8, max_ticks=20):
        strat.begin_offline(reqs, torch.zeros(len(reqs), 2, 4), torch.tensor([2] * len(reqs)))
        finals, partials = {}, {}
        for _ in range(max_ticks):
            if not strat.has_pending():
                break
            for out in strat.advance(StepBudget(max_steps=steps_per_tick)):
                if out.finished:
                    finals[out.request_id] = out
                else:
                    partials.setdefault(out.request_id, []).append(len(out.tokens[0]))
        return finals, partials

    def test_length_cap_finishes_every_row(self):
        strat, _ = _ar_strategy(partials=False, max_new_tokens=3)
        reqs = self._requests(3)
        finals, partials = self._drive(strat, reqs)
        assert sorted(finals) == ["r0", "r1", "r2"]
        assert all(len(f.tokens[0]) == 3 for f in finals.values())
        assert all(f.finish_reason == "length" for f in finals.values())
        assert partials == {}, "emit_partials=False must emit finals only"
        assert not strat.has_pending()

    def test_partials_when_enabled(self):
        strat, _ = _ar_strategy(partials=True, max_new_tokens=4)
        finals, partials = self._drive(strat, self._requests(2), steps_per_tick=1)
        assert sorted(partials) == ["r0", "r1"]
        # One partial per tick per active row, monotonically growing.
        for lens in partials.values():
            assert lens == sorted(lens) and lens[0] >= 1
        assert sorted(finals) == ["r0", "r1"]

    def test_eos_stops_a_row_without_emitting_the_token(self):
        # Row 0 emits ids 1,2,3...; make id 2 the EOS so r0 stops after one token.
        strat, _ = _ar_strategy(partials=False, eos_token=2, max_new_tokens=9)
        finals, _ = self._drive(strat, self._requests(2))
        assert finals["r0"].finish_reason == "stop"
        assert finals["r0"].tokens[0] == [1], "EOS itself must not be emitted"

    def test_rows_stay_aligned_after_compaction(self):
        """A row leaving must compact host bookkeeping and decoder state together.

        Row ``i`` emits ``i+1, i+2, ...``; with EOS = 2 the three rows retire in
        three different ways, so any row/state misalignment shows up as a wrong
        token stream rather than a crash: r1 hits EOS immediately (empty), r0 hits
        it on its second token, r2 never does and runs to the cap.
        """
        strat, decoder = _ar_strategy(partials=False, eos_token=2, max_new_tokens=4)
        finals, _ = self._drive(strat, self._requests(3))
        assert len(finals) == 3
        assert decoder.select_calls, "select must be called when a row retires"
        assert finals["r1"].tokens[0] == [] and finals["r1"].finish_reason == "stop"
        assert finals["r0"].tokens[0] == [1] and finals["r0"].finish_reason == "stop"
        assert finals["r2"].tokens[0] == [3, 4, 5, 6]
        assert finals["r2"].finish_reason == "length"

    def test_free_session_drops_one_row(self):
        strat, decoder = _ar_strategy(partials=False, max_new_tokens=50)
        reqs = self._requests(3)
        strat.begin_offline(reqs, torch.zeros(3, 2, 4), torch.tensor([2, 2, 2]))
        strat.advance(StepBudget(max_steps=1))
        strat.free_session(reqs[1])
        group = strat._groups[0]  # noqa: SLF001 - asserting internal alignment
        assert [r.request_id for r in group.requests] == ["r0", "r2"]
        assert group.last_logits.size(0) == 2
        assert decoder.select_calls[-1] == [0, 2]

    def test_non_applicable_surfaces_raise(self):
        strat, _ = _ar_strategy(partials=False)
        for call in (
            lambda: strat.decode_offline(torch.zeros(1, 2, 4), torch.tensor([2])),
            lambda: strat.decode_streaming_batch([], {}),
            lambda: strat.decode_streaming_chunk(None, torch.zeros(1, 2, 4)),
            lambda: strat.finalize(None),
        ):
            with pytest.raises(NotImplementedError, match="fake_ar"):
                call()


class TestArAdmissionWindow:
    """Coalescing thin arrivals into one decode batch (C2).

    An AR decoder step is weight-read bound, so its cost barely depends on how
    many rows it carries: two decode groups cost roughly twice one group of the
    same total rows.  Measured on Qwen2-Audio-7B (4 utterances, 124 tokens),
    arriving together took 922 ms vs 1614 ms arriving one per tick — identical
    work.  Groups cannot be merged afterwards (both decoder surfaces keep a
    shared scalar generation offset), so admission is where this is fixed.
    """

    def test_window_holds_a_thin_batch(self):
        ex, strat = _make_executor({"a": 4, "b": 4}, admit_window_ms=10_000.0, max_batch_size=8)
        _admit(ex, "a")
        # The window has not elapsed and the queue is far from max_batch_size,
        # so nothing is prefilled yet.
        assert ex.step() == []
        assert ex.num_running() == 0
        assert ex.num_waiting() == 1

    def test_window_releases_once_the_batch_is_wide(self):
        """Reaching max_batch_size releases immediately — no point waiting."""
        ex, strat = _make_executor({"a": 4, "b": 4}, admit_window_ms=10_000.0, max_batch_size=2)
        _admit(ex, "a")
        assert ex.step() == [] and ex.num_running() == 0  # 1 < 2, held
        _admit(ex, "b")
        ex.step()  # 2 >= 2 → prefilled together, as ONE group
        assert ex.num_running() == 2

    def test_window_releases_when_it_expires(self):
        ex, strat = _make_executor({"a": 3}, admit_window_ms=0.001, max_batch_size=8)
        _admit(ex, "a")
        time.sleep(0.005)  # blow through the 1 µs window
        ex.step()
        assert ex.num_running() == 1

    def test_disabled_by_default(self):
        """Zero window = today's behaviour: prefill the first arrival at once."""
        ex, strat = _make_executor({"a": 3}, max_batch_size=8)
        _admit(ex, "a")
        ex.step()
        assert ex.num_running() == 1

    def test_inert_for_one_shot_strategies(self):
        """A frame-synchronous strategy must never be held back by the window.

        Only label-synchronous decoding pays the per-group penalty the window
        exists to avoid; CTC / transducer / rescoring decode a batch in one shot,
        so holding them back would be pure added latency.
        """

        # A window that would stall an AR strategy indefinitely.
        ex, _ = _make_one_shot_executor(decode_admit_window_ms=10_000.0, max_batch_size=8)
        _admit(ex, "a")
        outs = ex.step()  # admitted and finalised despite the window
        assert [o.request_id for o in outs] == ["a"]


class TestDecodeSlotCap:
    """``max_decode_slots`` must be a hard cap, not a soft gate (C3).

    ``_admission_open`` only answers "is there *a* free slot".  Without a limit on
    the selection itself, a tick with one slot free still pulled a full
    ``max_batch_size`` batch and prefilled all of it — overshooting the cap by up
    to ``max_batch_size - 1`` requests' worth of preallocated decoder KV.  That is
    an OOM path, not a slowdown.
    """

    def test_batch_is_capped_at_the_free_slots(self):
        ex, strat = _make_executor({f"r{i}": 50 for i in range(6)}, slots=4, steps_per_tick=1)
        for i in range(6):
            _admit(ex, f"r{i}")
        ex.step()  # first tick: 4 slots free → prefill exactly 4
        assert ex.num_running() == 4
        assert ex.num_waiting() == 2, "the surplus must stay queued, not be prefilled"

    def test_partially_full_pool_admits_only_the_remainder(self):
        ex, strat = _make_executor({f"r{i}": 50 for i in range(5)}, slots=3, steps_per_tick=1)
        _admit(ex, "r0")
        ex.step()  # 1 in flight, 2 slots left
        for i in (1, 2, 3, 4):
            _admit(ex, f"r{i}")
        # Advance until a tick admits again (budget-spent ticks defer prefill).
        for _ in range(_MAX_SKIPPED_ADMITS + 2):
            ex.step()
            if ex.num_running() > 1:
                break
        assert ex.num_running() == 3, f"pool exceeded max_decode_slots: {ex.num_running()}"

    def test_unlimited_slots_are_bounded_by_max_batch_size_only(self):
        ex, strat = _make_executor({f"r{i}": 2 for i in range(4)}, slots=None, steps_per_tick=8)
        for i in range(4):
            _admit(ex, f"r{i}")
        ex.step()
        assert ex.num_running() == 4  # no slot cap → the whole batch prefills

    def test_one_shot_strategies_are_not_slot_limited(self):
        """A frame-synchronous family finalises within its tick and holds no slot."""
        ex, _ = _make_one_shot_executor(max_decode_slots=1)
        assert ex._admission_limit() is None  # noqa: SLF001


class TestPrefillRejection:
    """A prefill OOM must reject its own batch, not the whole tick (C3)."""

    def test_oom_during_prefill_rejects_the_batch(self):
        ex, strat = _make_executor({"a": 5, "b": 5}, steps_per_tick=4)

        def _boom(requests, enc_out, enc_lengths):
            raise torch.cuda.OutOfMemoryError("simulated")

        strat.begin_offline = _boom
        ra, rb = _admit(ex, "a"), _admit(ex, "b")
        outs = ex.step()

        assert sorted(o.request_id for o in outs) == ["a", "b"]
        assert all(o.finished and o.finish_reason == "error" and o.text == "" for o in outs)
        assert ra.state == RequestState.FINISHED and rb.state == RequestState.FINISHED
        # Nothing parked, so the next tick is clean rather than re-raising.
        assert ex.num_running() == 0 and not ex.has_pending()
        assert ex.step() == []


class TestDecodeKvByteBudget:
    """C3: admission must bound decoder-KV **bytes**, not just request count.

    A row's footprint is ``(prompt + max_new_tokens) * per-token rate`` and
    prefill preallocates all of it, so N slots of 30 s utterances cost far more
    than N slots of 2 s ones.  The slot cap alone therefore does not bound
    memory — which is an OOM path, not a slowdown.
    """

    def _executor(self, budget_gib, per_row_bytes, pending=0):
        from oasr.engine.executor.offline import OfflineExecutor

        strategy = SimpleNamespace(
            incremental=True,
            kv_bytes_per_row=lambda: per_row_bytes,
            has_pending=lambda: False,
        )
        ex = OfflineExecutor.__new__(OfflineExecutor)
        ex._op = SimpleNamespace(strategy=strategy)
        ex._max_decode_slots = None
        ex._decode_kv_budget_gib = budget_gib
        ex._pending = {f"r{i}": None for i in range(pending)}
        return ex

    def test_budget_caps_rows(self):
        gib = 1024**3
        ex = self._executor(budget_gib=1.0, per_row_bytes=gib // 4)
        assert ex._admission_limit() == 4

    def test_in_flight_rows_are_charged(self):
        gib = 1024**3
        ex = self._executor(budget_gib=1.0, per_row_bytes=gib // 4, pending=3)
        assert ex._admission_limit() == 1

    def test_a_full_budget_admits_nothing(self):
        gib = 1024**3
        ex = self._executor(budget_gib=1.0, per_row_bytes=gib // 2, pending=2)
        assert ex._admission_limit() == 0

    def test_disabled_budget_is_unlimited(self):
        ex = self._executor(budget_gib=None, per_row_bytes=1024)
        assert ex._admission_limit() is None

    def test_unmeasurable_model_is_not_throttled(self):
        """A model that declares no per-row footprint keeps slot-only behaviour.

        Guessing a footprint would silently reduce throughput on every model
        that has not declared ``decoder_cache_spec``.
        """
        ex = self._executor(budget_gib=1.0, per_row_bytes=None)
        assert ex._admission_limit() is None

    def test_the_tighter_of_slots_and_bytes_wins(self):
        gib = 1024**3
        ex = self._executor(budget_gib=1.0, per_row_bytes=gib // 8)
        ex._max_decode_slots = 3
        assert ex._admission_limit() == 3, "slot cap should bind here"
        ex._max_decode_slots = 32
        assert ex._admission_limit() == 8, "byte budget should bind here"

    def test_one_shot_families_are_unaffected(self):
        ex = self._executor(budget_gib=1.0, per_row_bytes=1024)
        ex._op.strategy.incremental = False
        assert ex._admission_limit() is None


# ---------------------------------------------------------------------------
# Failure isolation: one bad request must not take down its peers
#
# Offline retries singly, so a poisoned request costs only itself; streaming
# fails the whole cohort, because a partially-committed streaming step risks
# a double commit. Both halves are pinned here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The KV-pool invariant
# ---------------------------------------------------------------------------


def _request(rid: str) -> Request:
    req = Request(request_id=rid, streaming=False)
    req.state = RequestState.RUNNING
    return req


class _Poisoned:
    """Model runner whose forward raises for one request id."""

    def __init__(self, bad_id: str, exc: Exception):
        self.bad_id = bad_id
        self.exc = exc
        self.chunks_seen: list[list[str]] = []

    def forward_offline(self, features, lengths):
        ids = list(features)  # the fake collate hands us the id list
        self.chunks_seen.append(ids)
        if self.bad_id in ids:
            raise self.exc
        return SimpleNamespace(ids=ids), lengths


def _offline_executor(runner) -> OfflineExecutor:
    """An ``OfflineExecutor`` with only what ``run()`` touches.

    ``__init__`` wants a scheduler, an input processor and a real device; the
    isolation path needs none of them.
    """
    ex = OfflineExecutor.__new__(OfflineExecutor)
    ex._scheduler = SimpleNamespace(
        split_offline_batch=lambda batch: ([batch], None),
    )
    ex._mr = runner
    ex._enable_packing = False
    ex._pending = {}
    ex._op = SimpleNamespace(
        strategy=SimpleNamespace(incremental=False, consumes="log_probs"),
        # ``requests`` rides along for the families that read per-request
        # options at decode time (word timings); this stub ignores it.
        decode_offline=lambda enc, lens, requests=None: [
            RequestOutput(request_id=i, text=f"ok-{i}", tokens=[[1]]) for i in enc.ids
        ],
        fill_nbest_texts=lambda req, out: None,
    )
    # The fake collate passes the ids straight through as "features".
    ex._collate = lambda chunk: ([r.request_id for r in chunk], torch.tensor([1] * len(chunk)))
    return ex


class TestOfflineIsolation:
    def test_one_bad_request_does_not_take_its_peers(self):
        runner = _Poisoned("bad", ValueError("out-of-range vocab id"))
        ex = _offline_executor(runner)
        batch = [_request("a"), _request("bad"), _request("c")]

        outs = {o.request_id: o for o in ex.run(batch)}

        assert set(outs) == {"a", "bad", "c"}
        assert outs["a"].text == "ok-a" and outs["c"].text == "ok-c"
        assert outs["bad"].finish_reason == "error"
        assert outs["bad"].error_stage == "offline_forward"
        assert outs["bad"].finished and outs["bad"].text == ""

    def test_the_failing_batch_is_retried_singly(self):
        """Isolation is what makes the peers survive — check the mechanism."""
        runner = _Poisoned("bad", ValueError("boom"))
        ex = _offline_executor(runner)
        ex.run([_request("a"), _request("bad"), _request("c")])

        # First the whole micro-batch, then one pass per member.
        assert runner.chunks_seen[0] == ["a", "bad", "c"]
        assert runner.chunks_seen[1:] == [["a"], ["bad"], ["c"]]

    def test_a_singleton_failure_is_not_retried(self):
        runner = _Poisoned("bad", ValueError("boom"))
        ex = _offline_executor(runner)
        outs = ex.run([_request("bad")])
        assert len(runner.chunks_seen) == 1, "a batch of one has nothing to isolate"
        assert outs[0].error_stage == "offline_forward"

    def test_oom_rejects_the_batch_without_retrying(self):
        """Re-running under memory pressure is how one big request cascades."""
        runner = _Poisoned("bad", torch.cuda.OutOfMemoryError("CUDA out of memory"))
        ex = _offline_executor(runner)
        outs = ex.run([_request("a"), _request("bad"), _request("c")])

        assert len(runner.chunks_seen) == 1, "OOM must not trigger a retry pass"
        assert {o.error_stage for o in outs} == {"offline_oom"}
        assert all(o.finish_reason == "error" for o in outs)

    def test_a_healthy_batch_is_untouched(self):
        runner = _Poisoned("nobody", ValueError("never raised"))
        ex = _offline_executor(runner)
        outs = ex.run([_request("a"), _request("b")])
        assert [o.text for o in outs] == ["ok-a", "ok-b"]
        assert all(o.finish_reason is None for o in outs)
        assert len(runner.chunks_seen) == 1

    def test_a_post_collate_failure_isolates_over_the_features(self, caplog):
        """After collation the waveforms are gone, so the retry cannot re-collate.

        ``InputProcessor.collate`` releases ``request.audio`` once the GPU
        feature tensor owns the batch.  Re-running the whole micro-batch per
        request therefore dies on ``NoneType.size`` — and *that* is what every
        request in the batch got told, including the healthy ones, while the
        real cause never reached a log line.  It shipped: a conv kernel failing
        on an over-wide batch (``max_batch_size >= ~220``) returned empty
        transcripts for the entire corpus under an ``AttributeError`` about
        waveforms.  Past collate the isolation pass runs over the features
        already built, which needs no waveform.
        """
        import logging

        runner = _Poisoned("bad", ValueError("Conv2DActivation kernel failed"))
        ex = _offline_executor(runner)
        base_collate = ex._collate

        def releasing_collate(chunk):
            # Exactly the real one's hazard: refuses a second pass, because the
            # waveforms it needs were handed to the GPU on the first.
            if any(r.audio is None for r in chunk):
                raise AttributeError("'NoneType' object has no attribute 'size'")
            out = base_collate(chunk)
            for r in chunk:
                r.audio = None
            return out

        batch = [_request("a"), _request("bad"), _request("c")]
        for req in batch:
            req.audio = torch.zeros(4)
        ex._collate = releasing_collate

        with caplog.at_level(logging.WARNING, logger="oasr.engine.executor.offline"):
            outs = {o.request_id: o for o in ex.run(batch)}

        # The peers survive — under the old retry they died on the re-collate.
        assert outs["a"].text == "ok-a" and outs["c"].text == "ok-c"
        assert outs["a"].finish_reason is None and outs["c"].finish_reason is None
        assert outs["bad"].finish_reason == "error"
        assert outs["bad"].error_stage == "offline_forward"
        # The isolation pass reran the rows, not the collation.
        assert runner.chunks_seen == [["a", "bad", "c"], ["a"], ["bad"], ["c"]]
        # And the log names the real cause rather than the released waveform.
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "Conv2DActivation kernel failed" in text
        assert "NoneType" not in text


# ---------------------------------------------------------------------------
# Streaming: a failed cohort must not drain the pool
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.freed: list[str] = []
        self.finished: list[str] = []

    def free_session(self, req):
        self.freed.append(("session", req.request_id))

    def free_stream(self, req):
        self.freed.append(("stream", req.request_id))

    def finish_request(self, rid):
        self.finished.append(rid)


def _streaming_executor(rec: _Recorder) -> StreamingExecutor:
    ex = StreamingExecutor.__new__(StreamingExecutor)
    ex._op = SimpleNamespace(free_session=rec.free_session)
    ex._mr = SimpleNamespace(free_stream=rec.free_stream)
    ex._scheduler = SimpleNamespace(finish_request=rec.finish_request)
    return ex


class TestStreamingIsolation:
    def test_failed_cohort_is_finalized_and_freed(self):
        rec = _Recorder()
        ex = _streaming_executor(rec)
        cohort = [_request("s1"), _request("s2")]

        outs = ex._fail_cohort(cohort, RuntimeError("BlockPool exhausted"), "streaming_forward")

        assert [o.request_id for o in outs] == ["s1", "s2"]
        assert all(o.finish_reason == "error" for o in outs)
        assert all(o.error_stage == "streaming_forward" for o in outs)
        assert all(o.finished for o in outs)
        assert all(r.state is RequestState.FINISHED for r in cohort)
        # Both caches released for both streams, and the scheduler told.
        assert set(rec.freed) == {
            ("session", "s1"),
            ("stream", "s1"),
            ("session", "s2"),
            ("stream", "s2"),
        }
        assert rec.finished == ["s1", "s2"]

    def test_a_release_that_itself_raises_does_not_abort_teardown(self):
        """Teardown runs after an unknown failure; it cannot assume clean state.

        Leaking a cache slot per failure exhausts the pool in a way that looks
        like a capacity bug rather than an error path, so every release is
        attempted regardless of the previous one.
        """
        rec = _Recorder()
        ex = _streaming_executor(rec)

        def angry_free_session(req):
            raise RuntimeError("session already gone")

        ex._op = SimpleNamespace(free_session=angry_free_session)
        cohort = [_request("s1")]

        outs = ex._fail_cohort(cohort, RuntimeError("boom"), "streaming_features")

        assert outs[0].error_stage == "streaming_features"
        assert ("stream", "s1") in rec.freed, "the second release must still run"
        assert rec.finished == ["s1"]


class TestStreamingStepDoesNotRaise:
    """The contract C6 is actually about: a failed forward must not escape step().

    ``_fail_cohort`` above is the mechanism; this is the property.  Before the
    guard, ``RuntimeError("BlockPool exhausted")`` propagated out of
    ``ASREngine.step()`` and the dispatcher fanned it out as INTERNAL to every
    in-flight request — three such ticks and the process drained.
    """

    @staticmethod
    def _executor(rec, *, forward_raises, lookahead=False):
        ex = StreamingExecutor.__new__(StreamingExecutor)
        ready = [_request("s1"), _request("s2")]
        for r in ready:
            r.stream_id = 0

        def schedule():
            return [], list(ready)

        def forward(reqs):
            if forward_raises:
                raise RuntimeError("BlockPool exhausted: requested 1 block but 0 are free")
            return {r.request_id: torch.zeros(1) for r in reqs}

        ex._scheduler = SimpleNamespace(
            schedule_streaming=schedule, finish_request=rec.finish_request
        )
        ex._inp = SimpleNamespace(extract_streaming_batch=lambda reqs, cuda_stream=None: None)
        ex._mr = SimpleNamespace(forward_streaming_step=forward, free_stream=rec.free_stream)
        ex._op = SimpleNamespace(
            decode_streaming_batch=lambda reqs, m: [],
            finalize_streaming=lambda req: RequestOutput(req.request_id, "", [[]]),
            fill_nbest_texts=lambda req, out: None,
            free_session=rec.free_session,
        )
        ex._config = SimpleNamespace(decoding_window=1)
        ex._feat_stream = None
        # Both step orders have their own forward/extract/decode sequencing and
        # therefore their own teardown ordering; the contract is the same.
        ex._lookahead = lookahead
        return ex, ready

    @pytest.mark.parametrize("lookahead", [False, True], ids=["serial", "pipelined"])
    def test_a_raising_forward_returns_error_outputs(self, monkeypatch, lookahead):
        rec = _Recorder()
        ex, ready = self._executor(rec, forward_raises=True, lookahead=lookahead)
        # Every stream has a full window and no pending audio.
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: False))

        outs = ex.step()  # must not raise

        assert {o.request_id for o in outs} == {"s1", "s2"}
        assert all(o.finish_reason == "error" for o in outs)
        assert all(o.error_stage == "streaming_forward" for o in outs)
        assert rec.finished == ["s1", "s2"]

    @pytest.mark.parametrize("lookahead", [False, True], ids=["serial", "pipelined"])
    def test_a_healthy_forward_is_unaffected(self, monkeypatch, lookahead):
        rec = _Recorder()
        ex, ready = self._executor(rec, forward_raises=False, lookahead=lookahead)
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: False))

        outs = ex.step()

        assert outs == []
        assert rec.finished == [], "no stream should be torn down on the happy path"


# ---------------------------------------------------------------------------
# Pipelined ticks: streaming feature lookahead and offline collate prefetch
#
# Both changes are reorderings, so what has to be pinned is the *order* — which
# is invisible to a transcript comparison but is the entire point.  Each test
# below fails if its stage is moved back to where it used to be.
# ---------------------------------------------------------------------------


class _OrderRecorder:
    """Records the sequence of executor stages as they are entered."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def mark(self, name):
        def _record(*args, **kwargs):
            self.seen.append(name)
            return None

        return _record


def _lookahead_executor(order: _OrderRecorder, *, lookahead: bool):
    """A ``StreamingExecutor`` wired to record stage order and nothing else."""
    ex = StreamingExecutor.__new__(StreamingExecutor)
    running = [_request("s1")]
    for r in running:
        r.stream_id = 0

    def extract(reqs, cuda_stream=None):
        order.seen.append("extract")

    def forward(reqs):
        order.seen.append("forward")
        return {r.request_id: torch.zeros(1) for r in reqs}

    def decode(reqs, m):
        order.seen.append("decode")
        return []

    ex._scheduler = SimpleNamespace(
        schedule_streaming=lambda: ([], list(running)),
        finish_request=lambda rid: None,
    )
    ex._inp = SimpleNamespace(extract_streaming_batch=extract)
    ex._mr = SimpleNamespace(forward_streaming_step=forward, free_stream=lambda r: None)
    ex._op = SimpleNamespace(
        decode_streaming_batch=decode,
        finalize_streaming=lambda req: RequestOutput(req.request_id, "", [[]]),
        fill_nbest_texts=lambda req, out: None,
        free_session=lambda r: None,
    )
    ex._config = SimpleNamespace(decoding_window=1)
    ex._feat_stream = None
    ex._lookahead = lookahead
    return ex, running


class TestStreamingFeatureLookahead:
    """The pack has to run *behind* the encoder, not in front of it.

    ``pad+pin`` — the per-stream concat + ``audio_scale`` + write into pinned
    staging — is host work that issues no GPU operation, and it profiled as the
    largest single block of GPU-idle in a streaming step precisely because it
    sat ahead of the forward.  Moving it between the forward and the decode is
    the fix, and the decode is the boundary that makes the placement matter: it
    ends in a device->host readback, so anything after it overlaps nothing.
    """

    @staticmethod
    def _run(lookahead, monkeypatch):
        order = _OrderRecorder()
        ex, _ = _lookahead_executor(order, lookahead=lookahead)
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: True))
        ex.step()
        return order.seen

    def test_pipelined_extracts_between_the_forward_and_the_decode(self, monkeypatch):
        assert self._run(True, monkeypatch) == ["forward", "extract", "decode"]

    def test_serial_extracts_before_the_forward(self, monkeypatch):
        assert self._run(False, monkeypatch) == ["extract", "forward", "decode"]


class TestStreamingLookaheadDrains:
    """A stream's last chunk must still be forwarded, one step later.

    Under lookahead the features a step extracts are consumed by the *next*
    step, so the finalisation check has to keep a stream alive on the strength
    of a ready encoder chunk alone — its audio deque is already empty.  Getting
    this wrong finalises the stream one chunk early and silently truncates every
    transcript's last word.
    """

    def test_a_stream_is_not_finalised_while_a_chunk_is_still_unforwarded(self, monkeypatch):
        order = _OrderRecorder()
        ex, running = _lookahead_executor(order, lookahead=True)
        req = running[0]
        req.audio_final = True
        # The state right after the extract that consumed the last chunk: no
        # audio left, one encoder window built and not yet forwarded.
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: False))
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)

        outs = ex.step()

        assert outs == [], "a stream with an unforwarded window is not drained"
        assert req.state is not RequestState.FINISHED

        # Next step: the window has been forwarded, nothing is left.
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: False)
        outs = ex.step()
        assert [o.request_id for o in outs] == ["s1"]


def _prefetch_executor(order: _OrderRecorder, batches, *, prefetch: bool):
    """An ``OfflineExecutor`` wired for the pipelined tick, on CPU.

    ``batches`` is consumed one per ``schedule_offline`` call, so the fixture can
    drive several ticks and watch where each stage lands.  ``_prefetch`` selects
    the tick shape and ``_collate_stream`` stays ``None``, so the collate runs
    inline and ``_StagedBatch.ready`` is ``None`` — the ordering under test is
    the host's, and it is the same with or without the side stream.
    """
    ex = OfflineExecutor.__new__(OfflineExecutor)
    queue = list(batches)

    def schedule_offline(limit=None):
        order.seen.append("schedule")
        return queue.pop(0) if queue else []

    def collate(chunk):
        order.seen.append("collate")
        return [r.request_id for r in chunk], torch.tensor([1] * len(chunk))

    def forward_offline(features, lengths):
        order.seen.append("forward")
        return SimpleNamespace(ids=list(features)), lengths

    def decode_offline(enc, lens, requests=None):
        order.seen.append("decode")
        return [RequestOutput(request_id=i, text=f"ok-{i}", tokens=[[1]]) for i in enc.ids]

    ex._scheduler = SimpleNamespace(
        schedule_offline=schedule_offline,
        split_offline_batch=lambda batch: ([batch], None),
        num_waiting_offline=0,
    )
    ex._mr = SimpleNamespace(forward_offline=forward_offline)
    ex._enable_packing = False
    ex._pending = {}
    ex._op = SimpleNamespace(
        strategy=SimpleNamespace(
            incremental=False, consumes="log_probs", has_pending=lambda: False
        ),
        decode_offline=decode_offline,
        fill_nbest_texts=lambda req, out: None,
    )
    ex._collate = collate
    ex._prefetch = prefetch
    ex._collate_stream = None
    ex._collate_done = None
    ex._queued = deque()
    ex._staged = None
    ex._skipped_admits = 0
    ex._decode_admit_window_ms = 0.0
    ex._max_batch_size = 8
    ex._max_decode_slots = None
    ex._metrics = build_metrics(enabled=False)
    return ex


class TestOfflineCollatePrefetch:
    """The next batch has to be selected and collated *after* the forward.

    Batch selection is pure host work and the collate's GPU work cannot start
    until it finishes, so in the serial tick both sit ahead of the encoder with
    nothing queued for the GPU — ~4 ms of a ~23.5 ms step at ``max_batch_size``
    256.  Issuing them after the forward puts them in the window where the GPU
    is busy.  Issued *before* the forward, they are once again the thing the GPU
    is idle for, which is exactly the state being replaced.
    """

    @staticmethod
    def _batches():
        return [[_request("a1"), _request("a2")], [_request("b1")]]

    def test_pipelined_stages_the_next_batch_after_the_forward(self):
        order = _OrderRecorder()
        ex = _prefetch_executor(order, self._batches(), prefetch=True)

        first = ex.step()

        # Priming collates once in front of the GPU — then the second batch's
        # schedule + collate land between this batch's forward and its decode.
        assert order.seen == [
            "schedule",
            "collate",
            "forward",
            "schedule",
            "collate",
            "decode",
        ]
        assert [o.request_id for o in first] == ["a1", "a2"]

    def test_serial_collates_before_the_forward(self):
        order = _OrderRecorder()
        ex = _prefetch_executor(order, self._batches(), prefetch=False)

        ex.step()

        assert order.seen == ["schedule", "collate", "forward", "decode"]

    def test_a_staged_batch_keeps_the_engine_pending(self):
        """A drain loop asks ``has_pending``; a staged batch is invisible to the
        scheduler queue and is not parked in ``_pending``, so without it the
        loop exits one tick before running the batch it just collated."""
        order = _OrderRecorder()
        ex = _prefetch_executor(order, self._batches(), prefetch=True)

        ex.step()  # returns batch a, stages batch b

        assert ex._staged is not None
        assert ex.has_pending(), "the staged batch must keep the drain loop alive"
        assert ex.num_running() == 1

        second = ex.step()
        assert [o.request_id for o in second] == ["b1"]
        assert not ex.has_pending()

    def test_every_request_comes_back_exactly_once(self):
        """Across ticks, with the pipeline priming and draining."""
        order = _OrderRecorder()
        batches = [[_request(f"r{i}")] for i in range(5)]
        ex = _prefetch_executor(order, [list(b) for b in batches], prefetch=True)

        seen: list[str] = []
        for _ in range(8):
            seen.extend(o.request_id for o in ex.step())

        assert seen == [f"r{i}" for i in range(5)]


class TestOfflinePrefetchOrdering:
    """The partitioner's length sort must still be undone per micro-batch.

    ``split_offline_batch`` returns indices that are flat over the *whole*
    scheduled batch, and its chunks no longer finish in the same tick, so the
    restore has to work from each chunk's own slice.  Treating those indices as
    positions within the chunk is an ``IndexError`` waiting for the first batch
    the scheduler actually splits.
    """

    def test_scattered_indices_restore_by_rank(self):
        outs = [RequestOutput(request_id=r, text=r, tokens=[[1]]) for r in ("c", "a", "b")]
        # Length-sorted chunk whose members came 7th, 2nd and 5th in the batch.
        restored = OfflineExecutor._restore_order(outs, [7, 2, 5])
        assert [o.request_id for o in restored] == ["a", "b", "c"]

    def test_a_full_permutation_matches_the_serial_restore(self):
        outs = [RequestOutput(request_id=r, text=r, tokens=[[1]]) for r in ("c", "a", "b")]
        order = [2, 0, 1]
        restored = OfflineExecutor._restore_order(outs, order)
        expected: list = [None] * 3
        for pos, orig in enumerate(order):  # the pre-pipeline restore, verbatim
            expected[orig] = outs[pos]
        assert restored == expected
