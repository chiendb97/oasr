#!/usr/bin/env python3
"""Tests for the incremental decode protocol in the offline executor (K2).

Pure CPU: every engine component around ``OfflineExecutor`` is a small fake,
and the strategy is a synthetic ``incremental=True`` implementation that emits
one token per batched decoder step.  This pins the executor-side contract —
bounded steps per tick, the pending pool lifecycle, admission gating, abort —
independent of any real model (Whisper etc. plug into the same seam).
"""

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

    # -- incremental protocol ------------------------------------------
    def begin_offline(self, requests, enc_out, enc_lengths):
        for req in requests:
            self.states[req.request_id] = []

    def advance(self, budget: StepBudget) -> List[RequestOutput]:
        outputs: List[RequestOutput] = []
        steps = 0
        while self.states and budget.take():
            steps += 1
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
        """A request admitted later joins the same advance loop mid-flight."""
        ex, strat = _make_executor({"a": 6, "b": 2}, steps_per_tick=2)
        _admit(ex, "a")
        ex.step()  # prefill a
        ex.step()  # a: 2/6
        _admit(ex, "b")
        outs = ex.step()  # advance (a:4/6) then prefill b — same tick
        assert outs == [] and ex.num_running() == 2
        outs = ex.step()  # a hits 6 and b hits 2 in the same 2-step tick
        assert sorted(o.request_id for o in outs) == ["a", "b"]

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
