#!/usr/bin/env python3
"""Tests for the incremental decode protocol (K2), both sides of the seam.

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
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from oasr.engine.decode.base import DecodeStrategy
from oasr.engine.executor.offline import _MAX_SKIPPED_ADMITS, OfflineExecutor
from oasr.engine.generation import StepBudget
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
# Shared incremental-AR base (H1): the seam a third AR family plugs into
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
