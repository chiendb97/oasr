#!/usr/bin/env python3
"""Expected-length admission for the AR families (``kv_admission='expected'``).

A row used to reserve ``prompt + max_new_tokens`` of the paged decoder-KV pool
at admission, although a transcript is a few dozen tokens against a cap of
hundreds.  Now rows are admitted by the footprint finished rows reached, and
when the rows in flight outgrow that, the most recently admitted is swapped to
host memory and resumed later.

Pure CPU.  The decoder is a fake whose every token is a function of the K it
reads back **through the block table** -- the row's whole valid window -- so a
row resumed from the wrong bytes, at the wrong offset or with the wrong padding
window decodes a different transcript, rather than the same one by luck.
"""

from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch
from torch import nn

from oasr.cache import build_kv
from oasr.engine.decode.incremental import IncrementalArStrategy, Prefill
from oasr.engine.executor.offline import OfflineExecutor
from oasr.engine.generation import StepBudget
from oasr.engine.request import Request, RequestState

#: Small, so EOS is likely: like a transcript, a row ends far below its cap.
VOCAB = 4
EOS = 0
BLOCK = 4
#: fp32, one layer, one KV head of width 4: 2 * 4 * 4 bytes a position.
BYTES_PER_TOKEN = 32
GIB = 1024**3
#: The generation cap: a row's ceiling is the longest prompt plus this.
MAX_NEW = 64


class _PagedDecoder(nn.Module):
    """One layer of paged K/V; the next token is a hash of the row's whole window."""

    supports_paged_kv = True
    supports_step_graphs = False

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))  # dtype/device for the pool
        self.reads = 0

    def prefill(self, prompt, lens, capacity, kv_manager):
        """``prompt (B, P, 4)`` left-padded, ``lens`` each row's real length."""
        B, P, _ = prompt.shape
        starts = [P - int(n) for n in lens]
        kv = build_kv(
            1,
            B,
            prompt.device,
            prefill_len=P,
            cap=capacity,
            manager=kv_manager,
            starts=torch.tensor(starts, dtype=torch.int32),
            starts_host=starts,
        )
        kv.mask_kwargs(P)
        k = prompt.unsqueeze(1)  # (B, 1, P, 4)
        kv.append(0, k, k)
        kv.commit(P)
        return self._logits(kv), {"kv": kv}

    def step(self, tokens, state):
        kv = state["kv"]
        kv.mask_kwargs(1)
        k = (tokens.to(torch.float32) * 0.25 + 0.5).view(-1, 1, 1, 1).expand(-1, 1, 1, 4)
        kv.append(0, k.contiguous(), k.contiguous())
        kv.commit(1)
        return self._logits(kv), state

    def _logits(self, kv):
        self.reads += 1
        k_pool, _ = kv.manager.kv_view(0)
        table = kv.block_table()
        starts = kv.starts.tolist()
        out = torch.full((kv.batch, VOCAB), -1e4)
        for b in range(kv.batch):
            window = [
                k_pool[int(table[b, t // BLOCK]), t % BLOCK, 0]
                for t in range(starts[b], kv.lens_host[b])
            ]
            total = float(torch.stack(window).sum()) if window else 0.0
            # The row's own length, not its offset: a prefill pads every row to
            # its batch's longest prompt, and the token must not depend on whom
            # it was batched with.
            out[b, (int(abs(total) * 10.0) + kv.lens_host[b] - starts[b]) % VOCAB] = 10.0
        return out

    @staticmethod
    def select(state, keep):
        return {"kv": state["kv"].select(keep)}

    @staticmethod
    def can_merge(a, b):
        return a["kv"].can_merge(b["kv"])

    @staticmethod
    def merge(a, b):
        return {"kv": a["kv"].merge(b["kv"])}


#: The longest prompt :func:`_prompt` makes.
MAX_PROMPT = 10


class _PagedStrategy(IncrementalArStrategy):
    decode_type = "fake_paged"

    def _prompt_len_estimate(self):
        return MAX_PROMPT

    def _prefill(self, requests, enc_out, enc_lengths):
        max_new = [self._row_cap(r, 999) for r in requests]
        P = enc_out.size(1)
        logits, state = self._decoder().prefill(
            enc_out,
            enc_lengths.tolist(),
            capacity=[P + m for m in max_new],
            kv_manager=self.kv_manager(),
        )
        return Prefill(state=state, logits=logits, max_new=max_new)

    def _is_eos(self, token):
        return token == EOS


def _detok():
    def incremental(new_ids, state):
        state.setdefault("ids", []).extend(int(i) for i in new_ids)
        state["text"] = " ".join(map(str, state["ids"]))
        return state["text"]

    return SimpleNamespace(
        detokenize=lambda ids: " ".join(map(str, ids)),
        new_state=lambda: {"ids": [], "text": ""},
        detokenize_incremental=incremental,
    )


def _strategy(
    *, pool_blocks: int, admission: str = "expected", max_new: int = MAX_NEW, rows: int = 8
):
    model = SimpleNamespace(
        decoder=_PagedDecoder(),
        decoder_cache_spec=SimpleNamespace(
            num_layers=1, n_kv_head=1, head_dim=4, cross_attention_len=0
        ),
    )
    cfg = SimpleNamespace(
        max_new_tokens=max_new,
        max_decode_slots=None,
        max_batch_size=rows,
        decode_kv_budget_gib=pool_blocks * BLOCK * BYTES_PER_TOKEN / GIB,
        dtype=torch.float32,
        quantization=None,
        decode_options={
            "kv_storage": "paged",
            "kv_block_tokens": BLOCK,
            "kv_admission": admission,
            "step_graphs": False,
            "loop_steps": 0,
        },
    )
    return _PagedStrategy(cfg, _detok(), model)


def _prompt(rid: str):
    """A request's encoder output: its own length (3..10) and values."""
    seed = int(rid[1:])
    n = 3 + (seed * 5) % 8
    return (seed + 1) * 0.37 + torch.arange(n, dtype=torch.float32) * 0.11


def _encode(requests):
    rows = [_prompt(r.request_id) for r in requests]
    P = max(len(r) for r in rows)
    out = torch.zeros(len(rows), P, 4)
    for b, row in enumerate(rows):
        out[b, P - len(row) :] = row.unsqueeze(1)
    return out, torch.tensor([len(r) for r in rows])


def _requests(n, start=0):
    return [
        Request(audio=None, request_id=f"r{i}", streaming=False) for i in range(start, start + n)
    ]


def _drive(strategy, batches, steps_per_tick=4, max_ticks=2000):
    """Prefill each batch on its own tick, then advance until every row finishes."""
    finals: Dict[str, List[int]] = {}
    pending = list(batches)
    for _ in range(max_ticks):
        if pending:
            reqs = pending.pop(0)
            strategy.begin_offline(reqs, *_encode(reqs))
        for out in strategy.advance(StepBudget(max_steps=steps_per_tick)):
            if out.finished:
                assert out.finish_reason in ("stop", "length"), out
                finals[out.request_id] = out.tokens[0]
        if not pending and not strategy.has_pending():
            break
    return finals


def _reference(batches):
    """The same requests on a pool too big to run short: nothing is swapped."""
    strategy = _strategy(pool_blocks=4096)
    finals = _drive(strategy, batches)
    assert strategy.preemptions == 0
    return finals


class TestPreemption:
    """Rows swapped out of a pool that ran short finish exactly as if it had not."""

    def test_swapped_rows_decode_the_same_tokens(self):
        batches = [_requests(6), _requests(6, start=6)]
        want = _reference(batches)
        strategy = _strategy(pool_blocks=24)
        got = _drive(strategy, batches)
        assert strategy.preemptions > 0, "the pool was meant to run short"
        assert got == want
        pool = strategy.kv_manager().pool
        assert pool.num_free_blocks == pool.num_blocks, "every page came back"

    def test_the_most_recently_admitted_row_goes_first(self):
        strategy = _strategy(pool_blocks=24)
        first, second = _requests(3), _requests(3, start=3)
        strategy.begin_offline(first, *_encode(first))
        strategy.begin_offline(second, *_encode(second))
        assert strategy._preempt_youngest([])  # noqa: SLF001
        assert [p.request.request_id for p in strategy._preempted] == ["r5"]  # noqa: SLF001

    def test_a_prefill_that_does_not_fit_makes_room_instead_of_failing(self):
        """r3, r11, r19, ... (``seed % 8 == 3``) have the longest prompts, three
        pages each; the pool holds one row at its ceiling, as the budget does."""
        ceiling = -(-(MAX_PROMPT + MAX_NEW) // BLOCK)
        strategy = _strategy(pool_blocks=ceiling)
        ids = ("r3", "r11", "r27", "r35", "r43", "r51")
        first = [Request(audio=None, request_id=r, streaming=False) for r in ids]
        strategy.begin_offline(first, *_encode(first))
        assert strategy.kv_manager().pool.num_free_blocks == ceiling - 18
        late = [Request(audio=None, request_id="r19", streaming=False)]
        strategy.begin_offline(late, *_encode(late))  # would raise DecoderKvExhausted
        assert [p.request.request_id for p in strategy._preempted] == ["r51"]  # noqa: SLF001
        finals = _drive(strategy, [])
        assert set(finals) == set(ids) | {"r19"}
        assert finals == _reference([first, late])

    def test_an_abort_reaches_a_swapped_out_row(self):
        strategy = _strategy(pool_blocks=24)
        reqs = _requests(4)
        strategy.begin_offline(reqs, *_encode(reqs))
        strategy._preempt_youngest([])  # noqa: SLF001
        victim = strategy._preempted[0].request  # noqa: SLF001
        strategy.free_session(victim)
        assert not strategy._preempted  # noqa: SLF001
        finals = _drive(strategy, [])
        assert victim.request_id not in finals and len(finals) == 3
        pool = strategy.kv_manager().pool
        assert pool.num_free_blocks == pool.num_blocks

    def test_the_last_row_in_flight_is_never_swapped_out(self):
        """Alone, it would come back to the same pool and swap itself out again.
        A pool smaller than one row's growth fails that row instead."""
        strategy = _strategy(pool_blocks=4)
        reqs = [Request(audio=None, request_id="r3", streaming=False)]  # 3 pages of prompt
        strategy.begin_offline(reqs, *_encode(reqs))
        outs = []
        for _ in range(40):
            outs += strategy.advance(StepBudget(max_steps=4))
            if not strategy.has_pending():
                break
        assert strategy.preemptions == 0
        assert [(o.finish_reason, o.error_stage) for o in outs if o.finished] in (
            [("error", "decode_oom")],
            [("stop", None)],
        )
        pool = strategy.kv_manager().pool
        assert pool.num_free_blocks == pool.num_blocks

    def test_ceiling_admission_never_swaps(self):
        strategy = _strategy(pool_blocks=24, admission="ceiling")
        assert strategy.kv_admissible_rows() is None
        assert not strategy.kv_manager().overcommit


class TestExpectedFootprint:
    """What admission charges a row: the ceiling until rows have finished, then their mean."""

    def test_the_ceiling_is_the_prior(self):
        strategy = _strategy(pool_blocks=64)
        ceiling = -(-(MAX_PROMPT + MAX_NEW) // BLOCK)  # the longest prompt + the cap, in pages
        assert strategy.kv_admissible_rows() == 64 // (ceiling + 1)

    def test_finished_rows_set_the_expectation(self):
        strategy = _strategy(pool_blocks=64)
        _drive(strategy, [_requests(8), _requests(8, start=8)])
        expected = strategy._expected_row_pages()  # noqa: SLF001
        assert expected < -(-(MAX_PROMPT + MAX_NEW) // BLOCK)
        assert strategy.kv_admissible_rows() == 64 // (expected + 1)

    def test_nothing_is_admitted_while_a_row_waits_to_resume(self):
        strategy = _strategy(pool_blocks=64)
        reqs = _requests(2)
        strategy.begin_offline(reqs, *_encode(reqs))
        strategy._preempt_youngest([])  # noqa: SLF001
        assert strategy.kv_admissible_rows() == 0


# ---------------------------------------------------------------------------
# Through the offline executor
# ---------------------------------------------------------------------------


class _Scheduler:
    """FIFO, flooring a limit at one row the way ``_batch_cap`` does."""

    def __init__(self):
        self.waiting: List[Request] = []

    def add_request(self, req):
        self.waiting.append(req)

    def schedule_offline(self, limit=None):
        cap = len(self.waiting) if limit is None else max(1, int(limit))
        batch, self.waiting = self.waiting[:cap], self.waiting[cap:]
        return batch

    def split_offline_batch(self, batch):
        return [batch], None

    @property
    def num_waiting_offline(self):
        return len(self.waiting)

    def oldest_offline_wait(self):
        return None

    def abort_request(self, request_id):
        self.waiting = [r for r in self.waiting if r.request_id != request_id]

    def find_request(self, request_id):
        return next((r for r in self.waiting if r.request_id == request_id), None)


class _Inputs:
    def prepare_offline(self, req):
        pass

    def collate(self, chunk):
        return _encode(chunk)


class _Runner:
    def encode_offline(self, features, lengths):
        return features, lengths


class _Outputs:
    def __init__(self, strategy):
        self.strategy = strategy

    def fill_nbest_texts(self, request, output):
        return None


def _serve(n, *, pool_blocks, admission, slots=12):
    strategy = _strategy(pool_blocks=pool_blocks, admission=admission, rows=slots)
    ex = OfflineExecutor(
        scheduler=_Scheduler(),
        input_processor=_Inputs(),
        model_runner=_Runner(),
        output_processor=_Outputs(strategy),
        device=torch.device("cpu"),
        decode_steps_per_tick=4,
        max_decode_slots=slots,
        max_batch_size=slots,
        decode_kv_budget_gib=pool_blocks * BLOCK * BYTES_PER_TOKEN / GIB,
    )
    for req in _requests(n):
        ex.admit(req)
    finals: Dict[str, List[int]] = {}
    widest = 0
    for _ in range(5000):
        for out in ex.step():
            if out.finished:
                assert out.finish_reason in ("stop", "length"), out
                finals[out.request_id] = out.tokens[0]
        widest = max(widest, len(ex._pending))  # noqa: SLF001
        if not ex.has_pending():
            break
    assert all(r.state == RequestState.FINISHED for r in [])
    return finals, widest, strategy


class TestThroughTheExecutor:
    def test_more_rows_fly_and_every_transcript_is_unchanged(self):
        want, ceiling_widest, _ = _serve(40, pool_blocks=48, admission="ceiling")
        got, widest, strategy = _serve(40, pool_blocks=48, admission="expected")
        assert got == want and len(got) == 40
        assert widest > ceiling_widest, (widest, ceiling_widest)
        pool = strategy.kv_manager().pool
        assert pool.num_free_blocks == pool.num_blocks

    def test_one_prefill_stays_within_the_rows_the_budget_reserved(self):
        """The budget reserved prefill memory for the rows it fits at their
        ceilings; a prefill wider than that could run out of device memory."""
        _, _, strategy = _serve(24, pool_blocks=48, admission="expected")
        ex = OfflineExecutor.__new__(OfflineExecutor)
        ex._op = SimpleNamespace(strategy=strategy)  # noqa: SLF001
        ex._decode_kv_budget_gib = 48 * BLOCK * BYTES_PER_TOKEN / GIB  # noqa: SLF001
        ex._pending = {}  # noqa: SLF001
        ceiling_rows = 48 // -(-(MAX_PROMPT + MAX_NEW) // BLOCK)
        assert strategy.kv_admissible_rows() > ceiling_rows
        assert ex._rows_within_kv_budget(strategy) == ceiling_rows  # noqa: SLF001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
