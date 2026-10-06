# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The fused greedy path of ``TransducerDecodeStrategy`` (``oasr/engine/decode/transducer.py``).

What the strategy adds on top of the kernel -- and what these tests hold it to:
every entry point (offline, offline-async, streaming) takes the fused launch when
the model's surface declares the tensors and otherwise runs the op-by-op loop, a
batch that overflows the kernel's emission buffer is re-decoded rather than
truncated, and a weight written after the first decode is not decoded with a
stale copy.  The model comes from :mod:`helpers.transducer`, whose exact
arithmetic lets "the same transcript as the loop" be literal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from helpers.transducer import exact_transducer, integer_frames, transducer_strategy

import oasr.functionals.transducer as transducer_fn

pytestmark = pytest.mark.cuda

LENGTHS = [24, 17, 0, 1, 24, 9, 13, 2]


def _tokens(outputs):
    return [o.tokens[0] for o in outputs]


def _offline(strat, enc, lengths):
    with torch.no_grad():
        return _tokens(strat.decode_offline(enc, lengths))


def _other_frames(enc):
    """Frames of ``enc``'s shape and dtype, but not its values."""
    return integer_frames(enc.size(0), enc.size(1), enc.dtype, seed=99)


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


class TestOffline:
    def test_fused_equals_loop(self, dtype):
        model = exact_transducer(dtype)
        fused, loop = transducer_strategy(model), transducer_strategy(model, fused=False)
        enc = integer_frames(len(LENGTHS), max(LENGTHS), dtype)
        lengths = torch.tensor(LENGTHS, device="cuda")
        assert _offline(fused, enc, lengths) == _offline(loop, enc, lengths)
        assert fused.fused_stats == {"hits": 1, "fallbacks": 0, "overflows": 0}
        assert loop.fused_stats["hits"] == 0

    def test_async_equals_sync(self, dtype):
        model = exact_transducer(dtype, seed=3)
        strat = transducer_strategy(model)
        enc = integer_frames(len(LENGTHS), max(LENGTHS), dtype, seed=4)
        lengths = torch.tensor(LENGTHS, device="cuda")
        with torch.no_grad():
            collect = strat.decode_offline_async(enc, lengths)
            assert collect is not None
            got = _tokens(collect())
        assert got == _offline(transducer_strategy(model, fused=False), enc, lengths)

    def test_async_declines_what_decodes_synchronously(self):
        model = exact_transducer(torch.bfloat16)
        enc = integer_frames(2, 4, torch.bfloat16)
        lengths = torch.tensor([4, 4], device="cuda")
        with torch.no_grad():
            # A batch that asked for word timings reads its emissions after the
            # read-back, so it is not queued; neither is the loop's surface.
            asking = [SimpleNamespace(), SimpleNamespace()]
            assert transducer_strategy(model).decode_offline_async(enc, lengths, asking) is None
            assert (
                transducer_strategy(model, fused=False).decode_offline_async(enc, lengths) is None
            )

    def test_side_stream_equals_main_stream(self, dtype):
        model = exact_transducer(dtype, seed=12)
        enc = integer_frames(len(LENGTHS), max(LENGTHS), dtype, seed=13)
        lengths = torch.tensor(LENGTHS, device="cuda")
        got = {}
        for side in (True, False):
            strat = transducer_strategy(model, side_stream=side)
            with torch.no_grad():
                got[side] = _tokens(strat.decode_offline_async(enc, lengths)())
            assert (strat._side_stream is not None) == side
        assert got[True] == got[False]

    def test_side_stream_waits_for_the_forward(self):
        """The decode reads ``enc_proj``, which the main stream has only queued.

        The main stream is held back on purpose: a side stream that did not wait
        for it would start on a buffer nothing has written yet.  The same shapes
        are decoded once first, from other frames: a first call can load a JIT
        module, which synchronizes the device and hides the window.
        """
        model = exact_transducer(torch.bfloat16, seed=14)
        src = integer_frames(len(LENGTHS), max(LENGTHS), torch.bfloat16, seed=15)
        lengths = torch.tensor(LENGTHS, device="cuda")
        strat = transducer_strategy(model)
        with torch.no_grad():
            strat.decode_offline_async(_other_frames(src), lengths)()
            enc = torch.zeros_like(src)
            torch.cuda._sleep(50_000_000)
            enc.copy_(src)
            got = _tokens(strat.decode_offline_async(enc, lengths)())
        assert got == _offline(transducer_strategy(model, fused=False), src, lengths)

    def test_side_stream_keeps_its_inputs_alive(self):
        """``enc_proj`` dies with the call that made it; its block must not.

        The allocator would hand that block to the next main-stream allocation,
        and the main stream -- held back here, so the side kernel is still
        reading -- would overwrite it.  ``record_stream`` is what defers the reuse.
        """
        model = exact_transducer(torch.bfloat16, seed=16)
        enc = integer_frames(len(LENGTHS), max(LENGTHS), torch.bfloat16, seed=17)
        lengths = torch.tensor(LENGTHS, device="cuda")
        strat = transducer_strategy(model)
        with torch.no_grad():
            strat.decode_offline_async(_other_frames(enc), lengths)()
            torch.cuda._sleep(50_000_000)
            collect = strat.decode_offline_async(enc, lengths)
            scribbles = [torch.full_like(enc, 3) for _ in range(8)]
            got = _tokens(collect())
        del scribbles
        assert got == _offline(transducer_strategy(model, fused=False), enc, lengths)

    def test_overflow_falls_back_to_the_loop(self, monkeypatch):
        # Every step wants to emit: max_sym tokens per frame, past a tiny buffer.
        model = exact_transducer(torch.bfloat16, blank_bias=-64.0)
        strat = transducer_strategy(model)
        enc = integer_frames(3, 10, torch.bfloat16)
        lengths = torch.tensor([10, 4, 7], device="cuda")
        monkeypatch.setattr(transducer_fn, "transducer_greedy_capacity", lambda frames: 5)
        got = _offline(strat, enc, lengths)
        assert got == _offline(transducer_strategy(model, fused=False), enc, lengths)
        assert [len(h) for h in got] == [30, 12, 21]
        assert strat.fused_stats["overflows"] == 1

    def test_weights_written_after_first_decode_are_used(self):
        model = exact_transducer(torch.bfloat16)
        strat = transducer_strategy(model)
        enc = integer_frames(4, 16, torch.bfloat16, seed=6)
        lengths = torch.tensor([16, 16, 9, 12], device="cuda")
        before = _offline(strat, enc, lengths)
        with torch.no_grad():
            model.joiner.output_linear.weight.neg_()  # in place: same tensor, new version
        after = _offline(strat, enc, lengths)
        assert after == _offline(transducer_strategy(model, fused=False), enc, lengths)
        assert after != before


class TestFallsBack:
    def test_fp32_runs_the_loop(self):
        model = exact_transducer(torch.float32)
        strat = transducer_strategy(model)
        enc = integer_frames(2, 6, torch.float32)
        lengths = torch.tensor([6, 3], device="cuda")
        got = _offline(strat, enc, lengths)
        assert got == _offline(transducer_strategy(model, fused=False), enc, lengths)
        assert strat.fused_stats["hits"] == 0 and strat._fused_weights is None

    def test_a_surface_without_the_tensors_is_never_fused(self):
        model = exact_transducer(torch.bfloat16)
        model.joiner.additive_tensors = lambda: None  # e.g. a non-additive joint
        strat = transducer_strategy(model)
        enc = integer_frames(2, 6, torch.bfloat16)
        lengths = torch.tensor([6, 3], device="cuda")
        _offline(strat, enc, lengths)
        assert not strat._fused_enabled and strat.fused_stats["hits"] == 0


class TestStreaming:
    def _stream(self, strat, streams, splits):
        """Feed every stream its chunks tick by tick; return the final tokens."""
        reqs = {rid: SimpleNamespace(request_id=rid) for rid in streams}
        for req in reqs.values():
            strat.create_session(req)
        bounds = {rid: [0] + list(splits[rid]) for rid in streams}
        ticks = max(len(b) - 1 for b in bounds.values())
        with torch.no_grad():
            for k in range(ticks):
                ready = {
                    rid: streams[rid][:, bounds[rid][k] : bounds[rid][k + 1]]
                    for rid in streams
                    if k + 1 < len(bounds[rid])
                }
                strat.decode_streaming_batch([reqs[r] for r in ready], ready)
        return {rid: strat.finalize(req).tokens[0] for rid, req in reqs.items()}

    def test_fused_equals_loop_across_mixed_chunk_groups(self, dtype):
        model = exact_transducer(dtype, seed=8)
        enc = integer_frames(3, 24, dtype, seed=9)
        streams = {f"s{i}": enc[i : i + 1] for i in range(3)}
        # Different chunk lengths in one tick land in different groups.
        splits = {"s0": [8, 16, 24], "s1": [5, 16, 24], "s2": [8, 13, 24]}
        fused = self._stream(transducer_strategy(model), streams, splits)
        loop = self._stream(transducer_strategy(model, fused=False), streams, splits)
        assert fused == loop

    def test_streaming_equals_offline(self, dtype):
        model = exact_transducer(dtype, seed=10)
        enc = integer_frames(1, 24, dtype, seed=11)
        strat = transducer_strategy(model)
        streamed = self._stream(strat, {"s": enc}, {"s": [1, 7, 8, 20, 24]})["s"]
        assert streamed == _offline(transducer_strategy(model), enc, torch.tensor([24]).cuda())[0]
        assert strat.fused_stats["hits"] == 5
