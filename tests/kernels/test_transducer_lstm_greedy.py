# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.transducer_lstm_greedy_decode`` (``include/oasr/transducer/lstm_greedy.cuh``).

The oracle is :class:`~oasr.engine.decode.transducer.TransducerDecodeStrategy`'s
op-by-op loop on a tiny NeMo-style model (an LSTM predictor and an additive ReLU
joint).  The kernel keeps every rounding point of that loop but not the library
kernels' accumulation orders, so the two agree to the bit only where every sum is
exact.  They are held to agreement where it matters instead: the head's weights
are scaled until no decision sits near a tie, so **tokens and frames must be
identical**, and the carried state is compared within rounding.

What the kernel owns outright is exact: its result does not depend on how many
CTAs ran it or which rows shared its batch (each output is one warp's dot product
in a fixed order), and decoding in chunks carries the state bit for bit.
"""

from __future__ import annotations

import pytest
import torch
from helpers.transducer import transducer_strategy
from torch import nn

from oasr.functionals.transducer import (
    LSTM_GREEDY_MAX_ROWS,
    LstmGreedyResult,
    LstmGreedyWeights,
    transducer_lstm_greedy_decode,
)

pytestmark = pytest.mark.cuda

VOCAB, ENC, HIDDEN, LAYERS, BLANK = 37, 24, 32, 2, 0


def _model(dtype, *, seed=7, layers=LAYERS, blank_bias=None):
    """A tiny NeMo-style transducer whose logits are far apart (scale 6)."""
    from oasr.models.nemotron.predictor import NemotronRnntJoint, NemotronRnntPredictor
    from oasr.models.transducer import TransducerModel

    torch.manual_seed(seed)
    decoder = NemotronRnntPredictor(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_layers=layers, blank_id=BLANK
    )
    joiner = NemotronRnntJoint(ENC, HIDDEN, VOCAB)
    with torch.no_grad():
        joiner.head.weight.mul_(6.0)
        # Blank a little ahead, so rows both emit and advance.
        joiner.head.bias[BLANK] += 4.0 if blank_bias is None else blank_bias
    model = TransducerModel(nn.Identity(), decoder, joiner, blank_id=BLANK).eval()
    return model.to(device="cuda", dtype=dtype)


def _weights(model) -> LstmGreedyWeights:
    pred = model.decoder.recurrent_tensors()
    join = model.joiner.additive_tensors()
    return LstmGreedyWeights.prepare(
        embedding=pred.embedding,
        weight_ih=pred.weight_ih,
        weight_hh=pred.weight_hh,
        bias_ih=pred.bias_ih,
        bias_hh=pred.bias_hh,
        decoder_proj_weight=join.decoder_proj_weight,
        decoder_proj_bias=join.decoder_proj_bias,
        output_weight=join.output_weight,
        output_bias=join.output_bias,
        vocab=join.vocab_size,
        blank=pred.blank_id,
    )


def _enc(B, T, dtype, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(B, T, ENC, device="cuda", generator=g).to(dtype)


def _loop(model, enc, lengths, max_sym=3):
    """The op-by-op loop: (hyps, frames, (out, h, c), dec_proj)."""
    strat = transducer_strategy(model, max_sym=max_sym, fused=False)
    state, dec_proj = strat._init_state(enc.size(0), enc.device)
    with torch.no_grad():
        hyps, marks, state, dec_proj = strat._greedy_loop(enc, lengths, state, dec_proj, track=True)
    return hyps, [[f for f, _ in row] for row in marks], state, dec_proj


def _fused(model, enc, lengths, max_sym=3, **kwargs) -> LstmGreedyResult:
    strat = transducer_strategy(model, max_sym=max_sym, fused=False)
    (_out, h, c), dec_proj = strat._init_state(enc.size(0), enc.device)
    with torch.no_grad():
        res = transducer_lstm_greedy_decode(
            model.joiner.encoder_proj(enc),
            lengths,
            h,
            c,
            dec_proj,
            _weights(model),
            max_sym=max_sym,
            **kwargs,
        )
    torch.cuda.synchronize()
    return res


def _rows(res: LstmGreedyResult):
    counts = res.counts.tolist()
    hyps = [res.tokens[b, :n].tolist() for b, n in enumerate(counts)]
    frames = [res.frames[b, :n].tolist() for b, n in enumerate(counts)]
    return hyps, frames


def _assert_identical(a: LstmGreedyResult, b: LstmGreedyResult):
    assert _rows(a) == _rows(b)
    assert torch.equal(a.counts, b.counts)
    assert torch.equal(a.h, b.h) and torch.equal(a.c, b.c)
    assert torch.equal(a.dec_proj, b.dec_proj)


class TestMatchesOpByOpLoop:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("lengths", [[30], [30, 21], [30, 0, 12], [30, 21, 1, 30]])
    def test_tokens_frames_and_state(self, dtype, lengths):
        model = _model(dtype)
        enc = _enc(len(lengths), max(lengths), dtype)
        lens = torch.tensor(lengths, device="cuda")
        hyps, frames, (out, h, c), dec_proj = _loop(model, enc, lens)
        res = _fused(model, enc, lens)
        assert _rows(res) == (hyps, frames)
        # The state took the same tokens through the same rounding points; only
        # the dot products' summation order differs.
        tol = (
            {"atol": 2e-2, "rtol": 2e-2}
            if dtype == torch.bfloat16
            else {"atol": 2e-3, "rtol": 2e-3}
        )
        torch.testing.assert_close(res.h, h, **tol)
        torch.testing.assert_close(res.c, c, **tol)
        torch.testing.assert_close(res.dec_proj, dec_proj, **tol)
        torch.testing.assert_close(res.h[-1], out, **tol)
        # Both branches: frames that emit and frames that do not.
        emitted = sum(len(set(f)) for f in frames)
        assert 0 < emitted < sum(lengths)

    def test_one_layer(self):
        model = _model(torch.float16, layers=1, seed=3)
        enc = _enc(2, 25, torch.float16, seed=3)
        lens = torch.tensor([25, 18], device="cuda")
        hyps, frames, _, _ = _loop(model, enc, lens)
        assert _rows(_fused(model, enc, lens)) == (hyps, frames)

    def test_max_sym_caps_emissions_per_frame(self):
        model = _model(torch.float16, blank_bias=-80.0)  # never blank
        enc = _enc(2, 9, torch.float16)
        lens = torch.tensor([9, 4], device="cuda")
        res = _fused(model, enc, lens, max_sym=3)
        assert res.counts.tolist() == [27, 12]
        hyps, frames, _, _ = _loop(model, enc, lens, max_sym=3)
        assert _rows(res) == (hyps, frames)


class TestWhatTheKernelOwnsIsExact:
    def test_the_grid_size_does_not_change_the_result(self):
        """Every output is one warp's dot product in a fixed order, so any grid
        -- one CTA doing everything, or CTAs left with no hidden units -- gives
        the same bits."""
        model = _model(torch.float16, seed=5)
        enc = _enc(3, 26, torch.float16, seed=5)
        lens = torch.tensor([26, 19, 7], device="cuda")
        ref = _fused(model, enc, lens, grid=1)
        for grid in (2, 5, 0):
            _assert_identical(_fused(model, enc, lens, grid=grid), ref)

    def test_a_row_does_not_depend_on_its_batch(self):
        model = _model(torch.bfloat16, seed=6)
        enc = _enc(4, 28, torch.bfloat16, seed=6)
        lens = torch.tensor([28, 13, 20, 28], device="cuda")
        together = _fused(model, enc, lens)
        for b in range(4):
            alone = _fused(model, enc[b : b + 1], lens[b : b + 1])
            n = int(together.counts[b])
            assert torch.equal(alone.tokens[0, :n], together.tokens[b, :n])
            assert torch.equal(alone.frames[0, :n], together.frames[b, :n])
            assert torch.equal(alone.h[:, 0], together.h[:, b])
            assert torch.equal(alone.c[:, 0], together.c[:, b])
            assert torch.equal(alone.dec_proj[0], together.dec_proj[b])

    def test_chunks_carry_state_exactly(self):
        model = _model(torch.float16, seed=8)
        weights = _weights(model)
        enc = model.joiner.encoder_proj(_enc(2, 33, torch.float16, seed=8))
        strat = transducer_strategy(model, fused=False)
        (_out, h, c), dec_proj = strat._init_state(2, enc.device)
        with torch.no_grad():
            one = transducer_lstm_greedy_decode(
                enc, torch.tensor([33, 33], device="cuda"), h, c, dec_proj, weights, max_sym=3
            )
            hyps = [[], []]
            for t0, t1 in [(0, 7), (7, 8), (8, 25), (25, 33)]:
                part = transducer_lstm_greedy_decode(
                    enc[:, t0:t1].contiguous(),
                    torch.full((2,), t1 - t0, device="cuda"),
                    h,
                    c,
                    dec_proj,
                    weights,
                    max_sym=3,
                )
                for b, n in enumerate(part.counts.tolist()):
                    hyps[b] += part.tokens[b, :n].tolist()
                h, c, dec_proj = part.h, part.c, part.dec_proj
        assert hyps == _rows(one)[0]
        assert torch.equal(h, one.h) and torch.equal(c, one.c)
        assert torch.equal(dec_proj, one.dec_proj)


class TestTracking:
    def test_posteriors_match_the_loop(self):
        model = _model(torch.float16, seed=9)
        enc = _enc(2, 24, torch.float16, seed=9)
        lens = torch.tensor([24, 15], device="cuda")
        strat = transducer_strategy(model, fused=False)
        state, dec_proj = strat._init_state(2, enc.device)
        with torch.no_grad():
            _, marks, _, _ = strat._greedy_loop(enc, lens, state, dec_proj, track=True)
        res = _fused(model, enc, lens, track=True)
        assert res.probs is not None
        for b, row in enumerate(marks):
            got = res.probs[b, : len(row)].tolist()
            assert got == pytest.approx([p for _, p in row], rel=2e-2, abs=1e-4)


class TestLengthsAndCapacity:
    def test_lengths_are_clamped(self):
        model = _model(torch.float16, seed=10)
        enc_full = _enc(3, 20, torch.float16, seed=10)
        enc = enc_full[:, :14]
        got = _fused(model, enc, torch.tensor([30, -2, 0], device="cuda"))
        want = _fused(model, enc, torch.tensor([14, 0, 0], device="cuda"))
        _assert_identical(got, want)
        assert got.counts[1:].tolist() == [0, 0]

    def test_a_row_that_never_steps_hands_back_its_state(self):
        model = _model(torch.float16, seed=11)
        enc = _enc(2, 10, torch.float16, seed=11)
        strat = transducer_strategy(model, fused=False)
        (_out, h, c), dec_proj = strat._init_state(2, enc.device)
        res = _fused(model, enc, torch.tensor([10, 0], device="cuda"))
        assert torch.equal(res.h[:, 1], h[:, 1]) and torch.equal(res.c[:, 1], c[:, 1])
        assert torch.equal(res.dec_proj[1], dec_proj[1])

    def test_overflow_is_reported_not_silent(self):
        model = _model(torch.float16, blank_bias=-80.0)
        enc = _enc(2, 10, torch.float16)
        res = _fused(model, enc, torch.tensor([10, 2], device="cuda"), max_sym=3, capacity=8)
        assert res.counts.tolist() == [30, 6]


class TestSupports:
    def test_scope(self):
        model = _model(torch.float16)
        w = _weights(model)
        assert w.supports(torch.zeros(LSTM_GREEDY_MAX_ROWS, 4, HIDDEN, device="cuda").half())
        assert not w.supports(
            torch.zeros(LSTM_GREEDY_MAX_ROWS + 1, 4, HIDDEN, device="cuda").half()
        )
        assert not w.supports(torch.zeros(1, 4, HIDDEN, device="cuda"))  # fp32
        assert not _weights(_model(torch.float32)).supports(
            torch.zeros(1, 4, HIDDEN, device="cuda")
        )

    def test_quantized_joint_declares_nothing(self):
        model = _model(torch.float16)
        model.joiner.head.quant_method = object()
        assert model.joiner.additive_tensors() is None
