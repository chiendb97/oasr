# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.transducer_greedy_decode`` (``oasr/functionals/transducer.py``).

The oracle is :class:`~oasr.engine.decode.transducer.TransducerDecodeStrategy`'s
own op-by-op greedy loop on the same model.  The two cannot agree bit for bit on
arbitrary weights, so the model and frames come from :mod:`helpers.transducer`,
whose data makes every sum in both paths exact -- and the tests demand identical
tokens, frames and final predictor state rather than a tolerance.  Each was seen
to fail against a deliberately broken kernel (``max_sym`` off by one, an
unshifted label window, a prefetch buffer that never flips).
"""

from __future__ import annotations

import pytest
import torch
from helpers.transducer import (
    BLANK,
    DEC_DIM,
    JOINER_DIM,
    VOCAB,
    exact_transducer,
    integer_frames as _int_frames,
    transducer_strategy,
)

from oasr.functionals.transducer import (
    StatelessGreedyResult,
    StatelessGreedyWeights,
    transducer_greedy_capacity,
    transducer_greedy_decode,
)
from oasr.layers import Linear

pytestmark = pytest.mark.cuda

J, D = JOINER_DIM, DEC_DIM


def _model(dtype, **kwargs):
    return exact_transducer(dtype, **kwargs)


def _enc(B, T, dtype, seed=1):
    return _int_frames(B, T, dtype, seed=seed)


def _strategy(model, **options):
    return transducer_strategy(model, **options)


def _weights(model):
    pred = model.decoder.stateless_tensors()
    join = model.joiner.additive_tensors()
    return StatelessGreedyWeights.prepare(
        output_weight=join.output_weight,
        output_bias=join.output_bias,
        vocab=join.vocab_size,
        activation=join.activation,
        embedding=pred.embedding,
        conv_weight=pred.conv_weight,
        context=pred.context_size,
        group=pred.group_size,
        decoder_proj_weight=join.decoder_proj_weight,
        decoder_proj_bias=join.decoder_proj_bias,
        blank=pred.blank_id,
    )


def _loop(strat, enc, lengths, track=False):
    """The op-by-op loop's (hyps, marks, state, dec_proj) from blank state."""
    state, dec_proj = strat._init_state(enc.size(0), enc.device)
    saved, strat._fused_enabled = strat._fused_enabled, False
    try:
        with torch.no_grad():
            return strat._greedy_loop(enc, lengths, state, dec_proj, track=track)
    finally:
        strat._fused_enabled = saved


def _fused(strat, enc, lengths, track=False, rows_per_cta=0, capacity=None):
    joiner = strat._model.joiner
    state, dec_proj = strat._init_state(enc.size(0), enc.device)
    with torch.no_grad():
        res = transducer_greedy_decode(
            joiner.encoder_proj(enc),
            lengths,
            state,
            dec_proj,
            _weights(strat._model),
            max_sym=strat._max_sym,
            track=track,
            rows_per_cta=rows_per_cta,
            capacity=capacity,
        )
    torch.cuda.synchronize()
    counts = res.counts.tolist()
    hyps = [res.tokens[b, : counts[b]].tolist() for b in range(len(counts))]
    frames = [res.frames[b, : counts[b]].tolist() for b in range(len(counts))]
    return res, hyps, frames


LENGTHS = [24, 17, 0, 1, 24, 9, 13, 2]


class TestMatchesOpByOpLoop:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "context,group",
        [(1, 1), (2, 1), (2, 4), (3, 4)],
        ids=["ctx1", "ctx2-depthwise", "ctx2-group4", "ctx3-group4"],
    )
    @pytest.mark.parametrize("activation", ["tanh", "relu"])
    def test_tokens_frames_and_state(self, dtype, context, group, activation):
        model = _model(dtype, context=context, group=group, activation=activation)
        strat = _strategy(model)
        enc = _enc(len(LENGTHS), max(LENGTHS), dtype)
        lengths = torch.tensor(LENGTHS, device="cuda")

        hyps, marks, state, dec_proj = _loop(strat, enc, lengths, track=True)
        res, f_hyps, f_frames = _fused(strat, enc, lengths)

        assert f_hyps == hyps
        assert f_frames == [[f for f, _ in row] for row in marks]
        assert torch.equal(res.window, state)
        assert torch.equal(res.dec_proj, dec_proj)
        # The data must exercise both branches -- frames that emit and frames
        # that do not -- or "identical" says nothing about the bookkeeping.
        emitting = sum(len({f for f, _ in row}) for row in marks)
        assert 0 < emitting < sum(LENGTHS)

    def test_max_sym_caps_emissions_per_frame(self):
        # A hugely negative blank bias: every step wants to emit, so only the cap
        # ever advances a frame.
        model = _model(torch.bfloat16, blank_bias=-64.0)
        strat = _strategy(model)
        enc = _enc(3, 10, torch.bfloat16)
        lengths = torch.tensor([10, 4, 7], device="cuda")
        hyps, _, _, _ = _loop(strat, enc, lengths)
        _, f_hyps, f_frames = _fused(strat, enc, lengths)
        assert f_hyps == hyps
        assert [len(h) for h in f_hyps] == [3 * n for n in (10, 4, 7)]
        assert all(row.count(t) == 3 for row in f_frames for t in set(row))


class TestLaunchShape:
    def test_rows_per_cta_agree(self):
        model = _model(torch.bfloat16)
        strat = _strategy(model)
        enc = _enc(5, 20, torch.bfloat16, seed=3)  # odd: the last 2-row CTA is half empty
        lengths = torch.tensor([20, 0, 13, 20, 5], device="cuda")
        r1, h1, f1 = _fused(strat, enc, lengths, rows_per_cta=1)
        r2, h2, f2 = _fused(strat, enc, lengths, rows_per_cta=2)
        assert (h1, f1) == (h2, f2)
        assert torch.equal(r1.window, r2.window) and torch.equal(r1.dec_proj, r2.dec_proj)

    def test_single_row(self):
        model = _model(torch.float16)
        strat = _strategy(model)
        enc = _enc(1, 7, torch.float16)
        hyps, _, _, _ = _loop(strat, enc, torch.tensor([7], device="cuda"))
        _, f_hyps, _ = _fused(strat, enc, torch.tensor([7], device="cuda"))
        assert f_hyps == hyps

    def test_empty_batch_is_a_no_op(self):
        weights = _weights(_model(torch.float16))
        res = transducer_greedy_decode(
            torch.zeros(0, 5, J, dtype=torch.float16, device="cuda"),
            torch.zeros(0, dtype=torch.long, device="cuda"),
            torch.zeros(0, 2, dtype=torch.long, device="cuda"),
            torch.zeros(0, J, dtype=torch.float16, device="cuda"),
            weights,
            max_sym=3,
        )
        assert res.counts.numel() == 0 and res.tokens.shape[0] == 0


def _replay_posteriors(model, enc, length, tokens, frames):
    """fp32 posteriors of each emission, replayed on the model's own layers.

    The logits are exact here (integer data), so the only difference left from
    the kernel is the logsumexp itself: this one is ``torch.logsumexp`` in fp32,
    the kernel's an online one with fast-math exp/log.
    """
    decoder, joiner = model.decoder, model.joiner
    window = torch.full((1, decoder.context_size), BLANK, dtype=torch.long, device="cuda")
    enc_proj = joiner.encoder_proj(enc)
    out = []
    with torch.no_grad():
        for tok, t in zip(tokens, frames):
            dec_proj = joiner.decoder_proj(decoder(window))
            logits = joiner(enc_proj[t : t + 1], dec_proj, project_input=False)[0].float()
            out.append(float((logits[tok] - torch.logsumexp(logits, dim=-1)).exp()))
            window = torch.cat([window[:, 1:], torch.tensor([[tok]], device="cuda")], dim=1)
    return out


class TestTracking:
    def test_posteriors_are_the_fp32_softmax_of_the_emitted_token(self):
        model = _model(torch.bfloat16)
        strat = _strategy(model)
        enc = _enc(4, 16, torch.bfloat16, seed=5)
        lengths = torch.tensor([16, 11, 16, 3], device="cuda")
        joiner = model.joiner
        state, dec_proj = strat._init_state(4, enc.device)
        with torch.no_grad():
            res = transducer_greedy_decode(
                joiner.encoder_proj(enc),
                lengths,
                state,
                dec_proj,
                _weights(model),
                max_sym=strat._max_sym,
                track=True,
            )
        counts = res.counts.tolist()
        assert sum(counts) > 0
        for b in range(4):
            toks = res.tokens[b, : counts[b]].tolist()
            frames = res.frames[b, : counts[b]].tolist()
            got = res.probs[b, : counts[b]].tolist()
            want = _replay_posteriors(model, enc[b], int(lengths[b]), toks, frames)
            assert got == pytest.approx(want, rel=1e-4, abs=1e-6)


class TestStatefulChunks:
    def test_chunks_carry_state_exactly(self):
        """Decoding in chunks from the carried (window, dec_proj) == one call."""
        model = _model(torch.bfloat16)
        strat = _strategy(model)
        weights = _weights(model)
        enc = model.joiner.encoder_proj(_enc(2, 24, torch.bfloat16, seed=7))
        full = torch.tensor([24, 24], device="cuda")
        state, dec_proj = strat._init_state(2, enc.device)
        with torch.no_grad():
            one = transducer_greedy_decode(enc, full, state, dec_proj, weights, max_sym=3)
            hyps = [[], []]
            for t0, t1 in [(0, 5), (5, 6), (6, 17), (17, 24)]:
                part = transducer_greedy_decode(
                    enc[:, t0:t1].contiguous(),
                    torch.full((2,), t1 - t0, device="cuda"),
                    state,
                    dec_proj,
                    weights,
                    max_sym=3,
                )
                counts = part.counts.tolist()
                for b in range(2):
                    hyps[b] += part.tokens[b, : counts[b]].tolist()
                state, dec_proj = part.window, part.dec_proj
        counts = one.counts.tolist()
        assert hyps == [one.tokens[b, : counts[b]].tolist() for b in range(2)]
        assert torch.equal(state, one.window) and torch.equal(dec_proj, one.dec_proj)

    def test_state_may_alias_its_outputs(self):
        """``out.window`` / ``out.dec_proj`` may be the input state (decode in place)."""
        model = _model(torch.float16)
        strat = _strategy(model)
        weights = _weights(model)
        enc = model.joiner.encoder_proj(_enc(3, 12, torch.float16, seed=9))
        lengths = torch.tensor([12, 6, 12], device="cuda")
        state, dec_proj = strat._init_state(3, enc.device)
        with torch.no_grad():
            ref = transducer_greedy_decode(enc, lengths, state, dec_proj, weights, max_sym=3)
            out = StatelessGreedyResult(
                tokens=torch.empty_like(ref.tokens),
                frames=torch.empty_like(ref.frames),
                probs=None,
                counts=torch.empty_like(ref.counts),
                window=state,
                dec_proj=dec_proj,
            )
            res = transducer_greedy_decode(
                enc, lengths, state, dec_proj, weights, max_sym=3, out=out
            )
        assert res is out
        assert torch.equal(state, ref.window) and torch.equal(dec_proj, ref.dec_proj)
        assert torch.equal(res.counts, ref.counts)
        for b, n in enumerate(ref.counts.tolist()):
            assert torch.equal(res.tokens[b, :n], ref.tokens[b, :n])


class TestCapacity:
    def test_overflow_is_reported_not_silent(self):
        model = _model(torch.bfloat16, blank_bias=-64.0)  # emits max_sym per frame
        strat = _strategy(model)
        enc = _enc(2, 10, torch.bfloat16)
        lengths = torch.tensor([10, 2], device="cuda")
        res, hyps, _ = _fused(strat, enc, lengths, capacity=8)
        assert res.counts.tolist() == [30, 6]  # the true counts, past the buffer
        assert len(hyps[0]) == 8 and len(hyps[1]) == 6  # what fit

    def test_default_capacity_is_linear_in_frames(self):
        assert transducer_greedy_capacity(0) == 32
        assert transducer_greedy_capacity(250) == 532


class TestRejectsWhatItCannotServe:
    def test_fp32_is_refused_by_the_launcher(self):
        model = _model(torch.float32)
        weights = _weights(model)
        enc = _enc(2, 4, torch.float32)
        assert not weights.supports(enc)
        state = torch.zeros(2, 2, dtype=torch.long, device="cuda")
        dec_proj = enc[:, 0].contiguous()
        with pytest.raises(Exception, match="FP16 and BF16"):
            transducer_greedy_decode(
                enc, torch.tensor([4, 4], device="cuda"), state, dec_proj, weights, max_sym=3
            )

    def test_misaligned_joiner_dim_is_unsupported(self):
        model = _model(torch.bfloat16)
        weights = _weights(model)
        assert weights.supports(_enc(1, 2, torch.bfloat16))
        assert not weights.supports(torch.zeros(1, 2, J - 4, device="cuda", dtype=torch.bfloat16))

    def test_k_major_layout(self):
        model = _model(torch.bfloat16)
        w = _weights(model)
        assert tuple(w.w_out_t.shape) == (J, 512) and tuple(w.w_dp_t.shape) == (D, 512)
        out = model.joiner.output_linear.weight
        assert torch.equal(w.w_out_t[:, :VOCAB], out[:VOCAB].t())
        assert not w.w_out_t[:, VOCAB:].any()


def test_linear_layers_route_through_the_layer_waist():
    """The model in these tests is built from oasr.layers, like a real one."""
    model = _model(torch.bfloat16)
    assert isinstance(model.joiner.output_linear, Linear)
