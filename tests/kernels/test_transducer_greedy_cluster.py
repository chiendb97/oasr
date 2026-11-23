# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.transducer_greedy_decode(..., cluster=C)``: the cluster kernel
(``include/oasr/transducer/greedy_cluster.cuh``).

Its contract is stronger than the op-by-op loop's: an output's arithmetic is the
one-CTA kernel's chain, so the two must agree **bit for bit on arbitrary
weights**, not only on the exact-arithmetic data of ``helpers.transducer``.  That
is the oracle here -- random weights make every sum inexact, so only the same
fmaf chain in the same K-slice order, summed in the same slice order, passes --
and it is what keeps a row's result independent of which kernel (and cluster
size) its batch was given.  The exact-data model then ties both to the loop.
"""

from __future__ import annotations

import pytest
import torch
from helpers.transducer import exact_transducer, integer_frames, transducer_strategy

from oasr.functionals.transducer import (
    StatelessGreedyResult,
    StatelessGreedyWeights,
    transducer_greedy_decode,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="thread-block clusters need sm_90+",
    ),
]

CLUSTERS = [2, 4, 8, 16]


def _random_weights(J, D, V, ctx, group, activation, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(dtype)

    out_b = r(V, scale=0.1)
    out_b[0] += 3.0  # blank-heavy, so rows advance as well as emit
    conv = None
    if ctx > 1:
        conv = r(ctx, 1, D, scale=ctx**-0.5) if group == 1 else r(D, 1, ctx, group)
    return StatelessGreedyWeights.prepare(
        output_weight=r(V, J, scale=J**-0.5),
        output_bias=out_b,
        vocab=V,
        activation=activation,
        embedding=r(V, D),
        conv_weight=conv,
        context=ctx,
        group=group,
        decoder_proj_weight=r(J, D, scale=D**-0.5),
        decoder_proj_bias=r(J, scale=0.1),
        blank=0,
    )


def _random_batch(B, T, J, ctx, dtype, seed, lengths=None):
    g = torch.Generator(device="cuda").manual_seed(seed + 100)
    enc = torch.randn(B, T, J, device="cuda", generator=g).to(dtype)
    if lengths is None:
        lengths = torch.randint(T // 2, T + 1, (B,), device="cuda", generator=g)
    window = torch.full((B, ctx), -1, dtype=torch.long, device="cuda")
    window[:, -1] = 0
    dec = torch.randn(B, J, device="cuda", generator=g).to(dtype)
    return enc, lengths, window, dec


def _assert_same(a: StatelessGreedyResult, b: StatelessGreedyResult, probs=False):
    assert torch.equal(a.counts, b.counts)
    for row, n in enumerate(b.counts.tolist()):
        n = min(n, b.tokens.size(1))
        assert torch.equal(a.tokens[row, :n], b.tokens[row, :n])
        assert torch.equal(a.frames[row, :n], b.frames[row, :n])
        if probs:
            assert a.probs is not None and b.probs is not None
            torch.testing.assert_close(a.probs[row, :n], b.probs[row, :n], rtol=1e-5, atol=1e-7)
    assert torch.equal(a.window, b.window)
    assert torch.equal(a.dec_proj, b.dec_proj)


# (J, D, V, context, group, activation).  The icefall Zipformer shape; a wide
# vocabulary over a depthwise predictor, whose two-CTA cluster owns 18 output
# blocks per CTA (more than one per thread of the epilogue); and a tiny one
# whose 16-CTA cluster leaves 14 CTAs without a block of the vocabulary.
GEOMETRIES = [
    (512, 512, 500, 2, 4, "tanh"),
    (640, 320, 1100, 2, 1, "relu"),
    (64, 64, 37, 3, 4, "tanh"),
]


class TestMatchesOneCtaKernel:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("geometry", GEOMETRIES, ids=["icefall", "wide-vocab", "tiny"])
    def test_random_weights_bit_identical(self, dtype, geometry):
        J, D, V, ctx, group, act = geometry
        weights = _random_weights(J, D, V, ctx, group, act, dtype, seed=1)
        enc, lengths, window, dec = _random_batch(6, 96, J, ctx, dtype, seed=1)
        with torch.no_grad():
            ref = transducer_greedy_decode(enc, lengths, window, dec, weights, max_sym=5, cluster=1)
            for c in CLUSTERS:
                got = transducer_greedy_decode(
                    enc, lengths, window, dec, weights, max_sym=5, cluster=c
                )
                _assert_same(got, ref)
        # Both branches must be exercised: frames that emit and frames that do not.
        emitted = int(ref.counts.sum())
        assert 0 < emitted < int(lengths.sum()) * 5

    def test_auto_picks_the_same_answer(self):
        weights = _random_weights(512, 512, 500, 2, 4, "tanh", torch.float16, seed=2)
        for B in (1, 3, 40):  # cluster, cluster, one CTA
            enc, lengths, window, dec = _random_batch(B, 64, 512, 2, torch.float16, seed=B)
            with torch.no_grad():
                ref = transducer_greedy_decode(
                    enc, lengths, window, dec, weights, max_sym=10, cluster=1
                )
                got = transducer_greedy_decode(
                    enc, lengths, window, dec, weights, max_sym=10, cluster=0
                )
            _assert_same(got, ref)


def _model_weights(model):
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


def _exact(dtype, **kwargs):
    model = exact_transducer(dtype, **kwargs)
    return model, transducer_strategy(model), _model_weights(model)


LENGTHS = [24, 17, 0, 1, 24, 9, 13, 2]


class TestMatchesOpByOpLoop:
    @pytest.mark.parametrize("cluster", [2, 16])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "context,group",
        [(1, 1), (2, 1), (2, 4), (3, 4)],
        ids=["ctx1", "ctx2-depthwise", "ctx2-group4", "ctx3-group4"],
    )
    def test_tokens_frames_and_state(self, cluster, dtype, context, group):
        model, strat, weights = _exact(dtype, context=context, group=group, seed=3)
        enc = integer_frames(len(LENGTHS), max(LENGTHS), dtype, seed=1)
        lengths = torch.tensor(LENGTHS, device="cuda")
        state, dec_proj = strat._init_state(enc.size(0), enc.device)
        saved, strat._fused_enabled = strat._fused_enabled, False
        try:
            with torch.no_grad():
                hyps, marks, want_state, want_dec = strat._greedy_loop(
                    enc, lengths, state, dec_proj, track=True
                )
        finally:
            strat._fused_enabled = saved
        with torch.no_grad():
            res = transducer_greedy_decode(
                model.joiner.encoder_proj(enc),
                lengths,
                state,
                dec_proj,
                weights,
                max_sym=strat._max_sym,
                cluster=cluster,
            )
        counts = res.counts.tolist()
        assert [res.tokens[b, :n].tolist() for b, n in enumerate(counts)] == hyps
        assert [res.frames[b, :n].tolist() for b, n in enumerate(counts)] == [
            [f for f, _ in row] for row in marks
        ]
        assert torch.equal(res.window, want_state)
        assert torch.equal(res.dec_proj, want_dec)

    def test_max_sym_caps_emissions_per_frame(self):
        model, strat, weights = _exact(torch.bfloat16, blank_bias=-64.0)
        enc = integer_frames(3, 10, torch.bfloat16, seed=1)
        lengths = torch.tensor([10, 4, 7], device="cuda")
        state, dec_proj = strat._init_state(3, enc.device)
        with torch.no_grad():
            res = transducer_greedy_decode(
                model.joiner.encoder_proj(enc),
                lengths,
                state,
                dec_proj,
                weights,
                max_sym=strat._max_sym,
                cluster=16,
            )
        assert res.counts.tolist() == [3 * n for n in (10, 4, 7)]


class TestTracking:
    @pytest.mark.parametrize("cluster", [4, 16])
    def test_posteriors_match_the_one_cta_kernel(self, cluster):
        """Same tokens and frames; the posterior's logsumexp is combined in
        another order, so it may move in the last bits only."""
        weights = _random_weights(512, 512, 500, 2, 4, "tanh", torch.bfloat16, seed=4)
        enc, lengths, window, dec = _random_batch(4, 80, 512, 2, torch.bfloat16, seed=4)
        with torch.no_grad():
            ref = transducer_greedy_decode(
                enc, lengths, window, dec, weights, max_sym=10, track=True, cluster=1
            )
            got = transducer_greedy_decode(
                enc, lengths, window, dec, weights, max_sym=10, track=True, cluster=cluster
            )
        assert int(ref.counts.sum()) > 0
        _assert_same(got, ref, probs=True)


class TestState:
    def test_chunks_carry_state_exactly(self):
        weights = _random_weights(512, 512, 500, 2, 4, "tanh", torch.float16, seed=5)
        enc, _, window, dec = _random_batch(2, 48, 512, 2, torch.float16, seed=5)
        full = torch.tensor([48, 48], device="cuda")
        with torch.no_grad():
            one = transducer_greedy_decode(enc, full, window, dec, weights, max_sym=10, cluster=1)
            hyps = [[], []]
            for t0, t1 in [(0, 9), (9, 10), (10, 33), (33, 48)]:
                part = transducer_greedy_decode(
                    enc[:, t0:t1].contiguous(),
                    torch.full((2,), t1 - t0, device="cuda"),
                    window,
                    dec,
                    weights,
                    max_sym=10,
                    cluster=16,
                )
                for b, n in enumerate(part.counts.tolist()):
                    hyps[b] += part.tokens[b, :n].tolist()
                window, dec = part.window, part.dec_proj
        assert hyps == [one.tokens[b, :n].tolist() for b, n in enumerate(one.counts.tolist())]
        assert torch.equal(window, one.window) and torch.equal(dec, one.dec_proj)

    def test_state_may_alias_its_outputs(self):
        weights = _random_weights(512, 512, 500, 2, 4, "tanh", torch.float16, seed=6)
        enc, lengths, window, dec = _random_batch(3, 40, 512, 2, torch.float16, seed=6)
        with torch.no_grad():
            ref = transducer_greedy_decode(enc, lengths, window, dec, weights, max_sym=10)
            out = StatelessGreedyResult(
                tokens=torch.empty_like(ref.tokens),
                frames=torch.empty_like(ref.frames),
                probs=None,
                counts=torch.empty_like(ref.counts),
                window=window,
                dec_proj=dec,
            )
            transducer_greedy_decode(
                enc, lengths, window, dec, weights, max_sym=10, cluster=16, out=out
            )
        assert torch.equal(window, ref.window) and torch.equal(dec, ref.dec_proj)
        assert torch.equal(out.counts, ref.counts)


class TestLengthsAndCapacity:
    def test_lengths_are_clamped_like_the_one_cta_kernel(self):
        """Zero, negative and past-T lengths: a row the cluster never steps
        still hands back its state, and a long length reads no frame past T."""
        weights = _random_weights(512, 512, 500, 2, 4, "tanh", torch.float16, seed=7)
        enc, _, window, dec = _random_batch(4, 30, 512, 2, torch.float16, seed=7)
        lengths = torch.tensor([0, -3, 45, 30], device="cuda")
        with torch.no_grad():
            ref = transducer_greedy_decode(
                enc, lengths, window, dec, weights, max_sym=10, cluster=1
            )
            got = transducer_greedy_decode(
                enc, lengths, window, dec, weights, max_sym=10, cluster=16
            )
        _assert_same(got, ref)
        assert ref.counts[:2].tolist() == [0, 0]
        assert torch.equal(got.dec_proj[:2], dec[:2]) and torch.equal(got.window[:2], window[:2])

    def test_overflow_is_reported_not_silent(self):
        model, strat, weights = _exact(torch.bfloat16, blank_bias=-64.0)
        enc = integer_frames(2, 10, torch.bfloat16, seed=1)
        state, dec_proj = strat._init_state(2, enc.device)
        with torch.no_grad():
            res = transducer_greedy_decode(
                model.joiner.encoder_proj(enc),
                torch.tensor([10, 2], device="cuda"),
                state,
                dec_proj,
                weights,
                max_sym=strat._max_sym,
                capacity=8,
                cluster=8,
            )
        assert res.counts.tolist() == [30, 6]


class TestScope:
    def test_a_vocabulary_past_the_key_is_refused_and_auto_keeps_one_cta(self):
        """The cluster kernel packs a token into 16 bits of its candidate key."""
        V = 70000
        weights = _random_weights(64, 64, V, 1, 1, "tanh", torch.float16, seed=8)
        enc, lengths, window, dec = _random_batch(1, 8, 64, 1, torch.float16, seed=8)
        with torch.no_grad():
            with pytest.raises(Exception, match="stateless greedy decode failed"):
                transducer_greedy_decode(enc, lengths, window, dec, weights, max_sym=3, cluster=16)
            ref = transducer_greedy_decode(enc, lengths, window, dec, weights, max_sym=3, cluster=1)
            got = transducer_greedy_decode(enc, lengths, window, dec, weights, max_sym=3)
        _assert_same(got, ref)

    def test_an_unsupported_cluster_size_is_refused(self):
        weights = _random_weights(64, 64, 37, 1, 1, "tanh", torch.float16, seed=9)
        enc, lengths, window, dec = _random_batch(1, 8, 64, 1, torch.float16, seed=9)
        with pytest.raises(Exception, match="cluster must be"):
            transducer_greedy_decode(enc, lengths, window, dec, weights, max_sym=3, cluster=3)
