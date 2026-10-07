# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.transducer_beam_decode`` (``oasr/functionals/transducer.py``).

The oracle is :func:`~oasr.engine.decode.transducer_beam.beam_search_history`,
the per-frame loop the kernel replaces, with its selection fused
(:func:`oasr.transducer_beam_topk`) so the two order tied candidates the same
way.  The kernel's GEMVs cannot match the loop's library GEMMs bit for bit on
arbitrary weights, so the model and frames come from :mod:`helpers.transducer`,
whose data makes every sum exact -- and the tests demand equal beams, scores and
back-pointers rather than a tolerance.  That data is also full of tied
candidates, so the selection's tie order is checked on every frame.

Exact data is not enough on its own -- the library GEMMs round split-K
partials at some shapes -- so the oracle runs under
:func:`helpers.transducer.exact_projections`.
"""

from __future__ import annotations

import pytest
import torch
from helpers.transducer import JOINER_DIM, exact_projections, exact_transducer, integer_frames

from oasr.engine.decode.transducer_beam import (
    beam_search_history,
    init_beam_state,
    read_walk,
    walk_chunk,
)
from oasr.functionals.transducer import (
    StatelessGreedyWeights,
    beam_walk_buffer,
    transducer_beam_decode,
)

pytestmark = pytest.mark.cuda


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


def _both(model, enc, lengths, k, state=None, cluster=0):
    """``(reference, kernel)``: each ``(context, scores, parents, labels)``."""
    if state is None:
        state = init_beam_state(model.decoder, enc.size(0), k, enc.device)
        state = (state.context, state.scores)
    context, scores = state
    with torch.no_grad():
        enc_proj = model.joiner.encoder_proj(enc)
        with exact_projections(model):
            want = beam_search_history(model, enc_proj, lengths, context, scores, fused=True)
        got = transducer_beam_decode(
            enc_proj, lengths, context, scores, _weights(model), cluster=cluster
        )
    return want, got


def _clusters():
    """Cluster sizes this device launches: a pair needs sm_90+."""
    major, _ = torch.cuda.get_device_capability()
    return (1, 2) if major >= 9 else (1,)


def _assert_same(want, got):
    for name, a, b in zip(("context", "scores", "parents", "labels"), want, got):
        assert torch.equal(a, b), name


class TestMatchesTheFrameLoop:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8])
    @pytest.mark.parametrize(
        "context,group", [(1, 1), (2, 1), (2, 2), (2, 4), (3, 8), (2, 16)], ids=str
    )
    @pytest.mark.parametrize("activation", ["tanh", "relu"])
    def test_beam_scores_and_back_pointers(self, dtype, k, context, group, activation):
        """Every beam width the kernel instantiates (2, 4, 8 rows, with dead rows
        at k = 1, 3, 5), both conv layouts and the run-time group size (16), both
        joiner activations; ragged lengths, including an empty and a one-frame
        utterance."""
        model = exact_transducer(dtype, context=context, group=group, activation=activation)
        enc = integer_frames(5, 23, dtype, seed=k + 10 * context)
        lengths = torch.tensor([23, 0, 1, 17, 9], device="cuda")
        _assert_same(*_both(model, enc, lengths, k))

    @pytest.mark.parametrize(
        "joiner,decoder,vocab",
        [(64, 64, 700), (512, 512, 500), (1024, 64, 37), (64, 1024, 1024), (96, 64, 13)],
        ids=["two_vocab_passes", "icefall_dims", "wide_joiner", "max_vocab", "short_rows"],
    )
    def test_geometries(self, joiner, decoder, vocab):
        """More than one GEMV pass per frame (V > 512: the row maximum spans the
        passes; J > 512: the decoder projection does), the icefall model's dims,
        the largest vocabulary, and one under 32 -- torch's narrower warp
        log-softmax -- with a joiner dim that is no multiple of 64.  One CTA per
        utterance and a cluster pair, each for every instantiated row count
        (beam 1 at the icefall dims sits just past 48 KB of shared memory once
        the static part is counted)."""
        dtype = torch.bfloat16
        model = exact_transducer(dtype, joiner_dim=joiner, dec_dim=decoder, vocab=vocab, seed=3)
        enc = integer_frames(3, 19, dtype, seed=4, dim=joiner)
        lengths = torch.tensor([19, 11, 6], device="cuda")
        for k in (1, 2, 4, 8):
            for cluster in _clusters():
                _assert_same(*_both(model, enc, lengths, k, cluster=cluster))

    def test_chunks_carry_the_beam_exactly(self):
        """Two calls, the second starting from the first's beam, equal one call:
        what streaming does every tick."""
        dtype, k = torch.bfloat16, 4
        model = exact_transducer(dtype, seed=6)
        enc = integer_frames(3, 30, dtype, seed=8)
        whole = torch.tensor([30, 30, 30], device="cuda")
        want, _ = _both(model, enc, whole, k)
        _, first = _both(model, enc[:, :13], torch.full_like(whole, 13), k)
        _, second = _both(
            model, enc[:, 13:], torch.full_like(whole, 17), k, state=(first[0], first[1])
        )
        assert torch.equal(second[0], want[0]) and torch.equal(second[1], want[1])
        assert torch.equal(torch.cat([first[2], second[2]]), want[2])
        assert torch.equal(torch.cat([first[3], second[3]]), want[3])

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_a_cluster_pair_searches_like_one_cta(self, k):
        """The pair splits every GEMV's outputs between two CTAs, but an output's
        arithmetic is the same chain either way: equal beams, scores and
        history, on continuous data where any reordering would show."""
        if 2 not in _clusters():
            pytest.skip("cluster launch needs sm_90+")
        dtype = torch.bfloat16
        model = exact_transducer(dtype, joiner_dim=512, dec_dim=512, vocab=500, seed=k)
        gen = torch.Generator(device="cuda").manual_seed(k)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.shape, device="cuda", generator=gen).to(dtype) * 0.1)
        enc = torch.randn(4, 33, 512, device="cuda", generator=gen).to(dtype)
        lengths = torch.tensor([33, 20, 1, 33], device="cuda")
        st = init_beam_state(model.decoder, 4, k, enc.device)
        with torch.no_grad():
            enc_proj = model.joiner.encoder_proj(enc)
            one, pair = (
                transducer_beam_decode(
                    enc_proj, lengths, st.context, st.scores, _weights(model), cluster=c
                )
                for c in (1, 2)
            )
        _assert_same(one, pair)

    def test_a_row_is_independent_of_its_batch(self):
        """One CTA (or pair) per utterance: a row's beam is the same alone or
        among others, at any batch width -- which is what lets a caller pad the
        batch, and what keeps a cohort's width from moving a stream's result
        (past half the SM count the launch drops from pairs to single CTAs)."""
        dtype, k = torch.bfloat16, 4
        model = exact_transducer(dtype, seed=9)
        enc = integer_frames(6, 21, dtype, seed=2)
        lengths = torch.tensor([21, 4, 15, 0, 21, 9], device="cuda")
        _, batched = _both(model, enc, lengths, k)
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        wide = enc.repeat(sms // 6 + 1, 1, 1)  # past half the SMs: one CTA each
        _, widest = _both(model, wide, lengths.repeat(sms // 6 + 1), k)
        for b in range(6):
            _, solo = _both(model, enc[b : b + 1], lengths[b : b + 1], k)
            assert torch.equal(solo[1][0], batched[1][b]), b
            assert torch.equal(solo[2][:, 0], batched[2][:, b]), b
            assert torch.equal(widest[1][b], batched[1][b]), b


class TestDeviceWalk:
    @pytest.mark.parametrize("k", [2, 4, 8])
    @pytest.mark.parametrize("blank_bias", [8.0, 2.0], ids=["sparse", "dense"])
    def test_walk_is_the_host_walk_of_its_history(self, k, blank_bias):
        """The kernel's walk against :func:`walk_chunk` over the history the same
        launch wrote: roots, and every hypothesis's tokens -- across ragged
        lengths (an empty utterance walks to its own slot with no tokens), a
        chunk long enough for several 32-frame compaction rounds, and an
        emission rate from sparse to nearly every frame."""
        dtype = torch.bfloat16
        model = exact_transducer(dtype, blank_bias=blank_bias, seed=k)
        enc = integer_frames(4, 77, dtype, seed=k)
        lengths = torch.tensor([77, 0, 33, 64], device="cuda")
        st = init_beam_state(model.decoder, 4, k, enc.device)
        walk = beam_walk_buffer(4, k, 77, enc.device)
        with torch.no_grad():
            enc_proj = model.joiner.encoder_proj(enc)
            _, _, parents, labels = transducer_beam_decode(
                enc_proj, lengths, st.context, st.scores, _weights(model), walk=walk
            )
        want = walk_chunk(parents[:, :3], labels[:, :3], 0)
        got = read_walk(walk, k, 77, rows=3)
        assert got.root == want.root
        assert [got.tokens(h) for h in range(3 * k)] == [want.tokens(h) for h in range(3 * k)]
        assert any(len(want.tokens(h)) > 32 for h in range(3 * k)) or blank_bias > 4

    def test_an_empty_chunk_walks_to_its_own_slots(self):
        model = exact_transducer(torch.bfloat16)
        st = init_beam_state(model.decoder, 2, 4, torch.device("cuda"))
        walk = beam_walk_buffer(2, 4, 0, torch.device("cuda"))
        transducer_beam_decode(
            integer_frames(2, 0, torch.bfloat16),
            torch.zeros(2, device="cuda"),
            st.context,
            st.scores,
            _weights(model),
            walk=walk,
        )
        got = read_walk(walk, 4, 0, rows=2)
        assert got.root == [0, 1, 2, 3] * 2 and got.flat == []


class TestInterface:
    def test_destination_passing_and_aliased_state(self):
        """The beam may be updated in place: each CTA reads its utterance's beam
        before it writes any of it."""
        dtype, k = torch.bfloat16, 4
        model = exact_transducer(dtype)
        enc = integer_frames(3, 12, dtype)
        lengths = torch.tensor([12, 5, 0], device="cuda")
        want, _ = _both(model, enc, lengths, k)
        st = init_beam_state(model.decoder, 3, k, enc.device)
        context, scores = st.context.clone(), st.scores.clone()
        parents = torch.empty(12, 3, k, dtype=torch.long, device="cuda")
        labels = torch.empty_like(parents)
        with torch.no_grad():
            enc_proj = model.joiner.encoder_proj(enc)
            out = transducer_beam_decode(
                enc_proj,
                lengths,
                context,
                scores,
                _weights(model),
                out=(context, scores, parents, labels),
            )
        assert out[0] is context and out[2] is parents
        _assert_same(want, (context, scores, parents, labels))

    def test_frames_past_an_utterance_record_stay_and_blank(self):
        model = exact_transducer(torch.bfloat16)
        enc = integer_frames(2, 10, torch.bfloat16)
        _, got = _both(model, enc, torch.tensor([3, 0], device="cuda"), 4)
        parents, labels = got[2], got[3]
        stay = torch.arange(4, device="cuda").expand(7, 4)
        assert torch.equal(parents[3:, 0], stay) and bool((labels[3:, 0] == 0).all())
        assert torch.equal(parents[:, 1], torch.arange(4, device="cuda").expand(10, 4))

    def test_empty_batch_and_zero_frames(self):
        model = exact_transducer(torch.bfloat16)
        w = _weights(model)
        for B, T in ((0, 5), (3, 0)):
            beam = (
                torch.zeros(B, 4, 2, dtype=torch.long, device="cuda"),
                torch.zeros(B, 4, device="cuda"),
            )
            enc = integer_frames(B, T, torch.bfloat16)
            context, scores, parents, labels = transducer_beam_decode(
                enc, torch.full((B,), T, device="cuda"), *beam, w
            )
            assert parents.shape == (T, B, 4) and labels.shape == (T, B, 4)
            assert torch.equal(context, beam[0]) and torch.equal(scores, beam[1])


class TestScope:
    def test_supports_beam_bounds(self):
        model = exact_transducer(torch.bfloat16)
        w = _weights(model)
        enc_proj = integer_frames(1, 4, torch.bfloat16)
        assert w.supports_beam(enc_proj, 1) and w.supports_beam(enc_proj, 8)
        assert not w.supports_beam(enc_proj, 9)
        assert not w.supports_beam(enc_proj.float(), 4)

    def test_vocabulary_past_the_warp_softmax_is_refused(self):
        model = exact_transducer(torch.bfloat16, vocab=1025)
        enc_proj = integer_frames(1, 4, torch.bfloat16)
        assert not _weights(model).supports_beam(enc_proj, 4)

    def test_shared_memory_fit_predicts_the_launch(self):
        """``supports_beam`` asks the kernel module, which counts its static
        shared memory too: on either side of where eight rows' working set
        leaves the device, the answer is what the launch then does."""
        dtype = torch.bfloat16
        enc = integer_frames(1, 1, dtype, dim=8)

        def fits(dim):
            model = exact_transducer(dtype, joiner_dim=dim, dec_dim=dim, seed=1)
            return _weights(model).supports_beam(enc.new_zeros(1, 1, dim), 8)

        edge = next((d for d in range(1600, 0, -8) if fits(d)), None)
        assert edge is not None
        for dim, served in ((edge, True), (edge + 8, False)):
            if dim > 1600:
                continue  # a device with more shared memory than the scan covers
            model = exact_transducer(dtype, joiner_dim=dim, dec_dim=dim, seed=1)
            x = integer_frames(2, 6, dtype, dim=dim)
            assert _weights(model).supports_beam(x, 8) is served
            lengths = torch.tensor([6, 3], device="cuda")
            if served:
                _assert_same(*_both(model, x, lengths, 8))
            else:
                with pytest.raises(Exception, match="beam decode failed"):
                    _both(model, x, lengths, 8)

    def test_fp32_is_refused_by_the_launcher(self):
        model = exact_transducer(torch.float32)
        w = _weights(model)
        st = init_beam_state(model.decoder, 1, 4, torch.device("cuda"))
        enc = torch.zeros(1, 3, JOINER_DIM, device="cuda")
        assert not w.supports_beam(enc, 4)
        with pytest.raises(Exception, match="FP16 and BF16"):
            transducer_beam_decode(enc, torch.tensor([3], device="cuda"), st.context, st.scores, w)
