# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron ASR (FastConformer + RNN-T) tests.

Four tiers, deliberately separated by what each one can actually prove:

**Structure, no checkpoint** — the pieces that are easy to get subtly wrong and
whose wrongness a parity test would hide behind a tolerance: the Transformer-XL
relative shift, the ``chunked_limited`` window, the LSTM predictor's protocol,
and the fact that the start-of-sequence state is *not* zeros.

**Streaming equals offline, no checkpoint** — a chunked pass against a
whole-utterance one on random weights, fp32, CPU, with ``rtol=0``.  Every cache in
the streaming path is a *replacement for padding an offline pass applies*, so
equality is the right bar and a tolerance would hide exactly the failures worth
catching: a stride-grid phase shift, a mis-sized left context, a position table
built for the wrong length.  This tier does not need the real checkpoint and it is
where a regression will be diagnosable.

**Frontend parity** — the ``nemotron_logmel`` recipe against HuggingFace's own
feature extractor.  Bit-exact once the mel filterbank is the same table, which is
worth pinning separately: a feature-convention bug cancels in every downstream
parity test (both sides get the same features) and only shows up as WER.  That is
how the ``audio_scale`` defect shipped.

**Real-checkpoint parity** — encoder tensors and greedy *token ids* against
``transformers``, then the engine end to end in both service modes.  Token
exactness is the strong claim here; the tensor comparisons are what localise a
failure when it breaks.  The streaming class compares *words* rather than exact
strings against the offline engine, and says why: sentence-final punctuation is
allowed to differ, because offline uniquely produces the encoder frame its
utterance-end right pad creates while a stream's equivalent frames see the
finalize silence pad instead.

The end-to-end accuracy numbers live in ``ci/wer-reference.json`` — one entry per
service mode, on the same manifest and denominator — and are checked by
``tests/test_accuracy.py``, not here.
"""

from __future__ import annotations

import math

import assets
import pytest
import torch

from oasr.models.nemotron import (
    NemotronEncoderConfig,
    NemotronModel,
    NemotronModelConfig,
    NemotronRnntPredictor,
    chunked_limited_mask,
    rel_shift,
    relative_position_embedding,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _tiny_config(**overrides) -> NemotronModelConfig:
    """Small but structurally complete: 8x subsampling still means three stages."""
    encoder = NemotronEncoderConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        conv_kernel_size=9,
        num_mel_bins=32,
        subsampling_conv_channels=16,
        sliding_window=13,
        default_num_lookahead_tokens=3,
    )
    kwargs = {
        "vocab_size": 32,
        "blank_token_id": 31,
        "decoder_hidden_size": 24,
        "num_decoder_layers": 2,
        "num_prompts": 8,
        "prompt_intermediate_size": 48,
        "default_prompt_id": 1,
        "encoder": encoder,
    }
    kwargs.update(overrides)
    return NemotronModelConfig(**kwargs)


# ---------------------------------------------------------------------------
# Structure — no checkpoint, no CUDA
# ---------------------------------------------------------------------------


class TestConfig:
    def test_released_geometry_is_reproduced(self):
        """128 mels through three causally-padded stride-2 stages → 17 bins.

        ``256 * 17 == 4352`` is the released checkpoint's
        ``encoder.subsampling.linear`` input width, so this is the arithmetic that
        decides whether the weight loads at all.
        """
        cfg = NemotronEncoderConfig()
        assert cfg.num_mel_bins == 128
        assert cfg.num_subsampling_layers == 3
        assert cfg.subsampling_out_hidden_size == 4352
        assert cfg.head_dim == 128

    def test_blank_is_the_last_vocabulary_slot(self):
        """Blank is 13087, not 0 — 0 is ``<unk>``.  Getting this backwards would
        make every emission look like a blank and produce an empty transcript."""
        cfg = NemotronModelConfig()
        assert cfg.blank_token_id == cfg.vocab_size - 1
        assert cfg.blank_id == cfg.blank_token_id


class TestRelativeShift:
    def test_shift_gives_key_j_the_embedding_of_distance_i_minus_j(self):
        """Row ``i``, column ``j`` must end up holding source column
        ``(L - 1) + j - i``.

        The unshifted axis runs over relative distances ``L-1 … -(L-1)``, so
        source column ``p`` is distance ``(L-1) - p``; landing at
        ``p = (L-1) + j - i`` therefore means key ``j`` is scored against the
        embedding of distance ``i - j`` — Transformer-XL's ``R_{i-j}``, positive
        into the past.  Checked as an identity on an index-encoded input rather
        than against a second implementation, so the test cannot agree with a copy
        of the same mistake.  Values are ``p + 1`` so the ``F.pad`` zero the shift
        introduces is distinguishable from source column 0.
        """
        length = 6
        positions = 2 * length - 1
        scores = (
            (torch.arange(positions, dtype=torch.float32) + 1)
            .view(1, 1, 1, positions)
            .expand(1, 1, length, positions)
            .contiguous()
        )
        shifted = rel_shift(scores)
        for i in range(length):
            for j in range(length):
                source = (length - 1) + j - i
                assert 0 <= source < positions
                assert shifted[0, 0, i, j].item() == pytest.approx(source + 1)

    def test_shape_is_preserved(self):
        x = torch.randn(2, 3, 7, 13)
        assert rel_shift(x).shape == x.shape


class TestChunkedLimitedMask:
    def test_lookahead_is_within_the_chunk_not_a_fixed_offset(self):
        """A query sees its own chunk — so how much *future* it sees depends on
        where it sits inside the chunk, which is what makes one offline pass equal
        the concatenation of the streaming chunks."""
        right, left = 3, 12  # chunk = 4, left_chunks = 3
        m = chunked_limited_mask(12, 12, left, right, torch.device("cpu"))[0, 0]
        # Query 0 is at the start of chunk 0: it sees keys 0..3 (3 ahead).
        assert m[0].tolist() == [True] * 4 + [False] * 8
        # Query 3 is at the end of chunk 0: it sees the same 4 keys, 0 ahead.
        assert m[3].tolist() == [True] * 4 + [False] * 8
        # Query 4 opens chunk 1: chunks 0 and 1, i.e. keys 0..7.
        assert m[4].tolist() == [True] * 8 + [False] * 4

    def test_left_context_is_bounded_by_whole_chunks(self):
        right, left = 0, 3  # chunk = 1, left_chunks = 3
        m = chunked_limited_mask(8, 8, left, right, torch.device("cpu"))[0, 0]
        for q in range(8):
            allowed = m[q].nonzero().flatten().tolist()
            assert allowed == list(range(max(0, q - 3), q + 1))

    def test_negative_left_context_means_unlimited(self):
        m = chunked_limited_mask(6, 6, -1, 0, torch.device("cpu"))[0, 0]
        assert m.tolist() == torch.ones(6, 6).tril().bool().tolist()

    def test_every_query_keeps_at_least_its_own_chunk(self):
        """No row is empty, so the mask alone never produces a NaN softmax row —
        the only way to get one is padding, which the encoder handles separately."""
        m = chunked_limited_mask(37, 37, 20, 6, torch.device("cpu"))
        assert bool(m.any(dim=-1).all())


class TestRelativePositionEmbedding:
    def test_sin_and_cos_are_interleaved(self):
        """Upstream builds the table with ``stack([sin, cos], -1).reshape(...)``.
        Concatenating instead would be a silent permutation of the positional
        projection's input columns — no shape error, no exception, wrong bias."""
        dim, length = 8, 4
        table = relative_position_embedding(length, dim, torch.device("cpu"), torch.float32)
        assert table.shape == (1, 2 * length - 1, dim)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # Row 0 is relative distance +(length - 1).
        pos = float(length - 1)
        torch.testing.assert_close(table[0, 0, 0::2], torch.sin(pos * inv_freq))
        torch.testing.assert_close(table[0, 0, 1::2], torch.cos(pos * inv_freq))

    def test_positions_run_from_plus_to_minus(self):
        table = relative_position_embedding(5, 4, torch.device("cpu"), torch.float32)
        # The centre row is distance 0: sin 0 = 0, cos 0 = 1.
        centre = table[0, 4]
        torch.testing.assert_close(centre[0::2], torch.zeros(2))
        torch.testing.assert_close(centre[1::2], torch.ones(2))


class TestPredictorProtocol:
    """The four operations that let one greedy loop drive a recurrent predictor."""

    @staticmethod
    def _predictor() -> NemotronRnntPredictor:
        torch.manual_seed(0)
        return NemotronRnntPredictor(vocab_size=16, hidden_size=8, num_layers=2, blank_id=15)

    def test_start_state_is_the_sos_step_not_zeros(self):
        """NeMo runs the LSTM once on the blank embedding (a zero row, via
        ``padding_idx``) from a zero hidden state before the first frame.  Handing
        back zeros instead would drop a constant out of every first-frame joint —
        and would still decode, just worse."""
        p = self._predictor()
        with torch.no_grad():
            p.embedding.weight[15].zero_()
            state = p.init_state(3, torch.device("cpu"))
        prediction = p.predict(state)
        assert prediction.shape == (3, 8)
        assert prediction.abs().max() > 0, "SOS prediction collapsed to zeros"
        # Every row starts identically: the state depends only on the blank token.
        torch.testing.assert_close(prediction[0], prediction[1])

    def test_advance_leaves_non_emitting_rows_untouched(self):
        """The greedy loop steps the whole batch and keeps the result only where a
        row emitted, so a blanking row must come back bit-identical — otherwise its
        next projection silently drifts."""
        p = self._predictor()
        state = p.init_state(4, torch.device("cpu"))
        tokens = torch.tensor([1, 2, 3, 4])
        emit = torch.tensor([True, False, True, False])
        with torch.no_grad():
            new = p.advance(state, tokens, emit)
        for before, after in zip(state, new):
            torch.testing.assert_close(before[..., 1, :], after[..., 1, :])
            torch.testing.assert_close(before[..., 3, :], after[..., 3, :])
        assert not torch.allclose(state[0][0], new[0][0])

    def test_advance_matches_a_per_row_step(self):
        """Batched-and-masked must equal stepping the emitting row alone."""
        p = self._predictor()
        batched = p.init_state(2, torch.device("cpu"))
        with torch.no_grad():
            moved = p.advance(batched, torch.tensor([5, 9]), torch.tensor([True, False]))
            solo = p.advance(
                p.init_state(1, torch.device("cpu")), torch.tensor([5]), torch.tensor([True])
            )
        torch.testing.assert_close(moved[0][:1], solo[0])

    def test_stack_and_unstack_round_trip(self):
        p = self._predictor()
        states = [p.init_state(1, torch.device("cpu")) for _ in range(3)]
        with torch.no_grad():
            states[1] = p.advance(states[1], torch.tensor([7]), torch.tensor([True]))
        split = p.unstack_states(p.stack_states(states))
        assert len(split) == 3
        for original, restored in zip(states, split):
            for a, b in zip(original, restored):
                torch.testing.assert_close(a, b)

    def test_recurrent_state_is_not_a_label_window(self):
        """Which is what tells the decode strategy to refuse beam search rather
        than reorder a state it cannot express."""
        assert self._predictor().label_window_state is False


class TestTinyModel:
    """Built and run on CPU/fp32 — the property the parity oracles depend on."""

    def test_forward_shapes(self):
        cfg = _tiny_config()
        model = NemotronModel.from_config(cfg)
        model.eval()
        frames = 200
        features = torch.randn(2, frames, cfg.encoder.num_mel_bins)
        lengths = torch.tensor([frames, frames // 2])
        with torch.no_grad():
            hidden, out_lengths = model.encode_offline(features, lengths)
        expected = frames
        for _ in range(3):
            expected = expected // 2 + 1
        assert hidden.shape == (2, expected, cfg.encoder.hidden_size)
        assert out_lengths.tolist() == [expected, (((frames // 2) // 2 + 1) // 2 + 1) // 2 + 1]
        assert torch.isfinite(hidden).all()

    def test_joint_shapes_and_vocabulary_width(self):
        cfg = _tiny_config()
        model = NemotronModel.from_config(cfg)
        enc = torch.randn(3, cfg.encoder.hidden_size)
        dec = torch.randn(3, cfg.decoder_hidden_size)
        logits = model.joiner(enc, dec)
        assert logits.shape == (3, cfg.vocab_size)

    def test_joiner_aliases_the_registered_joint(self):
        """A property, not a second attribute: assigning the module twice would
        duplicate every joint tensor in the state dict."""
        model = NemotronModel.from_config(_tiny_config())
        assert model.joiner is model.joint
        keys = [k for k in model.state_dict() if "encoder_proj" in k]
        assert keys == ["joint.encoder_proj.weight", "joint.encoder_proj.bias"]

    def test_prompt_projector_is_optional(self):
        """``num_prompts = 0`` is the sibling ``nemotron_asr_streaming`` layout
        (no language conditioning); the encoder projection then consumes the
        encoder output directly."""
        model = NemotronModel.from_config(_tiny_config(num_prompts=0))
        assert model.prompt_projector is None
        with torch.no_grad():
            hidden, _ = model.encode_offline(torch.randn(1, 120, 32), torch.tensor([120]))
        assert torch.isfinite(hidden).all()

    def test_declared_metadata(self):
        model = NemotronModel.from_config(_tiny_config())
        assert model.default_decode_type == "transducer"
        assert model.capabilities == frozenset({"transducer"})
        assert model.streaming_kind == "paged"
        assert model.head is None
        assert model.blank_id == 31

    def test_cache_spec_declares_everything_a_chunk_carries(self):
        """The engine builds the whole streaming cache from this one descriptor, so
        a missing declaration is a silently reset piece of state rather than an
        error.  Four things: paged K/V geometry, the conv left-context width, the
        per-subsampling-stage tails, and the *trained* attention window that makes
        the engine pre-fill the K/V window instead of growing it."""
        model = NemotronModel.from_config(_tiny_config())
        spec = model.cache_spec
        assert spec is not None
        assert (spec.num_layers, spec.n_kv_head, spec.hidden_dim) == (2, 2, 32)
        assert spec.conv_kernel_size == 9  # -> 8 frames of conv left-context
        assert [s.name for s in spec.stream_states] == [
            "subsample.0",
            "subsample.1",
            "subsample.2",
        ]
        # sliding_window - 1: part of the trained mask, not a cache preference.
        assert spec.fixed_attention_window == 12

    def test_transducer_capability_surface_is_satisfied(self):
        from oasr.models.interfaces import missing_members

        model = NemotronModel.from_config(_tiny_config())
        assert missing_members(model, "transducer") == []


# ---------------------------------------------------------------------------
# Streaming, no checkpoint: does a chunked pass equal a whole-utterance one?
# ---------------------------------------------------------------------------


def _streaming_harness(enc, chunk_size, *, max_batch_size=2):
    """A hand-driven paged streaming rig for one encoder, on CPU fp32.

    Deliberately not the engine: this isolates the *encoder's* four caches from
    scheduling, feature extraction and decode, so a failure here is arithmetic
    rather than plumbing.  fp32 on CPU because the claim being tested is
    equality, and fp16 would replace it with a tolerance.
    """
    from oasr.cache import AttentionCacheManager, BlockPool, CacheConfig, SlotStateCache
    from oasr.cache.cnn_cache import conv_state_spec

    spec = enc.cache_spec
    num_left_chunks = -(-enc.fixed_attention_window // chunk_size) + 1
    cfg = CacheConfig(
        num_layers=spec.num_layers,
        n_kv_head=spec.n_kv_head,
        head_dim=spec.head_dim,
        hidden_dim=spec.hidden_dim,
        kernel_size=spec.conv_kernel_size,
        stream_states=spec.stream_states,
        prefill_kv_window=True,
        chunk_size=chunk_size,
        num_left_chunks=num_left_chunks,
        block_size_frames=chunk_size,
        max_num_blocks=max_batch_size * num_left_chunks + 8,
        max_blocks_per_seq=num_left_chunks + 4,
        max_batch_size=max_batch_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    att = AttentionCacheManager(BlockPool(cfg), cfg)
    states = SlotStateCache(
        [conv_state_spec(cfg), *cfg.stream_states],
        max_batch_size=cfg.max_batch_size,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    return cfg, att, states


def _stream_encoder(enc, cfg, att, states, feats, chunk_size, *, admit_at=None):
    """Feed ``feats (B, T, F)`` chunk by chunk; return ``(B, n_enc, D)`` per row.

    ``admit_at`` optionally staggers admission (``{row: step}``) so a *young*
    stream shares a cohort with a mature one — the case the prefilled window's
    leading zero columns exist for.
    """
    from oasr.cache.cnn_cache import CONV_STATE

    B, total, _ = feats.shape
    window = chunk_size * enc.subsampling_rate
    cache_t1 = cfg.prefilled_cache_frames
    admit_at = admit_at or dict.fromkeys(range(B), 0)
    offsets = [0] * B
    outs: dict = {b: [] for b in range(B)}
    live = []
    for step in range(total // window):
        for b in range(B):
            if admit_at[b] == step:
                att.allocate_stream(b, slot_id=b)
                states.allocate_stream(b, slot_id=b)
                live.append(b)
        if not live:
            continue
        slot_ids = torch.tensor(live, dtype=torch.long)
        att.prepare_chunks_batched(live)
        assert (att.cache_seqlens.index_select(0, slot_ids) == cache_t1).all()
        caches, _, _ = att.get_batched_paged_caches(slot_ids)
        views = states.views(slot_ids)
        xs = torch.stack(
            [
                feats[b, (step - admit_at[b]) * window : (step - admit_at[b] + 1) * window]
                for b in live
            ]
        )
        out = enc.forward_chunk_paged(
            xs,
            torch.tensor([offsets[b] for b in live], dtype=torch.int32),
            caches,
            views[CONV_STATE],
            cache_t1=cache_t1,
            states=views,
        )
        att.commit_chunks_paged_batched(live, out.size(1))
        for i, b in enumerate(live):
            offsets[b] += out.size(1)
            outs[b].append(out[i : i + 1])
    return {b: torch.cat(v, dim=1) for b, v in outs.items() if v}


class TestStreamingEqualsOffline:
    """The gate for the whole streaming path, on arithmetic rather than transcripts.

    Every cache in play is a *replacement for padding an offline pass applies*, so
    "equal" is the right bar and a tolerance would hide exactly the failures worth
    catching — a stride-grid phase shift, a mis-sized left context, a position
    table built for the wrong length.
    """

    @staticmethod
    def _encoder(**overrides):
        from oasr.models.nemotron.encoder import NemotronEncoder

        kwargs = {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "conv_kernel_size": 9,
            "num_mel_bins": 32,
            "subsampling_conv_channels": 16,
            "sliding_window": 13,
            "default_num_lookahead_tokens": 1,
            "max_position_embeddings": 500,
        }
        kwargs.update(overrides)
        cfg = NemotronEncoderConfig(**kwargs)
        torch.manual_seed(0)
        enc = NemotronEncoder(cfg).eval()
        for p in enc.parameters():
            torch.nn.init.normal_(p, std=0.05)
        return enc

    @pytest.mark.parametrize("chunk_size", [2, 4, 8])
    def test_a_chunked_pass_equals_the_whole_utterance(self, chunk_size):
        enc = self._encoder()
        cfg, att, states = _streaming_harness(enc, chunk_size)
        window = chunk_size * enc.subsampling_rate
        feats = torch.randn(2, window * 10, enc.config.num_mel_bins)
        with torch.no_grad():
            offline, _ = enc(feats, torch.tensor([feats.size(1)] * 2, dtype=torch.int32))
            streamed = _stream_encoder(enc, cfg, att, states, feats, chunk_size)
        got = torch.cat([streamed[0], streamed[1]], dim=0)
        n = got.size(1)
        # Offline emits one extra frame per stage from the utterance-end right pad;
        # in a stream those frames arrive with the next chunk.
        assert offline.size(1) >= n
        torch.testing.assert_close(got, offline[:, :n], rtol=0, atol=2e-6)

    @pytest.mark.parametrize("lookahead", [0, 3])
    def test_every_trained_lookahead_streams(self, lookahead):
        """The chunk width *is* ``lookahead + 1``, so this varies the mask geometry
        and the number of trained chunks a step covers at once."""
        enc = self._encoder(
            default_num_lookahead_tokens=lookahead, sliding_window=4 * (lookahead + 1) + 1
        )
        chunk_size = 2 * (lookahead + 1)
        cfg, att, states = _streaming_harness(enc, chunk_size)
        window = chunk_size * enc.subsampling_rate
        feats = torch.randn(1, window * 8, enc.config.num_mel_bins)
        with torch.no_grad():
            offline, _ = enc(feats, torch.tensor([feats.size(1)], dtype=torch.int32))
            streamed = _stream_encoder(enc, cfg, att, states, feats, chunk_size)[0]
        n = streamed.size(1)
        torch.testing.assert_close(streamed, offline[:, :n], rtol=0, atol=2e-6)

    def test_a_stream_admitted_late_still_matches_its_own_offline_pass(self):
        """The prefilled window's leading columns are zero placeholders for history
        a young stream does not have; if the mask let them through, or if the
        position table assumed they were real, this is where it shows."""
        enc = self._encoder()
        chunk_size = 2
        cfg, att, states = _streaming_harness(enc, chunk_size)
        window = chunk_size * enc.subsampling_rate
        steps, late = 14, 5
        feats = torch.randn(2, window * steps, enc.config.num_mel_bins)
        with torch.no_grad():
            streamed = _stream_encoder(
                enc, cfg, att, states, feats, chunk_size, admit_at={0: 0, 1: late}
            )
            # Row 1 only ever saw its first ``steps - late`` windows.
            own = feats[1:2, : window * (steps - late)]
            ref_late, _ = enc(own, torch.tensor([own.size(1)], dtype=torch.int32))
            ref_early, _ = enc(feats[0:1], torch.tensor([feats.size(1)], dtype=torch.int32))
        for got, ref in ((streamed[0], ref_early), (streamed[1], ref_late)):
            n = got.size(1)
            torch.testing.assert_close(got, ref[:, :n], rtol=0, atol=2e-6)


class TestSubsamplingStreamGrid:
    """The cached left context is ``kernel - 1``, and the precondition is real."""

    def test_cache_width_is_kernel_minus_one_not_kernel_minus_stride(self):
        """Upstream uses ``kernel - stride`` plus a first-chunk top-up, which shifts
        the stride grid from chunk two onward (measured ~3 absolute against its own
        offline pass at kernel 3 / stride 2).  With ``S`` a multiple of ``stride`` —
        which every legal chunk is — the frames the next output still needs are
        exactly ``kernel - 1``, uniformly and with no first-chunk case.
        """
        from oasr.models.nemotron.subsampling import NemotronSubsampling

        cfg = NemotronEncoderConfig(num_mel_bins=32, subsampling_conv_channels=16)
        sub = NemotronSubsampling(cfg)
        pad = sub._pad  # noqa: SLF001
        assert pad.stream_left == cfg.subsampling_conv_kernel_size - 1
        assert pad.stream_left != cfg.subsampling_conv_kernel_size - pad.stride

    def test_declared_specs_match_the_stage_shapes(self):
        from oasr.models.nemotron.subsampling import NemotronSubsampling

        cfg = NemotronEncoderConfig(num_mel_bins=128, subsampling_conv_channels=256)
        sub = NemotronSubsampling(cfg)
        specs = sub.state_specs(cfg.num_mel_bins, cfg.subsampling_conv_channels)
        # 128 mels -> 65 -> 33 bins; channels 1 into the stem, then 256.
        assert [s.shape for s in specs] == [(2, 128, 1), (2, 65, 256), (2, 33, 256)]
        assert [s.slot_axis for s in specs] == [0, 0, 0]

    def test_a_misaligned_chunk_is_refused_with_the_multiple_named(self):
        from oasr.models.nemotron.encoder import NemotronEncoder

        enc = NemotronEncoder(NemotronEncoderConfig(default_num_lookahead_tokens=3))
        assert enc.streaming_geometry(4) == (32, 32)
        assert enc.streaming_geometry(8) == (64, 64)
        with pytest.raises(ValueError, match="multiple of 4"):
            enc.streaming_geometry(6)
        with pytest.raises(ValueError, match="multiple of 4"):
            enc.streaming_geometry(0)


class TestFrontendGeometry:
    def test_n_fft_is_the_power_of_two_above_the_window(self):
        from oasr.features import FeatureConfig
        from oasr.features.nemotron import nemotron_stft_geometry

        cfg = FeatureConfig(feature_type="nemotron_logmel", num_mel_bins=128)
        n_fft, hop, win = nemotron_stft_geometry(cfg)
        assert (n_fft, hop, win) == (512, 160, 400)

    def test_frame_count_follows_the_attention_mask_convention(self):
        """``floor(len / hop)`` valid frames, in a tensor one frame wider: a
        centered STFT emits ``floor(len / hop) + 1`` and HuggingFace's mask drops
        the last.  Claiming it would feed the encoder a frame of zero padding as
        if it were audio."""
        from oasr.features import FeatureConfig
        from oasr.features.nemotron import batched_nemotron_logmel

        cfg = FeatureConfig(feature_type="nemotron_logmel", num_mel_bins=128)
        samples = 16000
        wav = torch.randn(2, samples)
        lengths = torch.tensor([samples, samples // 2])
        feats, feat_lengths = batched_nemotron_logmel(wav, lengths, cfg)
        assert feats.shape == (2, samples // 160 + 1, 128)
        assert feat_lengths.tolist() == [samples // 160, (samples // 2) // 160]
        # Frames past a row's count are zeroed, not left at the log floor.
        assert feats[1, feat_lengths[1] :].abs().max() == 0

    def test_registered_as_streamable_with_a_declared_grid(self):
        from oasr.features import FeatureConfig, build_extractor

        cfg = FeatureConfig(feature_type="nemotron_logmel")
        spec = build_extractor(cfg)
        assert spec.supports_streaming is True
        assert spec.window_seconds_attr is None  # cost tracks the real length
        framing = spec.framing_for(cfg)
        # ``span`` is n_fft, not win_length: a centered STFT reads the whole
        # transform width, so keying readiness off the 400-sample window would
        # emit a frame before its last samples had arrived.
        assert (framing.span, framing.hop) == (512, 160)
        # One sample of pre-emphasis history (NeMo pre-emphasises the *signal*),
        # plus n_fft // 2 for the centered grid's implicit left pad.
        assert (framing.history, framing.prefill) == (1, 257)

    def test_streaming_extraction_reproduces_the_offline_grid(self):
        """Bit-exact, which is the only acceptable answer for a frame grid.

        The offline pass is one ``center=True`` STFT over the utterance; a chunked
        caller reproduces it by starting the buffer with ``prefill`` zeros, framing
        past ``history`` context samples, and retaining ``buf[F * hop:]``.  Any
        drift here shifts every frame the encoder sees.
        """
        from oasr.features import FeatureConfig, build_extractor
        from oasr.features.nemotron import batched_nemotron_logmel

        cfg = FeatureConfig(feature_type="nemotron_logmel", num_mel_bins=128)
        framing = build_extractor(cfg).framing_for(cfg)
        spec = build_extractor(cfg)

        torch.manual_seed(0)
        total = 16000
        wav = torch.randn(1, total) * 0.1
        offline, off_len = batched_nemotron_logmel(wav, torch.tensor([total]), cfg)
        n_valid = int(off_len[0])

        for chunk_samples in (1280, 2560, 5120):
            buf = torch.zeros(framing.prefill)
            pos, emitted = 0, []
            while pos < total:
                buf = torch.cat([buf, wav[0, pos : pos + chunk_samples]])
                pos += chunk_samples
                n_frames = framing.frames_for(buf.numel())
                if n_frames <= 0:
                    continue
                feats, lens = spec.extract_streaming(
                    buf.unsqueeze(0), torch.tensor([buf.numel()]), cfg
                )
                assert int(lens[0]) == n_frames
                emitted.append(feats[0, :n_frames])
                buf = buf[n_frames * framing.hop :]
            got = torch.cat(emitted, 0)
            m = min(got.size(0), n_valid)
            # The last frame needs right context and arrives with the engine's
            # finalize silence pad, so streaming is one short mid-stream.
            assert m >= n_valid - 1
            # Tolerance, not equality: the offline side is one STFT over the whole
            # utterance and the streaming side is many short ones, so the mel
            # matmul is a differently-shaped reduction on each and their summation
            # order need not agree.  On CPU the split also depends on the intra-op
            # thread count, i.e. on how many cores the runner happens to have —
            # this assertion was `rtol=0, atol=0` and passed locally and on one
            # GitHub runner while failing on the next by 9.5e-07.
            #
            # 1e-4 is chosen against the thing the test exists to catch: a drift
            # in the framing grid.  Measured on this input, one frame of
            # misalignment moves a value by 0.98 on average and 10.2 at worst,
            # four orders of magnitude above the arithmetic noise, so the bound
            # sits ~100x above the noise and ~10,000x below a real drift.
            torch.testing.assert_close(got[:m], offline[0, :m], rtol=1e-4, atol=1e-4)

    def test_feature_spec_round_trips_and_maps(self):
        from oasr.features import FeatureSpec

        spec = FeatureSpec(
            kind="nemotron_logmel", feature_dim=128, preemphasis=0.97, audio_scale=1.0
        )
        assert FeatureSpec.from_dict(spec.to_dict()) == spec
        cfg = spec.to_feature_config()
        assert cfg.feature_type == "nemotron_logmel"
        assert cfg.num_mel_bins == 128
        assert cfg.preemphasis_coefficient == 0.97
        assert spec.mismatches(cfg) == []


# ---------------------------------------------------------------------------
# Frontend parity against HuggingFace
# ---------------------------------------------------------------------------


@pytest.mark.requires_assets("NEMOTRON_CKPT", "WAV_DIR")
class TestFrontendParity:
    """The one comparison a downstream parity test structurally cannot make.

    Every tensor comparison further down feeds *identical* features to both
    implementations, so a frontend-convention bug cancels on both sides and the
    suite stays green.  ``audio_scale`` shipped through exactly that gap.
    """

    @staticmethod
    def _inputs(n=3):
        import soundfile as sf

        paths = assets.require_wavs(n)
        return [sf.read(p, dtype="float32")[0] for p in paths]

    def test_matches_the_hf_feature_extractor(self):
        transformers = pytest.importorskip("transformers")
        from oasr.features.nemotron import batched_nemotron_logmel
        from oasr.models.nemotron.convert import HFNemotronConverter

        ckpt = assets.require("NEMOTRON_CKPT")
        audios = self._inputs()
        fe = transformers.AutoProcessor.from_pretrained(ckpt).feature_extractor
        ref = fe(audios, sampling_rate=16000, return_tensors="pt")

        cfg = HFNemotronConverter().build_feature_spec(ckpt).to_feature_config()
        width = max(len(a) for a in audios)
        wav = torch.zeros(len(audios), width)
        for i, a in enumerate(audios):
            wav[i, : len(a)] = torch.from_numpy(a)
        got, got_lengths = batched_nemotron_logmel(wav, torch.tensor([len(a) for a in audios]), cfg)

        assert got.shape == ref["input_features"].shape
        assert got_lengths.tolist() == ref["attention_mask"].sum(-1).tolist()
        # 2.3e-4 on values spanning [-16.6, 3.2] is the fp32 accumulation-order
        # difference between torchaudio's mel filterbank and librosa's — the whole
        # residual, verified below.  Two orders of magnitude below fp16 resolution
        # at this scale.
        torch.testing.assert_close(got, ref["input_features"], atol=5e-4, rtol=0)

    def test_only_the_mel_table_differs(self):
        """Substituting librosa's filterbank makes the pipeline **bit-exact**.

        Pinned because it says where the residual above comes from: if this ever
        stops being exact, the difference has moved into the STFT, the
        pre-emphasis or the log guard — a real divergence — rather than staying a
        table-precision artifact.
        """
        transformers = pytest.importorskip("transformers")
        librosa = pytest.importorskip("librosa")
        import oasr.features.nemotron as frontend
        from oasr.models.nemotron.convert import HFNemotronConverter

        ckpt = assets.require("NEMOTRON_CKPT")
        audio = self._inputs(1)[0]
        fe = transformers.AutoProcessor.from_pretrained(ckpt).feature_extractor
        ref = fe([audio], sampling_rate=16000, return_tensors="pt")["input_features"]

        table = torch.from_numpy(
            librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000, norm="slaney")
        ).float()
        original = frontend._mel_filters
        frontend._mel_filters = lambda *a, **k: table
        try:
            cfg = HFNemotronConverter().build_feature_spec(ckpt).to_feature_config()
            got, _ = frontend.batched_nemotron_logmel(
                torch.from_numpy(audio)[None, :], torch.tensor([len(audio)]), cfg
            )
        finally:
            frontend._mel_filters = original
        assert (got - ref).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Real-checkpoint parity
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_assets("NEMOTRON_CKPT", "WAV_DIR")
class TestRealCheckpointParity:
    """fp32 on CPU against ``transformers``, which is where an exact claim is
    meaningful; the fp16 GPU path is covered by the WER gate."""

    N_UTTERANCES = 4

    @pytest.fixture(scope="class")
    def hf(self):
        transformers = pytest.importorskip("transformers")
        ckpt = assets.require("NEMOTRON_CKPT")
        from transformers.models.nemotron3_5_asr import Nemotron3_5AsrForRNNT

        processor = transformers.AutoProcessor.from_pretrained(ckpt)
        model = Nemotron3_5AsrForRNNT.from_pretrained(ckpt, dtype=torch.float32).eval()
        return processor, model

    @pytest.fixture(scope="class")
    def ours(self):
        from oasr.models.loaders import load_pretrained

        loaded = load_pretrained(assets.require("NEMOTRON_CKPT"), device="cpu", dtype=torch.float32)
        # ``en-US`` — prompt slot 0 — so the comparison is against a pinned
        # language rather than the checkpoint's language-ID default.
        loaded.model.prompt_id = 0
        return loaded

    @pytest.fixture(scope="class")
    def strategy(self, ours):
        from types import SimpleNamespace

        from oasr.engine.decode.detokenize import Detokenizer
        from oasr.engine.decode.transducer import TransducerDecodeStrategy
        from oasr.tokenizers import build_tokenizer

        config = SimpleNamespace(
            decode_options={},
            transducer_max_sym_per_frame=int(ours.config.max_symbols_per_step),
            partial_decode_interval=1,
        )
        detok = Detokenizer(tokenizer=build_tokenizer(ours.tokenizer_spec))
        return TransducerDecodeStrategy(config, detok, ours.model)

    @staticmethod
    def _audios(n):
        import soundfile as sf

        return [sf.read(p, dtype="float32")[0] for p in assets.require_wavs(n)]

    def test_load_report_drops_nothing_unexpected(self, ours):
        report = ours.load_report
        assert report is not None
        assert report.dropped == []
        assert report.missing == []

    def test_encoder_and_prompt_fusion_match_hf(self, hf, ours):
        processor, reference = hf
        audio = self._audios(1)[0]
        inputs = processor(audio, sampling_rate=16000, language="en-US")
        lengths = torch.tensor([int(inputs["attention_mask"].sum())])

        with torch.no_grad():
            expected = reference.get_audio_features(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                prompt_ids=inputs["prompt_ids"],
                num_lookahead_tokens=inputs["num_lookahead_tokens"],
            )
            hidden, _ = ours.model.encoder(inputs["input_features"], lengths)
            fused, _ = ours.model.encode_offline(inputs["input_features"], lengths)
            projected = ours.model.joiner.encoder_proj(fused)

        # Raw encoder first, then the prompt projector + joint projection: when
        # this breaks, which of the two comparisons fails localises it.
        torch.testing.assert_close(hidden, expected.last_hidden_state, atol=1e-5, rtol=0)
        torch.testing.assert_close(projected, expected.pooler_output, atol=5e-5, rtol=0)

    def test_greedy_is_token_exact_vs_hf(self, hf, ours, strategy):
        processor, reference = hf
        blank = ours.model.blank_id
        for audio in self._audios(self.N_UTTERANCES):
            inputs = processor(audio, sampling_rate=16000, language="en-US")
            with torch.no_grad():
                generated = reference.generate(**inputs)
                lengths = torch.tensor([int(inputs["attention_mask"].sum())])
                hidden, out_lengths = ours.model.encode_offline(inputs["input_features"], lengths)
                got = strategy.decode_offline(hidden, out_lengths)[0]
            expected_ids = [t for t in generated.sequences[0].tolist() if t != blank]
            assert got.tokens[0] == expected_ids
            assert (
                got.text.strip()
                == processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
            )

    def test_batched_decode_matches_one_at_a_time(self, ours, strategy):
        """Mixed lengths in one batch, which is what exercises the key-padding
        mask and the fully-masked-row handling.

        Exact here, but exactness is empirical rather than structural: both this
        encoder and HuggingFace's are padding-invariant only to ~2e-5 (batched
        reductions run in a different order), and a genuine near-tie at an
        utterance boundary can flip a trailing token.  The WER gate is the
        backstop — it is bit-stable across batch 1/8/16/32.
        """
        from oasr.features.nemotron import batched_nemotron_logmel

        cfg = ours.feature_spec.to_feature_config()
        audios = self._audios(6)
        width = max(len(a) for a in audios)
        wav = torch.zeros(len(audios), width)
        for i, a in enumerate(audios):
            wav[i, : len(a)] = torch.from_numpy(a)
        lengths = torch.tensor([len(a) for a in audios])

        with torch.no_grad():
            feats, feat_lengths = batched_nemotron_logmel(wav, lengths, cfg)
            hidden, out_lengths = ours.model.encode_offline(feats, feat_lengths.long())
            batched = strategy.decode_offline(hidden, out_lengths)
            solo = []
            for a in audios:
                one, one_len = batched_nemotron_logmel(
                    torch.from_numpy(a)[None, :], torch.tensor([len(a)]), cfg
                )
                h, hl = ours.model.encode_offline(one, one_len.long())
                solo.append(strategy.decode_offline(h, hl)[0])
        assert [o.tokens[0] for o in batched] == [o.tokens[0] for o in solo]


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.requires_assets("NEMOTRON_CKPT", "WAV_DIR")
class TestEngineEndToEnd:
    @staticmethod
    def _audios(n):
        import soundfile as sf

        return [sf.read(p, dtype="float32")[0] for p in assets.require_wavs(n)]

    @pytest.fixture(scope="class")
    def engine(self, device):
        from oasr.engine import ASREngine, EngineConfig

        eng = ASREngine(
            EngineConfig(
                ckpt_dir=assets.require("NEMOTRON_CKPT"),
                service_mode="offline",
                dtype=torch.float16,
                max_batch_size=8,
            )
        )
        yield eng
        del eng
        torch.cuda.empty_cache()

    def test_transcribes_and_adopts_the_checkpoint_frontend(self, engine):
        """``audio_scale`` in particular: NeMo trains on the ``[-1, 1]`` waveform,
        and the engine's own default is WeNet's ``1 << 15``.  A wrong scale here
        offsets every mel bin by a constant, which costs the leading token and
        nothing else — invisible to every tensor comparison."""
        assert engine.sample_rate == 16000
        config = engine._config  # noqa: SLF001 — asserting the adopted spec
        assert config.audio_scale == 1.0
        assert config.feature_config.feature_type == "nemotron_logmel"
        assert config.feature_config.num_mel_bins == 128

        outputs = engine.transcribe_offline(self._audios(4))
        texts = [o if isinstance(o, str) else o.text for o in outputs]
        assert all(t.strip() for t in texts), texts

    def test_batch_size_does_not_change_the_transcript(self, engine):
        audios = self._audios(4)
        wide = [o if isinstance(o, str) else o.text for o in engine.transcribe_offline(audios)]
        narrow = [
            (lambda o: o if isinstance(o, str) else o.text)(engine.transcribe_offline([a])[0])
            for a in audios
        ]
        assert wide == narrow

    def test_streaming_request_is_refused_by_an_offline_engine(self, engine):
        with pytest.raises(ValueError, match="service_mode"):
            engine.add_streaming_request("nemotron-stream-1")


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.requires_assets("NEMOTRON_CKPT", "WAV_DIR")
class TestStreamingEngine:
    """The streaming service mode on the real checkpoint."""

    @staticmethod
    def _audios(n):
        import soundfile as sf

        return [sf.read(p, dtype="float32")[0] for p in assets.require_wavs(n)]

    @staticmethod
    def _texts(outputs):
        return [o if isinstance(o, str) else o.text for o in outputs]

    @pytest.fixture(scope="class")
    def streaming_engine(self, device):
        from oasr.engine import ASREngine, EngineConfig

        eng = ASREngine(
            EngineConfig(
                ckpt_dir=assets.require("NEMOTRON_CKPT"),
                service_mode="streaming",
                dtype=torch.float16,
                max_batch_size=4,
            )
        )
        yield eng
        del eng
        torch.cuda.empty_cache()

    def test_the_engine_derives_the_trained_window_not_the_config_default(self, streaming_engine):
        """``num_left_chunks`` is not a knob for this encoder — the mask is trained.

        The engine derives the retained cache from
        ``CacheSpec.fixed_attention_window`` and pre-fills it, which is what makes
        ``cache_seqlens`` uniform across the cohort and therefore the *shared*
        relative-position table correct.
        """
        backend = streaming_engine._model_runner.streaming_backend  # noqa: SLF001
        cache_config = backend._cache_config  # noqa: SLF001
        assert cache_config.prefill_kv_window is True
        # 56 trained left-context frames, retained in chunk-sized pages.
        window = streaming_engine._model.encoder.fixed_attention_window  # noqa: SLF001
        assert cache_config.prefilled_cache_frames >= window
        assert cache_config.block_size_frames == cache_config.chunk_size
        # Every declared state got a buffer: the conv left-context plus one tail
        # per subsampling stage.
        names = backend.state_cache.names
        assert names[0] == "conv"
        assert [n for n in names if n.startswith("subsample.")] == [
            "subsample.0",
            "subsample.1",
            "subsample.2",
        ]

    def test_window_equals_stride_for_a_cached_causal_frontend(self, streaming_engine):
        """No lookahead: a chunk consumes exactly ``chunk_size * 8`` input frames.

        The generic formula would ask for ``(chunk_size - 1) * 8 + right_context + 1``,
        which describes a *centred* subsampling front-end and is one stride grid
        away from this one.
        """
        cfg = streaming_engine._config  # noqa: SLF001
        assert cfg.decoding_window == cfg.stride == cfg.chunk_size * 8

    def test_streaming_transcript_matches_offline_word_for_word(self, streaming_engine):
        """The gate for the whole streaming path.

        Sentence-final punctuation is allowed to differ: offline uniquely produces
        the encoder frame that the utterance-end right pad creates, and a stream's
        equivalent frames see the finalize silence pad instead.  Words must not.
        """
        from oasr.engine import ASREngine, EngineConfig

        audios = self._audios(4)
        streamed = self._texts(streaming_engine.transcribe(audios))

        offline_engine = ASREngine(
            EngineConfig(
                ckpt_dir=assets.require("NEMOTRON_CKPT"),
                service_mode="offline",
                dtype=torch.float16,
                max_batch_size=4,
            )
        )
        try:
            offline = self._texts(offline_engine.transcribe_offline(audios))
        finally:
            del offline_engine
            torch.cuda.empty_cache()

        def words(text):
            return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).split()

        for a, b in zip(offline, streamed):
            assert words(a) == words(b), f"offline={a!r} streaming={b!r}"

    def test_batch_size_does_not_change_the_streaming_transcript(self, streaming_engine):
        audios = self._audios(4)
        wide = self._texts(streaming_engine.transcribe(audios))
        solo = [self._texts(streaming_engine.transcribe([a]))[0] for a in audios]
        assert wide == solo

    @pytest.mark.parametrize("chunk_size", [4, 8, 32])
    def test_a_wider_or_narrower_chunk_still_matches(self, chunk_size, device):
        """Any multiple of the trained attention chunk must decode the same words.

        A step wider than one trained chunk needs the ``chunked_limited`` structure
        *within* the step, which rides in the same additive bias as the rel-pos
        term; a narrower one exercises the case where the convolution cache is
        wider than a chunk's own output.
        """
        from oasr.engine import ASREngine, EngineConfig

        audios = self._audios(2)
        texts = {}
        for cs in (16, chunk_size):
            eng = ASREngine(
                EngineConfig(
                    ckpt_dir=assets.require("NEMOTRON_CKPT"),
                    service_mode="streaming",
                    dtype=torch.float16,
                    max_batch_size=2,
                    chunk_size=cs,
                )
            )
            try:
                texts[cs] = self._texts(eng.transcribe(audios))
            finally:
                del eng
                torch.cuda.empty_cache()
        assert texts[chunk_size] == texts[16]

    def test_a_misaligned_chunk_size_is_refused_at_construction(self):
        """``chunk_size`` must be a whole number of trained attention chunks.

        The ``chunked_limited`` mask groups **absolute** frame positions, so a
        partial trained chunk would need keys from the future — which fails
        silently, producing a plausible transcript from arithmetic that no longer
        matches the offline pass.  Hence a construction-time refusal naming the
        multiple.
        """
        from oasr.engine import ASREngine, EngineConfig

        with pytest.raises(ValueError, match="multiple of"):
            ASREngine(
                EngineConfig(
                    ckpt_dir=assets.require("NEMOTRON_CKPT"),
                    service_mode="streaming",
                    dtype=torch.float16,
                    max_batch_size=1,
                    chunk_size=6,  # not a multiple of num_lookahead_tokens + 1 == 4
                )
            )


@pytest.mark.slow
@pytest.mark.requires_assets("NEMOTRON_CKPT")
def test_native_round_trip_preserves_the_weights(tmp_path):
    """``oasr-convert`` → reload must be a no-op, which is not free here: the
    subsampling projection's columns are permuted and three convolution layouts
    are rewritten on load, so every one of those hooks has to be idempotent."""
    from oasr.checkpoints.convert import convert_to_native
    from oasr.models.loaders import load_pretrained

    source = assets.require("NEMOTRON_CKPT")
    destination = tmp_path / "native"
    convert_to_native(source, str(destination), checkpoint_name="model.safetensors")

    original = load_pretrained(source, device="cpu", dtype=torch.float32)
    restored = load_pretrained(str(destination), device="cpu", dtype=torch.float32)
    left = original.model.state_dict()
    right = restored.model.state_dict()
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], msg=f"{key} changed")


def test_registered_in_the_model_registry():
    from oasr.models.registry import get_model_entry, list_models

    assert "nemotron" in list_models()
    entry = get_model_entry("nemotron")
    assert entry.model_cls is NemotronModel
    assert entry.config_cls is NemotronModelConfig
    assert entry.converter.architecture == "nemotron"


@pytest.mark.requires_assets("NEMOTRON_CKPT")
def test_converter_detects_the_checkpoint_unambiguously():
    """Detection must resolve without an ``architecture=`` hint: the directory also
    holds a ``model.safetensors`` and a ``tokenizer.json``, which weaker
    filename-based detectors match."""
    from pathlib import Path

    from oasr.models.registry import resolve_architecture

    assert resolve_architecture(Path(assets.require("NEMOTRON_CKPT"))) == "nemotron"


def test_lookahead_is_a_documented_knob():
    """The trained set, and the fact that the encoder exposes it: right context
    trades accuracy against streaming latency and the checkpoint supports four
    values, so pinning it silently at the default would hide a real lever."""
    model = NemotronModel.from_config(_tiny_config())
    assert model.encoder.num_lookahead_tokens == 3
    left, right = model.encoder.attention_context()
    assert (left, right) == (model.encoder.config.sliding_window - 1, 3)
    model.encoder.num_lookahead_tokens = 13
    assert model.encoder.attention_context()[1] == 13
    assert 13 in NemotronEncoderConfig().supported_num_lookahead_tokens


def test_input_scale_is_off_on_the_release():
    """``scale_input`` multiplies the subsampling output by ``sqrt(hidden_size)``
    — a factor of 32 at 1024.  The upstream *config class* defaults it to True and
    the released checkpoint sets it False, so a default copied from the class
    would be catastrophically wrong rather than subtly so."""
    assert NemotronEncoderConfig().scale_input is False
    model = NemotronModel.from_config(_tiny_config())
    assert model.encoder.input_scale == 1.0
    scaled = NemotronModel.from_config(
        _tiny_config(encoder=NemotronEncoderConfig(hidden_size=32, scale_input=True))
    )
    assert scaled.encoder.input_scale == pytest.approx(math.sqrt(32))
