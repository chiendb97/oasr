# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron ASR (FastConformer + RNN-T) tests.

Three tiers, deliberately separated by what each one can actually prove:

**Structure, no checkpoint** — the pieces that are easy to get subtly wrong and
whose wrongness a parity test would hide behind a tolerance: the Transformer-XL
relative shift, the ``chunked_limited`` window, the LSTM predictor's protocol,
and the fact that the start-of-sequence state is *not* zeros.

**Frontend parity** — the ``nemotron_logmel`` recipe against HuggingFace's own
feature extractor.  Bit-exact once the mel filterbank is the same table, which is
worth pinning separately: a feature-convention bug cancels in every downstream
parity test (both sides get the same features) and only shows up as WER.  That is
how the ``audio_scale`` defect shipped.

**Real-checkpoint parity** — encoder tensors and greedy *token ids* against
``transformers``, then the engine end to end.  Token exactness is the strong
claim here; the tensor comparisons are what localise a failure when it breaks.

The end-to-end accuracy number lives in ``ci/wer-reference.json`` and is checked
by ``tests/test_accuracy.py``, not here.
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
        assert model.streaming_kind == "none"
        assert model.cache_spec is None
        assert model.head is None
        assert model.blank_id == 31

    def test_offline_only_model_advertises_no_streaming(self):
        """The checkpoint *is* a streaming model; OASR does not yet carry its
        subsampling conv cache, so the engine must refuse a streaming request at
        construction rather than serve one with a reset front-end."""
        from oasr.engine.streaming_backend import build_streaming_backend  # noqa: F401

        model = NemotronModel.from_config(_tiny_config())
        assert model.encoder.streaming_kind == "none"

    def test_transducer_capability_surface_is_satisfied(self):
        from oasr.models.interfaces import missing_members

        model = NemotronModel.from_config(_tiny_config())
        assert missing_members(model, "transducer") == []


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

    def test_registered_as_non_streaming(self):
        from oasr.features import FeatureConfig, build_extractor

        spec = build_extractor(FeatureConfig(feature_type="nemotron_logmel"))
        assert spec.supports_streaming is False
        assert spec.window_seconds_attr is None

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

    def test_streaming_mode_is_refused_at_construction(self):
        """An offline-only encoder must be refused when the engine is *built*, so
        the failure names the checkpoint rather than surfacing as a mysterious
        rejection on the first request.  The model's name says streaming; what is
        missing is OASR's side of its subsampling conv cache."""
        from oasr.engine import ASREngine, EngineConfig

        with pytest.raises(ValueError, match="offline-only"):
            ASREngine(
                EngineConfig(
                    ckpt_dir=assets.require("NEMOTRON_CKPT"),
                    service_mode="streaming",
                    dtype=torch.float16,
                    max_batch_size=2,
                )
            )

    def test_streaming_request_is_refused_by_an_offline_engine(self, engine):
        with pytest.raises(ValueError, match="service_mode"):
            engine.add_streaming_request("nemotron-stream-1")


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
