#!/usr/bin/env python3
"""Tests for the Whisper package: frontend, model, converter, aed strategy.

CPU tests use a tiny random Whisper (1 s window); the end-to-end engine tests
use the real ``openai/whisper-tiny`` snapshot at ``WHISPER_CKPT`` and skip
when it is absent.
"""

import glob
import os

import pytest
import torch

from oasr.features import FeatureConfig
from oasr.features.whisper import batched_whisper_logmel
from oasr.models.whisper import WhisperModel, WhisperModelConfig

WHISPER_CKPT = os.environ.get("WHISPER_CKPT", "/data01/kilm/users/chiendb/models/asr/whisper-tiny")
WAV_DIR = os.environ.get(
    "WAV_DIR", "/data01/kilm/users/chiendb/data/asr/ljspeech-sr16k-dataset/wavs"
)


def _tiny_config(**overrides):
    base = dict(
        vocab_size=64,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_mel_bins=80,
        max_source_positions=50,  # 1 s window → 100 frames → 50 positions
        max_target_positions=32,
        decoder_start_token_id=60,
        eos_token_id=61,
        forced_decoder_ids=[(1, 62)],
        suppress_tokens=[0, 1],
        begin_suppress_tokens=[2],
    )
    base.update(overrides)
    return WhisperModelConfig(**base)


# ---------------------------------------------------------------------------
# Log-mel frontend
# ---------------------------------------------------------------------------


class TestWhisperLogmel:
    CFG = FeatureConfig(feature_type="whisper_logmel", whisper_chunk_seconds=1.0)

    def test_shapes_and_uniform_lengths(self):
        wav = torch.randn(3, 12000) * 0.1
        feats, lens = batched_whisper_logmel(wav, torch.tensor([12000, 8000, 4000]), self.CFG)
        assert feats.shape == (3, 100, 80)
        assert lens.tolist() == [100, 100, 100]  # padded window is real input

    def test_normalization_range(self):
        wav = torch.randn(1, 16000) * 0.5
        feats, _ = batched_whisper_logmel(wav, torch.tensor([16000]), self.CFG)
        # (log10 clamped to [max-8, max] + 4) / 4 → span <= 2
        assert feats.max() - feats.min() <= 2.0 + 1e-5

    def test_padding_invariance(self):
        """Extra zero padding past the valid length must not change features."""
        torch.manual_seed(0)
        wav = torch.randn(4000) * 0.3
        a, _ = batched_whisper_logmel(wav.unsqueeze(0), torch.tensor([4000]), self.CFG)
        padded = torch.cat([wav, torch.randn(8000)]).unsqueeze(0)  # garbage tail
        b, _ = batched_whisper_logmel(padded, torch.tensor([4000]), self.CFG)
        assert torch.allclose(a, b, atol=1e-6)

    def test_batch_rows_independent(self):
        torch.manual_seed(1)
        w1, w2 = torch.randn(5000) * 0.2, torch.randn(7000) * 0.9
        both, _ = batched_whisper_logmel(
            torch.stack([torch.nn.functional.pad(w1, (0, 2000)), w2]),
            torch.tensor([5000, 7000]),
            self.CFG,
        )
        solo, _ = batched_whisper_logmel(w1.unsqueeze(0), torch.tensor([5000]), self.CFG)
        assert torch.allclose(both[0], solo[0], atol=1e-5)

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(WHISPER_CKPT, "config.json")),
        reason="whisper snapshot absent",
    )
    def test_matches_transformers_feature_extractor(self):
        transformers = pytest.importorskip("transformers")
        fe = transformers.WhisperFeatureExtractor()
        torch.manual_seed(2)
        wav = (torch.randn(24000) * 0.1).numpy()
        ref = torch.tensor(fe(wav, sampling_rate=16000, return_tensors="np").input_features[0])
        ours, _ = batched_whisper_logmel(
            torch.tensor(wav).unsqueeze(0),
            torch.tensor([len(wav)]),
            FeatureConfig(feature_type="whisper_logmel"),
        )
        assert torch.allclose(ours[0].t(), ref, atol=1e-4)


# ---------------------------------------------------------------------------
# Decoder incremental surface
# ---------------------------------------------------------------------------


class TestDecoderIncremental:
    def test_step_matches_prefill(self):
        """KV-cached step-by-step logits == one big teacher-forced prefill."""
        cfg = _tiny_config()
        torch.manual_seed(3)
        model = WhisperModel(cfg).eval()
        B, T_enc = 2, cfg.max_source_positions
        enc = torch.randn(B, T_enc, cfg.d_model)
        seq = torch.randint(3, 59, (B, 8))

        with torch.no_grad():
            # Reference: prefill over the whole sequence at once.
            ref_logits, _ = model.decoder.prefill(enc, seq)
            # Incremental: prefill the first 4, then step the remaining 4.
            logits, state = model.decoder.prefill(enc, seq[:, :4])
            for t in range(4, 8):
                logits, state = model.decoder.step(seq[:, t], state)
        assert torch.allclose(logits, ref_logits, atol=1e-4)

    def test_select_drops_rows(self):
        cfg = _tiny_config()
        torch.manual_seed(4)
        model = WhisperModel(cfg).eval()
        enc = torch.randn(3, cfg.max_source_positions, cfg.d_model)
        seq = torch.randint(3, 59, (3, 5))
        with torch.no_grad():
            _, state = model.decoder.prefill(enc, seq)
            state1 = model.decoder.select(state, torch.tensor([2]))
            logits1, _ = model.decoder.step(seq[2, -1:].expand(1), state1)
            _, solo = model.decoder.prefill(enc[2:3], seq[2:3])
            logits_solo, _ = model.decoder.step(seq[2, -1:], solo)
        assert torch.allclose(logits1, logits_solo, atol=1e-4)


# ---------------------------------------------------------------------------
# Converter + registry + native round trip (real snapshot)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(os.path.join(WHISPER_CKPT, "model.safetensors")),
    reason="whisper snapshot absent",
)
class TestConverter:
    def test_detect_and_bundle(self):
        from oasr.models.registry import load_checkpoint_bundle

        arch, bundle = load_checkpoint_bundle(WHISPER_CKPT)
        assert arch == "whisper"
        assert bundle.source_format == "huggingface"
        assert bundle.tokenizer.kind == "whisper"
        f = bundle.features
        assert (f.kind, f.feature_dim, f.audio_scale) == ("whisper_logmel", 80, 1.0)
        assert bundle.decoding.default_decode_type == "aed"
        assert bundle.model_config.sot_sequence()[0] == 50258

    def test_load_report_clean(self):
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        arch, bundle = load_checkpoint_bundle(WHISPER_CKPT)
        model, cfg, report = instantiate_from_bundle(arch, bundle)
        assert not report.missing
        assert not [k for k in report.dropped if not k.startswith("proj_out.")]
        assert sorted(model.capabilities) == ["aed"]

    def test_native_round_trip(self, tmp_path):
        pytest.importorskip("safetensors")
        from oasr.checkpoints.convert import convert_to_native
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        out = tmp_path / "native"
        convert_to_native(WHISPER_CKPT, str(out))
        arch, bundle = load_checkpoint_bundle(out)
        assert (arch, bundle.source_format) == ("whisper", "native")
        assert bundle.tokenizer.kind == "whisper"
        m2, cfg2, _ = instantiate_from_bundle(arch, bundle)
        assert cfg2.sot_sequence() == [50258, 50259, 50359, 50363]

        arch1, b1 = load_checkpoint_bundle(WHISPER_CKPT)
        m1, _, _ = instantiate_from_bundle(arch1, b1)
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        assert set(sd1) == set(sd2)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), k


# ---------------------------------------------------------------------------
# Engine end-to-end (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
@pytest.mark.skipif(
    not os.path.exists(os.path.join(WHISPER_CKPT, "model.safetensors")),
    reason="whisper snapshot absent",
)
class TestEngineWhisperE2E:
    @pytest.fixture(scope="class")
    def engine(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=WHISPER_CKPT,
            service_mode="offline",
            max_batch_size=4,
            decode_steps_per_tick=64,
        )
        eng = ASREngine(cfg)
        yield eng
        del eng
        torch.cuda.empty_cache()

    def test_transcribe_offline(self, engine):
        wavs = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))[:2]
        if not wavs:
            pytest.skip(f"no wavs under {WAV_DIR}")
        import torchaudio

        audios = [torchaudio.load(w)[0].squeeze(0) for w in wavs]
        texts = engine.transcribe_offline(audios)
        texts = [t.text if hasattr(t, "text") else t for t in texts]
        assert len(texts) == 2
        # LJ001-0001/0002 ground truth openers.
        assert "printing" in texts[0].lower()
        assert "modern" in texts[1].lower()

    def test_streaming_mode_rejected(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=WHISPER_CKPT, service_mode="streaming")
        with pytest.raises(ValueError, match="offline-only"):
            ASREngine(cfg)

    def test_wrong_decode_method_rejected(self):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=WHISPER_CKPT, service_mode="offline", decode_method="ctc")
        with pytest.raises(ValueError, match="not a capability"):
            ASREngine(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
