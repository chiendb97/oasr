# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for batched audio feature extraction (``oasr.features``).

Covers :class:`FeatureConfig`, offline batch APIs (``fbank_batch``, ``mfcc_batch``,
``extract_features_batch``), and :class:`BatchedStreamingFeatureExtractor`.
"""

from __future__ import annotations

from typing import List

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _have_torchaudio() -> bool:
    try:
        import torchaudio  # noqa: F401

        return True
    except ImportError:
        return False


requires_torchaudio = pytest.mark.skipif(
    not _have_torchaudio(), reason="torchaudio not installed"
)


def _generate_waveform(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> torch.Tensor:
    """Deterministic pseudo-random waveform, shape ``(samples,)``."""
    gen = torch.Generator().manual_seed(seed)
    n_samples = int(duration_s * sample_rate)
    return torch.randn(n_samples, generator=gen)


def _ref_fbank_1d(
    wav: torch.Tensor,
    *,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    dither: float = 0.0,
) -> torch.Tensor:
    """Reference FBANK for a 1-D waveform (torchaudio Kaldi)."""
    import torchaudio

    x = wav.unsqueeze(0) if wav.dim() == 1 else wav
    return torchaudio.compliance.kaldi.fbank(
        x,
        sample_frequency=float(sample_rate),
        num_mel_bins=num_mel_bins,
        frame_length=25.0,
        frame_shift=10.0,
        dither=dither,
        energy_floor=0.0,
        preemphasis_coefficient=0.97,
        window_type="povey",
        low_freq=20.0,
        high_freq=0.0,
        snip_edges=True,
    )


def _ref_mfcc_1d(
    wav: torch.Tensor,
    *,
    sample_rate: int = 16000,
    num_ceps: int = 13,
    num_mel_bins: int = 23,
    dither: float = 0.0,
) -> torch.Tensor:
    """Reference MFCC for a 1-D waveform (torchaudio Kaldi)."""
    import torchaudio

    x = wav.unsqueeze(0) if wav.dim() == 1 else wav
    return torchaudio.compliance.kaldi.mfcc(
        x,
        sample_frequency=float(sample_rate),
        num_mel_bins=num_mel_bins,
        num_ceps=num_ceps,
        frame_length=25.0,
        frame_shift=10.0,
        dither=dither,
        energy_floor=0.0,
        preemphasis_coefficient=0.97,
        window_type="povey",
        low_freq=20.0,
        high_freq=0.0,
        snip_edges=True,
    )


# torchaudio introduces small floating-point differences between full-buffer and
# chunked extraction; keep tolerances for batched-streaming vs offline batch.
_STREAMING_ATOL = 5e-5
_STREAMING_RTOL = 5e-4


# ===========================================================================
# FeatureConfig validation
# ===========================================================================


class TestFeatureConfig:
    """Unit tests for :class:`FeatureConfig` dataclass."""

    def test_defaults(self):
        from oasr.features import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.feature_type == "fbank"
        assert cfg.sample_rate == 16000
        assert cfg.num_mel_bins == 80
        assert cfg.frame_length_ms == 25.0
        assert cfg.frame_shift_ms == 10.0
        assert cfg.backend == "torchaudio"
        assert cfg.snip_edges is True

    def test_output_dim_fbank(self):
        from oasr.features import FeatureConfig

        cfg = FeatureConfig(feature_type="fbank", num_mel_bins=40)
        assert cfg.output_dim == 40

    def test_output_dim_mfcc(self):
        from oasr.features import FeatureConfig

        cfg = FeatureConfig(feature_type="mfcc", num_ceps=20)
        assert cfg.output_dim == 20

    def test_frame_samples(self):
        from oasr.features import FeatureConfig

        cfg = FeatureConfig(sample_rate=16000, frame_length_ms=25.0, frame_shift_ms=10.0)
        assert cfg.frame_length_samples == 400
        assert cfg.frame_shift_samples == 160

    def test_invalid_feature_type(self):
        from oasr.features import FeatureConfig

        with pytest.raises(ValueError, match="feature_type"):
            FeatureConfig(feature_type="spectrogram")

    def test_invalid_backend(self):
        from oasr.features import FeatureConfig

        with pytest.raises(ValueError, match="backend"):
            FeatureConfig(backend="librosa")

    def test_invalid_sample_rate(self):
        from oasr.features import FeatureConfig

        with pytest.raises(ValueError, match="sample_rate"):
            FeatureConfig(sample_rate=0)

    def test_shift_exceeds_length(self):
        from oasr.features import FeatureConfig

        with pytest.raises(ValueError, match="frame_shift_ms"):
            FeatureConfig(frame_length_ms=10.0, frame_shift_ms=25.0)


# ===========================================================================
# Batched extraction
# ===========================================================================


@requires_torchaudio
class TestBatchedFbank:
    """Batched FBANK extraction via :func:`fbank_batch`."""

    def test_uniform_batch_tensor(self):
        from oasr.features import fbank_batch

        B, T = 4, 16000
        torch.manual_seed(0)
        wavs = torch.randn(B, T)
        feats, feat_lens = fbank_batch(wavs, dither=0.0)

        assert feats.dim() == 3
        assert feats.size(0) == B
        assert feats.size(2) == 80
        assert feat_lens.shape == (B,)
        assert (feat_lens == feat_lens[0]).all(), "Uniform-length batch should have equal frame counts"

        single = _ref_fbank_1d(wavs[0], dither=0.0)
        torch.testing.assert_close(feats[0, : feat_lens[0]], single, rtol=0.0, atol=0.0)

    def test_variable_length_list(self):
        from oasr.features import fbank_batch

        torch.manual_seed(1)
        wavs = [torch.randn(16000), torch.randn(8000), torch.randn(24000)]
        feats, feat_lens = fbank_batch(wavs, dither=0.0)

        assert feats.dim() == 3
        assert feats.size(0) == 3
        assert feats.size(2) == 80
        assert feat_lens[0] != feat_lens[2], "Different lengths should yield different frame counts"

        for i, w in enumerate(wavs):
            single = _ref_fbank_1d(w, dither=0.0)
            n = int(feat_lens[i].item())
            torch.testing.assert_close(feats[i, :n], single, rtol=0.0, atol=0.0)

    def test_padded_with_lengths(self):
        from oasr.features import fbank_batch

        torch.manual_seed(2)
        actual_lens = [16000, 8000, 12000]
        max_len = max(actual_lens)
        B = len(actual_lens)
        wavs = torch.zeros(B, max_len)
        for i, L in enumerate(actual_lens):
            wavs[i, :L] = torch.randn(L)
        lengths = torch.tensor(actual_lens, dtype=torch.long)

        feats, feat_lens = fbank_batch(wavs, lengths=lengths, dither=0.0)

        assert feats.dim() == 3
        assert feats.size(0) == B
        for i in range(B):
            single = _ref_fbank_1d(wavs[i, : actual_lens[i]], dither=0.0)
            n = int(feat_lens[i].item())
            assert n == single.size(0)
            torch.testing.assert_close(feats[i, :n], single, rtol=0.0, atol=0.0)

    def test_single_item_batch(self):
        from oasr.features import fbank_batch

        wav = _generate_waveform(duration_s=1.0)
        feats, feat_lens = fbank_batch([wav], dither=0.0)

        assert feats.size(0) == 1
        single = _ref_fbank_1d(wav, dither=0.0)
        torch.testing.assert_close(feats[0, : feat_lens[0]], single, rtol=0.0, atol=0.0)

    def test_custom_mel_bins(self):
        from oasr.features import fbank_batch

        wavs = torch.randn(2, 8000)
        feats, _ = fbank_batch(wavs, num_mel_bins=40, dither=0.0)
        assert feats.size(2) == 40


@requires_torchaudio
class TestBatchedMfcc:
    """Batched MFCC extraction via :func:`mfcc_batch`."""

    def test_uniform_batch(self):
        from oasr.features import mfcc_batch

        B, T = 3, 16000
        torch.manual_seed(10)
        wavs = torch.randn(B, T)
        feats, feat_lens = mfcc_batch(wavs, dither=0.0, num_ceps=13)

        assert feats.dim() == 3
        assert feats.size(0) == B
        assert feats.size(2) == 13

        single = _ref_mfcc_1d(wavs[0], dither=0.0, num_ceps=13)
        torch.testing.assert_close(feats[0, : feat_lens[0]], single, rtol=0.0, atol=0.0)

    def test_variable_length_list(self):
        from oasr.features import mfcc_batch

        torch.manual_seed(11)
        wavs = [torch.randn(8000), torch.randn(16000)]
        feats, feat_lens = mfcc_batch(wavs, dither=0.0)

        assert feats.size(0) == 2
        assert feat_lens[0] < feat_lens[1]


@requires_torchaudio
class TestBatchedExtractFeatures:
    """Batched extraction via :func:`extract_features_batch`."""

    def test_fbank_config(self):
        from oasr.features import FeatureConfig, extract_features_batch

        cfg = FeatureConfig(feature_type="fbank", num_mel_bins=40, dither=0.0)
        wavs = [torch.randn(16000), torch.randn(8000)]
        feats, feat_lens = extract_features_batch(wavs, cfg)

        assert feats.size(2) == 40
        assert feats.size(0) == 2

    def test_mfcc_config(self):
        from oasr.features import FeatureConfig, extract_features_batch

        cfg = FeatureConfig(feature_type="mfcc", num_ceps=20, dither=0.0)
        wavs = torch.randn(2, 16000)
        feats, feat_lens = extract_features_batch(wavs, cfg)

        assert feats.size(2) == 20
        assert (feat_lens == feat_lens[0]).all()


# ===========================================================================
# Batch edge cases
# ===========================================================================


@requires_torchaudio
class TestBatchEdgeCases:
    def test_very_short_audio_batch(self):
        """Exactly one frame from minimal-length waveform in a batch."""
        from oasr.features import fbank_batch

        wav = torch.randn(400)
        feats, feat_lens = fbank_batch([wav], dither=0.0)
        assert feat_lens[0] == 1
        assert feats.size(2) == 80
        ref = _ref_fbank_1d(wav, dither=0.0)
        torch.testing.assert_close(feats[0, : feat_lens[0]], ref, rtol=0.0, atol=0.0)

    def test_8khz_sample_rate_batch(self):
        from oasr.features import fbank_batch

        wav = torch.randn(8000)
        feats, feat_lens = fbank_batch([wav], sample_rate=8000, dither=0.0)
        ref = _ref_fbank_1d(wav, sample_rate=8000, dither=0.0)
        torch.testing.assert_close(feats[0, : feat_lens[0]], ref, rtol=0.0, atol=0.0)


# ===========================================================================
# Batched streaming extraction
# ===========================================================================


@requires_torchaudio
class TestBatchedStreaming:
    """Tests for :class:`BatchedStreamingFeatureExtractor`."""

    @pytest.fixture()
    def fbank_config(self):
        from oasr.features import FeatureConfig

        return FeatureConfig(
            feature_type="fbank",
            num_mel_bins=80,
            dither=0.0,
        )

    def test_output_shape(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        B = 3
        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)
        wavs = torch.randn(B, 4000)
        feats, feat_lens = ext.process_chunk(wavs)

        assert feats.dim() == 3
        assert feats.size(0) == B
        assert feats.size(2) == 80
        assert feat_lens.shape == (B,)

    def test_matches_single_stream(self, fbank_config):
        """Each stream matches a private per-stream chunk extractor."""
        from oasr.features import BatchedStreamingFeatureExtractor
        from oasr.features.streaming import _StreamingFeatureExtractor

        B = 3
        chunk_size = 1600
        torch.manual_seed(42)
        waveforms = [torch.randn(32000) for _ in range(B)]

        batched_ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)
        batched_parts: List[List[torch.Tensor]] = [[] for _ in range(B)]
        n_chunks = 32000 // chunk_size
        for c in range(n_chunks):
            chunk_batch = [w[c * chunk_size : (c + 1) * chunk_size] for w in waveforms]
            feats, feat_lens = batched_ext.process_chunk(chunk_batch)
            for i in range(B):
                n = int(feat_lens[i].item())
                if n > 0:
                    batched_parts[i].append(feats[i, :n])

        flush_feats, flush_lens = batched_ext.flush()
        for i in range(B):
            n = int(flush_lens[i].item())
            if n > 0:
                batched_parts[i].append(flush_feats[i, :n])

        for i in range(B):
            ref_ext = _StreamingFeatureExtractor(fbank_config)
            ref_parts: List[torch.Tensor] = []
            for c in range(n_chunks):
                chunk = waveforms[i][c * chunk_size : (c + 1) * chunk_size]
                f = ref_ext.process_chunk(chunk)
                if f is not None:
                    ref_parts.append(f)
            fl = ref_ext.flush()
            if fl is not None:
                ref_parts.append(fl)

            batched_cat = torch.cat(batched_parts[i], dim=0) if batched_parts[i] else torch.empty(0, 80)
            ref_cat = torch.cat(ref_parts, dim=0) if ref_parts else torch.empty(0, 80)

            assert batched_cat.shape == ref_cat.shape, (
                f"Stream {i}: shape mismatch {batched_cat.shape} vs {ref_cat.shape}"
            )
            torch.testing.assert_close(
                batched_cat,
                ref_cat,
                rtol=0.0,
                atol=0.0,
                msg=f"Stream {i}: batched != single-stream",
            )

    def test_variable_chunk_sizes(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        B = 2
        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)

        torch.manual_seed(7)
        wavs = [torch.randn(3200), torch.randn(4800)]
        feats, feat_lens = ext.process_chunk(wavs)

        assert feats.size(0) == B
        assert feat_lens[0] < feat_lens[1], (
            "Longer chunk should produce more frames"
        )

    def test_padded_tensor_with_lengths(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        B = 3
        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)

        actual = [3200, 1600, 4800]
        max_len = max(actual)
        torch.manual_seed(8)
        padded = torch.zeros(B, max_len)
        for i, L in enumerate(actual):
            padded[i, :L] = torch.randn(L)
        lengths = torch.tensor(actual, dtype=torch.long)

        feats, feat_lens = ext.process_chunk(padded, lengths=lengths)
        assert feats.size(0) == B
        assert int(feat_lens[0].item()) < int(feat_lens[2].item())

    def test_flush_returns_correct_shape(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        B = 2
        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)

        wavs = [torch.randn(1600), torch.randn(100)]
        ext.process_chunk(wavs)

        flush_feats, flush_lens = ext.flush()
        assert flush_feats.size(0) == B
        assert flush_feats.size(2) == 80

    def test_reset_all(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        B = 2
        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)
        ext.process_chunk(torch.randn(B, 4000))
        ext.reset()

        counts = ext.num_frames_extracted
        assert (counts == 0).all()

    def test_reset_selective(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        B = 2
        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)
        ext.process_chunk(torch.randn(B, 4000))

        counts_before = ext.num_frames_extracted.clone()
        ext.reset(stream_indices=[0])

        counts_after = ext.num_frames_extracted
        assert counts_after[0] == 0
        assert counts_after[1] == counts_before[1]

    def test_batch_size_mismatch_raises(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=3)
        with pytest.raises(ValueError, match="Expected 3"):
            ext.process_chunk([torch.randn(1600), torch.randn(1600)])

    def test_num_frames_extracted(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        B = 2
        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)
        total = torch.zeros(B, dtype=torch.long)

        for _ in range(5):
            feats, feat_lens = ext.process_chunk(torch.randn(B, 1600))
            total += feat_lens

        flush_feats, flush_lens = ext.flush()
        total += flush_lens

        counts = ext.num_frames_extracted
        torch.testing.assert_close(counts, total)

    def test_matches_offline_batch(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor, fbank_batch

        B, total_samples = 3, 16000
        chunk_size = 2000
        torch.manual_seed(99)
        waveforms = torch.randn(B, total_samples)

        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)
        stream_parts: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            feats, feat_lens = ext.process_chunk(waveforms[:, start:end])
            for i in range(B):
                n = int(feat_lens[i].item())
                if n > 0:
                    stream_parts[i].append(feats[i, :n])

        flush_feats, flush_lens = ext.flush()
        for i in range(B):
            n = int(flush_lens[i].item())
            if n > 0:
                stream_parts[i].append(flush_feats[i, :n])

        offline_feats, offline_lens = fbank_batch(waveforms, dither=0.0)

        for i in range(B):
            stream_cat = torch.cat(stream_parts[i], dim=0)
            n_off = int(offline_lens[i].item())
            assert stream_cat.size(0) >= n_off, (
                f"Stream {i}: streaming {stream_cat.size(0)} < offline {n_off}"
            )
            torch.testing.assert_close(
                stream_cat[:n_off],
                offline_feats[i, :n_off],
                rtol=_STREAMING_RTOL,
                atol=_STREAMING_ATOL,
                msg=f"Stream {i}: batched streaming != offline",
            )

    def test_zero_length_chunk(self, fbank_config):
        from oasr.features import BatchedStreamingFeatureExtractor

        B = 2
        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=B)

        wavs = [torch.randn(1600), torch.empty(0)]
        feats, feat_lens = ext.process_chunk(wavs)

        assert feats.size(0) == B
        assert feat_lens[1] == 0

    def test_snip_edges_false_rejected(self):
        from oasr.features import BatchedStreamingFeatureExtractor, FeatureConfig

        with pytest.raises(ValueError, match="snip_edges"):
            BatchedStreamingFeatureExtractor(FeatureConfig(snip_edges=False), batch_size=1)

    def test_streaming_multiple_flush_calls(self, fbank_config):
        """Second flush after the first should return zero-length rows."""
        from oasr.features import BatchedStreamingFeatureExtractor

        ext = BatchedStreamingFeatureExtractor(fbank_config, batch_size=1)
        ext.process_chunk(torch.randn(1, 1600))
        f1, l1 = ext.flush()
        assert l1[0] >= 1
        f2, l2 = ext.flush()
        assert l2[0] == 0
