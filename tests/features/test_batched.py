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


requires_torchaudio = pytest.mark.skipif(not _have_torchaudio(), reason="torchaudio not installed")


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
        assert (
            feat_lens == feat_lens[0]
        ).all(), "Uniform-length batch should have equal frame counts"

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

            batched_cat = (
                torch.cat(batched_parts[i], dim=0) if batched_parts[i] else torch.empty(0, 80)
            )
            ref_cat = torch.cat(ref_parts, dim=0) if ref_parts else torch.empty(0, 80)

            assert (
                batched_cat.shape == ref_cat.shape
            ), f"Stream {i}: shape mismatch {batched_cat.shape} vs {ref_cat.shape}"
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
        assert feat_lens[0] < feat_lens[1], "Longer chunk should produce more frames"

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
            assert (
                stream_cat.size(0) >= n_off
            ), f"Stream {i}: streaming {stream_cat.size(0)} < offline {n_off}"
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


# ===========================================================================
# GraphedFeatureExtraction — CUDA Graph capture of batched fbank/mfcc
# ===========================================================================


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture"
)


@requires_cuda
def _torch_stft_frame(
    wav: torch.Tensor,
    lengths: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop: int,
    num_frames: int,
    *,
    center_offset: int,
    win_offset: int,
    preemph_coef: float,
    preemph_replicate: bool,
) -> torch.Tensor:
    """Reference for :func:`oasr.stft_frame`, written straight off its contract."""
    B, T = wav.shape
    out = wav.new_zeros(B, num_frames, n_fft)
    win_length = window.numel()
    for b in range(B):
        n = int(lengths[b])
        for f in range(num_frames):
            for i in range(win_offset, win_offset + win_length):
                t = f * hop - center_offset + i
                if t < 0 or t >= n:
                    continue
                y = float(wav[b, t])
                if preemph_coef:
                    if t == 0:
                        prev = y if preemph_replicate else 0.0
                    else:
                        prev = float(wav[b, t - 1])
                    y -= preemph_coef * prev
                out[b, f, i] = y * window[i - win_offset]
    return out


@requires_cuda
class TestStftFrameKernel:
    """The framing primitive KG16 named as its missing piece, tested directly."""

    @pytest.mark.parametrize("center_offset", [0, 8, -1])
    @pytest.mark.parametrize("preemph", [0.0, 0.97])
    @pytest.mark.parametrize("replicate", [False, True])
    def test_matches_the_reference_contract(self, center_offset, preemph, replicate):
        import oasr

        torch.manual_seed(0)
        n_fft, hop, win_length = 32, 8, 20
        B, T = 3, 200
        wav = torch.randn(B, T)
        lengths = torch.tensor([T, T - 17, 40], dtype=torch.int64)
        window = torch.hann_window(win_length, periodic=False)
        win_offset = (n_fft - win_length) // 2
        num_frames = 12

        ref = _torch_stft_frame(
            wav,
            lengths,
            window,
            n_fft,
            hop,
            num_frames,
            center_offset=center_offset,
            win_offset=win_offset,
            preemph_coef=preemph,
            preemph_replicate=replicate,
        )
        got = oasr.stft_frame(
            wav.cuda(),
            lengths.cuda(),
            window.cuda(),
            n_fft,
            hop,
            num_frames,
            center_offset=center_offset,
            preemph_coef=preemph,
            preemph_replicate=replicate,
        )
        torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=1e-6)

    def test_centered_framing_reproduces_torch_stft(self):
        """``center_offset = n_fft // 2`` == ``torch.stft(center=True, constant)``."""
        import oasr

        torch.manual_seed(1)
        n_fft, hop, win_length = 512, 160, 400
        T = 4000
        wav = torch.randn(1, T) * 0.1
        window = torch.hann_window(win_length, periodic=False)
        num_frames = T // hop + 1

        frames = oasr.stft_frame(
            wav.cuda(),
            torch.tensor([T]).cuda(),
            window.cuda(),
            n_fft,
            hop,
            num_frames,
            center_offset=n_fft // 2,
        )
        got = torch.fft.rfft(frames.cpu(), n=n_fft)
        ref = torch.stft(
            wav,
            n_fft,
            hop_length=hop,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).transpose(1, 2)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=2e-4)

    def test_reflect_padding_reproduces_a_logical_padded_window(self):
        """Whisper reflects the fixed window, not each row's valid prefix."""
        import oasr

        torch.manual_seed(4)
        B, T, signal_length = 2, 41, 64
        n_fft, hop = 32, 8
        center = n_fft // 2
        num_frames = signal_length // hop + 1
        wav = torch.randn(B, T)
        lengths = torch.tensor([T, 23])
        window = torch.hann_window(n_fft)

        logical = torch.zeros(B, signal_length)
        for b in range(B):
            logical[b, : lengths[b]] = wav[b, : lengths[b]]
        padded = torch.nn.functional.pad(logical, (center, center), mode="reflect")
        ref = padded.unfold(1, n_fft, hop)[:, :num_frames] * window
        got = oasr.stft_frame(
            wav.cuda(),
            lengths.cuda(),
            window.cuda(),
            n_fft,
            hop,
            num_frames,
            center_offset=center,
            pad_mode="reflect",
            signal_length=signal_length,
        )
        torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=1e-6)

    def test_dc_removal_is_per_frame_with_a_replicate_boundary(self):
        """The KG16 path frames and performs Kaldi preprocessing in one launch."""
        import oasr

        torch.manual_seed(5)
        B, T, frame_length, hop, n_fft = 3, 160, 40, 16, 64
        wav = torch.randn(B, T)
        lengths = torch.tensor([T, 101, 40])
        num_frames = (T - frame_length) // hop + 1
        window = torch.hamming_window(frame_length, periodic=False)
        frames = wav.unfold(1, frame_length, hop)
        centered = frames - frames.mean(-1, keepdim=True)
        preem = torch.empty_like(centered)
        preem[..., 0] = 0.03 * centered[..., 0]
        preem[..., 1:] = centered[..., 1:] - 0.97 * centered[..., :-1]
        ref = torch.nn.functional.pad(preem * window, (0, n_fft - frame_length))

        got = oasr.stft_frame(
            wav.cuda(),
            lengths.cuda(),
            window.cuda(),
            n_fft,
            hop,
            num_frames,
            win_offset=0,
            preemph_coef=0.97,
            preemph_replicate=True,
            remove_dc_offset=True,
        ).cpu()
        valid = torch.clamp((lengths - frame_length) // hop + 1, min=0)
        for b, count in enumerate(valid.tolist()):
            torch.testing.assert_close(got[b, :count], ref[b, :count], rtol=0, atol=2e-5)
            # The general framing primitive preserves the valid prefix of a
            # partial trailing window (centered STFT recipes need that). Kaldi
            # rejects that whole frame via the frame lengths passed to mel_log;
            # only windows whose *start* is beyond the row must be empty here.
            first_empty = (int(lengths[b]) + hop - 1) // hop
            if first_empty < num_frames:
                assert got[b, first_empty:].abs().max() == 0

    def test_window_is_zero_outside_its_offset(self):
        import oasr

        n_fft, win_length = 64, 20
        wav = torch.ones(1, 128)
        window = torch.ones(win_length)
        out = oasr.stft_frame(
            wav.cuda(), torch.tensor([128]).cuda(), window.cuda(), n_fft, 8, 4, center_offset=0
        )
        off = (n_fft - win_length) // 2
        assert out[:, :, :off].abs().max() == 0
        assert out[:, :, off + win_length :].abs().max() == 0
        assert out[:, :, off : off + win_length].abs().min() > 0

    def test_zero_frames_returns_an_empty_tensor(self):
        import oasr

        out = oasr.stft_frame(
            torch.zeros(2, 10).cuda(),
            torch.tensor([10, 10]).cuda(),
            torch.ones(8).cuda(),
            8,
            4,
            0,
        )
        assert out.shape == (2, 0, 8)

    def test_rejects_a_window_wider_than_the_transform(self):
        import oasr

        with pytest.raises(ValueError, match="win_length"):
            oasr.stft_frame(
                torch.zeros(1, 64).cuda(),
                torch.tensor([64]).cuda(),
                torch.ones(16).cuda(),
                8,
                4,
                2,
            )


@requires_cuda
class TestMelLogGuards:
    """The floor and the additive guard set a silent bin differently."""

    def _power(self):
        torch.manual_seed(2)
        return torch.rand(2, 5, 33, device="cuda") * 1e-3

    def _filters(self):
        torch.manual_seed(3)
        return torch.rand(8, 33, device="cuda")

    def test_additive_guard_matches_log_of_the_sum(self):
        import oasr

        power, filters = self._power(), self._filters()
        got = oasr.mel_log(power, filters, log_floor=0.0, log_offset=2.0**-24)
        ref = torch.log(power @ filters.t() + 2.0**-24)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_floor_still_works_and_is_independent(self):
        import oasr

        power, filters = self._power(), self._filters()
        got = oasr.mel_log(power, filters, log_floor=1e-2, log_offset=0.0)
        ref = torch.log((power @ filters.t()).clamp_min(1e-2))
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_frame_lengths_zero_the_padded_tail(self):
        """A padded frame's log is a large negative constant, not zero."""
        import oasr

        power, filters = self._power(), self._filters()
        lens = torch.tensor([5, 2], dtype=torch.int32, device="cuda")
        got = oasr.mel_log(power, filters, log_floor=0.0, log_offset=2.0**-24, frame_lengths=lens)
        unmasked = oasr.mel_log(power, filters, log_floor=0.0, log_offset=2.0**-24)
        assert got[1, 2:].abs().max() == 0
        assert unmasked[1, 2:].abs().min() > 1.0, "the tail was already zero — bad fixture"
        torch.testing.assert_close(got[0], unmasked[0])
        torch.testing.assert_close(got[1, :2], unmasked[1, :2])

    def test_frame_lengths_needs_a_batched_power_tensor(self):
        import oasr

        with pytest.raises(ValueError, match="3-D"):
            oasr.mel_log(
                torch.rand(5, 33, device="cuda"),
                self._filters(),
                frame_lengths=torch.tensor([5], dtype=torch.int32, device="cuda"),
            )


@requires_cuda
class TestKernelBackedBatchedKaldi:
    """The production batched extractor reaches the KG16 kernel chain."""

    @pytest.mark.parametrize("feature_type", ["fbank", "mfcc"])
    def test_varlen_kernel_path_matches_the_torch_oracle(self, feature_type, monkeypatch):
        from oasr.features import FeatureConfig
        from oasr.features.batched import batched_fbank, batched_mfcc

        monkeypatch.delenv("OASR_FEATURE_BACKEND", raising=False)
        torch.manual_seed(6)
        lengths = torch.tensor([16000, 9371, 400])
        waveforms = torch.zeros(3, 16000)
        for b, length in enumerate(lengths.tolist()):
            waveforms[b, :length] = torch.randn(length) * 1000.0
        cfg = FeatureConfig(
            feature_type=feature_type,
            num_mel_bins=23 if feature_type == "mfcc" else 80,
            num_ceps=13,
            dither=0.0,
        )
        extract = batched_mfcc if feature_type == "mfcc" else batched_fbank
        expected, expected_lengths = extract(waveforms, lengths, cfg)
        got, got_lengths = extract(waveforms.cuda(), lengths.cuda(), cfg)

        assert torch.equal(got_lengths.cpu(), expected_lengths)
        for b, count in enumerate(expected_lengths.tolist()):
            torch.testing.assert_close(
                got[b, :count].cpu(), expected[b, :count], rtol=1e-3, atol=1e-2
            )
