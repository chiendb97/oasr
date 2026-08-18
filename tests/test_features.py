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
class TestGraphedFeatureExtraction:
    """Bit-exact parity between captured replay and the eager batched path."""

    def _make_cfg(self, *, feature_type: str = "fbank"):
        from oasr.features import FeatureConfig

        return FeatureConfig(dither=0.0, feature_type=feature_type)

    def _eager_feats(
        self,
        wave_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
        bucket: int,
        t_pad: int,
        cfg,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run the unbatched-graph eager path on a (bucket, t_pad) buffer."""
        from oasr.features.batched import batched_fbank, batched_mfcc

        device = torch.device("cuda")
        B_active = wave_cpu.size(0)
        T = wave_cpu.size(1)
        wav_gpu = torch.zeros(bucket, t_pad, dtype=torch.float32, device=device)
        wav_gpu[:B_active, :T] = wave_cpu.to(device)
        len_gpu = torch.zeros(bucket, dtype=torch.int64, device=device)
        len_gpu[:B_active] = lengths_cpu.to(device)
        fn = batched_mfcc if cfg.feature_type == "mfcc" else batched_fbank
        feats_f32, _ = fn(wav_gpu, len_gpu, cfg)
        return feats_f32.to(dtype=out_dtype)

    def test_batched_fbank_cuda_graph_matches_eager(self):
        """Replay output matches an eager call on the same padded inputs."""
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg()
        device = torch.device("cuda")
        chunk_samples = 16 * 4 * cfg.frame_shift_samples  # ASR steady-state stride
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=device,
            feature_config=cfg,
            output_dtype=torch.float16,
            chunk_samples=chunk_samples,
            max_batch_size=8,
        )
        assert gfe.buckets == [1, 2, 4, 8]

        torch.manual_seed(0)
        for B_active in [1, 2, 3, 5, 8]:
            bucket = gfe.pick_bucket(B_active)
            assert bucket is not None
            # Vary T inside [frame_len, t_pad].
            for T in [
                cfg.frame_length_samples,
                chunk_samples,
                chunk_samples + 117,
                gfe.t_pad,
            ]:
                wave = torch.randn(B_active, T, dtype=torch.float32) * 1000.0
                lengths = torch.full((B_active,), T, dtype=torch.int64)

                feats_graph_view = gfe.replay(B_active, wave, lengths)
                assert feats_graph_view is not None
                assert feats_graph_view.dtype == torch.float16
                # Caller slices to the first B_active rows.
                feats_graph = feats_graph_view[:B_active].clone()

                feats_eager = self._eager_feats(
                    wave, lengths, bucket, gfe.t_pad, cfg, torch.float16
                )[:B_active]

                torch.testing.assert_close(
                    feats_graph,
                    feats_eager,
                    rtol=0,
                    atol=0,
                    msg=f"B_active={B_active} T={T}",
                )

    def test_batched_mfcc_cuda_graph_matches_eager(self):
        """Same bit-exact check on the MFCC path."""
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg(feature_type="mfcc")
        device = torch.device("cuda")
        chunk_samples = 16 * 4 * cfg.frame_shift_samples
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=device,
            feature_config=cfg,
            output_dtype=torch.float32,
            chunk_samples=chunk_samples,
            max_batch_size=4,
        )

        torch.manual_seed(1)
        for B_active in [1, 4]:
            bucket = gfe.pick_bucket(B_active)
            T = chunk_samples + 17
            wave = torch.randn(B_active, T, dtype=torch.float32) * 500.0
            lengths = torch.full((B_active,), T, dtype=torch.int64)
            feats_graph = gfe.replay(B_active, wave, lengths)[:B_active].clone()
            feats_eager = self._eager_feats(wave, lengths, bucket, gfe.t_pad, cfg, torch.float32)[
                :B_active
            ]
            torch.testing.assert_close(feats_graph, feats_eager, rtol=0, atol=0)

    def test_pick_bucket_returns_smallest_fit(self):
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg()
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=torch.device("cuda"),
            feature_config=cfg,
            output_dtype=torch.float16,
            chunk_samples=16 * 4 * cfg.frame_shift_samples,
            max_batch_size=16,
        )
        assert gfe.buckets == [1, 2, 4, 8, 16]
        assert gfe.pick_bucket(1) == 1
        assert gfe.pick_bucket(3) == 4
        assert gfe.pick_bucket(8) == 8
        assert gfe.pick_bucket(9) == 16
        assert gfe.pick_bucket(17) is None  # oversize → eager fallback

    def test_custom_batch_buckets_override(self):
        """Explicit ``batch_buckets`` overrides the default power-of-two ladder."""
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg()
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=torch.device("cuda"),
            feature_config=cfg,
            output_dtype=torch.float16,
            chunk_samples=16 * 4 * cfg.frame_shift_samples,
            max_batch_size=64,
            batch_buckets=[8, 32],
        )
        assert gfe.buckets == [8, 32]
        assert gfe.pick_bucket(1) == 8
        assert gfe.pick_bucket(8) == 8
        assert gfe.pick_bucket(9) == 32
        assert gfe.pick_bucket(33) is None

    def test_oversize_returns_none(self):
        """Combined waveform longer than ``t_pad`` triggers the eager fallback."""
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg()
        chunk_samples = 16 * 4 * cfg.frame_shift_samples
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=torch.device("cuda"),
            feature_config=cfg,
            output_dtype=torch.float16,
            chunk_samples=chunk_samples,
            max_batch_size=4,
        )
        oversize_T = gfe.t_pad + 1
        wave = torch.zeros(2, oversize_T, dtype=torch.float32)
        lengths = torch.tensor([oversize_T, oversize_T], dtype=torch.int64)
        assert gfe.replay(2, wave, lengths) is None


class TestCollateOutputIsNotReused:
    """``collate`` must return a fresh feature tensor every call.

    ``OfflineExecutor``'s collate prefetch holds *two* micro-batches' features
    alive at once — one being forwarded, one staged for the next tick.  Every
    other buffer on that path (``_wav_flat``, ``_wav_padded``) is deliberately
    reused, so the one that must not be is worth pinning: a reused feature
    buffer would let the staged batch overwrite the one the encoder is reading,
    which corrupts a transcript rather than raising.  The graph-replay buffers
    are the same class of hazard, and they *did* ship.
    """

    def _proc(self):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="x", device="cpu", dtype=torch.float32)
        return InputProcessor(cfg, torch.device("cpu"))

    def _batch(self, proc, n, samples):
        from oasr.engine.request import Request

        out = []
        for i in range(n):
            req = Request(request_id=f"c{i}", streaming=False)
            req.audio = torch.randn(samples)
            req.sample_rate = 16000
            proc.prepare_offline(req)
            out.append(req)
        return out

    def test_two_collates_do_not_alias(self):
        proc = self._proc()
        # Held simultaneously, which is what the prefetch does — otherwise the
        # caching allocator could hand back the same address legitimately.
        f1, l1 = proc.collate(self._batch(proc, 4, 16000))
        f2, l2 = proc.collate(self._batch(proc, 4, 16000))
        assert f1.shape == f2.shape
        assert f1.data_ptr() != f2.data_ptr()
        assert l1.data_ptr() != l2.data_ptr()


class TestStagingBuffers:
    """M4/M5: staging must be reused per step but bounded across the process."""

    def _proc(self, **overrides):

        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="x", device="cpu", **overrides)
        return InputProcessor(cfg, torch.device("cpu"))

    def test_offline_buffer_is_reused_between_calls(self):
        p = self._proc()
        a = p._flat_host(1024)
        b = p._flat_host(1024)
        assert a.data_ptr() == b.data_ptr()

    def test_offline_buffer_grows_geometrically(self):
        p = self._proc()
        p._flat_host(1024)
        first = p._wav_flat.numel()
        p._flat_host(first + 1)
        assert p._wav_flat.numel() >= 2 * first

    def test_an_outlier_batch_is_not_retained(self):
        """One huge request must not pin its peak for the process lifetime.

        Geometric growth sized by the longest utterance ever seen never shrinks,
        and pinned host memory is process-global.
        """
        p = self._proc()
        p._max_staging_elems = 4096
        p._flat_host(1024)
        retained = p._wav_flat.numel()
        big = p._flat_host(1_000_000)
        assert big.numel() == 1_000_000
        assert p._wav_flat.numel() == retained, "the outlier was retained"

    def test_retained_buffer_never_exceeds_the_cap(self):
        p = self._proc()
        p._max_staging_elems = 4096
        p._flat_host(4096)
        assert p._wav_flat.numel() <= 4096

    def test_streaming_staging_is_reused_within_a_slot(self):
        p = self._proc()
        slot = p._next_stream_slot()
        a = p._stream_host(slot, 4, 100)
        b = p._stream_host(slot, 4, 100)
        assert a.data_ptr() == b.data_ptr()
        la = p._stream_lengths_host(slot, 4)
        lb = p._stream_lengths_host(slot, 4)
        assert la.data_ptr() == lb.data_ptr()

    def test_consecutive_streaming_steps_get_different_buffers(self):
        """Double buffering: back-to-back steps must not share staging memory.

        The pinned pair is read by an async H2D, so a step that rewrites the
        buffer the previous step's copy is still reading corrupts it — measured
        as nemotron streaming WER 2.44% -> 2.53% under a co-tenant GPU load, and
        the *reason* the buffers rotate rather than being reused in place
        (`.artifacts/known_issues.md`
        "Lessons that outlived their bugs").  Reuse two steps apart is fine, and
        gated by the slot's completion event on CUDA.
        """
        p = self._proc()
        seen = []
        for _ in range(4):
            slot = p._next_stream_slot()
            seen.append(p._stream_host(slot, 4, 100).data_ptr())
            p._stream_lengths_host(slot, 4)
        assert seen[0] != seen[1], "consecutive steps shared a staging buffer"
        assert seen[0] == seen[2] and seen[1] == seen[3], "slots should cycle, not grow"

    def test_release_drops_everything(self):
        p = self._proc()
        p._flat_host(64)
        slot = p._next_stream_slot()
        p._stream_host(slot, 2, 8)
        p._stream_lengths_host(slot, 2)
        p.release_staging()
        assert p._wav_flat is None
        assert all(s.flat is None and s.lens is None for s in p._stream_slots)
        assert all(s.ready is None for s in p._stream_slots)


class TestPinnedAudioBuffers:
    """``new_audio_buffer`` — the buffer the front-end fills so ``collate``
    can DMA straight from it instead of packing the batch into staging."""

    def _proc(self, device="cpu", **overrides):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="x", device=device, **overrides)
        return InputProcessor(cfg, torch.device(device))

    def test_cpu_engine_declines(self):
        """No CUDA context to page-lock against — the caller uses the heap."""
        assert self._proc().new_audio_buffer(16000) is None

    @pytest.mark.cuda
    def test_offers_pinned_memory(self, device):
        p = self._proc(device="cuda")
        buf = p.new_audio_buffer(16000)
        assert buf is not None
        assert buf.is_pinned() and buf.dtype is torch.float32
        assert buf.numel() == 16000

    @pytest.mark.cuda
    def test_declines_past_the_cap(self, device):
        """Page-locked memory is process-global; one long request must not be
        able to reserve an unbounded amount of it."""
        p = self._proc(device="cuda", max_pinned_audio_seconds=1.0)
        sr = p._feature_config.sample_rate
        assert p.new_audio_buffer(sr) is not None
        assert p.new_audio_buffer(sr + 1) is None

    @pytest.mark.cuda
    def test_zero_cap_declines_everything(self, device):
        p = self._proc(device="cuda", max_pinned_audio_seconds=0.0)
        assert p.new_audio_buffer(16000) is None

    def test_non_positive_size_declines(self):
        p = self._proc()
        assert p.new_audio_buffer(0) is None
        assert p.new_audio_buffer(-1) is None

    @pytest.mark.cuda
    def test_pinned_and_unpinned_batches_agree(self, device):
        """The two collate paths must produce the same device batch.

        The pinned path skips the host pack entirely and DMAs each row into
        place; the unpinned one packs into staging first.  Padding and
        ``audio_scale`` are applied identically (on the GPU, after padding), so
        this is an exact comparison, not a tolerance.
        """
        p = self._proc(device="cuda")
        waves = [torch.randn(n) for n in (16000, 12345, 8000, 1)]
        plain = p._padded_waveform_batch(list(waves))
        pinned = p._padded_waveform_batch([w.pin_memory() for w in waves])
        torch.cuda.synchronize()
        assert torch.equal(plain, pinned)

    @pytest.mark.cuda
    def test_mixed_batch_takes_the_pack_path(self, device):
        """One unpinned row sends the whole micro-batch through the pack — and
        still produces the same batch."""
        p = self._proc(device="cuda")
        waves = [torch.randn(4000), torch.randn(4000)]
        mixed = [waves[0].pin_memory(), waves[1]]
        torch.cuda.synchronize()
        assert torch.equal(p._padded_waveform_batch(list(waves)), p._padded_waveform_batch(mixed))


class TestStreamingFeatureStreamHandoff:
    """The feature stream → default stream hand-off must be ordered.

    ``extract_streaming_batch`` runs the H2D and the frontend on a caller-supplied
    stream so they overlap the previous step's encoder forward, then appends the
    result into each request's ``feature_buffer`` **on the current stream**.  That
    cross-stream read has to be ordered inside the method: the step loop's own
    ``wait_stream`` fires only after it returns, which is after the append has
    already been issued.

    Unordered, the append reads feature memory the frontend has not finished
    writing.  An idle GPU always hides it — the kernels beat the host to it — so
    the reproduction here **congests the feature stream on purpose** rather than
    relying on timing.  Measured cost of the missing wait on real audio: conformer
    streaming 3.70% → 99.32% WER with 195 of 200 transcripts empty, nemotron
    2.44% → 59.71%, in both cases NaN arriving one whole mel frame at a time with
    nothing raised (`.artifacts/known_issues.md` "Lessons that outlived their bugs").
    """

    def _processor_and_requests(self, device, n=3, samples=8000):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor
        from oasr.engine.request import Request
        from oasr.features import FeatureConfig

        cfg = EngineConfig(
            ckpt_dir="x",
            device=str(device),
            dtype=torch.float32,
            max_batch_size=8,
            feature_config=FeatureConfig(feature_type="fbank", num_mel_bins=80, dither=0.0),
            use_cuda_graphs=False,  # exercise the eager path; the graph path
            # stages through its own captured buffers
        )
        proc = InputProcessor(cfg, device)
        torch.manual_seed(1234)  # same audio in both arms
        reqs = []
        for i in range(n):
            req = Request(None, request_id=f"r{i}", streaming=True)
            proc.prepare_streaming(req)
            proc.append_streaming_chunk(req, torch.randn(samples).clamp(-1, 1))
            reqs.append(req)
        return proc, reqs

    @pytest.mark.cuda
    def test_features_survive_a_congested_feature_stream(self, device):
        """Features must match the single-stream result, however backed-up the
        feature stream is when the append is issued."""
        proc_ref, reqs_ref = self._processor_and_requests(device)
        proc_ref.extract_streaming_batch(reqs_ref, cuda_stream=None)
        torch.cuda.synchronize()
        reference = [r.feature_buffer[: r.feature_frames].clone() for r in reqs_ref]

        proc, reqs = self._processor_and_requests(device)
        feat_stream = torch.cuda.Stream(device=device)
        # Queue enough work on the feature stream that the frontend's kernels
        # cannot possibly have completed by the time the append is enqueued.
        with torch.cuda.stream(feat_stream):
            a = torch.randn(2048, 2048, device=device)
            b = torch.randn(2048, 2048, device=device)
            for _ in range(60):
                a = (a @ b).mul_(1e-4)
        proc.extract_streaming_batch(reqs, cuda_stream=feat_stream)
        torch.cuda.synchronize()

        for i, (req, ref) in enumerate(zip(reqs, reference)):
            got = req.feature_buffer[: req.feature_frames]
            assert not torch.isnan(got).any(), f"stream {i}: NaN in features"
            assert got.shape == ref.shape
            torch.testing.assert_close(got, ref, rtol=0, atol=0)


# ===========================================================================
# stft_frame / mel_log kernels (KG23's shared primitives)
# ===========================================================================


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


class TestStreamingFeatureAppend:
    """``_plan_append_features`` — the batched feature-buffer append.

    Appending each stream's new frames one call at a time was 146 device-to-device
    copies per streaming step carrying 0.09 ms of work: 21% of the wall clock
    spent submitting copies, not making them.  The batched form plans every
    stream's growth on the host and commits the copies as one
    ``torch._foreach_copy_``, whose members are unordered with respect to each
    other — so what has to hold is that no queued pair reads what another writes,
    and that the buffer contents are what the per-stream form produced.

    The compaction path is the one that had to change shape: a standalone
    compaction left a buffer exactly as long as what it kept, so a grow *always*
    followed it, chaining old->keep->new.  Two copies of the same frames, and two
    that could not have shared a batch.
    """

    def _proc(self):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        return InputProcessor(EngineConfig(ckpt_dir="x", device="cpu"), torch.device("cpu"))

    @staticmethod
    def _req():
        from oasr.engine.request import Request

        req = Request(request_id="r", streaming=True)
        req.feature_buffer = None
        req.feature_frames = 0
        req.feature_cursor = 0
        return req

    def _append(self, proc, req, frames, feat_dim):
        dsts, srcs = [], []
        proc._plan_append_features(req, frames, feat_dim, dsts, srcs)
        for d, s in zip(dsts, srcs):
            d.copy_(s)
        return req

    def test_appends_are_contiguous_and_ordered(self):
        proc, req, F = self._proc(), self._req(), 4
        expected = []
        for step in range(6):
            frames = torch.full((3, F), float(step))
            expected.append(frames)
            self._append(proc, req, frames, F)
        want = torch.cat(expected)
        assert req.feature_frames == want.size(0)
        torch.testing.assert_close(req.feature_buffer[: req.feature_frames], want)

    def test_consumed_prefix_is_dropped_without_losing_live_frames(self):
        """Compaction moves the cursor to 0 and keeps everything after it."""
        proc, req, F = self._proc(), self._req(), 4
        for step in range(4):
            self._append(proc, req, torch.full((4, F), float(step)), F)
        assert req.feature_frames == 16

        req.feature_cursor = 12  # >= have // 2, so the next append compacts
        live = req.feature_buffer[12:16].clone()
        self._append(proc, req, torch.full((4, F), 99.0), F)

        assert req.feature_cursor == 0, "compaction must rebase the cursor"
        assert req.feature_frames == 8
        torch.testing.assert_close(req.feature_buffer[:4], live)
        torch.testing.assert_close(req.feature_buffer[4:8], torch.full((4, F), 99.0))

    def test_compaction_relocates_once(self):
        """Not old->keep->new: one allocation, one copy of the retained frames."""
        proc, req, F = self._proc(), self._req(), 4
        for step in range(4):
            self._append(proc, req, torch.full((4, F), float(step)), F)
        req.feature_cursor = 12

        dsts, srcs = [], []
        proc._plan_append_features(req, torch.full((4, F), 99.0), F, dsts, srcs)
        assert len(dsts) == 2, "one relocation copy plus the append, and no more"
        # Every destination lives in the new buffer; no source does — which is
        # what makes the pairs safe to run unordered.
        new_storage = req.feature_buffer.untyped_storage().data_ptr()
        assert all(d.untyped_storage().data_ptr() == new_storage for d in dsts)
        assert all(s.untyped_storage().data_ptr() != new_storage for s in srcs)

    def test_growth_preserves_the_unconsumed_prefix(self):
        """A grow with a small cursor must not drop the frames before it."""
        proc, req, F = self._proc(), self._req(), 4
        self._append(proc, req, torch.arange(200 * F, dtype=torch.float32).reshape(200, F), F)
        req.feature_cursor = 1  # well under have // 2: no compaction, only growth
        before = req.feature_buffer[:200].clone()
        self._append(proc, req, torch.full((200, F), 7.0), F)

        assert req.feature_cursor == 1, "growth alone must not rebase the cursor"
        assert req.feature_frames == 400
        torch.testing.assert_close(req.feature_buffer[:200], before)
        torch.testing.assert_close(req.feature_buffer[200:400], torch.full((200, F), 7.0))


class TestStreamingSegmentPack:
    """The streaming pack keeps each stream's pieces apart until one batched
    ``cat`` writes them straight into the staging row.

    Two costs were being paid to build a buffer that is copied again a moment
    later: concatenating the carried-over tail onto every stream's chunk
    (0.44 ms/step at 64 streams), then copying each result into the padded
    staging batch (0.46 ms/step).  Packing the pieces directly is one pass
    instead of two.

    What has to hold is that the packed batch is *exactly* what concatenating
    would have produced — same bytes, same padding, same lengths — for ragged
    cohorts as well as steady-state ones, since a wrong row here is a wrong
    transcript with nothing raised.
    """

    def _proc(self, **overrides):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="x", device="cpu", **overrides)
        return InputProcessor(cfg, torch.device("cpu"))

    @staticmethod
    def _inputs(pieces, flush=False):
        from oasr.engine.input_processor import _StreamInput

        return [
            _StreamInput(
                request=None,  # type: ignore[arg-type]
                segments=segs,
                n_samples=sum(int(s.numel()) for s in segs),
                flush=flush,
            )
            for segs in pieces
        ]

    def _pack(self, proc, inputs):
        """The real packing stage, not a re-implementation of it — a test that
        restates the layout rule cannot catch the layout rule changing."""
        t_max = max(inp.n_samples for inp in inputs)
        slot = proc._next_stream_slot()
        return proc._pack_streaming_waveforms(inputs, slot, t_max)

    @staticmethod
    def _reference(pieces, t_max):
        """What the per-stream concatenate-then-copy form produced."""
        out = torch.zeros(len(pieces), t_max)
        for i, segs in enumerate(pieces):
            cat = torch.cat(segs)
            out[i, : cat.numel()] = cat
        return out

    def test_equal_length_rows_match_the_concatenated_form(self):
        proc = self._proc()
        pieces = [[torch.randn(37), torch.randn(320)] for _ in range(6)]
        packed = self._pack(proc, self._inputs(pieces))
        assert torch.equal(packed, self._reference(pieces, 357))

    def test_ragged_rows_match_and_are_zero_padded(self):
        proc = self._proc()
        pieces = [
            [torch.randn(37), torch.randn(320)],
            [torch.randn(320)],  # freshly admitted stream: no carry-over tail
            [torch.randn(11), torch.randn(200), torch.randn(9)],  # closing flush pad
        ]
        packed = self._pack(proc, self._inputs(pieces))
        assert torch.equal(packed, self._reference(pieces, 357))
        assert torch.equal(packed[1, 320:], torch.zeros(37))

    def test_the_shared_pad_buffer_is_never_written_through(self):
        """Every ragged row borrows the same zero run, so a row that wrote back
        into it would silently corrupt its peers in the same step."""
        proc = self._proc()
        pieces = [[torch.randn(400)], [torch.randn(10)], [torch.randn(10)]]
        self._pack(proc, self._inputs(pieces))
        assert torch.count_nonzero(proc._stream_pad) == 0
        packed = self._pack(proc, self._inputs(pieces))
        assert torch.equal(packed, self._reference(pieces, 400))


class TestStreamingTailSuffix:
    """``_suffix`` — the retained carry-over, without materialising the join.

    ``torch.cat(segments)[start:]`` is the rule; the point is to get there
    without the ``cat``.  In steady state ``start`` lands inside the last
    segment and the result is a view, which is the whole reason the pieces are
    kept apart.
    """

    @staticmethod
    def _suffix(segments, start):
        from oasr.engine.input_processor import _suffix

        return _suffix(segments, start)

    @pytest.mark.parametrize("start", [0, 1, 9, 10, 11, 24, 25, 26, 39, 40])
    def test_matches_the_concatenated_slice(self, start):
        segs = [torch.randn(10), torch.randn(15), torch.randn(15)]
        assert torch.equal(self._suffix(segs, start), torch.cat(segs)[start:])

    def test_a_start_inside_the_last_segment_is_a_view(self):
        segs = [torch.randn(10), torch.randn(15)]
        out = self._suffix(segs, 20)
        assert out.data_ptr() == segs[1][10:].data_ptr(), "steady state must not copy"

    def test_a_start_past_the_end_is_empty(self):
        segs = [torch.randn(10), torch.randn(15)]
        assert self._suffix(segs, 25).numel() == 0

    def test_a_single_segment_behaves_like_a_slice(self):
        segs = [torch.randn(20)]
        assert torch.equal(self._suffix(segs, 7), segs[0][7:])


class TestStreamingAudioScaleSites:
    """``audio_scale`` must be applied exactly once, on whichever copy exists.

    The streaming pack writes **raw** samples so it stays one pass over the
    batch; the multiply then rides on the device copy.  Two paths have no device
    copy to ride on — a CPU engine, and the captured feature graph, which owns
    its own H2D — and scale the pinned host buffer instead.

    The hazard is the seam: a feature-graph *bucket miss* returns ``None`` and
    falls through to the eager path with a buffer that has already been scaled.
    Scaling it again is silent — every sample off by ``audio_scale``, which for
    a WeNet checkpoint is 32768.
    """

    def _processor(self, device, scale):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor
        from oasr.features import FeatureConfig

        cfg = EngineConfig(
            ckpt_dir="x",
            device=str(device),
            dtype=torch.float32,
            max_batch_size=8,
            audio_scale=scale,
            feature_config=FeatureConfig(feature_type="fbank", num_mel_bins=80, dither=0.0),
            use_cuda_graphs=False,
        )
        return InputProcessor(cfg, device)

    @staticmethod
    def _inputs(waves):
        from oasr.engine.input_processor import _StreamInput

        return [
            _StreamInput(request=None, segments=[w], n_samples=int(w.numel()), flush=False)  # type: ignore[arg-type]
            for w in waves
        ]

    @pytest.mark.cuda
    def test_a_feature_graph_miss_does_not_scale_twice(self, device):
        if device.type != "cuda":
            pytest.skip("the fallthrough only exists on the CUDA path")
        from types import SimpleNamespace

        torch.manual_seed(7)
        waves = [torch.randn(4000).clamp(-1, 1) for _ in range(3)]

        proc = self._processor(device, 32768.0)
        expected, _ = proc._run_streaming_features(self._inputs(waves), None)

        # A graph whose bucket never matches: replay returns None, so the eager
        # path runs on a buffer the graph path has already scaled.
        missing = self._processor(device, 32768.0)
        missing._feature_graph = SimpleNamespace(t_pad=1 << 30, replay=lambda *a, **k: None)
        got, _ = missing._run_streaming_features(self._inputs(waves), None)

        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @pytest.mark.cuda
    def test_the_device_scale_matches_scaling_on_the_host(self, device):
        if device.type != "cuda":
            pytest.skip("compares the device site against the host site")
        torch.manual_seed(11)
        waves = [torch.randn(4000).clamp(-1, 1) for _ in range(3)]

        scaled = self._processor(device, 32768.0)
        got, _ = scaled._run_streaming_features(self._inputs(waves), None)

        # The same audio pre-scaled on the host, through an engine that does not
        # scale at all — the two must agree bit for bit.
        plain = self._processor(device, 1.0)
        expected, _ = plain._run_streaming_features(
            self._inputs([w * 32768.0 for w in waves]), None
        )

        torch.testing.assert_close(got, expected, rtol=0, atol=0)
