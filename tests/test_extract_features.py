#!/usr/bin/env python3
"""End-to-end tests for the Fbank / Mfcc CUDA-backed modules.

Validates against ``torchaudio.compliance.kaldi.{fbank,mfcc}`` for the
inference profile (``dither=0``, ``snip_edges=True``, ``use_energy=False``).
"""

from __future__ import annotations

import math

import pytest
import torch

torchaudio = pytest.importorskip("torchaudio")
import torchaudio.compliance.kaldi as kaldi  # noqa: E402

import oasr  # noqa: E402
from oasr.feature import dct_lifter, fbank_preprocess, mel_log  # noqa: E402
from oasr.features import FeatureConfig  # noqa: E402
from oasr.layers import Fbank, Mfcc  # noqa: E402


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _kaldi_fbank(waveform: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    return kaldi.fbank(
        waveform.unsqueeze(0).cpu(),
        sample_frequency=float(cfg.sample_rate),
        num_mel_bins=cfg.num_mel_bins,
        frame_length=cfg.frame_length_ms,
        frame_shift=cfg.frame_shift_ms,
        dither=0.0,
        energy_floor=0.0,
        preemphasis_coefficient=cfg.preemphasis_coefficient,
        window_type=cfg.window_type,
        use_energy=False,
        low_freq=cfg.low_freq,
        high_freq=cfg.high_freq,
        snip_edges=True,
    )


def _kaldi_mfcc(waveform: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    return kaldi.mfcc(
        waveform.unsqueeze(0).cpu(),
        sample_frequency=float(cfg.sample_rate),
        num_mel_bins=cfg.num_mel_bins,
        num_ceps=cfg.num_ceps,
        cepstral_lifter=cfg.cepstral_lifter,
        frame_length=cfg.frame_length_ms,
        frame_shift=cfg.frame_shift_ms,
        dither=0.0,
        energy_floor=0.0,
        preemphasis_coefficient=cfg.preemphasis_coefficient,
        window_type=cfg.window_type,
        use_energy=False,
        low_freq=cfg.low_freq,
        high_freq=cfg.high_freq,
        snip_edges=True,
    )


# ---------------------------------------------------------------------------
# Low-level kernel parity tests
# ---------------------------------------------------------------------------


@CUDA
class TestFbankPreprocess:
    def test_matches_reference(self):
        torch.manual_seed(0)
        B, F, L = 4, 64, 400
        n_fft = 512
        preemph = 0.97
        frames = torch.randn(B, F, L, device="cuda", dtype=torch.float32)
        window = torch.rand(L, device="cuda", dtype=torch.float32)

        out = fbank_preprocess(frames, window, n_fft=n_fft, preemph_coef=preemph)

        # Reference: DC removal + preemphasis (replicate boundary) + window + zero-pad.
        ref = frames - frames.mean(dim=-1, keepdim=True)
        preem = torch.empty_like(ref)
        preem[..., 1:] = ref[..., 1:] - preemph * ref[..., :-1]
        preem[..., 0] = ref[..., 0] - preemph * ref[..., 0]
        windowed = preem * window
        ref_out = torch.nn.functional.pad(windowed, (0, n_fft - L))

        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)

    def test_no_preemph_no_dc(self):
        x = torch.randn(2, 8, 200, device="cuda", dtype=torch.float32)
        w = torch.ones(200, device="cuda", dtype=torch.float32)
        out = fbank_preprocess(
            x, w, n_fft=256, preemph_coef=0.0,
            remove_dc_offset=False, apply_preemph=False,
        )
        # With no DC removal, no preemph, unit window: out is just zero-padded x.
        ref = torch.nn.functional.pad(x, (0, 56))
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


@CUDA
class TestMelLog:
    def test_matches_reference(self):
        torch.manual_seed(0)
        T, F, M = 32, 257, 80
        power = torch.rand(T, F, device="cuda", dtype=torch.float32) + 1e-3
        mel_mat = torch.rand(M, F, device="cuda", dtype=torch.float32)

        out = mel_log(power, mel_mat, log_floor=1e-30)

        ref = torch.log((power @ mel_mat.t()).clamp_min(1e-30))
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_log_floor_applied(self):
        T, F, M = 4, 8, 4
        power = torch.zeros(T, F, device="cuda", dtype=torch.float32)
        mel_mat = torch.rand(M, F, device="cuda", dtype=torch.float32)
        out = mel_log(power, mel_mat, log_floor=1e-10)
        expected = math.log(1e-10)
        assert torch.allclose(out, torch.full_like(out, expected), rtol=1e-4)


@CUDA
class TestDctLifter:
    def test_matches_reference(self):
        torch.manual_seed(0)
        T, M, C = 16, 80, 13
        log_mel = torch.randn(T, M, device="cuda", dtype=torch.float32)
        dct = torch.randn(C, M, device="cuda", dtype=torch.float32)
        lifter = torch.rand(C, device="cuda", dtype=torch.float32) + 0.5

        out = dct_lifter(log_mel, dct, lifter=lifter)
        ref = (log_mel @ dct.t()) * lifter
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_no_lifter(self):
        T, M, C = 8, 23, 13
        log_mel = torch.randn(T, M, device="cuda", dtype=torch.float32)
        dct = torch.randn(C, M, device="cuda", dtype=torch.float32)
        out = dct_lifter(log_mel, dct, lifter=None)
        torch.testing.assert_close(out, log_mel @ dct.t(), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# End-to-end module parity vs torchaudio.compliance.kaldi
# ---------------------------------------------------------------------------


def _make_sine(sample_rate: int, duration_s: float, freq: float = 220.0) -> torch.Tensor:
    n = int(sample_rate * duration_s)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return torch.sin(2.0 * math.pi * freq * t) + 0.1 * torch.randn_like(t)


@CUDA
class TestFbank:
    @pytest.mark.parametrize(
        "num_mel_bins,frame_length_ms",
        [(80, 25.0), (40, 25.0), (80, 32.0)],
    )
    @pytest.mark.parametrize("window_type", ["povey", "hanning", "hamming"])
    def test_parity_kaldi(self, num_mel_bins, frame_length_ms, window_type):
        cfg = FeatureConfig(
            feature_type="fbank",
            sample_rate=16000,
            num_mel_bins=num_mel_bins,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=10.0,
            window_type=window_type,
            dither=0.0,
        )
        wav = _make_sine(cfg.sample_rate, 1.0)

        fb = Fbank(cfg).cuda()
        feats, feat_lens = fb(wav.cuda())

        ref = _kaldi_fbank(wav, cfg)
        # Drop the synthetic batch dim from torchaudio output.
        torch.testing.assert_close(
            feats[0, : feat_lens[0]], ref.to("cuda"), rtol=1e-3, atol=1e-2
        )

    def test_batched(self):
        cfg = FeatureConfig(num_mel_bins=80)
        fb = Fbank(cfg).cuda()
        wavs = torch.stack(
            [_make_sine(16000, 1.0), _make_sine(16000, 1.0, freq=440.0)], dim=0
        )
        feats, feat_lens = fb(wavs.cuda())

        for i in range(2):
            ref = _kaldi_fbank(wavs[i], cfg)
            torch.testing.assert_close(
                feats[i, : feat_lens[i]], ref.to("cuda"), rtol=1e-3, atol=1e-2
            )

    def test_lengths(self):
        cfg = FeatureConfig(num_mel_bins=40)
        fb = Fbank(cfg).cuda()
        wav = torch.randn(2, 16000, device="cuda")
        lengths = torch.tensor([12000, 16000], dtype=torch.int32, device="cuda")
        feats, feat_lens = fb(wav, lengths=lengths)

        # snip_edges=True: feat_len = (samples - frame_length) // frame_shift + 1
        expected_lens = (lengths - cfg.frame_length_samples) // cfg.frame_shift_samples + 1
        assert torch.equal(feat_lens, expected_lens.to(torch.int32))

    def test_short_input(self):
        cfg = FeatureConfig(num_mel_bins=40)
        fb = Fbank(cfg).cuda()
        # Less than one frame -> empty output.
        wav = torch.randn(2, 100, device="cuda")
        feats, feat_lens = fb(wav)
        assert feats.shape == (2, 0, 40)
        assert torch.equal(feat_lens, torch.zeros(2, dtype=torch.int32, device="cuda"))

    def test_cpu_input_rejected(self):
        cfg = FeatureConfig(num_mel_bins=40)
        fb = Fbank(cfg).cuda()
        with pytest.raises(ValueError):
            fb(torch.randn(2, 16000))


@CUDA
class TestMfcc:
    @pytest.mark.parametrize(
        "num_ceps,num_mel_bins,cepstral_lifter",
        [(13, 23, 22.0), (20, 40, 22.0), (13, 23, 0.0)],
    )
    def test_parity_kaldi(self, num_ceps, num_mel_bins, cepstral_lifter):
        cfg = FeatureConfig(
            feature_type="mfcc",
            sample_rate=16000,
            num_mel_bins=num_mel_bins,
            num_ceps=num_ceps,
            cepstral_lifter=cepstral_lifter,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            window_type="povey",
            dither=0.0,
        )
        wav = _make_sine(cfg.sample_rate, 1.0)

        mf = Mfcc(cfg).cuda()
        feats, feat_lens = mf(wav.cuda())

        ref = _kaldi_mfcc(wav, cfg)
        torch.testing.assert_close(
            feats[0, : feat_lens[0]], ref.to("cuda"), rtol=1e-3, atol=1e-2
        )

    def test_output_shape_and_dtype(self):
        cfg = FeatureConfig(feature_type="mfcc", num_mel_bins=23, num_ceps=13)
        mf = Mfcc(cfg).cuda()
        wav = torch.randn(3, 16000, device="cuda")
        feats, feat_lens = mf(wav)
        assert feats.shape == (3, feat_lens[0].item(), 13)
        assert feats.dtype == torch.float32

    def test_no_lifter(self):
        cfg = FeatureConfig(feature_type="mfcc", num_mel_bins=23, num_ceps=13,
                            cepstral_lifter=0.0)
        mf = Mfcc(cfg).cuda()
        # Just verify it runs and produces sensible output.
        wav = torch.randn(1, 16000, device="cuda")
        feats, _ = mf(wav)
        assert torch.isfinite(feats).all()
