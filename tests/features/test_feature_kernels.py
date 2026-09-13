#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The frontend kernels under ``oasr/functionals/feature.py``, and the layers on them.

Validated against ``torchaudio.compliance.kaldi.{fbank,mfcc}`` for the
inference profile (``dither=0``, ``snip_edges=True``, ``use_energy=False``),
which is the only oracle that can catch a frontend-convention bug -- a parity
test against our own batched path feeds both sides the same convention and so
cancels it out.

CMVN lives here too: it is the last pointwise step of the same frontend and
the same ``oasr/functionals/feature.py`` neighbourhood, and as its own
79-line module it was mostly a duplicate grid.
"""

from __future__ import annotations

import math

import pytest
import torch
from helpers import assert_dest_passing, tol

torchaudio = pytest.importorskip("torchaudio")
import torchaudio.compliance.kaldi as kaldi  # noqa: E402

import oasr  # noqa: E402
from oasr.features import FeatureConfig  # noqa: E402
from oasr.functionals.feature import (  # noqa: E402
    dct_lifter,
    fbank_preprocess,
    mel_log,
    whisper_logmel,
)
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
            x,
            w,
            n_fft=256,
            preemph_coef=0.0,
            remove_dc_offset=False,
            apply_preemph=False,
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


def _whisper_reference(power: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """``oasr.whisper_logmel`` written out, for a ``(n_freq, num_mel)`` table."""
    ref = torch.log10((power @ filters).clamp_min(1e-10))
    row_max = ref.amax(dim=(1, 2), keepdim=True)
    return (torch.maximum(ref, row_max - 8.0) + 4.0) / 4.0


@CUDA
class TestWhisperLogMelKernel:
    def test_projection_and_per_row_floor_match_reference(self):
        torch.manual_seed(9)
        power = torch.rand(3, 17, 33, device="cuda")
        filters = torch.rand(33, 8, device="cuda")
        filters *= torch.linspace(0.1, 1.0, 8, device="cuda")

        got = whisper_logmel(power, filters)
        torch.testing.assert_close(got, _whisper_reference(power, filters), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("num_mel", [8, 80, 128])
    @pytest.mark.parametrize("num_frames", [1, 13, 3000])
    def test_shapes_the_frontends_actually_use(self, num_mel, num_frames):
        """The projection tiles frames; a frame count that is not a whole
        number of tiles must not read or write past the utterance."""
        torch.manual_seed(11)
        power = torch.rand(2, num_frames, 201, device="cuda") * 1e-3
        filters = torch.rand(201, num_mel, device="cuda") * 0.05
        got = whisper_logmel(power, filters)
        torch.testing.assert_close(got, _whisper_reference(power, filters), rtol=1e-5, atol=1e-5)

    def test_complex_spectrum_matches_the_power_it_stands_for(self):
        """The frontend hands over the transform, not ``abs().square()``."""
        torch.manual_seed(13)
        z = torch.fft.rfft(torch.randn(3, 257, 400, device="cuda"), n=400)
        filters = torch.rand(201, 80, device="cuda") * 0.05

        got = whisper_logmel(z, filters)
        torch.testing.assert_close(
            got, _whisper_reference(z.abs().square(), filters), rtol=1e-5, atol=1e-5
        )

    def test_rows_are_independent(self):
        """The max floor is per utterance. A loud row must not floor a quiet
        one -- the reduction is carried on an atomic into a per-row slot, and
        one slot for the batch would be invisible in any single-row test."""
        torch.manual_seed(17)
        filters = torch.rand(201, 80, device="cuda") * 0.05
        quiet = torch.rand(1, 400, 201, device="cuda") * 1e-9
        loud = torch.rand(1, 400, 201, device="cuda") * 1e3

        solo = whisper_logmel(quiet, filters)
        both = whisper_logmel(torch.cat([quiet, loud]), filters)
        torch.testing.assert_close(both[:1], solo, rtol=0, atol=0)

    def test_rejects_a_mel_table_in_the_wrong_orientation(self):
        power = torch.rand(1, 4, 33, device="cuda")
        with pytest.raises(ValueError, match=r"\(n_freq, num_mel\)"):
            whisper_logmel(power, torch.rand(8, 33, device="cuda"))


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
    @pytest.mark.parametrize(
        "window_type", ["povey", "hanning", "hamming", "blackman", "rectangular"]
    )
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
        torch.testing.assert_close(feats[0, : feat_lens[0]], ref.to("cuda"), rtol=1e-3, atol=1e-2)

    def test_batched(self):
        cfg = FeatureConfig(num_mel_bins=80)
        fb = Fbank(cfg).cuda()
        wavs = torch.stack([_make_sine(16000, 1.0), _make_sine(16000, 1.0, freq=440.0)], dim=0)
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

    def test_cpu_runs_the_torch_path(self):
        """CPU is out of the kernels' scope, not out of the layer's.

        ``Fbank`` owns both paths -- the kernel chain and the torch reference
        beside it -- so the engine's CPU branch and ``OASR_FEATURE_BACKEND=torch``
        reach the same module rather than a second implementation.
        """
        cfg = FeatureConfig(num_mel_bins=40)
        fb = Fbank(cfg)
        wav = _make_sine(16000, 1.0)
        cpu_feats, cpu_lens = fb(wav)
        ref = _kaldi_fbank(wav, cfg)
        torch.testing.assert_close(cpu_feats[0, : cpu_lens[0]], ref, rtol=1e-3, atol=1e-2)

        cuda_feats, _ = fb(wav.cuda())
        torch.testing.assert_close(cuda_feats.cpu(), cpu_feats, rtol=1e-3, atol=1e-2)


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
        torch.testing.assert_close(feats[0, : feat_lens[0]], ref.to("cuda"), rtol=1e-3, atol=1e-2)

    def test_output_shape_and_dtype(self):
        cfg = FeatureConfig(feature_type="mfcc", num_mel_bins=23, num_ceps=13)
        mf = Mfcc(cfg).cuda()
        wav = torch.randn(3, 16000, device="cuda")
        feats, feat_lens = mf(wav)
        assert feats.shape == (3, feat_lens[0].item(), 13)
        assert feats.dtype == torch.float32

    def test_no_lifter(self):
        cfg = FeatureConfig(feature_type="mfcc", num_mel_bins=23, num_ceps=13, cepstral_lifter=0.0)
        mf = Mfcc(cfg).cuda()
        # Just verify it runs and produces sensible output.
        wav = torch.randn(1, 16000, device="cuda")
        feats, _ = mf(wav)
        assert torch.isfinite(feats).all()


# ---------------------------------------------------------------------------
# The mel-energy floor -- a frontend *convention*, so only the external oracle
# can check it
# ---------------------------------------------------------------------------


#: Kaldi's mel-energy floor: ``std::numeric_limits<float>::epsilon()``.  Spelled
#: out here rather than imported, so these tests keep checking the *convention*
#: even if OASR's own constant moves.
_FLT_EPSILON = float(torch.finfo(torch.float32).eps)


def _quiet_lowpass(sample_rate: int = 16000, duration_s: float = 1.0) -> torch.Tensor:
    """A quiet low-frequency tone: real speech's high mel bins, reproducibly.

    Kaldi's floor only shows itself where a mel bin's energy lands under
    ``FLT_EPSILON``, which needs two things at once -- an ``audio_scale=1.0``
    convention (icefall, lhotse) and a band with no content.  A 200 Hz tone at
    amplitude 1e-3 has both, and unlike a real recording it is one line.
    """
    t = torch.arange(int(sample_rate * duration_s), dtype=torch.float32) / sample_rate
    return 1e-3 * torch.sin(2.0 * math.pi * 200.0 * t)


@CUDA
class TestKaldiMelFloor:
    """``log(max(mel, FLT_EPSILON))`` -- Kaldi's floor, not ``float32`` tiny.

    The two differ by 31 orders of magnitude, so a floored bin is ``-15.94``
    under Kaldi and ``-87.34`` under ``tiny``.  Nothing internal can catch the
    difference: both OASR paths would agree with each other, and a parity test
    feeds the same convention to both sides.  These compare against
    ``torchaudio.compliance.kaldi``, which is also what the per-utterance
    reference path inside OASR calls -- so the two in-tree paths that serve the
    *same* config are pinned to the same answer.
    """

    CFG = FeatureConfig(num_mel_bins=80, dither=0.0)

    def test_the_fixture_actually_reaches_the_floor(self):
        """Guard: without floored bins the tests below pass on any floor.

        Asked of the *oracle*, not of us — a guard phrased in terms of OASR's own
        floor constant moves with the bug it is guarding against.
        """
        ref = _kaldi_fbank(_quiet_lowpass(), self.CFG)
        floored = (ref - math.log(_FLT_EPSILON)).abs() < 1e-4
        assert floored.any(), "torchaudio floored no bin on this signal — bad fixture"

    @pytest.mark.parametrize("backend", ["oasr", "torch"])
    @pytest.mark.parametrize("feature_type", ["fbank", "mfcc"])
    def test_floored_bins_match_torchaudio(self, feature_type, backend, monkeypatch):
        from oasr.features.batched import batched_fbank, batched_mfcc

        monkeypatch.setenv("OASR_FEATURE_BACKEND", "torch" if backend == "torch" else "")
        cfg = FeatureConfig(feature_type=feature_type, num_mel_bins=80, dither=0.0)
        wav = _quiet_lowpass()
        extract = batched_mfcc if feature_type == "mfcc" else batched_fbank
        got, lens = extract(wav.unsqueeze(0).cuda(), torch.tensor([wav.numel()]).cuda(), cfg)
        ref = _kaldi_mfcc(wav, cfg) if feature_type == "mfcc" else _kaldi_fbank(wav, cfg)
        # ``tiny`` instead of ``FLT_EPSILON`` puts this at 71 (fbank) / 639 (mfcc).
        torch.testing.assert_close(got[0, : lens[0]].cpu(), ref.cpu(), rtol=1e-3, atol=1e-1)

    def test_digital_silence_is_the_reference_constant(self):
        """An all-zero row is the floor and nothing else -- the clearest case.

        ``log(FLT_EPSILON) = -15.9424``; under ``float32`` tiny it is ``-87.3365``,
        a 71.4 step on every bin of every silent frame.
        """
        from oasr.features.batched import batched_fbank

        wav = torch.zeros(1, 16000, device="cuda")
        got, lens = batched_fbank(wav, torch.tensor([16000], device="cuda"), self.CFG)
        ref = _kaldi_fbank(torch.zeros(16000), self.CFG)
        torch.testing.assert_close(got[0, : lens[0]].cpu(), ref)
        assert abs(float(got[0, 0, 0]) - math.log(_FLT_EPSILON)) < 1e-5


# ---------------------------------------------------------------------------
# CMVN -- the last pointwise step of the frontend
# ---------------------------------------------------------------------------


# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")


class TestCMVN:
    """Tests for oasr.cmvn() functional API."""

    #: ``num_cols`` is the only kernel-relevant axis -- it picks the vector
    #: width.  40 and 80 are the real ASR feature dims; 256 crosses into the
    #: wide path.  Batch and sequence are grid parallelism over a broadcast
    #: elementwise op.
    @pytest.mark.parametrize("num_cols", [40, 256])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_cmvn_correctness(self, num_cols, dtype):
        """``(x - mean) * istd``, against the torch expression it fuses."""
        x = torch.randn(4, 200, num_cols, device="cuda", dtype=dtype)
        mean = torch.randn(num_cols, device="cuda", dtype=dtype)
        istd = torch.randn(num_cols, device="cuda", dtype=dtype).abs() + 0.1

        output = oasr.cmvn(x, mean, istd)

        torch.testing.assert_close(output, (x - mean) * istd, **tol(dtype))

    def test_cmvn_destination_passing(self):
        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.float16)
        mean = torch.randn(256, device="cuda", dtype=torch.float16)
        istd = torch.randn(256, device="cuda", dtype=torch.float16).abs() + 0.1
        assert_dest_passing(
            oasr.cmvn, x, mean, istd, out=torch.empty_like(x), expected=(x - mean) * istd
        )

    def test_cmvn_2d_input(self):
        """Test CMVN with 2D input [m, n]."""
        m, n = 100, 80
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        mean = torch.randn(n, device="cuda", dtype=torch.float32)
        istd = torch.randn(n, device="cuda", dtype=torch.float32).abs() + 0.1

        output = oasr.cmvn(x, mean, istd)

        expected = (x - mean) * istd
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
