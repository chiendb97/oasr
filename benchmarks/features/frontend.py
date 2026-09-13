# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Feature frontend -- ``oasr/functionals/feature.py`` and ``oasr/features/``.

Four stages measured on their own (``fbank_preprocess``, ``mel_log``,
``dct_lifter``, ``whisper_logmel``) and the three whole pipelines they compose
into.  The Kaldi pipelines carry a ``torchaudio`` arm as a speed reference: it
computes the same feature through a different FFT order, so its output is
compared and reported but never gates the run.

Every row has a ``torch`` arm on purpose.  These kernels replace an expression
a caller could have written in torch, so "is it faster than what it replaced"
is the only question that decides whether the kernel should be on the path at
all -- and a row that is missing cannot answer it.  ``whisper_logmel`` shipped
without one and spent its first weeks 4-8x slower than the three-line torch
recipe it displaced.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from typing import Any, Callable, Dict

import torch

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size

SUBROUTINES = [
    "fbank_preprocess",
    "mel_log",
    "dct_lifter",
    "whisper_logmel",
    "lfr_gather",
    "fbank_pipeline",
    "mfcc_pipeline",
    "whisper_pipeline",
]

#: The frontend kernels are single-precision only.
FORCE_DTYPE = "float32"

#: Compared and reported, never gating -- see the module docstring.
NON_GATING_BACKENDS = frozenset({"torchaudio"})

#: A pipeline accumulates rounding across four stages, so it is checked at the
#: tolerance the composition warrants rather than the one a single kernel gets.
TOLERANCES = {
    "fbank_pipeline": (5e-2, 5e-2),
    "mfcc_pipeline": (5e-2, 5e-2),
    # Whisper's output is a normalized [-1, 1] band, so the pipeline's spread
    # over a different FFT order stays much tighter than the Kaldi rows'.
    "whisper_pipeline": (1e-3, 1e-3),
}

# Per-kernel configs shape the intermediate tensor directly; pipeline configs
# shape the input waveform via (batch, audio_seconds).  16 kHz Kaldi-style.
DEFAULT_CONFIGS: Dict[str, list] = {
    "fbank_preprocess": [
        {"batch": 32, "num_frames": 250, "frame_length": 400, "n_fft": 512},
        {"batch": 64, "num_frames": 500, "frame_length": 400, "n_fft": 512},
        {"batch": 64, "num_frames": 500, "frame_length": 800, "n_fft": 1024},
    ],
    # The 30 s window is fixed, so the only free dimension is concurrency:
    # 3000 frames and 201 bins per request whatever the audio is.  80 mels is
    # whisper-tiny .. large-v2, 128 is large-v3 and Qwen2-Audio.
    "whisper_logmel": [
        {"batch": 1, "num_frames": 3000, "n_freq": 201, "num_mel": 128},
        {"batch": 8, "num_frames": 3000, "n_freq": 201, "num_mel": 80},
        {"batch": 32, "num_frames": 3000, "n_freq": 201, "num_mel": 128},
    ],
    # Paraformer's frontend: 80 mel, LFR 7/6 -> 560-dim at a 60 ms hop.
    "lfr_gather": [
        {"batch": 1, "num_frames": 1000, "feature_dim": 80, "lfr_m": 7, "lfr_n": 6},
        {"batch": 8, "num_frames": 1000, "feature_dim": 80, "lfr_m": 7, "lfr_n": 6},
        {"batch": 32, "num_frames": 3000, "feature_dim": 80, "lfr_m": 7, "lfr_n": 6},
    ],
    "whisper_pipeline": [
        {"batch": 1, "audio_seconds": 30.0},
        {"batch": 8, "audio_seconds": 30.0},
        {"batch": 32, "audio_seconds": 30.0},
    ],
    "mel_log": [
        {"batch": 32, "num_frames": 250, "n_freq": 257, "num_mel": 80},
        {"batch": 64, "num_frames": 500, "n_freq": 257, "num_mel": 80},
        {"batch": 64, "num_frames": 500, "n_freq": 513, "num_mel": 80},
    ],
    "dct_lifter": [
        {"batch": 32, "num_frames": 250, "num_mel": 23, "num_ceps": 13},
        {"batch": 64, "num_frames": 500, "num_mel": 23, "num_ceps": 13},
        {"batch": 64, "num_frames": 500, "num_mel": 80, "num_ceps": 40},
    ],
    "fbank_pipeline": [
        {"batch": 8, "audio_seconds": 5.0},
        {"batch": 32, "audio_seconds": 5.0},
        {"batch": 64, "audio_seconds": 5.0},
        {"batch": 32, "audio_seconds": 10.0},
    ],
    "mfcc_pipeline": [
        {"batch": 8, "audio_seconds": 5.0},
        {"batch": 32, "audio_seconds": 5.0},
        {"batch": 64, "audio_seconds": 5.0},
        {"batch": 32, "audio_seconds": 10.0},
    ],
}


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


@lru_cache(maxsize=8)
def _povey_window(frame_length: int, device_str: str) -> torch.Tensor:
    """Povey window: ``(0.5 - 0.5*cos(2π i/(N-1)))**0.85``."""
    device = torch.device(device_str)
    i = torch.arange(frame_length, device=device, dtype=torch.float32)
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * i / (frame_length - 1))
    return w.pow(0.85)


@lru_cache(maxsize=8)
def _mel_bank(
    num_mel: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    device_str: str,
) -> torch.Tensor:
    """Kaldi-style triangular mel filterbank, shape ``(num_mel, n_fft//2+1)``."""
    device = torch.device(device_str)
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq = nyquist + high_freq
    num_bins_fft = n_fft // 2 + 1

    def to_mel(f: torch.Tensor) -> torch.Tensor:
        return 1127.0 * torch.log(1.0 + f / 700.0)

    mel_low = to_mel(torch.tensor(low_freq))
    mel_high = to_mel(torch.tensor(high_freq))
    mel_edges = torch.linspace(mel_low.item(), mel_high.item(), num_mel + 2)
    bin_hz = torch.arange(num_bins_fft, dtype=torch.float32) * (sample_rate / n_fft)
    bin_mel = to_mel(bin_hz)

    left = mel_edges[:-2].unsqueeze(1)
    center = mel_edges[1:-1].unsqueeze(1)
    right = mel_edges[2:].unsqueeze(1)
    bin_mel = bin_mel.unsqueeze(0)
    up = (bin_mel - left) / (center - left)
    down = (right - bin_mel) / (right - center)
    return torch.clamp(torch.minimum(up, down), min=0.0).to(device=device, dtype=torch.float32)


@lru_cache(maxsize=8)
def _dct_matrix(num_ceps: int, num_mel: int, device_str: str) -> torch.Tensor:
    """Kaldi-style orthonormal DCT-II matrix, shape ``(num_ceps, num_mel)``."""
    device = torch.device(device_str)
    n = torch.arange(num_mel, dtype=torch.float32) + 0.5  # (num_mel,)
    k = torch.arange(num_ceps, dtype=torch.float32).unsqueeze(1)  # (num_ceps, 1)
    mat = torch.cos(math.pi / num_mel * k * n) * math.sqrt(2.0 / num_mel)
    mat[0] *= 1.0 / math.sqrt(2.0)
    return mat.to(device=device, dtype=torch.float32)


@lru_cache(maxsize=8)
def _cepstral_lifter(num_ceps: int, lifter: float, device_str: str) -> torch.Tensor:
    """Kaldi cepstral lifter weights, shape ``(num_ceps,)``."""
    device = torch.device(device_str)
    if lifter <= 0:
        return torch.ones(num_ceps, device=device, dtype=torch.float32)
    n = torch.arange(num_ceps, dtype=torch.float32, device=device)
    return 1.0 + 0.5 * lifter * torch.sin(math.pi * n / lifter)


# ---------------------------------------------------------------------------
# Setup: per-kernel benchmarks
# ---------------------------------------------------------------------------


def setup_fbank_preprocess(batch, num_frames, frame_length, n_fft, dtype=torch.float32):
    device = "cuda"
    frames = torch.randn(batch, num_frames, frame_length, device=device, dtype=dtype)
    window = _povey_window(frame_length, device)
    coef = 0.97

    def oasr_fn():
        return oasr.functionals.feature.fbank_preprocess(
            frames, window, n_fft=n_fft, preemph_coef=coef
        )

    def torch_fn():
        # Equivalent vectorized batched op chain matching the OASR kernel.
        f = frames - frames.mean(dim=-1, keepdim=True)
        preem = torch.empty_like(f)
        preem[..., 1:] = f[..., 1:] - coef * f[..., :-1]
        preem[..., 0] = f[..., 0] - coef * f[..., 0]
        windowed = preem * window
        if n_fft > frame_length:
            return torch.nn.functional.pad(windowed, (0, n_fft - frame_length))
        return windowed.contiguous()

    return oasr_fn, torch_fn


def setup_mel_log(batch, num_frames, n_freq, num_mel, dtype=torch.float32):
    device = "cuda"
    power = torch.rand(batch, num_frames, n_freq, device=device, dtype=dtype) + 1e-6
    n_fft = (n_freq - 1) * 2
    mel_mat = _mel_bank(num_mel, n_fft, 16000, 20.0, 0.0, device)
    eps = torch.finfo(dtype).eps  # Kaldi's FLT_EPSILON floor, as the frontend uses

    def oasr_fn():
        return oasr.functionals.feature.mel_log(power, mel_mat, log_floor=eps)

    def torch_fn():
        return torch.matmul(power, mel_mat.t()).clamp_min(eps).log()

    return oasr_fn, torch_fn


def setup_lfr_gather(batch, num_frames, feature_dim, lfr_m, lfr_n, dtype=torch.float32):
    """The kernel against the ``gather`` + mask the torch path still runs.

    Rows are deliberately ragged: the gather clamps each row's source index to
    its own valid length, and a uniform batch would hide the per-row work.
    """
    device = "cuda"
    feats = torch.randn(batch, num_frames, feature_dim, device=device, dtype=dtype)
    lengths = torch.randint(
        num_frames // 2, num_frames + 1, (batch,), device=device, dtype=torch.int32
    )
    lengths[0] = num_frames
    lengths_long = lengths.to(torch.long)
    t_out = (num_frames + lfr_n - 1) // lfr_n
    out_lengths = (lengths_long + lfr_n - 1) // lfr_n
    left = (lfr_m - 1) // 2

    def oasr_fn():
        return oasr.lfr_gather(feats, lengths, lfr_m, lfr_n, t_out)

    def torch_fn():
        base = (
            torch.arange(t_out, device=device).unsqueeze(1) * lfr_n
            + torch.arange(lfr_m, device=device).unsqueeze(0)
            - left
        )
        idx = base.reshape(1, -1).expand(batch, -1)
        idx = idx.clamp(min=0).minimum((lengths_long - 1).clamp(min=0).unsqueeze(1))
        gathered = torch.gather(
            feats, 1, idx.unsqueeze(-1).expand(batch, t_out * lfr_m, feature_dim)
        )
        out = gathered.reshape(batch, t_out, lfr_m * feature_dim)
        valid = torch.arange(t_out, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)
        return out.mul_(valid.unsqueeze(-1).to(out.dtype))

    return oasr_fn, torch_fn


def setup_whisper_logmel(batch, num_frames, n_freq, num_mel, dtype=torch.float32):
    """The kernel against the torch expression ``oasr/features/whisper.py`` ran.

    The input is the *complex* transform, which is what the frontend hands over:
    the torch arm pays ``abs().square()`` because that is the recipe, and the
    kernel folds |z|^2 into the load the projection already makes.
    """
    device = "cuda"
    n_fft = (n_freq - 1) * 2
    frames = torch.randn(batch, num_frames, n_fft, device=device, dtype=dtype)
    spectrum = torch.fft.rfft(frames, n=n_fft)
    mel_mat = _mel_bank(num_mel, n_fft, 16000, 0.0, 0.0, device).t().contiguous()

    def oasr_fn():
        return oasr.whisper_logmel(spectrum, mel_mat)

    def torch_fn():
        log_spec = torch.matmul(spectrum.abs().square(), mel_mat).clamp_min(1e-10).log10()
        row_max = log_spec.amax(dim=(1, 2), keepdim=True)
        return (torch.maximum(log_spec, row_max - 8.0) + 4.0) * 0.25

    return oasr_fn, torch_fn


def setup_dct_lifter(batch, num_frames, num_mel, num_ceps, dtype=torch.float32):
    device = "cuda"
    log_mel = torch.randn(batch, num_frames, num_mel, device=device, dtype=dtype)
    dct = _dct_matrix(num_ceps, num_mel, device)
    lifter = _cepstral_lifter(num_ceps, 22.0, device)

    def oasr_fn():
        return oasr.functionals.feature.dct_lifter(log_mel, dct, lifter=lifter)

    def torch_fn():
        return torch.matmul(log_mel, dct.t()) * lifter

    return oasr_fn, torch_fn


# ---------------------------------------------------------------------------
# Setup: end-to-end pipelines
# ---------------------------------------------------------------------------


def _oasr_fbank_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The shipped chain: stft_frame → rfft_power → mel_log.

    What ``oasr.layers.Fbank`` and every served fbank request run.  Three
    launches and no framed-waveform copy: the framing is inside ``stft_frame``.
    """
    lengths = torch.full(
        (waveforms.size(0),), waveforms.size(1), dtype=torch.int32, device=waveforms.device
    )
    num_frames = (waveforms.size(1) - frame_length) // frame_shift + 1
    frames = oasr.stft_frame(
        waveforms,
        lengths,
        window,
        n_fft,
        frame_shift,
        num_frames,
        center_offset=0,
        win_offset=0,
        preemph_coef=0.97,
        preemph_replicate=True,
        remove_dc_offset=True,
    )
    power = oasr.rfft_power(frames, n=n_fft)
    return oasr.functionals.feature.mel_log(power, mel_mat, log_floor=eps)


def _unfold_fbank_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The pre-framed chain: unfold → fbank_preprocess → rfft_power → mel_log.

    Kept as its own arm so the cost of framing *outside* the kernel — an
    ``unfold`` plus a ``(B, N, frame_length)`` contiguous copy, 2.5x the
    waveform at Kaldi's 25 ms / 10 ms grid — is measured rather than asserted.
    ``oasr.layers.Fbank`` ran this until the four kernels were wired through
    ``stft_frame``.
    """
    frames = waveforms.unfold(-1, frame_length, frame_shift).contiguous()
    preprocessed = oasr.functionals.feature.fbank_preprocess(frames, window, n_fft=n_fft)
    power = oasr.rfft_power(preprocessed)
    return oasr.functionals.feature.mel_log(power, mel_mat, log_floor=eps)


def _torch_fbank_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    preemph: float,
    eps: float,
) -> torch.Tensor:
    """Pure torch batched pipeline (matches batched.batched_fbank)."""
    frames = waveforms.unfold(-1, frame_length, frame_shift)
    f = frames - frames.mean(dim=-1, keepdim=True)
    preem = torch.empty_like(f)
    preem[..., 1:] = f[..., 1:] - preemph * f[..., :-1]
    preem[..., 0] = f[..., 0] - preemph * f[..., 0]
    windowed = preem * window
    if n_fft > frame_length:
        windowed = torch.nn.functional.pad(windowed, (0, n_fft - frame_length))
    spec = torch.fft.rfft(windowed, n=n_fft)
    power = spec.real.pow(2) + spec.imag.pow(2)
    mel = torch.matmul(power, mel_mat.t())
    return torch.log(mel.clamp_min(eps))


def _torchaudio_fbank_loop(waveforms: torch.Tensor, sample_rate: int, num_mel: int) -> torch.Tensor:
    """Per-utterance torchaudio.compliance.kaldi.fbank loop (the legacy default)."""
    import torchaudio

    out = []
    for i in range(waveforms.size(0)):
        out.append(
            torchaudio.compliance.kaldi.fbank(
                waveforms[i : i + 1],
                sample_frequency=float(sample_rate),
                num_mel_bins=num_mel,
                frame_length=25.0,
                frame_shift=10.0,
                dither=0.0,
                energy_floor=0.0,
                preemphasis_coefficient=0.97,
                window_type="povey",
                low_freq=20.0,
                high_freq=0.0,
                snip_edges=True,
            )
        )
    return torch.stack(out, dim=0)


def setup_whisper_pipeline(batch, audio_seconds, dtype=torch.float32):
    """The served Whisper frontend, kernels against ``OASR_FEATURE_BACKEND=torch``.

    Both arms go through ``batched_whisper_logmel``, so this measures what a
    request pays -- framing, transform, projection, log and the per-utterance
    floor -- and not a stage in isolation.
    """
    import os

    from oasr.features import FeatureConfig
    from oasr.features.whisper import batched_whisper_logmel

    cfg = FeatureConfig(feature_type="whisper_logmel", num_mel_bins=128)
    samples = int(audio_seconds * 16000)
    waveforms = (torch.randn(batch, samples, device="cuda", dtype=dtype) * 0.1).contiguous()
    lengths = torch.full((batch,), samples, dtype=torch.int32, device="cuda")

    def oasr_fn():
        os.environ.pop("OASR_FEATURE_BACKEND", None)
        return batched_whisper_logmel(waveforms, lengths, cfg)[0]

    def torch_fn():
        os.environ["OASR_FEATURE_BACKEND"] = "torch"
        try:
            return batched_whisper_logmel(waveforms, lengths, cfg)[0]
        finally:
            os.environ.pop("OASR_FEATURE_BACKEND", None)

    return oasr_fn, torch_fn


def setup_fbank_pipeline(batch, audio_seconds, dtype=torch.float32):
    device = "cuda"
    sample_rate = 16000
    frame_length = 400
    frame_shift = 160
    n_fft = _next_power_of_two(frame_length)
    num_mel = 80
    preemph = 0.97
    eps = torch.finfo(dtype).eps  # Kaldi's FLT_EPSILON floor, as the frontend uses

    n_samples = int(audio_seconds * sample_rate)
    waveforms = torch.randn(batch, n_samples, device=device, dtype=dtype)
    window = _povey_window(frame_length, device)
    mel_mat = _mel_bank(num_mel, n_fft, sample_rate, 20.0, 0.0, device)

    def oasr_fn():
        return _oasr_fbank_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            eps=eps,
        )

    def torch_fn():
        return _torch_fbank_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            preemph=preemph,
            eps=eps,
        )

    def torchaudio_fn():
        return _torchaudio_fbank_loop(waveforms, sample_rate, num_mel)

    def unfold_fn():
        return _unfold_fbank_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            eps=eps,
        )

    return oasr_fn, torch_fn, torchaudio_fn, unfold_fn


def _oasr_mfcc_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    dct: torch.Tensor,
    lifter: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    log_mel = _oasr_fbank_pipeline(
        waveforms,
        frame_length=frame_length,
        frame_shift=frame_shift,
        n_fft=n_fft,
        window=window,
        mel_mat=mel_mat,
        eps=eps,
    )
    return oasr.functionals.feature.dct_lifter(log_mel, dct, lifter=lifter)


def _torch_mfcc_pipeline(
    waveforms: torch.Tensor,
    *,
    frame_length: int,
    frame_shift: int,
    n_fft: int,
    window: torch.Tensor,
    mel_mat: torch.Tensor,
    dct: torch.Tensor,
    lifter: torch.Tensor,
    preemph: float,
    eps: float,
) -> torch.Tensor:
    log_mel = _torch_fbank_pipeline(
        waveforms,
        frame_length=frame_length,
        frame_shift=frame_shift,
        n_fft=n_fft,
        window=window,
        mel_mat=mel_mat,
        preemph=preemph,
        eps=eps,
    )
    return torch.matmul(log_mel, dct.t()) * lifter


def _torchaudio_mfcc_loop(
    waveforms: torch.Tensor, sample_rate: int, num_mel: int, num_ceps: int
) -> torch.Tensor:
    import torchaudio

    out = []
    for i in range(waveforms.size(0)):
        out.append(
            torchaudio.compliance.kaldi.mfcc(
                waveforms[i : i + 1],
                sample_frequency=float(sample_rate),
                num_mel_bins=num_mel,
                num_ceps=num_ceps,
                cepstral_lifter=22.0,
                frame_length=25.0,
                frame_shift=10.0,
                dither=0.0,
                energy_floor=0.0,
                preemphasis_coefficient=0.97,
                window_type="povey",
                low_freq=20.0,
                high_freq=0.0,
                snip_edges=True,
            )
        )
    return torch.stack(out, dim=0)


def setup_mfcc_pipeline(batch, audio_seconds, dtype=torch.float32):
    device = "cuda"
    sample_rate = 16000
    frame_length = 400
    frame_shift = 160
    n_fft = _next_power_of_two(frame_length)
    num_mel = 23
    num_ceps = 13
    preemph = 0.97
    eps = torch.finfo(dtype).eps  # Kaldi's FLT_EPSILON floor, as the frontend uses

    n_samples = int(audio_seconds * sample_rate)
    waveforms = torch.randn(batch, n_samples, device=device, dtype=dtype)
    window = _povey_window(frame_length, device)
    mel_mat = _mel_bank(num_mel, n_fft, sample_rate, 20.0, 0.0, device)
    dct = _dct_matrix(num_ceps, num_mel, device)
    lifter = _cepstral_lifter(num_ceps, 22.0, device)

    def oasr_fn():
        return _oasr_mfcc_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            dct=dct,
            lifter=lifter,
            eps=eps,
        )

    def torch_fn():
        return _torch_mfcc_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            dct=dct,
            lifter=lifter,
            preemph=preemph,
            eps=eps,
        )

    def torchaudio_fn():
        return _torchaudio_mfcc_loop(waveforms, sample_rate, num_mel, num_ceps)

    def unfold_fn():
        log_mel = _unfold_fbank_pipeline(
            waveforms,
            frame_length=frame_length,
            frame_shift=frame_shift,
            n_fft=n_fft,
            window=window,
            mel_mat=mel_mat,
            eps=eps,
        )
        return oasr.functionals.feature.dct_lifter(log_mel, dct, lifter=lifter)

    return oasr_fn, torch_fn, torchaudio_fn, unfold_fn


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

KERNEL_SETUP = {
    "fbank_preprocess": setup_fbank_preprocess,
    "mel_log": setup_mel_log,
    "dct_lifter": setup_dct_lifter,
    "whisper_logmel": setup_whisper_logmel,
    "lfr_gather": setup_lfr_gather,
    # Two arms, like the kernel rows: the Whisper frontend's torch path *is*
    # the reference implementation, reached through the same entry point with
    # OASR_FEATURE_BACKEND=torch, so there is no third library to compare.
    "whisper_pipeline": setup_whisper_pipeline,
}

PIPELINE_SETUP = {
    "fbank_pipeline": setup_fbank_pipeline,
    "mfcc_pipeline": setup_mfcc_pipeline,
}


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--num-frames", type=int, default=None, help="Frames per utterance")
    parser.add_argument(
        "--frame-length", type=int, default=None, help="Frame length in samples (fbank_preprocess)"
    )
    parser.add_argument("--n-fft", type=int, default=None, help="FFT length")
    parser.add_argument(
        "--n-freq", type=int, default=None, help="Frequency bins = n_fft/2+1 (mel_log)"
    )
    parser.add_argument("--num-mel", type=int, default=None, help="Mel bins")
    parser.add_argument("--num-ceps", type=int, default=None, help="Cepstral coefficients")
    parser.add_argument("--feature-dim", type=int, default=None, help="Input feature dim (LFR)")
    parser.add_argument("--lfr-m", type=int, default=None, help="Frames stacked per LFR frame")
    parser.add_argument("--lfr-n", type=int, default=None, help="LFR hop in input frames")
    parser.add_argument(
        "--audio-seconds", type=float, default=None, help="Audio duration per utterance (pipelines)"
    )


#: The CLI dimensions that fully determine each subroutine's config.
_CONFIG_KEYS = {
    "fbank_preprocess": ("batch", "num_frames", "frame_length", "n_fft"),
    "mel_log": ("batch", "num_frames", "n_freq", "num_mel"),
    "dct_lifter": ("batch", "num_frames", "num_mel", "num_ceps"),
    "whisper_logmel": ("batch", "num_frames", "n_freq", "num_mel"),
    "lfr_gather": ("batch", "num_frames", "feature_dim", "lfr_m", "lfr_n"),
    "whisper_pipeline": ("batch", "audio_seconds"),
    "fbank_pipeline": ("batch", "audio_seconds"),
    "mfcc_pipeline": ("batch", "audio_seconds"),
}


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    keys = _CONFIG_KEYS[subroutine]
    values = {k: getattr(args, k, None) for k in keys}
    if all(v is not None for v in values.values()):
        return [values]
    return DEFAULT_CONFIGS[subroutine]


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if subroutine in PIPELINE_SETUP:
        oasr_fn, torch_fn, torchaudio_fn, unfold_fn = PIPELINE_SETUP[subroutine](**cfg, dtype=dtype)
        return {
            "cuda": oasr_fn,
            "torch": torch_fn,
            "torchaudio": torchaudio_fn,
            "cuda_unfold": unfold_fn,
        }
    oasr_fn, torch_fn = KERNEL_SETUP[subroutine](**cfg, dtype=dtype)[:2]
    return {"cuda": oasr_fn, "torch": torch_fn}


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    elem = dtype_size(dtype)
    b = cfg["batch"]
    if subroutine == "fbank_preprocess":
        n, length, n_fft = cfg["num_frames"], cfg["frame_length"], cfg["n_fft"]
        shape = f"[B={b}, N={n}, L={length}, n_fft={n_fft}]"
        nbytes = b * n * (length + n_fft) * elem
    elif subroutine == "mel_log":
        n, n_freq, num_mel = cfg["num_frames"], cfg["n_freq"], cfg["num_mel"]
        shape = f"[B={b}, N={n}, n_freq={n_freq}, num_mel={num_mel}]"
        nbytes = b * n * (n_freq + num_mel) * elem + n_freq * num_mel * elem
    elif subroutine == "dct_lifter":
        n, num_mel, num_ceps = cfg["num_frames"], cfg["num_mel"], cfg["num_ceps"]
        shape = f"[B={b}, N={n}, num_mel={num_mel}, num_ceps={num_ceps}]"
        nbytes = b * n * (num_mel + num_ceps) * elem + num_mel * num_ceps * elem
    elif subroutine == "whisper_logmel":
        n, n_freq, num_mel = cfg["num_frames"], cfg["n_freq"], cfg["num_mel"]
        shape = f"[B={b}, N={n}, n_freq={n_freq}, num_mel={num_mel}]"
        # Complex spectrum in, log-mel out, and the normalization pass reads
        # and rewrites that output once more.
        nbytes = b * n * (2 * n_freq + 3 * num_mel) * elem + n_freq * num_mel * elem
    elif subroutine == "lfr_gather":
        n, fd, m, lfr_n = cfg["num_frames"], cfg["feature_dim"], cfg["lfr_m"], cfg["lfr_n"]
        t_out = (n + lfr_n - 1) // lfr_n
        shape = f"[B={b}, T={n}, F={fd}, lfr={m}/{lfr_n}]"
        # A gather: every output element is one element read and one written.
        nbytes = 2 * b * t_out * fd * m * elem
    elif subroutine == "whisper_pipeline":
        seconds = cfg["audio_seconds"]
        shape = f"[B={b}, {seconds}s]"
        samples = int(seconds * 16000)
        nbytes = b * (samples + (samples // 160) * 128) * elem
    else:
        seconds = cfg["audio_seconds"]
        shape = f"[B={b}, {seconds}s]"
        # Waveform in, 80 mel bins out at the 10 ms frame rate.
        samples = int(seconds * 16000)
        frames = samples // 160
        nbytes = b * (samples + frames * 80) * elem
    return Work(shape=shape, params=params_of(cfg), bytes=nbytes)
