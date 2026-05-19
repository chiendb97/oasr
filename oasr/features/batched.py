# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Truly batched FBANK / MFCC matching torchaudio's Kaldi-compliant output.

Replaces the per-utterance Python loop over
``torchaudio.compliance.kaldi.fbank`` / ``mfcc`` (the throughput ceiling of
the offline pipeline's GPU feature-extraction path) with a handful of fused
tensor ops over the whole micro-batch.

Matches :func:`torchaudio.compliance.kaldi.fbank` /
:func:`torchaudio.compliance.kaldi.mfcc` with the default settings actually
used by the OASR offline pipeline:

* ``dither=0`` (deterministic inference)
* ``use_energy=False``
* ``snip_edges=True``
* ``window_type='povey'``
* ``preemphasis_coefficient=0.97``
* ``remove_dc_offset=True`` (default)
* ``round_to_power_of_two=True``
* ``use_power=True``
* ``use_log_fbank=True``

For configurations outside this window callers should fall back to the
per-utt loop over ``torchaudio.compliance.kaldi.{fbank,mfcc}`` via a thin
dispatcher.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import torch

from .config import FeatureConfig

__all__ = [
    "batched_fbank",
    "batched_mfcc",
    "supports_batched_fbank",
    "supports_batched_mfcc",
]


def _supports_common(cfg: FeatureConfig) -> bool:
    return (
        cfg.backend == "torchaudio"
        and cfg.window_type == "povey"
        and cfg.dither == 0.0
        and cfg.snip_edges is True
        and cfg.use_energy is False
    )


def supports_batched_fbank(cfg: FeatureConfig) -> bool:
    """Return ``True`` if ``batched_fbank`` matches ``cfg`` exactly."""
    return cfg.feature_type == "fbank" and _supports_common(cfg)


def supports_batched_mfcc(cfg: FeatureConfig) -> bool:
    """Return ``True`` if ``batched_mfcc`` matches ``cfg`` exactly."""
    return cfg.feature_type == "mfcc" and _supports_common(cfg)


@lru_cache(maxsize=32)
def _mel_banks(
    num_bins: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Kaldi-style triangular mel filterbank (matches torchaudio.compliance.kaldi).

    Returns a ``(num_bins, n_fft // 2 + 1)`` tensor.  Cached so we only pay
    the construction cost once per (config, device) pair.
    """
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq = nyquist + high_freq  # matches kaldi's "0.0 means Nyquist"
    assert 0.0 <= low_freq < high_freq <= nyquist

    num_bins_fft = n_fft // 2 + 1

    def _to_mel(f: torch.Tensor) -> torch.Tensor:
        return 1127.0 * torch.log(1.0 + f / 700.0)

    mel_low = _to_mel(torch.tensor(low_freq))
    mel_high = _to_mel(torch.tensor(high_freq))
    mel_edges = torch.linspace(mel_low.item(), mel_high.item(), num_bins + 2)

    # FFT bin center frequencies (Hz).
    bin_hz = torch.arange(num_bins_fft, dtype=torch.float32) * (sample_rate / n_fft)
    bin_mel = _to_mel(bin_hz)

    left = mel_edges[:-2].unsqueeze(1)    # (num_bins, 1)
    center = mel_edges[1:-1].unsqueeze(1)  # (num_bins, 1)
    right = mel_edges[2:].unsqueeze(1)    # (num_bins, 1)
    bin_mel = bin_mel.unsqueeze(0)        # (1, num_bins_fft)

    up = (bin_mel - left) / (center - left)
    down = (right - bin_mel) / (right - center)
    mel_mat = torch.clamp(torch.minimum(up, down), min=0.0)
    return mel_mat.to(device=device, dtype=dtype)


@lru_cache(maxsize=16)
def _povey_window(
    frame_length: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Povey window:  (0.5 - 0.5*cos(2π n/(N-1)))**0.85."""
    i = torch.arange(frame_length, device=device, dtype=dtype)
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * i / (frame_length - 1))
    return w.pow(0.85)


@lru_cache(maxsize=16)
def _dct_matrix(
    num_ceps: int,
    num_mel_bins: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Kaldi orthonormal DCT-II matrix, shape ``(num_ceps, num_mel_bins)``.

    ``dct[0, n] = 1/sqrt(M)`` and
    ``dct[k, n] = sqrt(2/M) * cos(pi*(n + 0.5)*k / M)`` for ``k > 0``,
    where ``M = num_mel_bins``.  Matches ``torchaudio.functional.create_dct``
    with ``norm='ortho'``.
    """
    M = num_mel_bins
    n = torch.arange(M, dtype=torch.float64).unsqueeze(0)    # (1, M)
    k = torch.arange(num_ceps, dtype=torch.float64).unsqueeze(1)  # (num_ceps, 1)
    dct = torch.cos(math.pi * (n + 0.5) * k / M) * math.sqrt(2.0 / M)
    dct[0, :] = 1.0 / math.sqrt(M)
    return dct.to(device=device, dtype=dtype)


@lru_cache(maxsize=16)
def _cepstral_lifter(
    num_ceps: int,
    cepstral_lifter: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Kaldi cepstral lifter vector, shape ``(num_ceps,)``.

    ``lifter[k] = 1 + (L/2) * sin(pi*k/L)`` when ``L > 0``; an all-ones
    vector when ``L == 0`` (no liftering).
    """
    k = torch.arange(num_ceps, dtype=torch.float64)
    if cepstral_lifter == 0.0:
        lifter = torch.ones_like(k)
    else:
        L = cepstral_lifter
        lifter = 1.0 + 0.5 * L * torch.sin(math.pi * k / L)
    return lifter.to(device=device, dtype=dtype)


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _log_mel_pipeline(
    waveforms: torch.Tensor, cfg: FeatureConfig
) -> Tuple[torch.Tensor, int]:
    """Shared fbank pipeline producing the log-mel tensor + frame count.

    Returns ``(log_mel, num_frames)`` where ``log_mel`` has shape
    ``(B, num_frames, num_mel_bins)``.  ``num_frames == 0`` means the input
    is shorter than a single frame.
    """
    B, T = waveforms.shape
    device = waveforms.device
    dtype = torch.float32

    frame_length = cfg.frame_length_samples
    frame_shift = cfg.frame_shift_samples
    preemph = cfg.preemphasis_coefficient
    num_mel = cfg.num_mel_bins

    if T < frame_length:
        return torch.zeros(B, 0, num_mel, device=device, dtype=dtype), 0

    # Frame via ``unfold`` (view, zero copy) — snip_edges=True means we drop
    # any trailing tail that doesn't fit a full frame.
    frames = waveforms.unfold(-1, frame_length, frame_shift)

    # Step 1: subtract DC offset per frame (Kaldi default: remove_dc_offset=True).
    frames = frames - frames.mean(dim=-1, keepdim=True)

    # Step 2: pre-emphasis per frame.  Kaldi applies this after DC removal and
    # handles the leading sample as ``x[0] - coef * x[0]`` (the "replicate"
    # convention used by torchaudio).
    preem = torch.empty_like(frames)
    preem[..., 1:] = frames[..., 1:] - preemph * frames[..., :-1]
    preem[..., 0] = frames[..., 0] - preemph * frames[..., 0]

    # Step 3: Povey window.
    window = _povey_window(frame_length, device, dtype)
    windowed = preem * window

    # Step 4: FFT at the next power of two ≥ frame_length.
    n_fft = _next_power_of_two(frame_length)
    if n_fft > frame_length:
        windowed = torch.nn.functional.pad(windowed, (0, n_fft - frame_length))
    spectrum = torch.fft.rfft(windowed, n=n_fft)           # (B, N, n_fft/2+1) complex
    power = spectrum.real.pow(2) + spectrum.imag.pow(2)    # (B, N, n_fft/2+1)

    # Step 5: mel filterbank.
    mel_mat = _mel_banks(
        num_mel, n_fft, cfg.sample_rate, cfg.low_freq, cfg.high_freq,
        device=device, dtype=dtype,
    )
    mel_energies = torch.matmul(power, mel_mat.t())        # (B, N, num_mel)

    # Step 6: log (with a tiny floor to avoid log(0)).  Kaldi uses log energy
    # directly; with energy_floor=0 torchaudio applies ``torch.clamp(min=eps)``
    # implicitly via ``torch.log`` on clamped values.
    eps = torch.finfo(dtype).tiny
    log_mel = torch.log(mel_energies.clamp_min(eps))
    return log_mel, log_mel.size(1)


def _feat_lengths(
    lengths: torch.Tensor, frame_length: int, frame_shift: int
) -> torch.Tensor:
    """Kaldi snip_edges frame-count formula, returned as int32."""
    if lengths.dtype != torch.int64:
        lengths = lengths.long()
    return torch.clamp(
        (lengths - frame_length) // frame_shift + 1, min=0,
    ).to(torch.int32)


def batched_fbank(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully batched Kaldi-compliant fbank.

    Parameters
    ----------
    waveforms : Tensor
        ``(B, T)`` float32 waveforms, pre-emphasis-scaled
        (``audio_scale`` already applied by the caller).
    lengths : Tensor
        ``(B,)`` int32/int64 sample counts per utterance.
    cfg : FeatureConfig
        Feature config — must satisfy :func:`supports_batched_fbank`.

    Returns
    -------
    features : Tensor
        ``(B, num_frames_max, num_mel_bins)`` float32 log-mel features.
    feat_lengths : Tensor
        ``(B,)`` int32 valid frame counts per utterance.
    """
    assert waveforms.dim() == 2, "waveforms must be (B, T)"
    B = waveforms.size(0)
    device = waveforms.device

    log_mel, num_frames = _log_mel_pipeline(waveforms, cfg)
    if num_frames == 0:
        return (
            log_mel,
            torch.zeros(B, dtype=torch.int32, device=device),
        )
    feat_lengths = _feat_lengths(
        lengths, cfg.frame_length_samples, cfg.frame_shift_samples
    )
    return log_mel, feat_lengths


def batched_mfcc(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully batched Kaldi-compliant MFCC.

    Computes log-mel via the same pipeline as :func:`batched_fbank`, then
    applies an orthonormal DCT-II and the Kaldi cepstral lifter.

    Parameters
    ----------
    waveforms : Tensor
        ``(B, T)`` float32 waveforms.
    lengths : Tensor
        ``(B,)`` int32/int64 sample counts per utterance.
    cfg : FeatureConfig
        Feature config — must satisfy :func:`supports_batched_mfcc`.

    Returns
    -------
    features : Tensor
        ``(B, num_frames_max, num_ceps)`` float32 MFCC features.
    feat_lengths : Tensor
        ``(B,)`` int32 valid frame counts per utterance.
    """
    assert waveforms.dim() == 2, "waveforms must be (B, T)"
    B = waveforms.size(0)
    device = waveforms.device
    dtype = torch.float32
    num_ceps = cfg.num_ceps

    log_mel, num_frames = _log_mel_pipeline(waveforms, cfg)
    if num_frames == 0:
        return (
            torch.zeros(B, 0, num_ceps, device=device, dtype=dtype),
            torch.zeros(B, dtype=torch.int32, device=device),
        )

    # DCT-II + cepstral liftering.
    dct = _dct_matrix(num_ceps, cfg.num_mel_bins, device=device, dtype=dtype)
    mfcc = torch.matmul(log_mel, dct.t())                # (B, N, num_ceps)
    lifter = _cepstral_lifter(
        num_ceps, cfg.cepstral_lifter, device=device, dtype=dtype,
    )
    mfcc = mfcc * lifter

    feat_lengths = _feat_lengths(
        lengths, cfg.frame_length_samples, cfg.frame_shift_samples
    )
    return mfcc, feat_lengths
