# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Whisper log-mel spectrogram frontend (batched, GPU-friendly).

Reproduces ``whisper.audio.log_mel_spectrogram`` + ``pad_or_trim``:

* input waveforms at **[-1, 1] float scale** (``FeatureSpec.audio_scale = 1.0``
  — unlike the Kaldi int16-scale frontends);
* pad/trim every utterance to 30 s (480 000 samples @ 16 kHz);
* STFT ``n_fft=400, hop=160``, Hann window, centered/reflect, magnitude²,
  last frame dropped → 3000 frames;
* slaney-scale mel filterbank (matches librosa's default, which Whisper uses);
* ``log10`` clamped at 1e-10, floored at ``max - 8``, then ``(x + 4) / 4``.

The global max-normalization couples frames within one utterance but not
across utterances, so the batch dimension is safe.  Output frame counts are
uniform (all rows = 3000 frames), which is what makes the Whisper encoder's
fixed 1500-position geometry work.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch

from .config import FeatureConfig

__all__ = ["batched_whisper_logmel"]

_N_FFT = 400
_HOP = 160


@functools.lru_cache(maxsize=8)
def _mel_filters(sample_rate: int, n_mels: int, device_str: str) -> torch.Tensor:
    """Slaney-normalized slaney-scale mel filterbank ``(n_mels, n_fft//2 + 1)``.

    ``torchaudio.functional.melscale_fbanks(..., norm="slaney",
    mel_scale="slaney")`` is numerically identical to
    ``librosa.filters.mel(...)`` — the table Whisper ships in its assets.
    """
    from torchaudio.functional import melscale_fbanks

    fb = melscale_fbanks(
        n_freqs=_N_FFT // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )  # (n_freqs, n_mels)
    return fb.t().contiguous().to(torch.device(device_str))


def batched_whisper_logmel(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Whisper log-mel features.

    Parameters
    ----------
    waveforms : Tensor
        ``(B, T)`` float32 waveforms at **[-1, 1]** scale.
    lengths : Tensor
        ``(B,)`` valid sample counts (samples past each length are zeroed
        before the pad/trim so padding never leaks energy).
    cfg : FeatureConfig
        ``feature_type="whisper_logmel"``; reads ``sample_rate``,
        ``num_mel_bins``, ``whisper_chunk_seconds``.

    Returns
    -------
    features : Tensor
        ``(B, n_frames, num_mel_bins)`` float32 — ``n_frames`` = 3000 for the
        standard 30 s window.
    feat_lengths : Tensor
        ``(B,)`` int32, all equal to ``n_frames`` (Whisper consumes the padded
        30 s window as real input by design).
    """
    assert waveforms.dim() == 2, "waveforms must be (B, T)"
    B, T = waveforms.shape
    device = waveforms.device
    n_samples = int(cfg.sample_rate * cfg.whisper_chunk_seconds)

    # Zero anything past each row's valid length, then pad/trim to 30 s.
    idx = torch.arange(T, device=device).unsqueeze(0)
    wav = waveforms * (idx < lengths.to(device).unsqueeze(1))
    if T < n_samples:
        wav = torch.nn.functional.pad(wav, (0, n_samples - T))
    elif T > n_samples:
        wav = wav[:, :n_samples]

    window = torch.hann_window(_N_FFT, device=device, dtype=wav.dtype)
    stft = torch.stft(
        wav.float(), _N_FFT, _HOP, window=window.float(), center=True, return_complex=True
    )
    magnitudes = stft[..., :-1].abs() ** 2  # (B, n_freqs, n_frames)

    filters = _mel_filters(cfg.sample_rate, cfg.num_mel_bins, str(device))
    mel = filters @ magnitudes  # (B, n_mels, n_frames)

    log_spec = torch.clamp(mel, min=1e-10).log10()
    # Per-utterance max floor (amax over mel+time of each row, not the batch).
    row_max = log_spec.amax(dim=(1, 2), keepdim=True)
    log_spec = torch.maximum(log_spec, row_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    features = log_spec.transpose(1, 2).contiguous()  # (B, n_frames, n_mels)
    n_frames = features.size(1)
    return features, torch.full((B,), n_frames, dtype=torch.int32, device=device)
