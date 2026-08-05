# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""NeMo/Nemotron log-mel spectrogram frontend (batched, GPU-friendly).

Reproduces ``NemotronAsrStreamingFeatureExtractor`` (which in turn reproduces
NeMo's ``AudioToMelSpectrogramPreprocessor`` for the Nemotron ASR releases):

* input waveforms at **[-1, 1] float scale** (``FeatureSpec.audio_scale = 1.0``
  — like the icefall/Whisper frontends and *unlike* WeNet's int16 scale);
* per-row **pre-emphasis** ``x[t] - 0.97 * x[t-1]`` with ``x[0]`` kept, applied
  before the STFT and re-zeroed past each row's valid length;
* STFT ``n_fft = 512`` (the power of two above ``win_length = 400``),
  ``hop = 160``, **non-periodic** Hann window, centered with *constant* (zero)
  padding — not reflect;
* power spectrum, slaney-scale mel filterbank (``fmin = 0``, ``fmax = sr/2``);
* ``log(mel + 2**-24)`` — a **log of the raw mel**, not ``log10`` and not
  normalized in any way.  Nemotron never applies per-feature CMVN, so unlike
  every other frontend in this package there is nothing to normalize over and
  the recipe is exactly frame-local.

Three consequences of that last point are worth naming, because they are what
make this its own extractor rather than a :class:`~oasr.features.FeatureConfig`
of the Kaldi one:

1. the log guard ``2**-24`` and the missing ``log10``/``dither``/``povey``
   window put it outside what ``oasr.features.batched`` can express;
2. being frame-local, it *could* stream — but the frame grid is defined by one
   ``center=True`` pass over the whole utterance, and reproducing it chunk by
   chunk needs ``center=False`` plus a ``n_fft // 2`` look-back on every chunk
   boundary.  Until that exists the extractor declares
   ``supports_streaming=False`` so the engine refuses a streaming request
   instead of silently shifting the frame grid;
3. the returned frame count is ``floor(L / hop)`` while the tensor is
   ``floor(L_max / hop) + 1`` frames wide.  That is not an off-by-one: it is
   HF's ``attention_mask`` convention, and the encoder's causal subsampling
   reads the mask, so a frontend that claimed the extra frame would feed the
   encoder one frame of zero-padding as if it were audio.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch

from .config import FeatureConfig

__all__ = ["batched_nemotron_logmel", "nemotron_stft_geometry"]

#: The log-zero guard NeMo uses (``2 ** -24``); reproduced exactly because it
#: sets the floor of every silent bin and therefore the encoder's input scale.
LOG_ZERO_GUARD = 2.0**-24


def nemotron_stft_geometry(cfg: FeatureConfig) -> Tuple[int, int, int]:
    """``(n_fft, hop_length, win_length)`` in samples for ``cfg``.

    ``n_fft`` is the power of two at or above ``win_length`` (NeMo's
    ``2 ** ceil(log2(win_length))``), so the released 25 ms / 10 ms geometry at
    16 kHz gives ``(512, 160, 400)``.
    """
    win_length = cfg.frame_length_samples
    hop_length = cfg.frame_shift_samples
    n_fft = 1 << max(0, (win_length - 1).bit_length())
    return n_fft, hop_length, win_length


@functools.lru_cache(maxsize=8)
def _hann_window(win_length: int, device_str: str) -> torch.Tensor:
    """fp32 **non-periodic** Hann window, cached per (length, device).

    ``periodic=False`` matters: it is a different window from the
    ``torch.stft`` default and the one NeMo trained with.
    """
    return torch.hann_window(
        win_length, periodic=False, device=torch.device(device_str), dtype=torch.float32
    )


@functools.lru_cache(maxsize=8)
def _mel_filters(sample_rate: int, n_fft: int, n_mels: int, device_str: str) -> torch.Tensor:
    """Slaney-normalized slaney-scale mel filterbank ``(n_mels, n_fft // 2 + 1)``.

    ``torchaudio.functional.melscale_fbanks(norm="slaney", mel_scale="slaney")``
    is the same table ``librosa.filters.mel`` builds — the one the HF feature
    extractor uses, and which it prefers over ``transformers``'s own
    ``mel_filter_bank`` for exactly this reason.  Verified against librosa at
    ``(16 kHz, 512, 128)``: max absolute difference 2.6e-7, which is fp32
    accumulation order, not a different filter.
    """
    from torchaudio.functional import melscale_fbanks

    fb = melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )  # (n_freqs, n_mels)
    table: torch.Tensor = fb.t().contiguous().to(torch.device(device_str))
    return table


def batched_nemotron_logmel(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Nemotron log-mel features.

    Parameters
    ----------
    waveforms : Tensor
        ``(B, T)`` float32 waveforms at **[-1, 1]** scale.
    lengths : Tensor
        ``(B,)`` valid sample counts.  Everything past a row's length is zeroed
        *before* pre-emphasis and the resulting frames are zeroed after the
        mel, so padding contributes neither energy nor a log floor.
    cfg : FeatureConfig
        ``feature_type="nemotron_logmel"``; reads ``sample_rate``,
        ``num_mel_bins``, ``frame_length_ms``, ``frame_shift_ms`` and
        ``preemphasis_coefficient``.

    Returns
    -------
    features : Tensor
        ``(B, floor(T / hop) + 1, num_mel_bins)`` float32.
    feat_lengths : Tensor
        ``(B,)`` int32 — ``floor(len / hop)`` per row (HF's attention-mask
        count, one less than the number of STFT frames a centered transform
        produces; see the module docstring).
    """
    assert waveforms.dim() == 2, "waveforms must be (B, T)"
    device = waveforms.device
    n_fft, hop, win_length = nemotron_stft_geometry(cfg)

    lengths = lengths.to(device=device)
    time_mask = torch.arange(waveforms.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
    wav = (waveforms * time_mask).float()

    coeff = cfg.preemphasis_coefficient
    if coeff:
        # Keep sample 0, then first-difference.  The re-mask matters: at index
        # ``len`` the difference is ``0 - coeff * x[len - 1]``, which is not zero.
        wav = torch.cat([wav[:, :1], wav[:, 1:] - coeff * wav[:, :-1]], dim=1)
        wav = wav * time_mask

    stft = torch.stft(
        wav,
        n_fft,
        hop_length=hop,
        win_length=win_length,
        window=_hann_window(win_length, str(device)),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    # ``sqrt(re^2 + im^2)`` then square, in that order, mirroring the HF
    # extractor.  ``abs() ** 2`` is the same value up to fp rounding; matching
    # the upstream op sequence keeps the parity test's tolerance about the mel
    # table rather than about this.
    parts = torch.view_as_real(stft)
    power = parts.pow(2).sum(-1).sqrt().pow(2)  # (B, n_freqs, n_frames)

    filters = _mel_filters(cfg.sample_rate, n_fft, cfg.num_mel_bins, str(device))
    mel = filters @ power  # (B, n_mels, n_frames)
    log_mel = torch.log(mel + LOG_ZERO_GUARD)

    features = log_mel.transpose(1, 2)  # (B, n_frames, n_mels)
    feat_lengths = torch.div(lengths, hop, rounding_mode="floor")
    feat_lengths = feat_lengths.clamp(max=features.size(1)).to(torch.int32)
    # Zero the frames past each row's count: the log of a silent bin is a large
    # negative constant, not zero, so leaving them would hand the encoder's
    # causal subsampling real-looking energy in the padding.
    frame_mask = torch.arange(features.size(1), device=device).unsqueeze(
        0
    ) < feat_lengths.unsqueeze(1)
    return (features * frame_mask.unsqueeze(-1)).contiguous(), feat_lengths
