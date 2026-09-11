# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Batched FBANK and MFCC — the engine's entry into the Kaldi frontend.

The pipeline itself lives in :mod:`oasr.layers.feature`, on the four feature
kernels (``stft_frame`` → ``rfft_power`` → ``mel_log`` → ``dct_lifter``) with a
torch path beside them.  This module owns only the question the engine asks
first: *does the fused path reproduce this config exactly?*  When it does not,
:func:`oasr.features.extractors.kaldi_extract` falls to the per-utterance
reference, so an unsupported knob is served correctly and slowly rather than
approximated.
"""

from __future__ import annotations

from typing import Tuple

import torch

from oasr.layers.feature import WINDOW_TYPES, kaldi_fbank, kaldi_mfcc

from .config import FeatureConfig

__all__ = [
    "batched_fbank",
    "batched_mfcc",
    "supports_batched_fbank",
    "supports_batched_mfcc",
]


def _supports_common(cfg: FeatureConfig) -> bool:
    """Knobs the fused pipeline reproduces exactly, for either feature type.

    ``dither`` is refused rather than approximated because it is noise the
    reference draws per sample; ``snip_edges=False`` is a different frame grid,
    not a different constant; and ``use_energy`` appends a term the chain does
    not compute.  Every window Kaldi defines *is* supported — the table is built
    host-side once per config, so there is nothing to gain by narrowing it.
    """
    return (
        cfg.backend == "torchaudio"
        and cfg.window_type in WINDOW_TYPES
        and cfg.dither == 0.0
        and cfg.snip_edges is True
        and cfg.use_energy is False
    )


def supports_batched_fbank(cfg: FeatureConfig) -> bool:
    """Return ``True`` if :func:`batched_fbank` matches ``cfg`` exactly."""
    return cfg.feature_type == "fbank" and _supports_common(cfg)


def supports_batched_mfcc(cfg: FeatureConfig) -> bool:
    """Return ``True`` if :func:`batched_mfcc` matches ``cfg`` exactly."""
    return cfg.feature_type == "mfcc" and _supports_common(cfg)


def batched_fbank(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully batched Kaldi-compliant FBANK.

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
    return kaldi_fbank(waveforms, lengths, cfg)


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
    return kaldi_mfcc(waveforms, lengths, cfg)
