# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Batched audio feature extraction (FBANK / MFCC).

Supports **offline batch** extraction (padded ``(B, T)`` or list of waveforms)
and **batched chunk streaming** (``B`` parallel streams via
:class:`BatchedStreamingFeatureExtractor`).

Backends: **torchaudio** (default, ``torchaudio.compliance.kaldi``) or
**kaldifeat** (optional GPU path; ``pip install kaldifeat``).

Offline batch::

    feats, feat_lens = oasr.fbank_batch(waveforms, num_mel_bins=80)
    # feats: (B, max_frames, 80), feat_lens: (B,)

With a shared :class:`FeatureConfig`::

    from oasr.features import FeatureConfig, extract_features_batch

    cfg = FeatureConfig(feature_type="fbank", num_mel_bins=80, dither=0.0)
    feats, feat_lens = extract_features_batch(wavs, cfg)

Batched parallel streams::

    from oasr.features import FeatureConfig, BatchedStreamingFeatureExtractor

    ext = BatchedStreamingFeatureExtractor(
        FeatureConfig(num_mel_bins=80, dither=0.0), batch_size=B)
    for chunk_batch in chunks:
        feats, feat_lens = ext.process_chunk(chunk_batch)
    ext.flush()
"""

from __future__ import annotations

from .backends import extract_features_batch, fbank_batch, mfcc_batch
from .config import FeatureConfig
from .streaming import BatchedStreamingFeatureExtractor

__all__ = [
    "FeatureConfig",
    "fbank_batch",
    "mfcc_batch",
    "extract_features_batch",
    "BatchedStreamingFeatureExtractor",
]
