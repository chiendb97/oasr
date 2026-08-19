# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline and streaming batched audio feature extraction."""

from __future__ import annotations

from .backends import extract_features_batch, fbank_batch, mfcc_batch
from .batched import batched_fbank, batched_mfcc
from .config import FeatureConfig
from .lfr import apply_lfr_batch
from .registry import (
    ExtractorSpec,
    StreamingFraming,
    build_extractor,
    list_extractors,
    register_extractor,
)
from .spec import FeatureSpec
from .streaming import BatchedStreamingFeatureExtractor
from .whisper import batched_whisper_logmel

__all__ = [
    "FeatureConfig",
    "FeatureSpec",
    "apply_lfr_batch",
    "ExtractorSpec",
    "StreamingFraming",
    "register_extractor",
    "build_extractor",
    "list_extractors",
    "batched_fbank",
    "batched_mfcc",
    "batched_whisper_logmel",
    "fbank_batch",
    "mfcc_batch",
    "extract_features_batch",
    "BatchedStreamingFeatureExtractor",
]
