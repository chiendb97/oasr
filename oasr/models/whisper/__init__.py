# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Whisper (HF-format) AED model, registered as ``"whisper"``.

Offline-only: audio is padded/trimmed to the 30 s log-mel window
(``FeatureSpec(kind="whisper_logmel", audio_scale=1.0)``) and decoding runs
label-synchronously through the engine's incremental ``aed`` strategy —
bounded decoder steps per engine tick with continuous batching.  Streaming
requests are rejected at engine construction.
"""

from ..registry import register_model
from .config import WhisperModelConfig
from .convert import HFWhisperConverter
from .model import WhisperDecoder, WhisperEncoder, WhisperModel

register_model(
    "whisper",
    model_cls=WhisperModel,
    config_cls=WhisperModelConfig,
    converter=HFWhisperConverter(),
)

__all__ = [
    "WhisperModel",
    "WhisperModelConfig",
    "WhisperEncoder",
    "WhisperDecoder",
    "HFWhisperConverter",
]
