# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Speech-LLM (Qwen2-Audio-style) — LLM-based ASR, registered as ``"speech_llm"``.

Whisper-geometry audio tower + linear projector + Qwen2 causal LM.
Auto-detected from HF snapshots (``config.json`` with
``model_type: "qwen2_audio"``); decoded offline by the incremental ``llm``
strategy (bounded greedy generation with token-streaming partials).
"""

from ..registry import register_model
from .audio_tower import Qwen2AudioTower
from .config import SpeechLlmModelConfig
from .convert import HFQwen2AudioConverter
from .llm import Qwen2Lm
from .model import SpeechLlmModel

register_model(
    "speech_llm",
    model_cls=SpeechLlmModel,
    config_cls=SpeechLlmModelConfig,
    converter=HFQwen2AudioConverter(),
)

__all__ = [
    "SpeechLlmModel",
    "SpeechLlmModelConfig",
    "Qwen2AudioTower",
    "Qwen2Lm",
    "HFQwen2AudioConverter",
]
