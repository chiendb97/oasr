# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""OASR models.

Layering (vLLM / SGLang style): reusable layers (:mod:`oasr.layers`) compose
into an encoder (:class:`BaseEncoder`) + head (:class:`BaseHead`) wrapped by a
model (:class:`BaseAsrModel`).  Architectures self-register in their package
``__init__`` and are loaded generically via :func:`build_model_from_checkpoint`.
"""

from .base import (
    BaseAsrModel,
    BaseEncoder,
    BaseHead,
    BaseModelConfig,
    CacheSpec,
    DecodeType,
    LoadReport,
)

# Importing the architecture packages triggers their register_model() calls.
# Every built-in package is imported here, and the same list drives
# ``registry._BUILTIN_PACKAGES``; ``tests/test_model_registry.py`` asserts the
# two agree, since they drifted apart once already (this file exported only
# conformer / transducer / zipformer while the registry knew all six).
from .conformer import (
    ConformerEncoder,
    ConformerEncoderConfig,
    ConformerEncoderLayer,
    ConformerModel,
    ConformerModelConfig,
    ConvolutionModule,
    PositionwiseFeedForward,
)
from .heads import CTCHead
from .loaders import PretrainedModel, from_pretrained, load_pretrained
from .paraformer import ParaformerModel, ParaformerModelConfig
from .registry import (
    ModelEntry,
    build_model_from_checkpoint,
    get_model_entry,
    instantiate_from_bundle,
    list_models,
    load_checkpoint_bundle,
    register_model,
    resolve_architecture,
)
from .speech_llm import SpeechLlmModel, SpeechLlmModelConfig
from .transducer import (
    TransducerModel,
    TransducerModelConfig,
)
from .whisper import WhisperModel, WhisperModelConfig
from .zipformer import (
    ZipformerEncoder,
    ZipformerEncoderConfig,
    ZipformerModel,
    ZipformerModelConfig,
)

__all__ = [
    # Base abstractions
    "BaseAsrModel",
    "BaseEncoder",
    "BaseHead",
    "BaseModelConfig",
    "CacheSpec",
    "DecodeType",
    "LoadReport",
    # Registry / factory
    "ModelEntry",
    "PretrainedModel",
    "build_model_from_checkpoint",
    "from_pretrained",
    "get_model_entry",
    "instantiate_from_bundle",
    "list_models",
    "load_checkpoint_bundle",
    "load_pretrained",
    "register_model",
    "resolve_architecture",
    # Heads
    "CTCHead",
    # Conformer
    "ConformerModel",
    "ConformerEncoder",
    "ConformerEncoderLayer",
    "ConformerEncoderConfig",
    "ConformerModelConfig",
    "ConvolutionModule",
    "PositionwiseFeedForward",
    # Zipformer
    "ZipformerModel",
    "ZipformerEncoder",
    "ZipformerEncoderConfig",
    "ZipformerModelConfig",
    # Transducer (RNNT)
    "TransducerModel",
    "TransducerModelConfig",
    # Whisper (AED)
    "WhisperModel",
    "WhisperModelConfig",
    # Paraformer (NAR)
    "ParaformerModel",
    "ParaformerModelConfig",
    # Speech-LLM (Qwen2-Audio)
    "SpeechLlmModel",
    "SpeechLlmModelConfig",
]
