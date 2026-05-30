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
)
from .registry import (
    ModelEntry,
    build_model_from_checkpoint,
    get_model_entry,
    list_models,
    register_model,
    resolve_architecture,
)
from .heads import CTCHead

# Importing the architecture packages triggers their register_model() calls.
from .conformer import (
    ConformerEncoder,
    ConformerEncoderConfig,
    ConformerEncoderLayer,
    ConformerModel,
    ConformerModelConfig,
    ConvolutionModule,
    PositionwiseFeedForward,
)
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
    # Registry / factory
    "ModelEntry",
    "build_model_from_checkpoint",
    "get_model_entry",
    "list_models",
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
]
