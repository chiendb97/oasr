# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer model (encoder + CTC head), registered under ``"conformer"``."""

from ..registry import register_model
from .config import ConformerEncoderConfig, ConformerModelConfig
from .convert import WenetConverter, load_wenet_checkpoint
from .model import (
    CTC,
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerModel,
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    ConvolutionModule,
    GlobalCMVN,
    PositionwiseFeedForward,
)

register_model(
    "conformer",
    model_cls=ConformerModel,
    config_cls=ConformerModelConfig,
    converter=WenetConverter(),
)

__all__ = [
    "ConformerModel",
    "ConformerEncoder",
    "ConformerEncoderLayer",
    "ConformerEncoderConfig",
    "ConformerModelConfig",
    "Conv2dSubsampling",
    "Conv2dSubsampling2",
    "Conv2dSubsampling4",
    "Conv2dSubsampling6",
    "Conv2dSubsampling8",
    "ConvolutionModule",
    "GlobalCMVN",
    "PositionwiseFeedForward",
    "CTC",
    "WenetConverter",
    "load_wenet_checkpoint",
]
