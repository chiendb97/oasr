# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Zipformer CTC model (icefall ``egs/librispeech/ASR/zipformer``), registered as ``"zipformer"``."""

from ..registry import register_model
from .config import ZipformerEncoderConfig, ZipformerModelConfig
from .convert import IcefallConverter, load_icefall_checkpoint
from .model import ZipformerEncoder, ZipformerModel

register_model(
    "zipformer",
    model_cls=ZipformerModel,
    config_cls=ZipformerModelConfig,
    converter=IcefallConverter(),
)

__all__ = [
    "ZipformerModel",
    "ZipformerEncoder",
    "ZipformerEncoderConfig",
    "ZipformerModelConfig",
    "IcefallConverter",
    "load_icefall_checkpoint",
]
