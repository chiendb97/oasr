# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron ASR (FastConformer + RNN-T), registered under ``"nemotron"``."""

from ..registry import register_model
from .config import NemotronEncoderConfig, NemotronModelConfig
from .convert import HFNemotronConverter
from .encoder import (
    NemotronEncoder,
    NemotronEncoderLayer,
    chunked_limited_mask,
    rel_shift,
    relative_position_embedding,
)
from .model import NemotronModel
from .predictor import NemotronPromptProjector, NemotronRnntJoint, NemotronRnntPredictor
from .subsampling import NemotronSubsampling

register_model(
    "nemotron",
    model_cls=NemotronModel,
    config_cls=NemotronModelConfig,
    converter=HFNemotronConverter(),
)

__all__ = [
    "HFNemotronConverter",
    "NemotronEncoder",
    "NemotronEncoderConfig",
    "NemotronEncoderLayer",
    "NemotronModel",
    "NemotronModelConfig",
    "NemotronPromptProjector",
    "NemotronRnntJoint",
    "NemotronRnntPredictor",
    "NemotronSubsampling",
    "chunked_limited_mask",
    "rel_shift",
    "relative_position_embedding",
]
