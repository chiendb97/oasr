# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer (FunASR) — non-autoregressive ASR, registered as ``"paraformer"``.

SANM encoder + CIF predictor + SANM NAR decoder.  Auto-detected from FunASR
model dirs (``config.yaml`` with ``model: Paraformer``); decoded offline by
the ``paraformer`` strategy in one parallel pass (no AR loop).
"""

from ..registry import register_model
from .config import ParaformerModelConfig
from .convert import FunASRParaformerConverter
from .decoder import ParaformerSANMDecoder
from .encoder import SANMEncoder
from .model import ParaformerModel
from .predictor import CifPredictor

register_model(
    "paraformer",
    model_cls=ParaformerModel,
    config_cls=ParaformerModelConfig,
    converter=FunASRParaformerConverter(),
)

__all__ = [
    "ParaformerModel",
    "ParaformerModelConfig",
    "SANMEncoder",
    "ParaformerSANMDecoder",
    "CifPredictor",
    "FunASRParaformerConverter",
]
