# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) model: encoder + stateless predictor + joiner, registered
as ``"transducer"``.

The decode-side (predictor + joiner) is small pure-PyTorch; the engine's
:class:`~oasr.engine.decode.TransducerDecodeStrategy` drives it with
frame-synchronous greedy search over raw encoder hidden states (offline and
streaming).

Registered **without auto-detection**: icefall experiment dirs are claimed by
the Zipformer CTC converter and hybrid checkpoints carry both branches — load
with ``from_pretrained(dir, architecture="transducer")``.
"""

from ..registry import register_model
from .config import TransducerModelConfig
from .convert import IcefallTransducerConverter, load_icefall_transducer_checkpoint
from .decoder import StatelessDecoder
from .joiner import TransducerJoiner
from .model import TransducerModel

register_model(
    "transducer",
    model_cls=TransducerModel,
    config_cls=TransducerModelConfig,
    converter=IcefallTransducerConverter(),
)

__all__ = [
    "TransducerModel",
    "TransducerModelConfig",
    "StatelessDecoder",
    "TransducerJoiner",
    "IcefallTransducerConverter",
    "load_icefall_transducer_checkpoint",
]
