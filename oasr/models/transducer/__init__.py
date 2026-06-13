# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) model: encoder + stateless predictor + joiner.

The decode-side (predictor + joiner) is small pure-PyTorch; the engine's
:class:`~oasr.engine.decode.TransducerDecodeStrategy` drives it with
frame-synchronous greedy search over raw encoder hidden states.

Not auto-registered yet: a ``CheckpointConverter`` (e.g. for icefall
pruned-transducer checkpoints) is the remaining piece to wire ``from_pretrained``.
The model is fully constructible via ``TransducerModel(...)`` /
``TransducerModel.from_config(...)``.
"""

from .config import TransducerModelConfig
from .decoder import StatelessDecoder
from .joiner import TransducerJoiner
from .model import TransducerModel

__all__ = [
    "TransducerModel",
    "TransducerModelConfig",
    "StatelessDecoder",
    "TransducerJoiner",
]
