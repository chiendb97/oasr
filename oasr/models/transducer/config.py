# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) model config."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..base import BaseModelConfig, CacheSpec
from ..conformer.config import ConformerEncoderConfig


@dataclass
class TransducerModelConfig(BaseModelConfig):
    """Encoder + stateless-predictor + joiner transducer config.

    The default encoder is a Conformer (the common transducer acoustic model);
    swap ``encoder`` for another encoder config to build a different acoustic
    front-end.
    """

    model_type: str = "transducer"
    encoder: ConformerEncoderConfig = field(default_factory=ConformerEncoderConfig)
    decoder_dim: int = 512
    joiner_dim: int = 512
    context_size: int = 2
    blank_id: int = 0

    @property
    def cache_spec(self) -> CacheSpec:
        return self.encoder.cache_spec
