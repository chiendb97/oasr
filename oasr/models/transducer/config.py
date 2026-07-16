# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) model config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..base import BaseModelConfig, CacheSpec
from ..conformer.config import ConformerEncoderConfig


@dataclass
class TransducerModelConfig(BaseModelConfig):
    """Encoder + stateless-predictor + joiner transducer config.

    ``encoder_type`` selects the acoustic front-end: ``"conformer"`` (default,
    paged-KV streaming) or ``"zipformer"`` (icefall pruned-transducer
    checkpoints, stateful streaming); ``encoder`` holds the matching encoder
    config dataclass.
    """

    model_type: str = "transducer"
    encoder_type: str = "conformer"
    encoder: Any = field(default_factory=ConformerEncoderConfig)
    decoder_dim: int = 512
    joiner_dim: int = 512
    context_size: int = 2
    blank_id: int = 0

    @property
    def cache_spec(self) -> CacheSpec:
        return self.encoder.cache_spec

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransducerModelConfig":
        """Build from a dict (native-format ``oasr_config.json``)."""
        encoder_type = d.get("encoder_type", "conformer")
        encoder_dict = d.get("encoder", {})
        if encoder_type == "zipformer":
            from ..zipformer.config import ZipformerEncoderConfig

            fields = set(ZipformerEncoderConfig.__dataclass_fields__)
            encoder = ZipformerEncoderConfig(
                **{
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in encoder_dict.items()
                    if k in fields
                }
            )
        else:
            fields = set(ConformerEncoderConfig.__dataclass_fields__)
            encoder = ConformerEncoderConfig(
                **{k: v for k, v in encoder_dict.items() if k in fields}
            )
        known = ("vocab_size", "decoder_dim", "joiner_dim", "context_size", "blank_id")
        return cls(
            encoder_type=encoder_type,
            encoder=encoder,
            **{k: d[k] for k in known if k in d},
        )
