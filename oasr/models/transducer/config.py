# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) model config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping

from ..base import BaseModelConfig, CacheSpec, coerce_config
from ..conformer.config import ConformerEncoderConfig


@dataclass
class TransducerModelConfig(BaseModelConfig):
    """Encoder + stateless-predictor + joiner transducer config.

    ``encoder_type`` selects the acoustic front-end: ``"conformer"`` (default,
    paged-KV streaming) or ``"zipformer"`` (icefall pruned-transducer
    checkpoints, stateful streaming); ``encoder`` holds the matching encoder
    config dataclass.

    ``decoder_conv_group_size`` is the predictor conv's **input channels per
    group** — icefall's stateless decoder is
    ``nn.Conv1d(C, C, context_size, groups=C // group_size)``, and the group
    size is a *recipe* decision, not a layout one, so it has to be carried:

    * ``1`` — fully depthwise (``groups == decoder_dim``), the old
      ``pruned_transducer_stateless2/3/5`` recipes.  The default, so every
      existing config and native checkpoint keeps its meaning.
    * ``4`` — every Zipformer recipe (``zipformer``,
      ``pruned_transducer_stateless7``), i.e. what a real icefall release ships.

    The converter infers it from ``decoder.conv.weight``, so a checkpoint never
    depends on the default being right.
    """

    model_type: str = "transducer"
    encoder_type: str = "conformer"
    encoder: Any = field(default_factory=ConformerEncoderConfig)
    decoder_dim: int = 512
    joiner_dim: int = 512
    context_size: int = 2
    decoder_conv_group_size: int = 1
    blank_id: int = 0

    @property
    def cache_spec(self) -> CacheSpec:
        return self.encoder.cache_spec

    # ``encoder: Any`` is genuinely polymorphic — its class is decided by the sibling
    # ``encoder_type`` key, which no annotation can express — so it is the one field
    # here that needs a hook.  The flat scalars used to be listed in a hardcoded
    # ``known`` tuple, a fourth spelling of "filter to known fields" that silently
    # dropped any field added after it was written; they now come from
    # ``__dataclass_fields__`` like everywhere else.
    _from_dict_overrides: ClassVar[Mapping[str, Any]] = {
        "encoder": lambda d: _encoder_config_from_dict(
            d.get("encoder_type", "conformer"), d.get("encoder", {})
        ),
    }


def _encoder_config_from_dict(encoder_type: str, d: Dict[str, Any]):
    """Build the acoustic front-end config named by ``encoder_type``."""
    if encoder_type == "zipformer":
        from ..zipformer.config import ZipformerEncoderConfig

        return coerce_config(ZipformerEncoderConfig, d)
    return coerce_config(ConformerEncoderConfig, d)
