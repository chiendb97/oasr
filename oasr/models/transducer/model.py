# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) model: encoder + stateless predictor + joiner.

Engine integration: ``decode_type`` resolves to ``"transducer"`` (via the
predictor's class attribute), so the engine selects
:class:`~oasr.engine.decode.TransducerDecodeStrategy`.  That strategy consumes
raw encoder hidden states (``ModelRunner.encode_offline``) and drives this
model's ``decoder`` (predictor) + ``joiner`` directly — frame-synchronous greedy
search — rather than the fused CTC head path.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from ..base import BaseAsrModel, BaseEncoder
from ..decoders.base import BaseDecoder, Joiner
from .config import TransducerModelConfig
from .decoder import StatelessDecoder
from .joiner import TransducerJoiner


class TransducerModel(BaseAsrModel):
    """Encoder + stateless predictor (``decoder``) + ``joiner``."""

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        joiner: Joiner,
        *,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder  # stateless predictor; carries decode_type
        self.joiner = joiner
        self._blank_id = blank_id

    @property
    def blank_id(self) -> int:
        return getattr(self.decoder, "blank_id", self._blank_id)

    # head is unused for transducers; expose a None-safe alias so generic code
    # that probes ``model.head`` doesn't crash.
    @property
    def head(self):  # type: ignore[override]
        return None

    @classmethod
    def from_config(
        cls, config: TransducerModelConfig, global_cmvn: Optional[Any] = None, **aux: Any
    ) -> "TransducerModel":
        from ..conformer.model import ConformerEncoder

        encoder = ConformerEncoder(config.encoder, global_cmvn)
        enc_dim = config.encoder.output_size
        vocab = config.vocab_size
        assert vocab is not None, "TransducerModelConfig.vocab_size must be set"
        decoder = StatelessDecoder(
            vocab, config.decoder_dim, blank_id=config.blank_id, context_size=config.context_size
        )
        joiner = TransducerJoiner(enc_dim, config.decoder_dim, config.joiner_dim, vocab)
        return cls(encoder, decoder, joiner, blank_id=config.blank_id)

    def load_weights(self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False) -> None:
        """Load ``encoder.*`` / ``decoder.*`` / ``joiner.*`` weights.

        Placeholder loader (no transducer checkpoint is validated yet): a real
        ``CheckpointConverter`` (e.g. an icefall pruned-transducer converter)
        would handle vocab padding + the pruned ``simple_*_proj`` aux heads.
        """
        missing, unexpected = self.load_state_dict(dict(state_dict), strict=strict)
        if missing or unexpected:
            import logging

            logging.getLogger(__name__).warning(
                "Transducer load_weights: missing=%s unexpected=%s", missing, unexpected
            )
