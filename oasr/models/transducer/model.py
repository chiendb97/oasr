# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) model: encoder + stateless predictor + joiner.

Engine integration: ``decode_type`` resolves to ``"transducer"`` (via the
predictor's class attribute), so the engine selects
:class:`~oasr.engine.decode.TransducerDecodeStrategy`.  That strategy consumes
raw encoder hidden states (``consumes="hidden"``) and drives this model's
``decoder`` (predictor) + ``joiner`` directly — frame-synchronous greedy
search — rather than the fused CTC head path.  Streaming follows the encoder:
a Conformer front-end streams through the paged-KV backend
(``encode_chunk_paged``), a Zipformer front-end through the stateful backend
(``encoder.streaming_forward``); both feed the strategy hidden states.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Tuple

import torch

from ..base import BaseAsrModel, BaseEncoder, LoadReport
from ..decoders.base import BaseDecoder, Joiner
from .config import TransducerModelConfig
from .decoder import StatelessDecoder
from .joiner import TransducerJoiner

logger = logging.getLogger(__name__)


class TransducerModel(BaseAsrModel):
    """Encoder + stateless predictor (``decoder``) + ``joiner``."""

    # Conformer front-ends rebuild the positional-encoding table from config;
    # harmless no-op for Zipformer front-ends (no matching keys).
    _computed_buffer_suffixes = ("pos_enc.pe",)

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
        if config.encoder_type == "zipformer":
            from ..zipformer.model import ZipformerEncoder

            encoder: BaseEncoder = ZipformerEncoder(config.encoder)
            enc_dim = encoder.output_size
        else:
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

    # -- stateful streaming passthrough (Zipformer-style front-ends) ---------
    def get_streaming_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: Optional[torch.dtype] = None,
    ) -> List[torch.Tensor]:
        """Initial encoder streaming state (``dtype`` defaults to param dtype)."""
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self.encoder.get_streaming_init_states(batch_size, device, dtype)

    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> LoadReport:
        """Map an icefall transducer state-dict into this model.

        Encoder keys follow the front-end: Zipformer front-ends remap icefall's
        ``encoder_embed.*`` / ``encoder.*`` to ``encoder.encoder_embed.*`` /
        ``encoder.encoder.*`` (same rule as :class:`ZipformerModel`); Conformer
        front-ends take ``encoder.*`` as-is.  The predictor (``decoder.*`` —
        icefall's stateless ``Decoder``: ``embedding`` / ``conv``) and the
        ``joiner.*`` (``encoder_proj`` / ``decoder_proj`` / ``output_linear``)
        map 1:1.  Everything else (``simple_*_proj`` pruned-RNNT training
        heads, a hybrid ``ctc_output.*`` branch) lands in
        ``LoadReport.dropped`` for the registry to account for.
        """
        zipformer_front = type(self.encoder).__name__ == "ZipformerEncoder"
        remapped = {}
        dropped = []
        for k, v in state_dict.items():
            if k.startswith("decoder.") or k.startswith("joiner."):
                remapped[k] = v
            elif zipformer_front and k.startswith("encoder_embed."):
                remapped["encoder.encoder_embed." + k[len("encoder_embed.") :]] = v
            elif zipformer_front and k.startswith("encoder."):
                remapped["encoder.encoder." + k[len("encoder.") :]] = v
            elif not zipformer_front and k.startswith("encoder."):
                remapped[k] = v
            else:
                dropped.append(k)

        missing, unexpected = self.load_state_dict(remapped, strict=strict)
        expected_missing_suffixes = self._computed_buffer_suffixes
        real_missing = [
            k for k in missing if not any(k.endswith(s) for s in expected_missing_suffixes)
        ]
        if real_missing:
            logger.warning("Missing keys when loading transducer weights: %s", real_missing)
        if unexpected:
            logger.warning("Unexpected keys when loading transducer weights: %s", unexpected)
        mapped = [k for k in remapped if k not in unexpected]
        return LoadReport(mapped=mapped, dropped=dropped, missing=real_missing)
