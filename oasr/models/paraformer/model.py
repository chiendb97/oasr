# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer: non-autoregressive ASR (SANM encoder + CIF predictor + NAR decoder).

Offline-only (``streaming_kind == "none"``).  The engine drives it through the
``paraformer`` decode strategy (``consumes="hidden"``): one
:meth:`encode_offline` pass, then :meth:`predict` (CIF → per-token acoustic
embeddings + token counts + fire positions) and :meth:`nar_decode` (one
parallel decoder pass → ``(B, U, V)`` log-probs) — no autoregressive loop, so
the one-shot offline executor path fits unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

import torch

from oasr.models.base import BaseAsrModel, LoadReport

from .config import ParaformerModelConfig
from .decoder import ParaformerSANMDecoder
from .encoder import SANMEncoder
from .predictor import CifPredictor

logger = logging.getLogger(__name__)


class ParaformerModel(BaseAsrModel):
    """FunASR Paraformer for OASR (offline NAR decoding)."""

    def __init__(self, config: ParaformerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SANMEncoder(config)
        self.predictor = CifPredictor(config)
        self.decoder = ParaformerSANMDecoder(config)

    @classmethod
    def from_config(cls, config: ParaformerModelConfig, **aux: Any) -> "ParaformerModel":
        del aux
        return cls(config)

    # -- engine-facing metadata ---------------------------------------------
    @property
    def default_decode_type(self) -> str:
        return "paraformer"

    @property
    def capabilities(self) -> frozenset:
        return frozenset({"paraformer"})

    # -- weights --------------------------------------------------------------
    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> LoadReport:
        """FunASR keys map 1:1 onto this module tree (``encoder.encoders0.*``,
        ``encoder.encoders.*``, ``decoder.decoders.*``, ``predictor.*``); the
        converter additionally injects ``encoder.cmvn_shift`` /
        ``encoder.cmvn_scale`` parsed from ``am.mvn``."""
        sd = {}
        dropped = []
        for k, v in state_dict.items():
            if k.startswith(("encoder.", "decoder.", "predictor.")):
                sd[k] = v
            else:
                dropped.append(k)
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        if unexpected:
            logger.warning("Unexpected keys in Paraformer checkpoint: %s", unexpected[:8])
            dropped.extend(unexpected)
        if missing:
            logger.warning("Paraformer model keys not filled: %s", missing[:8])
        mapped = [k for k in sd if k not in set(unexpected)]
        return LoadReport(mapped=mapped, dropped=dropped, missing=list(missing))

    # -- SupportsParaformer compute contract ----------------------------------
    def predict(
        self, enc_out: torch.Tensor, enc_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CIF → ``(acoustic_embeds (B, U, D) fp32, token_lens (B,) int32,
        fires (B, T+1) fp32)``."""
        T = enc_out.size(1)
        mask = (
            torch.arange(T, device=enc_out.device).unsqueeze(0)
            < enc_lens.to(enc_out.device).unsqueeze(1)
        ).float()
        acoustic_embeds, token_num, _alphas, fires = self.predictor(enc_out, mask)
        token_lens = token_num.round().long().clamp(min=0).to(torch.int32)
        return acoustic_embeds, token_lens, fires

    def nar_decode(
        self,
        enc_out: torch.Tensor,
        enc_lens: torch.Tensor,
        acoustic_embeds: torch.Tensor,
        token_lens: torch.Tensor,
    ) -> torch.Tensor:
        """One parallel decoder pass → ``(B, U, V)`` log-probs."""
        log_probs, _ = self.decoder(
            enc_out, enc_lens, acoustic_embeds.to(enc_out.dtype), token_lens
        )
        return log_probs
