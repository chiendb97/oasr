# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Stateless transducer predictor (icefall ``Decoder``).

Conditions only on the last ``context_size`` labels (no recurrent state): an
embedding followed by a depthwise 1-D convolution over the label window.  Param
names (``embedding`` / ``conv``) mirror icefall ``egs/.../ASR/*/decoder.py`` so a
future converter can load icefall pruned-transducer checkpoints 1:1.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..decoders.base import BaseDecoder


class StatelessDecoder(BaseDecoder):
    """Label predictor: ``(B, context_size)`` token window → ``(B, decoder_dim)``.

    The transducer "decode state" is just the last ``context_size`` emitted
    labels (blank-filled at init); :class:`~oasr.engine.decode.TransducerDecodeStrategy`
    shifts a new label in after each emission.
    """

    decode_type = "transducer"

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int = 0,
        context_size: int = 2,
    ) -> None:
        super().__init__()
        assert context_size >= 1, context_size
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.blank_id = blank_id
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, decoder_dim, padding_idx=blank_id)
        if context_size > 1:
            # Depthwise conv over the label window (no padding: U == context_size
            # in → 1 frame out at decode time).
            self.conv = nn.Conv1d(
                decoder_dim,
                decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim,
                bias=False,
            )

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Blank-filled label window ``(batch_size, context_size)`` (int64)."""
        del dtype  # labels are always int64
        return torch.full(
            (batch_size, self.context_size), self.blank_id, dtype=torch.long, device=device
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """``(B, context_size)`` label window → ``(B, decoder_dim)`` prediction."""
        emb = self.embedding(y)  # (B, U, dim)
        if self.context_size > 1:
            emb = emb.permute(0, 2, 1)  # (B, dim, U)
            emb = self.conv(emb)  # (B, dim, U - context_size + 1) == (B, dim, 1)
            emb = emb.permute(0, 2, 1)  # (B, 1, dim)
        emb = F.relu(emb)
        # (B, U_out, dim) -> (B, dim): decode feeds exactly one window per step.
        return emb[:, -1, :]
