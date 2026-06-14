# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer joiner (icefall ``Joiner``).

Combines an encoder frame and a predictor output into vocab logits:
``output_linear(tanh(encoder_proj(enc) + decoder_proj(dec)))``.  ``project_input``
lets the greedy decoder pre-project the whole encoder output once and add the
per-step predictor projection (the icefall fast path).  Param names mirror
icefall so checkpoints load 1:1.
"""

from __future__ import annotations

import torch
from torch import nn

from ..decoders.base import Joiner


class TransducerJoiner(Joiner):
    def __init__(
        self, encoder_dim: int, decoder_dim: int, joiner_dim: int, vocab_size: int
    ) -> None:
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joiner_dim)
        self.decoder_proj = nn.Linear(decoder_dim, joiner_dim)
        self.output_linear = nn.Linear(joiner_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """``enc`` ⊕ ``dec`` → vocab logits.

        With ``project_input=True`` (default) ``encoder_out`` / ``decoder_out``
        are raw and projected here; with ``False`` they are already in joiner
        space and only summed (used by the greedy fast path).
        """
        if project_input:
            x = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            x = encoder_out + decoder_out
        return self.output_linear(torch.tanh(x))
