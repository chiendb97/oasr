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

from oasr.layers import ColumnParallelLinear, Linear, Tanh
from oasr.models.base import align_out_features

from ..decoders.base import Joiner


class TransducerJoiner(Joiner):
    def __init__(
        self, encoder_dim: int, decoder_dim: int, joiner_dim: int, vocab_size: int
    ) -> None:
        super().__init__()
        self.encoder_proj = ColumnParallelLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ColumnParallelLinear(decoder_dim, joiner_dim)
        # Vocabulary out-features are rarely 8-aligned (500 for the icefall BPE
        # releases) and the GEMM kernels cannot address an unaligned width at
        # all, so the head is allocated aligned and
        # ``TransducerModel.load_weights`` widens the checkpoint to match.
        self.vocab_size = vocab_size
        self.output_linear = Linear(joiner_dim, align_out_features(vocab_size))
        self.tanh = Tanh()

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
        # Slice off the alignment padding so the logit width is the true
        # vocabulary; the padding rows carry a hugely negative bias anyway, so
        # this is presentation, not masking.
        return self.output_linear(self.tanh(x))[..., : self.vocab_size]
