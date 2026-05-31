# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CTC output head."""

from __future__ import annotations

import torch

import oasr
from oasr.layers.linear import Linear

from ..base import BaseHead


class CTCHead(BaseHead):
    """CTC head: fused (Linear -> log_softmax) via ``oasr.gemm_log_softmax``."""

    decode_type = "ctc"

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
    ):
        """Construct CTC head.

        Args:
            vocab_size: number of output classes.
            encoder_output_size: number of encoder projection units.
        """
        super().__init__()
        self.ctc_lo = Linear(encoder_output_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project + log-softmax encoder hidden states ``(B, T, D)`` → ``(B, T, V)``."""
        return oasr.gemm_log_softmax(hidden_states, self.ctc_lo.weight, self.ctc_lo.bias)
