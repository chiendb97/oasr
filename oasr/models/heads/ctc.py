# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CTC output head."""

from __future__ import annotations

import torch

from oasr.layers.ctc import CtcProjection

from ..base import BaseHead


class CTCHead(BaseHead):
    """CTC head: thin :class:`~oasr.models.base.BaseHead` over the fused
    :class:`~oasr.layers.ctc.CtcProjection` (Linear → log_softmax).

    The compute lives in the reusable layer; this class only attaches the
    engine-facing ``decode_type`` and the ``BaseHead`` contract.
    """

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
        # Named ``ctc_lo`` so external checkpoint keys ``ctc.ctc_lo.{weight,bias}``
        # map straight onto the projection's ``weight`` / ``bias`` parameters.
        self.ctc_lo = CtcProjection(encoder_output_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project + log-softmax encoder hidden states ``(B, T, D)`` → ``(B, T, V)``."""
        return self.ctc_lo(hidden_states)
