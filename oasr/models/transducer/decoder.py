# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Stateless transducer predictor (icefall ``Decoder``).

Conditions only on the last ``context_size`` labels (no recurrent state): an
embedding followed by a depthwise 1-D convolution over the label window.  Param
names (``embedding`` / ``conv``) mirror icefall ``egs/.../ASR/*/decoder.py`` so a
future converter can load icefall pruned-transducer checkpoints 1:1.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Sequence

import torch
import torch.nn.functional as F

from oasr.layers import DepthwiseConv1d, Embedding

from ..decoders.base import TransducerPredictor


class StatelessDecoder(TransducerPredictor):
    """Label predictor: ``(B, context_size)`` token window → ``(B, decoder_dim)``.

    The transducer "decode state" is just the last ``context_size`` emitted
    labels (blank-filled at init); :class:`~oasr.engine.decode.TransducerDecodeStrategy`
    shifts a new label in after each emission via :meth:`advance`.
    """

    decode_type = "transducer"
    #: The state *is* a label window, so beam search can gather-reorder it.
    label_window_state: ClassVar[bool] = True

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
        self.embedding = Embedding(vocab_size, decoder_dim, padding_idx=blank_id)
        if context_size > 1:
            # Depthwise conv over the label window (no padding: U == context_size
            # in → 1 frame out at decode time).
            self.conv = DepthwiseConv1d(decoder_dim, kernel_size=context_size, bias=False)

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
            emb = self.conv(emb)  # (B, U - context_size + 1, dim) == (B, 1, dim)
        emb = F.relu(emb)
        # (B, U_out, dim) -> (B, dim): decode feeds exactly one window per step.
        return emb[:, -1, :]

    # -- TransducerPredictor protocol ---------------------------------------
    #
    # These are the lines the greedy loop used to inline.  Stateless is the
    # degenerate case of the protocol: the state carries no history beyond the
    # window, so ``predict`` is a full recompute and ``advance`` is a shift.

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Recompute the prediction from the label window."""
        return self(state)

    def advance(
        self, state: torch.Tensor, tokens: torch.Tensor, emit: torch.Tensor
    ) -> torch.Tensor:
        """Shift ``tokens`` into the window of every row where ``emit``.

        Rows that did not emit keep their window untouched, which is what makes
        the batched :meth:`predict` that follows reproduce their previous
        projection exactly rather than approximately.
        """
        shifted = torch.cat([state[:, 1:], tokens.unsqueeze(1)], dim=1)
        merged: torch.Tensor = torch.where(emit.unsqueeze(1), shifted, state)
        return merged

    def stack_states(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(states), dim=0)

    def unstack_states(self, state: torch.Tensor) -> List[torch.Tensor]:
        return [state[b : b + 1] for b in range(state.size(0))]
