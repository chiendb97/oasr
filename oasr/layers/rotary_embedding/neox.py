# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GPT-NeoX / HF-style rotary embedding over arbitrary position tensors.

The sibling :class:`~oasr.layers.rotary_embedding.base.RotaryEmbedding` builds
a *complex* ``freqs_cis`` table for positions ``0..seq_len`` and applies it to
``(B, T, H, D)``.  That shape cannot express what a batched LLM decoder needs:
**per-row** positions (a left-padded prompt has ``cumsum(mask) - 1``, so row 0
and row 1 are at different positions in the same column) applied to
``(B, H, T, D)``.  Hence a second, real-valued implementation rather than a
bent one.

Naming trap worth stating once: this module's rotation is what HF and vLLM
call *NeoX style* — split the head into halves and rotate one against the
other.  The registry style named ``"llama"`` in :mod:`.common` is the
*interleaved* (GPT-J) pairing, and ``"google"`` is the half-split one.  Those
names predate this file; go by the math, not the label.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]`` over the last dimension's two halves."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q``/``k`` ``(B, H, T, D)`` by ``cos``/``sin`` ``(B, T, D)``.

    ``unsqueeze_dim=1`` broadcasts the tables across the head axis, matching
    HF's ``apply_rotary_pos_emb``.
    """
    cos = cos.to(q.dtype).unsqueeze(unsqueeze_dim)
    sin = sin.to(q.dtype).unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class NeoxRotaryEmbedding(nn.Module):
    """Real ``cos``/``sin`` tables for an arbitrary integer position tensor.

    Frequencies are held as a non-persistent fp32 buffer and the tables are
    built in fp32 regardless of the model dtype — the position ladder and the
    trig values are exactly the places half precision costs accuracy, and HF
    computes them in fp32 too, so it is a parity requirement as much as a
    numerical one.
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float = 10000.0, device=None) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``positions (...)`` integer → ``cos``/``sin`` ``(..., head_dim)`` fp32."""
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, theta={self.theta}"


__all__ = ["NeoxRotaryEmbedding", "apply_rotary_pos_emb", "rotate_half"]
