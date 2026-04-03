#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Base rotary embedding implementation.

Provides the standard RoPE (Rotary Position Embedding) with precomputed
complex exponentials. Designed to be extensible for scaling variants
(NTK, YaRN, linear, etc.) by overriding ``_compute_inv_freq`` or
``_compute_freqs_cis``.

Reference: https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .common import get_apply_rotary_emb


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute complex exponentials used for rotary embeddings.

    This is the standard RoPE formulation. For scaling variants (NTK, YaRN),
    use ``RotaryEmbedding`` subclasses that override ``_compute_inv_freq``.

    Args:
        dim: Head dimension (rotary_dim). Typically ``head_dim``.
        end: Maximum sequence length (max_position).
        theta: Base for frequency computation. Default 10000.
        device: Device for the output tensor. Defaults to CPU.

    Returns:
        Complex tensor of shape ``(end, dim // 2)`` (complex64).
        Each row t contains exp(i * t * inv_freq) for all frequencies.

    Reference:
        https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/rope_utils.py
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=inv_freq.device if device is None else device)
    freqs = torch.outer(t, inv_freq).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class RotaryEmbeddingBase(nn.Module):
    """Base class for rotary position embeddings.

    Subclasses can override ``_compute_inv_freq`` or ``_compute_freqs_cis``
    to implement scaling variants (NTK, YaRN, linear scaling, etc.).

    This class does not register buffers; it computes freqs_cis on demand
    for flexibility with variable sequence lengths (encoder, decoder,
    encoder-decoder with different max lengths).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        style: str = "llama",
    ):
        """Construct a RotaryEmbeddingBase.

        Args:
            head_dim: Per-head dimension (rotary_dim).
            max_position_embeddings: Maximum sequence length for precomputation.
            base: Base value (theta) for frequency computation.
            style: RoPE apply style: ``"google"``, ``"llama"``, or ``"neox"``.
        """
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim  # Full head by default; subclasses may use partial
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.style = style
        self._apply_fn = get_apply_rotary_emb(style)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute inverse frequencies. Override in subclasses for scaling variants."""
        arange = torch.arange(0, self.rotary_dim, 2, dtype=torch.float)[
            : (self.rotary_dim // 2)
        ]
        inv_freq = 1.0 / (base ** (arange / self.rotary_dim))
        return inv_freq

    def _compute_freqs_cis(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Compute freqs_cis for a given sequence length."""
        inv_freq = self._compute_inv_freq(self.base)
        if device is not None:
            inv_freq = inv_freq.to(device)
        t = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        if dtype is not None:
            freqs = freqs.to(dtype)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def get_freqs_cis(
        self,
        seq_len: int,
        batch_size: int = 1,
        n_heads: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Get freqs_cis with shape suitable for broadcasting with Q/K.

        Q/K are typically ``(batch, time, heads, head_dim)``.
        Returns ``(1, seq_len, 1, rotary_dim//2)`` for broadcasting.

        Args:
            seq_len: Sequence length.
            batch_size: Batch dimension (for shape, not used in computation).
            n_heads: Number of heads (for shape, not used in computation).
            device: Device for the tensor.
            dtype: Not used (freqs_cis is complex64).

        Returns:
            Complex tensor of shape ``(1, seq_len, 1, rotary_dim//2)``.
        """
        freqs_cis = self._compute_freqs_cis(seq_len, device=device, dtype=dtype)
        # Shape for broadcasting: (1, seq_len, 1, rotary_dim//2)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        return freqs_cis

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding to x using precomputed freqs_cis.

        Args:
            x: Query or key tensor, shape ``(batch, time, heads, head_dim)``.
            freqs_cis: Precomputed complex exponentials from ``get_freqs_cis``
                or ``precompute_freqs_cis``.

        Returns:
            Rotated tensor of same shape as x.
        """
        return self._apply_fn(x, freqs_cis)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, max_position={self.max_position_embeddings}, "
            f"base={self.base}, style={self.style}"
        )


class RotaryEmbedding(RotaryEmbeddingBase):
    """Standard rotary position embedding (no scaling).

    Use this for standard transformer models. For long-context scaling
    (NTK, YaRN, etc.), use or implement the corresponding subclasses.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        style: str = "llama",
    ):
        super().__init__(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            style=style,
        )
