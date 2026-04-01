#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
NTK-aware scaled rotary embedding.

Extends the base RoPE with NTK scaling for long-context extrapolation.
Reference: https://kexue.fm/archives/9706

Use ``rope_type="ntk"`` in ``get_rotary_embedding`` factory.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import RotaryEmbeddingBase


class NTKScalingRotaryEmbedding(RotaryEmbeddingBase):
    """RoPE with NTK scaling for extended context length."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        style: str = "llama",
        scaling_factor: float = 1.0,
        mixed_b: Optional[float] = None,
    ):
        """Construct NTK-scaled RoPE.

        Args:
            head_dim: Per-head dimension.
            max_position_embeddings: Maximum sequence length.
            base: Base (theta) for frequency computation.
            style: Apply style (google, llama, neox).
            scaling_factor: NTK scaling factor. Typically
                ``extended_len / original_max_len``.
            mixed_b: Optional mixed scaling exponent. If set, uses
                dimension-dependent scaling.
        """
        super().__init__(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            style=style,
        )
        self.scaling_factor = scaling_factor
        self.mixed_b = mixed_b

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        base_eff = base * (
            self.scaling_factor if self.mixed_b is None else 1.0
        )
        inv_freq = 1.0 / (
            base_eff
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)[
                    : (self.rotary_dim // 2)
                ]
                / self.rotary_dim
            )
        )
        if self.mixed_b is None:
            inv_freq = inv_freq / (self.scaling_factor ** (2 / self.rotary_dim))
        else:
            a = (
                torch.tensor(self.scaling_factor).log()
                / (self.rotary_dim / 2) ** self.mixed_b
            )
            lambda_1_m = (
                a
                * torch.arange(1, self.rotary_dim // 2 + 1, dtype=torch.float)
                ** self.mixed_b
            ).exp()
            inv_freq = inv_freq / lambda_1_m
        return inv_freq
