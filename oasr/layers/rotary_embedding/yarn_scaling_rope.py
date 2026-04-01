#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
YaRN (Yet another RoPE extensioN) scaled rotary embedding.

Provides improved long-context extrapolation via attention scaling.
Reference: https://arxiv.org/abs/2309.00071

Use ``rope_type="yarn"`` in ``get_rotary_embedding`` factory.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import RotaryEmbeddingBase


def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    """Inverse dim formula to find dim based on number of rotations."""
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
    truncate: bool = True,
) -> tuple[float, float]:
    """Find dim range bounds based on rotations."""
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


class YaRNScalingRotaryEmbedding(RotaryEmbeddingBase):
    """RoPE with YaRN scaling for long-context extrapolation."""

    def __init__(
        self,
        head_dim: int,
        original_max_position_embeddings: int,
        base: float = 10000.0,
        style: str = "llama",
        scaling_factor: float = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        """Construct YaRN-scaled RoPE.

        Args:
            head_dim: Per-head dimension.
            original_max_position_embeddings: Original training max length.
            base: Base (theta) for frequency computation.
            style: Apply style (google, llama, neox).
            scaling_factor: YaRN scaling factor (extended_len / original_len).
            extrapolation_factor: Extrapolation factor for interpolation.
            attn_factor: Attention scaling factor.
            beta_fast: Fast beta for ramp.
            beta_slow: Slow beta for ramp.
        """
        super().__init__(
            head_dim=head_dim,
            max_position_embeddings=int(
                original_max_position_embeddings * scaling_factor
            ),
            base=base,
            style=style,
        )
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Precompute YaRN ramp mask
        low, high = _yarn_find_correction_range(
            1, scaling_factor, self.rotary_dim // 2, base,
            original_max_position_embeddings
        )
        ramp = (torch.arange(self.rotary_dim // 2, dtype=torch.float) - low) / (
            high - low + 1e-6
        )
        self.register_buffer(
            "yarn_ramp",
            torch.clamp(ramp, 0, 1),
            persistent=False,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)[
                    : (self.rotary_dim // 2)
                ]
                / self.rotary_dim
            )
        )
        # Apply YaRN interpolation scaling (ramp on same device as inv_freq)
        ramp = self.yarn_ramp.to(inv_freq.device)
        inv_freq = inv_freq / (self.scaling_factor ** ramp)
        return inv_freq
