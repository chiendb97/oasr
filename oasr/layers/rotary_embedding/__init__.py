#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Rotary Position Embeddings (RoPE).

This module provides extensible, reusable rotary embedding implementations
for Transformer-based models (encoder-only, decoder-only, encoder-decoder).

Usage:
    # Precompute freqs_cis (standalone, e.g. for compatibility with WeNet)
    from oasr.layers.rotary_embedding import precompute_freqs_cis
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    # Use RotaryEmbedding module (for attention layers)
    from oasr.layers.rotary_embedding import RotaryEmbedding, get_apply_rotary_emb
    rope = RotaryEmbedding(head_dim=64, style="llama")
    freqs_cis = rope.get_freqs_cis(seq_len)
    q_rot = rope(q, freqs_cis)

    # Factory for different RoPE variants
    from oasr.layers.rotary_embedding import get_rotary_embedding
    rope = get_rotary_embedding(head_dim=64, rope_type="ntk", scaling_factor=4.0)
"""

from __future__ import annotations

from typing import Any, Optional

from .base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
    precompute_freqs_cis,
)
from .common import (
    apply_rotary_emb_google,
    apply_rotary_emb_llama,
    get_apply_rotary_emb,
    list_rope_styles,
    register_apply_rotary_emb,
)
from .ntk_scaling_rope import NTKScalingRotaryEmbedding
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding


def get_rotary_embedding(
    head_dim: int,
    max_position_embeddings: int = 2048,
    base: float = 10000.0,
    style: str = "llama",
    rope_type: str = "default",
    **kwargs: Any,
) -> RotaryEmbeddingBase:
    """Factory for creating rotary embedding instances.

    Args:
        head_dim: Per-head dimension.
        max_position_embeddings: Maximum sequence length.
        base: Base (theta) for frequency computation.
        style: Apply style: ``"google"``, ``"llama"``, or ``"neox"``.
        rope_type: RoPE variant:
            - ``"default"``: Standard RoPE
            - ``"ntk"``: NTK-aware scaling (requires ``scaling_factor``)
            - ``"yarn"``: YaRN scaling (requires ``scaling_factor``,
              ``original_max_position_embeddings``)
        **kwargs: Additional arguments for specific rope types.

    Returns:
        RotaryEmbeddingBase instance.
    """
    if rope_type == "default":
        return RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            style=style,
        )
    if rope_type == "ntk":
        scaling_factor = kwargs.get("scaling_factor", 1.0)
        mixed_b = kwargs.get("mixed_b")
        return NTKScalingRotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            style=style,
            scaling_factor=scaling_factor,
            mixed_b=mixed_b,
        )
    if rope_type == "yarn":
        scaling_factor = kwargs.get("scaling_factor", 1.0)
        original_max = kwargs.get(
            "original_max_position_embeddings", max_position_embeddings
        )
        return YaRNScalingRotaryEmbedding(
            head_dim=head_dim,
            original_max_position_embeddings=original_max,
            base=base,
            style=style,
            scaling_factor=scaling_factor,
            extrapolation_factor=kwargs.get("extrapolation_factor", 1.0),
            attn_factor=kwargs.get("attn_factor", 1.0),
            beta_fast=kwargs.get("beta_fast", 32.0),
            beta_slow=kwargs.get("beta_slow", 1.0),
        )
    raise ValueError(f"Unknown rope_type: {rope_type}")


__all__ = [
    # Base
    "RotaryEmbedding",
    "RotaryEmbeddingBase",
    "precompute_freqs_cis",
    # Common / apply
    "apply_rotary_emb_google",
    "apply_rotary_emb_llama",
    "get_apply_rotary_emb",
    "list_rope_styles",
    "register_apply_rotary_emb",
    # Variants
    "NTKScalingRotaryEmbedding",
    "YaRNScalingRotaryEmbedding",
    # Factory
    "get_rotary_embedding",
]
