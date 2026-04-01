#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Common rotary embedding apply functions.

This module provides the core apply functions for different RoPE styles
(Google/Gemma, LLaMA/Neox). These are pure PyTorch implementations
that work with precomputed complex exponentials (freqs_cis).

Reference: https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/rope_utils.py
"""

from __future__ import annotations

from typing import Callable

import torch

# Registry of apply functions: style name -> (x, freqs_cis) -> x_rotated
_APPLY_ROTARY_EMB: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}


def apply_rotary_emb_google(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding (Google/Gemma style) to query/key tensors.

    Uses half-dimension chunking: pairs (x[0], x[1]), (x[2], x[3]), ...
    as complex numbers.

    Args:
        x: Input tensor of shape ``(batch, time, heads, dim)``.
        freqs_cis: Precomputed complex exponentials, shape ``(..., dim/2)``
            (complex64). Must broadcast with x.

    Returns:
        Rotated tensor of same shape as x.
    """
    # x: [batch, time, heads, dim]
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.float(), 2, dim=-1), dim=-1)
    )
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    return x_out


def apply_rotary_emb_llama(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding (LLaMA/Neox style) to query/key tensors.

    Uses interleaved pairing: (x[0], x[1]), (x[2], x[3]), ... as complex numbers.
    Equivalent to GPT-J / Neox style in vLLM terminology.

    Args:
        x: Input tensor of shape ``(batch, time, heads, dim)``.
        freqs_cis: Precomputed complex exponentials, shape ``(..., dim/2)``
            (complex64). Must broadcast with x.

    Returns:
        Rotated tensor of same shape as x.
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


def register_apply_rotary_emb(style: str):
    """Decorator to register an apply function for a RoPE style."""

    def decorator(
        fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        _APPLY_ROTARY_EMB[style] = fn
        return fn

    return decorator


# Register built-in styles
_APPLY_ROTARY_EMB["google"] = apply_rotary_emb_google
_APPLY_ROTARY_EMB["llama"] = apply_rotary_emb_llama
# Alias: Neox is the same as LLaMA style
_APPLY_ROTARY_EMB["neox"] = apply_rotary_emb_llama


def get_apply_rotary_emb(style: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get the apply function for a given RoPE style.

    Args:
        style: One of ``"google"``, ``"llama"``, ``"neox"``, or any
            custom-registered style.

    Returns:
        Callable ``(x, freqs_cis) -> x_rotated``.

    Raises:
        ValueError: If style is not registered.
    """
    if style not in _APPLY_ROTARY_EMB:
        raise ValueError(
            f"Unknown RoPE style '{style}', supported: {list(_APPLY_ROTARY_EMB)}"
        )
    return _APPLY_ROTARY_EMB[style]


def list_rope_styles() -> list[str]:
    """Return list of registered RoPE apply styles."""
    return list(_APPLY_ROTARY_EMB.keys())
