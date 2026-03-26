# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for normalization operations."""

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api


@functools.cache
def _get_norm_module():
    from oasr.jit.norm import gen_norm_module

    return gen_norm_module().build_and_load()


@oasr_api
def layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply layer normalization.

    Args:
        input: Input tensor [batch, seq_len, hidden_size].
        weight: Scale parameter [hidden_size].
        bias: Optional offset parameter [hidden_size].
        eps: Epsilon for numerical stability.
        out: Optional pre-allocated output tensor.

    Returns:
        Normalized tensor with same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().layernorm(out, input, weight, bias, eps)
    return out


@oasr_api
def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply RMS normalization.

    Args:
        input: Input tensor [batch, seq_len, hidden_size].
        weight: Scale parameter [hidden_size].
        bias: Optional offset parameter [hidden_size].
        eps: Epsilon for numerical stability.
        out: Optional pre-allocated output tensor.

    Returns:
        Normalized tensor with same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().rmsnorm(out, input, weight, bias, eps)
    return out


@oasr_api
def batch_norm_1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply batch normalization (inference mode).

    Args:
        input: Input tensor [batch, seq_len, channels].
        weight: Scale parameter [channels].
        bias: Offset parameter [channels].
        running_mean: Running mean [channels].
        running_var: Running variance [channels].
        eps: Epsilon for numerical stability.
        out: Optional pre-allocated output tensor.

    Returns:
        Normalized tensor with same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().batchnorm1d(out, input, weight, bias, running_mean, running_var, eps)
    return out


@oasr_api
def group_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply group normalization.

    Args:
        input: Input tensor [batch, seq_len, channels].
        weight: Scale parameter [channels].
        bias: Offset parameter [channels].
        num_groups: Number of groups.
        eps: Epsilon for numerical stability.
        out: Optional pre-allocated output tensor.

    Returns:
        Normalized tensor with same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().groupnorm(out, input, weight, bias, num_groups, eps)
    return out


@oasr_api
def add_layer_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply fused Add + LayerNorm: output = LayerNorm(input + residual).

    Args:
        input: Input tensor [batch, seq_len, hidden_size].
        residual: Residual tensor [batch, seq_len, hidden_size].
        weight: Scale parameter [hidden_size].
        bias: Offset parameter [hidden_size].
        eps: Epsilon for numerical stability.
        out: Optional pre-allocated output tensor.

    Returns:
        Normalized tensor with same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().addlayernorm(out, input, residual, weight, bias, eps)
    return out


@oasr_api
def layer_norm_activation(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    activation_type: int = 2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused LayerNorm + Activation."""
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().layernorm_activation(out, input, weight, bias, eps, activation_type)
    return out


@oasr_api
def rms_norm_activation(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    activation_type: int = 2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused RMSNorm + Activation."""
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().rmsnorm_activation(out, input, weight, bias, eps, activation_type)
    return out


@oasr_api
def batch_norm_activation(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    activation_type: int = 2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused BatchNorm + Activation (inference mode)."""
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().batchnorm_activation(
        out, input, weight, bias, running_mean, running_var, eps, activation_type
    )
    return out


@oasr_api
def batch_norm_swish(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused BatchNorm + Swish (inference mode)."""
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().batchnorm_swish(out, input, weight, bias, running_mean, running_var, eps)
    return out
