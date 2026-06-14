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
def bias_norm(
    input: torch.Tensor,
    bias: torch.Tensor,
    log_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Zipformer BiasNorm over the last dimension.

    Computes ``scales = mean((x - bias)**2, dim=-1, keepdim=True)**-0.5 *
    exp(log_scale)`` then ``output = x * scales``.  No eps term (matches
    icefall's inference-time BiasNorm).

    Args:
        input: Input tensor ``[..., num_channels]``.
        bias: Per-channel bias ``[num_channels]``.
        log_scale: Scalar (1-element) log-scale tensor.
        out: Optional pre-allocated output tensor.

    Returns:
        Normalized tensor with the same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().bias_norm(out, input, bias, log_scale)
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
def add_layer_norm_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
):
    """Fused Add + LayerNorm that also returns the un-normalized sum.

    Computes, in a single kernel launch::

        s   = residual + alpha * input
        out = LayerNorm(s)            # == s normalized, scaled, shifted

    and returns ``(out, s)``.  ``s`` is the value a pre-norm residual stream
    carries forward, so this folds the per-sublayer ``x = residual + a*sub(x)``
    add (and its scale) into the *following* LayerNorm with no extra elementwise
    kernels.  Bit-identical to ``layer_norm(residual + alpha * input)``.

    Args:
        input: Sub-layer output ``[..., hidden]``.
        residual: Residual stream ``[..., hidden]``.
        weight: LayerNorm scale ``[hidden]``.
        bias: LayerNorm offset ``[hidden]`` (optional).
        eps: Epsilon for numerical stability.
        alpha: Scale applied to ``input`` before the residual add.
        out: Optional pre-allocated normalized output tensor.
        residual_out: Optional pre-allocated sum (``s``) output tensor.

    Returns:
        ``(out, residual_out)`` — the normalized tensor and the un-normalized
        residual sum, both shaped like ``input``.
    """
    if out is None:
        out = torch.empty_like(input)
    if residual_out is None:
        residual_out = torch.empty_like(input)
    _get_norm_module().addlayernorm_residual(
        out, residual_out, input, residual, weight, bias, eps, alpha
    )
    return out, residual_out


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
def cmvn(
    input: torch.Tensor,
    mean: torch.Tensor,
    istd: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Cepstral Mean and Variance Normalization (CMVN).

    Computes: output = (input - mean) * istd, where mean and istd are
    broadcast along the feature dimension.

    Args:
        input: Input tensor [..., num_cols].
        mean: Mean vector [num_cols].
        istd: Inverse standard deviation vector [num_cols].
        out: Optional pre-allocated output tensor.

    Returns:
        Normalized tensor with same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().cmvn(out, input, mean, istd)
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
