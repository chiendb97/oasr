# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for normalization operations."""

import functools
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from oasr.api_logging import oasr_api
from oasr.functionals.quant import QUANT_EPS, quant_qmax, reference_quantize_per_token


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
def add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply fused residual add + RMSNorm.

    Computes ``RMSNorm(residual + alpha * input)`` with the residual sum kept
    in fp32 through normalization and no intermediate global-memory tensor.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().addrmsnorm(out, input, residual, weight, bias, eps, alpha)
    return out


@oasr_api
def add_rms_norm_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply fused residual add + RMSNorm and return the unnormalized sum.

    The normalized output consumes ``residual + alpha * input`` directly in
    fp32. The returned residual is that sum rounded to the input dtype for the
    next pre-norm sublayer.
    """
    if out is None:
        out = torch.empty_like(input)
    if residual_out is None:
        residual_out = torch.empty_like(input)
    _get_norm_module().addrmsnorm_residual(
        out, residual_out, input, residual, weight, bias, eps, alpha
    )
    return out, residual_out


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
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply fused Add + LayerNorm: output = LayerNorm(residual + alpha * input).

    Args:
        input: Input tensor [batch, seq_len, hidden_size].
        residual: Residual tensor [batch, seq_len, hidden_size].
        weight: Scale parameter [hidden_size].
        bias: Offset parameter [hidden_size].
        eps: Epsilon for numerical stability.
        out: Optional pre-allocated output tensor.
        alpha: Scale applied to ``input`` before the residual add.

    Returns:
        Normalized tensor with same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_norm_module().addlayernorm(out, input, residual, weight, bias, eps, alpha)
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

        s_float = float(residual) + alpha * float(input)
        out     = LayerNorm(s_float)
        s       = cast_to_input_dtype(s_float)

    and returns ``(out, s)``.  ``s`` is the value a pre-norm residual stream
    carries forward, so this folds the per-sublayer ``x = residual + a*sub(x)``
    add (and its scale) into the *following* LayerNorm with no extra elementwise
    kernels. The normalization consumes the fp32 sum directly; only the
    carried residual and normalized output are rounded to the input dtype.

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


def _quant_outputs(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``quantize_per_token``'s destinations: 8-bit (*dtype*) like *input*, fp32 per row."""
    if out is None:
        quant_qmax(dtype)  # refuses a format the kernels do not write
        out = torch.empty(input.shape, dtype=dtype, device=input.device)
    if scale is None:
        scale = torch.empty(input.shape[:-1], dtype=torch.float32, device=input.device)
    return out, scale


@oasr_api
def layer_norm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    qeps: float = QUANT_EPS,
    out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm emitting per-token 8-bit values directly: one launch for norm and quantize.

    Bit-exact with the unfused pair::

        layer_norm_quant(x, w, b, eps, qeps, dtype=d)
            == quantize_per_token(layer_norm(x, w, b, eps), None, qeps, dtype=d)

    The normalized value is rounded to the input dtype before it is quantized,
    exactly as the stored norm output would be, and the quantize follows
    ``include/oasr/quant/quantize.cuh`` (no smoother: SmoothQuant folds its
    scale into the norm's affine for these layers).

    Args:
        input: fp16/bf16 CUDA tensor ``[..., hidden]``, contiguous.
        weight: LayerNorm scale ``[hidden]``, the input's dtype.
        bias: Optional LayerNorm offset ``[hidden]``.
        eps: LayerNorm epsilon.
        qeps: Floor on each row's amax (:data:`oasr.functionals.quant.QUANT_EPS`).
        out: Optional pre-allocated 8-bit tensor shaped like *input*; its dtype
            wins over *dtype*.
        scale: Optional pre-allocated fp32 tensor, one entry per row.
        dtype: ``torch.int8`` or ``torch.float8_e4m3fn``, when *out* is not given.

    Returns:
        ``(q, scale)``: 8-bit values shaped like *input* and the fp32 per-row
        scale, shaped ``input.shape[:-1]`` (as :func:`oasr.quantize_per_token`).
    """
    out, scale = _quant_outputs(input, out, scale, dtype)
    _get_norm_module().layernorm_quant(out, scale, input, weight, bias, eps, qeps)
    return out, scale


@oasr_api
def add_layer_norm_residual_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    alpha: float = 1.0,
    qeps: float = QUANT_EPS,
    out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`add_layer_norm_residual` whose normalized output is per-token 8-bit.

    Bit-exact with the unfused pair: for ``(y, s) = add_layer_norm_residual(
    input, residual, weight, bias, eps, alpha)`` this returns
    ``(*quantize_per_token(y, None, qeps, dtype=dtype), s)``.  ``residual_out``
    (``s``) is stored exactly as :func:`add_layer_norm_residual` stores it.

    Args:
        input: Sub-layer output ``[..., hidden]``, fp16/bf16, contiguous.
        residual: Residual stream, shaped and laid out like *input*.
        weight: LayerNorm scale ``[hidden]``.
        bias: Optional LayerNorm offset ``[hidden]``.
        eps: LayerNorm epsilon.
        alpha: Scale applied to *input* before the residual add.
        qeps: Floor on each row's amax.
        out: Optional pre-allocated 8-bit tensor shaped like *input*; its dtype
            wins over *dtype*.
        scale: Optional pre-allocated fp32 tensor, one entry per row.
        residual_out: Optional pre-allocated sum (``s``) tensor.
        dtype: ``torch.int8`` or ``torch.float8_e4m3fn``, when *out* is not given.

    Returns:
        ``(q, scale, residual_out)``.
    """
    out, scale = _quant_outputs(input, out, scale, dtype)
    if residual_out is None:
        residual_out = torch.empty_like(input)
    _get_norm_module().addlayernorm_residual_quant(
        out, scale, residual_out, input, residual, weight, bias, eps, alpha, qeps
    )
    return out, scale, residual_out


def _reference_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float
) -> torch.Tensor:
    """LayerNorm in fp32, rounded once to the input dtype (the kernels' order)."""
    bias_float = None if bias is None else bias.float()
    return F.layer_norm(x.float(), (x.size(-1),), weight.float(), bias_float, eps).to(x.dtype)


def reference_layer_norm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    qeps: float = QUANT_EPS,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch twin of :func:`layer_norm_quant`, on any device.

    The quantize half is :func:`oasr.functionals.quant.reference_quantize_per_token`
    and bit-exact given the same normalized values; the LayerNorm half is torch's
    reduction, not the kernel's, so a normalized value can differ by one ulp of
    the input dtype -- and its 8-bit code by one step -- from the kernel.
    """
    return reference_quantize_per_token(
        _reference_layer_norm(input, weight, bias, eps), None, qeps, dtype=dtype
    )


def reference_add_layer_norm_residual_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    alpha: float = 1.0,
    qeps: float = QUANT_EPS,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch twin of :func:`add_layer_norm_residual_quant`, on any device.

    The fp32 sum is normalized directly and only rounded for the carried
    residual, as in the kernel; see :func:`reference_layer_norm_quant` for what
    is and is not bit-exact.
    """
    summed = residual.float() + alpha * input.float()
    bias_float = None if bias is None else bias.float()
    normalized = F.layer_norm(summed, (input.size(-1),), weight.float(), bias_float, eps)
    q, scale = reference_quantize_per_token(normalized.to(input.dtype), None, qeps, dtype=dtype)
    return q, scale, summed.to(input.dtype)


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
