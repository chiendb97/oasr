# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for convolution operations."""

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api


@functools.cache
def _get_conv_module():
    from oasr.jit.conv import gen_conv_module

    return gen_conv_module().build_and_load()


@functools.cache
def _get_conv2d_module():
    from oasr.jit.conv import gen_conv2d_module

    return gen_conv2d_module().build_and_load()


@functools.cache
def _get_cudnn_conv2d_module():
    from oasr.jit.conv import gen_cudnn_conv2d_module

    return gen_cudnn_conv2d_module().build_and_load()


def _default_conv2d_fn():
    from oasr.jit.conv import CONV2D_DEFAULT, conv2d_func_name

    return getattr(_get_conv2d_module(), conv2d_func_name(CONV2D_DEFAULT))


def _default_conv2d_activation_fn():
    from oasr.jit.conv import CONV2D_DEFAULT, conv2d_activation_func_name

    return getattr(_get_conv2d_module(), conv2d_activation_func_name(CONV2D_DEFAULT))


# IC threshold below which cuDNN is used instead of CUTLASS.
# CUTLASS implicit GEMM uses scalar alignment (=1) for all IC values, but
# cuDNN can pick better algorithms when IC is small (e.g. IC=1 in conformer
# subsampling).
_CUDNN_IC_THRESHOLD = 8


@oasr_api
def depthwise_conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    padding: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Depthwise separable 1D convolution.

    Args:
        input: Input [batch, seq_len, channels].
        weight: Weight [kernel_size, channels].
        bias: Optional bias [channels].
        padding: Padding size.
        out: Optional pre-allocated output tensor.

    Returns:
        Output [batch, out_len, channels] where out_len = seq_len + 2*padding - kernel_size + 1.
    """
    if out is None:
        kernel_size = weight.shape[0]
        out_len = input.shape[1] + 2 * padding - kernel_size + 1
        out = torch.empty(
            input.shape[0], out_len, input.shape[2],
            device=input.device, dtype=input.dtype,
        )
    _get_conv_module().depthwise_conv1d(out, input, weight, bias, padding)
    return out


@oasr_api
def pointwise_conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pointwise (1x1) convolution.

    Args:
        input: Input [batch, seq_len, in_channels].
        weight: Weight [out_channels, in_channels].
        bias: Optional bias [out_channels].
        out: Optional pre-allocated output tensor.

    Returns:
        Output [batch, seq_len, out_channels].
    """
    if out is None:
        batch, seq_len = input.shape[0], input.shape[1]
        out_channels = weight.shape[0]
        out = torch.empty(batch, seq_len, out_channels, device=input.device, dtype=input.dtype)
    _get_conv_module().pointwise_conv1d(out, input, weight, bias)
    return out


@oasr_api
def conv2d(
    input: torch.Tensor,
    filter: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    pad_h: int = 0,
    pad_w: int = 0,
    stride_h: int = 1,
    stride_w: int = 1,
    dilation_h: int = 1,
    dilation_w: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """2D convolution (NHWC layout).

    Uses cuDNN when IC < 8 (better algorithm selection for small channel counts),
    CUTLASS Implicit GEMM otherwise.

    When autotuning is enabled (``oasr.tune.autotune``), the autotuner selects
    the fastest backend by profiling.

    Args:
        input: Input [N, H, W, IC].
        filter: Filter [K, R, S, IC].
        bias: Optional per-channel bias [K].
        pad_h, pad_w: Symmetric padding.
        stride_h, stride_w: Convolution stride.
        dilation_h, dilation_w: Dilation.
        out: Optional pre-allocated output tensor.

    Returns:
        Output [N, P, Q, K].
    """
    IC = input.shape[3]
    if out is None:
        N = input.shape[0]
        H, W = input.shape[1], input.shape[2]
        K, R, S = filter.shape[0], filter.shape[1], filter.shape[2]
        P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
        Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1
        out = torch.empty(N, P, Q, K, device=input.device, dtype=input.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        N, H, W, _IC = input.shape
        K, R, S, _ = filter.shape
        get_tuner().dispatch(
            op_key=OpKey("conv", "conv2d"),
            shape_sig=(N, H, W, _IC, K, R, S,
                       stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w),
            dtype=input.dtype,
            device=input.device,
            runner_args=(out, input, filter, bias,
                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w),
        )
        return out

    if IC < _CUDNN_IC_THRESHOLD:
        _get_cudnn_conv2d_module().cudnn_conv2d(
            out, input, filter, bias, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w
        )
    else:
        _default_conv2d_fn()(
            out, input, filter, bias, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w
        )
    return out


@oasr_api
def depthwise_conv1d_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    padding: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused depthwise 1D convolution + SiLU activation."""
    if out is None:
        kernel_size = weight.shape[0]
        out_len = input.shape[1] + 2 * padding - kernel_size + 1
        out = torch.empty(
            input.shape[0], out_len, input.shape[2],
            device=input.device, dtype=input.dtype,
        )
    _get_conv_module().depthwise_conv1d_silu(out, input, weight, bias, padding)
    return out


@oasr_api
def pointwise_conv1d_activation(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation_type: int = 2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pointwise (1x1) convolution with fused activation."""
    if out is None:
        batch, seq_len = input.shape[0], input.shape[1]
        out_channels = weight.shape[0]
        out = torch.empty(batch, seq_len, out_channels, device=input.device, dtype=input.dtype)
    _get_conv_module().pointwise_conv1d_activation(out, input, weight, bias, activation_type)
    return out


@oasr_api
def causal_conv1d(
    input: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Causal 1D convolution with state management (streaming)."""
    if out is None:
        out = torch.empty_like(input)
    _get_conv_module().causal_conv1d(out, input, state, weight, bias)
    return out


@oasr_api
def conv2d_activation(
    input: torch.Tensor,
    filter: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation_type: int = 2,
    pad_h: int = 0,
    pad_w: int = 0,
    stride_h: int = 1,
    stride_w: int = 1,
    dilation_h: int = 1,
    dilation_w: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """2D convolution with fused activation (NHWC layout).

    Uses cuDNN when IC < 8, CUTLASS Implicit GEMM otherwise.
    When autotuning is enabled, the autotuner selects the fastest backend.
    """
    IC = input.shape[3]
    if out is None:
        N = input.shape[0]
        H, W = input.shape[1], input.shape[2]
        K, R, S = filter.shape[0], filter.shape[1], filter.shape[2]
        P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
        Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1
        out = torch.empty(N, P, Q, K, device=input.device, dtype=input.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        N, H, W, _IC = input.shape
        K, R, S, _ = filter.shape
        get_tuner().dispatch(
            op_key=OpKey("conv", "conv2d_activation"),
            shape_sig=(N, H, W, _IC, K, R, S,
                       stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w),
            dtype=input.dtype,
            device=input.device,
            runner_args=(out, input, filter, bias, activation_type,
                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w),
        )
        return out

    if IC < _CUDNN_IC_THRESHOLD:
        _get_cudnn_conv2d_module().cudnn_conv2d_activation(
            out, input, filter, bias, activation_type, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w,
        )
    else:
        _default_conv2d_activation_fn()(
            out, input, filter, bias, activation_type, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w,
        )
    return out
