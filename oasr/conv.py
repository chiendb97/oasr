# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for convolution operations."""

import functools
from typing import Optional

import torch

import oasr.jit.conv as _jit_conv
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


def _default_conv1d_fn():
    from oasr.jit.conv import CONV2D_DEFAULT, conv1d_func_name

    return getattr(_get_conv2d_module(), conv1d_func_name(CONV2D_DEFAULT))


def _default_conv1d_activation_fn():
    from oasr.jit.conv import CONV2D_DEFAULT, conv1d_activation_func_name

    return getattr(_get_conv2d_module(), conv1d_activation_func_name(CONV2D_DEFAULT))


@functools.cache
def _target_sm() -> int:
    from oasr.jit.core import _get_target_sm

    return _get_target_sm()


@functools.cache
def _conv1d_fn(compile_name: str):
    return getattr(_get_conv2d_module(), f"conv1d_{compile_name}")


@functools.cache
def _conv1d_activation_fn(compile_name: str):
    return getattr(_get_conv2d_module(), f"conv1d_{compile_name}_activation")


def _dispatch_conv1d(
    out: torch.Tensor,
    input: torch.Tensor,
    filter: torch.Tensor,
    bias: Optional[torch.Tensor],
    padding: int,
    stride: int,
    dilation: int,
) -> None:
    """Run the measured shape-specific tile, retaining a safe fixed fallback."""
    try:
        choice = _jit_conv.select_default_conv1d_config(
            input.shape[0],
            input.shape[1],
            input.shape[2],
            filter.shape[0],
            filter.shape[1],
            padding,
            stride,
            dilation,
            input.dtype,
            _target_sm(),
        )
        _conv1d_fn(choice.compile_name)(out, input, filter, bias, padding, stride, dilation)
        return
    except Exception:
        pass
    _default_conv1d_fn()(out, input, filter, bias, padding, stride, dilation)


def _dispatch_conv1d_activation(
    out: torch.Tensor,
    input: torch.Tensor,
    filter: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation_type: int,
    padding: int,
    stride: int,
    dilation: int,
) -> None:
    """Run a measured fused-activation tile, retaining a safe fixed fallback."""
    try:
        choice = _jit_conv.select_default_conv1d_activation_config(
            input.shape[0],
            input.shape[1],
            input.shape[2],
            filter.shape[0],
            filter.shape[1],
            padding,
            stride,
            dilation,
            input.dtype,
            _target_sm(),
        )
        _conv1d_activation_fn(choice.compile_name)(
            out, input, filter, bias, activation_type, padding, stride, dilation
        )
        return
    except Exception:
        pass
    _default_conv1d_activation_fn()(
        out, input, filter, bias, activation_type, padding, stride, dilation
    )


# IC threshold below which cuDNN is used instead of CUTLASS.
# CUTLASS implicit GEMM uses scalar alignment (=1) for all IC values, but
# cuDNN can pick better algorithms when IC is small (e.g. IC=1 in conformer
# subsampling).
_CUDNN_IC_THRESHOLD = 8


def _conv1d_output_length(
    seq_len: int, kernel_size: int, padding: int, stride: int, dilation: int
) -> int:
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    if stride <= 0 or dilation <= 0:
        raise ValueError(f"stride and dilation must be positive, got {stride=} {dilation=}")
    out_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    if out_len <= 0:
        raise ValueError(
            f"Conv1D output length must be positive, got {out_len} from "
            f"T={seq_len}, kernel={kernel_size}, padding={padding}, "
            f"stride={stride}, dilation={dilation}"
        )
    return out_len


@oasr_api
def conv1d(
    input: torch.Tensor,
    filter: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense cross-channel 1D convolution in packed BTC layout.

    The CUTLASS path specializes the NHWC implicit-GEMM convolution to a
    height-one problem, so neither the activation nor the KSC filter is
    transposed or materialized.  cuDNN remains an autotuning candidate and
    serves channel counts that do not meet CUTLASS's alignment requirement.

    Args:
        input: Input ``[batch, seq_len, in_channels]``.
        filter: Filter ``[out_channels, kernel_size, in_channels]`` (KSC).
        bias: Optional bias ``[out_channels]``.
        padding: Symmetric temporal padding.
        stride: Temporal stride.
        dilation: Temporal dilation.
        out: Optional pre-allocated ``[batch, out_len, out_channels]`` tensor.
    """
    if input.ndim != 3 or filter.ndim != 3:
        raise ValueError(
            f"conv1d expects 3D BTC input and KSC filter, got {input.ndim}D and {filter.ndim}D"
        )
    batch, seq_len, in_channels = input.shape
    out_channels, kernel_size, filter_channels = filter.shape
    if filter_channels != in_channels:
        raise ValueError(
            f"conv1d channel mismatch: input has {in_channels}, filter has {filter_channels}"
        )
    out_len = _conv1d_output_length(seq_len, kernel_size, padding, stride, dilation)
    if out is None:
        out = torch.empty(batch, out_len, out_channels, device=input.device, dtype=input.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled() and in_channels % 8 == 0 and out_channels % 8 == 0:
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("conv", "conv1d"),
            shape_sig=(
                batch,
                seq_len,
                in_channels,
                out_channels,
                kernel_size,
                padding,
                stride,
                dilation,
            ),
            dtype=input.dtype,
            device=input.device,
            runner_args=(out, input, filter, bias, padding, stride, dilation),
        )
    elif in_channels % 8 or out_channels % 8:
        _get_cudnn_conv2d_module().cudnn_conv1d(out, input, filter, bias, padding, stride, dilation)
    else:
        _dispatch_conv1d(out, input, filter, bias, padding, stride, dilation)
    return out


@oasr_api
def conv1d_activation(
    input: torch.Tensor,
    filter: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation_type: int = 2,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense BTC Conv1D with a fused ReLU, GELU-tanh, or SiLU epilogue."""
    if input.ndim != 3 or filter.ndim != 3:
        raise ValueError(
            "conv1d_activation expects 3D BTC input and KSC filter, "
            f"got {input.ndim}D and {filter.ndim}D"
        )
    batch, seq_len, in_channels = input.shape
    out_channels, kernel_size, filter_channels = filter.shape
    if filter_channels != in_channels:
        raise ValueError(
            "conv1d_activation channel mismatch: "
            f"input has {in_channels}, filter has {filter_channels}"
        )
    out_len = _conv1d_output_length(seq_len, kernel_size, padding, stride, dilation)
    if out is None:
        out = torch.empty(batch, out_len, out_channels, device=input.device, dtype=input.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled() and in_channels % 8 == 0 and out_channels % 8 == 0:
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("conv", "conv1d_activation"),
            shape_sig=(
                batch,
                seq_len,
                in_channels,
                out_channels,
                kernel_size,
                padding,
                stride,
                dilation,
            ),
            dtype=input.dtype,
            device=input.device,
            runner_args=(
                out,
                input,
                filter,
                bias,
                activation_type,
                padding,
                stride,
                dilation,
            ),
        )
    elif in_channels % 8 or out_channels % 8:
        _get_cudnn_conv2d_module().cudnn_conv1d_activation(
            out, input, filter, bias, activation_type, padding, stride, dilation
        )
    else:
        _dispatch_conv1d_activation(
            out, input, filter, bias, activation_type, padding, stride, dilation
        )
    return out


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
            input.shape[0],
            out_len,
            input.shape[2],
            device=input.device,
            dtype=input.dtype,
        )
    _get_conv_module().depthwise_conv1d(out, input, weight, bias, padding)
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
            shape_sig=(
                N,
                H,
                W,
                _IC,
                K,
                R,
                S,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
            ),
            dtype=input.dtype,
            device=input.device,
            runner_args=(
                out,
                input,
                filter,
                bias,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
            ),
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
            input.shape[0],
            out_len,
            input.shape[2],
            device=input.device,
            dtype=input.dtype,
        )
    _get_conv_module().depthwise_conv1d_silu(out, input, weight, bias, padding)
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
            shape_sig=(
                N,
                H,
                W,
                _IC,
                K,
                R,
                S,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
            ),
            dtype=input.dtype,
            device=input.device,
            runner_args=(
                out,
                input,
                filter,
                bias,
                activation_type,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
            ),
        )
        return out

    if IC < _CUDNN_IC_THRESHOLD:
        _get_cudnn_conv2d_module().cudnn_conv2d_activation(
            out,
            input,
            filter,
            bias,
            activation_type,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
        )
    else:
        _default_conv2d_activation_fn()(
            out,
            input,
            filter,
            bias,
            activation_type,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
        )
    return out
