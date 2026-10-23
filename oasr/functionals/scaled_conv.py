# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""int8 convolution with a per-channel dequantize, bias and activation epilogue.

The forward of a quantized ``Conv2d`` / ``Conv1d``:

    out = act(scale[k] * conv(q_x, q_w)[..., k] + bias[k])

``q_x`` is the int8 NHWC (or ``(B, T, C)``) input quantized with one static
per-tensor scale, ``q_w`` the int8 KRSC (``(K, S, C)``) filter quantized per output
channel, and ``scale = input_scale * weight_scale`` folded into one fp32 vector.
A per-token activation scale cannot be used here: an output pixel sums over a
receptive field spanning several input pixels, so their scales would not factor
out of the sum.

The int32 accumulator is exact, so every tile computes the same bits -- the
kernel is batch-invariant -- and the reference below reproduces it exactly up to
the epilogue's single fp32 FMA and the output rounding.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from oasr.api_logging import oasr_api

from .conv import _row_slices
from .scaled_mm import _TORCH_ACTIVATION, activation_id

Pair = Union[int, Tuple[int, int]]


def _pair(v: Pair) -> Tuple[int, int]:
    return (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))


@functools.cache
def _get_scaled_conv_module():
    from oasr.jit.scaled_conv import gen_scaled_conv_module

    return gen_scaled_conv_module().build_and_load()


def _output_hw(h: int, w: int, r: int, s: int, padding, stride, dilation) -> Tuple[int, int]:
    p = (h + 2 * padding[0] - dilation[0] * (r - 1) - 1) // stride[0] + 1
    q = (w + 2 * padding[1] - dilation[1] * (s - 1) - 1) // stride[1] + 1
    return p, q


@oasr_api
def scaled_conv2d(
    input: torch.Tensor,
    filter: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    padding: Pair = 0,
    stride: Pair = 1,
    dilation: Pair = 1,
    out_dtype: torch.dtype = torch.float16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """int8 NHWC convolution with the per-channel dequant epilogue.

    Args:
        input: int8 ``[N, H, W, C]`` CUDA tensor, contiguous (``C % 16 == 0``).
        filter: int8 ``[K, R, S, C]`` (``K % 8 == 0``), contiguous.
        scale: fp32 ``[K]``: ``input_scale * weight_scale``.
        bias: Optional ``[K]`` in the output dtype.
        activation: Layer-level activation name (``relu``, ``swish``,
            ``gelu_tanh``, ``gelu`` = exact erf), or ``None``.
        padding, stride, dilation: ``(h, w)`` pairs or one int for both.
        out_dtype: fp16 or bf16, when *out* is not given.
        out: Optional pre-allocated ``[N, P, Q, K]`` output.

    Returns:
        ``[N, P, Q, K]`` in the output dtype.
    """
    from oasr.jit.scaled_conv import scaled_conv2d_func_name, select_config

    padding, stride, dilation = _pair(padding), _pair(stride), _pair(dilation)
    n, h, w, c = input.shape
    k, r, s, _ = filter.shape
    p, q = _output_hw(h, w, r, s, padding, stride, dilation)
    if out is None:
        out = torch.empty((n, p, q, k), dtype=out_dtype, device=input.device)
    # A batch too wide for one launch (2 GiB per tensor) runs as several.
    slices = _row_slices("scaled_conv2d", input, out)
    if slices is not None:
        for rows in slices:
            scaled_conv2d(
                input[rows],
                filter,
                scale,
                bias,
                activation,
                padding,
                stride,
                dilation,
                out=out[rows],
            )
        return out
    cfg = select_config(n * p * q, k, r * s * c)
    fn = getattr(_get_scaled_conv_module(), scaled_conv2d_func_name(cfg))
    fn(
        out,
        input,
        filter,
        scale,
        bias,
        activation_id(activation),
        padding[0],
        padding[1],
        stride[0],
        stride[1],
        dilation[0],
        dilation[1],
    )
    return out


@oasr_api
def scaled_conv1d(
    input: torch.Tensor,
    filter: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
    out_dtype: torch.dtype = torch.float16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """int8 ``(B, T, C)`` convolution over time: :func:`scaled_conv2d` at height one.

    Args:
        input: int8 ``[B, T, C]``; filter: int8 ``[K, S, C]`` (``Conv1d``'s KSC layout).
        The rest as :func:`scaled_conv2d`, along time only.

    Returns:
        ``[B, T_out, K]`` in the output dtype.
    """
    b, t, c = input.shape
    k, width, _ = filter.shape
    t_out = (t + 2 * padding - dilation * (width - 1) - 1) // stride + 1
    if out is None:
        out = torch.empty((b, t_out, k), dtype=out_dtype, device=input.device)
    scaled_conv2d(
        input.view(b, 1, t, c),
        filter.view(k, 1, width, c),
        scale,
        bias,
        activation,
        padding=(0, padding),
        stride=(1, stride),
        dilation=(1, dilation),
        out=out.view(b, 1, t_out, k),
    )
    return out


def reference_scaled_conv2d(
    input: torch.Tensor,
    filter: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    padding: Pair = 0,
    stride: Pair = 1,
    dilation: Pair = 1,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Torch twin of :func:`scaled_conv2d` (any device; the CPU/fp32 path and the oracle).

    The integer convolution runs in fp64, where every sum of int8 products is
    exact; it is rounded to fp32 like the kernel's int32 -> fp32 conversion, then
    ``scale * acc + bias`` once in fp64 (an FMA's single rounding, up to a
    double rounding on ties) and the activation in fp32.
    """
    padding, stride, dilation = _pair(padding), _pair(stride), _pair(dilation)
    acc = F.conv2d(
        input.permute(0, 3, 1, 2).double(),
        filter.permute(0, 3, 1, 2).double(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).permute(0, 2, 3, 1)
    y = acc.float().double() * scale.double().to(acc.device)
    if bias is not None:
        y = y + bias.double().to(acc.device)
    return _TORCH_ACTIVATION[activation](y.float()).to(out_dtype)


def reference_scaled_conv1d(
    input: torch.Tensor,
    filter: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Torch twin of :func:`scaled_conv1d`."""
    b, t, c = input.shape
    k, width, _ = filter.shape
    out = reference_scaled_conv2d(
        input.reshape(b, 1, t, c),
        filter.reshape(k, 1, width, c),
        scale,
        bias,
        activation,
        padding=(0, padding),
        stride=(1, stride),
        dilation=(1, dilation),
        out_dtype=out_dtype,
    )
    return out.reshape(b, out.shape[2], k)


__all__ = [
    "reference_scaled_conv1d",
    "reference_scaled_conv2d",
    "scaled_conv1d",
    "scaled_conv2d",
]
