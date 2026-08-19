# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for pooling operations."""

from __future__ import annotations

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api


def _single_int(value: int | tuple[int], name: str) -> int:
    """Normalize a PyTorch-style one-dimensional scalar/1-tuple."""
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"{name} must be an int or a one-element tuple, got {value!r}")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int or a one-element tuple, got {value!r}")
    return value


def _avg_pool1d_output_length(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
) -> int:
    numerator = input_length + 2 * padding - kernel_size
    if ceil_mode:
        numerator += stride - 1
    output_length = numerator // stride + 1
    if ceil_mode and output_length > 0 and (output_length - 1) * stride >= input_length + padding:
        output_length -= 1
    return output_length


@functools.cache
def _get_pooling_module():
    from oasr.jit.pooling import gen_pooling_module

    return gen_pooling_module().build_and_load()


@oasr_api
def avg_pool1d(
    input: torch.Tensor,
    kernel_size: int | tuple[int],
    stride: Optional[int | tuple[int]] = None,
    padding: int | tuple[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average pool a contiguous TC or BTC tensor along its time dimension.

    Unlike :func:`torch.nn.functional.avg_pool1d`, the native layout is
    ``(batch, time, channels)`` so encoder activations do not need BCT
    transposes around the operation.  Padding is symmetric, matching PyTorch.
    FP16, BF16, and FP32 CUDA tensors are supported by the functional API.
    """
    kernel = _single_int(kernel_size, "kernel_size")
    step = kernel if stride is None else _single_int(stride, "stride")
    pad = _single_int(padding, "padding")
    if kernel <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel}")
    if step <= 0:
        raise ValueError(f"stride must be positive, got {step}")
    if pad < 0 or pad > kernel // 2:
        raise ValueError(
            f"padding must be non-negative and at most half of kernel_size, got "
            f"padding={pad}, kernel_size={kernel}"
        )
    if input.dim() not in (2, 3):
        raise ValueError(f"avg_pool1d expects TC or BTC input, got shape {tuple(input.shape)}")

    input_length = input.shape[-2]
    output_length = _avg_pool1d_output_length(input_length, kernel, step, pad, ceil_mode)
    if output_length <= 0:
        raise ValueError(
            f"avg_pool1d produces an invalid output length {output_length} from "
            f"T={input_length}, kernel_size={kernel}, stride={step}, padding={pad}"
        )
    output_shape = input.shape[:-2] + (output_length, input.shape[-1])
    if out is None:
        out = input.new_empty(output_shape)

    _get_pooling_module().avg_pool1d(
        out, input, kernel, step, pad, bool(ceil_mode), bool(count_include_pad)
    )
    return out
