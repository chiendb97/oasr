# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Pooling layers in the encoder's native time-major-within-batch layout."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

import oasr

from ._backend import use_pooling_kernel


def _single_int(value: int | tuple[int], name: str) -> int:
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"{name} must be an int or a one-element tuple, got {value!r}")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int or a one-element tuple, got {value!r}")
    return value


class AvgPool1d(nn.Module):
    """Average pooling over contiguous TC/BTC activations.

    The public attributes and pooling semantics match :class:`torch.nn.AvgPool1d`,
    but the tensor layout is ``(time, channels)`` or ``(batch, time, channels)``.
    This is the same BTC convention as :class:`oasr.layers.Conv1d`.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: Optional[int | tuple[int]] = None,
        padding: int | tuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
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
        self.kernel_size = kernel
        self.stride = step
        self.padding = pad
        self.ceil_mode = bool(ceil_mode)
        self.count_include_pad = bool(count_include_pad)

    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool1d(
            x.transpose(-2, -1),
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
        return pooled.transpose(-2, -1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not use_pooling_kernel(x):
            return self._torch_forward(x)
        out: torch.Tensor = oasr.avg_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
        return out

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, ceil_mode={self.ceil_mode}, "
            f"count_include_pad={self.count_include_pad}"
        )
