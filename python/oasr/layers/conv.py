# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Convolution kernel wrappers (PyTorch-style interface).

These classes live under ``oasr.layers`` to mirror structures like
`vllm.model_executor.layers` while providing a thin, torch.nn-like API.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from oasr import ActivationType, ConvType, kernels
from oasr.utils import _torch_dtype_to_oasr

_BACKEND_ERROR = (
    "oasr._C extension not found. Build the project with pip install -e . or set PYTHONPATH."
)


class Conv1d(nn.Module):
    """Wrapper for 1D convolution kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        conv_type: str = "standard",
        is_causal: bool = False,
        channels_last: bool = True,
        activation: str = "SWISH",
        fuse_activation: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv_type = conv_type
        self.is_causal = is_causal
        self.channels_last = channels_last
        self.activation = activation
        self.fuse_activation = fuse_activation

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, device=device, dtype=dtype
        ))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = (in_channels // groups) * kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if kernels is None:
            raise ImportError(_BACKEND_ERROR)
        batch_size, seq_len = x.shape[0], x.shape[1]
        oasr_dtype = _torch_dtype_to_oasr(x.dtype)

        return kernels.conv.conv1d(
            x, self.weight, self.bias,
            batch_size, seq_len,
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride, self.padding,
            self.dilation, self.groups,
            getattr(ConvType, self.conv_type.upper(), ConvType.STANDARD),
            oasr_dtype,
            self.channels_last, self.is_causal,
            getattr(ActivationType, self.activation.upper(), ActivationType.SWISH),
            self.fuse_activation,
        )

    def __call__(self, x):
        return self.forward(x)


class DepthwiseConv1d(nn.Module):
    """Wrapper for depthwise 1D convolution kernel."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        padding: int = 0,
        is_causal: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.is_causal = is_causal

        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(channels, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if kernels is None:
            raise ImportError(_BACKEND_ERROR)
        batch_size, seq_len, _ = x.shape
        return kernels.conv.depthwise_conv1d(
            x, self.weight, self.bias,
            batch_size, seq_len, self.channels,
            self.kernel_size, self.padding,
            _torch_dtype_to_oasr(x.dtype),
        )

    def __call__(self, x):
        return self.forward(x)


class PointwiseConv1d(nn.Module):
    """Wrapper for pointwise (1x1) convolution kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "swish",
        fuse_activation: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.fuse_activation = fuse_activation

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if kernels is None:
            raise ImportError(_BACKEND_ERROR)
        batch_size, seq_len, _ = x.shape
        return kernels.conv.pointwise_conv1d(
            x, self.weight, self.bias,
            batch_size, seq_len,
            self.in_channels, self.out_channels,
            getattr(ActivationType, self.activation.upper(), ActivationType.SWISH),
            self.fuse_activation,
            _torch_dtype_to_oasr(x.dtype),
        )

    def __call__(self, x):
        return self.forward(x)


__all__ = ["Conv1d", "DepthwiseConv1d", "PointwiseConv1d"]
