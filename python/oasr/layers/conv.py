# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Convolution kernel wrappers (PyTorch-style interface).

These classes live under ``oasr.layers`` to mirror structures like
`vllm.model_executor.layers` while providing a thin, torch.nn-like API.
"""

from __future__ import annotations

from typing import Optional

from oasr import ActivationType, ConvType, kernels
from oasr.utils import _torch_dtype_to_oasr

_BACKEND_ERROR = (
    "oasr._C extension not found. Build the project with pip install -e . or set PYTHONPATH."
)


class Conv1d:
    """
    Wrapper for 1D convolution kernel.

    Usage:
        conv = Conv1d(in_channels, out_channels, kernel_size, ...)
        y = conv(x, weight, bias)
    """

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
    ):
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

    def forward(
        self,
        x: "torch.Tensor",
        weight: "torch.Tensor",
        bias: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch

        if kernels is None:
            raise ImportError(_BACKEND_ERROR)
        batch_size, seq_len = x.shape[0], x.shape[1]
        oasr_dtype = _torch_dtype_to_oasr(x.dtype)
        bias_ptr = bias.data_ptr() if bias is not None else 0

        out_seq = (
            seq_len + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1
        if self.channels_last:
            output = torch.empty(
                batch_size, out_seq, self.out_channels, device=x.device, dtype=x.dtype
            )
        else:
            output = torch.empty(
                batch_size, self.out_channels, out_seq, device=x.device, dtype=x.dtype
            )

        kernels.conv.conv1d(
            x.data_ptr(),
            weight.data_ptr(),
            bias_ptr,
            output.data_ptr(),
            batch_size,
            seq_len,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            getattr(ConvType, self.conv_type.upper(), ConvType.STANDARD),
            oasr_dtype,
            self.channels_last,
            self.is_causal,
            getattr(ActivationType, self.activation.upper(), ActivationType.SWISH),
            self.fuse_activation,
        )
        return output

    def __call__(self, x, weight, bias=None):
        return self.forward(x, weight, bias)


class DepthwiseConv1d:
    """Wrapper for depthwise 1D convolution kernel."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        padding: int = 0,
        is_causal: bool = False,
    ):
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.is_causal = is_causal

    def forward(
        self,
        x: "torch.Tensor",
        weight: "torch.Tensor",
        bias: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch

        if kernels is None:
            raise ImportError(_BACKEND_ERROR)
        batch_size, seq_len, _ = x.shape
        output = torch.empty_like(x)
        bias_ptr = bias.data_ptr() if bias is not None else 0
        kernels.conv.depthwise_conv1d(
            x.data_ptr(),
            weight.data_ptr(),
            bias_ptr,
            output.data_ptr(),
            batch_size,
            seq_len,
            self.channels,
            self.kernel_size,
            self.padding,
            self.is_causal,
            _torch_dtype_to_oasr(x.dtype),
        )
        return output

    def __call__(self, x, weight, bias=None):
        return self.forward(x, weight, bias)


class PointwiseConv1d:
    """Wrapper for pointwise (1x1) convolution kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "SWISH",
        fuse_activation: bool = False,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.fuse_activation = fuse_activation

    def forward(
        self,
        x: "torch.Tensor",
        weight: "torch.Tensor",
        bias: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch

        if kernels is None:
            raise ImportError(_BACKEND_ERROR)
        batch_size, seq_len, _ = x.shape
        output = torch.empty(
            batch_size, seq_len, self.out_channels, device=x.device, dtype=x.dtype
        )
        bias_ptr = bias.data_ptr() if bias is not None else 0
        kernels.conv.pointwise_conv1d(
            x.data_ptr(),
            weight.data_ptr(),
            bias_ptr,
            output.data_ptr(),
            batch_size,
            seq_len,
            self.in_channels,
            self.out_channels,
            getattr(ActivationType, self.activation.upper(), ActivationType.SWISH),
            self.fuse_activation,
            _torch_dtype_to_oasr(x.dtype),
        )
        return output

    def __call__(self, x, weight, bias=None):
        return self.forward(x, weight, bias)


__all__ = ["Conv1d", "DepthwiseConv1d", "PointwiseConv1d"]
