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

import oasr
from oasr.utils import get_activation_type


class DepthwiseConv1d(nn.Module):
    """Wrapper for depthwise 1D convolution kernel."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(
            kernel_size, 1, channels, device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(
                channels, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.kernels.conv.depthwise_conv1d(x, self.weight, self.bias, self.padding)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Support loading WeNet Conv1d depthwise weights.

        WeNet depthwise conv uses PyTorch Conv1d layout [C, 1, K].
        OASR depthwise kernel expects [K, 1, C], so transpose on load.
        """
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            w = state_dict[weight_key]
            # Detect WeNet-style layout and convert to OASR layout.
            if (
                isinstance(w, torch.Tensor)
                and w.ndim == 3
                and w.shape[1] == 1
                and w.shape[0] == self.channels
                and w.shape[2] == self.kernel_size
                and w.shape != self.weight.shape
            ):
                state_dict[weight_key] = w.permute(2, 1, 0).contiguous()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class PointwiseConv1d(nn.Module):
    """Wrapper for pointwise (1x1) convolution kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str | None = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = (
            None if activation_type is None
            else get_activation_type(activation_type)
        )

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, 1, device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(
                out_channels, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation is not None:
            return oasr.kernels.conv.pointwise_conv1d_activation(x, self.weight, self.bias, self.activation)
        else:
            return oasr.kernels.conv.pointwise_conv1d(x, self.weight, self.bias)


__all__ = ["DepthwiseConv1d", "PointwiseConv1d"]
