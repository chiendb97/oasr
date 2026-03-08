# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Normalization kernel wrappers (PyTorch-style interface)."""

from __future__ import annotations

import torch
import torch.nn as nn

import oasr


class LayerNorm(nn.Module):
    """Wrapper for layer normalization kernel."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.kernels.norm.layer_norm(x, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """Wrapper for RMS normalization kernel."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.kernels.norm.rms_norm(x, self.weight, self.bias, self.eps)


class GroupNorm(nn.Module):
    """Wrapper for group normalization kernel."""

    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.kernels.norm.group_norm(x, self.weight, self.bias, self.num_groups, self.eps)


class BatchNorm1d(nn.Module):
    """Wrapper for 1D batch normalization kernel (inference).

    running_mean and running_var are registered as buffers (non-trainable).
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))
        else:
            self.bias = None
        self.register_buffer("running_mean", torch.zeros(num_channels, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(num_channels, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.kernels.norm.batch_norm_1d(x, self.weight, self.bias, self.running_mean, self.running_var, self.eps)


class AddLayerNorm(nn.Module):
    """Wrapper for fused add + layer norm: output = LayerNorm(x + residual)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        return oasr.kernels.norm.add_layer_norm(x, residual, self.weight, self.bias, self.eps)

__all__ = ["LayerNorm", "RMSNorm", "GroupNorm", "BatchNorm1d", "AddLayerNorm"]
