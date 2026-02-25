# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Normalization kernel wrappers (PyTorch-style interface)."""

from __future__ import annotations

import torch

from oasr import kernels
from oasr.utils import _torch_dtype_to_oasr


class LayerNorm:
    """Wrapper for layer normalization kernel."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.ones(normalized_shape, device=device, dtype=dtype)

        if bias:
            self.bias = torch.zeros(normalized_shape, device=device, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        batch_size, seq_len, hidden_size = x.shape
        return kernels.norm.layer_norm(
            x,
            self.weight, self.bias,
            batch_size, seq_len, hidden_size,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )

    def __call__(self, x):
        return self.forward(x)


class RMSNorm:
    """Wrapper for RMS normalization kernel."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.ones(normalized_shape, device=device, dtype=dtype)
        if bias:
            self.bias = torch.zeros(normalized_shape, device=device, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        batch_size, seq_len, hidden_size = x.shape
        return kernels.norm.rms_norm(
            x,
            self.weight, self.bias,
            batch_size, seq_len, hidden_size,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )

    def __call__(self, x):
        return self.forward(x)


class GroupNorm:
    """Wrapper for group normalization kernel."""

    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = torch.ones(num_channels, device=device, dtype=dtype)
        if bias:
            self.bias = torch.zeros(num_channels, device=device, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        batch_size, seq_len, channels = x.shape
        return kernels.norm.group_norm(
            x,
            self.weight, self.bias,
            batch_size, seq_len, channels, self.num_groups,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )

    def __call__(self, x):
        return self.forward(x)


class BatchNorm1d:
    """Wrapper for 1D batch normalization kernel (inference)."""

    def __init__(self, num_channels: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        self.num_channels = num_channels
        self.eps = eps
        self.weight = torch.ones(num_channels, device=device, dtype=dtype)
        if bias:
            self.bias = torch.zeros(num_channels, device=device, dtype=dtype)
        else:
            self.bias = None
        self.running_mean = torch.zeros(num_channels, device=device, dtype=dtype)
        self.running_var = torch.ones(num_channels, device=device, dtype=dtype)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        batch_size, seq_len, channels = x.shape
        return kernels.norm.batch_norm_1d(
            x,
            self.weight, self.bias,
            self.running_mean, self.running_var,
            batch_size, seq_len, channels,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )

    def __call__(self, x):
        return self.forward(x)


class AddLayerNorm:
    """Wrapper for fused add + layer norm: output = LayerNorm(x + residual)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.ones(normalized_shape, device=device, dtype=dtype)
        if bias:
            self.bias = torch.zeros(normalized_shape, device=device, dtype=dtype)
        else:
            self.bias = None

    def forward(
        self,
        x: "torch.Tensor",
        residual: "torch.Tensor",
    ) -> "torch.Tensor":
        batch_size, seq_len, hidden_size = x.shape
        return kernels.norm.add_layer_norm(
            x, residual,
            self.weight, self.bias,
            batch_size, seq_len, hidden_size,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )

    def __call__(self, x, residual):
        return self.forward(x, residual)


__all__ = ["LayerNorm", "RMSNorm", "GroupNorm", "BatchNorm1d", "AddLayerNorm"]
