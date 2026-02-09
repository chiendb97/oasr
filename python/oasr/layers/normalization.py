# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Normalization kernel wrappers (PyTorch-style interface)."""

from __future__ import annotations

from oasr import kernels
from oasr.utils import _torch_dtype_to_oasr


class LayerNorm:
    """Wrapper for layer normalization kernel."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(
        self,
        x: "torch.Tensor",
        gamma: "torch.Tensor",
        beta: "torch.Tensor",
    ) -> "torch.Tensor":
        batch_size, seq_len, hidden_size = x.shape
        output = x.new_empty(x.shape)
        kernels.normalization.layer_norm(
            x.data_ptr(),
            output.data_ptr(),
            gamma.data_ptr(),
            beta.data_ptr(),
            batch_size,
            seq_len,
            hidden_size,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )
        return output

    def __call__(self, x, gamma, beta):
        return self.forward(x, gamma, beta)


class RMSNorm:
    """Wrapper for RMS normalization kernel."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(
        self,
        x: "torch.Tensor",
        gamma: "torch.Tensor",
    ) -> "torch.Tensor":
        batch_size, seq_len, hidden_size = x.shape
        output = x.new_empty(x.shape)
        kernels.normalization.rms_norm(
            x.data_ptr(),
            output.data_ptr(),
            gamma.data_ptr(),
            batch_size,
            seq_len,
            hidden_size,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )
        return output

    def __call__(self, x, gamma):
        return self.forward(x, gamma)


class GroupNorm:
    """Wrapper for group normalization kernel."""

    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5):
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps

    def forward(
        self,
        x: "torch.Tensor",
        gamma: "torch.Tensor",
        beta: "torch.Tensor",
    ) -> "torch.Tensor":
        batch_size, seq_len, channels = x.shape
        output = x.new_empty(x.shape)
        kernels.normalization.group_norm(
            x.data_ptr(),
            output.data_ptr(),
            gamma.data_ptr(),
            beta.data_ptr(),
            batch_size,
            seq_len,
            channels,
            self.num_groups,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )
        return output

    def __call__(self, x, gamma, beta):
        return self.forward(x, gamma, beta)


class BatchNorm1d:
    """Wrapper for 1D batch normalization kernel (inference)."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        self.num_channels = num_channels
        self.eps = eps

    def forward(
        self,
        x: "torch.Tensor",
        gamma: "torch.Tensor",
        beta: "torch.Tensor",
        running_mean: "torch.Tensor",
        running_var: "torch.Tensor",
    ) -> "torch.Tensor":
        batch_size, seq_len, channels = x.shape
        output = x.new_empty(x.shape)
        kernels.normalization.batch_norm_1d(
            x.data_ptr(),
            output.data_ptr(),
            gamma.data_ptr(),
            beta.data_ptr(),
            running_mean.data_ptr(),
            running_var.data_ptr(),
            batch_size,
            seq_len,
            channels,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )
        return output

    def __call__(self, x, gamma, beta, running_mean, running_var):
        return self.forward(x, gamma, beta, running_mean, running_var)


class AddLayerNorm:
    """Wrapper for fused add + layer norm: output = LayerNorm(x + residual)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(
        self,
        x: "torch.Tensor",
        residual: "torch.Tensor",
        gamma: "torch.Tensor",
        beta: "torch.Tensor",
    ) -> "torch.Tensor":
        batch_size, seq_len, hidden_size = x.shape
        output = x.new_empty(x.shape)
        kernels.normalization.add_layer_norm(
            x.data_ptr(),
            residual.data_ptr(),
            output.data_ptr(),
            gamma.data_ptr(),
            beta.data_ptr(),
            batch_size,
            seq_len,
            hidden_size,
            self.eps,
            _torch_dtype_to_oasr(x.dtype),
        )
        return output

    def __call__(self, x, residual, gamma, beta):
        return self.forward(x, residual, gamma, beta)


__all__ = ["LayerNorm", "RMSNorm", "GroupNorm", "BatchNorm1d", "AddLayerNorm"]

