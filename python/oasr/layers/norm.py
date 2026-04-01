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
        self.weight = nn.Parameter(torch.ones(
            normalized_shape, device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(
                normalized_shape, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.layer_norm(x, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """Wrapper for RMS normalization kernel."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(
            normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                normalized_shape, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.rms_norm(x, self.weight, self.bias, self.eps)


class GroupNorm(nn.Module):
    """Wrapper for group normalization kernel."""

    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(
            num_channels, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                num_channels, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.group_norm(x, self.weight, self.bias, self.num_groups, self.eps)


class BatchNorm1d(nn.Module):
    """Wrapper for 1D batch normalization kernel (inference).

    running_mean and running_var are registered as buffers (non-trainable).
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(
            num_channels, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                num_channels, device=device, dtype=dtype))
        else:
            self.bias = None
        self.register_buffer("running_mean", torch.zeros(
            num_channels, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(
            num_channels, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.batch_norm_1d(x, self.weight, self.bias, self.running_mean, self.running_var, self.eps)


class BatchNormSwish(nn.Module):
    """Wrapper for fused BatchNorm + Swish kernel."""

    def __init__(self, num_channels: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(
            num_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(
            num_channels, device=device, dtype=dtype))
        self.register_buffer("running_mean", torch.zeros(
            num_channels, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(
            num_channels, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.batch_norm_swish(x, self.weight, self.bias, self.running_mean, self.running_var, self.eps)


class AddLayerNorm(nn.Module):
    """Wrapper for fused add + layer norm: output = LayerNorm(x + residual)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(
            normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                normalized_shape, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        return oasr.add_layer_norm(x, residual, self.weight, self.bias, self.eps)


class LayerNormActivation(nn.Module):
    """Fused LayerNorm + Activation: output = activation(LayerNorm(x))."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
        activation: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.activation_type = oasr.get_activation_type_id(activation)
        self.weight = nn.Parameter(torch.ones(
            normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                normalized_shape, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.layer_norm_activation(
            x, self.weight, self.bias, self.eps, self.activation_type)


class RMSNormActivation(nn.Module):
    """Fused RMSNorm + Activation: output = activation(RMSNorm(x))."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
        activation: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.activation_type = oasr.get_activation_type_id(activation)
        self.weight = nn.Parameter(torch.ones(
            normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                normalized_shape, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.rms_norm_activation(
            x, self.weight, self.bias, self.eps, self.activation_type)


class BatchNormActivation(nn.Module):
    """Fused BatchNorm + Activation: output = activation(BatchNorm(x))."""

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        activation: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.activation_type = oasr.get_activation_type_id(activation)
        self.weight = nn.Parameter(torch.ones(
            num_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(
            num_channels, device=device, dtype=dtype))
        self.register_buffer("running_mean", torch.zeros(
            num_channels, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(
            num_channels, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.batch_norm_activation(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.eps, self.activation_type)


class GlobalCMVN(nn.Module):
    """Global cepstral mean and variance normalization.

    Stores pre-computed mean and inverse std-dev as buffers
    and applies ``oasr.cmvn(x, mean, istd)`` to input features.
    """

    def __init__(self, mean: torch.Tensor, istd: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.cmvn(x, self.mean, self.istd)


__all__ = [
    "LayerNorm", "RMSNorm", "GroupNorm",
    "BatchNorm1d", "BatchNormSwish", "AddLayerNorm",
    "LayerNormActivation", "RMSNormActivation", "BatchNormActivation",
    "GlobalCMVN",
]
