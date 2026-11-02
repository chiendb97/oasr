# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Parameter-free activation modules at the layer waist."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import oasr

from ._backend import use_conv_kernel


class Gelu(nn.Module):
    """Exact-erf GELU on the OASR kernel for served CUDA dtypes.

    CPU/fp32 and an explicitly selected torch backend use :func:`F.gelu` with
    its default exact-erf formulation. The distinction from ``gelu_tanh`` is
    intentional: Whisper and Qwen2-Audio checkpoints were trained with this
    form.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_conv_kernel(x):
            activated: torch.Tensor = oasr.gelu(x)
            return activated
        return F.gelu(x)


class Sigmoid(nn.Module):
    """Elementwise sigmoid on the OASR kernel for served CUDA dtypes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_conv_kernel(x):
            activated: torch.Tensor = oasr.sigmoid(x)
            return activated
        return torch.sigmoid(x)


class Tanh(nn.Module):
    """Elementwise tanh on the OASR kernel for served CUDA dtypes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_conv_kernel(x):
            activated: torch.Tensor = oasr.tanh(x)
            return activated
        return torch.tanh(x)


class Relu(nn.Module):
    """Elementwise ReLU on the OASR kernel for served CUDA dtypes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_conv_kernel(x):
            activated: torch.Tensor = oasr.relu(x)
            return activated
        return F.relu(x)


def swoosh_l(x: torch.Tensor) -> torch.Tensor:
    """Zipformer's SwooshL, ``log(1 + exp(x - 4)) - 0.08 x - 0.035``, routed.

    The function form of :class:`SwooshL`, for a caller that applies it inline
    (Zipformer's fused activation + linear) rather than as a child module.  The
    torch path evaluates in fp32 and rounds once, as the kernel does
    (``include/oasr/common/math.h``).
    """
    if use_conv_kernel(x):
        activated: torch.Tensor = oasr.swoosh_l(x)
        return activated
    xf = x.float()
    return (F.softplus(xf - 4.0) - 0.08 * xf - 0.035).to(x.dtype)


def swoosh_r(x: torch.Tensor) -> torch.Tensor:
    """Zipformer's SwooshR, ``log(1 + exp(x - 1)) - 0.08 x - 0.313261687``, routed."""
    if use_conv_kernel(x):
        activated: torch.Tensor = oasr.swoosh_r(x)
        return activated
    xf = x.float()
    return (F.softplus(xf - 1.0) - 0.08 * xf - 0.313261687).to(x.dtype)


class SwooshL(nn.Module):
    """Zipformer's SwooshL on the OASR kernel for served CUDA dtypes, torch elsewhere.

    It lived in ``oasr/models/zipformer/scaling.py`` and called the kernel
    unconditionally, so it raised on a CPU tensor where every other activation
    here routes to torch.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swoosh_l(x)


class SwooshR(nn.Module):
    """Zipformer's SwooshR; see :class:`SwooshL`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swoosh_r(x)


__all__ = ["Gelu", "Relu", "Sigmoid", "SwooshL", "SwooshR", "Tanh", "swoosh_l", "swoosh_r"]
