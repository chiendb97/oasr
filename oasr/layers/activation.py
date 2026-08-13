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


__all__ = ["Gelu"]
