# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Linear layer (GEMM kernel wrapper, PyTorch-style interface)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import oasr


class Linear(nn.Module):
    """Linear layer: y = x @ weight + bias, backed by a GEMM kernel."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            self.bias = nn.Parameter(torch.empty(
                out_features, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (*, in_features) -> (*, out_features)."""

        return oasr.kernels.gemm.gemm(x, self.weight, self.bias)


__all__ = ["Linear"]
