# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CTC projection layer: fused (Linear -> log_softmax)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import oasr


class CtcProjection(nn.Module):
    """Fused CTC projection: ``log_softmax(x @ weight.T + bias)``.

    Mirrors :class:`~oasr.layers.linear.Linear` (holds ``weight`` / ``bias``
    parameters directly) but fuses the trailing vocab ``log_softmax`` via
    ``oasr.gemm_log_softmax``.  Pure compute — it carries no decode metadata;
    :class:`~oasr.models.heads.ctc.CTCHead` wraps it to add ``decode_type`` and
    the engine-facing ``BaseHead`` contract.
    """

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

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(*, in_features)`` → ``log_softmax`` over ``(*, out_features)``."""

        return oasr.gemm_log_softmax(x, self.weight, self.bias)


__all__ = ["CtcProjection"]
