# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Softmax layer wrapper (PyTorch-style interface)."""

from __future__ import annotations

import torch
import torch.nn as nn

import oasr


class Softmax(nn.Module):
    """Wrapper for softmax kernel (operates on the last dimension)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return oasr.softmax(x)
