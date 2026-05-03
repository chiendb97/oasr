# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""TopK layer wrapper (PyTorch-style interface)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import oasr


class TopK(nn.Module):
    """Wrapper for top-k kernel (operates on the last dimension).

    Args:
        k: Number of top elements to return per row.
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return oasr.topk(x, self.k)
