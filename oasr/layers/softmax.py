# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Softmax layer wrapper (PyTorch-style interface)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import oasr


class Softmax(nn.Module):
    """Wrapper for softmax kernel (operates on the last dimension)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Annotated rather than returned directly: ``@oasr_api`` is an untyped
        # decorator, so every functional it wraps is ``Any`` to mypy.
        out: torch.Tensor = oasr.softmax(x)
        return out


class MaskedSoftmax(nn.Module):
    """Fused additive-bias + boolean-mask softmax over the last dimension.

    The waist member for the op sequence a relative-position attention runs:
    add the positional bias, floor the masked keys, softmax.  ``bias`` and both
    masks are read through their own strides, so a shifted or step-sliced view
    is consumed where it is, not copied.
    """

    def __init__(self, mask_value: float = -1000.0) -> None:
        super().__init__()
        self.mask_value = mask_value

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out: torch.Tensor = oasr.masked_softmax(
            x, bias=bias, mask=mask, mask2=mask2, mask_value=self.mask_value
        )
        return out
