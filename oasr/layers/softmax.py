# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Softmax layer wrapper (PyTorch-style interface)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import oasr

from ._backend import use_masked_softmax_kernel, use_softmax_kernel


def masked_softmax(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
    mask_value: float = -1000.0,
) -> torch.Tensor:
    """:func:`oasr.masked_softmax`, routed: the kernel for CUDA fp16/bf16/fp32,
    the same function in torch elsewhere.

    The function form of :class:`MaskedSoftmax`, for model code that calls it
    from a free function.  The torch path is the kernel's documented
    definition: the biased score is rounded back to ``x``'s dtype before the
    mask, and a masked score becomes ``mask_value``.
    """
    if use_masked_softmax_kernel(x):
        out: torch.Tensor = oasr.masked_softmax(
            x, bias=bias, mask=mask, mask2=mask2, mask_value=mask_value
        )
        return out
    scores = x if bias is None else (x + bias).to(x.dtype)
    for m in (mask, mask2):
        if m is not None:
            scores = scores.masked_fill(m, mask_value)
    return torch.softmax(scores, dim=-1)


class Softmax(nn.Module):
    """Softmax over the last dimension.

    Owns both paths, like every other member of the waist.  It used to call the
    kernel unconditionally, which made it the one layer that *raised* on a CPU
    tensor instead of routing to torch — and CPU is an out-of-scope input, not
    a kernel gap, so raising there contradicts the backend contract.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not use_softmax_kernel(x):
            return torch.softmax(x, dim=-1)
        # Annotated rather than returned directly: ``@oasr_api`` is an untyped
        # decorator, so every functional it wraps is ``Any`` to mypy.
        out: torch.Tensor = oasr.softmax(x)
        return out


class MaskedSoftmax(nn.Module):
    """Fused additive-bias + boolean-mask softmax over the last dimension.

    The waist member for the op sequence a relative-position attention runs:
    add the positional bias, floor the masked keys, softmax.  ``bias`` and both
    masks are read through their own strides, so a shifted or step-sliced view
    is consumed where it is, not copied.  Off the kernel's scope (CPU, the
    torch backend) it runs the same function in torch; it used to call the
    kernel unconditionally and raise there.
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
        return masked_softmax(x, bias=bias, mask=mask, mask2=mask2, mask_value=self.mask_value)
