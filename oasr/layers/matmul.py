# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The batched matrix product at the layer waist."""

from __future__ import annotations

from typing import Optional

import torch

import oasr

from ._backend import use_bmm_kernel


def bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """:func:`oasr.bmm` (``A @ B.transpose(-1, -2)``, batch axes broadcast), routed.

    The kernel for CUDA fp16/bf16; ``torch.matmul`` for CPU, fp32 and the torch
    backend, which the kernel does not serve.  A function rather than a module
    because its callers (Zipformer's attention products) are host-issue-bound
    and write through an ``out`` view.  ``out``, when given, is written in place
    on either path, so a strided view keeps its layout.
    """
    if use_bmm_kernel(A):
        result: torch.Tensor = oasr.bmm(A, B, out=out)
        return result
    product = torch.matmul(A, B.transpose(-1, -2))
    if out is None:
        return product
    return out.copy_(product)


__all__ = ["bmm"]
