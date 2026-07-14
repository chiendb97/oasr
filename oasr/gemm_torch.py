# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Torch/cuBLAS GEMM runners — a backend candidate and a production dispatch target.

These mirror the OASR CUTLASS GEMM launcher contract exactly so they can be used
interchangeably by the autotuner (as a ``Tactic("torch")`` candidate) and by the
shape-aware production selector in :mod:`oasr.gemm`:

  * output tensor first, written **in place** (no new allocations — CUDA-graph safe);
  * ``B`` is the ``[N, K]`` weight (transB), so ``D = A @ Bᵀ`` (== ``F.linear``);
  * bias ``C`` is an optional ``[N]`` vector broadcast over rows;
  * ``split_k_slices`` is accepted for signature parity and ignored (cuBLAS picks
    its own internal split).

Activation IDs match :mod:`oasr.activation`: RELU=0, GELU=1, SWISH=2 (SiLU).
GELU uses the erf form to match the CUTLASS ``LinearCombinationGELU`` epilogue
(``oasr/common/epilogue_functors.h``); SWISH matches ``LinearCombinationSilu``.

Kept deliberately free of any ``oasr.tune`` import so the production GEMM path can
import it without pulling in the autotuner.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_gemm(out, A, B, C=None, split_k_slices: int = 1) -> None:
    """``out[M,N] = A[M,K] @ B[N,K]ᵀ (+ C[N])`` — written in place into ``out``.

    With a bias, ``torch.addmm`` fuses ``C + A@Bᵀ`` into a single cuBLAS GEMM
    (the bias ``[N]`` broadcasts over rows) instead of a separate matmul + add —
    one fewer kernel launch, and CUDA-graph safe.
    """
    if C is None:
        torch.matmul(A, B.t(), out=out)
    else:
        torch.addmm(C, A, B.t(), out=out)


def torch_gemm_activation(out, A, B, C, activation_type: int, split_k_slices: int = 1) -> None:
    """``out = act(A @ Bᵀ + C)`` — written in place into ``out`` (bias fused via addmm)."""
    if C is None:
        torch.matmul(A, B.t(), out=out)
    else:
        torch.addmm(C, A, B.t(), out=out)
    if activation_type == 0:  # RELU
        torch.relu_(out)
    elif activation_type == 1:  # GELU (erf form, matches LinearCombinationGELU)
        out.copy_(F.gelu(out))
    elif activation_type == 2:  # SWISH / SiLU (matches LinearCombinationSilu)
        F.silu(out, inplace=True)
    else:
        raise ValueError(f"Unsupported activation_type: {activation_type}")


def torch_bmm(out, A, B) -> None:
    """``out[b,M,N] = A[b,M,K] @ B[b,N,K]ᵀ`` — written in place into ``out``."""
    torch.bmm(A, B.transpose(-1, -2), out=out)


def torch_gemm_log_softmax(out, A, B, C=None, split_k_slices: int = 1) -> None:
    """``out = log_softmax(A @ Bᵀ (+ C), dim=-1)`` — written in place into ``out``.

    The all-torch counterpart of the fused CUTLASS ``gemm_log_softmax`` (the
    CTC head).  ``torch.log_softmax`` supports ``out is input`` (element-wise
    final pass), so no intermediate buffer is allocated.
    """
    if C is None:
        torch.matmul(A, B.t(), out=out)
    else:
        torch.addmm(C, A, B.t(), out=out)
    torch.log_softmax(out, dim=-1, out=out)
