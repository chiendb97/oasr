# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Linear layer (GEMM kernel wrapper, PyTorch-style interface)."""

from __future__ import annotations

from typing import Optional

from oasr import kernels
from oasr.utils import _torch_dtype_to_oasr


class Linear:
    """Wrapper for GEMM kernel: D = alpha * A @ B + beta * C."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        A: "torch.Tensor",
        B: "torch.Tensor",
        C: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """D = alpha * A @ B + beta * C. A: (M, K), B: (K, N), C optional (M, N)."""
        import torch

        M, K = A.shape
        _, N = B.shape
        D = torch.empty(M, N, device=A.device, dtype=A.dtype)
        oasr_dtype = _torch_dtype_to_oasr(A.dtype)

        params = kernels.gemm.GemmParams()
        params.A = A.data_ptr()
        params.B = B.data_ptr()
        params.D = D.data_ptr()
        params.M, params.N, params.K = M, N, K
        params.lda, params.ldb, params.ldd = K, N, N
        params.alpha = self.alpha
        params.beta = self.beta
        params.dtype_a = params.dtype_b = params.dtype_d = oasr_dtype
        if C is not None:
            params.C = C.data_ptr()
            params.ldc = N
            params.dtype_c = oasr_dtype
        else:
            params.C = 0
            params.ldc = N

        status = kernels.gemm.invoke_gemm(params)
        if status != kernels.gemm.GemmStatus.SUCCESS:
            raise RuntimeError(f"GEMM failed: {kernels.gemm.get_gemm_status_string(status)}")
        return D

    def __call__(self, A, B, C=None):
        return self.forward(A, B, C)


__all__ = ["Linear"]
