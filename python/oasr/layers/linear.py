# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Linear layer (GEMM kernel wrapper, PyTorch-style interface)."""

from __future__ import annotations

import math

import torch

from oasr import kernels
from oasr.utils import _torch_dtype_to_oasr


class Linear:
    """Linear layer: y = x @ weight + bias, backed by a GEMM kernel."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.empty(in_features, out_features, device=device, dtype=dtype)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            self.bias = torch.empty(out_features, device=device, dtype=dtype)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """x: (*, in_features) -> (*, out_features)."""
        input_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        M, K = x_2d.shape
        N = self.out_features
        oasr_dtype = _torch_dtype_to_oasr(x.dtype)

        D, status = kernels.gemm.invoke_gemm(
            x_2d, self.weight,
            M, N, K, K, N, N,
            1.0, 0.0,
            kernels.gemm.TransposeOp.NoTranspose,
            kernels.gemm.TransposeOp.NoTranspose,
            oasr_dtype,
        )
        if status != kernels.gemm.GemmStatus.SUCCESS:
            raise RuntimeError(f"GEMM failed: {kernels.gemm.get_gemm_status_string(status)}")

        if self.bias is not None:
            D += self.bias

        return D.reshape(*input_shape[:-1], N)

    def __call__(self, x):
        return self.forward(x)


__all__ = ["Linear"]
