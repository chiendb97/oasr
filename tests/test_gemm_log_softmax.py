#!/usr/bin/env python3
"""Unit tests for the fused GEMM + log_softmax API (oasr.gemm_log_softmax)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import oasr


def _reference(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Reference: F.log_softmax(F.linear(A, B, bias), dim=-1) in fp32."""
    out = torch.matmul(A.float(), B.float().T)
    if bias is not None:
        out = out + bias.float()
    return F.log_softmax(out, dim=-1).to(A.dtype)


class TestGemmLogSoftmax:
    @pytest.mark.parametrize(
        "M,N,K",
        [
            (64, 128, 256),     # power-of-two
            (32, 1024, 128),    # wider vocab
            (16, 2048, 256),    # vocab > tile_N (forces multi-tile rows)
            (8, 5000, 512),     # CTC-like
            (4, 40, 64),        # small N — CUTLASS still requires 8-alignment
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_torch(self, M, N, K, dtype):
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
        B = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
        bias = torch.randn(N, device="cuda", dtype=dtype) * 0.1

        out = oasr.gemm_log_softmax(A, B, bias)
        expected = _reference(A, B, bias)

        # log_softmax has tighter requirements than softmax; allow slightly looser
        # tolerance for FP16/BF16 accumulation.
        rtol = 5e-2 if dtype == torch.bfloat16 else 5e-3
        atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_no_bias(self, dtype):
        torch.manual_seed(1)
        A = torch.randn(16, 128, device="cuda", dtype=dtype) * 0.1
        B = torch.randn(256, 128, device="cuda", dtype=dtype) * 0.1

        out = oasr.gemm_log_softmax(A, B, None)
        expected = _reference(A, B, None)

        torch.testing.assert_close(out, expected, rtol=5e-3, atol=1e-2)

    def test_3d_leading_dims(self):
        """A may have arbitrary leading dims (B, T, K) — same as oasr.gemm."""
        torch.manual_seed(2)
        B_dim, T, K, V = 2, 50, 256, 1024
        A = torch.randn(B_dim, T, K, device="cuda", dtype=torch.float16) * 0.1
        W = torch.randn(V, K, device="cuda", dtype=torch.float16) * 0.1
        bias = torch.randn(V, device="cuda", dtype=torch.float16) * 0.1

        out = oasr.gemm_log_softmax(A, W, bias)
        assert out.shape == (B_dim, T, V)

        expected = F.log_softmax(F.linear(A.float(), W.float(), bias.float()), dim=-1).half()
        torch.testing.assert_close(out, expected, rtol=5e-3, atol=1e-2)

    def test_out_is_log_distribution(self):
        """Each row of the output should satisfy logsumexp(row) ≈ 0."""
        torch.manual_seed(3)
        A = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        B = torch.randn(1024, 256, device="cuda", dtype=torch.float16)

        out = oasr.gemm_log_softmax(A, B, None)
        lse = torch.logsumexp(out.float(), dim=-1)
        torch.testing.assert_close(lse, torch.zeros_like(lse), rtol=0, atol=5e-3)

    def test_destination_passing(self):
        torch.manual_seed(4)
        A = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        B = torch.randn(512, 128, device="cuda", dtype=torch.float16)
        out = torch.empty(32, 512, device="cuda", dtype=torch.float16)

        ret = oasr.gemm_log_softmax(A, B, None, out=out)
        assert ret.data_ptr() == out.data_ptr()
