#!/usr/bin/env python3
"""
Unit tests for functional GEMM API (TVM-FFI JIT path).
"""

import pytest
import torch

import oasr

# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")


class TestGemm:
    """Tests for oasr.gemm() functional API."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (64, 128, 256),
            (32, 32, 32),
            (256, 32, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm(self, M, N, K, dtype):
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)

        D = oasr.gemm(A, B)

        expected = torch.matmul(A, B.T)
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm_destination_passing(self, dtype):
        """Test GEMM with pre-allocated output."""
        M, N, K = 64, 128, 256
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)

        result = oasr.gemm(A, B, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = torch.matmul(A, B.T)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


class TestBmm:
    """Tests for oasr.bmm() functional API."""

    @pytest.mark.parametrize(
        "batch_size,M,N,K",
        [
            (4, 64, 128, 256),
            (3, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bmm(self, batch_size, M, N, K, dtype):
        A = torch.randn(batch_size, M, K, device="cuda", dtype=dtype)
        B = torch.randn(batch_size, N, K, device="cuda", dtype=dtype)

        D = oasr.bmm(A, B)

        expected = torch.bmm(A, B.permute(0, 2, 1))
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bmm_destination_passing(self, dtype):
        """Test BMM with pre-allocated output."""
        batch_size, M, N, K = 4, 64, 128, 256
        A = torch.randn(batch_size, M, K, device="cuda", dtype=dtype)
        B = torch.randn(batch_size, N, K, device="cuda", dtype=dtype)
        out = torch.empty(batch_size, M, N, device="cuda", dtype=dtype)

        result = oasr.bmm(A, B, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = torch.bmm(A, B.permute(0, 2, 1))
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
