#!/usr/bin/env python3
"""
Unit tests for GEMM, Batched GEMM (BMM), and Grouped GEMM kernels.
Uses torch.testing.assert_close for correctness verification.
"""

import pytest
import torch

import sys
sys.path.insert(0, 'python')


@pytest.fixture
def oasr():
    """Import oasr module."""
    import oasr
    try:
        from oasr import DataType
    except ImportError:
        from oasr._C import DataType
        oasr.DataType = DataType
    return oasr


# -----------------------------------------------------------------------------
# Smoke tests (no CUDA required)
# -----------------------------------------------------------------------------

class TestGemmModule:
    """Smoke tests: module and enums exist."""

    def test_gemm_module_import(self, oasr):
        """GEMM submodule is importable."""
        assert oasr.kernels.gemm is not None

    def test_gemm_status_enum(self, oasr):
        """GemmStatus enum has expected values."""
        assert hasattr(oasr.kernels.gemm, 'GemmStatus')
        assert getattr(oasr.kernels.gemm.GemmStatus, 'SUCCESS') is not None

    def test_get_gemm_status_string(self, oasr):
        """get_gemm_status_string returns non-empty string."""
        s = oasr.kernels.gemm.get_gemm_status_string(oasr.kernels.gemm.GemmStatus.SUCCESS)
        assert isinstance(s, str) and len(s) > 0


# -----------------------------------------------------------------------------
# Single GEMM: D = alpha * A @ B + beta * C
# -----------------------------------------------------------------------------

class TestGemm:
    """Tests for single GEMM kernel."""

    @pytest.mark.parametrize('M,N,K', [
        (64, 128, 256),
        (32, 32, 32),
        (256, 32, 128),
    ])
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    def test_gemm(self, oasr, M, N, K, dtype):
        """Test GEMM against torch.matmul."""
        A = torch.randn(M, K, device='cuda', dtype=dtype)
        B = torch.randn(N, K, device='cuda', dtype=dtype)

        D = oasr.kernels.gemm.invoke_gemm(A, B)
        oasr.synchronize()

        expected = torch.matmul(A, B.T)
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)


# -----------------------------------------------------------------------------
# Batched GEMM (strided): D[b] = A[b] @ B[b]
# -----------------------------------------------------------------------------

class TestBmm:
    """Tests for batched GEMM (strided) kernel."""

    @pytest.mark.parametrize('batch_size,M,N,K', [
        (4, 64, 128, 256),
        (3, 32, 32, 32),
        (3, 200, 200, 32),
    ])
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    def test_bmm(self, oasr, batch_size, M, N, K, dtype):
        """Test BMM against torch.bmm."""
        A = torch.randn(batch_size, M, K, device='cuda', dtype=dtype)
        B = torch.randn(batch_size, N, K, device='cuda', dtype=dtype)

        D = oasr.kernels.gemm.invoke_bmm(A, B)
        oasr.synchronize()

        expected = torch.bmm(A, B.permute(0, 2, 1))
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)


class TestGroupGemm:
    """Tests for grouped GEMM kernel (variable M per group, fixed N, K)."""

    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    def test_group_gemm_single_problem(self, oasr, dtype):
        """Test grouped GEMM with one problem."""
        # Fixed N, K; single group with variable M
        M, N, K = 32, 64, 64
        A = torch.randn(M, K, device='cuda', dtype=dtype)
        B = torch.randn(1, K, N, device='cuda', dtype=dtype)
        offset = torch.tensor([M], dtype=torch.int64, device='cuda')

        D = oasr.kernels.gemm.invoke_group_gemm(A, B, offset)
        oasr.synchronize()

        assert D.shape == (M, N)
        expected = torch.matmul(A, B[0].T)
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    def test_group_gemm_variable_sizes(self, oasr, dtype):
        """Test grouped GEMM with different M per group (same N, K)."""
        # M per group: 32, 64, 16; shared N=64, K=64
        M_list = [32, 64, 16]
        N, K = 128, 64
        L = sum(M_list)
        num_groups = len(M_list)

        A = torch.randn(L, K, device='cuda', dtype=dtype)
        B = torch.randn(num_groups, N, K, device='cuda', dtype=dtype)
        offset = torch.cumsum(torch.tensor(M_list, dtype=torch.int32, device='cuda'), dim=0, dtype=torch.int32)

        D = oasr.kernels.gemm.invoke_group_gemm(A, B, offset)
        oasr.synchronize()

        assert D.shape == (L, N)

        # Use grouped_mm for the expected result
        B_transposed = B.transpose(1, 2).contiguous()
        expected = torch.nn.functional.grouped_mm(A, B_transposed, offs=offset)

        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)


# -----------------------------------------------------------------------------
# Helper APIs (workspace, status, config)
# -----------------------------------------------------------------------------

class TestGemmHelpers:
    """Tests for GEMM helper APIs."""

    def test_get_gemm_status_string(self, oasr):
        """get_gemm_status_string returns string containing status name."""
        s = oasr.kernels.gemm.get_gemm_status_string(oasr.kernels.gemm.GemmStatus.SUCCESS)
        assert isinstance(s, str) and 'SUCCESS' in s

    def test_get_sm_version(self, oasr):
        """get_sm_version returns non-negative int."""
        sm = oasr.kernels.gemm.get_sm_version(-1)
        assert isinstance(sm, int) and sm >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
