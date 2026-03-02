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


# -----------------------------------------------------------------------------
# Grouped GEMM (variable-sized problems)
# -----------------------------------------------------------------------------

def _invoke_group_gemm(oasr, problem_sizes, A_tensors, B_tensors, D_tensors, dtype):
    """Build workspace and call invoke_group_gemm with direct parameters."""
    oasr_dtype = oasr.DataType.FP16 if dtype == torch.float16 else oasr.DataType.BF16
    num_problems = len(problem_sizes)

    problems_MNK = []
    lda_list, ldb_list, ldd_list = [], [], []
    for M, N, K in problem_sizes:
        problems_MNK.extend([M, N, K])
        lda_list.append(K)
        ldb_list.append(N)
        ldd_list.append(N)

    a_ptrs = torch.tensor([t.data_ptr() for t in A_tensors], dtype=torch.int64, device='cuda')
    b_ptrs = torch.tensor([t.data_ptr() for t in B_tensors], dtype=torch.int64, device='cuda')
    d_ptrs = torch.tensor([t.data_ptr() for t in D_tensors], dtype=torch.int64, device='cuda')
    problems_tensor = torch.tensor(problems_MNK, dtype=torch.int32, device='cuda')
    lda_tensor = torch.tensor(lda_list, dtype=torch.int64, device='cuda')
    ldb_tensor = torch.tensor(ldb_list, dtype=torch.int64, device='cuda')
    ldd_tensor = torch.tensor(ldd_list, dtype=torch.int64, device='cuda')

    float_ws, int_ws = oasr.kernels.gemm.query_group_gemm_workspace_size(num_problems, oasr_dtype)
    workspace_float = torch.empty(float_ws, dtype=torch.uint8, device='cuda')

    return oasr.kernels.gemm.invoke_group_gemm(
        problems_tensor.data_ptr(),
        num_problems,
        a_ptrs.data_ptr(), b_ptrs.data_ptr(), d_ptrs.data_ptr(),
        lda_tensor.data_ptr(), ldb_tensor.data_ptr(), ldd_tensor.data_ptr(),
        oasr_dtype,
        workspace_float.data_ptr(), float_ws,
    )


class TestGroupGemm:
    """Tests for grouped GEMM kernel (variable-sized problems)."""

    def test_group_gemm_single_problem(self, oasr):
        """Test grouped GEMM with one problem."""
        problem_sizes = [(64, 128, 256)]
        dtype = torch.float16
        A_tensors = [torch.randn(M, K, device='cuda', dtype=dtype) for M, N, K in problem_sizes]
        B_tensors = [torch.randn(K, N, device='cuda', dtype=dtype) for M, N, K in problem_sizes]
        D_tensors = [torch.empty(M, N, device='cuda', dtype=dtype) for M, N, K in problem_sizes]

        status = _invoke_group_gemm(oasr, problem_sizes, A_tensors, B_tensors, D_tensors, dtype)
        assert status == oasr.kernels.gemm.GemmStatus.SUCCESS
        oasr.synchronize()

        for i, (M, N, K) in enumerate(problem_sizes):
            expected = torch.matmul(A_tensors[i], B_tensors[i])
            torch.testing.assert_close(D_tensors[i], expected, rtol=1e-2, atol=1e-2)

    def test_group_gemm_multiple_same_size(self, oasr):
        """Test grouped GEMM with multiple identical problem sizes."""
        problem_sizes = [(32, 64, 64)] * 4
        dtype = torch.float16
        A_tensors = [torch.randn(M, K, device='cuda', dtype=dtype) for M, N, K in problem_sizes]
        B_tensors = [torch.randn(K, N, device='cuda', dtype=dtype) for M, N, K in problem_sizes]
        D_tensors = [torch.empty(M, N, device='cuda', dtype=dtype) for M, N, K in problem_sizes]

        status = _invoke_group_gemm(oasr, problem_sizes, A_tensors, B_tensors, D_tensors, dtype)
        assert status == oasr.kernels.gemm.GemmStatus.SUCCESS
        oasr.synchronize()

        for i in range(len(problem_sizes)):
            expected = torch.matmul(A_tensors[i], B_tensors[i])
            torch.testing.assert_close(D_tensors[i], expected, rtol=1e-2, atol=1e-2)

    def test_group_gemm_variable_sizes(self, oasr):
        """Test grouped GEMM with different M, N, K per problem."""
        problem_sizes = [(32, 64, 48), (64, 32, 96), (16, 128, 64)]
        dtype = torch.float16
        A_tensors = [torch.randn(M, K, device='cuda', dtype=dtype) for M, N, K in problem_sizes]
        B_tensors = [torch.randn(K, N, device='cuda', dtype=dtype) for M, N, K in problem_sizes]
        D_tensors = [torch.empty(M, N, device='cuda', dtype=dtype) for M, N, K in problem_sizes]

        status = _invoke_group_gemm(oasr, problem_sizes, A_tensors, B_tensors, D_tensors, dtype)
        assert status == oasr.kernels.gemm.GemmStatus.SUCCESS
        oasr.synchronize()

        for i, (M, N, K) in enumerate(problem_sizes):
            expected = torch.matmul(A_tensors[i], B_tensors[i])
            torch.testing.assert_close(D_tensors[i], expected, rtol=1e-2, atol=1e-2)


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
