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
    def test_gemm_fp16(self, oasr, M, N, K):
        """Test GEMM FP16 against torch.matmul."""
        dtype = torch.float16
        A = torch.randn(M, K, device='cuda', dtype=dtype)
        B = torch.randn(K, N, device='cuda', dtype=dtype)
        D = torch.empty(M, N, device='cuda', dtype=dtype)

        params = oasr.kernels.gemm.GemmParams()
        params.A = A.data_ptr()
        params.B = B.data_ptr()
        params.D = D.data_ptr()
        params.M, params.N, params.K = M, N, K
        params.lda, params.ldb, params.ldd = K, N, N
        params.alpha = 1.0
        params.beta = 0.0
        params.dtype_a = params.dtype_b = params.dtype_d = oasr.DataType.FP16

        status = oasr.kernels.gemm.invoke_gemm(params)
        assert status == oasr.kernels.gemm.GemmStatus.SUCCESS, (
            f'invoke_gemm failed: {oasr.kernels.gemm.get_gemm_status_string(status)}'
        )
        oasr.synchronize()

        expected = torch.matmul(A, B)
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    def test_gemm_alpha_beta(self, oasr):
        """Test GEMM with alpha=2.0, beta=0 -> D = 2 * A @ B."""
        M, N, K = 32, 64, 48
        dtype = torch.float16
        A = torch.randn(M, K, device='cuda', dtype=dtype)
        B = torch.randn(K, N, device='cuda', dtype=dtype)
        D = torch.empty(M, N, device='cuda', dtype=dtype)

        params = oasr.kernels.gemm.GemmParams()
        params.A = A.data_ptr()
        params.B = B.data_ptr()
        params.D = D.data_ptr()
        params.M, params.N, params.K = M, N, K
        params.lda, params.ldb, params.ldd = K, N, N
        params.alpha = 2.0
        params.beta = 0.0
        params.dtype_a = params.dtype_b = params.dtype_d = oasr.DataType.FP16

        status = oasr.kernels.gemm.invoke_gemm(params)
        assert status == oasr.kernels.gemm.GemmStatus.SUCCESS
        oasr.synchronize()

        expected = 2.0 * torch.matmul(A, B)
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize('M,N,K', [(64, 128, 256)])
    def test_gemm_bf16(self, oasr, M, N, K):
        """Test GEMM BF16 against torch.matmul (skipped if BF16 not supported)."""
        if not torch.cuda.is_bf16_supported():
            pytest.skip('BF16 not supported on this device')
        dtype = torch.bfloat16
        A = torch.randn(M, K, device='cuda', dtype=dtype)
        B = torch.randn(K, N, device='cuda', dtype=dtype)
        D = torch.empty(M, N, device='cuda', dtype=dtype)

        params = oasr.kernels.gemm.GemmParams()
        params.A = A.data_ptr()
        params.B = B.data_ptr()
        params.D = D.data_ptr()
        params.M, params.N, params.K = M, N, K
        params.lda, params.ldb, params.ldd = K, N, N
        params.alpha = 1.0
        params.beta = 0.0
        params.dtype_a = params.dtype_b = params.dtype_d = oasr.DataType.BF16

        status = oasr.kernels.gemm.invoke_gemm(params)
        assert status == oasr.kernels.gemm.GemmStatus.SUCCESS
        oasr.synchronize()

        expected = torch.matmul(A, B)
        torch.testing.assert_close(D.float(), expected.float(), rtol=1e-2, atol=1e-2)


# -----------------------------------------------------------------------------
# Batched GEMM (strided): D[b] = A[b] @ B[b]
# -----------------------------------------------------------------------------

class TestBmm:
    """Tests for batched GEMM (strided) kernel."""

    @pytest.mark.parametrize('batch_size,M,N,K', [
        (4, 64, 64, 128),
        (2, 32, 32, 32),
    ])
    def test_bmm_fp16(self, oasr, batch_size, M, N, K):
        """Test BMM FP16 against torch.bmm."""
        dtype = torch.float16
        A = torch.randn(batch_size, M, K, device='cuda', dtype=dtype)
        B = torch.randn(batch_size, K, N, device='cuda', dtype=dtype)
        D = torch.empty(batch_size, M, N, device='cuda', dtype=dtype)

        params = oasr.kernels.gemm.BmmParams.Strided(
            A.data_ptr(), B.data_ptr(), D.data_ptr(),
            batch_size, M, N, K, oasr.DataType.FP16,
        )
        status = oasr.kernels.gemm.invoke_bmm(params)
        assert status == oasr.kernels.gemm.GemmStatus.SUCCESS
        oasr.synchronize()

        expected = torch.bmm(A, B)
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    def test_bmm_bf16(self, oasr):
        """Test BMM BF16 against torch.bmm (skipped if BF16 not supported)."""
        if not torch.cuda.is_bf16_supported():
            pytest.skip('BF16 not supported on this device')
        batch_size, M, N, K = 4, 64, 64, 128
        dtype = torch.bfloat16
        A = torch.randn(batch_size, M, K, device='cuda', dtype=dtype)
        B = torch.randn(batch_size, K, N, device='cuda', dtype=dtype)
        D = torch.empty(batch_size, M, N, device='cuda', dtype=dtype)

        params = oasr.kernels.gemm.BmmParams.Strided(
            A.data_ptr(), B.data_ptr(), D.data_ptr(),
            batch_size, M, N, K, oasr.DataType.BF16,
        )
        status = oasr.kernels.gemm.invoke_bmm(params)
        assert status == oasr.kernels.gemm.GemmStatus.SUCCESS
        oasr.synchronize()

        expected = torch.bmm(A, B)
        torch.testing.assert_close(D.float(), expected.float(), rtol=1e-2, atol=1e-2)


# -----------------------------------------------------------------------------
# Grouped GEMM (variable-sized problems)
# -----------------------------------------------------------------------------

def _build_group_gemm_params(oasr, problem_sizes, A_tensors, B_tensors, D_tensors, dtype):
    """Build GroupGemmParams and workspace for a list of (M, N, K) problems."""
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
    workspace_int = torch.empty(int_ws, dtype=torch.uint8, device='cuda')

    params = oasr.kernels.gemm.GroupGemmParams()
    params.problems = problems_tensor.data_ptr()
    params.num_problems = num_problems
    params.A_array = a_ptrs.data_ptr()
    params.B_array = b_ptrs.data_ptr()
    params.D_array = d_ptrs.data_ptr()
    params.lda_array = lda_tensor.data_ptr()
    params.ldb_array = ldb_tensor.data_ptr()
    params.ldd_array = ldd_tensor.data_ptr()
    params.alpha = 1.0
    params.beta = 0.0
    params.dtype_a = params.dtype_b = params.dtype_d = oasr_dtype
    params.workspace_float = workspace_float.data_ptr()
    params.workspace_float_size = float_ws
    params.workspace_int = workspace_int.data_ptr()
    params.workspace_int_size = int_ws
    return params


class TestGroupGemm:
    """Tests for grouped GEMM kernel (variable-sized problems)."""

    def test_group_gemm_single_problem(self, oasr):
        """Test grouped GEMM with one problem."""
        problem_sizes = [(64, 128, 256)]
        dtype = torch.float16
        A_tensors = [torch.randn(M, K, device='cuda', dtype=dtype) for M, N, K in problem_sizes]
        B_tensors = [torch.randn(K, N, device='cuda', dtype=dtype) for M, N, K in problem_sizes]
        D_tensors = [torch.empty(M, N, device='cuda', dtype=dtype) for M, N, K in problem_sizes]

        params = _build_group_gemm_params(oasr, problem_sizes, A_tensors, B_tensors, D_tensors, dtype)
        status = oasr.kernels.gemm.invoke_group_gemm(params)
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

        params = _build_group_gemm_params(oasr, problem_sizes, A_tensors, B_tensors, D_tensors, dtype)
        status = oasr.kernels.gemm.invoke_group_gemm(params)
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

        params = _build_group_gemm_params(oasr, problem_sizes, A_tensors, B_tensors, D_tensors, dtype)
        status = oasr.kernels.gemm.invoke_group_gemm(params)
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

    def test_query_gemm_workspace_size(self, oasr):
        """query_gemm_workspace_size returns non-negative int."""
        size = oasr.kernels.gemm.query_gemm_workspace_size(128, 256, 512, oasr.DataType.FP16)
        assert isinstance(size, int) and size >= 0

    def test_query_bmm_workspace_size(self, oasr):
        """query_bmm_workspace_size returns non-negative int."""
        size = oasr.kernels.gemm.query_bmm_workspace_size(8, 64, 64, 128, oasr.DataType.FP16)
        assert isinstance(size, int) and size >= 0

    def test_query_group_gemm_workspace_size(self, oasr):
        """query_group_gemm_workspace_size returns (float_ws, int_ws) >= 0."""
        float_ws, int_ws = oasr.kernels.gemm.query_group_gemm_workspace_size(4, oasr.DataType.FP16)
        assert isinstance(float_ws, int) and float_ws >= 0
        assert isinstance(int_ws, int) and int_ws >= 0

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
