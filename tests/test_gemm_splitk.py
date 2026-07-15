#!/usr/bin/env python3
"""Tests for the GEMM split-K decompositions and the persistent workspace cache.

Covers:
  * serial split-K (single-launch, pre-zeroed semaphore workspace) numerics
    across deep split factors — repeated launches must stay correct, proving
    the self-restoring-semaphore invariant the workspace cache relies on;
  * parallel split-K (``GemmSplitKParallel``) numerics, including the fused
    activation epilogue (applied once post-reduction) and the no-bias path;
  * serial split-K + activation is rejected (per-partition activation would be
    mathematically wrong);
  * CUDA-graph capture/replay of the serial split-K fast path (workspace
    address captured; semaphores restored every replay).
"""

import pytest
import torch
import torch.nn.functional as F

import oasr  # noqa: F401  (triggers JIT module availability)
from oasr.gemm import _get_gemm_module
from oasr.jit.core import _get_target_sm
from oasr.jit.gemm import CutlassGemmConfig, get_unique_compile_configs

_SM = _get_target_sm()


def _tol(dtype, K, split_k=1, serial=False):
    """Accumulation-order tolerance: bf16/f16 outputs at deep K; serial split-K
    round-trips partials through the output dtype, one rounding per slice."""
    base = 4e-2 if dtype == torch.float16 else 2.5e-1
    scale = (K / 256) ** 0.5
    if serial:
        scale *= max(1, split_k) ** 0.5
    return base * scale


def _find_cfg(*, parallel_split_k=False, block_m=None):
    for cfg in get_unique_compile_configs(_SM).values():
        if not isinstance(cfg, CutlassGemmConfig):
            continue
        if getattr(cfg, "stream_k", False):
            continue
        if bool(getattr(cfg, "parallel_split_k", False)) != parallel_split_k:
            continue
        if block_m is not None and cfg.block_m != block_m:
            continue
        return cfg
    return None


_SERIAL_CFG = _find_cfg(parallel_split_k=False, block_m=16)
_PK_CFG = _find_cfg(parallel_split_k=True, block_m=16)


@pytest.mark.skipif(_SERIAL_CFG is None, reason="no serial CUTLASS 2.x config for this arch")
class TestSerialSplitK:
    @pytest.mark.parametrize("split_k", [2, 4, 8, 16])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_numerics_repeated(self, dtype, split_k):
        """Deep serial split-K stays correct over REPEATED launches — this is
        what proves the pre-zeroed cached semaphores are restored each run."""
        torch.manual_seed(0)
        M, N, K = 64, 256, 2048
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        C = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        ref = torch.addmm(C.float(), A.float(), B.float().t())

        fn = getattr(_get_gemm_module(), f"gemm_{_SERIAL_CFG.compile_name}")
        for _ in range(5):
            out.zero_()
            fn(out, A, B, C, split_k)
            torch.cuda.synchronize()
            err = (out.float() - ref).abs().max().item()
            assert err < _tol(dtype, K, split_k, serial=True), f"split_k={split_k} err={err}"

    def test_activation_rejected(self):
        """Serial split-K + fused activation must fail loudly, not silently
        produce nested-activation garbage."""
        M, N, K = 64, 2048, 256
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        fn = getattr(_get_gemm_module(), f"gemm_{_SERIAL_CFG.compile_name}_activation")
        with pytest.raises(RuntimeError):
            fn(out, A, B, C, 2, 4)

    def test_cuda_graph_capture_replay(self):
        """Warm the workspace cache, capture the split-K launch in a CUDA
        graph, replay several times with mutated inputs — every replay must be
        correct (the captured kernel must restore the semaphores itself)."""
        dtype = torch.bfloat16
        M, N, K = 64, 256, 2048
        split_k = 8
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        C = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        fn = getattr(_get_gemm_module(), f"gemm_{_SERIAL_CFG.compile_name}")

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn(out, A, B, C, split_k)  # warm-up: allocates + zeroes the cached workspace
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn(out, A, B, C, split_k)

        for i in range(3):
            A.normal_(generator=None)
            graph.replay()
            torch.cuda.synchronize()
            ref = torch.addmm(C.float(), A.float(), B.float().t())
            err = (out.float() - ref).abs().max().item()
            assert err < _tol(dtype, K, split_k, serial=True), f"replay {i}: err={err}"


@pytest.mark.skipif(_PK_CFG is None, reason="parallel split-K configs not built for this arch")
class TestParallelSplitK:
    @pytest.mark.parametrize("split_k", [2, 8, 16])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_numerics(self, dtype, split_k):
        torch.manual_seed(0)
        M, N, K = 64, 256, 4864
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        C = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)
        ref = torch.addmm(C.float(), A.float(), B.float().t())

        fn = getattr(_get_gemm_module(), f"gemm_{_PK_CFG.compile_name}")
        fn(out, A, B, C, split_k)
        torch.cuda.synchronize()
        err = (out.float() - ref).abs().max().item()
        # fp32 partials: error must NOT grow with split depth
        assert err < _tol(dtype, K), f"split_k={split_k} err={err}"

    def test_no_bias(self):
        M, N, K = 32, 256, 2048
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        fn = getattr(_get_gemm_module(), f"gemm_{_PK_CFG.compile_name}")
        fn(out, A, B, None, 8)
        torch.cuda.synchronize()
        ref = A.float() @ B.float().t()
        assert (out.float() - ref).abs().max().item() < _tol(torch.bfloat16, K)

    @pytest.mark.parametrize("act,ref_fn", [(0, F.relu), (2, F.silu)])
    def test_activation_applied_once(self, act, ref_fn):
        """The activation epilogue runs in the reduction kernel — exactly once
        over the FULL sum (the property serial split-K cannot provide)."""
        torch.manual_seed(1)
        M, N, K = 64, 128, 2048
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        cfg = _find_cfg(parallel_split_k=True, block_m=16)
        fn = getattr(_get_gemm_module(), f"gemm_{cfg.compile_name}_activation")
        fn(out, A, B, C, act, 8)
        torch.cuda.synchronize()
        ref = ref_fn(torch.addmm(C.float(), A.float(), B.float().t()))
        err = (out.float() - ref).abs().max().item()
        assert err < _tol(torch.bfloat16, K), f"err={err}"

    def test_slices_one_rejected(self):
        """parallel split-K variants require split_k_slices > 1."""
        M, N, K = 32, 256, 2048
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        fn = getattr(_get_gemm_module(), f"gemm_{_PK_CFG.compile_name}")
        with pytest.raises(RuntimeError):
            fn(out, A, B, None, 1)
