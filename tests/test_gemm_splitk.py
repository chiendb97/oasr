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
from oasr.functionals.gemm import _gemm_fn, _get_gemm_module
from oasr.jit.core import _get_target_sm
from oasr.jit.gemm import CutlassGemmConfig, get_unique_compile_configs

# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")


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

    @pytest.mark.parametrize("act,ref_fn", [(0, F.relu), (2, F.silu), (4, F.gelu)])
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


def _split_k_variant(kind: str):
    """A registered variant of one kind, or ``None``.

    ``pk`` deliberately for the size tests: a parallel split-K workspace is
    ``M*N*4*split`` bytes of fp32 partials, where a *serial* split-K semaphore is
    one int per output tile.  With the small one the bound is untestable — 32
    keys of 4 KiB is 128 KiB whether it is enforced or not.
    """
    for c in get_unique_compile_configs(_SM).values():
        if kind == "pk" and getattr(c, "parallel_split_k", False):
            return c
        if kind == "sk" and getattr(c, "stream_k", False):
            return c
        if kind == "any" and (getattr(c, "split_k", 1) > 1 or getattr(c, "stream_k", False)):
            return c
    return None


def _run_on_streams(fn, split_k, M, N, K, device, n_streams):
    """Run one GEMM per fresh ``torch.cuda.Stream()``; return the relative error.

    ``torch.cuda.Stream()`` does not create a stream — it takes the next of a
    POOL of 32 per device and cycles.  That is the whole reason the first version
    of this test was blind: the cache is keyed on the stream handle, so a caller
    like this saturates at 32 keys after the first 32 iterations, and a test that
    warmed up and then measured a *second* burst measured zero growth whether the
    cache was bounded or not.  What grows is not the key count but the bytes
    behind each key.
    """
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    C = torch.randn(N, device=device, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    ref = A.float() @ B.float().t() + C.float()
    for _ in range(n_streams):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(out, A, B, C, split_k)
        torch.cuda.current_stream().wait_stream(s)
        del s
    torch.cuda.synchronize()
    return (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


class TestWorkspaceCacheBytesBound:
    """The split-K / Stream-K workspace cache must bound the BYTES it holds.

    ``include/oasr/common/workspace_cache.h`` keeps one grow-only buffer per
    ``(device, stream, pool)`` and never frees — a retired buffer's address may
    be baked into a captured graph.  So every byte it hands out is held until the
    process exits, which makes it a cache that has to be bounded in bytes rather
    than an allocator.

    It was not.  A parallel split-K workspace is ``M*N*4*split`` bytes, so a
    single 4096x5008 shape is 328 MiB per key; run across the 32-handle stream
    pool that is 10,016 MiB held forever, measured, with the next ladder step
    failing the kernel outright.  ``scripts/tune_asr_gemm.py`` hit exactly this
    over a 121-shape sweep and died on a 66 MiB allocation while PyTorch's own
    allocator held 1 GiB.

    Serving never sees it — every architecture in the tree asks for 152-296 bytes
    of semaphore and at most 1 MiB of scratch — which is exactly why it needs a
    test rather than a comment.

    These assert on ``ws_cache_bytes()`` rather than on ``cudaMemGetInfo``,
    because ``cudaMallocAsync`` recycles: "cached once" and "allocated and freed
    every call" both read as flat free memory, and a bound whose test cannot see
    it is how the unbounded version shipped.
    """

    @pytest.mark.cuda
    def test_a_large_workspace_is_not_cached(self, device):
        cfg = _split_k_variant("pk") or _split_k_variant("any")
        if cfg is None:
            pytest.skip("no split-K / Stream-K variant registered for this arch")
        mod = _get_gemm_module()
        fn = _gemm_fn(cfg.compile_name, False)
        split_k = max(2, int(getattr(cfg, "split_k", 1)))
        # 4096 x 5008 fp32 partials x split -> >= 156 MiB, far over the per-key
        # ceiling.  96 iterations to walk the whole 32-handle stream pool three
        # times over.
        before = mod.ws_cache_bytes()
        err = _run_on_streams(fn, split_k, 4096, 5008, 256, device, 96)
        grew = (mod.ws_cache_bytes() - before) / 2**20
        assert grew < 64, (
            f"a 156+ MiB workspace across 96 stream-per-call GEMMs added {grew:.1f} MiB "
            "to a cache that never frees — the per-key size ceiling is not holding"
        )
        # And it still computed the right answer on the per-call path.
        assert err < 1e-2, f"declining to cache changed the result (rel err {err:.2e})"

    @pytest.mark.cuda
    def test_a_small_workspace_is_still_cached(self, device):
        """The guard in the other direction.

        A "fix" that simply stopped caching would satisfy every bound here, and
        would silently give back the win the cache exists for (a serial split-K
        semaphore that stays zeroed lets the kernel skip a per-launch memset —
        one whole kernel launch, ~5 us on this box, against a kernel of the same
        order).  So: a small workspace must still be cached, and re-used.
        """
        cfg = _split_k_variant("pk") or _split_k_variant("any")
        if cfg is None:
            pytest.skip("no split-K / Stream-K variant registered for this arch")
        mod = _get_gemm_module()
        fn = _gemm_fn(cfg.compile_name, False)
        split_k = max(2, int(getattr(cfg, "split_k", 1)))
        M, N, K = 128, 256, 512  # partials ~ 256 KiB * split
        # One stream, so one key.  Enough calls that a per-call allocation would
        # be unmistakable if it were counted.
        stream = torch.cuda.Stream()
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        C = torch.randn(N, device=device, dtype=torch.bfloat16)
        out = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(out, A, B, C, split_k)  # first call: allocates
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        keys_after_first = mod.ws_cache_keys()
        bytes_after_first = mod.ws_cache_bytes()
        assert keys_after_first >= 1, (
            "a 256 KiB workspace was not cached at all — the cache is disabled or "
            "its ceiling is below what production actually asks for"
        )
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(64):
                fn(out, A, B, C, split_k)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert mod.ws_cache_bytes() == bytes_after_first, (
            "64 further calls on the same stream allocated again — the cached "
            "buffer is not being re-used"
        )
        assert mod.ws_cache_keys() == keys_after_first

    @pytest.mark.cuda
    def test_the_total_stays_bounded_across_shapes_and_streams(self, device):
        """Many shapes x the whole stream pool: the total is what must not grow."""
        cfg = _split_k_variant("pk") or _split_k_variant("any")
        if cfg is None:
            pytest.skip("no split-K / Stream-K variant registered for this arch")
        mod = _get_gemm_module()
        fn = _gemm_fn(cfg.compile_name, False)
        split_k = max(2, int(getattr(cfg, "split_k", 1)))
        for M, N in ((128, 256), (512, 1024), (1024, 2048), (2048, 5008), (4096, 5008)):
            err = _run_on_streams(fn, split_k, M, N, 256, device, 40)
            held = mod.ws_cache_bytes() / 2**20
            assert held <= 64, (
                f"cache holds {held:.1f} MiB after M={M} N={N} — over its documented "
                "64 MiB total ceiling"
            )
            assert err < 1e-2, f"M={M} N={N}: rel err {err:.2e}"

    @pytest.mark.cuda
    def test_results_stay_correct_when_the_cache_declines(self, device):
        """Past a ceiling the cache returns nullptr and the caller allocates a
        per-call workspace — correct, just without the saved ritual.  Prove the
        numerics do not change, on the same shape, either side of the ceiling."""
        cfg = _split_k_variant("any")
        if cfg is None:
            pytest.skip("no split-K / Stream-K variant registered for this arch")
        fn = _gemm_fn(cfg.compile_name, False)
        M, N, K = 128, 256, 512
        torch.manual_seed(3)
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        C = torch.randn(N, device=device, dtype=torch.bfloat16)
        ref = A.float() @ B.float().t() + C.float()
        split_k = max(2, int(getattr(cfg, "split_k", 1)))
        out = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        for _ in range(140):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                out.zero_()
                fn(out, A, B, C, split_k)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-1)
            del s
