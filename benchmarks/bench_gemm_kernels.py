#!/usr/bin/env python3
"""
Performance benchmarks for GEMM, Batched GEMM, and Grouped GEMM kernels.
Uses triton.testing.do_bench for accurate GPU timing.

Profiling mode for NVIDIA Nsight Compute:
    ncu --set full -o profile_report python bench_gemm_kernels.py \
    --profile --kernel gemm --target oasr \
    --warmup 0 --iters 1
"""

import sys
sys.path.insert(0, 'python')

import argparse
import torch

import oasr

try:
    gemm_mod = oasr.kernels.gemm
    DataType = oasr.DataType
except AttributeError:
    from oasr._C import kernels
    gemm_mod = kernels.gemm
    DataType = oasr.DataType if hasattr(oasr, 'DataType') else getattr(kernels, 'DataType', None)
if DataType is None:
    import oasr._C
    DataType = oasr._C.DataType


# =============================================================================
# Profiling utilities
# =============================================================================

def profile_kernel(name: str, fn, warmup: int = 3, profile_iters: int = 1):
    """Run kernel for profiling with Nsight Compute."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(name)
    for _ in range(profile_iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


# =============================================================================
# Kernel setup functions (shared between benchmark and profile modes)
# =============================================================================

def setup_gemm(M, N, K, dtype=torch.float16):
    """Setup tensors for single GEMM: D = A @ B."""
    dtype_map = {
        torch.float32: DataType.FP32,
        torch.float16: DataType.FP16,
        torch.bfloat16: DataType.BF16,
    }
    if dtype not in dtype_map:
        dtype = torch.float16
    oasr_dtype = dtype_map[dtype]

    A = torch.randn(M, K, device='cuda', dtype=dtype)
    B = torch.randn(K, N, device='cuda', dtype=dtype)

    trans = gemm_mod.TransposeOp.NoTranspose
    lda, ldb, ldd = K, N, N

    # Pre-allocate D for benchmarking (avoid alloc overhead in hot loop)
    D_holder = [None]

    def oasr_fn():
        D_holder[0], _ = gemm_mod.invoke_gemm(
            A, B,
            M, N, K, lda, ldb, ldd,
            alpha=1.0, beta=0.0,
            trans_a=trans, trans_b=trans,
            dtype=oasr_dtype, stream=None,
        )

    D_pt = torch.empty(M, N, device='cuda', dtype=dtype)

    def pytorch_fn():
        torch.matmul(A, B, out=D_pt)

    return oasr_fn, pytorch_fn


def setup_bmm(batch_size, M, N, K, dtype=torch.float16):
    """Setup tensors for strided batched GEMM: D[b] = A[b] @ B[b]."""
    dtype_map = {
        torch.float32: DataType.FP32,
        torch.float16: DataType.FP16,
        torch.bfloat16: DataType.BF16,
    }
    if dtype not in dtype_map:
        dtype = torch.float16
    oasr_dtype = dtype_map[dtype]

    A = torch.randn(batch_size, M, K, device='cuda', dtype=dtype)
    B = torch.randn(batch_size, K, N, device='cuda', dtype=dtype)

    lda, ldb, ldd = K, N, N
    stride_a, stride_b, stride_d = M * K, K * N, M * N
    trans = gemm_mod.TransposeOp.NoTranspose

    D_holder = [None]

    def oasr_fn():
        D_holder[0], _ = gemm_mod.invoke_bmm(
            A, B,
            batch_size, M, N, K,
            lda, ldb, ldd,
            stride_a, stride_b, stride_d,
            alpha=1.0, beta=0.0,
            trans_a=trans, trans_b=trans,
            dtype=oasr_dtype, stream=None,
        )

    D_pt = torch.empty(batch_size, M, N, device='cuda', dtype=dtype)

    def pytorch_fn():
        torch.bmm(A, B, out=D_pt)

    return oasr_fn, pytorch_fn


def setup_group_gemm(problem_sizes, dtype=torch.float16):
    """Setup device arrays for invoke_group_gemm."""
    dtype_map = {
        torch.float32: DataType.FP32,
        torch.float16: DataType.FP16,
        torch.bfloat16: DataType.BF16,
    }
    if dtype not in dtype_map:
        dtype = torch.float16
    oasr_dtype = dtype_map[dtype]
    num_problems = len(problem_sizes)

    A_tensors = []
    B_tensors = []
    D_tensors = []
    lda_list = []
    ldb_list = []
    ldd_list = []
    problems_MNK = []

    for M, N, K in problem_sizes:
        A_tensors.append(torch.randn(M, K, device='cuda', dtype=dtype))
        B_tensors.append(torch.randn(K, N, device='cuda', dtype=dtype))
        D_tensors.append(torch.empty(M, N, device='cuda', dtype=dtype))
        lda_list.append(K)
        ldb_list.append(N)
        ldd_list.append(N)
        problems_MNK.extend([M, N, K])

    a_ptrs = torch.tensor([t.data_ptr() for t in A_tensors], dtype=torch.int64, device='cuda')
    b_ptrs = torch.tensor([t.data_ptr() for t in B_tensors], dtype=torch.int64, device='cuda')
    d_ptrs = torch.tensor([t.data_ptr() for t in D_tensors], dtype=torch.int64, device='cuda')
    lda_tensor = torch.tensor(lda_list, dtype=torch.int64, device='cuda')
    ldb_tensor = torch.tensor(ldb_list, dtype=torch.int64, device='cuda')
    ldd_tensor = torch.tensor(ldd_list, dtype=torch.int64, device='cuda')
    problems_tensor = torch.tensor(problems_MNK, dtype=torch.int32, device='cuda')

    float_ws_size, _ = gemm_mod.query_group_gemm_workspace_size(num_problems, oasr_dtype)
    workspace_float = torch.empty(float_ws_size, dtype=torch.uint8, device='cuda')

    def oasr_fn():
        gemm_mod.invoke_group_gemm(
            problems_tensor.data_ptr(), num_problems,
            a_ptrs.data_ptr(), b_ptrs.data_ptr(), d_ptrs.data_ptr(),
            lda_tensor.data_ptr(), ldb_tensor.data_ptr(), ldd_tensor.data_ptr(),
            oasr_dtype,
            workspace_float.data_ptr(), float_ws_size,
            stream=None,
        )

    def pytorch_fn():
        for i in range(num_problems):
            torch.matmul(A_tensors[i], B_tensors[i], out=D_tensors[i])

    return oasr_fn, pytorch_fn


# =============================================================================
# Benchmark functions
# =============================================================================


def benchmark_gemm():
    """Benchmark GEMM: OASR vs PyTorch."""
    import triton

    configs = [
        (8000, 256, 256),
        (16000, 256, 256),
        (16000, 2048, 256),
        (16000, 256, 2048),
        (16000, 512, 512),
        (16000, 2048, 512),
        (16000, 512, 2048),
        (32000, 256, 256),
        (32000, 512, 512),
    ]

    print("\n" + "=" * 70)
    print("GEMM Benchmark (Conformer-base / Conformer-large workload)")
    print("=" * 70)
    print(f"\n{'Shape (M, N, K)':<25} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 65)

    for M, N, K in configs:
        oasr_fn, pytorch_fn = setup_gemm(M, N, K)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"({M}, {N}, {K})"
        print(f"{shape_str:<25} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_bmm():
    """Benchmark Batched GEMM: OASR vs PyTorch."""
    import triton

    configs = [
        (256, 250, 250, 64),
        (512, 250, 250, 64),
        (256, 125, 125, 64),
        (512, 500, 500, 64),
        (64, 250, 64, 64),
    ]

    print("\n" + "=" * 70)
    print("Batched GEMM (BMM) Benchmark (Conformer attention workload)")
    print("=" * 70)
    print(f"\n{'Shape (B, M, N, K)':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, M, N, K in configs:
        oasr_fn, pytorch_fn = setup_bmm(batch_size, M, N, K)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"({batch_size}, {M}, {N}, {K})"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_group_gemm():
    """Benchmark Grouped GEMM: OASR vs PyTorch loop."""
    import triton

    configs = [
        (64, (250, 64, 64)),
        (256, (250, 64, 64)),
        (12, (16000, 256, 2048)),
        (12, (16000, 512, 2048)),
    ]

    print("\n" + "=" * 70)
    print("Grouped GEMM Benchmark (Conformer multi-block/head workload)")
    print("=" * 70)
    print(f"\n{'Config':<40} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for num_problems, (M, N, K) in configs:
        problem_sizes = [(M, N, K)] * num_problems
        oasr_fn, pytorch_fn = setup_group_gemm(problem_sizes)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        config_str = f"{num_problems} x ({M}, {N}, {K})"
        print(f"{config_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


# =============================================================================
# Profiling functions
# =============================================================================

PROFILE_CONFIGS = {
    'gemm': (16000, 256, 2048),
    'bmm': (256, 250, 250, 64),
    'group_gemm': (64, (250, 64, 64)),
}


def profile_kernels(kernels: list, target: str = 'oasr', warmup: int = 3, iters: int = 1):
    """Profile specified kernels for Nsight Compute."""
    print("=" * 70)
    print("OASR GEMM Kernels - Profiling Mode")
    print("=" * 70)
    print(f"Target: {target}, Warmup: {warmup}, Profile iterations: {iters}")
    print("=" * 70)

    setup_funcs = {
        'gemm': lambda: setup_gemm(*PROFILE_CONFIGS['gemm']),
        'bmm': lambda: setup_bmm(*PROFILE_CONFIGS['bmm']),
        'group_gemm': lambda: setup_group_gemm(
            [PROFILE_CONFIGS['group_gemm'][1]] * PROFILE_CONFIGS['group_gemm'][0]
        ),
    }

    for kernel_name in kernels:
        if kernel_name not in setup_funcs:
            print(f"Unknown kernel: {kernel_name}")
            continue

        config = PROFILE_CONFIGS[kernel_name]
        print(f"\nProfiling: {kernel_name}")
        print(f"  Config: {config}")

        oasr_fn, pytorch_fn = setup_funcs[kernel_name]()

        if target in ('oasr', 'both'):
            print(f"  Running OASR kernel...")
            profile_kernel(f"oasr_{kernel_name}", oasr_fn, warmup=warmup, profile_iters=iters)

        if target in ('pytorch', 'both'):
            print(f"  Running PyTorch kernel...")
            profile_kernel(f"pytorch_{kernel_name}", pytorch_fn, warmup=warmup, profile_iters=iters)

        print(f"  Done.")

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OASR GEMM/BMM/GroupGEMM Kernels - Benchmark & Profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python bench_gemm_kernels.py

  # Profile specific kernel with Nsight Compute (OASR only)
  ncu --set full -o gemm_profile python bench_gemm_kernels.py --profile --kernel gemm --target oasr

  # Profile all kernels
  python bench_gemm_kernels.py --profile --kernel all

Available kernels:
  gemm, bmm, group_gemm
        """
    )

    parser.add_argument('--profile', action='store_true',
        help='Run in profiling mode (single iterations for ncu)')
    parser.add_argument('--kernel', nargs='+', default=['all'],
        help='Kernel(s) to profile (use "all" for all kernels)')
    parser.add_argument('--target', choices=['oasr', 'pytorch', 'both'], default='oasr',
        help='Which implementation to profile (default: oasr)')
    parser.add_argument('--warmup', type=int, default=3,
        help='Number of warmup iterations for profiling (default: 3)')
    parser.add_argument('--iters', type=int, default=1,
        help='Number of profile iterations (default: 1)')

    args = parser.parse_args()

    if args.profile:
        if 'all' in args.kernel:
            kernels = list(PROFILE_CONFIGS.keys())
        else:
            kernels = args.kernel
        profile_kernels(kernels, target=args.target, warmup=args.warmup, iters=args.iters)
    else:
        print("=" * 70)
        print("OASR GEMM Kernels - Performance Benchmarks")
        print("=" * 70)

        benchmark_gemm()
        benchmark_bmm()
        benchmark_group_gemm()

        print("\n" + "=" * 70)
        print("Benchmarks complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
