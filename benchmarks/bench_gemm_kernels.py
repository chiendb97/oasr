#!/usr/bin/env python3
"""
Performance benchmarks for GEMM, Batched GEMM, and Grouped GEMM kernels.
Uses triton.testing.do_bench for accurate GPU timing.

Problem sizes align with WeNet Conformer-base (d_model=256, linear_units=2048) /
Conformer-large (d_model=512); 10 sec audio, batch up to 64.

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

# Resolve gemm submodule and types
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
    """
    Run kernel for profiling with Nsight Compute.

    Args:
        name: Kernel name for NVTX range
        fn: Kernel function to profile
        warmup: Number of warmup iterations
        profile_iters: Number of iterations to profile
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Profile iterations with NVTX markers
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
    D = torch.empty(M, N, device='cuda', dtype=dtype)

    params = gemm_mod.GemmParams()
    params.A = A.data_ptr()
    params.B = B.data_ptr()
    params.D = D.data_ptr()
    params.M = M
    params.N = N
    params.K = K
    params.lda = K
    params.ldb = N
    params.ldd = N
    params.alpha = 1.0
    params.beta = 0.0
    params.dtype_a = oasr_dtype
    params.dtype_b = oasr_dtype
    params.dtype_d = oasr_dtype

    def oasr_fn():
        gemm_mod.invoke_gemm(params)

    def pytorch_fn():
        torch.matmul(A, B, out=D)

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
    D = torch.empty(batch_size, M, N, device='cuda', dtype=dtype)

    params = gemm_mod.BmmParams.Strided(
        A.data_ptr(), B.data_ptr(), D.data_ptr(),
        batch_size, M, N, K, oasr_dtype, None
    )

    def oasr_fn():
        gemm_mod.invoke_bmm(params)

    def pytorch_fn():
        torch.bmm(A, B, out=D)

    return oasr_fn, pytorch_fn


def setup_group_gemm(problem_sizes, dtype=torch.float16):
    """
    problem_sizes: list of (M, N, K) for each problem.
    Setup device arrays and GroupGemmParams for invoke_group_gemm.
    """
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

    # Device array of pointers (as int64 on GPU)
    a_ptrs = torch.tensor([t.data_ptr() for t in A_tensors], dtype=torch.int64, device='cuda')
    b_ptrs = torch.tensor([t.data_ptr() for t in B_tensors], dtype=torch.int64, device='cuda')
    d_ptrs = torch.tensor([t.data_ptr() for t in D_tensors], dtype=torch.int64, device='cuda')
    lda_tensor = torch.tensor(lda_list, dtype=torch.int64, device='cuda')
    ldb_tensor = torch.tensor(ldb_list, dtype=torch.int64, device='cuda')
    ldd_tensor = torch.tensor(ldd_list, dtype=torch.int64, device='cuda')

    # problems: device buffer of GemmProblemDesc (M,N,K per problem -> 3 ints each)
    problems_tensor = torch.tensor(problems_MNK, dtype=torch.int32, device='cuda')

    float_ws_size, int_ws_size = gemm_mod.query_group_gemm_workspace_size(num_problems, oasr_dtype)
    workspace_float = torch.empty(float_ws_size, dtype=torch.uint8, device='cuda')
    workspace_int = torch.empty(int_ws_size, dtype=torch.uint8, device='cuda')

    params = gemm_mod.GroupGemmParams()
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
    params.dtype_a = oasr_dtype
    params.dtype_b = oasr_dtype
    params.dtype_d = oasr_dtype
    params.workspace_float = workspace_float.data_ptr()
    params.workspace_float_size = float_ws_size
    params.workspace_int = workspace_int.data_ptr()
    params.workspace_int_size = int_ws_size

    def oasr_fn():
        gemm_mod.invoke_group_gemm(params)

    def pytorch_fn():
        for i in range(num_problems):
            torch.matmul(A_tensors[i], B_tensors[i], out=D_tensors[i])

    return oasr_fn, pytorch_fn


# =============================================================================
# Benchmark functions
# =============================================================================


def benchmark_gemm():
    """Benchmark GEMM: OASR vs PyTorch. Sizes from WeNet (10 sec audio, batch up to 64)."""
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
    """Benchmark Batched GEMM: OASR vs PyTorch. Sizes from WeNet Conformer attention (B*heads, T, head_dim)."""
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
    """Benchmark Grouped GEMM: OASR vs PyTorch loop. Sizes from Conformer multi-block/head."""
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

# Default profiling configs (Conformer representative sizes)
PROFILE_CONFIGS = {
    'gemm': (16000, 256, 2048),   # FFN reduce, B=64 T=250
    'bmm': (256, 250, 250, 64),   # attention Q@K^T
    'group_gemm': (64, (250, 64, 64)),  # num_problems, (M, N, K)
}


def profile_kernels(kernels: list, target: str = 'oasr', warmup: int = 3, iters: int = 1):
    """
    Profile specified kernels for Nsight Compute.

    Args:
        kernels: List of kernel names to profile
        target: Which implementation to profile ('oasr', 'pytorch', or 'both')
        warmup: Warmup iterations
        iters: Profile iterations
    """
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

        # Profile OASR kernel
        if target in ('oasr', 'both'):
            print(f"  Running OASR kernel...")
            profile_kernel(f"oasr_{kernel_name}", oasr_fn, warmup=warmup, profile_iters=iters)

        # Profile PyTorch reference
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

  # Profile PyTorch implementation only
  python bench_gemm_kernels.py --profile --kernel gemm --target pytorch

  # Profile both implementations
  python bench_gemm_kernels.py --profile --kernel gemm --target both

  # Profile multiple kernels
  python bench_gemm_kernels.py --profile --kernel gemm bmm --target both

  # Profile all kernels
  python bench_gemm_kernels.py --profile --kernel all

Available kernels:
  gemm, bmm, group_gemm
        """
    )

    parser.add_argument(
        '--profile', action='store_true',
        help='Run in profiling mode (single iterations for ncu)'
    )
    parser.add_argument(
        '--kernel', nargs='+', default=['all'],
        help='Kernel(s) to profile (use "all" for all kernels)'
    )
    parser.add_argument(
        '--target', choices=['oasr', 'pytorch', 'both'], default='oasr',
        help='Which implementation to profile: oasr, pytorch, or both (default: oasr)'
    )
    parser.add_argument(
        '--warmup', type=int, default=3,
        help='Number of warmup iterations for profiling (default: 3)'
    )
    parser.add_argument(
        '--iters', type=int, default=1,
        help='Number of profile iterations (default: 1)'
    )

    args = parser.parse_args()

    if args.profile:
        # Profiling mode
        if 'all' in args.kernel:
            kernels = list(PROFILE_CONFIGS.keys())
        else:
            kernels = args.kernel

        profile_kernels(kernels, target=args.target, warmup=args.warmup, iters=args.iters)
    else:
        # Benchmark mode
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
