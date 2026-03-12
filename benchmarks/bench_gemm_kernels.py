#!/usr/bin/env python3
"""
Performance benchmarks for GEMM, Batched GEMM, and Grouped GEMM kernels.
Uses triton.testing.do_bench for accurate GPU timing.

Profiling mode for NVIDIA Nsight Compute:
    ncu --set full -o profile_report python bench_gemm_kernels.py \
    --profile --kernel gemm --target oasr \
    --warmup 0 --iters 1
"""

import argparse
import torch
import torch.nn.functional as F

import oasr



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

    A = torch.randn(M, K, device='cuda', dtype=dtype)
    B = torch.randn(N, K, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.gemm.gemm(
            A, B, stream=None,
        )

    def pytorch_fn():
        return F.linear(A, B)

    return oasr_fn, pytorch_fn


def setup_bmm(batch_size, M, N, K, dtype=torch.float16):
    """Setup tensors for strided batched GEMM: D[b] = A[b] @ B[b]."""
    A = torch.randn(batch_size, M, K, device='cuda', dtype=dtype)
    B = torch.randn(batch_size, N, K, device='cuda', dtype=dtype)
    B_transposed = B.transpose(1, 2).contiguous()

    def oasr_fn():
        return oasr.kernels.gemm.bmm(A, B_transposed, stream=None)

    def pytorch_fn():
        return torch.bmm(A, B_transposed)

    return oasr_fn, pytorch_fn


def setup_group_gemm(problem_sizes, dtype=torch.bfloat16):
    """Setup tensors for grouped GEMM using high-level invoke_group_gemm.

    problem_sizes: list of (M, N, K) for each group. N and K are assumed equal
    across all groups (matches the grouped GEMM tests); M may vary.
    """
    if len(problem_sizes) == 0:
        raise ValueError("problem_sizes must be non-empty")

    # Assume common N, K across groups
    _, N, K = problem_sizes[0]
    Ms = [M for M, _, _ in problem_sizes]
    assert all(N_i == N and K_i == K for M, N_i, K_i in problem_sizes), \
        "All grouped GEMM problems must share the same N and K"

    num_problems = len(problem_sizes)
    L = sum(Ms)

    # Build concatenated A [L, K] and per-group B [num_problems, K, N]
    A = torch.randn(L, K, device='cuda', dtype=dtype)
    B = torch.randn(num_problems, N, K, device='cuda', dtype=dtype)
    B_transposed = B.transpose(1, 2).contiguous()
    offset = torch.cumsum(torch.tensor(Ms, dtype=torch.int32, device='cuda'), dim=0, dtype=torch.int32)

    def oasr_fn():
        return oasr.kernels.gemm.group_gemm(A, B, offset, stream=None)

    def pytorch_fn():
        return F.grouped_mm(A, B_transposed, offs=offset)

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
    print("GEMM Benchmark")
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
        (256, 200, 200, 64),
        (512, 250, 250, 64),
        (256, 100, 100, 64),
        (512, 200, 200, 64),
        (64, 200, 64, 64),
    ]

    print("\n" + "=" * 70)
    print("Batched GEMM (BMM) Benchmark")
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

    # Each config: (num_groups, (base_M, N, K)).
    # For benchmarking we keep N, K fixed per config and vary M across groups.
    configs = [
        (8, (256, 64, 64)),
        (8, (256, 128, 64)),
        (16, (16000, 256, 2048)),
        (16, (16000, 512, 2048)),
    ]

    print("\n" + "=" * 70)
    print("Grouped GEMM Benchmark")
    print("=" * 70)
    print(f"\n{'Config':<40} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for problem_count, (base_M, N, K) in configs:
        # Use a fixed small number of problems with random M while keeping N, K fixed.
        torch.manual_seed(0)
        # Sample M in a reasonable range around base_M
        low = max(16, base_M // 2)
        high = max(low + 1, base_M * 2)
        Ms = torch.randint(low=low, high=high, size=(problem_count,), device='cuda').tolist()

        problem_sizes = [(M_i, N, K) for M_i in Ms]
        oasr_fn, pytorch_fn = setup_group_gemm(problem_sizes)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        config_str = f"{problem_count} x (M∈[{min(Ms)},{max(Ms)}], N={N}, K={K})"
        print(f"{config_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


# =============================================================================
# Profiling functions
# =============================================================================

PROFILE_CONFIGS = {
    'gemm': (16000, 256, 2048),
    'bmm': (256, 200, 200, 64),
    'group_gemm': (64, (200, 64, 64)),
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
