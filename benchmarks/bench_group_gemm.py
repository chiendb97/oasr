#!/usr/bin/env python3
"""Performance benchmarks for Grouped GEMM kernel."""

import torch

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


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
    offset = torch.cumsum(torch.tensor(
        Ms, dtype=torch.int32, device='cuda'), dim=0, dtype=torch.int32)

    def oasr_fn():
        return oasr.group_gemm(A, B, offset)

    def pytorch_fn():
        try:
            D = torch.nn.functional.grouped_mm(
                A, B_transposed, offs=offset)
            return D
        except:
            D = torch.zeros(L, N, device='cuda', dtype=dtype)
            num_groups = len(Ms)
            s_idx = 0
            for i in range(num_groups):
                m_i = Ms[i]
                D[s_idx: s_idx + m_i] = torch.matmul(
                    A[s_idx: s_idx + m_i], B_transposed[i]
                )
                s_idx += m_i

            return D

    return oasr_fn, pytorch_fn


def benchmark_group_gemm():
    """Benchmark Grouped GEMM: OASR vs PyTorch loop."""
    import triton

    # Each config: (num_groups, (base_M, N, K)).
    # For benchmarking we keep N, K fixed per config and vary M across groups.
    configs = [
        (32, (256, 64, 64)),
        (32, (256, 128, 64)),
        (64, (16000, 256, 2048)),
        (64, (16000, 512, 2048)),
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
        Ms = torch.randint(low=low, high=high, size=(
            problem_count,), device='cuda').tolist()

        problem_sizes = [(M_i, N, K) for M_i in Ms]
        oasr_fn, pytorch_fn = setup_group_gemm(problem_sizes)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        config_str = f"{problem_count} x (M∈[{min(Ms)},{max(Ms)}], N={N}, K={K})"
        print(
            f"{config_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


PROFILE_CONFIGS = {
    'group_gemm': (64, (200, 64, 64)),
}


def main():
    setup_funcs = {
        'group_gemm': lambda: setup_group_gemm(
            [PROFILE_CONFIGS['group_gemm'][1]] *
            PROFILE_CONFIGS['group_gemm'][0]
        ),
    }
    run_main("Grouped GEMM Kernel", PROFILE_CONFIGS,
             setup_funcs, benchmark_group_gemm)


if __name__ == "__main__":
    main()
