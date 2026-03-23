#!/usr/bin/env python3
"""Performance benchmarks for Batched GEMM (BMM) kernel."""

import torch

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


def setup_bmm(batch_size, M, N, K, dtype=torch.float16):
    """Setup tensors for strided batched GEMM: D[b] = A[b] @ B[b]."""
    A = torch.randn(batch_size, M, K, device='cuda', dtype=dtype)
    B = torch.randn(batch_size, N, K, device='cuda', dtype=dtype)
    B_transposed = B.transpose(1, 2).contiguous()

    def oasr_fn():
        return oasr.bmm(A, B)

    def pytorch_fn():
        return torch.bmm(A, B_transposed)

    return oasr_fn, pytorch_fn


def benchmark_bmm():
    """Benchmark Batched GEMM: OASR vs PyTorch."""
    import triton

    configs = [
        (256, 200, 200, 64),
        (512, 400, 400, 64),
        (512, 200, 200, 64),
        (64, 200, 200, 64),
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


PROFILE_CONFIGS = {
    'bmm': (256, 200, 200, 64),
}


def main():
    setup_funcs = {
        'bmm': lambda: setup_bmm(*PROFILE_CONFIGS['bmm']),
    }
    run_main("Batched GEMM (BMM) Kernel", PROFILE_CONFIGS, setup_funcs, benchmark_bmm)


if __name__ == "__main__":
    main()
