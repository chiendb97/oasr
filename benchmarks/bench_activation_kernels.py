#!/usr/bin/env python3
"""
Performance benchmarks for convolution kernels.
Uses triton.testing.do_bench for accurate GPU timing.

Profiling mode for NVIDIA Nsight Compute:
    ncu --set full -o profile_report python bench_conv_kernels.py \
    --profile --kernel swish --target oasr \
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


def setup_glu(batch_size, seq_len, channels, dtype=torch.float16):
    """Setup tensors for GLU."""
    x = torch.randn(batch_size, seq_len, 2 * channels,
                    device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.conv.glu(
            x
        )

    def pytorch_fn():
        return F.glu(x, dim=-1)

    return oasr_fn, pytorch_fn


def setup_swish(batch_size, seq_len, channels, dtype=torch.float16):
    """Setup tensors for Swish."""
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.kernels.conv.swish(
            x
        )

    def pytorch_fn():
        return F.silu(x)

    return oasr_fn, pytorch_fn




def benchmark_glu():
    """Benchmark GLU activation."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 256),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("GLU Activation Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_glu(batch_size, seq_len, channels)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {2*channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_swish():
    """Benchmark Swish (SiLU) activation."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("Swish Activation Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_swish(batch_size, seq_len, channels)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")





PROFILE_CONFIGS = {
    'glu': (64, 250, 512),
    'swish': (64, 250, 512),    
}


def profile_kernels(kernels: list, target: str = 'both', warmup: int = 3, iters: int = 1):
    """Profile specified kernels for Nsight Compute."""
    print("=" * 70)
    print("OASR Convolution Kernels - Profiling Mode")
    print("=" * 70)
    print(f"Target: {target}, Warmup: {warmup}, Profile iterations: {iters}")
    print("=" * 70)

    setup_funcs = {
        'glu': lambda: setup_glu(*PROFILE_CONFIGS['glu']),
        'swish': lambda: setup_swish(*PROFILE_CONFIGS['swish']),
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
            profile_kernel(f"oasr_{kernel_name}", oasr_fn,
                           warmup=warmup, profile_iters=iters)

        if target in ('pytorch', 'both'):
            print(f"  Running PyTorch kernel...")
            profile_kernel(
                f"pytorch_{kernel_name}", pytorch_fn, warmup=warmup, profile_iters=iters)

        print(f"  Done.")

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OASR Activation Kernels - Benchmark & Profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python bench_activation_kernels.py

  # Profile specific kernel with Nsight Compute (OASR only)
  ncu --set full -o activation_profile python bench_activation_kernels.py --profile --kernel glu --target oasr

  # Profile all kernels
  python bench_activation_kernels.py --profile --kernel all

Available kernels:
  glu, swish
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
        profile_kernels(kernels, target=args.target,
                        warmup=args.warmup, iters=args.iters)
    else:
        print("=" * 70)
        print("OASR Activation Kernels - Performance Benchmarks")
        print("=" * 70)

        benchmark_glu()
        benchmark_swish()

        print("\n" + "=" * 70)
        print("Benchmarks complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
