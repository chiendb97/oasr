"""Shared utilities for OASR benchmark scripts."""

import argparse

import torch


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


def make_bench_parser(description: str, kernels: list[str]) -> argparse.ArgumentParser:
    """Create a standard benchmark argument parser."""
    kernel_list = ", ".join(kernels)
    parser = argparse.ArgumentParser(
        description=f"OASR {description} - Benchmark & Profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available kernels:
  {kernel_list}
        """,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run in profiling mode (single iterations for ncu)",
    )
    parser.add_argument(
        "--kernel",
        nargs="+",
        default=["all"],
        help='Kernel(s) to profile (use "all" for all kernels)',
    )
    parser.add_argument(
        "--target",
        choices=["oasr", "pytorch", "both"],
        default="oasr",
        help="Which implementation to profile (default: oasr)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations for profiling (default: 3)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Number of profile iterations (default: 1)",
    )
    return parser


def run_profile(
    title: str,
    profile_configs: dict,
    setup_funcs: dict,
    kernels: list,
    target: str = "oasr",
    warmup: int = 3,
    iters: int = 1,
):
    """Run profiling for specified kernels."""
    print("=" * 70)
    print(f"OASR {title} - Profiling Mode")
    print("=" * 70)
    print(f"Target: {target}, Warmup: {warmup}, Profile iterations: {iters}")
    print("=" * 70)

    for kernel_name in kernels:
        if kernel_name not in setup_funcs:
            print(f"Unknown kernel: {kernel_name}")
            continue

        config = profile_configs[kernel_name]
        print(f"\nProfiling: {kernel_name}")
        print(f"  Config: {config}")

        oasr_fn, pytorch_fn = setup_funcs[kernel_name]()

        if target in ("oasr", "both"):
            print(f"  Running OASR kernel...")
            profile_kernel(
                f"oasr_{kernel_name}", oasr_fn, warmup=warmup, profile_iters=iters
            )

        if target in ("pytorch", "both"):
            print(f"  Running PyTorch kernel...")
            profile_kernel(
                f"pytorch_{kernel_name}",
                pytorch_fn,
                warmup=warmup,
                profile_iters=iters,
            )

        print(f"  Done.")

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)


def run_main(title: str, profile_configs: dict, setup_funcs: dict, benchmark_fn):
    """Standard main entry point for benchmark scripts."""
    parser = make_bench_parser(title, list(profile_configs.keys()))
    args = parser.parse_args()

    if args.profile:
        if "all" in args.kernel:
            kernels = list(profile_configs.keys())
        else:
            kernels = args.kernel
        run_profile(
            title,
            profile_configs,
            setup_funcs,
            kernels,
            target=args.target,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        print("=" * 70)
        print(f"OASR {title} - Performance Benchmarks")
        print("=" * 70)

        benchmark_fn()

        print("\n" + "=" * 70)
        print("Benchmarks complete!")
        print("=" * 70)
