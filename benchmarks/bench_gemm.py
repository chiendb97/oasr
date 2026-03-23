#!/usr/bin/env python3
"""Performance benchmarks for GEMM kernel with optional autotuning.

Usage:
  # Default benchmark (single tile config)
  python benchmarks/bench_gemm.py

  # Autotune: profile all tile variants, pick best per shape
  python benchmarks/bench_gemm.py --autotune

  # Autotune with persistent cache
  python benchmarks/bench_gemm.py --autotune --cache gemm_tune.json

  # Replay cached configs (no profiling)
  python benchmarks/bench_gemm.py --autotune --cache gemm_tune.json --no-tune

  # Profiling mode for Nsight Compute
  ncu --set full -o gemm_profile python benchmarks/bench_gemm.py \
      --profile --kernel gemm --target oasr --warmup 0 --iters 1
"""

import argparse

import torch
import torch.nn.functional as F

import oasr

try:
    from .utils import run_profile
except ImportError:
    from utils import run_profile


BENCH_CONFIGS = [
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


def setup_gemm(M, N, K, dtype=torch.float16):
    """Setup tensors for single GEMM: D = A @ B."""
    A = torch.randn(M, K, device='cuda', dtype=dtype)
    B = torch.randn(N, K, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.gemm(A, B)

    def pytorch_fn():
        return F.linear(A, B)

    return oasr_fn, pytorch_fn


# =============================================================================
# Plain benchmark (default tile config only)
# =============================================================================


def benchmark_gemm():
    """Benchmark GEMM with default config: OASR vs PyTorch."""
    import triton

    print("\n" + "=" * 70)
    print("GEMM Benchmark (default tile config)")
    print("=" * 70)
    print(
        f"\n{'Shape (M, N, K)':<25} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 65)

    for M, N, K in BENCH_CONFIGS:
        oasr_fn, pytorch_fn = setup_gemm(M, N, K)
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        speedup = pytorch_ms / oasr_ms
        shape_str = f"({M}, {N}, {K})"
        print(f"{shape_str:<25} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


# =============================================================================
# Autotuning benchmark
# =============================================================================


def _format_tactic(tactic):
    """Format a Tactic's config as a compact string."""
    cfg = dict(tactic.config)
    parts = [f"{cfg['tile_m']}x{cfg['tile_n']}x{cfg['tile_k']}"]
    parts.append(f"w{cfg['warp_m']}x{cfg['warp_n']}x{cfg['warp_k']}")
    parts.append(f"s{cfg['stages']}")
    if cfg.get('split_k', 1) > 1:
        parts.append(f"sk{cfg['split_k']}")
    return "_".join(parts)


def benchmark_gemm_autotune(cache_path=None, tune_mode=True):
    """Benchmark GEMM with autotuning: profile all tile variants per shape.

    For each problem size, compares:
      - Autotuned OASR (best tile config selected by profiler)
      - Default OASR (fixed 128x128x32 config)
      - PyTorch (cuBLAS via F.linear)
    """
    import triton
    from oasr.tune import get_selected_config
    from oasr.tune._types import OpKey
    from oasr.tune.kernel_configs import GEMM_TILE_CONFIGS
    import oasr.tune as _tune_mod

    num_variants = len(GEMM_TILE_CONFIGS)
    mode_str = "PROFILING" if tune_mode else "CACHED"

    print("\n" + "=" * 110)
    print(f"GEMM Autotuning Benchmark ({mode_str}, {num_variants} tile variants)")
    print("=" * 110)
    if cache_path:
        print(f"Cache: {cache_path}")
    print()

    sm = torch.cuda.get_device_capability()
    sm_version = sm[0] * 10 + sm[1]

    with oasr.autotune(tune_mode, cache=cache_path):
        print(f"{'Shape (M,N,K)':<25} {'Best (ms)':<11} {'Config':<35} "
              f"{'Default (ms)':<14} {'PyTorch (ms)':<14} {'vs Def':<9} {'vs PT':<9}")
        print("-" * 120)

        for M, N, K in BENCH_CONFIGS:
            # 1) Autotuned OASR — triggers profiling on first call
            oasr_fn, pytorch_fn = setup_gemm(M, N, K)
            best_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)

            # Look up which tactic was selected
            tactic = get_selected_config(
                op_key=OpKey("gemm", "gemm"),
                shape_sig=(M, N, K),
                dtype="float16",
                device_sm=sm_version,
            )
            config_str = _format_tactic(tactic) if tactic else "default"

            # 2) Default OASR (tuning disabled, uses fixed default config)
            prev = _tune_mod._enabled
            _tune_mod._enabled = False
            default_fn, _ = setup_gemm(M, N, K)
            default_ms = triton.testing.do_bench(default_fn, warmup=100, rep=500)
            _tune_mod._enabled = prev

            # 3) PyTorch baseline
            pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

            vs_default = default_ms / best_ms
            vs_pytorch = pytorch_ms / best_ms
            shape_str = f"({M}, {N}, {K})"

            print(f"{shape_str:<25} {best_ms:<11.4f} {config_str:<35} "
                  f"{default_ms:<14.4f} {pytorch_ms:<14.4f} {vs_default:<9.2f}x {vs_pytorch:<9.2f}x")

    print("\n" + "=" * 110)
    print("Autotuning benchmark complete!")
    print("=" * 110)


# =============================================================================
# Profile mode (for Nsight Compute)
# =============================================================================

PROFILE_CONFIGS = {
    'gemm': (16000, 256, 2048),
}


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="OASR GEMM Kernel - Benchmark, Autotune & Profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default benchmark
  python benchmarks/bench_gemm.py

  # Autotune all shapes
  python benchmarks/bench_gemm.py --autotune

  # Autotune and save cache for reuse
  python benchmarks/bench_gemm.py --autotune --cache gemm_tune.json

  # Replay cached configs (no profiling)
  python benchmarks/bench_gemm.py --autotune --cache gemm_tune.json --no-tune

  # Nsight Compute profiling
  ncu --set full -o gemm_profile python benchmarks/bench_gemm.py \\
      --profile --kernel gemm --target oasr --warmup 0 --iters 1
        """,
    )

    parser.add_argument('--autotune', action='store_true',
                        help='Enable autotuning to find best tile config per shape')
    parser.add_argument('--cache', type=str, default=None,
                        help='Path to JSON cache file for autotuning results')
    parser.add_argument('--no-tune', action='store_true',
                        help='Use cached configs only, skip profiling (requires --cache)')
    parser.add_argument('--profile', action='store_true',
                        help='Run in profiling mode (single iterations for ncu)')
    parser.add_argument('--kernel', nargs='+', default=['all'],
                        help='Kernel(s) to profile')
    parser.add_argument('--target', choices=['oasr', 'pytorch', 'both'], default='oasr',
                        help='Which implementation to profile (default: oasr)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations for ncu profiling (default: 3)')
    parser.add_argument('--iters', type=int, default=1,
                        help='Profile iterations for ncu (default: 1)')

    args = parser.parse_args()

    if args.profile:
        kernels = list(PROFILE_CONFIGS.keys()) if 'all' in args.kernel else args.kernel
        setup_funcs = {
            'gemm': lambda: setup_gemm(*PROFILE_CONFIGS['gemm']),
        }
        run_profile("GEMM Kernel", PROFILE_CONFIGS, setup_funcs,
                     kernels, target=args.target,
                     warmup=args.warmup, iters=args.iters)
    elif args.autotune:
        tune_mode = not args.no_tune
        benchmark_gemm_autotune(cache_path=args.cache, tune_mode=tune_mode)
    else:
        print("=" * 70)
        print("OASR GEMM Kernel - Performance Benchmarks")
        print("=" * 70)
        benchmark_gemm()
        print("\n" + "=" * 70)
        print("Benchmarks complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
