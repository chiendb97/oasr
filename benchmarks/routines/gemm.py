"""GEMM family benchmark routines (gemm, bmm, group_gemm, gemm_activation)."""

from __future__ import annotations

import argparse
from typing import Any, Callable

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import (
    BenchResult,
    OutputWriter,
    bench_fn,
    check_close,
    compute_bmm_tflops,
    compute_gemm_tflops,
    make_bench_parser,
    profile_kernel,
    run_profile,
)

SUBROUTINES = ["gemm", "bmm", "group_gemm", "gemm_activation"]

# ---------------------------------------------------------------------------
# Default configs (from existing bench_*.py files)
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "gemm": [
        {"M": 8000, "N": 256, "K": 256},
        {"M": 16000, "N": 256, "K": 256},
        {"M": 16000, "N": 2048, "K": 256},
        {"M": 16000, "N": 256, "K": 2048},
        {"M": 16000, "N": 512, "K": 512},
        {"M": 16000, "N": 2048, "K": 512},
        {"M": 16000, "N": 512, "K": 2048},
        {"M": 32000, "N": 256, "K": 256},
        {"M": 32000, "N": 512, "K": 512},
    ],
    "bmm": [
        {"B": 256, "M": 200, "N": 200, "K": 64},
        {"B": 512, "M": 400, "N": 400, "K": 64},
        {"B": 512, "M": 200, "N": 200, "K": 64},
        {"B": 64, "M": 200, "N": 200, "K": 64},
    ],
    "group_gemm": [
        {"num_groups": 32, "M": 256, "N": 64, "K": 64},
        {"num_groups": 32, "M": 256, "N": 128, "K": 64},
        {"num_groups": 64, "M": 16000, "N": 256, "K": 2048},
        {"num_groups": 64, "M": 16000, "N": 512, "K": 2048},
    ],
    "gemm_activation": [
        {"M": 16000, "N": 256, "K": 256},
        {"M": 16000, "N": 512, "K": 512},
        {"M": 16000, "N": 2048, "K": 256},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "gemm": (16000, 256, 2048),
    "bmm": (256, 200, 200, 64),
    "group_gemm": (64, 200, 64, 64),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------


def setup_gemm(M: int, N: int, K: int, dtype=torch.float16):
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(N, K, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.gemm(A, B)

    def pytorch_fn():
        return F.linear(A, B)

    return oasr_fn, pytorch_fn


def setup_bmm(B: int, M: int, N: int, K: int, dtype=torch.float16):
    A = torch.randn(B, M, K, device="cuda", dtype=dtype)
    Bmat = torch.randn(B, N, K, device="cuda", dtype=dtype)
    B_transposed = Bmat.transpose(1, 2).contiguous()

    def oasr_fn():
        return oasr.bmm(A, Bmat)

    def pytorch_fn():
        return torch.bmm(A, B_transposed)

    return oasr_fn, pytorch_fn


def setup_group_gemm(problem_sizes: list[tuple[int, int, int]], dtype=torch.bfloat16):
    if len(problem_sizes) == 0:
        raise ValueError("problem_sizes must be non-empty")

    _, N, K = problem_sizes[0]
    Ms = [m for m, _, _ in problem_sizes]
    assert all(
        n == N and k == K for _, n, k in problem_sizes
    ), "All grouped GEMM problems must share the same N and K"

    num_problems = len(problem_sizes)
    L = sum(Ms)

    A = torch.randn(L, K, device="cuda", dtype=dtype)
    B = torch.randn(num_problems, N, K, device="cuda", dtype=dtype)
    B_transposed = B.transpose(1, 2).contiguous()
    offset = torch.cumsum(
        torch.tensor(Ms, dtype=torch.int32, device="cuda"), dim=0, dtype=torch.int32
    )

    def oasr_fn():
        return oasr.group_gemm(A, B, offset)

    def pytorch_fn():
        try:
            return torch.nn.functional.grouped_mm(A, B_transposed, offs=offset)
        except Exception:
            D = torch.zeros(L, N, device="cuda", dtype=dtype)
            s_idx = 0
            for i in range(num_problems):
                m_i = Ms[i]
                D[s_idx : s_idx + m_i] = torch.matmul(A[s_idx : s_idx + m_i], B_transposed[i])
                s_idx += m_i
            return D

    return oasr_fn, pytorch_fn


def setup_gemm_activation(M: int, N: int, K: int, dtype=torch.float16):
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(N, K, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.gemm_activation(A, B, activation_type=oasr.ACTIVATION_SWISH)

    def pytorch_fn():
        return F.silu(F.linear(A, B))

    return oasr_fn, pytorch_fn


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    """Add GEMM-specific CLI arguments."""
    parser.add_argument("--M", type=int, default=None, help="M dimension")
    parser.add_argument("--N", type=int, default=None, help="N dimension")
    parser.add_argument("--K", type=int, default=None, help="K dimension")
    parser.add_argument("--batch-count", type=int, default=None, help="Batch count (bmm/group_gemm)")
    parser.add_argument("--autotune", action="store_true", help="Enable autotuning (gemm only)")
    parser.add_argument("--cache", type=str, default=None, help="Autotune cache path")
    parser.add_argument("--no-tune", action="store_true", help="Use cached configs only")


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    """Run a single GEMM-family benchmark test."""
    subroutine = getattr(args, "subroutine", "gemm")
    dtype_str = getattr(args, "dtype", "float16")
    dtype = torch.float16
    if dtype_str:
        from benchmarks.routines.bench_utils import parse_dtype
        dtype = parse_dtype(dtype_str)

    backends = getattr(args, "backends", ["oasr", "pytorch"])
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    configs = _resolve_configs(args, subroutine)
    backend_fns = {"oasr": 0, "pytorch": 1}

    for cfg in configs:
        oasr_fn, pytorch_fn = _setup_for_config(subroutine, cfg, dtype)
        fn_map = {"oasr": oasr_fn, "pytorch": pytorch_fn}
        shape_str = _shape_str(subroutine, cfg)

        output.write_verbose(f"shape = {shape_str}, dtype = {dtype_str}", level=2)

        if do_check and "oasr" in backends and "pytorch" in backends:
            oasr_out = oasr_fn()
            pytorch_out = pytorch_fn()
            passed, max_diff = check_close(oasr_out, pytorch_out)
            if not passed:
                msg = f"[ERROR] Output mismatch for {shape_str} (max_diff={max_diff:.6f})"
                print(msg)
                if not allow_mismatch:
                    continue

        for backend in backends:
            if backend not in fn_map:
                continue
            median_ms, std_ms = bench_fn(
                fn_map[backend],
                dry_run_iters=dry_run_iters,
                num_iters=num_iters,
                use_cuda_events=use_cuda_events,
            )
            tflops = _compute_tflops(subroutine, cfg, median_ms)
            output.write_result(BenchResult(
                routine="gemm",
                subroutine=subroutine,
                backend=backend,
                shape=shape_str,
                dtype=dtype_str,
                median_ms=median_ms,
                std_ms=std_ms,
                tflops=tflops,
            ))


def _resolve_configs(args: argparse.Namespace, subroutine: str) -> list[dict]:
    """Build config list from CLI args or defaults."""
    M = getattr(args, "M", None)
    N = getattr(args, "N", None)
    K = getattr(args, "K", None)
    batch_count = getattr(args, "batch_count", None)

    if subroutine == "bmm" and all(v is not None for v in [M, N, K]) and batch_count is not None:
        return [{"B": batch_count, "M": M, "N": N, "K": K}]
    elif subroutine == "group_gemm" and all(v is not None for v in [M, N, K]):
        num_groups = batch_count or 32
        return [{"num_groups": num_groups, "M": M, "N": N, "K": K}]
    elif subroutine in ("gemm", "gemm_activation") and all(v is not None for v in [M, N, K]):
        return [{"M": M, "N": N, "K": K}]

    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["gemm"])


def _setup_for_config(subroutine: str, cfg: dict, dtype: torch.dtype):
    """Return (oasr_fn, pytorch_fn) for a given subroutine and config."""
    if subroutine == "gemm":
        return setup_gemm(cfg["M"], cfg["N"], cfg["K"], dtype)
    elif subroutine == "bmm":
        return setup_bmm(cfg["B"], cfg["M"], cfg["N"], cfg["K"], dtype)
    elif subroutine == "group_gemm":
        num_groups = cfg["num_groups"]
        M, N, K = cfg["M"], cfg["N"], cfg["K"]
        torch.manual_seed(0)
        low = max(16, M // 2)
        high = max(low + 1, M * 2)
        Ms = torch.randint(low=low, high=high, size=(num_groups,), device="cuda").tolist()
        problem_sizes = [(m_i, N, K) for m_i in Ms]
        return setup_group_gemm(problem_sizes, dtype)
    elif subroutine == "gemm_activation":
        return setup_gemm_activation(cfg["M"], cfg["N"], cfg["K"], dtype)
    else:
        raise ValueError(f"Unknown gemm subroutine: {subroutine}")


def _compute_tflops(subroutine: str, cfg: dict, time_ms: float) -> float:
    if subroutine == "bmm":
        return compute_bmm_tflops(cfg["B"], cfg["M"], cfg["N"], cfg["K"], time_ms)
    elif subroutine == "group_gemm":
        # Approximate: num_groups * M * N * K
        return compute_gemm_tflops(
            cfg["num_groups"] * cfg["M"], cfg["N"], cfg["K"], time_ms
        )
    else:  # gemm, gemm_activation
        return compute_gemm_tflops(cfg["M"], cfg["N"], cfg["K"], time_ms)


def _shape_str(subroutine: str, cfg: dict) -> str:
    if subroutine == "bmm":
        return f"({cfg['B']}, {cfg['M']}, {cfg['N']}, {cfg['K']})"
    elif subroutine == "group_gemm":
        return f"{cfg['num_groups']}x({cfg['M']}, {cfg['N']}, {cfg['K']})"
    else:
        return f"({cfg['M']}, {cfg['N']}, {cfg['K']})"


# ---------------------------------------------------------------------------
# Standalone entry (backwards compat)
# ---------------------------------------------------------------------------


def run_standalone() -> None:
    """Standalone entry point replicating the old bench_gemm.py / bench_bmm.py behavior."""
    import sys

    # Detect which subroutine based on caller filename
    caller = sys.argv[0]
    if "bmm" in caller and "group" not in caller:
        _run_standalone_bmm()
    elif "group_gemm" in caller:
        _run_standalone_group_gemm()
    else:
        _run_standalone_gemm()


def _run_standalone_gemm() -> None:
    """Replicate bench_gemm.py main()."""
    parser = argparse.ArgumentParser(
        description="OASR GEMM Kernel - Benchmark, Autotune & Profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--kernel", nargs="+", default=["all"])
    parser.add_argument("--target", choices=["oasr", "pytorch", "both"], default="oasr")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    if args.profile:
        kernels = list(PROFILE_CONFIGS.keys()) if "all" in args.kernel else args.kernel
        setup_funcs = {"gemm": lambda: setup_gemm(*PROFILE_CONFIGS["gemm"])}
        run_profile(
            "GEMM Kernel", {"gemm": PROFILE_CONFIGS["gemm"]}, setup_funcs,
            kernels, target=args.target, warmup=args.warmup, iters=args.iters,
        )
    elif args.autotune:
        _benchmark_gemm_autotune(cache_path=args.cache, tune_mode=not args.no_tune)
    else:
        output = OutputWriter()
        output.write_header("GEMM Benchmark (default tile config)")
        for cfg in DEFAULT_CONFIGS["gemm"]:
            oasr_fn, pytorch_fn = setup_gemm(cfg["M"], cfg["N"], cfg["K"])
            shape = f"({cfg['M']}, {cfg['N']}, {cfg['K']})"
            for backend, fn in [("oasr", oasr_fn), ("pytorch", pytorch_fn)]:
                median_ms, std_ms = bench_fn(fn)
                tflops = compute_gemm_tflops(cfg["M"], cfg["N"], cfg["K"], median_ms)
                output.write_result(BenchResult(
                    routine="gemm", subroutine="gemm", backend=backend,
                    shape=shape, dtype="float16",
                    median_ms=median_ms, std_ms=std_ms, tflops=tflops,
                ))
        output.finalize()


def _run_standalone_bmm() -> None:
    """Replicate bench_bmm.py main()."""
    setup_funcs = {"bmm": lambda: setup_bmm(*PROFILE_CONFIGS["bmm"])}

    def benchmark():
        output = OutputWriter()
        output.write_header("Batched GEMM (BMM) Benchmark")
        for cfg in DEFAULT_CONFIGS["bmm"]:
            oasr_fn, pytorch_fn = setup_bmm(cfg["B"], cfg["M"], cfg["N"], cfg["K"])
            shape = f"({cfg['B']}, {cfg['M']}, {cfg['N']}, {cfg['K']})"
            for backend, fn in [("oasr", oasr_fn), ("pytorch", pytorch_fn)]:
                median_ms, std_ms = bench_fn(fn)
                tflops = compute_bmm_tflops(cfg["B"], cfg["M"], cfg["N"], cfg["K"], median_ms)
                output.write_result(BenchResult(
                    routine="gemm", subroutine="bmm", backend=backend,
                    shape=shape, dtype="float16",
                    median_ms=median_ms, std_ms=std_ms, tflops=tflops,
                ))
        output.finalize()

    from benchmarks.routines.bench_utils import run_main
    run_main("Batched GEMM (BMM) Kernel", {"bmm": PROFILE_CONFIGS["bmm"]}, setup_funcs, benchmark)


def _run_standalone_group_gemm() -> None:
    """Replicate bench_group_gemm.py main()."""
    profile_cfg = PROFILE_CONFIGS["group_gemm"]
    setup_funcs = {
        "group_gemm": lambda: setup_group_gemm(
            [profile_cfg[1:]] * profile_cfg[0]
            if len(profile_cfg) == 4
            else [(profile_cfg[1], profile_cfg[2], profile_cfg[3])] * profile_cfg[0],
        ),
    }

    def benchmark():
        output = OutputWriter()
        output.write_header("Grouped GEMM Benchmark")
        for cfg in DEFAULT_CONFIGS["group_gemm"]:
            num_groups = cfg["num_groups"]
            M, N, K = cfg["M"], cfg["N"], cfg["K"]
            torch.manual_seed(0)
            low = max(16, M // 2)
            high = max(low + 1, M * 2)
            Ms = torch.randint(low=low, high=high, size=(num_groups,), device="cuda").tolist()
            problem_sizes = [(m_i, N, K) for m_i in Ms]
            oasr_fn, pytorch_fn = setup_group_gemm(problem_sizes)
            config_str = f"{num_groups}x(M~{M}, N={N}, K={K})"
            for backend, fn in [("oasr", oasr_fn), ("pytorch", pytorch_fn)]:
                median_ms, std_ms = bench_fn(fn)
                tflops = compute_gemm_tflops(num_groups * M, N, K, median_ms)
                output.write_result(BenchResult(
                    routine="gemm", subroutine="group_gemm", backend=backend,
                    shape=config_str, dtype="bfloat16",
                    median_ms=median_ms, std_ms=std_ms, tflops=tflops,
                ))
        output.finalize()

    from benchmarks.routines.bench_utils import run_main
    run_main(
        "Grouped GEMM Kernel",
        {"group_gemm": PROFILE_CONFIGS["group_gemm"]},
        setup_funcs,
        benchmark,
    )


def _benchmark_gemm_autotune(cache_path=None, tune_mode=True):
    """Autotuning benchmark (preserved from bench_gemm.py)."""
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
        print(
            f"{'Shape (M,N,K)':<25} {'Best (ms)':<11} {'Config':<35} "
            f"{'Default (ms)':<14} {'PyTorch (ms)':<14} {'vs Def':<9} {'vs PT':<9}"
        )
        print("-" * 120)

        for cfg in DEFAULT_CONFIGS["gemm"]:
            M, N, K = cfg["M"], cfg["N"], cfg["K"]
            oasr_fn, pytorch_fn = setup_gemm(M, N, K)
            best_ms, _ = bench_fn(oasr_fn)

            tactic = get_selected_config(
                op_key=OpKey("gemm", "gemm"),
                shape_sig=(M, N, K),
                dtype="float16",
                device_sm=sm_version,
            )

            def _format_tactic(t):
                c = dict(t.config)
                parts = [f"{c['tile_m']}x{c['tile_n']}x{c['tile_k']}"]
                parts.append(f"w{c['warp_m']}x{c['warp_n']}x{c['warp_k']}")
                parts.append(f"s{c['stages']}")
                if c.get("split_k", 1) > 1:
                    parts.append(f"sk{c['split_k']}")
                return "_".join(parts)

            config_str = _format_tactic(tactic) if tactic else "default"

            prev = _tune_mod._enabled
            _tune_mod._enabled = False
            default_fn, _ = setup_gemm(M, N, K)
            default_ms, _ = bench_fn(default_fn)
            _tune_mod._enabled = prev

            pytorch_ms, _ = bench_fn(pytorch_fn)
            vs_default = default_ms / best_ms if best_ms > 0 else 0
            vs_pytorch = pytorch_ms / best_ms if best_ms > 0 else 0
            shape_str = f"({M}, {N}, {K})"
            print(
                f"{shape_str:<25} {best_ms:<11.4f} {config_str:<35} "
                f"{default_ms:<14.4f} {pytorch_ms:<14.4f} {vs_default:<9.2f}x {vs_pytorch:<9.2f}x"
            )

    print("\n" + "=" * 110)
    print("Autotuning benchmark complete!")
    print("=" * 110)
