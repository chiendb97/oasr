#!/usr/bin/env python3
"""Unified OASR benchmark CLI.

Usage:
  # Single test with defaults
  python benchmarks/oasr_benchmark.py --routine gemm

  # Specific subroutine and shape
  python benchmarks/oasr_benchmark.py --routine gemm --subroutine bmm \\
      --batch-count 256 --M 200 --N 200 --K 64

  # Select backends to compare
  python benchmarks/oasr_benchmark.py --routine norm --subroutine layer_norm \\
      --backends cuda torch --batch 64 --seq 250 --hidden 512

  # Correctness check
  python benchmarks/oasr_benchmark.py --routine gemm --backends cutlass torch --refcheck

  # Export to CSV
  python benchmarks/oasr_benchmark.py --routine norm --output_path results.csv

  # Batch test from testlist
  python benchmarks/oasr_benchmark.py --testlist benchmarks/testlists/conformer_base.txt

  # Profiling mode (NVTX markers, run under ncu manually)
  python benchmarks/oasr_benchmark.py --routine gemm --subroutine gemm \\
      --M 200 --N 200 --K 64 --backends cutlass --profile --dry_run_iters 0

  # Verbose output
  python benchmarks/oasr_benchmark.py --routine gemm -vv

  # List available routines
  python benchmarks/oasr_benchmark.py --list
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

# Ensure project root is on sys.path so `benchmarks.routines` is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.routines import get_routine, list_all_subroutines, list_routines
from benchmarks.routines.bench_utils import OutputWriter, get_device_info, profile_kernel


# ---------------------------------------------------------------------------
# Argument parsing (two-stage)
# ---------------------------------------------------------------------------


def _build_global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OASR Unified Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )

    # Stage-1 global args
    parser.add_argument(
        "--routine",
        type=str,
        default=None,
        choices=list_routines(),
        help="Kernel routine family (gemm, norm, conv, activation, composite)",
    )
    parser.add_argument(
        "--subroutine",
        type=str,
        default=None,
        help="Specific kernel within the routine (e.g. bmm, layer_norm, depthwise_conv1d)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backend(s) to benchmark. Per-family defaults: gemm/conv2d → cutlass torch; "
             "norm/conv1d/activation/composite → cuda torch",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type (float16, bfloat16, float32; default: float16)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to write CSV results",
    )
    parser.add_argument(
        "--testlist",
        type=str,
        default=None,
        help="Path to testlist file (one test per line)",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=30,
        help="Number of measurement iterations (default: 30)",
    )
    parser.add_argument(
        "--dry_run_iters",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--refcheck",
        action="store_true",
        help="Verify output correctness between backends",
    )
    parser.add_argument(
        "--allow_output_mismatch",
        action="store_true",
        help="Continue benchmarking even if refcheck fails",
    )
    parser.add_argument(
        "--use_cuda_events",
        action="store_true",
        help="Force CUDA events timing (skip Triton do_bench)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run in profiling mode (Nsight Compute / NVTX)",
    )
    parser.add_argument(
        "--generate_repro_command",
        action="store_true",
        help="Print reproducer command for each test",
    )
    parser.add_argument(
        "--case_tag",
        type=str,
        default=None,
        help="Tag string to include in CSV output",
    )
    parser.add_argument(
        "-v",
        action="count",
        default=0,
        dest="verbosity",
        help="Increase verbosity (-v verbose, -vv very verbose)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available routines and subroutines",
    )
    return parser


def _build_epilog() -> str:
    lines = ["\nAvailable routines and subroutines:"]
    try:
        for routine, subs in list_all_subroutines().items():
            lines.append(f"  {routine}: {', '.join(subs)}")
    except Exception:
        lines.append("  (run with --list to see available routines)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Testlist parsing
# ---------------------------------------------------------------------------


def _parse_testlist(path: str) -> list[list[str]]:
    """Parse a testlist file. Each non-empty, non-comment line is a CLI invocation."""
    tests = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tokens = shlex.split(line)
            except ValueError as e:
                print(f"[WARNING] testlist line {line_num} parse error: {e}")
                continue
            tests.append(tokens)
    return tests


# ---------------------------------------------------------------------------
# Repro command generation
# ---------------------------------------------------------------------------


def _build_repro_command(args: argparse.Namespace, remaining: list[str]) -> str:
    """Build a reproducible CLI command from the parsed args."""
    parts = ["python benchmarks/oasr_benchmark.py"]
    parts.append(f"--routine {args.routine}")
    if args.subroutine:
        parts.append(f"--subroutine {args.subroutine}")
    parts.append(f"--backends {' '.join(args.backends)}")
    parts.append(f"--dtype {args.dtype}")
    parts.append(f"--num_iters {args.num_iters}")
    parts.append(f"--dry_run_iters {args.dry_run_iters}")
    if args.refcheck:
        parts.append("--refcheck")
    if args.use_cuda_events:
        parts.append("--use_cuda_events")
    # Include remaining (routine-specific) args
    if remaining:
        parts.extend(remaining)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Single test execution
# ---------------------------------------------------------------------------


def _run_single_test(
    global_args: argparse.Namespace,
    remaining: list[str],
    output: OutputWriter,
) -> None:
    """Execute a single benchmark test."""
    routine_name = global_args.routine
    if routine_name is None:
        print("[ERROR] --routine is required (or use --testlist / --list)")
        sys.exit(1)

    routine_mod = get_routine(routine_name)

    # Stage-2: parse routine-specific args
    sub_parser = argparse.ArgumentParser(add_help=False)
    routine_mod.parse_args(sub_parser)
    routine_args, _ = sub_parser.parse_known_args(remaining)

    # Merge global + routine args into a single namespace
    merged = argparse.Namespace(**vars(global_args), **vars(routine_args))

    # Default subroutine to first available
    if merged.subroutine is None:
        merged.subroutine = routine_mod.SUBROUTINES[0]

    if merged.subroutine not in routine_mod.SUBROUTINES:
        print(
            f"[ERROR] Unknown subroutine '{merged.subroutine}' for routine '{routine_name}'. "
            f"Available: {routine_mod.SUBROUTINES}"
        )
        sys.exit(1)

    if merged.profile:
        _run_profile_mode(routine_mod, merged)
    else:
        # Generate repro command
        if merged.generate_repro_command:
            repro = _build_repro_command(merged, remaining)
            output.add_repro_command(repro)

        routine_mod.run_test(merged, output)


def _run_profile_mode(routine_mod, args: argparse.Namespace) -> None:
    """Run in profiling mode with NVTX markers."""
    from benchmarks.routines.bench_utils import parse_dtype

    subroutine = args.subroutine
    warmup = args.dry_run_iters
    dtype = parse_dtype(args.dtype)

    configs = routine_mod._resolve_configs(args, subroutine)
    cfg = configs[0]
    oasr_fn, pytorch_fn = routine_mod._setup_for_config(subroutine, cfg, dtype)
    fn_map = routine_mod.get_fn_map(subroutine, oasr_fn, pytorch_fn)
    backends = args.backends or list(fn_map.keys())

    print(f"[INFO] Profiling: {args.routine}/{subroutine}  config={cfg}")
    for name, fn in fn_map.items():
        if name not in backends:
            continue
        print(f"[INFO] Running {name}...")
        profile_kernel(f"{name}_{subroutine}", fn, warmup=warmup)
    print("[INFO] Profiling done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global_parser = _build_global_parser()
    global_args, remaining = global_parser.parse_known_args()

    # --list mode
    if global_args.list:
        print("Available routines and subroutines:\n")
        for routine, subs in list_all_subroutines().items():
            print(f"  {routine}:")
            for sub in subs:
                print(f"    - {sub}")
        return

    # Print device info at verbosity >= 2
    if global_args.verbosity >= 2:
        info = get_device_info()
        print(f"[VVERBOSE] gpu_name = '{info['name']}'")
        print(f"[VVERBOSE] sm = {info['sm']}, memory = {info['memory_gb']} GB")

    output = OutputWriter(
        output_path=global_args.output_path,
        verbosity=global_args.verbosity,
        case_tag=global_args.case_tag,
    )

    if global_args.testlist:
        # Batch mode: run each testlist line
        tests = _parse_testlist(global_args.testlist)
        print(f"[INFO] Running {len(tests)} tests from {global_args.testlist}")

        for i, tokens in enumerate(tests):
            test_args, test_remaining = global_parser.parse_known_args(tokens)
            # Inherit global settings
            test_args.output_path = global_args.output_path
            if global_args.refcheck:
                test_args.refcheck = True
            if global_args.generate_repro_command:
                test_args.generate_repro_command = True
            test_args.verbosity = global_args.verbosity
            test_args.case_tag = global_args.case_tag
            test_args.use_cuda_events = global_args.use_cuda_events

            routine_name = test_args.routine
            if routine_name is None:
                print(f"[WARNING] testlist line {i + 1} missing --routine, skipping")
                continue

            sub = test_args.subroutine or "default"
            output.write_header(f"[{i + 1}/{len(tests)}] {routine_name}/{sub}")

            _run_single_test(test_args, test_remaining, output)
    else:
        # Single test mode
        if global_args.routine is None:
            global_parser.print_help()
            return

        sub = global_args.subroutine
        if sub is None:
            routine_mod = get_routine(global_args.routine)
            sub = routine_mod.SUBROUTINES[0]
        output.write_header(f"{global_args.routine}/{sub}")
        _run_single_test(global_args, remaining, output)

    output.finalize()


if __name__ == "__main__":
    main()
