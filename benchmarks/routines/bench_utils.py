"""Shared infrastructure for OASR benchmark routines.

Provides timing, performance metrics, structured output (terminal + CSV),
correctness checking, and device utilities.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import math
import statistics
import sys
from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _cuda_events_bench(
    fn: Callable, dry_run_iters: int = 5, num_iters: int = 30
) -> tuple[float, float]:
    """Time *fn* with CUDA events. Returns (median_ms, std_ms)."""
    for _ in range(dry_run_iters):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return median, std


try:
    import triton.testing as _triton_testing

    def _triton_bench(
        fn: Callable, dry_run_iters: int = 5, num_iters: int = 30
    ) -> tuple[float, float]:
        """Time *fn* with Triton do_bench. Returns (median_ms, std_ms).

        Triton's do_bench only returns the median; we run a second lightweight
        pass with CUDA events to capture std when num_iters is large enough.
        """
        median = _triton_testing.do_bench(fn, warmup=dry_run_iters, rep=num_iters)
        # Quick std estimate via CUDA events (small pass)
        _, std = _cuda_events_bench(fn, dry_run_iters=2, num_iters=min(num_iters, 10))
        return median, std

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def bench_fn(
    fn: Callable,
    dry_run_iters: int = 5,
    num_iters: int = 30,
    use_cuda_events: bool = False,
) -> tuple[float, float]:
    """Time *fn* and return ``(median_ms, std_ms)``.

    Uses Triton ``do_bench`` by default, falls back to CUDA events if Triton
    is not installed or ``use_cuda_events=True``.
    """
    if use_cuda_events or not _HAS_TRITON:
        return _cuda_events_bench(fn, dry_run_iters=dry_run_iters, num_iters=num_iters)
    return _triton_bench(fn, dry_run_iters=dry_run_iters, num_iters=num_iters)


def profile_kernel(name: str, fn: Callable, warmup: int = 3, profile_iters: int = 1):
    """Run kernel for profiling with Nsight Compute (NVTX markers)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(name)
    for _ in range(profile_iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def compute_gemm_tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Compute TFLOPS for a GEMM operation (2*M*N*K FLOPs)."""
    if time_ms <= 0:
        return 0.0
    return 2.0 * M * N * K / (time_ms * 1e-3) / 1e12


def compute_bmm_tflops(B: int, M: int, N: int, K: int, time_ms: float) -> float:
    """Compute TFLOPS for a batched GEMM operation."""
    if time_ms <= 0:
        return 0.0
    return 2.0 * B * M * N * K / (time_ms * 1e-3) / 1e12


def compute_bandwidth_tb_s(bytes_accessed: int, time_ms: float) -> float:
    """Compute memory bandwidth in TB/s for memory-bound kernels."""
    if time_ms <= 0:
        return 0.0
    return bytes_accessed / (time_ms * 1e-3) / 1e12


def dtype_size(dtype: torch.dtype) -> int:
    """Return byte size of a torch dtype."""
    return torch.tensor([], dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------


def check_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> tuple[bool, float]:
    """Compare two tensors. Returns (passed, max_abs_diff)."""
    diff = (actual.float() - expected.float()).abs()
    max_diff = diff.max().item()
    passed = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    return passed, max_diff


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def parse_dtype(s: str) -> torch.dtype:
    """Convert a human-friendly dtype string to ``torch.dtype``."""
    key = s.strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{s}'. Choose from: {list(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[key]


def get_device_info() -> dict[str, Any]:
    """Return a dict with GPU name, SM version, and memory."""
    if not torch.cuda.is_available():
        return {"name": "N/A", "sm": "N/A", "memory_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm": f"{props.major}.{props.minor}",
        "memory_gb": round(props.total_mem / 1e9, 1),
    }


# ---------------------------------------------------------------------------
# BenchResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchResult:
    """Structured result of a single benchmark measurement."""

    routine: str
    subroutine: str
    backend: str  # "oasr" or "pytorch"
    shape: str
    dtype: str
    median_ms: float
    std_ms: float = 0.0
    tflops: float | None = None
    bandwidth_tb_s: float | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        extra = d.pop("extra", {})
        d.update(extra)
        return d


# ---------------------------------------------------------------------------
# OutputWriter
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "routine",
    "subroutine",
    "backend",
    "shape",
    "dtype",
    "median_ms",
    "std_ms",
    "tflops",
    "bandwidth_tb_s",
    "device",
    "case_tag",
    "repro_command",
]


class OutputWriter:
    """Manages structured output to terminal and optional CSV file.

    Terminal output follows the ``[PERF]`` log-line convention::

        [PERF] oasr    :: median time 0.145 ms; std 0.002 ms; achieved tflops 125.3 TFLOPs/sec; achieved tb_per_sec 1.87 TB/sec
    """

    def __init__(
        self,
        output_path: str | None = None,
        verbosity: int = 0,
        case_tag: str | None = None,
    ):
        self.output_path = output_path
        self.verbosity = verbosity
        self.case_tag = case_tag
        self._results: list[BenchResult] = []
        self._csv_file: io.TextIOWrapper | None = None
        self._csv_writer: csv.DictWriter | None = None
        self._repro_commands: list[str] = []

        if output_path:
            self._csv_file = open(output_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_COLUMNS)
            self._csv_writer.writeheader()

    # -- Terminal output ----------------------------------------------------

    def write_header(self, title: str) -> None:
        """Print a section header to the terminal."""
        print(f"\n[INFO] {title}")

    def write_verbose(self, msg: str, level: int = 1) -> None:
        """Print a message if verbosity >= level."""
        if self.verbosity >= level:
            tag = "[VERBOSE]" if level == 1 else "[VVERBOSE]"
            print(f"{tag} {msg}")

    def write_result(self, result: BenchResult) -> None:
        """Print a [PERF] line and append to CSV."""
        self._results.append(result)

        # Build [PERF] line
        parts = [f"median time {result.median_ms:.3f} ms"]
        parts.append(f"std {result.std_ms:.3f} ms")
        if result.tflops is not None and result.tflops > 0:
            parts.append(f"achieved tflops {result.tflops:.1f} TFLOPs/sec")
        if result.bandwidth_tb_s is not None and result.bandwidth_tb_s > 0:
            parts.append(f"achieved tb_per_sec {result.bandwidth_tb_s:.2f} TB/sec")

        print(f"[PERF] {result.backend:<12} :: {'; '.join(parts)}")

        # CSV
        if self._csv_writer is not None:
            device_info = get_device_info()
            repro = self._repro_commands[-1] if self._repro_commands else ""
            row = {
                "routine": result.routine,
                "subroutine": result.subroutine,
                "backend": result.backend,
                "shape": result.shape,
                "dtype": result.dtype,
                "median_ms": f"{result.median_ms:.4f}",
                "std_ms": f"{result.std_ms:.4f}",
                "tflops": f"{result.tflops:.2f}" if result.tflops is not None else "",
                "bandwidth_tb_s": (
                    f"{result.bandwidth_tb_s:.4f}" if result.bandwidth_tb_s is not None else ""
                ),
                "device": device_info["name"],
                "case_tag": self.case_tag or "",
                "repro_command": repro,
            }
            self._csv_writer.writerow(row)

    def add_repro_command(self, cmd: str) -> None:
        """Store a reproducer command for the current test."""
        self._repro_commands.append(cmd)
        print(f"[REPRO] {cmd}")

    def finalize(self) -> None:
        """Print summary and close CSV file if open."""
        print(f"\n[INFO] Benchmarks complete! ({len(self._results)} measurements)")
        if self._csv_file is not None:
            self._csv_file.close()
            print(f"[INFO] Results saved to: {self.output_path}")


# ---------------------------------------------------------------------------
# Standalone helpers (backwards-compat with old utils.py)
# ---------------------------------------------------------------------------

import argparse


def make_bench_parser(description: str, kernels: list[str]) -> argparse.ArgumentParser:
    """Create a standard benchmark argument parser (legacy API)."""
    kernel_list = ", ".join(kernels)
    parser = argparse.ArgumentParser(
        description=f"OASR {description} - Benchmark & Profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"\nAvailable kernels:\n  {kernel_list}\n",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Run in profiling mode (single iterations for ncu)",
    )
    parser.add_argument(
        "--kernel", nargs="+", default=["all"],
        help='Kernel(s) to profile (use "all" for all kernels)',
    )
    parser.add_argument(
        "--target", choices=["oasr", "pytorch", "both"], default="oasr",
        help="Which implementation to profile (default: oasr)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations for profiling (default: 3)",
    )
    parser.add_argument(
        "--iters", type=int, default=1,
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
    """Run profiling for specified kernels (legacy API)."""
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
            print("  Running OASR kernel...")
            profile_kernel(f"oasr_{kernel_name}", oasr_fn, warmup=warmup, profile_iters=iters)

        if target in ("pytorch", "both"):
            print("  Running PyTorch kernel...")
            profile_kernel(
                f"pytorch_{kernel_name}", pytorch_fn, warmup=warmup, profile_iters=iters
            )

        print("  Done.")

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)


def run_main(title: str, profile_configs: dict, setup_funcs: dict, benchmark_fn: Callable):
    """Standard main entry point for benchmark scripts (legacy API)."""
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
