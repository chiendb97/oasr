"""Pooling benchmark routines."""

from __future__ import annotations

import argparse
import statistics
from typing import Any, Callable

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import (
    BenchResult,
    OutputWriter,
    bench_fn,
    check_close,
    compute_bandwidth_tb_s,
    dtype_size,
    parse_dtype,
    profile_kernel,
)
from oasr.functionals.pooling import _avg_pool1d_output_length

SUBROUTINES = ["avg_pool1d"]

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "avg_pool1d": [
        {
            "batch": batch,
            "seq": 1500,
            "channels": 1280,
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True,
        }
        for batch in (1, 2, 4, 8)
    ]
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


def setup_avg_pool1d(
    batch: int,
    seq: int,
    channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: torch.dtype = torch.float16,
):
    """Return allocation-including OASR/BTC and PyTorch/BCT closures."""
    x_btc = torch.randn(batch, seq, channels, device="cuda", dtype=dtype)
    x_bct = x_btc.transpose(1, 2).contiguous()

    def oasr_fn():
        return oasr.avg_pool1d(
            x_btc,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        )

    def pytorch_fn():
        # BCT is pre-created: this is the raw PyTorch/cuDNN pooling baseline.
        return F.avg_pool1d(
            x_bct,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        ).transpose(1, 2)

    return oasr_fn, pytorch_fn


def _setup_previous_model(cfg: dict[str, Any], dtype: torch.dtype):
    """The exact pre-KG4 Speech-LLM BTC -> BCT -> BTC expression."""
    x = torch.randn(cfg["batch"], cfg["seq"], cfg["channels"], device="cuda", dtype=dtype)

    def previous_model_fn():
        return (
            F.avg_pool1d(
                x.transpose(1, 2),
                cfg["kernel_size"],
                cfg["stride"],
                cfg["padding"],
                cfg["ceil_mode"],
                cfg["count_include_pad"],
            )
            .transpose(1, 2)
            .contiguous()
        )

    return x, previous_model_fn


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Input time extent")
    parser.add_argument("--channels", type=int, default=None, help="Channel dimension")
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--ceil-mode", action="store_true")
    parser.add_argument("--exclude-pad", action="store_true")


def _pool_bytes(cfg: dict[str, Any], dtype: torch.dtype) -> int:
    output_length = _avg_pool1d_output_length(
        cfg["seq"],
        cfg["kernel_size"],
        cfg["stride"],
        cfg["padding"],
        cfg["ceil_mode"],
    )
    elements = cfg["batch"] * cfg["channels"] * (cfg["seq"] + output_length)
    return elements * dtype_size(dtype)


def _shape_str(cfg: dict[str, Any]) -> str:
    return (
        f"[{cfg['batch']},{cfg['seq']},{cfg['channels']}] "
        f"k={cfg['kernel_size']} s={cfg['stride']} p={cfg['padding']}"
    )


def _resolve_configs(args: argparse.Namespace, subroutine: str):
    del subroutine
    batch = getattr(args, "batch", None)
    seq = getattr(args, "seq", None)
    channels = getattr(args, "channels", None)
    if batch is not None and seq is not None and channels is not None:
        return [
            {
                "batch": batch,
                "seq": seq,
                "channels": channels,
                "kernel_size": getattr(args, "kernel_size", 2),
                "stride": getattr(args, "stride", 2),
                "padding": getattr(args, "padding", 0),
                "ceil_mode": getattr(args, "ceil_mode", False),
                "count_include_pad": not getattr(args, "exclude_pad", False),
            }
        ]
    return DEFAULT_CONFIGS["avg_pool1d"]


def _setup_for_config(subroutine: str, cfg: dict[str, Any], dtype: torch.dtype):
    if subroutine != "avg_pool1d":
        raise ValueError(f"Unknown pooling subroutine: {subroutine}")
    return setup_avg_pool1d(**cfg, dtype=dtype)


def get_fn_map(subroutine: str, cuda_fn: Callable, torch_fn: Callable):
    del subroutine
    return {"cuda": cuda_fn, "torch": torch_fn}


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "avg_pool1d")
    dtype_str = getattr(args, "dtype", "float16")
    dtype = parse_dtype(dtype_str)
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    for cfg in _resolve_configs(args, subroutine):
        cuda_fn, torch_fn = _setup_for_config(subroutine, cfg, dtype)
        fn_map = get_fn_map(subroutine, cuda_fn, torch_fn)
        backends = getattr(args, "backends", None) or list(fn_map)
        shape = _shape_str(cfg)

        if do_check and "cuda" in backends and "torch" in backends:
            passed, max_diff = check_close(cuda_fn(), torch_fn())
            if not passed:
                print(f"  [ERROR] Output mismatch for {shape} (max_diff={max_diff:.6f})")
                if not allow_mismatch:
                    continue

        for backend in backends:
            if backend not in fn_map:
                print(f"  [WARNING] Unknown backend {backend!r}, skipping")
                continue
            median_ms, std_ms = bench_fn(
                fn_map[backend],
                dry_run_iters=dry_run_iters,
                num_iters=num_iters,
                use_cuda_events=use_cuda_events,
            )
            output.write_result(
                BenchResult(
                    routine="pooling",
                    subroutine=subroutine,
                    backend=backend,
                    shape=shape,
                    dtype=dtype_str,
                    median_ms=median_ms,
                    std_ms=std_ms,
                    bandwidth_tb_s=compute_bandwidth_tb_s(_pool_bytes(cfg, dtype), median_ms),
                )
            )


def _measure_interleaved(
    arms: dict[str, Callable], rounds: int, iterations: int, use_cuda_events: bool
):
    samples = {name: [] for name in arms}
    names = list(arms)
    for round_idx in range(rounds):
        offset = round_idx % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            median_ms, _ = bench_fn(
                arms[name],
                dry_run_iters=2,
                num_iters=iterations,
                use_cuda_events=use_cuda_events,
            )
            samples[name].append(median_ms)
    return {
        name: (
            statistics.median(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
        for name, values in samples.items()
    }


def _fmt_time(timing: tuple[float, float]) -> str:
    return f"{timing[0]:.4f}±{timing[1]:.4f}ms"


def _run_report_shape(
    cfg: dict[str, Any],
    dtype: torch.dtype,
    rounds: int,
    iterations: int,
    use_cuda_events: bool,
):
    cuda_fn, torch_fn = _setup_for_config("avg_pool1d", cfg, dtype)
    previous_input, previous_fn = _setup_previous_model(cfg, dtype)

    # Use the previous expression's input for a separate correctness check;
    # setup_avg_pool1d owns another random tensor by design so every timing arm
    # keeps its preferred pre-created layout.
    previous_ref = previous_fn()
    previous_cuda = oasr.avg_pool1d(
        previous_input,
        cfg["kernel_size"],
        cfg["stride"],
        cfg["padding"],
        cfg["ceil_mode"],
        cfg["count_include_pad"],
    )
    torch.testing.assert_close(previous_cuda, previous_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cuda_fn(), torch_fn(), rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()

    return _measure_interleaved(
        {"cuda": cuda_fn, "torch": torch_fn, "previous": previous_fn},
        rounds,
        iterations,
        use_cuda_events,
    )


def run_standalone(variant: str = "avg_pool1d") -> None:
    if variant != "avg_pool1d":
        raise ValueError(f"Unknown pooling variant: {variant}")
    parser = argparse.ArgumentParser(description="OASR AvgPool1D benchmark")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "all"), default="all")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--shape-index", type=int)
    parser.add_argument("--use-cuda-events", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--target", choices=("cuda", "torch", "previous", "all"), default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rounds <= 0 or args.iterations <= 0:
        parser.error("--rounds and --iterations must be positive")

    configs = DEFAULT_CONFIGS[variant]
    if args.shape_index is not None:
        configs = [configs[args.shape_index]]
    dtypes = (torch.float16, torch.bfloat16) if args.dtype == "all" else (parse_dtype(args.dtype),)

    if args.profile:
        cfg = configs[0]
        cuda_fn, torch_fn = _setup_for_config(variant, cfg, dtypes[0])
        _, previous_fn = _setup_previous_model(cfg, dtypes[0])
        arms = {"cuda": cuda_fn, "torch": torch_fn, "previous": previous_fn}
        for name, fn in arms.items():
            if args.target in (name, "all"):
                profile_kernel(f"{name}_{variant}", fn)
        return

    column_shape = 34
    column_time = 20
    column_speedup = 12
    header = (
        f"{'shape':>{column_shape}}"
        f"  {'OASR BTC':>{column_time}}"
        f"  {'PyTorch BCT':>{column_time}}"
        f"  {'Previous model':>{column_time}}"
        f"  {'Torch/OASR':>{column_speedup}}"
        f"  {'Old/OASR':>{column_speedup}}"
    )
    separator = "-" * len(header)
    device = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()

    print("OASR AvgPool1D Benchmark")
    print("=" * len(header))
    print(
        f"device={device}, sm={capability[0]}{capability[1]}, "
        f"rounds={args.rounds}, iterations={args.iterations}"
    )
    for dtype in dtypes:
        print(f"\n--- {str(dtype).removeprefix('torch.')} ---")
        print(header)
        print(separator)
        for cfg in configs:
            timings = _run_report_shape(
                cfg, dtype, args.rounds, args.iterations, args.use_cuda_events
            )
            cuda_ms = timings["cuda"][0]
            row = (
                f"{_shape_str(cfg):>{column_shape}}"
                f"  {_fmt_time(timings['cuda']):>{column_time}}"
                f"  {_fmt_time(timings['torch']):>{column_time}}"
                f"  {_fmt_time(timings['previous']):>{column_time}}"
                f"  {timings['torch'][0] / cuda_ms:>{column_speedup}.2f}x"
                f"  {timings['previous'][0] / cuda_ms:>{column_speedup}.2f}x"
            )
            print(row)
    print()
