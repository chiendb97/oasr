"""Softmax family benchmark routines."""

from __future__ import annotations

import argparse
from typing import Any

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
    make_bench_parser,
    run_main,
    run_profile,
)

SUBROUTINES = ["softmax"]

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "softmax": [
        {"batch": 32, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 512},
        {"batch": 64, "seq": 500, "channels": 256},
        {"batch": 64, "seq": 500, "channels": 512},
        {"batch": 64, "seq": 250, "channels": 1024},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "softmax": (64, 250, 512),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------


def setup_softmax(batch_size, seq_len, channels, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.softmax(x)

    def pytorch_fn():
        return F.softmax(x, dim=-1)

    return oasr_fn, pytorch_fn


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--channels", type=int, default=None, help="Channel dimension (softmax dim)")


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def _softmax_bytes(batch, seq, channels, dtype):
    """Bytes accessed: read input + write output."""
    elem = dtype_size(dtype)
    return 2 * batch * seq * channels * elem


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "softmax")
    dtype_str = getattr(args, "dtype", "float16")
    from benchmarks.routines.bench_utils import parse_dtype

    dtype = parse_dtype(dtype_str)
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    configs = _resolve_configs(args, subroutine)

    for cfg in configs:
        oasr_fn, pytorch_fn = setup_softmax(
            cfg["batch"], cfg["seq"], cfg["channels"], dtype
        )
        fn_map = get_fn_map(subroutine, oasr_fn, pytorch_fn)
        backends = getattr(args, "backends", None) or list(fn_map.keys())

        b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
        bytes_accessed = _softmax_bytes(b, s, c, dtype)
        shape_str = f"[{b}, {s}, {c}]"

        if do_check and "torch" in backends and any(
            bk in fn_map and bk != "torch" for bk in backends
        ):
            oasr_out = oasr_fn()
            pytorch_out = pytorch_fn()
            passed, max_diff = check_close(oasr_out, pytorch_out.to(dtype))
            if not passed:
                print(f"  [ERROR] Output mismatch for {shape_str} (max_diff={max_diff:.6f})")
                if not allow_mismatch:
                    continue

        for backend in backends:
            if backend not in fn_map:
                print(f"  [WARNING] Unknown backend '{backend}', skipping")
                continue
            median_ms, std_ms = bench_fn(
                fn_map[backend],
                dry_run_iters=dry_run_iters,
                num_iters=num_iters,
                use_cuda_events=use_cuda_events,
            )
            bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
            output.write_result(BenchResult(
                routine="softmax",
                subroutine=subroutine,
                backend=backend,
                shape=shape_str,
                dtype=dtype_str,
                median_ms=median_ms,
                std_ms=std_ms,
                bandwidth_tb_s=bw,
            ))


def _resolve_configs(args, subroutine):
    batch = getattr(args, "batch", None)
    seq = getattr(args, "seq", None)
    channels = getattr(args, "channels", None)

    if all(v is not None for v in [batch, seq, channels]):
        return [{"batch": batch, "seq": seq, "channels": channels}]

    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["softmax"])


def get_fn_map(subroutine, cuda_fn, torch_fn):
    return {"cuda": cuda_fn, "torch": torch_fn}


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------


def run_standalone(variant: str = "softmax") -> None:
    """Standalone entry point for backwards-compat bench_softmax.py wrapper."""
    subs = [variant]
    pcfg = {k: PROFILE_CONFIGS[k] for k in subs if k in PROFILE_CONFIGS}

    setup_funcs = {}
    for sub in subs:
        if sub in PROFILE_CONFIGS:
            setup_funcs[sub] = _make_profile_setup(sub)

    def benchmark():
        output = OutputWriter()
        for sub in subs:
            configs = DEFAULT_CONFIGS.get(sub, [])
            output.write_header(f"{sub.upper()} Benchmark")
            for cfg in configs:
                oasr_fn, pytorch_fn = setup_softmax(
                    cfg["batch"], cfg["seq"], cfg["channels"], torch.float16
                )
                b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
                bytes_accessed = _softmax_bytes(b, s, c, torch.float16)
                shape_str = f"[{b}, {s}, {c}]"
                for backend, fn in get_fn_map(sub, oasr_fn, pytorch_fn).items():
                    median_ms, std_ms = bench_fn(fn)
                    bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
                    output.write_result(BenchResult(
                        routine="softmax", subroutine=sub, backend=backend,
                        shape=shape_str, dtype="float16",
                        median_ms=median_ms, std_ms=std_ms,
                        bandwidth_tb_s=bw,
                    ))
        output.finalize()

    run_main(f"{variant.upper()} Softmax Kernel", pcfg, setup_funcs, benchmark)


def _make_profile_setup(subroutine):
    cfg_tuple = PROFILE_CONFIGS[subroutine]

    def _setup():
        return setup_softmax(*cfg_tuple)

    return _setup
