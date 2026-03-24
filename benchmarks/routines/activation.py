"""Activation family benchmark routines (glu, swish)."""

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

SUBROUTINES = ["glu", "swish"]

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "glu": [
        {"batch": 32, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 512},
        {"batch": 64, "seq": 500, "channels": 256},
        {"batch": 64, "seq": 500, "channels": 512},
    ],
    "swish": [
        {"batch": 32, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 512},
        {"batch": 64, "seq": 500, "channels": 512},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "glu": (64, 250, 512),
    "swish": (64, 250, 512),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------


def setup_glu(batch_size, seq_len, channels, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len, 2 * channels, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.glu(x)

    def pytorch_fn():
        return F.glu(x, dim=-1)

    return oasr_fn, pytorch_fn


def setup_swish(batch_size, seq_len, channels, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.swish(x)

    def pytorch_fn():
        return F.silu(x)

    return oasr_fn, pytorch_fn


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--channels", type=int, default=None, help="Channel dimension")


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def _activation_bytes(batch, seq, channels, dtype, subroutine):
    """Estimate bytes accessed for activation kernels."""
    elem = dtype_size(dtype)
    if subroutine == "glu":
        input_bytes = batch * seq * 2 * channels * elem
        output_bytes = batch * seq * channels * elem
    else:  # swish
        input_bytes = batch * seq * channels * elem
        output_bytes = batch * seq * channels * elem
    return input_bytes + output_bytes


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "glu")
    dtype_str = getattr(args, "dtype", "float16")
    from benchmarks.routines.bench_utils import parse_dtype

    dtype = parse_dtype(dtype_str)
    backends = getattr(args, "backends", ["oasr", "pytorch"])
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    configs = _resolve_configs(args, subroutine)

    for cfg in configs:
        oasr_fn, pytorch_fn = _setup_for_config(subroutine, cfg, dtype)
        fn_map = {"oasr": oasr_fn, "pytorch": pytorch_fn}

        b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
        bytes_accessed = _activation_bytes(b, s, c, dtype, subroutine)
        shape_str = _shape_str(subroutine, cfg)

        if do_check and "oasr" in backends and "pytorch" in backends:
            oasr_out = oasr_fn()
            pytorch_out = pytorch_fn()
            passed, max_diff = check_close(oasr_out, pytorch_out)
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
                routine="activation",
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

    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["glu"])


def _setup_for_config(subroutine, cfg, dtype):
    b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
    if subroutine == "glu":
        return setup_glu(b, s, c, dtype)
    elif subroutine == "swish":
        return setup_swish(b, s, c, dtype)
    else:
        raise ValueError(f"Unknown activation subroutine: {subroutine}")


def _shape_str(subroutine, cfg):
    b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
    if subroutine == "glu":
        return f"[{b}, {s}, {2 * c}]"
    return f"[{b}, {s}, {c}]"


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------


def run_standalone(variant: str = "glu") -> None:
    """Standalone entry point for backwards-compat bench_*.py wrappers."""
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
            output.write_header(f"{sub.upper()} Activation Benchmark")
            for cfg in configs:
                oasr_fn, pytorch_fn = _setup_for_config(sub, cfg, torch.float16)
                b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
                bytes_accessed = _activation_bytes(b, s, c, torch.float16, sub)
                shape_str = _shape_str(sub, cfg)
                for backend, fn in [("oasr", oasr_fn), ("pytorch", pytorch_fn)]:
                    median_ms, std_ms = bench_fn(fn)
                    bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
                    output.write_result(BenchResult(
                        routine="activation", subroutine=sub, backend=backend,
                        shape=shape_str, dtype="float16",
                        median_ms=median_ms, std_ms=std_ms,
                        bandwidth_tb_s=bw,
                    ))
        output.finalize()

    title = f"{variant.upper()} Activation Kernel"
    run_main(title, pcfg, setup_funcs, benchmark)


def _make_profile_setup(subroutine):
    cfg_tuple = PROFILE_CONFIGS[subroutine]

    def _setup():
        if subroutine == "glu":
            return setup_glu(*cfg_tuple)
        else:
            return setup_swish(*cfg_tuple)

    return _setup
