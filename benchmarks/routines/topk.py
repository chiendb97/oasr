"""Top-k family benchmark routines."""

from __future__ import annotations

import argparse
from typing import Any

import torch

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

SUBROUTINES = ["topk"]

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "topk": [
        {"batch": 64, "seq": 250, "channels": 256,  "k": 5},
        {"batch": 64, "seq": 250, "channels": 512,  "k": 5},
        {"batch": 64, "seq": 250, "channels": 1024, "k": 5},
        {"batch": 64, "seq": 250, "channels": 2048, "k": 5},
        {"batch": 64, "seq": 250, "channels": 4096, "k": 5},
        {"batch": 64, "seq": 250, "channels": 4096, "k": 50},
        {"batch": 64, "seq": 250, "channels": 8192, "k": 5},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "topk": (64, 250, 4096, 5),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------


def setup_topk(batch_size, seq_len, channels, k, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.topk(x, k)

    def pytorch_fn():
        return torch.topk(x, k, dim=-1)

    return oasr_fn, pytorch_fn


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--channels", type=int, default=None, help="Channel dimension")
    parser.add_argument("--k", type=int, default=None, help="Number of top elements")


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def _topk_bytes(batch, seq, channels, k, dtype):
    """Approximate bytes accessed: read input + write values + write indices."""
    elem = dtype_size(dtype)
    input_bytes = batch * seq * channels * elem
    values_bytes = batch * seq * k * elem
    indices_bytes = batch * seq * k * 4  # int32
    return input_bytes + values_bytes + indices_bytes


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "topk")
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
        oasr_fn, pytorch_fn = setup_topk(
            cfg["batch"], cfg["seq"], cfg["channels"], cfg["k"], dtype
        )
        fn_map = get_fn_map(subroutine, oasr_fn, pytorch_fn)
        backends = getattr(args, "backends", None) or list(fn_map.keys())

        b, s, c, k = cfg["batch"], cfg["seq"], cfg["channels"], cfg["k"]
        bytes_accessed = _topk_bytes(b, s, c, k, dtype)
        shape_str = f"[{b}, {s}, {c}] k={k}"

        if do_check and "torch" in backends and any(
            bk in fn_map and bk != "torch" for bk in backends
        ):
            oasr_vals, oasr_idxs = oasr_fn()
            pt_vals, _ = pytorch_fn()
            passed, max_diff = check_close(oasr_vals, pt_vals.to(dtype))
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
                routine="topk",
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
    k = getattr(args, "k", None)

    if all(v is not None for v in [batch, seq, channels, k]):
        return [{"batch": batch, "seq": seq, "channels": channels, "k": k}]

    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["topk"])


def get_fn_map(subroutine, cuda_fn, torch_fn):
    return {"cuda": cuda_fn, "torch": torch_fn}


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------


def run_standalone(variant: str = "topk") -> None:
    """Standalone entry point for backwards-compat bench_topk.py wrapper."""
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
                oasr_fn, pytorch_fn = setup_topk(
                    cfg["batch"], cfg["seq"], cfg["channels"], cfg["k"], torch.float16
                )
                b, s, c, k = cfg["batch"], cfg["seq"], cfg["channels"], cfg["k"]
                bytes_accessed = _topk_bytes(b, s, c, k, torch.float16)
                shape_str = f"[{b}, {s}, {c}] k={k}"
                for backend, fn in get_fn_map(sub, oasr_fn, pytorch_fn).items():
                    median_ms, std_ms = bench_fn(fn)
                    bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
                    output.write_result(BenchResult(
                        routine="topk", subroutine=sub, backend=backend,
                        shape=shape_str, dtype="float16",
                        median_ms=median_ms, std_ms=std_ms,
                        bandwidth_tb_s=bw,
                    ))
        output.finalize()

    run_main(f"{variant.upper()} TopK Kernel", pcfg, setup_funcs, benchmark)


def _make_profile_setup(subroutine):
    cfg_tuple = PROFILE_CONFIGS[subroutine]

    def _setup():
        return setup_topk(*cfg_tuple)

    return _setup
