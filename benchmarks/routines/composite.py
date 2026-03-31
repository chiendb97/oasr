"""Composite / end-to-end benchmark routines (conv_block)."""

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
    make_bench_parser,
    run_main,
    run_profile,
)

SUBROUTINES = ["conv_block"]

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "conv_block": [
        {"batch": 32, "seq": 250, "d_model": 256, "kernel_size": 15},
        {"batch": 64, "seq": 250, "d_model": 256, "kernel_size": 15},
        {"batch": 64, "seq": 250, "d_model": 512, "kernel_size": 31},
        {"batch": 64, "seq": 500, "d_model": 256, "kernel_size": 15},
        {"batch": 64, "seq": 500, "d_model": 512, "kernel_size": 31},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "conv_block": (64, 250, 512, 31),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------


def setup_conv_block(batch_size, seq_len, d_model, kernel_size, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype)
    pw1_weight = torch.randn(2 * d_model, d_model, device="cuda", dtype=dtype)
    pw1_bias = torch.randn(2 * d_model, device="cuda", dtype=dtype)
    dw_weight = torch.randn(kernel_size, d_model, device="cuda", dtype=dtype)
    dw_bias = torch.randn(d_model, device="cuda", dtype=dtype)
    pw2_weight = torch.randn(d_model, d_model, device="cuda", dtype=dtype)
    pw2_bias = torch.randn(d_model, device="cuda", dtype=dtype)

    dw_weight_pt = dw_weight.view(d_model, 1, kernel_size)

    def oasr_fn():
        pw1_out = oasr.pointwise_conv1d(x, pw1_weight, pw1_bias)
        glu_out = oasr.glu(pw1_out)
        dw_out = oasr.depthwise_conv1d(glu_out, dw_weight, dw_bias, kernel_size // 2)
        swish_out = oasr.swish(dw_out)
        return oasr.pointwise_conv1d(swish_out, pw2_weight, pw2_bias)

    def pytorch_fn():
        pw1 = F.linear(x, pw1_weight, pw1_bias)
        glu = F.glu(pw1, dim=-1)
        glu_nchw = glu.permute(0, 2, 1)
        dw_nchw = F.conv1d(
            glu_nchw, dw_weight_pt, dw_bias, padding=kernel_size // 2, groups=d_model
        )
        dw = dw_nchw.permute(0, 2, 1)
        swish = F.silu(dw)
        return F.linear(swish, pw2_weight, pw2_bias)

    return oasr_fn, pytorch_fn


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument("--kernel-size", type=int, default=None, help="Depthwise conv kernel size")


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "conv_block")
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
        oasr_fn, pytorch_fn = setup_conv_block(
            cfg["batch"], cfg["seq"], cfg["d_model"], cfg["kernel_size"], dtype
        )
        fn_map = {"cuda": oasr_fn, "torch": pytorch_fn}
        backends = getattr(args, "backends", None) or list(fn_map.keys())

        shape_str = f"[{cfg['batch']}, {cfg['seq']}, {cfg['d_model']}] k={cfg['kernel_size']}"

        if do_check and "torch" in backends and any(b in fn_map and b != "torch" for b in backends):
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
            output.write_result(BenchResult(
                routine="composite",
                subroutine=subroutine,
                backend=backend,
                shape=shape_str,
                dtype=dtype_str,
                median_ms=median_ms,
                std_ms=std_ms,
            ))


def _resolve_configs(args, subroutine):
    batch = getattr(args, "batch", None)
    seq = getattr(args, "seq", None)
    d_model = getattr(args, "d_model", None)
    kernel_size = getattr(args, "kernel_size", None)

    if all(v is not None for v in [batch, seq, d_model, kernel_size]):
        return [{"batch": batch, "seq": seq, "d_model": d_model, "kernel_size": kernel_size}]

    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["conv_block"])


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------


def run_standalone() -> None:
    pcfg = {"conv_block": PROFILE_CONFIGS["conv_block"]}
    setup_funcs = {
        "conv_block": lambda: setup_conv_block(*PROFILE_CONFIGS["conv_block"]),
    }

    def benchmark():
        output = OutputWriter()
        output.write_header("Conv Block (End-to-End) Benchmark")
        for cfg in DEFAULT_CONFIGS["conv_block"]:
            oasr_fn, pytorch_fn = setup_conv_block(
                cfg["batch"], cfg["seq"], cfg["d_model"], cfg["kernel_size"]
            )
            shape_str = f"[{cfg['batch']}, {cfg['seq']}, {cfg['d_model']}] k={cfg['kernel_size']}"
            for backend, fn in [("cuda", oasr_fn), ("torch", pytorch_fn)]:
                median_ms, std_ms = bench_fn(fn)
                output.write_result(BenchResult(
                    routine="composite", subroutine="conv_block", backend=backend,
                    shape=shape_str, dtype="float16",
                    median_ms=median_ms, std_ms=std_ms,
                ))
        output.finalize()

    run_main("Conv Block Kernel", pcfg, setup_funcs, benchmark)
