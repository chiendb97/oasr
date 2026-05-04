"""FFT family benchmark routines."""

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
    parse_dtype,
    run_main,
)

SUBROUTINES = ["rfft", "rfft_power"]

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

# Typical ASR feature-extraction shapes: micro-batch * num_frames flattened FFTs.
# n_fft is constrained to a power of two in [8, 2048].
DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "rfft": [
        {"batch": 32, "num_frames": 250, "n_fft": 256},
        {"batch": 32, "num_frames": 250, "n_fft": 512},
        {"batch": 64, "num_frames": 500, "n_fft": 512},
        {"batch": 64, "num_frames": 500, "n_fft": 1024},
        {"batch": 64, "num_frames": 500, "n_fft": 2048},
    ],
    "rfft_power": [
        {"batch": 32, "num_frames": 250, "n_fft": 256},
        {"batch": 32, "num_frames": 250, "n_fft": 512},
        {"batch": 64, "num_frames": 500, "n_fft": 512},
        {"batch": 64, "num_frames": 500, "n_fft": 1024},
        {"batch": 64, "num_frames": 500, "n_fft": 2048},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "rfft": (64, 500, 512),
    "rfft_power": (64, 500, 512),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions  -- return (oasr_fn, pytorch_fn) closures
# ---------------------------------------------------------------------------


def setup_rfft(batch, num_frames, n_fft, dtype=torch.float32):
    x = torch.randn(batch, num_frames, n_fft, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.rfft(x)

    def pytorch_fn():
        return torch.fft.rfft(x)

    return oasr_fn, pytorch_fn


def setup_rfft_power(batch, num_frames, n_fft, dtype=torch.float32):
    x = torch.randn(batch, num_frames, n_fft, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.rfft_power(x)

    def pytorch_fn():
        return torch.fft.rfft(x).abs().pow(2)

    return oasr_fn, pytorch_fn


SETUP_FOR = {"rfft": setup_rfft, "rfft_power": setup_rfft_power}


# ---------------------------------------------------------------------------
# CLI args (used by oasr_benchmark.py)
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--num-frames", type=int, default=None, help="Frames per utterance")
    parser.add_argument("--n-fft", type=int, default=None, help="FFT length (power of two)")


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def _fft_bytes(batch, num_frames, n_fft, subroutine, dtype):
    """Bytes accessed: read input + write output."""
    elem = dtype_size(dtype)
    in_bytes = batch * num_frames * n_fft * elem
    out_elem = (n_fft // 2 + 1) * (2 if subroutine == "rfft" else 1)
    out_bytes = batch * num_frames * out_elem * elem
    return in_bytes + out_bytes


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "rfft")
    dtype_str = getattr(args, "dtype", "float32")
    dtype = parse_dtype(dtype_str)
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    if dtype != torch.float32:
        print(f"  [WARNING] FFT only supports float32, got {dtype_str}. Forcing float32.")
        dtype_str = "float32"
        dtype = torch.float32

    setup_fn = SETUP_FOR.get(subroutine, setup_rfft)
    configs = _resolve_configs(args, subroutine)

    for cfg in configs:
        oasr_fn, pytorch_fn = setup_fn(cfg["batch"], cfg["num_frames"], cfg["n_fft"], dtype)
        fn_map = get_fn_map(subroutine, oasr_fn, pytorch_fn)
        backends = getattr(args, "backends", None) or list(fn_map.keys())

        b, f, n = cfg["batch"], cfg["num_frames"], cfg["n_fft"]
        bytes_accessed = _fft_bytes(b, f, n, subroutine, dtype)
        shape_str = f"[{b}, {f}, n_fft={n}]"

        if do_check and "torch" in backends and any(
            bk in fn_map and bk != "torch" for bk in backends
        ):
            oasr_out = oasr_fn()
            pytorch_out = pytorch_fn()
            if oasr_out.dtype != pytorch_out.dtype:
                pytorch_out = pytorch_out.to(oasr_out.dtype)
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
                routine="fft",
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
    num_frames = getattr(args, "num_frames", None)
    n_fft = getattr(args, "n_fft", None)

    if all(v is not None for v in [batch, num_frames, n_fft]):
        return [{"batch": batch, "num_frames": num_frames, "n_fft": n_fft}]

    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["rfft"])


def get_fn_map(subroutine, cuda_fn, torch_fn):
    return {"cuda": cuda_fn, "torch": torch_fn}


# ---------------------------------------------------------------------------
# Standalone entry  -- used by bench_fft.py / bench_rfft_power.py wrappers
# ---------------------------------------------------------------------------


def run_standalone(variant: str = "rfft") -> None:
    setup_fn = SETUP_FOR.get(variant, setup_rfft)
    subs = [variant]
    pcfg = {k: PROFILE_CONFIGS[k] for k in subs if k in PROFILE_CONFIGS}

    setup_funcs = {sub: _make_profile_setup(sub) for sub in subs if sub in PROFILE_CONFIGS}

    def benchmark():
        output = OutputWriter()
        for sub in subs:
            output.write_header(f"{sub.upper()} Kernel Benchmark")
            for cfg in DEFAULT_CONFIGS.get(sub, []):
                oasr_fn, pytorch_fn = setup_fn(
                    cfg["batch"], cfg["num_frames"], cfg["n_fft"], torch.float32
                )
                b, f, n = cfg["batch"], cfg["num_frames"], cfg["n_fft"]
                bytes_accessed = _fft_bytes(b, f, n, sub, torch.float32)
                shape_str = f"[{b}, {f}, n_fft={n}]"
                for backend, fn in get_fn_map(sub, oasr_fn, pytorch_fn).items():
                    median_ms, std_ms = bench_fn(fn)
                    bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
                    output.write_result(BenchResult(
                        routine="fft", subroutine=sub, backend=backend,
                        shape=shape_str, dtype="float32",
                        median_ms=median_ms, std_ms=std_ms,
                        bandwidth_tb_s=bw,
                    ))
        output.finalize()

    run_main(f"{variant.upper()} FFT Kernel", pcfg, setup_funcs, benchmark)


def _make_profile_setup(subroutine):
    cfg_tuple = PROFILE_CONFIGS[subroutine]
    setup_fn = SETUP_FOR.get(subroutine, setup_rfft)

    def _setup():
        return setup_fn(*cfg_tuple)

    return _setup
