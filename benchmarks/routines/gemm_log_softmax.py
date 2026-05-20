"""Fused GEMM + log_softmax benchmark routine.

Compares ``oasr.gemm_log_softmax(x, W, b)`` against the unfused PyTorch
baseline ``F.log_softmax(F.linear(x, W, b), dim=-1)`` — i.e. the CTC head
``F.log_softmax(self.ctc_lo(hidden_states), dim=2)``.

Shapes follow CTC conventions: M = batch * frames, K = encoder hidden,
N = vocab size (typically padded to a multiple of 8).
"""

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
    compute_gemm_tflops,
    dtype_size,
    parse_dtype,
    run_main,
)

SUBROUTINES = ["gemm_log_softmax"]

# ---------------------------------------------------------------------------
# Default configs — CTC head shapes
#   M = effective_batch * frames, K = encoder hidden, N = vocab_size
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "gemm_log_softmax": [
        # (batch, frames, hidden, vocab) — typical Conformer-CTC heads.
        # CUTLASS FP16/BF16 GEMM requires N % 8 == 0; the real model's
        # convert.py pads the vocab the same way.
        {"M": 16 * 250,  "N": 4240,  "K": 256},   # WeNet ASR vocab ≈ 4233 → 4240
        {"M": 16 * 250,  "N": 4240,  "K": 512},
        {"M": 32 * 500,  "N": 5000,  "K": 512},
        {"M": 32 * 1000, "N": 8000,  "K": 512},
        {"M": 64 * 250,  "N": 16000, "K": 512},
        {"M": 64 * 500,  "N": 32000, "K": 512},
        # Single-frame streaming step
        {"M": 1,         "N": 4240,  "K": 256},
        {"M": 1,         "N": 4240,  "K": 512},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    # (M, N, K) for Nsight Compute profiling
    "gemm_log_softmax": (8000, 5000, 512),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_gemm_log_softmax(M: int, N: int, K: int, dtype: torch.dtype = torch.float16):
    """Build matched closures for ``oasr.gemm_log_softmax`` and the unfused baseline.

    Mirrors the CTC head shapes: ``A`` is [M, K] activations, ``W`` is [N, K]
    weight, ``bias`` is [N]. All vocab dims should be multiples of 8 to satisfy
    CUTLASS FP16/BF16 alignment (the model pads the real vocab the same way).
    """
    A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    W = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
    bias = torch.randn(N, device="cuda", dtype=dtype) * 0.1

    def oasr_fn():
        return oasr.gemm_log_softmax(A, W, bias)

    def pytorch_fn():
        return F.log_softmax(F.linear(A, W, bias), dim=-1)

    return oasr_fn, pytorch_fn


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--M", type=int, default=None, help="Rows (batch * frames)")
    parser.add_argument("--N", type=int, default=None, help="Vocab size")
    parser.add_argument("--K", type=int, default=None, help="Encoder hidden dim")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _softmax_bytes(M: int, N: int, dtype: torch.dtype) -> int:
    """Bytes touched by the log_softmax pass: read + write of [M, N]."""
    return 2 * M * N * dtype_size(dtype)


def _shape_str(cfg: dict[str, Any]) -> str:
    return f"({cfg['M']}, {cfg['N']}, {cfg['K']})"


# ---------------------------------------------------------------------------
# run_test — invoked by oasr_benchmark.py
# ---------------------------------------------------------------------------


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "gemm_log_softmax")
    dtype_str = getattr(args, "dtype", "float16")
    dtype = parse_dtype(dtype_str)
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    configs = _resolve_configs(args, subroutine)

    for cfg in configs:
        M, N, K = cfg["M"], cfg["N"], cfg["K"]
        oasr_fn, pytorch_fn = setup_gemm_log_softmax(M, N, K, dtype)
        fn_map = get_fn_map(subroutine, oasr_fn, pytorch_fn)
        backends = getattr(args, "backends", None) or list(fn_map.keys())
        shape_str = _shape_str(cfg)

        output.write_verbose(f"shape = {shape_str}, dtype = {dtype_str}", level=2)

        if do_check and "torch" in backends and any(
            b in fn_map and b != "torch" for b in backends
        ):
            oasr_out = oasr_fn()
            pytorch_out = pytorch_fn().to(dtype)
            # log_softmax tolerance must absorb FP16 GEMM noise on a wide N.
            rtol = 5e-2 if dtype == torch.bfloat16 else 5e-3
            atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
            passed, max_diff = check_close(oasr_out, pytorch_out, rtol=rtol, atol=atol)
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
            tflops = compute_gemm_tflops(M, N, K, median_ms)
            output.write_result(BenchResult(
                routine="gemm_log_softmax",
                subroutine=subroutine,
                backend=backend,
                shape=shape_str,
                dtype=dtype_str,
                median_ms=median_ms,
                std_ms=std_ms,
                tflops=tflops,
            ))


def _resolve_configs(args: argparse.Namespace, subroutine: str) -> list[dict]:
    M = getattr(args, "M", None)
    N = getattr(args, "N", None)
    K = getattr(args, "K", None)
    if all(v is not None for v in (M, N, K)):
        return [{"M": M, "N": N, "K": K}]
    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["gemm_log_softmax"])


def get_fn_map(subroutine, cutlass_fn, torch_fn):
    """Return {backend_name: fn}. Mirrors the gemm family — backend keys cutlass/torch."""
    return {"cutlass": cutlass_fn, "torch": torch_fn}


# ---------------------------------------------------------------------------
# Standalone entry — invoked by bench_gemm_log_softmax.py
# ---------------------------------------------------------------------------


def run_standalone() -> None:
    subroutine = "gemm_log_softmax"
    setup_funcs = {subroutine: _make_profile_setup(subroutine)}

    def benchmark():
        output = OutputWriter()
        output.write_header("Fused GEMM + log_softmax Benchmark")
        for dtype in (torch.float16, torch.bfloat16):
            dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
            for cfg in DEFAULT_CONFIGS[subroutine]:
                M, N, K = cfg["M"], cfg["N"], cfg["K"]
                oasr_fn, pytorch_fn = setup_gemm_log_softmax(M, N, K, dtype)
                shape_str = _shape_str(cfg)
                for backend, fn in get_fn_map(subroutine, oasr_fn, pytorch_fn).items():
                    median_ms, std_ms = bench_fn(fn)
                    tflops = compute_gemm_tflops(M, N, K, median_ms)
                    output.write_result(BenchResult(
                        routine="gemm_log_softmax",
                        subroutine=subroutine,
                        backend=backend,
                        shape=shape_str,
                        dtype=dtype_str,
                        median_ms=median_ms,
                        std_ms=std_ms,
                        tflops=tflops,
                    ))
        output.finalize()

    run_main(
        "Fused GEMM + log_softmax Kernel",
        {subroutine: PROFILE_CONFIGS[subroutine]},
        setup_funcs,
        benchmark,
    )


def _make_profile_setup(subroutine: str):
    cfg_tuple = PROFILE_CONFIGS[subroutine]

    def _setup():
        return setup_gemm_log_softmax(*cfg_tuple)

    return _setup
