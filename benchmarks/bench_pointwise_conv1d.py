#!/usr/bin/env python3
"""OASR Pointwise Conv1D Benchmark — Auto-Tuning vs CUTLASS vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import bench_fn

SHAPES = [
    (4,  250,  256,  512),
    (32,  250,  256,  512),
    (64,  250,  256,  512),
    (64,  250,  512, 1024),
    (64,  250,  256, 2048),
    (64,  250,  512, 2048),
    (64,  500,  256,  512),
    (64,  500,  512, 1024),
]

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}

_COL_SHAPE  = 28
_COL_TIME   = 14
_COL_TFLOPS =  8

_HEADER = (
    f"{'(B,T,IC,OC)':>{_COL_SHAPE}}"
    f"  {'Auto-Tuning':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
    f"  {'CUTLASS':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR Pointwise Conv1D Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _tflops(B, T, IC, OC, ms):
    return 2.0 * B * T * IC * OC / (ms * 1e-3) / 1e12


def _fmt_time(ms):
    return f"{ms:.4f}ms"


def _fmt_tflops(t):
    return f"{t:.1f}T"


def _row(shape_str, times, B, T, IC, OC):
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    for ms in times:
        parts.append(
            f"  {_fmt_time(ms):>{_COL_TIME}}  {_fmt_tflops(_tflops(B, T, IC, OC, ms)):>{_COL_TFLOPS}}"
        )
    return "".join(parts)


def _run_shape(B, T, IC, OC, dtype):
    x = torch.randn(B, T, IC, device="cuda", dtype=dtype)
    weight = torch.randn(OC, IC, device="cuda", dtype=dtype)
    bias = torch.randn(OC, device="cuda", dtype=dtype)

    with oasr.autotune():
        oasr.gemm(x, weight, bias)
        autotune_ms, _ = bench_fn(lambda: oasr.gemm(x, weight, bias))

    cutlass_ms, _ = bench_fn(lambda: oasr.gemm(x, weight, bias))
    pytorch_ms, _ = bench_fn(lambda: F.linear(x, weight, bias))

    # order: Auto-Tuning, CUTLASS, PyTorch
    return autotune_ms, cutlass_ms, pytorch_ms


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); return

    print(_TITLE)
    print(_TITLE_SEP)

    for dtype in DTYPES:
        print(f"\n--- {DTYPE_NAMES[dtype]} ---")
        print(_HEADER)
        print(_SEP)
        for B, T, IC, OC in SHAPES:
            times = _run_shape(B, T, IC, OC, dtype)
            print(_row(f"({B},{T},{IC},{OC})", list(times), B, T, IC, OC))

    print()


if __name__ == "__main__":
    main()
