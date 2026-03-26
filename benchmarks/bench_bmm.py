#!/usr/bin/env python3
"""OASR BMM Benchmark — Auto-Tuning vs CUTLASS vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import oasr
from benchmarks.routines.bench_utils import bench_fn
from benchmarks.routines.gemm import setup_bmm

SHAPES = [
    (256, 200, 200,  64),
    (512, 200, 200,  64),
    (512, 400, 400,  64),
    ( 64, 200, 200,  64),
    (256, 400, 400, 128),
    (128, 500, 500,  64),
]

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}

_COL_SHAPE  = 22
_COL_TIME   = 14
_COL_METRIC =  8

_HEADER = (
    f"{'(B,M,N,K)':>{_COL_SHAPE}}"
    f"  {'Auto-Tuning':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
    f"  {'CUTLASS':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR BMM Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _tflops(B, M, N, K, ms):
    return 2.0 * B * M * N * K / (ms * 1e-3) / 1e12


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_t(t):
    return f"{t:.1f}T"


def _row(shape_str, times, bmnk):
    B, M, N, K = bmnk
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    for ms in times:
        parts.append(
            f"  {_fmt(ms):>{_COL_TIME}}  {_fmt_t(_tflops(B, M, N, K, ms)):>{_COL_METRIC}}"
        )
    return "".join(parts)


def _run_shape(B, M, N, K, dtype):
    A    = torch.randn(B, M, K, device="cuda", dtype=dtype)
    Bmat = torch.randn(B, N, K, device="cuda", dtype=dtype)

    with oasr.autotune():
        oasr.bmm(A, Bmat)
        autotune_ms, _ = bench_fn(lambda: oasr.bmm(A, Bmat))

    cutlass_ms, _ = bench_fn(lambda: oasr.bmm(A, Bmat))

    _, pytorch_fn = setup_bmm(B, M, N, K, dtype)
    pytorch_ms, _ = bench_fn(pytorch_fn)

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
        for B, M, N, K in SHAPES:
            times = _run_shape(B, M, N, K, dtype)
            print(_row(f"({B},{M},{N},{K})", list(times), (B, M, N, K)))

    print()


if __name__ == "__main__":
    main()
