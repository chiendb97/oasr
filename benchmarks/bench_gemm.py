#!/usr/bin/env python3
"""OASR GEMM Benchmark — Auto-Tuning vs cuBLAS vs CUTLASS vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import bench_fn

SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (1, 4096, 4096),
    (32, 4096, 4096),
    (128, 4096, 11008),
]

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}

_COL_SHAPE = 20
_COL_TIME  = 14
_COL_TFLOPS = 8

_HEADER = (
    f"{'(M,N,K)':>{_COL_SHAPE}}"
    f"  {'Auto-Tuning':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
    f"  {'cuBLAS':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
    f"  {'CUTLASS':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR GEMM Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _tflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def _fmt_time(ms):
    return f"{ms:.4f}ms"


def _fmt_tflops(t):
    return f"{t:.1f}T"


def _row(shape_str, times, mnk):
    M, N, K = mnk
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    for ms in times:
        parts.append(
            f"  {_fmt_time(ms):>{_COL_TIME}}  {_fmt_tflops(_tflops(M, N, K, ms)):>{_COL_TFLOPS}}"
        )
    return "".join(parts)


def _run_shape(M, N, K, dtype):
    A  = torch.randn(M, K, device="cuda", dtype=dtype)
    B  = torch.randn(N, K, device="cuda", dtype=dtype)
    Bt = B.t().contiguous()

    with oasr.autotune():
        oasr.gemm(A, B)
        autotune_ms, _ = bench_fn(lambda: oasr.gemm(A, B))

    cublas_ms,  _ = bench_fn(lambda: torch.mm(A, Bt))
    cutlass_ms, _ = bench_fn(lambda: oasr.gemm(A, B))
    pytorch_ms, _ = bench_fn(lambda: F.linear(A, B))

    # order: Auto-Tuning, cuBLAS, CUTLASS, PyTorch
    return autotune_ms, cublas_ms, cutlass_ms, pytorch_ms


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); return

    print(_TITLE)
    print(_TITLE_SEP)

    for dtype in DTYPES:
        print(f"\n--- {DTYPE_NAMES[dtype]} ---")
        print(_HEADER)
        print(_SEP)
        for M, N, K in SHAPES:
            times = _run_shape(M, N, K, dtype)
            print(_row(f"({M},{N},{K})", list(times), (M, N, K)))

    print()


if __name__ == "__main__":
    main()
