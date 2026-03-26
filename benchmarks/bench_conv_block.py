#!/usr/bin/env python3
"""OASR Conv Block Benchmark — CUDA vs PyTorch (end-to-end latency)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.bench_utils import bench_fn
from benchmarks.routines.composite import setup_conv_block

CONFIGS = [
    {"batch": 32, "seq": 250, "d_model": 256, "kernel_size": 15},
    {"batch": 64, "seq": 250, "d_model": 256, "kernel_size": 15},
    {"batch": 64, "seq": 250, "d_model": 512, "kernel_size": 31},
    {"batch": 64, "seq": 500, "d_model": 256, "kernel_size": 15},
    {"batch": 64, "seq": 500, "d_model": 512, "kernel_size": 31},
]

DTYPES = [torch.float16]
DTYPE_NAMES = {torch.float16: "float16"}

_COL_SHAPE   = 28
_COL_TIME    = 12
_COL_SPEEDUP =  8

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'CUDA':>{_COL_TIME}}"
    f"  {'PyTorch':>{_COL_TIME}}"
    f"  {'Speedup':>{_COL_SPEEDUP}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR Conv Block Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_speedup(s):
    return f"{s:.2f}x"


def _row(shape_str, cuda_ms, pytorch_ms):
    speedup = pytorch_ms / cuda_ms if cuda_ms > 0 else 0.0
    return (
        f"{shape_str:>{_COL_SHAPE}}"
        f"  {_fmt(cuda_ms):>{_COL_TIME}}"
        f"  {_fmt(pytorch_ms):>{_COL_TIME}}"
        f"  {_fmt_speedup(speedup):>{_COL_SPEEDUP}}"
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); return

    print(_TITLE)
    print(_TITLE_SEP)

    for dtype in DTYPES:
        print(f"\n--- {DTYPE_NAMES[dtype]} ---")
        print(_HEADER)
        print(_SEP)
        for cfg in CONFIGS:
            cuda_fn, pt_fn = setup_conv_block(
                cfg["batch"], cfg["seq"], cfg["d_model"], cfg["kernel_size"], dtype
            )
            cuda_ms, _ = bench_fn(cuda_fn)
            pt_ms,   _ = bench_fn(pt_fn)
            shape_str = f"[{cfg['batch']},{cfg['seq']},{cfg['d_model']}] k={cfg['kernel_size']}"
            # order: CUDA, PyTorch
            print(_row(shape_str, cuda_ms, pt_ms))

    print()


if __name__ == "__main__":
    main()
