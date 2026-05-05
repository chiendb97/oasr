#!/usr/bin/env python3
"""OASR Softmax Benchmark — CUDA vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.bench_utils import bench_fn, dtype_size
from benchmarks.routines.softmax import setup_softmax

CONFIGS = [
    {"batch": 32, "seq": 250,  "channels": 256},
    {"batch": 64, "seq": 250,  "channels": 256},
    {"batch": 64, "seq": 250,  "channels": 512},
    {"batch": 64, "seq": 500,  "channels": 256},
    {"batch": 64, "seq": 500,  "channels": 512},
    {"batch": 64, "seq": 250,  "channels": 1024},
    {"batch": 64, "seq": 1000, "channels": 512},
    {"batch": 64, "seq": 1000, "channels": 1024},
    {"batch": 1, "seq": 1, "channels": 131072},
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}

_COL_SHAPE  = 22
_COL_TIME   = 12
_COL_METRIC = 10

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'CUDA':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
    f"  {'Speedup':>8}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR Softmax Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _bw_tb_s(cfg, dtype, ms):
    b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
    elem    = dtype_size(dtype)
    nbytes  = 2 * b * s * c * elem   # read input + write output
    return nbytes / (ms * 1e-3) / 1e12


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_bw(bw):
    return f"{bw:.3f}TB/s"


def _row(shape_str, cuda_ms, pt_ms, cfg, dtype):
    cuda_bw = _bw_tb_s(cfg, dtype, cuda_ms)
    pt_bw   = _bw_tb_s(cfg, dtype, pt_ms)
    speedup = pt_ms / cuda_ms
    return (
        f"{shape_str:>{_COL_SHAPE}}"
        f"  {_fmt(cuda_ms):>{_COL_TIME}}  {_fmt_bw(cuda_bw):>{_COL_METRIC}}"
        f"  {_fmt(pt_ms):>{_COL_TIME}}  {_fmt_bw(pt_bw):>{_COL_METRIC}}"
        f"  {speedup:>7.2f}x"
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(_TITLE)
    print(_TITLE_SEP)

    for dtype in DTYPES:
        print(f"\n--- softmax ({DTYPE_NAMES[dtype]}) ---")
        print(_HEADER)
        print(_SEP)

        for cfg in CONFIGS:
            b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
            cuda_fn, pt_fn = setup_softmax(b, s, c, dtype)
            cuda_ms, _ = bench_fn(cuda_fn)
            pt_ms,   _ = bench_fn(pt_fn)
            shape_str = f"[{b},{s},{c}]"
            print(_row(shape_str, cuda_ms, pt_ms, cfg, dtype))

    print()


if __name__ == "__main__":
    main()
