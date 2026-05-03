#!/usr/bin/env python3
"""OASR TopK Benchmark — CUDA vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.bench_utils import bench_fn, dtype_size
from benchmarks.routines.topk import setup_topk

# (batch, seq, channels, k)
CONFIGS = [
    {"batch": 64, "seq": 250, "channels": 256,  "k": 5},
    {"batch": 64, "seq": 250, "channels": 512,  "k": 5},
    {"batch": 64, "seq": 250, "channels": 1024, "k": 5},
    {"batch": 64, "seq": 250, "channels": 2048, "k": 5},
    {"batch": 64, "seq": 250, "channels": 4096, "k": 5},
    {"batch": 64, "seq": 250, "channels": 4096, "k": 50},
    {"batch": 64, "seq": 250, "channels": 8192, "k": 5},
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}

_COL_SHAPE  = 26
_COL_TIME   = 12
_COL_METRIC = 10

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'CUDA':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
    f"  {'Speedup':>8}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR TopK Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _bw_tb_s(cfg, dtype, ms):
    b, s, c, k = cfg["batch"], cfg["seq"], cfg["channels"], cfg["k"]
    elem   = dtype_size(dtype)
    nbytes = b * s * c * elem + b * s * k * (elem + 4)  # read input + write values + indices
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
        print(f"\n--- topk ({DTYPE_NAMES[dtype]}) ---")
        print(_HEADER)
        print(_SEP)

        for cfg in CONFIGS:
            b, s, c, k = cfg["batch"], cfg["seq"], cfg["channels"], cfg["k"]
            cuda_fn, pt_fn = setup_topk(b, s, c, k, dtype)
            cuda_ms, _ = bench_fn(cuda_fn)
            pt_ms,   _ = bench_fn(pt_fn)
            shape_str = f"[{b},{s},{c}] k={k}"
            print(_row(shape_str, cuda_ms, pt_ms, cfg, dtype))

    print()


if __name__ == "__main__":
    main()
