#!/usr/bin/env python3
"""OASR GLU Benchmark — CUDA vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.bench_utils import bench_fn, dtype_size
from benchmarks.routines.activation import setup_glu

CONFIGS = [
    {"batch": 32, "seq": 250, "channels": 256},
    {"batch": 64, "seq": 250, "channels": 256},
    {"batch": 64, "seq": 250, "channels": 512},
    {"batch": 64, "seq": 500, "channels": 256},
    {"batch": 64, "seq": 500, "channels": 512},
]

DTYPES = [torch.float16]
DTYPE_NAMES = {torch.float16: "float16"}

_COL_SHAPE  = 22
_COL_TIME   = 12
_COL_METRIC = 10

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'CUDA':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR GLU Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _bw_tb_s(cfg, dtype, ms):
    b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
    nbytes  = (b * s * 2 * c + b * s * c) * dtype_size(dtype)
    return nbytes / (ms * 1e-3) / 1e12


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_bw(bw):
    return f"{bw:.3f}TB/s"


def _row(shape_str, times, cfg, dtype):
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    for ms in times:
        parts.append(
            f"  {_fmt(ms):>{_COL_TIME}}  {_fmt_bw(_bw_tb_s(cfg, dtype, ms)):>{_COL_METRIC}}"
        )
    return "".join(parts)


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
            cuda_fn, pytorch_fn = setup_glu(cfg["batch"], cfg["seq"], cfg["channels"], dtype)
            cuda_ms,    _ = bench_fn(cuda_fn)
            pytorch_ms, _ = bench_fn(pytorch_fn)
            shape_str = f"[{cfg['batch']},{cfg['seq']},{cfg['channels']*2}]"
            # order: CUDA, PyTorch
            print(_row(shape_str, [cuda_ms, pytorch_ms], cfg, dtype))

    print()


if __name__ == "__main__":
    main()
