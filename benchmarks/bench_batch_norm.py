#!/usr/bin/env python3
"""OASR BatchNorm Benchmark — CUDA vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.bench_utils import bench_fn, dtype_size
from benchmarks.routines.norm import setup_batch_norm, setup_batch_norm_swish, setup_batch_norm_activation

SUBROUTINES = {
    "batch_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "batch_norm_swish": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "batch_norm_activation": [
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
}

DTYPES = [torch.float32]
DTYPE_NAMES = {torch.float32: "float32"}

_COL_SHAPE  = 20
_COL_TIME   = 12
_COL_METRIC = 10

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'CUDA':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR BatchNorm Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _bw_tb_s(cfg, dtype, ms):
    b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
    elem    = dtype_size(dtype)
    nbytes  = b * s * h * elem * 2 + h * elem * 4
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


def _bench_sub(sub, cfgs, dtype):
    for cfg in cfgs:
        b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
        if sub == "batch_norm":
            cuda_fn, pt_fn = setup_batch_norm(b, s, h, dtype)
        elif sub == "batch_norm_swish":
            cuda_fn, pt_fn = setup_batch_norm_swish(b, s, h, dtype)
        else:
            cuda_fn, pt_fn = setup_batch_norm_activation(b, s, h, dtype=dtype)
        cuda_ms, _ = bench_fn(cuda_fn)
        pt_ms,   _ = bench_fn(pt_fn)
        shape_str = f"[{b},{s},{h}]"
        # order: CUDA, PyTorch
        print(_row(shape_str, [cuda_ms, pt_ms], cfg, dtype))


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); return

    print(_TITLE)
    print(_TITLE_SEP)

    for dtype in DTYPES:
        for sub, cfgs in SUBROUTINES.items():
            print(f"\n--- {sub} ({DTYPE_NAMES[dtype]}) ---")
            print(_HEADER)
            print(_SEP)
            _bench_sub(sub, cfgs, dtype)

    print()


if __name__ == "__main__":
    main()
