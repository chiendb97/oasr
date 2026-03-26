#!/usr/bin/env python3
"""OASR LayerNorm Benchmark — CUDA vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.bench_utils import bench_fn, dtype_size
from benchmarks.routines.norm import setup_layer_norm, setup_add_layer_norm, setup_layer_norm_activation

SUBROUTINES = {
    "layer_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 256},
        {"batch": 64, "seq": 500, "hidden": 512},
        {"batch": 32, "seq": 500, "hidden": 512},
    ],
    "add_layer_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "layer_norm_activation": [
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
}

DTYPES = [torch.float16]
DTYPE_NAMES = {torch.float16: "float16"}

_COL_SHAPE  = 20
_COL_TIME   = 12
_COL_METRIC = 10

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'CUDA':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR LayerNorm Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _bw_tb_s(cfg, dtype, ms, extra_inputs=0):
    b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
    elem    = dtype_size(dtype)
    nbytes  = b * s * h * elem * (2 + extra_inputs) + h * elem * 2
    return nbytes / (ms * 1e-3) / 1e12


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_bw(bw):
    return f"{bw:.3f}TB/s"


def _row(shape_str, times, cfg, dtype, extra_inputs=0):
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    for ms in times:
        parts.append(
            f"  {_fmt(ms):>{_COL_TIME}}  {_fmt_bw(_bw_tb_s(cfg, dtype, ms, extra_inputs)):>{_COL_METRIC}}"
        )
    return "".join(parts)


def _bench_sub(sub, cfgs, dtype):
    for cfg in cfgs:
        if sub == "layer_norm":
            cuda_fn, pt_fn = setup_layer_norm(cfg["batch"], cfg["seq"], cfg["hidden"], dtype)
            extra = 0
        elif sub == "add_layer_norm":
            cuda_fn, pt_fn = setup_add_layer_norm(cfg["batch"], cfg["seq"], cfg["hidden"], dtype)
            extra = 1
        else:
            cuda_fn, pt_fn = setup_layer_norm_activation(cfg["batch"], cfg["seq"], cfg["hidden"], dtype=dtype)
            extra = 0
        cuda_ms, _ = bench_fn(cuda_fn)
        pt_ms,   _ = bench_fn(pt_fn)
        shape_str = f"[{cfg['batch']},{cfg['seq']},{cfg['hidden']}]"
        # order: CUDA, PyTorch
        print(_row(shape_str, [cuda_ms, pt_ms], cfg, dtype, extra))


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
