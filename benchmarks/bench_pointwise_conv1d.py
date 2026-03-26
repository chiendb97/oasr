#!/usr/bin/env python3
"""OASR Pointwise Conv1D Benchmark — CUTLASS vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.bench_utils import bench_fn
from benchmarks.routines.conv import setup_pointwise_conv1d

CONFIGS = [
    {"batch": 32, "seq": 250, "channels": 256,  "out_channels":  512},
    {"batch": 64, "seq": 250, "channels": 256,  "out_channels":  512},
    {"batch": 64, "seq": 250, "channels": 512,  "out_channels": 1024},
    {"batch": 64, "seq": 250, "channels": 256,  "out_channels": 2048},
    {"batch": 64, "seq": 250, "channels": 512,  "out_channels": 2048},
    {"batch": 64, "seq": 500, "channels": 256,  "out_channels":  512},
    {"batch": 64, "seq": 500, "channels": 512,  "out_channels": 1024},
]

DTYPES = [torch.float16]
DTYPE_NAMES = {torch.float16: "float16"}

_COL_SHAPE  = 34
_COL_TIME   = 12
_COL_METRIC =  8

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'CUTLASS':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR Pointwise Conv1D Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _tflops(cfg, ms):
    b, s = cfg["batch"], cfg["seq"]
    ic, oc = cfg["channels"], cfg["out_channels"]
    return 2.0 * b * s * oc * ic / (ms * 1e-3) / 1e12


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_t(t):
    return f"{t:.1f}T"


def _row(shape_str, times, cfg):
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    for ms in times:
        parts.append(
            f"  {_fmt(ms):>{_COL_TIME}}  {_fmt_t(_tflops(cfg, ms)):>{_COL_METRIC}}"
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
            cutlass_fn, pytorch_fn = setup_pointwise_conv1d(
                cfg["batch"], cfg["seq"], cfg["channels"], cfg["out_channels"], dtype
            )
            cutlass_ms, _ = bench_fn(cutlass_fn)
            pytorch_ms, _ = bench_fn(pytorch_fn)
            shape_str = f"[{cfg['batch']},{cfg['seq']},{cfg['channels']}] -> {cfg['out_channels']}"
            # order: CUTLASS, PyTorch
            print(_row(shape_str, [cutlass_ms, pytorch_ms], cfg))

    print()


if __name__ == "__main__":
    main()
