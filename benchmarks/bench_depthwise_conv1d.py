#!/usr/bin/env python3
"""OASR Depthwise Conv1D Benchmark — CUDA vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.bench_utils import bench_fn, dtype_size
from benchmarks.routines.conv import setup_depthwise_conv1d

CONFIGS = [
    {"batch": 32, "seq": 250, "channels": 256, "kernel_size": 15},
    {"batch": 64, "seq": 250, "channels": 256, "kernel_size": 15},
    {"batch": 64, "seq": 250, "channels": 512, "kernel_size": 31},
    {"batch": 64, "seq": 500, "channels": 256, "kernel_size": 15},
    {"batch": 64, "seq": 500, "channels": 512, "kernel_size": 31},
    {"batch": 32, "seq": 125, "channels": 256, "kernel_size": 15},
    {"batch": 32, "seq": 125, "channels": 256, "kernel_size": 31},
    {"batch": 16, "seq": 125, "channels": 256, "kernel_size": 15},
    {"batch": 64, "seq": 125, "channels": 512, "kernel_size": 31},
]

DTYPES = [torch.float16]
DTYPE_NAMES = {torch.float16: "float16"}

_COL_SHAPE  = 30
_COL_TIME   = 12
_COL_METRIC =  9

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'CUDA':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'BW(TB/s)':>{_COL_METRIC}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR Depthwise Conv1D Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _bw_tb_s(cfg, dtype, ms):
    batch, seq, ch, ks = cfg["batch"], cfg["seq"], cfg["channels"], cfg["kernel_size"]
    out_len = seq - ks + 1 + ks // 2 * 2
    elem    = dtype_size(dtype)
    nbytes  = (batch * seq * ch + ks * ch + ch + batch * out_len * ch) * elem
    return nbytes / (ms * 1e-3) / 1e12


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_bw(bw):
    return f"{bw:.2f}TB/s"


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
            cuda_fn, pytorch_fn = setup_depthwise_conv1d(
                cfg["batch"], cfg["seq"], cfg["channels"], cfg["kernel_size"], dtype
            )
            cuda_ms,    _ = bench_fn(cuda_fn)
            pytorch_ms, _ = bench_fn(pytorch_fn)
            shape_str = f"[{cfg['batch']},{cfg['seq']},{cfg['channels']}] k={cfg['kernel_size']}"
            # order: CUDA, PyTorch
            print(_row(shape_str, [cuda_ms, pytorch_ms], cfg, dtype))

    print()


if __name__ == "__main__":
    main()
