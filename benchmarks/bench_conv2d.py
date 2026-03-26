#!/usr/bin/env python3
"""OASR Conv2D Benchmark — Auto-Tuning vs CUTLASS vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import oasr
from benchmarks.routines.bench_utils import bench_fn
from benchmarks.routines.conv import setup_conv2d

CONFIGS = [
    {"N": 16, "H": 200, "W": 80,  "IC":   1, "K":  64, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 16, "H": 100, "W": 40,  "IC":  64, "K":  64, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 32, "H": 200, "W": 80,  "IC":   1, "K":  64, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 32, "H": 100, "W": 40,  "IC":  64, "K":  64, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N":  8, "H": 300, "W": 80,  "IC":   1, "K": 256, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N":  8, "H": 150, "W": 40,  "IC": 256, "K": 256, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 16, "H": 100, "W": 40,  "IC": 128, "K": 128, "R": 3, "S": 3, "pad": 1, "stride": 1},
    {"N": 16, "H": 100, "W": 40,  "IC": 256, "K": 256, "R": 3, "S": 3, "pad": 1, "stride": 1},
]

DTYPES = [torch.float16]
DTYPE_NAMES = {torch.float16: "float16"}

_COL_SHAPE  = 36
_COL_TIME   = 12
_COL_METRIC =  8

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'Auto-Tuning':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
    f"  {'CUTLASS':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR Conv2D Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _tflops(cfg, ms):
    N, H, W = cfg["N"], cfg["H"], cfg["W"]
    IC, K, R, S = cfg["IC"], cfg["K"], cfg["R"], cfg["S"]
    pad, stride = cfg["pad"], cfg["stride"]
    OH = (H + 2 * pad - R) // stride + 1
    OW = (W + 2 * pad - S) // stride + 1
    return 2.0 * N * OH * OW * K * IC * R * S / (ms * 1e-3) / 1e12


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_t(t):
    return f"{t:.2f}T"


def _row(shape_str, times, cfg):
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    for ms in times:
        parts.append(
            f"  {_fmt(ms):>{_COL_TIME}}  {_fmt_t(_tflops(cfg, ms)):>{_COL_METRIC}}"
        )
    return "".join(parts)


def _shape_str(cfg):
    c = cfg
    return (
        f"[{c['N']},{c['H']},{c['W']},{c['IC']}]->[{c['N']},{c['K']}]"
        f" {c['R']}x{c['S']} p={c['pad']} s={c['stride']}"
    )


def _run_shape(cfg, dtype):
    N, H, W = cfg["N"], cfg["H"], cfg["W"]
    IC, K   = cfg["IC"], cfg["K"]
    R, S    = cfg["R"], cfg["S"]
    pad, stride = cfg["pad"], cfg["stride"]

    x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype)
    w = torch.randn(K, R, S, IC, device="cuda", dtype=dtype)
    b = torch.randn(K,          device="cuda", dtype=dtype)

    with oasr.autotune():
        oasr.conv2d(x, w, b, pad, pad, stride, stride)
        autotune_ms, _ = bench_fn(lambda: oasr.conv2d(x, w, b, pad, pad, stride, stride))

    _, pytorch_fn = setup_conv2d(N, H, W, IC, K, R, S, pad, stride, dtype)
    cutlass_ms, _ = bench_fn(lambda: oasr.conv2d(x, w, b, pad, pad, stride, stride))
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
        for cfg in CONFIGS:
            times = _run_shape(cfg, dtype)
            print(_row(_shape_str(cfg), list(times), cfg))

    print()


if __name__ == "__main__":
    main()
