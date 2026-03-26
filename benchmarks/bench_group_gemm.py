#!/usr/bin/env python3
"""OASR Group GEMM Benchmark — Auto-Tuning vs CUTLASS vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import oasr
from benchmarks.routines.bench_utils import bench_fn
from benchmarks.routines.gemm import setup_group_gemm

CONFIGS = [
    {"num_groups": 32,  "M": 256,   "N":  64, "K":  64},
    {"num_groups": 32,  "M": 256,   "N": 128, "K":  64},
    {"num_groups": 64,  "M": 16000, "N": 256, "K": 2048},
    {"num_groups": 64,  "M": 16000, "N": 512, "K": 2048},
]

DTYPES = [torch.bfloat16]
DTYPE_NAMES = {torch.bfloat16: "bfloat16"}

_COL_SHAPE  = 28
_COL_TIME   = 14
_COL_METRIC =  8

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'Auto-Tuning':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
    f"  {'CUTLASS':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'TFLOPS':>{_COL_METRIC}}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR Group GEMM Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _tflops(num_groups, M, N, K, ms):
    return 2.0 * num_groups * M * N * K / (ms * 1e-3) / 1e12


def _fmt(ms):
    return f"{ms:.4f}ms"


def _fmt_t(t):
    return f"{t:.1f}T"


def _row(shape_str, times, cfg):
    g, M, N, K = cfg["num_groups"], cfg["M"], cfg["N"], cfg["K"]
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    for ms in times:
        parts.append(
            f"  {_fmt(ms):>{_COL_TIME}}  {_fmt_t(_tflops(g, M, N, K, ms)):>{_COL_METRIC}}"
        )
    return "".join(parts)


def _make_tensors(cfg, dtype):
    torch.manual_seed(0)
    g, M, N, K = cfg["num_groups"], cfg["M"], cfg["N"], cfg["K"]
    low  = max(16, M // 2)
    high = max(low + 1, M * 2)
    Ms     = torch.randint(low=low, high=high, size=(g,), device="cuda").tolist()
    L      = sum(Ms)
    A      = torch.randn(L, K, device="cuda", dtype=dtype)
    B      = torch.randn(g, N, K, device="cuda", dtype=dtype)
    offset = torch.cumsum(torch.tensor(Ms, dtype=torch.int32, device="cuda"), dim=0)
    return A, B, offset


def _run_shape(cfg, dtype):
    g, M, N, K = cfg["num_groups"], cfg["M"], cfg["N"], cfg["K"]
    low  = max(16, M // 2)
    high = max(low + 1, M * 2)
    torch.manual_seed(0)
    Ms   = torch.randint(low=low, high=high, size=(g,), device="cuda").tolist()
    problem_sizes = [(m, N, K) for m in Ms]
    oasr_fn, pytorch_fn = setup_group_gemm(problem_sizes, dtype)

    A, B, offset = _make_tensors(cfg, dtype)

    with oasr.autotune():
        oasr.group_gemm(A, B, offset)
        autotune_ms, _ = bench_fn(lambda: oasr.group_gemm(A, B, offset))

    cutlass_ms, _ = bench_fn(oasr_fn)
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
            g, M, N, K = cfg["num_groups"], cfg["M"], cfg["N"], cfg["K"]
            times = _run_shape(cfg, dtype)
            print(_row(f"{g}x({M},{N},{K})", list(times), cfg))

    print()


if __name__ == "__main__":
    main()
