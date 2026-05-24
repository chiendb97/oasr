#!/usr/bin/env python3
"""OASR FMHA Benchmark -- CuteDSL kernel vs PyTorch SDPA."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.routines.attention import (
    DEFAULT_CONFIGS,
    SUBROUTINES,
    compute_fmha_tflops,
    setup_fmha_bias,
    setup_fmha_bias_seqlens,
    setup_fmha_offline,
    setup_fmha_paged,
    setup_fmha_paged_bias,
    setup_fmha_seqlens,
)
from benchmarks.routines.bench_utils import bench_fn

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}

# Mirrors ``benchmarks/routines/attention.py::_SETUP_FNS``; kept local so
# bench_fmha.py can stay a thin standalone wrapper.
_SETUP = {
    "fmha_offline": setup_fmha_offline,
    "fmha_bias": setup_fmha_bias,
    "fmha_seqlens": setup_fmha_seqlens,
    "fmha_bias_seqlens": setup_fmha_bias_seqlens,
    "fmha_paged": setup_fmha_paged,
    "fmha_paged_bias": setup_fmha_paged_bias,
}

_COL_SHAPE = 36
_COL_TIME  = 14
_COL_TFLOPS = 8

_HEADER = (
    f"{'(B,H,Hkv,Tq,Tk,D)':>{_COL_SHAPE}}"
    f"  {'CuteDSL':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
    f"  {'PyTorch':>{_COL_TIME}}  {'TFLOPS':>{_COL_TFLOPS}}"
    f"  {'speedup':>9}"
)
_SEP       = "-" * len(_HEADER)
_TITLE     = "OASR FMHA Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _fmt_time(ms):
    return f"{ms:.4f}ms"


def _fmt_tflops(t):
    return f"{t:.1f}T"


def _fmt_speedup(cute_ms, pt_ms):
    if cute_ms <= 0:
        return "    n/a"
    return f"{pt_ms / cute_ms:7.2f}x"


def _row(shape_str, cute_ms, pt_ms, cfg):
    cute_tf = compute_fmha_tflops(cfg["B"], cfg["H"], cfg["T_q"], cfg["T_k"], cfg["D"], cute_ms)
    pt_tf   = compute_fmha_tflops(cfg["B"], cfg["H"], cfg["T_q"], cfg["T_k"], cfg["D"], pt_ms)
    parts = [f"{shape_str:>{_COL_SHAPE}}"]
    parts.append(f"  {_fmt_time(cute_ms):>{_COL_TIME}}  {_fmt_tflops(cute_tf):>{_COL_TFLOPS}}")
    parts.append(f"  {_fmt_time(pt_ms):>{_COL_TIME}}  {_fmt_tflops(pt_tf):>{_COL_TFLOPS}}")
    parts.append(f"  {_fmt_speedup(cute_ms, pt_ms):>9}")
    return "".join(parts)


def _shape_str(cfg):
    return (
        f"({cfg['B']},{cfg['H']},{cfg['H_kv']},"
        f"{cfg['T_q']},{cfg['T_k']},{cfg['D']})"
    )


def _run_shape(sub, cfg, dtype):
    setup = _SETUP[sub]
    cute_fn, pt_fn = setup(cfg, dtype)
    # Warm caches + first-call CuteDSL compile.
    cute_fn(); pt_fn()
    torch.cuda.synchronize()
    cute_ms, _ = bench_fn(cute_fn)
    pt_ms,   _ = bench_fn(pt_fn)
    return cute_ms, pt_ms


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); return

    print(_TITLE)
    print(_TITLE_SEP)

    for dtype in DTYPES:
        for sub in SUBROUTINES:
            print(f"\n--- {sub} ({DTYPE_NAMES[dtype]}) ---")
            print(_HEADER)
            print(_SEP)
            for cfg in DEFAULT_CONFIGS[sub]:
                try:
                    cute_ms, pt_ms = _run_shape(sub, cfg, dtype)
                except Exception as exc:
                    print(f"{_shape_str(cfg):>{_COL_SHAPE}}  FAILED: {exc}")
                    continue
                print(_row(_shape_str(cfg), cute_ms, pt_ms, cfg))

    print()


if __name__ == "__main__":
    main()
