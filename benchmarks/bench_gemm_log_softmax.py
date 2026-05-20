#!/usr/bin/env python3
"""OASR Fused GEMM + log_softmax Benchmark — CUTLASS vs PyTorch.

Compares ``oasr.gemm_log_softmax(x, W, b)`` against the unfused PyTorch
baseline ``F.log_softmax(F.linear(x, W, b), dim=-1)`` (the CTC head).

Run:
    python benchmarks/bench_gemm_log_softmax.py
    python benchmarks/bench_gemm_log_softmax.py --profile --target cutlass
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.routines.gemm_log_softmax import run_standalone


if __name__ == "__main__":
    run_standalone()
