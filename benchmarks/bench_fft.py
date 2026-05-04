#!/usr/bin/env python3
"""OASR rFFT Benchmark -- CUDA vs PyTorch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.routines.fft import run_standalone


if __name__ == "__main__":
    run_standalone("rfft")
