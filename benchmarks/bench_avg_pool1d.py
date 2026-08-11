#!/usr/bin/env python3
"""OASR AvgPool1D benchmark -- BTC CUDA vs PyTorch/cuDNN."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.routines.pooling import run_standalone

if __name__ == "__main__":
    run_standalone("avg_pool1d")
