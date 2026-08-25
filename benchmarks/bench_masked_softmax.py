#!/usr/bin/env python3
"""OASR fused masked/biased softmax benchmark — fused vs the op sequence it replaces."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.routines.softmax import run_standalone

if __name__ == "__main__":
    run_standalone("masked_softmax")
