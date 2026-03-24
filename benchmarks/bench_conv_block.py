#!/usr/bin/env python3
"""Performance benchmarks for Conformer conv block. See oasr_benchmark.py for unified CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.routines.composite import run_standalone

if __name__ == "__main__":
    run_standalone()
