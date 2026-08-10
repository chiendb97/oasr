#!/usr/bin/env python3
"""OASR grouped/depthwise and pointwise Conv2D benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.routines.conv import run_grouped_conv2d_report

if __name__ == "__main__":
    run_grouped_conv2d_report()
