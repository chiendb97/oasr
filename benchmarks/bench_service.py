#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim -- see benchmarks/run.py --family service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._compat import forward  # noqa: E402

if __name__ == "__main__":
    sys.exit(forward("service"))
