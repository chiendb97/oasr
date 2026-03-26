# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-export from autotuner.py.

AutoTuner has been consolidated into autotuner.py (FlashInfer-style).
Import from oasr.tune or oasr.tune.autotuner instead.
"""

from .autotuner import AutoTuner

__all__ = ["AutoTuner"]
