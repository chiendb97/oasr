# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-exports from autotuner.py.

All registry types have been consolidated into autotuner.py (FlashInfer-style).
Import from oasr.tune or oasr.tune.autotuner instead.
"""

from .autotuner import BackendEntry, BackendRegistry, _global_registry

__all__ = ["BackendEntry", "BackendRegistry", "_global_registry"]
