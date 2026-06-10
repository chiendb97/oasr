# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-mode executor implementations and the shared :class:`Executor` ABC.

The engine is mode-agnostic: it builds exactly one executor at construction
time and delegates every public entry point (admit, feed_chunk, abort,
step, status) to it.  See :mod:`oasr.engine.executor.base` for the
protocol.
"""

from .base import Executor
from .offline import OfflineExecutor
from .streaming import StreamingExecutor

__all__ = [
    "Executor",
    "OfflineExecutor",
    "StreamingExecutor",
]
