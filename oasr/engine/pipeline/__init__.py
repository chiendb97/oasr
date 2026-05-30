# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-mode pipeline implementations and the shared :class:`Pipeline` ABC.

The engine is mode-agnostic: it builds exactly one pipeline at construction
time and delegates every public entry point (admit, feed_chunk, abort,
step, status) to it.  See :mod:`oasr.engine.pipeline.base` for the
protocol.
"""

from .base import Pipeline
from .offline import OfflinePipeline
from .packing import PackingPipeline
from .streaming import StreamingPipeline

__all__ = [
    "Pipeline",
    "OfflinePipeline",
    "PackingPipeline",
    "StreamingPipeline",
]
