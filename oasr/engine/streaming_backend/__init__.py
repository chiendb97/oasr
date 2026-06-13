# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Pluggable streaming-encoder backends for the ASR engine.

Importing this package registers the built-in backends (paged + stateful) so
:func:`build_streaming_backend` resolves them by ``encoder.streaming_kind``.  Add
a new streaming runtime by subclassing :class:`StreamingEncoderBackend` and
decorating it with :func:`register_streaming_backend`.
"""

# Import for side effects: each module registers its backend on import.
from . import paged, stateful  # noqa: E402,F401
from .base import (
    StreamingEncoderBackend,
    build_streaming_backend,
    register_streaming_backend,
)

__all__ = [
    "StreamingEncoderBackend",
    "build_streaming_backend",
    "register_streaming_backend",
]
