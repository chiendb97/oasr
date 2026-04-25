# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Cheap NVTX range helpers for profiling streaming hot paths.

Set ``OASR_NVTX=1`` to emit NVTX markers around the engine step stages.
When disabled (default), ``nvtx_push`` / ``nvtx_pop`` are bound to no-op
functions so they cost nothing in production.
"""

from __future__ import annotations

import os

_ENABLED = os.environ.get("OASR_NVTX", "0") == "1"

if _ENABLED:  # pragma: no cover - profiling-only path
    from torch.cuda import nvtx as _nvtx

    nvtx_push = _nvtx.range_push
    nvtx_pop = _nvtx.range_pop
else:
    def nvtx_push(name: str) -> None:
        pass

    def nvtx_pop() -> None:
        pass


class nvtx_range:
    """Context manager wrapping ``nvtx_push`` / ``nvtx_pop``.

    Cheap no-op when ``OASR_NVTX`` is not set.  Prefer the bare
    ``nvtx_push``/``nvtx_pop`` calls in tight loops to avoid the
    context-manager overhead.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __enter__(self) -> "nvtx_range":
        nvtx_push(self._name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        nvtx_pop()


__all__ = ["nvtx_push", "nvtx_pop", "nvtx_range"]
