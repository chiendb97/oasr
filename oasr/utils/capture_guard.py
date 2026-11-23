# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""One lock between CUDA-graph capture and CUDA work on another thread.

``torch.cuda.graph`` captures in ``"global"`` mode, in which the driver refuses
an allocation, a pinned-memory request or a synchronisation from *any* thread
while a capture is open, and the refusal invalidates the capture as well.  The
engine's captures all run on the step thread, so this never came up until a
second thread issued CUDA work of its own: the offline VAD thread
(``vad.mode="segment"`` with ``vad.device="cuda"``) failed its batch with
``operation not permitted when stream is capturing`` whenever a first-seen shape
was being captured, and that capture was lost.

Every capture in the process holds :func:`capture_guard` while it is open, and
so does any thread that issues CUDA work alongside the step loop.  Captures are
rare (start-up and first-seen shapes) and a few tens of milliseconds long, so
the other thread waits for one at most; it never waits for the step loop itself.
Switching the captures to ``"thread_local"`` mode would lift the restriction
without a lock, but it changes what a capture tolerates on its own thread too,
across every capture cache in the engine.
"""

from __future__ import annotations

import threading

__all__ = ["capture_guard"]

_LOCK = threading.RLock()


def capture_guard() -> "threading.RLock":
    """The process-wide lock a capture, or concurrent CUDA work, holds."""
    return _LOCK
