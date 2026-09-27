# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Keep full garbage collections off the engine thread.

A constructed engine leaves ~440k objects tracked by the cyclic garbage
collector — torch, the model's module tree, the tokenizer tables, the JIT
modules.  Almost none of them will ever become garbage, but a generation-2
collection walks all of them, and on this box that walk measured **162-170 ms**
of wall clock with the step loop stopped (5 configs, one full collection every
few hundred milliseconds of offline decode).  In a server that is one tick in
which every in-flight stream stalls at once: invisible in a throughput median,
and the whole p99.

``gc.freeze()`` moves every object tracked at that moment into a permanent
generation the collector never walks, so later collections cost what was
allocated *since* — requests, outputs, per-tick tensors.  It is the standard
remedy for a long-lived serving heap, and it is process-global, which is why
this module owns it rather than each engine calling it directly:

* **Refcounted.**  Freezing is idempotent per engine, and the heap is unfrozen
  only when the last engine that asked for it releases it.  An engine that
  shuts down while another is still serving re-freezes what survives, so the
  survivor keeps its short collections.
* **Collected first.**  ``gc.collect()`` runs before each freeze, so only live
  objects are frozen.  An object frozen while live and later orphaned *in a
  reference cycle* is not reclaimed until :func:`release` unfreezes it; the
  engine itself is not such an object — a dropped engine returns its device
  memory through reference counting alone (measured: 182 -> 32 MiB offline,
  641 -> 32 MiB streaming, with the cyclic collector disabled).

``EngineConfig.gc_freeze`` is the switch; ``False`` restores the default
collector behaviour for an embedding application that manages its own heap.
"""

from __future__ import annotations

import gc
import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_holders = 0


def freeze() -> None:
    """Collect, then freeze the heap; one call per engine that wants it."""
    global _holders
    with _lock:
        gc.collect()
        gc.freeze()
        _holders += 1
        logger.debug("gc heap frozen (%d objects, %d holders)", gc.get_freeze_count(), _holders)


def release() -> None:
    """Drop one engine's claim; unfreeze when it was the last one.

    With other holders left, what survives is frozen again: unfreezing is the
    only way to let this engine's cyclic garbage go, and re-freezing is what
    keeps the remaining engines' collections short.
    """
    global _holders
    with _lock:
        if _holders == 0:
            return
        _holders -= 1
        gc.unfreeze()
        if _holders > 0:
            gc.collect()
            gc.freeze()


def holders() -> int:
    """Engines currently holding the heap frozen (for tests)."""
    return _holders


__all__ = ["freeze", "holders", "release"]
