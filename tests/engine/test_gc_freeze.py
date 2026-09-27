# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The engine's process-global heap freeze (``oasr.engine.gc_freeze``).

What is pinned here is the refcounting, because that is what a process with more
than one engine depends on: the heap stays frozen while *any* engine holds it,
and an engine shutting down must neither leave the heap frozen for nobody nor
unfreeze it under an engine that is still serving.
"""

import gc

import pytest

from oasr.engine import gc_freeze


@pytest.fixture(autouse=True)
def _isolated_gc_state(monkeypatch):
    """Start each test with no holders, and leave the process as found.

    Engines built earlier in the same process hold the heap frozen (a test that
    never calls ``shutdown`` keeps its claim), so the module's count is pinned
    to zero here rather than assumed; the heap is re-frozen afterwards if those
    engines were holding it.
    """
    outer = gc_freeze._holders  # noqa: SLF001
    monkeypatch.setattr(gc_freeze, "_holders", 0)
    gc.unfreeze()
    yield
    gc.unfreeze()
    monkeypatch.setattr(gc_freeze, "_holders", outer)
    if outer:
        gc.freeze()


class _Cycle:
    def __init__(self):
        self.me = self


def test_freeze_moves_live_objects_out_of_collection():
    gc_freeze.freeze()
    assert gc.get_freeze_count() > 0
    assert gc_freeze.holders() == 1


def test_last_release_unfreezes():
    gc_freeze.freeze()
    gc_freeze.release()
    assert gc_freeze.holders() == 0
    assert gc.get_freeze_count() == 0


def test_release_with_another_holder_keeps_the_heap_frozen():
    gc_freeze.freeze()
    gc_freeze.freeze()
    gc_freeze.release()
    assert gc_freeze.holders() == 1
    assert gc.get_freeze_count() > 0


def test_release_frees_cyclic_garbage_that_was_frozen_live():
    """A cycle frozen while alive and orphaned later is reclaimed on release —
    including when another holder re-freezes the survivors."""
    import weakref

    obj = _Cycle()
    ref = weakref.ref(obj)
    gc_freeze.freeze()  # the survivor that keeps the heap frozen
    gc_freeze.freeze()  # the engine that shuts down below
    del obj
    gc.collect()
    assert ref() is not None, "a frozen cycle is exempt from collection"
    gc_freeze.release()
    assert ref() is None, "release must unfreeze, collect, then re-freeze the survivors"
    assert gc_freeze.holders() == 1


def test_unbalanced_release_is_a_no_op():
    gc_freeze.release()
    assert gc_freeze.holders() == 0
