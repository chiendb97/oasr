# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Undo the process-global damage an aborted CUDA-graph capture leaves behind.

Every capture cache in this package treats capture as best-effort: a forward
that reads a device value host-side (``.item()`` / ``.tolist()``) invalidates the
capture stream, and the honest response is to remember it, log it and run eager
(``offline_graph``, ``decoder_graph``, ``predictor_graph``, ``graph_cache``).
Swallowing the exception is right.  It is also not enough, because the abort does
not stay inside the cache: ``torch.cuda.graph.__exit__`` raises alongside the
body, so the two things ``capture_end`` would have done never happen.

**The allocator stays bound to the capture's private pool.**  ``__enter__`` called
``beginAllocateToPool``; without the matching ``endAllocateToPool`` *every later
allocation in the process* is served out of that pool and is never returned --
not by ``del``, not by ``gc.collect()``, not by ``torch.cuda.empty_cache()``.
Measured: a 256 MiB tensor allocated long after an unrelated abort, then freed,
stays gone.  And every later capture on that pool dies with
``beginAllocateToPool: already recording to mempool_id``, so one bad shape ends
capture for the whole process while the feature still reports itself on.

That is not hypothetical.  Nemotron's FastConformer offline forward reads a
device value host-side, so its capture aborts; the engine then allocated its KV
pool and decode buffers into the stuck pool and stranded 2.7--3.2 GiB per engine.
Six engines into ``tests/test_accuracy.py`` the 7B speech-LLM could no longer
size its decoder KV pool, rejected most batches, and returned empty transcripts
-- a 42% WER for a checkpoint that scores 5.6% run on its own.

**The generator stays in capture mode.**  ``__enter__`` put the device's default
generator into capture mode and only ``capture_end`` takes it out, so every CUDA
RNG call afterwards fails with ``Offset increment outside graph capture
encountered unexpectedly`` -- in this engine ``torch.multinomial`` in
:mod:`oasr.engine.generation.sampling`, i.e. one unrelated capture failure turns
every sampled decode into an error, far from the edit.

The two need different remedies, and the overlap between them is partial --
measured, because it decides whether either half may be dropped:

* ``_cuda_endAllocateToPool`` + ``_cuda_releasePool`` stops the diversion **and**
  hands back blocks already stranded, but leaves the generator capturing.
* A throwaway *successful* capture runs the generator epilogue, and its own
  ``__exit__`` incidentally ends the diversion -- but it cannot return bytes the
  process already allocated into the stuck pool.
* ``manual_seed`` and ``Generator.set_state`` do nothing for either.

So both are needed, in that order: release first (recovering what is already
lost), then reset the generator.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = ["recover_from_failed_capture", "restore_rng_after_failed_capture"]


def _release_pool(device: torch.device, pool: Any) -> bool:
    """Stop the allocator serving out of ``pool`` and give its blocks back."""
    if pool is None:
        return True
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        torch._C._cuda_endAllocateToPool(index, pool)
        torch._C._cuda_releasePool(index, pool)
    except Exception as exc:  # pragma: no cover - private API, version-dependent
        logger.warning(
            "could not release the CUDA graph pool after a failed capture (%s); "
            "allocations on %s may not be returned until the process restarts",
            exc,
            device,
        )
        return False
    return True


def restore_rng_after_failed_capture(device: torch.device) -> bool:
    """Clear ``device``'s generator capture flag with one throwaway capture.

    Returns whether CUDA RNG works on ``device`` afterwards.  Uses its own pool,
    so a recovery cannot grow the pool the engine replays out of.
    """
    if device.type != "cuda":
        return True
    graph = None
    try:
        with torch.cuda.device(device):
            probe = torch.empty(1, device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                probe.fill_(0)
            torch.cuda.synchronize(device)
    except Exception as exc:  # pragma: no cover - the recovery itself failing
        logger.warning(
            "could not reset the CUDA generator after a failed graph capture (%s); "
            "sampled decoding on %s may raise until the process restarts",
            exc,
            device,
        )
        return False
    finally:
        if graph is not None:
            try:
                graph.reset()
            except Exception:  # pragma: no cover - teardown must not raise
                pass
    return True


def recover_from_failed_capture(device: torch.device, pool: Optional[Any] = None) -> bool:
    """Full undo for an aborted capture: the allocator first, then the generator.

    ``pool`` is the handle the aborted capture was recording into; pass it, or
    the process keeps allocating into it.  The pool is released, so the caller
    must take a fresh handle for any later capture.  Returns whether both halves
    succeeded.
    """
    if device.type != "cuda":
        return True
    freed = _release_pool(device, pool)
    return restore_rng_after_failed_capture(device) and freed
