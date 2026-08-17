# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Host→device staging for the small index tensors a step loop rebuilds.

Every engine tick ships a handful of tiny host-side lists to the device — slot
ids, per-row offsets, block ids, sequence lengths.  Written the obvious way,

    torch.tensor(slots, dtype=torch.long, device="cuda")

each one **drains the pipeline**: the source is pageable, and CUDA specifies
that a pageable host→device copy first synchronises the stream, then stages the
bytes through a driver buffer.  On an idle GPU that is ~13 us and invisible; in
a step loop it costs *everything already queued*.  Measured on an RTX 5090 with
four 4096³ fp16 GEMMs in flight: **2.6 ms** per call against 13 us for the same
copy out of pinned memory.  The streaming encoder commit
(:meth:`oasr.cache.attention_cache.PagedAttentionCacheManager.commit_chunks_paged_batched`)
was one such call, placed immediately after the encoder forward — so it waited
out the entire forward and the host could never run a step ahead of the device.

:func:`to_device` is the fix: build the tensor in **pinned** host memory and
copy it with ``non_blocking=True``, which enqueues a real async DMA and returns.
Reuse safety is PyTorch's caching host allocator's job — it records an event on
the pinned block when the copy is issued and will not hand that block out again
until the event fires — so no staging-slot rotation is needed here (unlike the
feature buffers in :class:`~oasr.engine.input_processor.InputProcessor`, whose
pinned pages the *engine* rewrites in place).

The returned tensor is only defined once the copy has run, which for any
consumer enqueued on the same stream is automatic.  A consumer on a **different**
stream needs the usual ``wait_stream``/event ordering, exactly as it would for
any other async producer.

**Not for use inside a CUDA-graph capture.**  The copy would be recorded with
this pinned block's address baked in, and the block goes back to the caching
allocator when the tensor dies — a later replay would then read whatever took
its place.  Every graph in this engine builds its index tensors into static
buffers *before* the capture opens (``graph_cache._capture``,
``decoder_graph._GraphKv``) and should keep doing so; the pageable form used to
make this a loud failure (a capture cannot synchronise) rather than a silent one,
which is the one thing lost here and the reason it is written down.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch

__all__ = ["to_device"]


def to_device(
    values: Sequence[int],
    *,
    dtype: torch.dtype,
    device: Union[torch.device, str],
) -> torch.Tensor:
    """Ship a host-side list of indices to ``device`` without a stream drain.

    Parameters
    ----------
    values : sequence of int
        Host-side indices — slot ids, block ids, offsets, lengths.  Small: this
        is for per-step metadata, not for payload tensors.
    dtype : torch.dtype
        Element type of the resulting tensor.
    device : torch.device or str
        Destination device.  A non-CUDA destination takes the plain path — there
        is no copy to overlap and pinning would only cost a page-lock.

    Returns
    -------
    Tensor
        ``(len(values),)`` on ``device``.  The copy is asynchronous on the
        current stream; readers on that stream need no further ordering.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    host = torch.tensor(values, dtype=dtype, pin_memory=True)
    return host.to(device, non_blocking=True)
