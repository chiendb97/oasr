# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Decoder-side KV cache manager for autoregressive generation (AED / LLM).

Storage layer under the incremental decode protocol (multi-paradigm keystone
K6): per-request decoder *self-attention* KV that grows one token per step to
EOS.  Reuses :class:`~oasr.cache.block_pool.BlockPool` — a **separate** pool
instance from the encoder's streaming cache, with decoder geometry
(``block_size_frames`` = tokens per page, ``num_layers`` = decoder layers) —
because the policies must differ:

* the encoder streaming cache is chunk-quantized with sliding-window
  **eviction** (bounded left context);
* AR decode is **append-per-step growth with no eviction** — blocks are
  allocated as the sequence crosses page boundaries and freed only when the
  request finishes.

An AED's cross-attention K/V can live here too, as a second slot per row with a
fixed extent (the encoder output): created at prefill with ``prefill_len`` equal
to that extent and ``max_new_tokens=0``, so admission reserves exactly the pages
it maps and the slot never grows (``decoder_state.build_cross_kv``).  Paging it
is what leaves a decode step with nothing at a per-batch address.

The block tables / ``cache_seqlens`` this manager produces feed the same paged
FMHA path the encoder uses (``oasr.fmha`` with ``block_table``).

**Admission.**  By default a slot claims its whole ceiling (``prefill_len +
max_new_tokens``) at :meth:`DecoderKVCacheManager.can_admit`, so growth can
never fail.  An ``overcommit`` manager checks only the pages a slot maps now:
the AR strategy then budgets rows by their *expected* length and, when a row
outgrows the guess, swaps the youngest row's pages to host memory
(:meth:`~DecoderKVCacheManager.export_slot` /
:meth:`~DecoderKVCacheManager.import_slot`) until there is room.

**Left padding.**  A left-padded prompt's leading pages hold nothing a row ever
reads, so a slot can map them to the pool's scratch page instead of pages of its
own (``create(lead=...)``): every row of a batch is as long as its longest
prompt, and without this a short row held the long one's footprint.

A pool built with ``dtype=torch.float8_e4m3fn`` stores the KV **quantized**
(``EngineConfig.quantization.kv_cache = "fp8_e4m3"``): :meth:`kv_view` then
returns :class:`~oasr.functionals.kv_quant.QuantizedKv` pairs, which the paged
attention reads and dequantizes inside its tile loop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from oasr.functionals.kv_quant import QuantizedKv
from oasr.utils.staging import to_device

from .block_pool import BlockPool
from .types import CacheConfig

__all__ = ["DecoderKVCacheManager", "DecoderKvExhausted", "SwappedSlot"]


class DecoderKvExhausted(RuntimeError):
    """The decoder-KV pool cannot reserve another row at its declared ceiling.

    Its own type because the offline executor treats it the way it treats a
    prefill OOM — reject *this batch* with a reason, rather than letting the
    exception escape the step and fail every unrelated request in flight.
    """


@dataclass
class _SlotState:
    """Bookkeeping for one active request's decoder KV."""

    block_ids: List[int] = field(default_factory=list)
    seqlen: int = 0
    max_tokens: int = 0
    #: Leading ``block_ids`` that are the pool's scratch page (whole pages of
    #: left padding); never freed, never counted as held.
    lead: int = 0


@dataclass
class SwappedSlot:
    """One slot's pages on the host, from :meth:`DecoderKVCacheManager.export_slot`."""

    seqlen: int
    max_tokens: int
    lead: int
    #: :meth:`BlockPool.read_blocks` of the slot's own pages, moved to the host.
    data: Tuple[torch.Tensor, ...]

    @property
    def pages(self) -> int:
        """Pages the slot needs back: its own, not the scratch lead."""
        return int(self.data[0].size(1))


class DecoderKVCacheManager:
    """Per-request paged decoder KV: append-per-step growth, no eviction.

    Lifecycle per request::

        mgr.create(request_id, max_new_tokens=..., prefill_len=...)
        while decoding:
            pos = mgr.append_step([request_id, ...])   # write index per slot
            ... write K/V at pos via block_tables()/the paged FMHA path ...
        mgr.free(request_id)

    Parameters
    ----------
    pool : BlockPool
        A dedicated pool shaped for the decoder
        (``num_layers`` = decoder layers, ``block_size_frames`` = tokens per
        block).  Never share the encoder streaming pool — geometry and
        eviction policy differ.  A pool built with a scratch block
        (``build_pool(scratch_blocks=1)``) can map left padding onto it.
    overcommit : bool
        Admit a slot when the pages it maps *now* are free, rather than when
        its whole ceiling fits beside everyone else's (see the module
        docstring).  Growth can then run the pool out, and whoever drives the
        slots must make room first.
    """

    def __init__(self, pool: BlockPool, *, overcommit: bool = False) -> None:
        self._pool = pool
        self._block_tokens = pool.config.block_size_frames
        self._slots: Dict[str, _SlotState] = {}
        self._views: Dict[int, Tuple[Any, Any]] = {}
        self._slot_views: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self._lock = threading.Lock()
        self.overcommit = bool(overcommit)
        scratch = pool.scratch_block_ids
        self._scratch = int(scratch[0]) if scratch else None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pool(self) -> BlockPool:
        return self._pool

    @property
    def quantized(self) -> bool:
        """Whether the pool stores fp8 K/V with per-(token, head) scales."""
        return bool(self._pool.quantized)

    def num_active(self) -> int:
        return len(self._slots)

    def seqlen(self, request_id: str) -> int:
        return self._slots[request_id].seqlen

    def num_blocks(self, request_id: str) -> int:
        """Block-table entries currently mapped for one slot (scratch lead included).

        Read by callers that cache a block table across a decode step: the table
        only changes when a slot crosses a page boundary, so this is what says
        whether the cached one is still valid.
        """
        return len(self._slots[request_id].block_ids)

    def held_blocks(self, request_id: str) -> int:
        """Pool pages one slot holds: its table, less the scratch lead it shares."""
        state = self._slots[request_id]
        return len(state.block_ids) - state.lead

    @property
    def block_tokens(self) -> int:
        """Tokens per page."""
        return int(self._block_tokens)

    @property
    def has_scratch(self) -> bool:
        """Whether the pool has a scratch page to map left padding onto."""
        return self._scratch is not None

    def reserved_blocks(self) -> int:
        """Blocks every live slot could still demand under its declared ceiling.

        Not the same as *allocated*: a slot holds only the pages it has filled,
        and this is what it would hold at ``prefill_len + max_new_tokens``.
        """
        return sum(self._ceiling(s) for s in self._slots.values())

    def can_admit(self, max_new_tokens: int, prefill_len: int = 0, lead: int = 0) -> bool:
        """Whether one more slot fits: **at everyone's ceiling**, or for an
        ``overcommit`` manager, its prompt's pages are free.

        Admission is what makes growth safe: a decode step allocates a page as a
        row crosses a boundary, and there is no eviction to fall back on, so a
        pool that can run out mid-generation runs out several seconds into a
        request that is already half-answered.  Checking the *reserved* total
        here moves that failure to admission, where it is one rejected batch with
        a reason.  An overcommitting manager leaves that to its caller, which
        makes room by swapping rows out (see the module docstring).
        """
        if self.overcommit:
            return self._pool.num_free_blocks >= self._first_pages(prefill_len) - int(lead)
        need = self._blocks_for(prefill_len + max_new_tokens) - int(lead)
        return self.reserved_blocks() + need <= self._pool.num_blocks

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(
        self, request_id: str, max_new_tokens: int, prefill_len: int = 0, lead: int = 0
    ) -> None:
        """Register a request and pre-allocate blocks for its prefill.

        ``prefill_len`` tokens (prompt / forced-decoder ids) are accounted
        immediately so :meth:`block_tables` covers them; the caller writes
        their K/V in one batched prefill pass.  ``max_new_tokens`` bounds the
        request's total growth (a scheduler admission input, not enforced
        per-step here).

        ``lead`` maps the first ``lead`` pages to the pool's scratch page: whole
        pages of left padding, which the row's window starts past (its
        ``kv_starts``), so no read of them is ever unmasked.  The prefill still
        writes the padding's K/V through the table, and every lead page of
        every row lands on the one scratch page, which is what makes it a
        discard.  At least one page stays the row's own.
        """
        if request_id in self._slots:
            raise ValueError(f"decoder KV slot already exists for request {request_id!r}")
        first = self._first_pages(prefill_len)
        lead = int(lead)
        if lead and self._scratch is None:
            raise ValueError("lead pages need a pool built with a scratch block")
        if not 0 <= lead < first:
            raise ValueError(f"lead must leave the row a page of its own: {lead} of {first}")
        state = _SlotState(max_tokens=prefill_len + max_new_tokens, lead=lead)
        state.block_ids = [int(self._scratch or 0)] * lead + self._pool.allocate(first - lead)
        state.seqlen = prefill_len
        with self._lock:
            self._slots[request_id] = state

    def free(self, request_id: str) -> None:
        """Release every block the request holds (finalize/abort)."""
        with self._lock:
            state = self._slots.pop(request_id, None)
        if state is not None and len(state.block_ids) > state.lead:
            self._pool.free(state.block_ids[state.lead :])

    def export_slot(self, request_id: str) -> SwappedSlot:
        """Copy one slot's pages and bookkeeping to host memory; the slot stays.

        The copy is synchronous: it waits for whatever the stream has queued,
        the writes into these pages included.  A preemption is rare and its
        alternative is recomputing the row, so the simplicity is worth the wait.
        Freeing the slot is the caller's (a ``select`` that drops the row).
        """
        state = self._slots[request_id]
        own = state.block_ids[state.lead :]
        data = tuple(t.to("cpu") for t in self._pool.read_blocks(own))
        return SwappedSlot(state.seqlen, state.max_tokens, state.lead, data)

    def import_slot(self, request_id: str, swapped: SwappedSlot) -> None:
        """Register ``request_id`` with fresh pages holding what ``swapped`` copied out.

        Raises ``RuntimeError`` from the pool when its pages are not free.  The
        K/V come back byte for byte, so a resumed row decodes as if it had never
        left.
        """
        if request_id in self._slots:
            raise ValueError(f"decoder KV slot already exists for request {request_id!r}")
        if swapped.lead and self._scratch is None:
            raise ValueError("a slot with lead pages needs a pool built with a scratch block")
        own = self._pool.allocate(swapped.pages)
        try:
            self._pool.write_blocks(own, swapped.data)
        except Exception:
            self._pool.free(own)
            raise
        state = _SlotState(
            block_ids=[int(self._scratch or 0)] * swapped.lead + own,
            seqlen=swapped.seqlen,
            max_tokens=swapped.max_tokens,
            lead=swapped.lead,
        )
        with self._lock:
            self._slots[request_id] = state

    def free_all(self) -> None:
        for rid in list(self._slots):
            self.free(rid)

    # ------------------------------------------------------------------
    # Step growth
    # ------------------------------------------------------------------

    def append_step(self, request_ids: Sequence[str]) -> List[int]:
        """Advance each slot by one token; returns the write position per slot.

        The returned positions are the *pre-increment* sequence lengths — the
        token index whose K/V the caller is about to write.  A new block is
        allocated transparently when a slot crosses a page boundary
        (raising ``RuntimeError`` from the pool on exhaustion — admission
        control should size ``max_new_tokens`` against pool capacity).
        """
        positions: List[int] = []
        for rid in request_ids:
            state = self._slots[rid]
            if state.seqlen + 1 > len(state.block_ids) * self._block_tokens:
                state.block_ids.extend(self._pool.allocate(1))
            positions.append(state.seqlen)
            state.seqlen += 1
        return positions

    # ------------------------------------------------------------------
    # Paged-attention views
    # ------------------------------------------------------------------

    def block_tables(self, request_ids: Sequence[str], device: torch.device = None) -> torch.Tensor:
        """``(N, max_blocks)`` int32 block table over the given slots (0-padded).

        Built in **pinned** host memory when a device is asked for: this runs on
        every AR decode step, and a pageable host→device copy synchronises the
        stream before it stages (CUDA's rule), so the plain version parks the
        host until the previous step's decoder forward has finished — the one
        thing an incremental decode loop cannot afford to do per step.
        """
        rows = [self._slots[rid].block_ids for rid in request_ids]
        lens = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
        width = int(lens.max()) if len(rows) else 1
        pin = device is not None and torch.device(device).type == "cuda"
        table = torch.zeros(len(rows), width, dtype=torch.int32, pin_memory=pin)
        # One vectorised fill: a ``torch.tensor`` per row was 19 % of Whisper's
        # host samples on page-crossing steps.
        flat = np.fromiter(chain.from_iterable(rows), dtype=np.int32, count=int(lens.sum()))
        table.numpy()[np.arange(width) < lens[:, None]] = flat
        return table.to(device, non_blocking=True) if device is not None else table

    def cache_seqlens(
        self, request_ids: Sequence[str], device: torch.device = None
    ) -> torch.Tensor:
        """``(N,)`` int32 current sequence length per slot.

        Pinned + async for the same reason as :meth:`block_tables`.
        """
        if device is not None:
            return to_device(
                [self._slots[rid].seqlen for rid in request_ids],
                dtype=torch.int32,
                device=device,
            )
        return torch.tensor([self._slots[rid].seqlen for rid in request_ids], dtype=torch.int32)

    def kv_view(self, layer: int) -> Tuple[Any, Any]:
        """Full-pool ``(K, V)`` views for one decoder layer (paged FMHA input).

        Memoized: the pool's backing tensors never move, so the views never
        change, and an AR decode step asks for every layer's on every step —
        rebuilding them there is two tensor allocations per layer per step for an
        answer that is a constant.

        A quantized pool answers with two
        :class:`~oasr.functionals.kv_quant.QuantizedKv` — fp8 data plus its
        ``(blocks, page, H_kv)`` scales — which :class:`oasr.layers.Attention`
        takes in place of a tensor.
        """
        view = self._views.get(layer)
        if view is None:
            k, v = self._pool.get_kv_view(layer)
            if self.quantized:
                ks, vs = self._pool.get_kv_scale_view(layer)
                view = (QuantizedKv(k, ks), QuantizedKv(v, vs))
            else:
                view = (k, v)
            self._views[layer] = view
        return view

    def slot_views(self, layer: int) -> Tuple[torch.Tensor, ...]:
        """``(k, k_scale, v, v_scale)`` of a quantized pool, as the store kernel addresses them.

        ``(1, H_kv, slots, D)`` and ``(1, H_kv, slots)``: one shared row whose
        slot axis is the flattened ``(block, offset)`` index
        (:func:`~oasr.cache.paged_kv.flat_write_index`).  Views, memoized for the
        same reason as :meth:`kv_view`.
        """
        views = self._slot_views.get(layer)
        if views is None:
            k, v = self.kv_view(layer)
            out = []
            for part in (k, v):
                blocks, page, heads, dim = part.data.shape
                out.append(part.data.view(1, blocks * page, heads, dim).permute(0, 2, 1, 3))
                out.append(part.scale.view(1, blocks * page, heads).permute(0, 2, 1))
            views = tuple(out)
            self._slot_views[layer] = views
        return views

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _blocks_for(self, tokens: int) -> int:
        return max(1, -(-tokens // self._block_tokens))

    def _first_pages(self, prefill_len: int) -> int:
        """Table entries a new slot maps at creation (one even for an empty prompt)."""
        return self._blocks_for(int(prefill_len))

    def _ceiling(self, state: _SlotState) -> int:
        """Pool pages a slot can come to hold: its ceiling, less its scratch lead."""
        return max(0, self._blocks_for(state.max_tokens) - state.lead)

    @staticmethod
    def build_pool(
        *,
        num_layers: int,
        n_kv_head: int,
        head_dim: int,
        block_tokens: int = 16,
        max_num_blocks: int = 1024,
        max_batch_size: int = 32,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        scratch_blocks: int = 0,
    ) -> BlockPool:
        """Convenience constructor for a decoder-shaped :class:`BlockPool`.

        ``scratch_blocks=1`` adds the page left padding is mapped onto (see
        :meth:`DecoderKVCacheManager.create`); it is outside ``max_num_blocks``.
        """
        cfg = CacheConfig(
            num_layers=num_layers,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            hidden_dim=n_kv_head * head_dim,
            kernel_size=1,
            chunk_size=block_tokens,
            block_size_frames=block_tokens,
            max_num_blocks=max_num_blocks,
            max_batch_size=max_batch_size,
            device=device if device is not None else torch.device("cuda"),
            dtype=dtype,
        )
        return BlockPool(cfg, scratch_blocks=int(scratch_blocks))
