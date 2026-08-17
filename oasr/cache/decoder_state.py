# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-row self-attention KV for the incremental (AED / speech-LLM) decoders.

Both AR decoder surfaces used to track the generation offset as a **shared
scalar** — ``WhisperDecoder`` kept an ``int`` ``pos`` used for the position
embedding *and* the KV write offset, ``Qwen2Lm`` wrote new KV at ``len`` into a
shared-capacity buffer.  That scalar is why two decode groups could never be
merged after the fact: rows sitting at different offsets cannot share a forward,
so a trickle of arrivals became N narrow groups whose step counts *add* (an AR
step is weight-read bound, so N groups cost ~N times one group of the same total
rows — measured 1.75x on Qwen2-Audio-7B).

This module is the scalar made per-row.  One KV object holds every layer's K/V
for one decode group plus three row-indexed vectors:

``lens``
    Keys cached per row — and therefore the row's next **write index**.
``starts``
    First valid key per row, i.e. left padding.  The speech-LLM prompt is
    variable length and left-padded (HF's masked-generate convention), so its
    valid window is ``[starts, lens)``; an AED prompt is a fixed SOT sequence and
    leaves this ``None``.
``positions``
    Rotary / absolute position ids, which are *derived* (``lens - starts``)
    rather than stored: a fourth vector that must agree with the other two is a
    fourth vector that can disagree with them.

Together they are exactly the ``kv_lens`` / ``kv_starts`` window pair
:class:`oasr.layers.Attention` already takes as two length vectors — so per-row
offsets cost no materialized mask and reach the fused kernel on the same terms
the uniform case did.

Storage modes
-------------
:class:`DecoderKv`, ``cap`` given (the strategies pass prompt + generation cap)
    One ``(B, H_kv, cap, D)`` buffer per layer, allocated once; a step writes its
    own slot in place.  Rows at different offsets scatter; the uniform case —
    every group that has never been merged — keeps the single-slice write it had
    before.  Overflow **grows** the buffer rather than degrading, because a
    merged group has no uniform length to degrade to.
:class:`DecoderKv`, ``cap is None``
    Exact-size ``torch.cat`` growth, for direct ``prefill``/``step`` callers and
    the teacher-forced alignment pass.  Legacy mode cannot express per-row
    offsets (``cat`` appends one width to every row) and therefore cannot merge;
    :meth:`~DecoderKv.can_merge` says so rather than producing a silently wrong
    cache.
:class:`PagedDecoderKv`
    Pages out of a shared :class:`~oasr.cache.block_pool.BlockPool` via
    :class:`~oasr.cache.decoder_kv.DecoderKVCacheManager`, addressed by a per-row
    block table.  A row holds only the pages it has *filled*, the pool is
    allocated once for the process rather than per prefill, merging becomes free
    (the block tables concatenate and no K/V moves), and the addresses are stable
    enough to capture a step into a CUDA graph.  It is **not** the default: the
    block-table indirection measures 3% slower on Qwen2-Audio-7B and 9% on
    whisper-tiny, and it saves no VRAM while admission reserves each row's
    ceiling — see ``.artifacts/engine_perf.md`` §3.2b.  A row's pages are also its
    own, so the repeated-index ``select`` that expands and reorders a beam grid
    would alias two slots onto one page; :meth:`~PagedDecoderKv.select` refuses
    rather than corrupting, and the strategy keeps beam search on dense storage.

The untouched tail of a capacity buffer (and of a recycled page) is zeroed, never
``empty``: the buffer is handed to the attention kernel *whole* (buffer + length,
which is what skips a per-layer copy of a stride-gapped slice), so the kernel may
read the in-bounds tail of its final partial K block.  Uninitialized memory there
can hold a NaN bit pattern, and a NaN in ``v`` survives any mask through
``P @ V``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

from .paged_kv import flat_write_index

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .decoder_kv import DecoderKVCacheManager

__all__ = ["DecoderKv", "PagedDecoderKv", "build_kv", "consume_cat_rows"]

#: Capacity-buffer growth granularity, in tokens.  Growth is a safety valve —
#: the strategies size ``cap`` from the batch's generation cap — so this only
#: has to avoid reallocating once per token when it does happen.
_GROW_ROUND = 32


class _RowIndexed:
    """The row vectors and everything derivable from them.

    Shared by both storage modes so ``positions``, the window pair and the
    consumed-state guard have exactly one definition — a paged cache and a dense
    one differ in *where the K/V lives*, not in what a row's offset means.
    """

    lens: torch.Tensor
    lens_host: List[int]
    starts: Optional[torch.Tensor]
    consumed: bool

    @property
    def batch(self) -> int:
        return len(self.lens_host)

    @property
    def max_len(self) -> int:
        """Longest cached row (host-side; no device sync)."""
        return max(self.lens_host, default=0)

    @property
    def uniform(self) -> bool:
        """Whether every row sits at the same offset — the never-merged case."""
        return len(set(self.lens_host)) <= 1

    def positions(self) -> torch.Tensor:
        """``(B,)`` int64 next position id per row (``lens - starts``).

        Meaningful once the cached region really is the row's prefix — that is,
        after a left-padded prefill has committed.  A left-padding family builds
        its *prompt's* position ids from the validity mask instead (HF's
        ``cumsum(valid) - 1``), because during that one forward ``lens`` is still
        zero while ``starts`` already describes where the prompt will land.
        """
        pos = self.lens if self.starts is None else self.lens - self.starts
        return pos.to(torch.int64)

    def position_ids(self, t_new: int) -> torch.Tensor:
        """``(B, t_new)`` position ids for the tokens about to be appended."""
        base = self.positions().unsqueeze(1)
        return base + torch.arange(t_new, device=base.device, dtype=base.dtype).unsqueeze(0)

    def _window(self, t_new: int) -> Dict[str, Any]:
        """The ``kv_lens`` / ``kv_starts`` pair after ``t_new`` more keys."""
        kwargs: Dict[str, Any] = {"kv_lens": self.lens + t_new}
        if self.starts is not None:
            kwargs["kv_starts"] = self.starts
        return kwargs

    def _advance_rows(self, t_new: int) -> None:
        self.lens = self.lens + t_new
        self.lens_host = [n + t_new for n in self.lens_host]

    def _check_live(self) -> None:
        if self.consumed:
            raise RuntimeError(
                "this decoder KV state was consumed by merge(); use the merged "
                "state returned by it, not either operand"
            )


@dataclass
class DecoderKv(_RowIndexed):
    """Row-indexed self-attention KV for one AR decode group.

    Attributes
    ----------
    k, v : list[Tensor | None]
        Per layer, ``(B, H_kv, width, D)``.  ``width`` is ``cap`` in capacity
        mode and the exact cached length in legacy mode; ``None`` until the
        layer's first write.
    lens : Tensor
        ``(B,)`` int32 on the compute device — cached keys per row, i.e. the next
        write index.
    lens_host : list[int]
        Host mirror of ``lens``.  Every consumer of the *maximum* length is
        host-side (the attention extent, the overflow check), and reading it off
        the device would be a per-step ``cudaStreamSynchronize`` on the hottest
        AR path — the same reason ``PagedKVCache`` mirrors ``host_seqlen_max``.
    starts : Tensor | None
        ``(B,)`` int32 first valid key per row, or ``None`` when the family never
        left-pads.
    cap : int | None
        Capacity-buffer width for *newly allocated* layers; ``None`` selects
        legacy cat-growth.  An individual layer's real width is its buffer's,
        which can already be wider after a grow.
    """

    k: List[Optional[torch.Tensor]]
    v: List[Optional[torch.Tensor]]
    lens: torch.Tensor
    lens_host: List[int]
    starts: Optional[torch.Tensor] = None
    cap: Optional[int] = None
    #: Set by :meth:`merge`, which releases this cache's tensors into the merged
    #: one.  Checked rather than left to fail later: a consumed state's layers are
    #: ``None``, and the first symptom without this is SDPA reporting that its
    #: ``key`` argument is not a tensor, several frames from the real mistake.
    consumed: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def empty(
        cls,
        num_layers: int,
        batch_size: int,
        device: torch.device,
        *,
        starts: Optional[torch.Tensor] = None,
        cap: Optional[int] = None,
    ) -> "DecoderKv":
        """A group with no keys cached yet.

        ``starts`` is the left-padding vector, known at prefill from the prompt's
        validity mask; families with a fixed-length prompt pass ``None``.
        """
        return cls(
            k=[None] * num_layers,
            v=[None] * num_layers,
            lens=torch.zeros(batch_size, dtype=torch.int32, device=device),
            lens_host=[0] * batch_size,
            starts=None if starts is None else starts.to(torch.int32),
            cap=None if cap is None else int(cap),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return len(self.k)

    def mask_kwargs(self, t_new: int, *, trimmed: bool = False) -> Dict[str, Any]:
        """``Attention`` window kwargs for a forward that appends ``t_new`` keys.

        Empty when there is nothing to mask — a trimmed or exact-size cache with
        no left padding *is* the valid region for every row, and passing a
        redundant length vector there would only move the call onto a different
        attention backend for identical math.
        """
        if (self.cap is None or trimmed) and self.starts is None:
            return {}
        return self._window(t_new)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, *, trim: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """Write one layer's new K/V at each row's own offset.

        ``k_new`` / ``v_new`` are ``(B, H_kv, t_new, D)``.  Returns
        ``(k, v, kv_extent)`` ready for :class:`oasr.layers.Attention` —
        ``kv_extent`` is the cache's logical length when the returned tensors are
        a wider capacity buffer, and ``None`` when they are exact.

        ``trim`` returns just the region this call wrote instead of the whole
        capacity buffer.  It is for a **prefill** — every row at offset 0, the
        caller masking with ``is_causal`` alone: handing a 4-token Whisper prompt
        the full 452-wide buffer plus a length vector would compute (and mask) two
        orders of magnitude more score matrix for the same answer.  A *step* wants
        the opposite — the buffer whole, because slicing it is the stride gap that
        costs the fused kernel a per-layer copy.  Requiring an empty cache rather
        than merely a uniform one is what lets the paged store honour the same
        flag: there, "the region this call wrote" is only addressable densely
        while it is the whole cache.
        """
        self._check_live()
        t_new = k_new.size(2)
        if self.cap is None:
            return self._append_legacy(layer, k_new, v_new)
        if trim and self.max_len:
            raise RuntimeError("trim is for a prefill: every row must still be empty")

        extent = self.max_len + t_new
        buf_k = self._buffer(layer, k_new, extent)
        buf_v = self._buffer(layer, v_new, extent, value=True)
        if self.uniform:
            # Never-merged group: one slice assignment, as before per-row offsets.
            off = self.lens_host[0]
            buf_k[:, :, off : off + t_new] = k_new
            buf_v[:, :, off : off + t_new] = v_new
        else:
            idx = self._scatter_index(t_new)
            rows = torch.arange(self.batch, device=idx.device).unsqueeze(1)
            # ``(B, width, H_kv, D)`` view: advanced indexing on the token axis
            # needs it in front of the head axis, and permute is a view.
            buf_k.permute(0, 2, 1, 3)[rows, idx] = k_new.permute(0, 2, 1, 3)
            buf_v.permute(0, 2, 1, 3)[rows, idx] = v_new.permute(0, 2, 1, 3)
        if trim:
            return buf_k[:, :, :extent], buf_v[:, :, :extent], None
        return buf_k, buf_v, extent

    def _buffer(
        self, layer: int, like: torch.Tensor, needed: int, value: bool = False
    ) -> torch.Tensor:
        """This layer's capacity buffer, allocated or grown to hold ``needed``."""
        store = self.v if value else self.k
        buf = store[layer]
        if buf is not None and buf.size(2) >= needed:
            return buf
        width = max(int(self.cap or 0), needed)
        if buf is not None:
            # Growth rounds up so an overflowing generation does not reallocate
            # once per token; the strategies size ``cap`` so this is rare.
            width = max(width, -(-needed // _GROW_ROUND) * _GROW_ROUND)
        batch, h_kv, _, head_dim = like.shape
        fresh = like.new_zeros(batch, h_kv, width, head_dim)
        if buf is not None:
            fresh[:, :, : buf.size(2)] = buf
        store[layer] = fresh
        self.cap = max(int(self.cap or 0), width)
        return fresh

    def _scatter_index(self, t_new: int) -> torch.Tensor:
        """``(B, t_new)`` write positions for rows at different offsets."""
        base = self.lens.to(torch.int64).unsqueeze(1)
        return base + torch.arange(t_new, device=base.device, dtype=base.dtype).unsqueeze(0)

    def _append_legacy(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """Exact-size ``cat`` growth (no capacity buffer, so no per-row offsets)."""
        if not self.uniform:
            raise RuntimeError(
                "legacy cat-growth decoder KV cannot hold rows at different "
                "offsets; prefill with a capacity to get per-row offsets"
            )
        buf_k, buf_v = self.k[layer], self.v[layer]
        t_prev = self.lens_host[0]
        if buf_k is None or buf_v is None or not t_prev:
            k, v = k_new, v_new
        else:
            k = torch.cat([buf_k[:, :, :t_prev], k_new], dim=2)
            v = torch.cat([buf_v[:, :, :t_prev], v_new], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v, None

    def reserve(self, t_new: int) -> None:
        """Make room for ``t_new`` more keys per row, before anything writes.

        A no-op here — a capacity buffer grows on the write that needs it — and
        the paged store's page mapping.  It exists so a caller that has to do
        host work *before* a captured step (which runs no Python at all) has one
        call to make.
        """
        del t_new

    def commit(self, t_new: int) -> None:
        """Advance every row by the ``t_new`` keys the layers just wrote."""
        self._advance_rows(t_new)

    # ------------------------------------------------------------------
    # Row bookkeeping
    # ------------------------------------------------------------------

    def select(self, keep: torch.Tensor) -> "DecoderKv":
        """Rows ``keep`` (repeats allowed — that is what expands a beam grid)."""
        self._check_live()
        keep_host = keep.tolist()
        return DecoderKv(
            k=[None if t is None else t.index_select(0, keep) for t in self.k],
            v=[None if t is None else t.index_select(0, keep) for t in self.v],
            lens=self.lens.index_select(0, keep),
            lens_host=[self.lens_host[i] for i in keep_host],
            starts=None if self.starts is None else self.starts.index_select(0, keep),
            cap=self.cap,
        )

    def can_merge(self, other: "DecoderKv") -> bool:
        """Whether :meth:`merge` can put both groups' rows in one forward.

        Capacity mode on both sides (legacy ``cat`` growth has no room to hold
        rows at different offsets), the same layer count, and the same left-pad
        discipline — a family that left-pads and one that does not are not the
        same decoder.
        """
        if self.cap is None or other.cap is None:
            return False
        if self.num_layers != other.num_layers:
            return False
        if (self.starts is None) != (other.starts is None):
            return False
        return all((a is None) == (b is None) for a, b in zip(self.k, other.k))

    def merge(self, other: "DecoderKv") -> "DecoderKv":
        """Concatenate ``other``'s rows after this group's.

        Both sides are padded to a common capacity first, which is a copy of both
        caches — the price of dense storage, and the reason paged decoder KV
        makes this free (block tables concatenate; the KV never moves).

        **Consumes both operands**: each layer's source tensors are released as
        soon as the merged one holds them, so the transient peak is the merged
        cache plus *one* layer rather than the merged cache plus both sources.
        At the shapes this runs on (a 7B row's decoder KV is ~0.4 GiB) that is the
        difference between a merge and an OOM.  Neither ``self`` nor ``other`` is
        usable afterwards; the caller replaces both with the result.
        """
        self._check_live()
        other._check_live()
        if not self.can_merge(other):
            raise ValueError("decoder KV states are not mergeable (see can_merge)")
        width = max(
            max((t.size(2) for t in self.k if t is not None), default=0),
            max((t.size(2) for t in other.k if t is not None), default=0),
            int(self.cap or 0),
            int(other.cap or 0),
        )
        k = _consume_cat(self.k, other.k, width)
        v = _consume_cat(self.v, other.v, width)
        self.consumed = other.consumed = True
        starts = None
        if self.starts is not None and other.starts is not None:
            starts = torch.cat([self.starts, other.starts])
        return DecoderKv(
            k=k,
            v=v,
            lens=torch.cat([self.lens, other.lens]),
            lens_host=list(self.lens_host) + list(other.lens_host),
            starts=starts,
            cap=width,
        )


class PagedDecoderKv(_RowIndexed):
    """Row-indexed self-attention KV paged out of a shared block pool.

    Same surface as :class:`DecoderKv` — the decoders call ``append`` /
    ``commit`` / ``mask_kwargs`` / ``position_ids`` / ``select`` / ``merge`` and
    never learn which one they have — over
    :class:`~oasr.cache.decoder_kv.DecoderKVCacheManager` for the block
    bookkeeping and :class:`~oasr.cache.paged_kv.PagedKVCache` for the scatter.

    Two behaviours differ from dense storage and both are load-bearing:

    * :meth:`merge` moves no K/V at all — the block tables concatenate — where the
      dense merge copies both caches;
    * :meth:`select` **frees** the rows it drops and refuses a repeated index.
      A row's pages are its own, so expanding a beam grid with
      ``[0, 0, 1, 1, ...]`` would alias two slots onto one page and each would
      overwrite the other's K/V.  Forking pages copy-on-write is what would lift
      that; until then the strategy keeps beam search on dense storage.

    A prefill still attends its own K/V densely (``trim=True``) — the paged read
    is for the steps, which is where the cache is large and the query is one
    token.  The prompt is written into the pool on the same call.
    """

    _slot_ids = itertools.count()

    def __init__(
        self,
        manager: "DecoderKVCacheManager",
        slots: List[str],
        lens: torch.Tensor,
        lens_host: List[int],
        starts: Optional[torch.Tensor] = None,
    ) -> None:
        self._mgr = manager
        self.slots = slots
        self.lens = lens
        self.lens_host = lens_host
        self.starts = starts
        self.consumed = False
        self._block_size = int(manager.pool.config.block_size_frames)
        self._table: Optional[torch.Tensor] = None
        self._widx: Optional[torch.Tensor] = None
        self._widx_t = 0

    # ------------------------------------------------------------------
    # Construction / teardown
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        manager: "DecoderKVCacheManager",
        batch_size: int,
        device: torch.device,
        *,
        prefill_len: int,
        capacity: int,
        starts: Optional[torch.Tensor] = None,
    ) -> "PagedDecoderKv":
        """Reserve one slot per row, with pages for the prompt already mapped.

        ``capacity`` is the row's whole position budget (prompt + generation
        cap); only the prompt's pages are allocated now, the rest as the rows
        fill them — which is the difference from a capacity buffer.  The pool is
        checked against every live slot's *ceiling* first, so a batch that could
        run the pool out several seconds into generation is refused here instead;
        and anything already registered is rolled back, so a refused batch leaks
        no pages.
        """
        from .decoder_kv import DecoderKvExhausted

        slots: List[str] = []
        max_new = max(0, int(capacity) - int(prefill_len))
        try:
            for _ in range(batch_size):
                if not manager.can_admit(max_new, int(prefill_len)):
                    raise DecoderKvExhausted(
                        f"decoder KV pool cannot admit {batch_size} more row(s) at "
                        f"{int(capacity)} positions each: "
                        f"{manager.pool.num_free_blocks}/{manager.pool.num_blocks} "
                        "blocks free and the live slots' ceilings claim the rest. "
                        "Lower max_decode_slots or max_new_tokens, or raise "
                        "decode_kv_budget_gib."
                    )
                slot = f"ar-{next(cls._slot_ids)}"
                manager.create(slot, max_new_tokens=max_new, prefill_len=int(prefill_len))
                slots.append(slot)
        except Exception:
            for slot in slots:
                manager.free(slot)
            raise
        # ``create`` accounts the prompt's pages; the rows have not written yet,
        # so their length is still zero until the prefill commits.
        return cls(
            manager,
            slots,
            torch.zeros(batch_size, dtype=torch.int32, device=device),
            [0] * batch_size,
            None if starts is None else starts.to(torch.int32),
        )

    def free(self) -> None:
        """Release every page this group holds.  Idempotent."""
        for slot in self.slots:
            self._mgr.free(slot)
        self.slots = []

    def __del__(self) -> None:
        """Backstop for a group dropped without a ``select`` or a ``free``.

        The strategy frees at both points where a group ends, so this should
        never have anything to do — but a pool is a shared resource with no
        eviction, and the failure mode of one missed path is not a leak that
        shows up in a profile, it is ``BlockPool exhausted`` on an unrelated
        request an hour later.  Guarded because a finalizer can run at
        interpreter shutdown with the pool already torn down.
        """
        try:
            self.free()
        except Exception:  # pragma: no cover - shutdown ordering
            pass

    @property
    def num_layers(self) -> int:
        return int(self._mgr.pool.config.num_layers)

    @property
    def manager(self) -> "DecoderKVCacheManager":
        return self._mgr

    def block_table(self) -> torch.Tensor:
        """``(B, max_blocks)`` int32 table, rebuilt only when the pages change."""
        if self._table is None:
            self._table = self._mgr.block_tables(self.slots, device=self.lens.device)
        return self._table

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def mask_kwargs(self, t_new: int, *, trimmed: bool = False) -> Dict[str, Any]:
        """``Attention`` kwargs after ``t_new`` more keys.

        ``trimmed`` is the prefill's dense read, so it gets the same window pair
        (or nothing) a dense cache would; otherwise the block table travels with
        the window and turns the call into a paged one.

        Maps this forward's pages **first**.  Both decoders build the mask once
        before the layer loop, so a table built before the growth would be one
        page short of the ``kv_lens`` beside it exactly on the steps that cross a
        page boundary — the kernel then indexes a row of the table that is not
        there, which is an illegal access rather than a wrong answer.
        """
        self._grow_to([n + t_new for n in self.lens_host])
        if trimmed:
            return {} if self.starts is None else self._window(t_new)
        kwargs = self._window(t_new)
        kwargs["block_table"] = self.block_table()
        return kwargs

    def append(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, *, trim: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """Scatter one layer's new K/V into the pool at each row's own offset.

        Returns the **pool views** — the paged read addresses them through the
        block table, so unlike dense storage there is nothing per-group to hand
        back.  ``trim`` returns the fresh K/V instead, for the prefill, which
        attends its own prompt densely (see the class docstring).

        One write path, not a uniform fast path beside a scatter: the flat index
        is computed once per *step* rather than per layer, so the scatter is
        already two ``index_put_`` per layer, which is what the row-slice
        alternative costs anyway — and one path is one path to keep correct as
        rows diverge.
        """
        self._check_live()
        if trim and self.max_len:
            raise RuntimeError("trim is for a prefill: every row must still be empty")
        # Pages for this step are mapped before the first layer writes, not at
        # ``commit`` after the last one: the write itself needs the page.
        # Idempotent, so layers 1..n-1 cost one comparison each.
        t_new = k_new.size(2)
        self._grow_to([n + t_new for n in self.lens_host])
        k_pool, v_pool = self._mgr.kv_view(layer)
        # Frame-major is the pool's layout, and for K/V that came out of a head
        # split it is also the original one, so this permute is a view.
        index = self._write_index(t_new)
        h_kv, head_dim = k_pool.size(2), k_pool.size(3)
        k_pool.view(-1, h_kv, head_dim)[index] = k_new.permute(0, 2, 1, 3).reshape(
            -1, h_kv, head_dim
        )
        v_pool.view(-1, h_kv, head_dim)[index] = v_new.permute(0, 2, 1, 3).reshape(
            -1, h_kv, head_dim
        )
        if trim:
            return k_new, v_new, None
        return k_pool, v_pool, None

    def _write_index(self, t_new: int) -> torch.Tensor:
        """Flat pool slots this step writes, computed once for every layer.

        The positions depend on the block table and the row offsets, neither of
        which moves within a step, so recomputing them per layer would be four
        extra kernels times the decoder's depth.
        """
        if self._widx is None or self._widx_t != t_new:
            self._widx = flat_write_index(
                self.block_table(), self.lens.to(torch.int64), t_new, self._block_size
            )
            self._widx_t = t_new
        return self._widx

    def reserve(self, t_new: int) -> None:
        """Map the pages this forward will write into."""
        self._grow_to([n + t_new for n in self.lens_host])

    def commit(self, t_new: int) -> None:
        """Advance every row past the keys the layers just wrote."""
        self._grow_to([n + t_new for n in self.lens_host])  # no-op: append mapped them
        self._advance_rows(t_new)
        self._widx = None

    def _grow_to(self, target: Sequence[int]) -> None:
        """Make sure every slot has pages mapped up to its target length.

        The prompt's pages were mapped at :meth:`create`, so this is a no-op for
        the prefill and one ``append_step`` per row for a step — which is where
        pages are actually allocated, one at a time, as a row fills the one it
        has.  That is the whole difference from reserving ``prompt + max_new``.
        """
        need = [w - self._mgr.seqlen(s) for s, w in zip(self.slots, target)]
        if not any(need):
            return
        before = self._pages()
        if all(n == 1 for n in need):
            self._mgr.append_step(self.slots)
        else:
            for slot, n in zip(self.slots, need):
                for _ in range(max(0, n)):
                    self._mgr.append_step([slot])
        if self._pages() != before:
            self._table = None  # a row grew a page; the table it indexes changed

    def _pages(self) -> int:
        return sum(self._mgr.num_blocks(s) for s in self.slots)

    # ------------------------------------------------------------------
    # Row bookkeeping
    # ------------------------------------------------------------------

    def select(self, keep: torch.Tensor) -> "PagedDecoderKv":
        """Keep rows ``keep``, freeing the pages of every row not in it.

        Destructive, unlike the dense ``index_select``: the dropped rows' pages
        go back to the pool immediately, which is the point of paging them.  The
        source is marked consumed for the same reason ``merge`` marks its
        operands.
        """
        self._check_live()
        order = [int(i) for i in keep.tolist()]
        if len(set(order)) != len(order):
            raise RuntimeError(
                "paged decoder KV cannot expand or reorder a beam grid: a "
                "repeated row index would alias two slots onto one page. Beam "
                "search runs on dense storage."
            )
        kept = set(order)
        for row, slot in enumerate(self.slots):
            if row not in kept:
                self._mgr.free(slot)
        out = PagedDecoderKv(
            self._mgr,
            [self.slots[i] for i in order],
            self.lens.index_select(0, keep),
            [self.lens_host[i] for i in order],
            None if self.starts is None else self.starts.index_select(0, keep),
        )
        self.slots = []
        self.consumed = True
        return out

    def can_merge(self, other: "PagedDecoderKv") -> bool:
        """Same pool, same left-pad discipline, neither already consumed."""
        if not isinstance(other, PagedDecoderKv) or self.consumed or other.consumed:
            return False
        if self._mgr is not other._mgr:
            return False
        return (self.starts is None) == (other.starts is None)

    def merge(self, other: "PagedDecoderKv") -> "PagedDecoderKv":
        """Concatenate ``other``'s rows after this group's — no K/V moves.

        Consumes both operands, like the dense merge, but only so the freed
        slots cannot be released twice: the pages themselves are handed straight
        over.
        """
        self._check_live()
        other._check_live()
        if not self.can_merge(other):
            raise ValueError("decoder KV states are not mergeable (see can_merge)")
        starts = None
        if self.starts is not None and other.starts is not None:
            starts = torch.cat([self.starts, other.starts])
        merged = PagedDecoderKv(
            self._mgr,
            list(self.slots) + list(other.slots),
            torch.cat([self.lens, other.lens]),
            list(self.lens_host) + list(other.lens_host),
            starts,
        )
        self.slots, other.slots = [], []
        self.consumed = other.consumed = True
        return merged


def build_kv(
    num_layers: int,
    batch_size: int,
    device: torch.device,
    *,
    prefill_len: int,
    cap: Optional[int],
    manager: Optional["DecoderKVCacheManager"] = None,
    starts: Optional[torch.Tensor] = None,
) -> Any:
    """Storage for one freshly prefilled group — paged when a pool is given.

    One call site per decoder so the choice between the two is made in one place
    and neither decoder grows a branch on it.
    """
    if manager is None:
        return DecoderKv.empty(num_layers, batch_size, device, starts=starts, cap=cap)
    if cap is None:
        raise ValueError(
            "paged decoder KV needs a capacity: the pool reserves each row's "
            "position budget at admission so growth cannot fail mid-generation"
        )
    return PagedDecoderKv.create(
        manager,
        batch_size,
        device,
        prefill_len=prefill_len,
        capacity=cap,
        starts=starts,
    )


def _consume_cat(
    a: List[Optional[torch.Tensor]], b: List[Optional[torch.Tensor]], width: int
) -> List[Optional[torch.Tensor]]:
    """Per-layer batch-concatenation that releases each source as it goes.

    One allocation per layer, written from both sources — rather than padding
    each to ``width`` and then concatenating, which would hold five tensors at
    once where this holds three.
    """
    out: List[Optional[torch.Tensor]] = []
    for i in range(len(a)):
        left, right = a[i], b[i]
        a[i] = b[i] = None  # the only other references, so the pair frees below
        if left is None or right is None:
            out.append(None)
            continue
        n_left = left.size(0)
        merged = left.new_zeros(n_left + right.size(0), left.size(1), width, left.size(3))
        merged[:n_left, :, : left.size(2)] = left
        merged[n_left:, :, : right.size(2)] = right
        out.append(merged)
        del left, right, merged
    return out


def consume_cat_rows(a: List[torch.Tensor], b: List[torch.Tensor]) -> List[torch.Tensor]:
    """Batch-concatenate two per-layer tensor lists, releasing sources as it goes.

    For the fixed side-caches a decoder carries next to its self-attention KV
    (an AED's cross-attention K/V), which merge by plain concatenation because
    their token axis does not grow.
    """
    out: List[torch.Tensor] = []
    for i in range(len(a)):
        left, right = a[i], b[i]
        a[i] = b[i] = None  # type: ignore[call-overload]
        out.append(torch.cat([left, right], dim=0))
        del left, right
    return out
