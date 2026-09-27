# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph capture of one autoregressive decoder step.

A decode step is a fixed sequence of small kernels over fixed shapes — the same
launch-bound shape the streaming encoder's graph cache (:mod:`graph_cache`)
exists for, one layer of the engine up.  Replaying it collapses a 28-layer LM's
~200 launches into one.

What makes it capturable at all is **paged** decoder KV.  A graph records the
addresses it reads, so every tensor the step touches has to live at a stable one
across replays; a capacity buffer is allocated per decode group and moves with
every prefill, while a block pool is allocated once for the process and never
moves.  Everything that *does* change per step — the tokens, each row's length
and left-pad, the block table — is small, and is copied into pre-allocated
buffers the graph was captured reading.

Shape key
---------
``(rows, block-table width bucket, cross block-table width bucket)``.  Rows are
exact: a decoder step is weight-read bound so padding a batch up to a bucket
would cost nearly a full step, and the reachable row counts are bounded by
``max_decode_slots`` anyway.  The width is bucketed because it grows by one page
every ``block_size`` tokens and would otherwise key a capture per page; the real
table is copied into the bucket's buffer and the surplus columns point at page 0,
which the kernel loads and then gives zero softmax weight — every column past
``cache_seqlens`` is masked, and a pool page is finite, which is the one thing
masked columns must be.  The third member is ``0`` for a decoder-only LM.

The cross-attention side cache
------------------------------
An AED also reads the encoder output's K/V every step.  Dense, that is a
per-group tensor — allocated per prefill, far too large to copy into a static
buffer per step — and a state carrying one is refused (:meth:`capturable`).
Paged (``state["cross"]``, a fixed-extent region of the *same* pool), it is only
a block table and a key-length vector, copied into static buffers like the
self-attention's, so the step captures whole.

What is *not* captured
----------------------
* **Prefill.** Its shapes follow the prompt, so it would key a capture per
  prompt length, and it runs once per batch against a step's many.
* **Any state with a component outside the pool** (a dense ``cross_k`` list).
  Declared by :attr:`~oasr.models.decoders.base.BaseDecoder.supports_step_graphs`
  and checked per state rather than discovered — a decoder that quietly
  captured a stale pointer would return a plausible transcript of the previous
  batch's audio.
* **Beam search**, which does not page its KV in the first place.

Correctness notes
-----------------
* Page mapping is host work and cannot be inside the graph, so :meth:`step`
  calls ``kv.reserve(1)`` first — the page a row is about to cross into must
  exist before the recorded write lands.
* The returned logits are the graph's **output buffer**, live only until the
  next replay of the same key.  :meth:`step` clones, because two decode groups
  can hit one key in a single tick and the caller keeps ``last_logits`` across
  ticks (the same aliasing rule the encoder cache documents).
* The captured path and the eager path must pick the same kernels, per the
  repo's rule against branching dispatch on capture state.  Nothing here
  branches: the routing in :class:`oasr.layers.Attention` is a function of
  shapes and dtypes, which are identical by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import torch
import tvm_ffi

from oasr.cache import PagedDecoderKv

from .capture_recovery import recover_from_failed_capture

logger = logging.getLogger(__name__)

__all__ = ["DecoderStepGraphCache"]


class _GraphKv(PagedDecoderKv):
    """A paged KV view whose row vectors are the graph's static buffers.

    Two overrides, both saying "the host half already happened": page mapping is
    done by :meth:`DecoderStepGraphCache.step` before the replay, and the row
    advance by the caller after it.  Without them the captured ``step`` would try
    to allocate pages during capture and would rebind ``lens`` away from the
    buffer the graph reads.
    """

    def __init__(
        self,
        manager: Any,
        lens: torch.Tensor,
        starts: Optional[torch.Tensor],
        table: torch.Tensor,
        lens_host: List[int],
    ) -> None:
        super().__init__(manager, [], lens, list(lens_host), starts)
        self._table = table

    def block_table(self) -> torch.Tensor:
        return self._table  # type: ignore[return-value]

    def _grow_to(self, target) -> None:  # noqa: ANN001 - matches the base signature
        del target

    def commit(self, t_new: int) -> None:
        del t_new
        self._widx = None

    def free(self) -> None:
        return  # owns no slots; the real state does

    def __del__(self) -> None:
        return


@dataclass
class _Captured:
    """One captured graph plus the buffers it was captured reading/writing."""

    graph: "torch.cuda.CUDAGraph"
    tokens: torch.Tensor
    lens: torch.Tensor
    table: torch.Tensor
    logits: torch.Tensor
    starts: Optional[torch.Tensor] = None
    #: The cross-attention side cache's key lengths and block table, when the
    #: decoder pages one (an AED); ``None`` for a decoder-only LM.
    cross_lens: Optional[torch.Tensor] = None
    cross_table: Optional[torch.Tensor] = None


_Key = Tuple[int, int, int]


class DecoderStepGraphCache:
    """Lazily captured decoder steps, keyed by ``(rows, width bucket, cross bucket)``.

    Parameters
    ----------
    decoder :
        The AR decoder surface; must declare ``supports_step_graphs``.
    manager :
        The paged decoder-KV pool the captured step reads.  A state on any other
        pool is not capturable here, because the pool's addresses are what the
        graph baked in.
    width_pages :
        Block-table bucket granularity, in pages.  Larger buckets mean fewer
        captures and more masked columns per step.
    max_captures :
        Ceiling on distinct shapes; past it :meth:`step` returns ``None`` and the
        caller runs eager, rather than growing graph memory without bound.

    Capture is best-effort and each attempt costs a warm-up forward, so a failure
    is remembered rather than retried: a shape that raised is never attempted
    again, and an *out-of-memory* stops capture for the whole cache, because that
    is a fact about the process rather than about the shape.
    """

    def __init__(
        self,
        decoder: Any,
        manager: Any,
        *,
        width_pages: int = 8,
        max_captures: int = 64,
        pool: Any = None,
    ) -> None:
        self._decoder = decoder
        self._mgr = manager
        self._width_pages = max(1, int(width_pages))
        self._max_captures = int(max_captures)
        self._pool: Optional[Tuple[int, int]] = (
            pool if pool is not None else torch.cuda.graph_pool_handle()
        )
        self._captured: Dict[_Key, _Captured] = {}
        self._refused = False
        #: Shapes whose capture failed.  A capture costs a warm-up forward, so
        #: retrying one that already failed would pay that on *every* step of
        #: that shape and still run eager — the slowest possible outcome.
        self._failed: Set[_Key] = set()
        #: Set when a capture ran out of memory.  That is a property of the
        #: process, not of the shape: the next shape is no more likely to fit,
        #: and each attempt burns a forward.  Stop trying and run eager.
        self._disabled = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_captured(self) -> int:
        return len(self._captured)

    def capturable(self, state: Any) -> bool:
        """Whether this decode state's storage is one a captured step can read.

        ``state`` is the decoder's state dict (a bare :class:`PagedDecoderKv` is
        accepted as ``{"kv": it}``).  Every component a step reads must be paged
        on this cache's pool — the self-attention KV and, for an AED, the
        cross-attention side cache; anything else in the state holds per-group
        addresses a graph would bake in.
        """
        return self._parts(state) is not None

    def _parts(self, state: Any) -> Optional[Tuple[PagedDecoderKv, Optional[PagedDecoderKv]]]:
        """``(kv, cross)`` for a capturable state, else ``None``."""
        if not bool(getattr(self._decoder, "supports_step_graphs", False)):
            return None
        if isinstance(state, PagedDecoderKv):
            state = {"kv": state}
        if not isinstance(state, dict) or set(state) - {"kv", "cross"}:
            return None
        kv, cross = state.get("kv"), state.get("cross")
        for part in (kv, cross):
            if part is None:
                continue
            if not isinstance(part, PagedDecoderKv) or part.manager is not self._mgr:
                return None
            if part.consumed:
                return None
        if kv is None:
            return None
        return kv, cross

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, tokens: torch.Tensor, state: Any) -> Optional[torch.Tensor]:
        """One decoder step through a captured graph.

        Returns the ``(B, V)`` logits, or ``None`` when this state or shape is
        not capturable and the caller should step eagerly.  On return the
        self-attention KV has **not** advanced — the caller commits, the same as
        it would after an eager step.  ``state`` is the decoder's state dict, or
        a bare self-attention :class:`PagedDecoderKv`.
        """
        parts = None if self._disabled else self._parts(state)
        if parts is None:
            return None
        kv, cross = parts
        kv.reserve(1)  # host-side page mapping; the graph runs no Python
        table = kv.block_table()
        cross_table = cross.block_table() if cross is not None else None
        key = (
            kv.batch,
            self._bucket(table.size(1)),
            0 if cross_table is None else self._bucket(cross_table.size(1)),
        )
        captured = self._captured.get(key)
        if captured is None:
            if key in self._failed:
                return None
            if len(self._captured) >= self._max_captures:
                if not self._refused:
                    self._refused = True
                    logger.info(
                        "decoder-step graph cache full (%d shapes); further shapes " "run eager",
                        self._max_captures,
                    )
                return None
            captured = self._capture(key, tokens, kv, cross)
            if captured is None:
                return None
            self._captured[key] = captured

        captured.tokens.copy_(tokens)
        captured.lens.copy_(kv.lens)
        if captured.starts is not None and kv.starts is not None:
            captured.starts.copy_(kv.starts)
        self._fill_table(captured.table, table)
        if cross is not None and cross_table is not None:
            assert captured.cross_lens is not None and captured.cross_table is not None
            captured.cross_lens.copy_(cross.lens)
            self._fill_table(captured.cross_table, cross_table)
        captured.graph.replay()
        # The buffer is live only until the next replay of this key.
        return captured.logits.clone()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _bucket(self, width: int) -> int:
        return -(-max(1, width) // self._width_pages) * self._width_pages

    @staticmethod
    def _fill_table(dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy the live table in and point the surplus columns at page 0.

        Those columns sit past every row's ``cache_seqlens``, so the kernel loads
        them and then gives them zero softmax weight; page 0 is a real pool page
        and therefore finite, which is what stops a masked column poisoning the
        row through ``P @ V``.
        """
        dst.zero_()
        dst[:, : src.size(1)].copy_(src)

    def release(self) -> None:
        """Return the graph pool's VRAM.  Idempotent.

        ``CUDAGraph.reset()`` is the only thing that frees a capture's private
        memory pool; dropping the object and calling ``empty_cache()`` leaves it
        held for the life of the process.  See
        :meth:`oasr.engine.offline_graph.GraphedOfflineForward.release`.
        """
        for state in self._captured.values():
            try:
                state.graph.reset()
            except Exception:  # pragma: no cover - teardown must not raise
                pass
        self._captured.clear()
        self._failed.clear()

    def _capture(
        self,
        key: _Key,
        tokens: torch.Tensor,
        kv: PagedDecoderKv,
        cross: Optional[PagedDecoderKv] = None,
    ) -> Optional[_Captured]:
        rows, width, cross_width = key
        device = kv.lens.device
        tokens_buf = torch.empty_like(tokens)
        tokens_buf.copy_(tokens)
        lens_buf = torch.empty(rows, dtype=torch.int32, device=device)
        lens_buf.copy_(kv.lens)
        starts_buf: Optional[torch.Tensor] = None
        if kv.starts is not None:
            starts_buf = torch.empty(rows, dtype=torch.int32, device=device)
            starts_buf.copy_(kv.starts)
        table_buf = torch.zeros(rows, width, dtype=torch.int32, device=device)
        self._fill_table(table_buf, kv.block_table())

        graph_state: Dict[str, Any] = {
            "kv": _GraphKv(self._mgr, lens_buf, starts_buf, table_buf, kv.lens_host)
        }
        cross_lens_buf: Optional[torch.Tensor] = None
        cross_table_buf: Optional[torch.Tensor] = None
        if cross is not None:
            # The side cache is only read by a step, so its twin needs nothing
            # but the two vectors the paged read takes, at static addresses.
            cross_lens_buf = torch.empty(rows, dtype=torch.int32, device=device)
            cross_lens_buf.copy_(cross.lens)
            cross_table_buf = torch.zeros(rows, cross_width, dtype=torch.int32, device=device)
            self._fill_table(cross_table_buf, cross.block_table())
            graph_state["cross"] = _GraphKv(
                self._mgr, cross_lens_buf, None, cross_table_buf, cross.lens_host
            )

        def _run() -> torch.Tensor:
            logits, _ = self._decoder.step(tokens_buf, graph_state)
            return cast(torch.Tensor, logits)

        try:
            # Warm up before capture so libraries allocate workspaces. The first
            # replay overwrites the temporary next-position KV writes.
            with torch.no_grad():
                _run()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            # ``tvm_ffi.use_torch_stream`` is what gets a TVM-FFI kernel launch
            # recorded into the graph instead of escaping to the default stream.
            with torch.no_grad():
                # torch types the pool handle as an opaque ``_POOL_HANDLE``;
                # the engine passes the ``(int, int)`` tuple it actually is.
                ctx = torch.cuda.graph(graph, pool=self._pool)  # type: ignore[arg-type]
                with tvm_ffi.use_torch_stream(ctx):
                    logits_buf = _run()
        except torch.cuda.OutOfMemoryError as exc:
            # Treat capture OOM as process-wide and stop retrying costly warmups.
            self._disabled = True
            recover_from_failed_capture(device, self._pool)
            self._pool = None
            torch.cuda.empty_cache()
            logger.warning(
                "decoder-step graph capture ran out of memory at rows=%d width=%d "
                "(%s); step graphs are off for this engine and steps run eager",
                rows,
                width,
                exc,
            )
            return None
        except Exception as exc:  # pragma: no cover - capture is best-effort
            self._failed.add(key)
            recover_from_failed_capture(device, self._pool)
            self._pool = torch.cuda.graph_pool_handle()
            logger.warning(
                "decoder-step graph capture failed for rows=%d width=%d (%s); "
                "this shape runs eager",
                rows,
                width,
                exc,
            )
            return None
        logger.info(
            "captured decoder step: rows=%d block-table width=%d cross width=%d",
            rows,
            width,
            cross_width,
        )
        return _Captured(
            graph=graph,
            tokens=tokens_buf,
            lens=lens_buf,
            table=table_buf,
            logits=logits_buf,
            starts=starts_buf,
            cross_lens=cross_lens_buf,
            cross_table=cross_table_buf,
        )
