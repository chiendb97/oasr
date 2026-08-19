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
``(rows, block-table width bucket)``.  Rows are exact: a decoder step is
weight-read bound so padding a batch up to a bucket would cost nearly a full
step, and the reachable row counts are bounded by ``max_decode_slots`` anyway.
The width is bucketed because it grows by one page every ``block_size`` tokens
and would otherwise key a capture per page; the real table is copied into the
bucket's buffer and the surplus columns point at page 0, which the kernel loads
and then gives zero softmax weight — every column past ``cache_seqlens`` is
masked, and a pool page is finite, which is the one thing masked columns must be.

What is *not* captured
----------------------
* **Prefill.** Its shapes follow the prompt, so it would key a capture per
  prompt length, and it runs once per batch against a step's many.
* **A decoder with a per-group side cache.** An AED's cross-attention K/V is
  allocated per prefill and is far too large to copy into a static buffer per
  step, so ``aed`` is not capturable without pooling that as well.  Declared by
  :attr:`~oasr.models.decoders.base.BaseDecoder.supports_step_graphs` rather
  than discovered — a decoder that quietly captured a stale pointer would return
  a plausible transcript of the previous batch's audio.
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


class DecoderStepGraphCache:
    """Lazily captured decoder steps, keyed by ``(rows, width bucket)``.

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
        self._pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self._captured: Dict[Tuple[int, int], _Captured] = {}
        self._refused = False
        #: Shapes whose capture failed.  A capture costs a warm-up forward, so
        #: retrying one that already failed would pay that on *every* step of
        #: that shape and still run eager — the slowest possible outcome.
        self._failed: Set[Tuple[int, int]] = set()
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

    def capturable(self, kv: Any) -> bool:
        """Whether this KV state's storage is one a captured step can read."""
        return (
            isinstance(kv, PagedDecoderKv)
            and kv.manager is self._mgr
            and not kv.consumed
            and bool(getattr(self._decoder, "supports_step_graphs", False))
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, tokens: torch.Tensor, kv: PagedDecoderKv) -> Optional[torch.Tensor]:
        """One decoder step through a captured graph.

        Returns the ``(B, V)`` logits, or ``None`` when this state or shape is
        not capturable and the caller should step eagerly.  On return ``kv`` has
        **not** advanced — the caller commits, the same as it would after an
        eager step.
        """
        if self._disabled or not self.capturable(kv):
            return None
        kv.reserve(1)  # host-side page mapping; the graph runs no Python
        table = kv.block_table()
        key = (kv.batch, self._bucket(table.size(1)))
        state = self._captured.get(key)
        if state is None:
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
            state = self._capture(key, tokens, kv)
            if state is None:
                return None
            self._captured[key] = state

        state.tokens.copy_(tokens)
        state.lens.copy_(kv.lens)
        if state.starts is not None and kv.starts is not None:
            state.starts.copy_(kv.starts)
        self._fill_table(state.table, table)
        state.graph.replay()
        # The buffer is live only until the next replay of this key.
        return state.logits.clone()

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

    def _capture(
        self, key: Tuple[int, int], tokens: torch.Tensor, kv: PagedDecoderKv
    ) -> Optional[_Captured]:
        rows, width = key
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

        graph_kv = _GraphKv(self._mgr, lens_buf, starts_buf, table_buf, kv.lens_host)

        def _run() -> torch.Tensor:
            logits, _ = self._decoder.step(tokens_buf, {"kv": graph_kv})
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
                with tvm_ffi.use_torch_stream(torch.cuda.graph(graph, pool=self._pool)):
                    logits_buf = _run()
        except torch.cuda.OutOfMemoryError as exc:
            # Treat capture OOM as process-wide and stop retrying costly warmups.
            self._disabled = True
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
            logger.warning(
                "decoder-step graph capture failed for rows=%d width=%d (%s); "
                "this shape runs eager",
                rows,
                width,
                exc,
            )
            return None
        logger.info("captured decoder step: rows=%d block-table width=%d", rows, width)
        return _Captured(
            graph=graph,
            tokens=tokens_buf,
            lens=lens_buf,
            table=table_buf,
            logits=logits_buf,
            starts=starts_buf,
        )
