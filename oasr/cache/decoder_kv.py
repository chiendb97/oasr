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

Cross-attention KV is *not* managed here: it has a fixed length (the encoder
output), is computed once at prefill, and lives as a dense per-request tensor
owned by the decode strategy.

The block tables / ``cache_seqlens`` this manager produces feed the same paged
FMHA path the encoder uses (``oasr.fmha`` with ``block_table``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from .block_pool import BlockPool
from .types import CacheConfig

__all__ = ["DecoderKVCacheManager"]


@dataclass
class _SlotState:
    """Bookkeeping for one active request's decoder KV."""

    block_ids: List[int] = field(default_factory=list)
    seqlen: int = 0
    max_tokens: int = 0


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
        eviction policy differ.
    """

    def __init__(self, pool: BlockPool) -> None:
        self._pool = pool
        self._block_tokens = pool.config.block_size_frames
        self._slots: Dict[str, _SlotState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pool(self) -> BlockPool:
        return self._pool

    def num_active(self) -> int:
        return len(self._slots)

    def seqlen(self, request_id: str) -> int:
        return self._slots[request_id].seqlen

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, request_id: str, max_new_tokens: int, prefill_len: int = 0) -> None:
        """Register a request and pre-allocate blocks for its prefill.

        ``prefill_len`` tokens (prompt / forced-decoder ids) are accounted
        immediately so :meth:`block_tables` covers them; the caller writes
        their K/V in one batched prefill pass.  ``max_new_tokens`` bounds the
        request's total growth (a scheduler admission input, not enforced
        per-step here).
        """
        if request_id in self._slots:
            raise ValueError(f"decoder KV slot already exists for request {request_id!r}")
        state = _SlotState(max_tokens=prefill_len + max_new_tokens)
        need = self._blocks_for(prefill_len) if prefill_len > 0 else 1
        state.block_ids = self._pool.allocate(need)
        state.seqlen = prefill_len
        with self._lock:
            self._slots[request_id] = state

    def free(self, request_id: str) -> None:
        """Release every block the request holds (finalize/abort)."""
        with self._lock:
            state = self._slots.pop(request_id, None)
        if state is not None and state.block_ids:
            self._pool.free(state.block_ids)

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
        """``(N, max_blocks)`` int32 block table over the given slots (0-padded)."""
        rows = [self._slots[rid].block_ids for rid in request_ids]
        width = max((len(r) for r in rows), default=1)
        table = torch.zeros(len(rows), width, dtype=torch.int32)
        for i, r in enumerate(rows):
            table[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
        return table.to(device) if device is not None else table

    def cache_seqlens(
        self, request_ids: Sequence[str], device: torch.device = None
    ) -> torch.Tensor:
        """``(N,)`` int32 current sequence length per slot."""
        lens = torch.tensor([self._slots[rid].seqlen for rid in request_ids], dtype=torch.int32)
        return lens.to(device) if device is not None else lens

    def kv_view(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-pool ``(K, V)`` views for one decoder layer (paged FMHA input)."""
        return self._pool.get_kv_view(layer)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _blocks_for(self, tokens: int) -> int:
        return max(1, -(-tokens // self._block_tokens))

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
    ) -> BlockPool:
        """Convenience constructor for a decoder-shaped :class:`BlockPool`."""
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
        return BlockPool(cfg)
