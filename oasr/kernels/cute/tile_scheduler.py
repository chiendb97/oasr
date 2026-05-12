# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tile schedulers for the OASR FMHA forward kernel.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/tile_scheduler.py``.
The implementation is OASR-native and only covers two single-tile dispatchers:

* :class:`SingleTileScheduler` -- dense / paged path. ``gridDim ==
  (num_m_blocks, B, H)``; the scheduler returns ``(m_block, batch, q_head)``
  directly from ``block_idx()``.
* :class:`SingleTileVarlenScheduler` -- varlen path. ``gridDim == (num_tiles,
  H, 1)`` where ``num_tiles = sum(ceil(seqlen_q[b] / M_BLOCK))`` is a host-
  computed total. The kernel decodes ``(batch, m_block)`` from the flat
  tile id via a linear walk of ``cu_seqlens_q``.

Persistent / LPT scheduling is deliberately omitted (single-tile is enough
for the SM80 forward; persistent can come later).
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

import cutlass
import cutlass.cute as cute


class TileSchedulerArguments:
    """Host-side args bundled to make a scheduler."""

    def __init__(
        self,
        *,
        num_m_blocks: int,
        num_batches: int,
        num_heads: int,
        m_block_size: int,
        varlen: bool,
        num_varlen_tiles: int = 0,
    ):
        self.num_m_blocks = num_m_blocks
        self.num_batches = num_batches
        self.num_heads = num_heads
        self.m_block_size = m_block_size
        self.varlen = varlen
        self.num_varlen_tiles = num_varlen_tiles


class SingleTileScheduler:
    """Trivial scheduler for dense / paged: blockIdx -> (m, b, h)."""

    @staticmethod
    def grid_dim(args: TileSchedulerArguments):
        if args.varlen:
            raise ValueError(
                "SingleTileScheduler is for dense/paged; use "
                "SingleTileVarlenScheduler for varlen"
            )
        return (args.num_m_blocks, args.num_batches, args.num_heads)

    @staticmethod
    @cute.jit
    def get_work_tile():
        """Return ``(m_block, batch, q_head)`` from block_idx()."""
        m_block, batch, q_head = cute.arch.block_idx()
        return m_block, batch, q_head


class SingleTileVarlenScheduler:
    """Decode ``(batch, m_block)`` from a flat tile id for varlen.

    The host launches ``gridDim = (num_varlen_tiles, H, 1)``; the kernel
    reads ``tile_id = blockIdx.x`` and walks ``cu_seqlens_q`` to find which
    batch this tile belongs to.

    The walk is linear in B (matches FA's SM80 path); for typical ASR
    batch sizes (B <= 32) this is fine, and avoids the more complex
    warp-prefix-sum lookup that pays off only at much larger B.
    """

    @staticmethod
    def grid_dim(args: TileSchedulerArguments):
        if not args.varlen:
            raise ValueError(
                "SingleTileVarlenScheduler is for varlen; use "
                "SingleTileScheduler for dense/paged"
            )
        return (args.num_varlen_tiles, args.num_heads, 1)

    @staticmethod
    @cute.jit
    def get_work_tile(
        cu_seqlens_q: cute.Tensor,
        m_block_size: cutlass.Constexpr,
        num_batches: cutlass.Constexpr,
    ):
        """Return ``(m_block_within_batch, batch, q_head)``.

        Iterates per-batch tile counts via integer ceil-div over
        ``cu_seqlens_q[b+1] - cu_seqlens_q[b]``. ``-1`` for batch
        signals an out-of-range tile (returned ``num_batches``); the
        kernel should early-exit on this.
        """
        tile_id, q_head, _ = cute.arch.block_idx()
        accum = cutlass.Int32(0)
        batch_out = cutlass.Int32(num_batches)  # sentinel
        m_block_out = cutlass.Int32(0)
        for b in cutlass.range_constexpr(num_batches):
            seqlen_b = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
            n_tiles_b = (seqlen_b + (m_block_size - 1)) // m_block_size
            if tile_id >= accum and tile_id < accum + n_tiles_b:
                batch_out = cutlass.Int32(b)
                m_block_out = tile_id - accum
            accum = accum + n_tiles_b
        return m_block_out, batch_out, q_head


__all__ = [
    "TileSchedulerArguments",
    "SingleTileScheduler",
    "SingleTileVarlenScheduler",
]
