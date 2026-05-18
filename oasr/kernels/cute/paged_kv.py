# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paged-KV gmem loader for the OASR FMHA forward kernel.

Notes
-----
Structurally modelled on FlashAttention's PagedKVManager. The
implementation is OASR-native and uses the simpler "block-aligned n_block"
constraint already enforced by the existing OASR kernel: each n-tile of
``N_BLOCK`` rows reads ``blocks_per_n_tile = N_BLOCK // block_size``
consecutive logical block-IDs from the block table.

That constraint means ``divmod(row, block_size)`` collapses to a fixed
``(i, _)`` pair per cp.async, so we don't need a full per-row fastdiv --
the kernel just walks ``i = 0..blocks_per_n_tile`` and looks up
``mBlockTable[b, n_logical_base + i]``.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

import cutlass
import cutlass.cute as cute


class PagedKVManager:
    """Constexpr struct + per-tile loader for paged K/V.

    Fields:

    * ``block_size`` -- pool block size in rows. Constexpr.
    * ``blocks_per_n_tile`` -- ``N_BLOCK // block_size``. Constexpr.
    """

    def __init__(
        self,
        *,
        block_size: cutlass.Constexpr,
        n_block_size: cutlass.Constexpr,
        head_dim_padded: cutlass.Constexpr,
    ):
        if n_block_size % block_size != 0:
            raise ValueError(
                f"PagedKVManager requires n_block_size ({n_block_size}) "
                f"divisible by block_size ({block_size})"
            )
        self.block_size = block_size
        self.n_block_size = n_block_size
        self.head_dim_padded = head_dim_padded
        self.blocks_per_n_tile = n_block_size // block_size

    @cute.jit
    def load_tile(
        self,
        mKV: cute.Tensor,
        sKV: cute.Tensor,
        mBlockTable: cute.Tensor,
        batch_idx: cutlass.Int32,
        kv_head: cutlass.Int32,
        n_block: cutlass.Int32,
        gmem_thr_copy: cute.TiledCopy,
        gmem_tiled_copy: cute.TiledCopy,
    ):
        """Gather one N-tile of K (or V) from the paged pool into smem.

        ``mKV`` is ``(num_blocks, block_size, H_kv, D)`` (the per-layer pool
        view). ``sKV`` is the destination smem with ``(N_BLOCK, D)`` shape.
        We slice the smem into ``blocks_per_n_tile`` sub-tiles and issue
        one cp.async per logical block.
        """
        n_logical_base = n_block * self.blocks_per_n_tile
        for i in cutlass.range_constexpr(self.blocks_per_n_tile):
            phys = mBlockTable[batch_idx, n_logical_base + i]
            gKV_block = cute.local_tile(
                mKV[phys, None, kv_head, None],
                (self.block_size, self.head_dim_padded),
                (0, 0),
            )
            sKV_sub = cute.local_tile(
                sKV,
                (self.block_size, self.head_dim_padded),
                (i, 0),
            )
            tKVgKV = gmem_thr_copy.partition_S(gKV_block)
            tKVsKV = gmem_thr_copy.partition_D(sKV_sub)
            cute.copy(gmem_tiled_copy, tKVgKV, tKVsKV)


__all__ = ["PagedKVManager"]
