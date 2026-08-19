# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paged KV cache descriptor + read/write helpers.

The descriptor carries pool views and per-stream paging state for one
encoder layer. K/V scatter/gather lives on the descriptor (not on the
attention layer), so callers only need to talk to one object to write
new K/V into the pool or gather frames back out for the SDPA fallback.

The cute paged kernel reads pool views directly via ``block_table`` and
``cache_seqlens``, so :meth:`PagedKVCache.gather_full_kv` is only used
by the SDPA fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch


def flat_write_index(
    block_table: torch.Tensor,
    offsets: torch.Tensor,
    t_new: int,
    block_size: int,
) -> torch.Tensor:
    """``(B * t_new,)`` indices into a pool flattened to ``(blocks * size, ...)``.

    ``offsets`` is the ``(B,)`` per-stream logical write position; row ``b``'s
    ``t``-th new frame lands at logical ``offsets[b] + t``, which the block table
    maps to a physical page and an offset within it.
    """
    steps = torch.arange(t_new, device=offsets.device, dtype=offsets.dtype)
    time_pos = offsets.unsqueeze(1) + steps.unsqueeze(0)  # (B, t_new)
    logical = (time_pos // block_size).long()
    within = (time_pos % block_size).long()
    physical = torch.gather(block_table.long(), dim=1, index=logical)
    return (physical * block_size + within).view(-1)


@dataclass
class PagedKVCache:
    """Per-layer pool views and shared per-batch paging metadata.

    ``block_table`` maps logical to physical pages. ``cache_seqlens`` records
    committed frames before the current write. ``host_seqlen_max`` mirrors its
    maximum to avoid a per-step device-to-host synchronization.
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    block_size: int
    host_seqlen_max: int = 0

    # ------------------------------------------------------------------
    # K/V mutation -- write a new chunk into the pool
    # ------------------------------------------------------------------

    def write_kv_chunk(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        *,
        offset: Union[int, torch.Tensor],
    ) -> None:
        """Write new K/V frames into the paged pool.

        Parameters
        ----------
        new_k, new_v : Tensor
            ``(B, n_kv_head, T, head_dim)`` head-first new K/V to write.
        offset : int or Tensor
            Logical write offset. ``int`` (homogeneous case): every
            stream writes at the same offset; uses a cheap row-slice fast
            path with no D2H sync, and the chunk must land within **two**
            consecutive pages (which every encoder chunk does, being at most
            one page wide). ``(B,)`` int Tensor (heterogeneous case):
            per-stream offsets dispatched via a vectorised scatter, with no
            such bound.
        """
        T = new_k.size(2)
        block_size = self.block_size

        # Frame-major layout matching the pool's per-block tile.
        k_data = new_k.permute(0, 2, 1, 3).contiguous()  # (B, T, H_kv, D)
        v_data = new_v.permute(0, 2, 1, 3).contiguous()

        if isinstance(offset, int):
            # Homogeneous fast path -- same offset for every stream.
            blk_logical = offset // block_size
            blk_offset = offset % block_size
            if blk_offset + T <= block_size:
                phys_blks = self.block_table[:, blk_logical].long()  # (B,)
                self.k_cache[phys_blks, blk_offset : blk_offset + T] = k_data
                self.v_cache[phys_blks, blk_offset : blk_offset + T] = v_data
            else:
                if blk_offset + T > 2 * block_size:
                    # Named here rather than surfacing as a broadcast-shape error
                    # from inside the second index_put: the slice path writes at
                    # most two pages by construction, and a caller with a wider
                    # chunk wants the per-row scatter instead.
                    raise ValueError(
                        f"homogeneous write of {T} frames at offset {offset} spans "
                        f"more than two {block_size}-frame pages; pass a per-row "
                        "offset tensor to take the scatter path"
                    )
                first_n = block_size - blk_offset
                phys_blks = self.block_table[:, blk_logical].long()
                phys_blks_next = self.block_table[:, blk_logical + 1].long()
                self.k_cache[phys_blks, blk_offset:block_size] = k_data[:, :first_n]
                self.v_cache[phys_blks, blk_offset:block_size] = v_data[:, :first_n]
                self.k_cache[phys_blks_next, 0 : T - first_n] = k_data[:, first_n:]
                self.v_cache[phys_blks_next, 0 : T - first_n] = v_data[:, first_n:]
            return

        # Heterogeneous-offset scatter.
        flat_idx = flat_write_index(self.block_table, offset, T, block_size)
        self.scatter_flat(k_data, v_data, flat_idx)

    def scatter_flat(
        self, k_data: torch.Tensor, v_data: torch.Tensor, flat_index: torch.Tensor
    ) -> None:
        """Write frame-major ``(B, T, H_kv, D)`` K/V at precomputed pool slots.

        Split out of :meth:`write_kv_chunk` because the index depends only on the
        block table and the offsets, not on the layer — so a caller writing every
        layer at the same positions (an AR decode step) computes it once and pays
        two ``index_put_`` per layer instead of recomputing the address
        arithmetic each time.
        """
        H_kv, D = self.k_cache.size(2), self.k_cache.size(3)
        self.k_cache.view(-1, H_kv, D)[flat_index] = k_data.reshape(-1, H_kv, D)
        self.v_cache.view(-1, H_kv, D)[flat_index] = v_data.reshape(-1, H_kv, D)

    # ------------------------------------------------------------------
    # K/V access -- gather a contiguous slice for the SDPA fallback
    # ------------------------------------------------------------------

    def gather_full_kv(
        self,
        max_total_kv: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather the first ``max_total_kv`` K/V frames for every stream.

        Streams whose actual valid length is < ``max_total_kv`` end up
        with stale tail data in the gathered tensor; the caller must
        mask those positions via ``cache_seqlens`` (kernel) or pad_bias
        (SDPA fallback).

        Returns ``(k, v)`` both shaped
        ``(B, n_kv_head, max_total_kv, head_dim)``.
        """
        B = self.block_table.size(0)
        H_kv, D = self.k_cache.size(2), self.k_cache.size(3)
        if max_total_kv == 0:
            empty = torch.zeros(
                B,
                H_kv,
                0,
                D,
                dtype=self.k_cache.dtype,
                device=self.k_cache.device,
            )
            return empty, empty.clone()

        block_size = self.block_size
        num_blocks = (max_total_kv + block_size - 1) // block_size
        block_ids = self.block_table[:, :num_blocks].long()  # (B, num_blocks)

        k_gathered = self.k_cache[block_ids].reshape(B, num_blocks * block_size, H_kv, D)[
            :, :max_total_kv
        ]
        v_gathered = self.v_cache[block_ids].reshape(B, num_blocks * block_size, H_kv, D)[
            :, :max_total_kv
        ]
        return k_gathered.permute(0, 2, 1, 3), v_gathered.permute(0, 2, 1, 3)


__all__ = ["PagedKVCache", "flat_write_index"]
