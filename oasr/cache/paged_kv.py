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


@dataclass
class PagedKVCache:
    """Per-layer paged KV cache descriptor.

    All tensors except ``k_cache`` / ``v_cache`` are shared across
    encoder layers within the same forward call (one ``block_table`` and
    one ``cache_seqlens`` per stream batch).

    Attributes
    ----------
    k_cache, v_cache : Tensor
        ``(max_num_blocks, block_size, n_kv_head, head_dim)`` views into
        the block pool for one encoder layer.
    block_table : Tensor
        ``(B, max_blocks_per_seq)`` int32 -- per-stream logical -> physical
        block mapping.
    cache_seqlens : Tensor
        ``(B,)`` int32 -- committed K/V frames in the pool **before** the
        current chunk's K/V write. Lives on the same device as ``k_cache``.
    block_size : int
        Frames per physical block (= ``block_size_frames`` in
        :class:`~oasr.cache.types.CacheConfig`).
    host_seqlen_max : int
        Host-side mirror of ``cache_seqlens.max().item()`` so the encoder
        can compute the kernel's max key-sequence length without a
        per-step D2H sync (the engine already tracks ``Request.offset``
        on the host).
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
            path with no D2H sync. ``(B,)`` int Tensor (heterogeneous
            case): per-stream offsets dispatched via a vectorised
            scatter.
        """
        B, _, T, _ = new_k.shape
        H_kv = new_k.size(1)
        D = new_k.size(3)
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
                self.k_cache[phys_blks, blk_offset: blk_offset + T] = k_data
                self.v_cache[phys_blks, blk_offset: blk_offset + T] = v_data
            else:
                first_n = block_size - blk_offset
                phys_blks = self.block_table[:, blk_logical].long()
                phys_blks_next = self.block_table[:, blk_logical + 1].long()
                self.k_cache[phys_blks, blk_offset:block_size] = k_data[:, :first_n]
                self.v_cache[phys_blks, blk_offset:block_size] = v_data[:, :first_n]
                self.k_cache[phys_blks_next, 0: T - first_n] = k_data[:, first_n:]
                self.v_cache[phys_blks_next, 0: T - first_n] = v_data[:, first_n:]
            return

        # Heterogeneous-offset scatter.
        arange_T = torch.arange(T, device=offset.device, dtype=offset.dtype)
        time_pos = offset.unsqueeze(1) + arange_T.unsqueeze(0)  # (B, T)
        blk_logical_t = (time_pos // block_size).long()
        blk_offset_t = (time_pos % block_size).long()

        phys_blk = torch.gather(self.block_table.long(), dim=1, index=blk_logical_t)
        flat_idx = (phys_blk * block_size + blk_offset_t).view(-1)

        k_flat = self.k_cache.view(-1, H_kv, D)
        v_flat = self.v_cache.view(-1, H_kv, D)
        k_flat[flat_idx] = k_data.view(B * T, H_kv, D)
        v_flat[flat_idx] = v_data.view(B * T, H_kv, D)

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
                B, H_kv, 0, D,
                dtype=self.k_cache.dtype, device=self.k_cache.device,
            )
            return empty, empty.clone()

        block_size = self.block_size
        num_blocks = (max_total_kv + block_size - 1) // block_size
        block_ids = self.block_table[:, :num_blocks].long()  # (B, num_blocks)

        k_gathered = self.k_cache[block_ids].reshape(
            B, num_blocks * block_size, H_kv, D
        )[:, :max_total_kv]
        v_gathered = self.v_cache[block_ids].reshape(
            B, num_blocks * block_size, H_kv, D
        )[:, :max_total_kv]
        return k_gathered.permute(0, 2, 1, 3), v_gathered.permute(0, 2, 1, 3)


__all__ = ["PagedKVCache"]
