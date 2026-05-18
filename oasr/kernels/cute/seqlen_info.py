# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-batch sequence-length info for the FMHA forward kernel.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/seqlen_info.py``.
The implementation is OASR-native and only covers the four KV-layout modes
the forward kernel needs:

* **dense**           -- ``mK`` is ``(B, H_kv, T_k, D)``; ``seqlen_k = T_k``.
* **dense + cache_seqlens** -- ``seqlen_k = cache_seqlens[b]``; columns past
                              it are masked to ``-inf`` in softmax.
* **varlen**          -- ``mK`` is packed ``(total_k, H_kv, D)``;
                         ``cu_seqlens_k`` gives the per-batch offset and length.
* **paged**           -- ``mK`` is ``(num_blocks, block_size, H_kv, D)``;
                         logical length is derived from ``block_table`` width
                         and tightened by ``cache_seqlens``.

The struct is *value-typed*: all fields are loaded once at the top of the
kernel and passed downstream as a flat record. cute.jit treats this as a
constexpr aggregate.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

import cutlass
import cutlass.cute as cute


class SeqlenInfoQK:
    """Per-batch Q + K sequence-length descriptor.

    Fields (all loaded once at the top of the kernel):

    * ``offset_q``   -- row offset into Q's leading axis. 0 for dense, the
                        ``cu_seqlens_q[b]`` prefix-sum entry for varlen.
    * ``seqlen_q``   -- valid Q rows for this batch.
    * ``offset_k``   -- row offset into K/V's leading axis. 0 for dense /
                        paged, ``cu_seqlens_k[b]`` for varlen.
    * ``seqlen_k``   -- logical KV length actually attended to. In dense
                        mode = ``mK.shape[2]`` (or ``cache_seqlens[b]``).
                        In paged mode = ``cache_seqlens[b]`` clamped to
                        ``mBlockTable.shape[1] * block_size``.
    * ``has_cache_seqlens`` -- constexpr flag; gates the per-stream column
                        mask in softmax.
    """

    def __init__(
        self,
        *,
        offset_q: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        offset_k: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        has_cache_seqlens: cutlass.Constexpr,
    ):
        self.offset_q = offset_q
        self.seqlen_q = seqlen_q
        self.offset_k = offset_k
        self.seqlen_k = seqlen_k
        self.has_cache_seqlens = has_cache_seqlens

    @staticmethod
    @cute.jit
    def from_dense(
        batch_idx: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k_full: cutlass.Int32,
        mCacheSeqlens: cute.Tensor,
        has_cache_seqlens: cutlass.Constexpr,
    ) -> "SeqlenInfoQK":
        """Build a dense (or dense + cache_seqlens) descriptor.

        ``seqlen_k_full`` is the contiguous T_k axis size; when
        ``has_cache_seqlens`` is True, the per-stream value is read
        from ``mCacheSeqlens[batch_idx]`` and used as ``seqlen_k``.
        Otherwise ``seqlen_k = seqlen_k_full``.
        """
        if cutlass.const_expr(has_cache_seqlens):
            seqlen_k = mCacheSeqlens[batch_idx]
        else:
            seqlen_k = seqlen_k_full
        return SeqlenInfoQK(
            offset_q=cutlass.Int32(0),
            seqlen_q=seqlen_q,
            offset_k=cutlass.Int32(0),
            seqlen_k=seqlen_k,
            has_cache_seqlens=has_cache_seqlens,
        )

    @staticmethod
    @cute.jit
    def from_varlen(
        batch_idx: cutlass.Int32,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
    ) -> "SeqlenInfoQK":
        """Build a varlen descriptor from cu_seqlens prefix-sum tables."""
        offset_q = cu_seqlens_q[batch_idx]
        offset_k = cu_seqlens_k[batch_idx]
        seqlen_q = cu_seqlens_q[batch_idx + 1] - offset_q
        seqlen_k = cu_seqlens_k[batch_idx + 1] - offset_k
        return SeqlenInfoQK(
            offset_q=offset_q,
            seqlen_q=seqlen_q,
            offset_k=offset_k,
            seqlen_k=seqlen_k,
            has_cache_seqlens=False,
        )

    @staticmethod
    @cute.jit
    def from_paged(
        batch_idx: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        mCacheSeqlens: cute.Tensor,
    ) -> "SeqlenInfoQK":
        """Build a paged descriptor; ``seqlen_k = cache_seqlens[b]``.

        The kernel still walks ``mBlockTable.shape[1] * block_size`` logical
        columns; the per-stream length narrows attention via the column
        mask in softmax.
        """
        return SeqlenInfoQK(
            offset_q=cutlass.Int32(0),
            seqlen_q=seqlen_q,
            offset_k=cutlass.Int32(0),
            seqlen_k=mCacheSeqlens[batch_idx],
            has_cache_seqlens=True,
        )


__all__ = ["SeqlenInfoQK"]
