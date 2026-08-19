# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared types and configuration for the ASR streaming cache manager."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from oasr.cache.state import StreamStateSpec

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Master configuration for the ASR streaming cache system.

    Parameters
    ----------
    num_layers : int
        Number of conformer encoder layers.
    n_kv_head : int
        Number of KV attention heads per layer.
    head_dim : int
        Dimension per attention head (d_k). K and V are packed along the last
        dimension, so the stored last dim is ``head_dim * 2``.
    hidden_dim : int
        Encoder hidden dimension (output_size), used for CNN cache.
    kernel_size : int
        Depthwise conv kernel size in the ConvolutionModule. The CNN cache
        stores ``kernel_size - 1`` frames per layer (causal left-padding).
    chunk_size : int
        Number of encoder output frames per chunk (after subsampling).
    num_left_chunks : int
        Maximum number of left-context chunks to retain for attention.
        ``-1`` means unlimited history (all past frames kept) — bounded in
        practice by :attr:`blocks_per_stream`, past which the stream is either
        terminated or recycled (see ``recycle_streaming_history``).
    block_size_frames : int
        Number of time frames per physical block (page) in the attention
        KV pool. Setting this equal to ``chunk_size`` (the default) means
        each chunk maps to exactly one block, which simplifies commit logic.
    max_num_blocks : int
        Total number of physical blocks in the shared pool. Must be large
        enough to cover all concurrent streams:
        ``max_num_blocks >= max_batch_size * max_logical_blocks``.
    device : torch.device
        CUDA device used for all tensor allocations.
    dtype : torch.dtype
        Floating-point dtype for attention and CNN cache tensors.
    """

    num_layers: int = 12
    n_kv_head: int = 4
    head_dim: int = 64
    hidden_dim: int = 256
    kernel_size: int = 15
    chunk_size: int = 16
    num_left_chunks: int = -1
    recycle_streaming_history: bool = False
    """Recycle the oldest KV block instead of terminating a stream at capacity.

    Only meaningful with ``num_left_chunks < 0``.  Unlimited history is unlimited
    only until the stream reaches :attr:`blocks_per_stream` — the ceiling the
    block table and the pool's fair share already impose — at which point the
    engine must either stop the stream (the default, ``finish_reason="length"``)
    or start dropping history.  Setting this picks the latter: memory stays
    bounded by construction and a long-running stream keeps transcribing.

    Measured on the WeNet conformer (4 streams, vs unlimited history): audio
    *inside* the retained window is bit-identical (0/4 transcripts differ,
    0.00% WER); audio past it decodes in full where unlimited truncates.
    """
    stream_states: Tuple[StreamStateSpec, ...] = ()
    """Fixed-extent per-stream tensors **beyond** the convolutional left-context.

    Declared by the encoder (``CacheSpec.stream_states``) and allocated as one
    persistent slot-addressed buffer each; see :mod:`oasr.cache.state`.  Empty for
    every encoder whose only cross-chunk state is K/V plus the conv cache.
    """
    prefill_kv_window: bool = False
    """Allocate a stream's whole retained K/V window, zeroed, at admission.

    For an encoder whose attention span is a **trained constant** rather than
    "whatever history fits" (Nemotron's ``sliding_window``).  Three things follow,
    and they are why this is a mode rather than an optimisation:

    * every stream reports the **same** ``cache_seqlens`` from its first chunk, so
      one shared relative-position table is correct for the whole cohort.  A
      Transformer-XL table's distances are ``cache + i - j``, so a per-row
      ``cache`` would need a per-row table — and ``relative_k_proj`` is per layer,
      which at ``B = 32`` costs more than the encoder layer it serves;
    * ``cache_t1`` is constant, so the CUDA-graph cache captures **one** graph per
      batch size instead of one per ``(B, cache_t1 bucket)``;
    * a young stream's leading key columns are zeros, which the encoder masks with
      its additive bias.  They are *finite*, which is what
      ``oasr.fmha``'s "v must be finite where the kernel can read" precondition
      requires — hence zeroed, never ``empty``.

    Requires ``num_left_chunks >= 0`` (a bounded window is the whole premise) and
    costs the pool ``max_batch_size * blocks_per_stream`` blocks from admission,
    which is the invariant :meth:`__post_init__` already enforces there.
    """
    block_size_frames: int = 16
    max_num_blocks: int = 1024
    max_batch_size: int = 32
    """Maximum concurrent streams. Sizes the persistent batched block_table,
    cache_seqlens, CNN cache, and feature buffer tensors. Each admitted stream
    gets a slot id in ``[0, max_batch_size)`` via ``StreamSlotPool``."""
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    dtype: torch.dtype = torch.bfloat16

    @property
    def kv_last_dim(self) -> int:
        """Packed KV last dimension: ``head_dim * 2``."""
        return self.head_dim * 2

    @property
    def cnn_cache_frames(self) -> int:
        """Frames stored in CNN cache per layer: ``kernel_size - 1``."""
        return self.kernel_size - 1

    @property
    def max_cache_frames(self) -> int:
        """Maximum attention cache length in frames.

        Returns ``-1`` when ``num_left_chunks < 0`` (unlimited history).
        """
        if self.num_left_chunks < 0:
            return -1
        return self.chunk_size * self.num_left_chunks

    max_blocks_per_seq: int = 512
    """Maximum number of logical blocks per stream for the block_table tensor.

    Must satisfy ``max_blocks_per_seq >= max_logical_blocks`` (when
    ``num_left_chunks >= 0``).  Used to size the ``block_table`` tensor
    allocated for each stream in :class:`AttentionCacheManager`.
    """

    @property
    def max_logical_blocks(self) -> Optional[int]:
        """Maximum logical blocks per stream for the attention KV cache.

        Returns ``None`` when history is unlimited.
        """
        if self.num_left_chunks < 0:
            if not self.recycle_streaming_history:
                return None
            # Unlimited history still has a physical ceiling.  Recycling drops
            # the oldest block there instead of terminating the stream.
            return self.blocks_per_stream
        total_frames = self.chunk_size * self.num_left_chunks
        return (total_frames + self.block_size_frames - 1) // self.block_size_frames

    # ------------------------------------------------------------------
    # Capacity accounting
    # ------------------------------------------------------------------

    @property
    def blocks_per_stream(self) -> int:
        """Logical blocks one stream may hold before it is at capacity.

        With eviction enabled (``num_left_chunks >= 0``) this is
        :attr:`max_logical_blocks`.  With unlimited history the binding
        constraints are the ``block_table`` row width and the pool's fair share,
        so a stream is capped by ``min(max_blocks_per_seq, max_num_blocks //
        max_batch_size)`` — the point past which growth would either index off
        the block table or starve peer streams.
        """
        if self.num_left_chunks >= 0:
            # An explicit cap is the binding constraint.  Read the field rather
            # than ``max_logical_blocks`` — that property calls *this* one for
            # the derived case, and going through it would recurse.
            total = self.chunk_size * self.num_left_chunks
            return max(1, (total + self.block_size_frames - 1) // self.block_size_frames)
        fair_share = self.max_num_blocks // max(1, self.max_batch_size)
        return max(1, min(self.max_blocks_per_seq, fair_share))

    @property
    def max_stream_frames(self) -> int:
        """Encoder frames one stream may accumulate at :attr:`blocks_per_stream`."""
        return self.blocks_per_stream * self.block_size_frames

    @property
    def prefilled_cache_frames(self) -> int:
        """Constant ``cache_seqlens`` a prefilled window reports during a forward.

        ``prepare_chunks_batched`` evicts to ``max_logical_blocks - 1`` before
        allocating the chunk's block, so the *past* frames visible to the kernel
        are exactly that many blocks' worth — and with the window prefilled, every
        stream is there from its first chunk.
        """
        blocks = self.max_logical_blocks
        if blocks is None:
            raise ValueError(
                "prefill_kv_window needs a bounded window (num_left_chunks >= 0); "
                "with unlimited history there is no window to prefill"
            )
        return max(0, blocks - 1) * self.block_size_frames

    def __post_init__(self) -> None:
        if self.prefill_kv_window and self.num_left_chunks < 0:
            raise ValueError(
                "prefill_kv_window requires num_left_chunks >= 0: the point is a "
                "fixed, trained attention window, and unlimited history has none"
            )
        # One block is allocated per encoder chunk (``prepare_chunks_batched``),
        # so a chunk that doesn't fit in a page would spill into the *next*
        # logical block — which ``PagedKVCache``'s two-block write path reads as
        # the following chunk's block, silently corrupting KV.
        if self.chunk_size > self.block_size_frames:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be <= block_size_frames "
                f"({self.block_size_frames}): one paged block is allocated per "
                "encoder chunk, so a larger chunk would overflow its block."
            )
        evict_cap = self.max_logical_blocks
        if evict_cap is not None and evict_cap > self.max_blocks_per_seq:
            raise ValueError(
                f"max_blocks_per_seq ({self.max_blocks_per_seq}) is smaller than "
                f"the retained history ({evict_cap} blocks for num_left_chunks="
                f"{self.num_left_chunks}); the block table cannot address the "
                "cache this config asks for."
            )
        if self.num_left_chunks >= 0:
            # Eviction disables the per-stream capacity gate, so the pool must
            # hold every concurrent stream's retained history.
            needed = self.max_batch_size * self.blocks_per_stream
            if self.max_num_blocks < needed:
                raise ValueError(
                    f"max_num_blocks ({self.max_num_blocks}) cannot hold "
                    f"{self.max_batch_size} concurrent streams of "
                    f"{self.blocks_per_stream} blocks each = {needed} blocks "
                    f"(num_left_chunks={self.num_left_chunks}, chunk_size="
                    f"{self.chunk_size}, block_size_frames={self.block_size_frames}). "
                    f"With eviction enabled there is no capacity gate, so the pool "
                    f"would run dry inside the encoder forward. Raise max_num_blocks "
                    f"to >= {needed}, lower max_batch_size, or shorten the retained "
                    f"history."
                )
        if evict_cap is None:
            # Unlimited history: growth is bounded by the pool's fair share and
            # the block-table width, so every stream has a finite ceiling.  Say
            # what it is — a stream that reaches it is finalized early (with
            # ``finish_reason="length"``), which is otherwise silent.
            # The seconds figure assumes the common 4x-subsampling / 10 ms-hop
            # geometry; it is an operator hint, not a load-bearing number.
            logger.info(
                "streaming KV cache: unlimited history (num_left_chunks=-1) → each "
                "stream is capped at %d blocks / %d encoder frames (~%.0fs of audio "
                "at 4x subsampling); pool=%d blocks, max_batch_size=%d, "
                "max_blocks_per_seq=%d. Raise max_num_blocks to lift the cap, or "
                "set num_left_chunks to bound history by eviction instead.",
                self.blocks_per_stream,
                self.max_stream_frames,
                self.max_stream_frames * 4 * 0.01,
                self.max_num_blocks,
                self.max_batch_size,
                self.max_blocks_per_seq,
            )
