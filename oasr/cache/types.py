# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared types and configuration for the ASR streaming cache manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


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
        ``-1`` means unlimited history (all past frames kept).
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
    block_size_frames: int = 16
    max_num_blocks: int = 1024
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    dtype: torch.dtype = torch.float16

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

    @property
    def max_logical_blocks(self) -> Optional[int]:
        """Maximum logical blocks per stream for the attention KV cache.

        Returns ``None`` when history is unlimited.
        """
        if self.num_left_chunks < 0:
            return None
        total_frames = self.chunk_size * self.num_left_chunks
        return (total_frames + self.block_size_frames - 1) // self.block_size_frames
