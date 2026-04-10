#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Multi-head attention layers.

This module ports the attention implementations from
`wenet/models/transformer/attention.py` into OASR, keeping the
algorithms unchanged while adapting them to live inside the OASR
package.

The core classes mirror WeNet:

- MultiHeadedAttention
- RelPositionMultiHeadedAttention
- MultiHeadedCrossAttention
- ShawRelPositionMultiHeadedAttention
- RopeMultiHeadedAttention

These are standard PyTorch ``nn.Module`` implementations. They do
not yet call into OASR CUDA attention kernels; see the project
documentation for what kernel bindings would be needed to make
that possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from oasr.layers.rotary_embedding import get_apply_rotary_emb

T_CACHE = Tuple[torch.Tensor, torch.Tensor]

# ---------------------------------------------------------------------------
# Optional flash-attention import
# ---------------------------------------------------------------------------

try:
    from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache

    _HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover
    _flash_attn_with_kvcache = None  # type: ignore[assignment]
    _HAS_FLASH_ATTN = False


# ---------------------------------------------------------------------------
# Paged KV cache descriptor
# ---------------------------------------------------------------------------


@dataclass
class PagedKVCache:
    """Per-layer paged KV cache descriptor for inference-only decoding.

    ``k_cache`` and ``v_cache`` are views into the shared
    :class:`~oasr.cache.block_pool.BlockPool` for a single encoder layer.
    All layers belonging to the same stream share the **same Python tensor
    objects** for ``block_table`` and ``cache_seqlens``, so that a single
    in-place update to ``cache_seqlens`` propagates to every layer.

    Attributes
    ----------
    k_cache : Tensor
        ``(max_num_blocks, block_size, n_kv_head, head_dim)`` — physical K pool.
    v_cache : Tensor
        ``(max_num_blocks, block_size, n_kv_head, head_dim)`` — physical V pool.
    block_table : Tensor
        ``(1, max_blocks_per_seq)`` int32 — logical → physical block mapping.
    cache_seqlens : Tensor
        ``(1,)`` int32 — number of K/V tokens committed to the cache **before**
        the current chunk.  Updated by the cache manager's ``commit_chunk``
        *after* ``forward_chunk`` returns.
    block_size : int
        Frames per physical block (= ``block_size_frames`` in
        :class:`~oasr.cache.types.CacheConfig`).
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    block_size: int


# ---------------------------------------------------------------------------
# Paged-cache low-level helpers
# ---------------------------------------------------------------------------


def _paged_write_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_offset: int,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> None:
    """Write new K/V frames into the paged block pool.

    Parameters
    ----------
    k_cache, v_cache : Tensor
        Pool tensors ``(max_blocks, block_size, n_kv_head, head_dim)``.
    block_table : Tensor
        ``(1, max_blocks_per_seq)`` int32.
    seqlen_offset : int
        Absolute frame index at which to start writing (= ``cache_seqlens[0]``
        before any increment).
    new_k, new_v : Tensor
        ``(1, n_kv_head, chunk_size, head_dim)`` — **head-first** layout as
        returned by :meth:`MultiHeadedAttention.forward_qkv`.
    """
    block_size = k_cache.size(1)
    chunk_size = new_k.size(2)
    # Pool layout: (block_size, n_kv_head, head_dim) → transpose to time-first.
    k_data = new_k[0].permute(1, 0, 2).contiguous()  # (chunk_size, n_kv_head, d_k)
    v_data = new_v[0].permute(1, 0, 2).contiguous()

    written, pos = 0, seqlen_offset
    while written < chunk_size:
        blk_logical = pos // block_size
        blk_offset = pos % block_size
        phys_blk = block_table[0, blk_logical].item()
        frames = min(block_size - blk_offset, chunk_size - written)
        k_cache[phys_blk, blk_offset: blk_offset + frames].copy_(
            k_data[written: written + frames]
        )
        v_cache[phys_blk, blk_offset: blk_offset + frames].copy_(
            v_data[written: written + frames]
        )
        written += frames
        pos += frames


def _paged_gather_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    total_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather K and V from the paged pool for all ``total_frames`` tokens.

    The frames returned include any frames just written by :func:`_paged_write_kv`
    in the same forward pass (because those writes happened in-place to the
    pool tensors before this gather).

    Parameters
    ----------
    k_cache, v_cache : Tensor
        Pool tensors ``(max_blocks, block_size, n_kv_head, head_dim)``.
    block_table : Tensor
        ``(1, max_blocks_per_seq)`` int32.
    total_frames : int
        Total number of frames to retrieve.  Pass
        ``cache_seqlens[0] + chunk_size`` to include the current chunk.

    Returns
    -------
    k, v : Tensor
        Both shaped ``(1, n_kv_head, total_frames, head_dim)`` — head-first,
        matching SDPA's expected layout.
    """
    if total_frames == 0:
        n_kv_head, head_dim = k_cache.size(2), k_cache.size(3)
        empty = torch.zeros(
            1, n_kv_head, 0, head_dim, dtype=k_cache.dtype, device=k_cache.device
        )
        return empty, empty.clone()

    block_size = k_cache.size(1)
    num_blocks = (total_frames + block_size - 1) // block_size
    block_ids = block_table[0, :num_blocks].long()  # int32 → int64 for indexing

    # (num_blocks, block_size, n_kv_head, head_dim) → (total_frames, ...)
    k_flat = k_cache[block_ids].reshape(-1, k_cache.size(2), k_cache.size(3))[:total_frames]
    v_flat = v_cache[block_ids].reshape(-1, v_cache.size(2), v_cache.size(3))[:total_frames]

    # → (1, n_kv_head, total_frames, head_dim)  head-first
    k = k_flat.permute(1, 0, 2).unsqueeze(0)
    v = v_flat.permute(1, 0, 2).unsqueeze(0)
    return k, v


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Always uses PyTorch SDPA kernels.

    If ``n_kv_head`` is not ``None`` and ``n_kv_head != n_head``, this
    implements Multi-Query or Grouped-Query attention:

    - case 1: ``n_kv_head is None`` and ``head_dim is None``:
      standard multi-head self-attention (MHSA)
    - case 2: ``n_kv_head == 1`` and ``n_head > 1``:
      multi-query attention (MQA)
    - case 3: ``1 < n_kv_head < n_head``:
      grouped-query attention (GQA)

    Args:
        n_head: Number of attention heads.
        n_feat: Model dimension (input and output size).
        query_bias: Whether to include bias in the query projection.
        key_bias: Whether to include bias in the key projection.
        value_bias: Whether to include bias in the value projection.
        n_kv_head: Number of key/value heads (for MQA/GQA). If ``None``,
            defaults to ``n_head`` (standard multi-head).
        head_dim: Per-head dimension. If ``None``, inferred as
            ``n_feat // n_head``.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        """Construct a MultiHeadedAttention object (ported from WeNet)."""
        super().__init__()

        # Inner dimensions for Q/K/V projections
        self.inner_dim = n_feat if head_dim is None else head_dim * n_head
        if n_kv_head is not None:
            assert head_dim is not None, "head_dim must be set when n_kv_head is not None"
            self.inner_kv_dim = head_dim * n_kv_head
            n_kv_head = n_kv_head
        else:
            self.inner_kv_dim = self.inner_dim
            n_kv_head = n_head

        # We assume d_v always equals d_k
        self.d_k = self.inner_dim // n_head
        assert self.d_k == self.inner_kv_dim // n_kv_head

        self.h = n_head
        self.h_kv = n_kv_head

        self.linear_q = nn.Linear(n_feat, self.inner_dim, bias=query_bias)
        self.linear_k = nn.Linear(n_feat, self.inner_kv_dim, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, self.inner_kv_dim, bias=value_bias)
        self.linear_out = nn.Linear(self.inner_dim, n_feat, bias=query_bias)

    def _forward_linearx(
        self,
        name: str,
        x: torch.Tensor,
        head_first: bool = True,
    ) -> torch.Tensor:
        """Apply the Q/K/V projection and reshape into head dimensions."""

        assert x.ndim >= 3
        if name == "query":
            x = self.linear_q(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h, self.d_k])
        elif name == "key":
            x = self.linear_k(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])
        else:
            assert name == "value"
            x = self.linear_v(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])

        # split last dim
        x = x.view(x_shape)
        if head_first:
            # (batch, ..., head or head_kv, time, d_k)
            x = x.transpose(-3, -2)
        return x

    def forward_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key, and value.

        Args:
            query: Query tensor of shape ``(batch, ..., time1, size)``.
            key: Key tensor of shape ``(batch, ..., time2, size)``.
            value: Value tensor of shape ``(batch, ..., time2, size)``.

        Returns:
            Transformed Q/K/V tensors:

            - ``q``: ``(batch, ..., n_head, time1, d_k)``
            - ``k``: ``(batch, ..., n_head_kv, time2, d_k)``
            - ``v``: ``(batch, ..., n_head_kv, time2, d_k)``
        """

        q = self._forward_linearx("query", query)
        k = self._forward_linearx("key", key)
        v = self._forward_linearx("value", value)
        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value: Transformed value, shape
                ``(batch, ..., n_head, time2, d_k)``.
            scores: Attention scores, shape
                ``(batch, ..., n_head, time1, time2)``.
            mask: Attention mask, either of shape

                - ``(batch, 1, time2)`` or
                - ``(batch, ..., time1, time2)``

                A value of 0 means the position is masked.

        Returns:
            Output tensor of shape ``(batch, ..., time1, d_model)``.
        """

        # NOTE: When will `if mask.size(-1) > 0` be True?
        # 1. ONNX export for first chunk
        # 2. PyTorch training
        if mask.size(-1) > 0:  # time2 > 0
            # (batch, ..., 1, *, time2)
            mask = mask.unsqueeze(-3).eq(0)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[..., : scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores.float(), dim=-1).type_as(value).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            # NOTE: This path is used for some ONNX/JIT export modes.
            attn = torch.softmax(scores.float(), dim=-1).type_as(
                value
            )  # (batch, ..., head, time1, time2)

        x = torch.matmul(attn, value)  # (batch, ..., head, time1, d_k)
        # [batch, ..., time1, head, d_k]
        x = x.transpose(-3, -2).contiguous()
        x_shape = x.size()[:-2] + torch.Size([self.h * self.d_k])
        x = x.view(x_shape)  # (batch, ..., time1, d_model)
        return self.linear_out(x)  # (batch, ..., time1, d_model)

    def _update_kv_and_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: T_CACHE,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE]:
        """Update KV cache for streaming / incremental decoding."""

        new_cache = cache
        seq_axis = -2 if head_first else -3
        head_axis = -3 if head_first else -2

        if not self.training:
            # When exporting ONNX or running inference, we append new K/V
            # to the cache. Zero-shaped tensors are handled gracefully.
            key_cache, value_cache = cache
            if key_cache.size(0) > 0:
                k = torch.cat([key_cache, k], dim=seq_axis)
            if value_cache.size(0) > 0:
                v = torch.cat([value_cache, v], dim=seq_axis)

            # Cache slicing for chunked inference is handled at the
            # encoder layer level; here we simply concatenate.
            new_cache = (k, v)

        # For multi-query or grouped-query attention, expand KV heads.
        if self.h_kv != self.h and self.h_kv != 1:
            n_repeat = self.h // self.h_kv

            k_shape = k.size()
            repeat_axis = head_axis + 1
            k = (
                k.unsqueeze(head_axis)
                .expand(
                    k_shape[:repeat_axis]
                    + torch.Size([n_repeat])
                    + k_shape[repeat_axis:]
                )
                .reshape(
                    k_shape[:head_axis]
                    + torch.Size([self.h_kv * n_repeat])
                    + k_shape[repeat_axis:]
                )
            )

            v_shape = v.size()
            v = (
                v.unsqueeze(head_axis)
                .expand(
                    v_shape[:repeat_axis]
                    + torch.Size([n_repeat])
                    + v_shape[repeat_axis:]
                )
                .reshape(
                    v_shape[:head_axis]
                    + torch.Size([self.h_kv * n_repeat])
                    + v_shape[repeat_axis:]
                )
            )

        return k, v, new_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: Union[T_CACHE, PagedKVCache] = (
            torch.zeros(0, 0, 0, 0),
            torch.zeros(0, 0, 0, 0),
        ),
    ) -> Tuple[torch.Tensor, Union[T_CACHE, PagedKVCache]]:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape ``(batch, time1, size)``.
            key: Key tensor of shape ``(batch, time2, size)``.
            value: Value tensor of shape ``(batch, time2, size)``.
            mask: Mask tensor of shape ``(batch, 1, time2)`` or
                ``(batch, time1, time2)``.
            pos_emb: Positional embedding (unused in base class).
            cache: Dense KV cache pair (training / non-paged inference) or
                a :class:`PagedKVCache` descriptor for paged inference.

        Returns:
            A tuple of:

            - Output tensor ``(batch, time1, d_model)``.
            - Updated cache (same type as input).
        """

        del pos_emb  # Not used in standard attention

        if isinstance(cache, PagedKVCache):
            q, k, v = self.forward_qkv(query, key, value)
            # q: (1, n_head, chunk_size, d_k)
            seqlen_offset = int(cache.cache_seqlens[0].item())
            total_frames = seqlen_offset + q.size(2)

            if (
                _HAS_FLASH_ATTN
                and query.is_cuda
                and query.dtype in (torch.float16, torch.bfloat16)
            ):
                # flash_attn_with_kvcache expects (batch, seqlen, nheads, head_dim).
                # It writes k/v into the paged pool and computes attention atomically.
                q_fa = q.permute(0, 2, 1, 3)   # (1, chunk_size, n_head, d_k)
                k_fa = k.permute(0, 2, 1, 3)   # (1, chunk_size, n_kv_head, d_k)
                v_fa = v.permute(0, 2, 1, 3)
                output = _flash_attn_with_kvcache(
                    q_fa,
                    cache.k_cache,
                    cache.v_cache,
                    k_fa,
                    v_fa,
                    block_table=cache.block_table,
                    cache_seqlens=cache.cache_seqlens,
                    softmax_scale=1.0 / math.sqrt(self.d_k),
                    causal=False,
                )  # (1, chunk_size, n_head, d_k)
                output = output.reshape(query.size(0), -1, self.h * self.d_k)
            else:
                # Fallback: write K/V into pool, gather all frames, run SDPA.
                _paged_write_kv(
                    cache.k_cache, cache.v_cache,
                    cache.block_table, seqlen_offset, k, v,
                )
                k_full, v_full = _paged_gather_kv(
                    cache.k_cache, cache.v_cache, cache.block_table, total_frames,
                )
                # Expand KV heads for GQA (not needed for MQA: SDPA broadcasts).
                if self.h_kv != self.h and self.h_kv != 1:
                    n_repeat = self.h // self.h_kv
                    k_full = k_full.repeat_interleave(n_repeat, dim=1)
                    v_full = v_full.repeat_interleave(n_repeat, dim=1)
                assert mask.dtype != torch.bool
                output = F.scaled_dot_product_attention(
                    q, k_full, v_full,
                    attn_mask=mask.unsqueeze(1),
                    scale=1 / math.sqrt(self.d_k),
                )
                output = (
                    output.transpose(1, 2)
                    .contiguous()
                    .view(query.size(0), -1, self.h * self.d_k)
                )
            return self.linear_out(output), cache

        # ----------------------------------------------------------------
        # Dense T_CACHE path (training / non-paged inference)
        # ----------------------------------------------------------------
        q, k, v = self.forward_qkv(query, key, value)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        # SDPA-only path. The `mask` tensor is expected to be an additive
        # bias (float/bfloat16/half), not a boolean padding mask.
        assert mask.dtype != torch.bool
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(1),
            scale=1 / math.sqrt(self.d_k),
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(query.size(0), -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(output), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.

    Paper: https://arxiv.org/abs/1901.02860
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        """Construct a RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head,
            n_feat,
            query_bias,
            key_bias,
            value_bias,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )

        # Linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        # Learnable biases used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor, zero_triu: bool = False) -> torch.Tensor:
        """Compute relative positional encoding shift.

        Args:
            x: Input tensor ``(batch, head, time1, time2)``.
            zero_triu: If True, zero out the upper triangular part.
        """

        zero_pad = torch.zeros(
            (x.size()[0], x.size()[1], x.size()[2], 1),
            device=x.device,
            dtype=x.dtype,
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(
            x.size()[0],
            x.size()[1],
            x.size(3) + 1,
            x.size(2),
        )
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)),
                              device=x.device, dtype=x.dtype)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor = torch.empty(0),
        cache: Union[T_CACHE, PagedKVCache] = (
            torch.zeros((0, 0, 0, 0)),
            torch.zeros((0, 0, 0, 0)),
        ),
    ) -> Tuple[torch.Tensor, Union[T_CACHE, PagedKVCache]]:
        """Compute scaled dot-product attention with relative positional encoding.

        When ``cache`` is a :class:`PagedKVCache` the method writes new K/V
        frames into the shared block pool *before* gathering the full context,
        so that the rel-pos bias can be applied over the complete
        ``(cache_seqlens + chunk_size)``-frame key sequence.
        """

        q, k, v = self.forward_qkv(query, key, value)
        # q: (batch, n_head, time1, d_k)
        q = q.transpose(1, 2)  # (batch, time1, n_head, d_k)

        if isinstance(cache, PagedKVCache):
            seqlen_offset = int(cache.cache_seqlens[0].item())
            total_frames = seqlen_offset + q.size(1)
            # Write new K/V into the paged pool.
            _paged_write_kv(
                cache.k_cache, cache.v_cache,
                cache.block_table, seqlen_offset, k, v,
            )
            # Gather the full key/value context (past + current chunk).
            k_full, v_full = _paged_gather_kv(
                cache.k_cache, cache.v_cache, cache.block_table, total_frames,
            )
            # k_full: (1, n_kv_head, total_frames, d_k)
            # Expand KV heads for GQA (not needed for MQA: SDPA broadcasts).
            if self.h_kv != self.h and self.h_kv != 1:
                n_repeat = self.h // self.h_kv
                k_full = k_full.repeat_interleave(n_repeat, dim=1)
                v_full = v_full.repeat_interleave(n_repeat, dim=1)
            new_cache: Union[T_CACHE, PagedKVCache] = cache
        else:
            k, v, new_cache = self._update_kv_and_cache(k, v, cache)
            k_full, v_full = k, v

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, n_head, time_pos, d_k)

        # (batch, n_head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # (batch, n_head, time1, time_pos)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        # SDPA-only path: matrix_bd becomes part of the mask bias. The
        # incoming `mask` must already be an additive bias tensor
        # (float/bfloat16/half), not a boolean padding mask.
        assert mask.dtype != torch.bool
        mask = mask.unsqueeze(1)
        mask = (matrix_bd + mask) / math.sqrt(self.d_k)
        output = F.scaled_dot_product_attention(
            q_with_bias_u,
            k_full,
            v_full,
            attn_mask=mask,
            scale=1 / math.sqrt(self.d_k),
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(query.size(0), -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(output), new_cache


class MultiHeadedCrossAttention(MultiHeadedAttention):
    """Cross-attention variant of MultiHeadedAttention."""

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        super().__init__(
            n_head,
            n_feat,
            query_bias,
            key_bias,
            value_bias,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (
            torch.zeros((0, 0, 0, 0)),
            torch.zeros((0, 0, 0, 0)),
        ),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute cross-attention between decoder query and encoder memory."""

        del pos_emb

        key_cache, value_cache = cache
        assert key_cache.size(0) == value_cache.size(0)

        if key_cache.size(0) > 0:
            # During inference with cache, we reuse pre-computed keys/values.
            assert not self.training
            q = self._forward_linearx("query", query)
            k, v = key_cache, value_cache
            new_cache = cache
        else:
            q, k, v = self.forward_qkv(query, key, value)
            new_cache = (k, v) if not self.training else cache

        # For multi-query or multi-group attention, repeat KV heads.
        if self.h_kv != self.h and self.h_kv != 1:
            k = torch.repeat_interleave(
                k,
                self.h // self.h_kv,
                dim=-3,
            )
            v = torch.repeat_interleave(
                v,
                self.h // self.h_kv,
                dim=-3,
            )

        B = query.size(0)
        beams = 1
        if B != k.size(0):
            # Beam search case: batch is expanded.
            assert not self.training
            beams = B // k.size(0)
            B = k.size(0)
            q = q.view(B, beams, q.size(-3), q.size(-2), q.size(-1))
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            mask = mask.unsqueeze(1)

        # SDPA-only path. The `mask` tensor is expected to be an additive
        # bias (float/bfloat16/half), not a boolean padding mask.
        assert mask.dtype != torch.bool
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(1),
            scale=1 / math.sqrt(self.d_k),
        )
        output = output.transpose(-2, -3).contiguous()
        output_shape = output.size()[:-2] + torch.Size([self.h * self.d_k])
        output = output.view(output_shape)  # (batch, ..., time1, d_model)
        output = self.linear_out(output)

        if query.size(0) != B:
            # Fold beams back into batch.
            assert not self.training
            output_shape = torch.Size([B * beams]) + output.size()[2:]
            output = output.view(output_shape)

        return output, new_cache


class ShawRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-head attention with Shaw-style relative position embeddings.

    Reference: https://arxiv.org/pdf/1803.02155.pdf
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        # n_kv_head / head_dim not used here; use standard multi-head config.
        del n_kv_head, head_dim
        super().__init__(
            n_head,
            n_feat,
            query_bias,
            key_bias,
            value_bias,
            None,
            None,
        )

        # TODO: make these configurable if needed
        self.max_right_rel_pos = 8
        self.max_left_rel_pos = 64
        self.rel_k_embed = torch.nn.Embedding(
            self.max_left_rel_pos + self.max_right_rel_pos + 1,
            self.d_k,
        )

    def _relative_indices(self, keys: torch.Tensor) -> torch.Tensor:
        """Compute relative position indices."""

        # (S, 1)
        indices = torch.arange(keys.size(2), device=keys.device).unsqueeze(0)

        # (S, S)
        rel_indices = indices - indices.transpose(0, 1)

        rel_indices = torch.clamp(
            rel_indices,
            -self.max_left_rel_pos,
            self.max_right_rel_pos,
        )

        return rel_indices + self.max_left_rel_pos

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (
            torch.zeros((0, 0, 0, 0)),
            torch.zeros(0, 0, 0, 0),
        ),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute Shaw-style relative position attention."""

        del pos_emb

        q, k, v = self.forward_qkv(query, key, value)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        # rel_k: (t2, t2, d_k)
        rel_k = self.rel_k_embed(self._relative_indices(k))
        rel_k = rel_k[-q.size(2):]
        # rel_att_weights: (batch, head, time1, time2)
        rel_att_weights = torch.einsum("bhld,lrd->bhlr", q, rel_k)

        # SDPA-only path: rel_att_weights is used as bias in the mask. The
        # incoming `mask` must already be an additive bias tensor
        # (float/bfloat16/half), not a boolean padding mask.
        assert mask.dtype != torch.bool
        mask = mask.unsqueeze(1)
        mask = (rel_att_weights + mask) / math.sqrt(self.d_k)
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            scale=1 / math.sqrt(self.d_k),
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(query.size(0), -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(output), new_cache


class RopeMultiHeadedAttention(MultiHeadedAttention):
    """Multi-head attention with rotary position embeddings (RoPE)."""

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        style: str = "google",
    ):
        super().__init__(
            n_head,
            n_feat,
            query_bias,
            key_bias,
            value_bias,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )
        self.style = style
        self._apply_rotary_emb = get_apply_rotary_emb(style)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (
            torch.zeros((0, 0, 0, 0)),
            torch.zeros(0, 0, 0, 0),
        ),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute RoPE-scaled dot-product attention.

        Args:
            query: Query tensor of shape ``(batch, time1, size)``.
            key: Key tensor of shape ``(batch, time2, size)``.
            value: Value tensor of shape ``(batch, time2, size)``.
            mask: Attention mask, same semantics as in ``MultiHeadedAttention``.
            pos_emb: Rotary embedding tensor (precomputed frequencies) of
                shape matching the style.
            cache: KV cache for streaming decoding.
        """

        # Explicitly construct Q/K/V with `head_first=False` so that RoPE
        # can operate on (batch, time, heads, dim).
        q = self._forward_linearx("query", query, head_first=False)
        k = self._forward_linearx("key", key, head_first=False)
        v = self._forward_linearx("value", value, head_first=False)

        # Apply rotary embeddings via dedicated rotary_embedding module
        q = self._apply_rotary_emb(q, pos_emb)
        k = self._apply_rotary_emb(k, pos_emb)

        k, v, new_cache = self._update_kv_and_cache(
            k,
            v,
            cache,
            head_first=False,
        )

        # Convert to (batch, heads, time, dim) for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # SDPA-only path. The `mask` tensor is expected to be an additive
        # bias (float/bfloat16/half), not a boolean padding mask.
        assert mask.dtype != torch.bool
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(1),
            scale=1 / math.sqrt(self.d_k),
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(query.size(0), -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(output), new_cache


__all__ = [
    "MultiHeadedAttention",
    "RelPositionMultiHeadedAttention",
    "MultiHeadedCrossAttention",
    "ShawRelPositionMultiHeadedAttention",
    "RopeMultiHeadedAttention",
    "PagedKVCache",
]
