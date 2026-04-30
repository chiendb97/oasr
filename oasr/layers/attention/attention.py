# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer-style attention layer for OASR streaming inference.

A single inference-only attention class is provided:
:class:`RelPositionMultiHeadedAttention`. The actual attention compute
goes through PyTorch SDPA
(``torch.nn.functional.scaled_dot_product_attention``) for all three
cache modes:

* offline (``cache=None``)
* dense streaming (``cache=(K_cached, V_cached)``)
* paged streaming (``cache=PagedKVCache``)

The Transformer-XL rel-pos bias ``matrix_bd`` is combined with the
padding / per-stream length mask and passed in via ``attn_mask``.

A FlexAttention path was prototyped (the helpers ``build_paged_block_mask``,
``_flex_attention_paged_with_bias``, and ``warmup_flex_attention`` below
remain for future re-enablement). It produced numerical parity with SDPA
but was significantly slower on Conformer streaming shapes — the chunk
``T_q`` (8 frames after subsampling) is much smaller than FlexAttention's
``BLOCK_M=128`` Q-tile, so most of every tile was wasted. SDPA's
cuBLAS-sized tiles match our shapes directly. Re-enabling FlexAttention
will likely require packing queries across requests (variable-length
attention) so the packed Q length exceeds ``BLOCK_M``.

Removed in this revision (relative to the WeNet-port era):
    * ``MultiHeadedAttention`` / ``MultiHeadedCrossAttention`` /
      ``ShawRelPositionMultiHeadedAttention`` / ``RopeMultiHeadedAttention`` —
      none of them were used by the active Conformer encoder.
    * The ``if not self.training`` cache-update branch — inference-only.
    * The ``flash_attn_with_kvcache`` path — flash_attn 2.7 cannot express
      ``matrix_bd`` (no ``attn_bias`` argument).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask as _create_block_mask,
    flex_attention as _flex_attention,
)


# Compile the BlockMask constructor with ``dynamic=True`` so a single
# compiled callable adapts to varying ``(B, T_q, T_kv)`` across calls.
_create_block_mask_compiled = torch.compile(_create_block_mask, dynamic=True)


# FlexAttention itself is compiled via the wrapper in
# :func:`_flex_attention_paged_with_bias` rather than ``torch.compile``-ing
# ``flex_attention`` directly. The wrapper closure-captures the additive
# ``bias`` for ``score_mod``; placing the closure construction *inside*
# the compiled function keeps everything in a single Dynamo trace, which
# is the pattern PyTorch's FlexAttention API expects (otherwise
# ``flex_attention`` falls back to the unfused eager implementation).
@torch.compile(dynamic=True)
def _flex_attention_paged_with_bias(
    q: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    bias: torch.Tensor,
    block_mask: Optional[BlockMask],
    softmax_scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    """FlexAttention with a precomputed additive bias added via score_mod.

    Wrapped in ``torch.compile`` so ``flex_attention`` runs through the
    fused Triton kernel rather than the unfused fallback.
    """
    def score_mod(score, b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        return score + bias[b, h, q_idx, kv_idx]

    return _flex_attention(
        q, k_full, v_full,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=softmax_scale,
        enable_gqa=enable_gqa,
    )


# Dense KV cache pair ``(K_cached, V_cached)`` used by the legacy
# ``forward_chunk`` path. Both tensors are head-first
# ``(B, n_kv_head, T_cached, head_dim)``. New chunks concatenate along the
# time axis to produce the next cache.
T_CACHE = Tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Paged KV cache descriptor
# ---------------------------------------------------------------------------


@dataclass
class PagedKVCache:
    """Per-layer paged KV cache descriptor.

    All tensors except ``k_cache`` / ``v_cache`` are shared across encoder
    layers within the same forward call (one ``block_table`` and one
    ``cache_seqlens`` per stream batch).

    Attributes
    ----------
    k_cache, v_cache : Tensor
        ``(max_num_blocks, block_size, n_kv_head, head_dim)`` views into the
        block pool for one encoder layer.
    block_table : Tensor
        ``(B, max_blocks_per_seq)`` int32 — per-stream logical → physical
        block mapping.
    cache_seqlens : Tensor
        ``(B,)`` int32 — committed K/V frames in the pool **before** the
        current chunk's K/V write. Lives on the same device as ``k_cache``.
    block_size : int
        Frames per physical block (= ``block_size_frames`` in
        :class:`~oasr.cache.types.CacheConfig`).
    host_seqlen_max : int
        Host-side mirror of ``cache_seqlens.max().item()`` so the encoder
        can compute the kernel's max key-sequence length without a per-step
        D2H sync (the engine already tracks ``Request.offset`` on the host).
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    block_size: int
    host_seqlen_max: int = 0
    # When set (>=0), every stream in this batch shares this offset, so
    # the paged write can take the cheap scalar-offset fast path and the
    # per-stream length mask collapses to "no mask" (all streams have
    # ``T_kv_max`` valid frames). ``-1`` falls back to the per-stream
    # ``cache_seqlens`` tensor and a real ``BlockMask``.
    host_seqlen_homogeneous: int = -1
    # FlexAttention BlockMask, shared across encoder layers within one
    # ``forward_chunk_paged`` call. Built once by the encoder before the
    # layer loop (depends only on ``cache_seqlens``, ``T_q``, ``T_kv_max``)
    # and attached to every per-layer ``PagedKVCache`` so the attention
    # layer doesn't pay 12× the construction cost. ``None`` means full
    # attention with no masking (homogeneous-offset case).
    block_mask: Optional[Any] = None


# ---------------------------------------------------------------------------
# Paged-cache write / gather helpers
# ---------------------------------------------------------------------------


def _paged_write_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    offsets: Union[int, torch.Tensor],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> None:
    """Write new K/V frames into the paged pool.

    Parameters
    ----------
    k_cache, v_cache : Tensor
        Per-layer pool views ``(max_blocks, block_size, n_kv_head, head_dim)``.
    block_table : Tensor
        ``(B, max_blocks_per_seq)`` int32 — one row per stream.
    offsets : int or Tensor
        Logical write offset. ``int`` (homogeneous case): every stream
        writes at the same offset; uses a cheap row-slice fast path with
        no D2H sync. ``(B,)`` int Tensor (heterogeneous case): per-stream
        offsets dispatched via a vectorised scatter.
    new_k, new_v : Tensor
        ``(B, n_kv_head, T, head_dim)`` head-first new K/V to write.
    """
    B, _, T, _ = new_k.shape
    H_kv = new_k.size(1)
    D = new_k.size(3)
    block_size = k_cache.size(1)

    # Frame-major layout matching the pool's per-block tile.
    k_data = new_k.permute(0, 2, 1, 3).contiguous()  # (B, T, H_kv, D)
    v_data = new_v.permute(0, 2, 1, 3).contiguous()

    if isinstance(offsets, int):
        # Homogeneous fast path — same as the pre-cohort-relax code.
        blk_logical = offsets // block_size
        blk_offset = offsets % block_size
        if blk_offset + T <= block_size:
            phys_blks = block_table[:, blk_logical].long()  # (B,)
            k_cache[phys_blks, blk_offset: blk_offset + T] = k_data
            v_cache[phys_blks, blk_offset: blk_offset + T] = v_data
        else:
            first_n = block_size - blk_offset
            phys_blks = block_table[:, blk_logical].long()
            phys_blks_next = block_table[:, blk_logical + 1].long()
            k_cache[phys_blks, blk_offset:block_size] = k_data[:, :first_n]
            v_cache[phys_blks, blk_offset:block_size] = v_data[:, :first_n]
            k_cache[phys_blks_next, 0: T - first_n] = k_data[:, first_n:]
            v_cache[phys_blks_next, 0: T - first_n] = v_data[:, first_n:]
        return

    # Heterogeneous-offset scatter.
    arange_T = torch.arange(T, device=offsets.device, dtype=offsets.dtype)
    time_pos = offsets.unsqueeze(1) + arange_T.unsqueeze(0)  # (B, T)
    blk_logical_t = (time_pos // block_size).long()
    blk_offset_t = (time_pos % block_size).long()

    phys_blk = torch.gather(block_table.long(), dim=1, index=blk_logical_t)
    flat_idx = (phys_blk * block_size + blk_offset_t).view(-1)

    k_flat = k_cache.view(-1, H_kv, D)
    v_flat = v_cache.view(-1, H_kv, D)
    k_flat[flat_idx] = k_data.view(B * T, H_kv, D)
    v_flat[flat_idx] = v_data.view(B * T, H_kv, D)


def _paged_gather_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    max_total_kv: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather the first ``max_total_kv`` K/V frames for every stream.

    Streams whose actual valid length is < ``max_total_kv`` end up with
    stale tail data in the gathered tensor; the FlexAttention block-mask
    enforces per-stream length and discards it.

    Returns ``(k, v)`` both shaped ``(B, n_kv_head, max_total_kv, head_dim)``.
    """
    B = block_table.size(0)
    H_kv, D = k_cache.size(2), k_cache.size(3)
    if max_total_kv == 0:
        empty = torch.zeros(
            B, H_kv, 0, D, dtype=k_cache.dtype, device=k_cache.device
        )
        return empty, empty.clone()

    block_size = k_cache.size(1)
    num_blocks = (max_total_kv + block_size - 1) // block_size
    block_ids = block_table[:, :num_blocks].long()  # (B, num_blocks)

    k_gathered = k_cache[block_ids].reshape(
        B, num_blocks * block_size, H_kv, D
    )[:, :max_total_kv]
    v_gathered = v_cache[block_ids].reshape(
        B, num_blocks * block_size, H_kv, D
    )[:, :max_total_kv]
    return k_gathered.permute(0, 2, 1, 3), v_gathered.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Per-stream length → additive padding bias (SDPA paged / dense / offline)
# ---------------------------------------------------------------------------


def _length_to_pad_bias(
    total_kv_lens: torch.Tensor,
    T_kv_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a ``(B, 1, 1, T_kv_max)`` additive bias that masks each
    stream's attention to ``[0, total_kv_lens[b])`` (``-inf`` outside).
    """
    arange = torch.arange(T_kv_max, device=total_kv_lens.device)
    keep = arange.unsqueeze(0) < total_kv_lens.unsqueeze(1)  # (B, T_kv_max)
    bias = torch.where(keep, 0.0, float("-inf")).to(dtype)
    return bias.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T_kv_max)


# ---------------------------------------------------------------------------
# Paged FlexAttention BlockMask (kept for future re-enablement; not used
# by the active SDPA path).
# ---------------------------------------------------------------------------


def build_paged_block_mask(
    total_kv_lens: torch.Tensor,
    B: int,
    T_q: int,
    T_kv_max: int,
    device: torch.device,
) -> BlockMask:
    """Build a per-stream length BlockMask for FlexAttention paged forward.

    Stream ``b`` attends to ``kv_idx ∈ [0, total_kv_lens[b])`` only.
    Built once per encoder ``forward_chunk_paged`` call and shared
    across all encoder layers via ``PagedKVCache.block_mask``.

    Goes through the compiled ``create_block_mask`` so per-call Python
    overhead is amortised across all subsequent calls.
    """
    def mask_mod(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        del h, q_idx
        return kv_idx < total_kv_lens[b]

    return _create_block_mask_compiled(
        mask_mod, B=B, H=None, Q_LEN=T_q, KV_LEN=T_kv_max,
        device=device,
    )


# ---------------------------------------------------------------------------
# Conformer rel-pos multi-head attention
# ---------------------------------------------------------------------------


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-Head Attention with Transformer-XL relative-position bias.

    Reference: https://arxiv.org/abs/1901.02860.

    Inference-only. The actual attention compute uses SDPA; the
    Transformer-XL ``matrix_bd`` rel-pos bias is combined with the
    padding / per-stream length mask and passed in as ``attn_mask``.

    Supports MHA / MQA / GQA layouts:

    * ``n_kv_head=None``: standard multi-head (``n_kv_head = n_head``).
    * ``n_kv_head=1``: multi-query attention.
    * ``1 < n_kv_head < n_head``: grouped-query attention.

    Parameters
    ----------
    n_head : int
        Number of query/output attention heads.
    n_feat : int
        Model dimension (input / output size).
    query_bias, key_bias, value_bias : bool
        Whether the corresponding linear projection uses a bias term.
    n_kv_head : int, optional
        Number of K/V heads; defaults to ``n_head``. Requires ``head_dim``
        when set explicitly.
    head_dim : int, optional
        Per-head dimension; defaults to ``n_feat // n_head``.
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
        super().__init__()

        # Inner dimensions for Q/K/V projections.
        self.inner_dim = n_feat if head_dim is None else head_dim * n_head
        if n_kv_head is not None:
            assert head_dim is not None, (
                "head_dim must be set when n_kv_head is not None"
            )
            self.inner_kv_dim = head_dim * n_kv_head
        else:
            self.inner_kv_dim = self.inner_dim
            n_kv_head = n_head

        self.d_k = self.inner_dim // n_head
        assert self.d_k == self.inner_kv_dim // n_kv_head

        self.h = n_head
        self.h_kv = n_kv_head

        self.linear_q = nn.Linear(n_feat, self.inner_dim, bias=query_bias)
        self.linear_k = nn.Linear(n_feat, self.inner_kv_dim, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, self.inner_kv_dim, bias=value_bias)
        self.linear_out = nn.Linear(self.inner_dim, n_feat, bias=query_bias)

        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name == "query":
            x = self.linear_q(x)
            heads = self.h
        elif name == "key":
            x = self.linear_k(x)
            heads = self.h_kv
        else:
            assert name == "value"
            x = self.linear_v(x)
            heads = self.h_kv
        x = x.view(*x.shape[:-1], heads, self.d_k)
        return x.transpose(-3, -2)  # (..., heads, T, d_k)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.zeros((0, 0, 0)),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: Union[T_CACHE, PagedKVCache, None] = None,
    ) -> Tuple[torch.Tensor, Union[T_CACHE, PagedKVCache, None]]:
        """Compute Conformer rel-pos attention.

        Three cache modes are supported:

        * **Offline** (``cache is None`` or empty tuple). Standard
          attention over ``(query, key, value)``; ``mask`` is the additive
          padding bias of shape ``(B, 1, T)``.
        * **Paged streaming** (:class:`PagedKVCache`). K/V are written
          into the shared block pool and attended over the full per-stream
          context ``[0, cache_seqlens[b] + T_q)``.
        * **Dense streaming** (tuple of ``(K_cached, V_cached)`` head-first
          tensors). New K/V are concatenated with the cached pair and
          returned as the next cache. ``mask`` is an additive bias over the
          full attended range; same algebra as the legacy SDPA path.

        Returns
        -------
        out : Tensor
            ``(B, T_q, n_feat)``.
        cache : same type as input
            Updated cache; ``None`` for the offline path, the same descriptor
            (pool was updated in place) for paged, or a fresh
            ``(K_full, V_full)`` pair for dense.
        """
        B, T_q, _ = query.shape

        # 1. Q/K/V projections.
        q = self._project("query", query)  # (B, H,    T_q, d_k)
        k = self._project("key",   key)    # (B, H_kv, T_q, d_k)
        v = self._project("value", value)  # (B, H_kv, T_q, d_k)

        if isinstance(cache, PagedKVCache):
            return self._forward_paged(q, k, v, pos_emb, cache, B, T_q)

        # Tuple form: dense streaming when the cache tensors are populated,
        # otherwise (placeholder zeros) it's the offline / first-chunk path.
        if isinstance(cache, tuple) and cache[0].numel() > 0:
            return self._forward_dense(q, k, v, mask, pos_emb, cache, B, T_q)

        return self._forward_offline(q, k, v, mask, pos_emb, B, T_q)

    # ------------------------------------------------------------------
    # Mode-specific implementations
    # ------------------------------------------------------------------

    def _forward_paged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: torch.Tensor,
        cache: PagedKVCache,
        B: int,
        T_q: int,
    ) -> Tuple[torch.Tensor, PagedKVCache]:
        homogeneous = cache.host_seqlen_homogeneous >= 0

        # 1. Write the new K/V into the paged pool. Scalar-offset fast
        #    path when every stream in the batch shares an offset.
        if homogeneous:
            _paged_write_kv(
                cache.k_cache, cache.v_cache, cache.block_table,
                cache.host_seqlen_homogeneous, k, v,
            )
        else:
            _paged_write_kv(
                cache.k_cache, cache.v_cache, cache.block_table,
                cache.cache_seqlens, k, v,
            )

        # 2. Relative-position bias.  pos_emb: (B_pos, T_kv_max, n_feat)
        #    with T_kv_max == host_seqlen_max + T_q.
        n_batch_pos = pos_emb.size(0)
        T_kv_max = pos_emb.size(1)
        p = self.linear_pos(pos_emb).view(n_batch_pos, T_kv_max, self.h, self.d_k)
        p = p.transpose(1, 2)  # (B_pos, H, T_kv_max, d_k)

        q_t = q.transpose(1, 2)  # (B, T_q, H, d_k)
        q_with_bias_u = (q_t + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q_t + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # (B, H, T_q, T_kv_max)

        # 3. Gather K/V from the paged pool (includes the just-written chunk).
        k_full, v_full = _paged_gather_kv(
            cache.k_cache, cache.v_cache, cache.block_table, T_kv_max,
        )
        if self.h_kv != self.h and self.h_kv != 1:
            # Expand KV heads for grouped-query attention; SDPA broadcasts
            # H_kv=1 (MQA) automatically, but not generic H_kv groups.
            n_repeat = self.h // self.h_kv
            k_full = k_full.repeat_interleave(n_repeat, dim=1)
            v_full = v_full.repeat_interleave(n_repeat, dim=1)

        # 4. SDPA with combined attn_mask = (matrix_bd + pad_bias) / sqrt(d_k).
        #    Skip the pad_bias build entirely when offsets are homogeneous —
        #    every stream's valid kv length is exactly T_kv_max so the bias
        #    would be all zeros.
        if homogeneous:
            attn_mask = matrix_bd * (1.0 / math.sqrt(self.d_k))
        else:
            total_kv_lens = cache.cache_seqlens + T_q  # (B,) on GPU
            pad_bias = _length_to_pad_bias(total_kv_lens, T_kv_max, q.dtype)
            attn_mask = (matrix_bd + pad_bias) * (1.0 / math.sqrt(self.d_k))
        out = F.scaled_dot_product_attention(
            q_with_bias_u, k_full, v_full,
            attn_mask=attn_mask,
            scale=1.0 / math.sqrt(self.d_k),
        )  # (B, H, T_q, d_k)

        # 5. Output projection.
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.h * self.d_k)
        return self.linear_out(out), cache

    def _forward_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        cache: T_CACHE,
        B: int,
        T_q: int,
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Dense streaming — concatenate with cached K/V, then attend."""
        k_old, v_old = cache
        k_full = torch.cat([k_old, k], dim=-2)  # (B, H_kv, T_old + T_q, d_k)
        v_full = torch.cat([v_old, v], dim=-2)
        out = self._attend_with_relpos_bias(q, k_full, v_full, mask, pos_emb, B, T_q)
        return out, (k_full, v_full)

    def _forward_offline(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        B: int,
        T_q: int,
    ) -> Tuple[torch.Tensor, None]:
        """Full-sequence attention with padding bias and rel-pos bias."""
        out = self._attend_with_relpos_bias(q, k, v, mask, pos_emb, B, T_q)
        return out, None

    def _attend_with_relpos_bias(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        B: int,
        T_q: int,
    ) -> torch.Tensor:
        """Rel-pos attention over already-prepared Q/K/V via SDPA.

        ``mask`` is the additive padding bias (``0`` valid, ``-inf``
        padding). Combined with ``matrix_bd`` and scaled by
        ``1/sqrt(d_k)``, then passed to SDPA as ``attn_mask`` — same
        algebra as the original WeNet RelPos path.
        """
        n_batch_pos = pos_emb.size(0)
        T_pos = pos_emb.size(1)
        p = self.linear_pos(pos_emb).view(n_batch_pos, T_pos, self.h, self.d_k)
        p = p.transpose(1, 2)  # (B_pos, H, T_pos, d_k)

        q_t = q.transpose(1, 2)
        q_with_bias_u = (q_t + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q_t + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # (B, H, T_q, T_pos)
        if mask.numel() > 0:
            attn_mask = (matrix_bd + mask.unsqueeze(1)) * (1.0 / math.sqrt(self.d_k))
        else:
            attn_mask = matrix_bd * (1.0 / math.sqrt(self.d_k))

        if self.h_kv != self.h and self.h_kv != 1:
            n_repeat = self.h // self.h_kv
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q_with_bias_u, k, v,
            attn_mask=attn_mask,
            scale=1.0 / math.sqrt(self.d_k),
        )  # (B, H, T_q, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.h * self.d_k)
        return self.linear_out(out)


def warmup_flex_attention(
    *,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    max_batch_size: int,
    chunk_size: int,
    max_attention_key_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Trigger ``torch.compile`` for the FlexAttention kernel and the
    BlockMask constructor on representative shapes.

    Run once at engine init so the first benchmark / request doesn't
    pay the compilation tax. With ``dynamic=True`` on both compiled
    callables, a single warmup amortises across all subsequent shape
    combinations the engine produces.

    Walks through both branches:
    * **homogeneous** — ``block_mask=None`` (the common case where every
      streaming request in a step shares an offset).
    * **heterogeneous** — a real BlockMask built from per-stream
      ``cache_seqlens`` (cohort-relaxed admission, varying offsets).
    """
    B = max_batch_size
    T_q = chunk_size
    T_kv = max_attention_key_size

    q = torch.randn(B, n_head, T_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, n_kv_head, T_kv, head_dim, device=device, dtype=dtype)
    v = torch.randn(B, n_kv_head, T_kv, head_dim, device=device, dtype=dtype)
    bias = torch.zeros(B, n_head, T_q, T_kv, device=device, dtype=dtype)

    enable_gqa = n_kv_head != n_head

    # Homogeneous branch — no BlockMask, full attention.
    _flex_attention_paged_with_bias(
        q, k, v, bias, block_mask=None,
        softmax_scale=1.0 / math.sqrt(head_dim),
        enable_gqa=enable_gqa,
    )

    # Heterogeneous branch — real BlockMask with varying lengths so the
    # mask_mod path actually traces (cache_seqlens=0 here means stream b
    # attends only to its own freshly-written T_q frames; trivial but
    # non-uniform across the batch).
    base_lens = torch.full((B,), 0, device=device, dtype=torch.int32)
    base_lens[: B // 2] = max(0, T_kv - T_q)  # half full, half empty
    block_mask = build_paged_block_mask(
        base_lens + T_q, B=B, T_q=T_q, T_kv_max=T_kv, device=device,
    )
    _flex_attention_paged_with_bias(
        q, k, v, bias, block_mask=block_mask,
        softmax_scale=1.0 / math.sqrt(head_dim),
        enable_gqa=enable_gqa,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()


__all__ = [
    "PagedKVCache",
    "RelPositionMultiHeadedAttention",
    "T_CACHE",
    "build_paged_block_mask",
    "warmup_flex_attention",
]
