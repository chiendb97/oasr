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
from typing import Optional, Tuple, Union

import torch
from torch import nn

from oasr.cache.paged_kv import PagedKVCache


# Dense KV cache pair ``(K_cached, V_cached)`` used by the legacy
# ``forward_chunk`` path. Both tensors are head-first
# ``(B, n_kv_head, T_cached, head_dim)``. New chunks concatenate along the
# time axis to produce the next cache.
T_CACHE = Tuple[torch.Tensor, torch.Tensor]


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
        # 1. Write the new K/V into the paged pool. The cache descriptor
        #    owns the scatter logic.
        cache.write_kv_chunk(k, v, offset=cache.cache_seqlens)

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

        # 3. The fmha kernel reads paged K/V directly via the block table.
        #    Pass pool views unexpanded -- the kernel handles GQA head fan-out.
        #    The per-stream length mask is enforced inside the kernel via
        #    ``cache_seqlens``; no host-side pad_bias add is needed.
        scale = 1.0 / math.sqrt(self.d_k)
        total_kv_lens = cache.cache_seqlens + T_q  # (B,) on GPU
        # The bias passed to fmha must already be scaled to match the
        # post-softmax_scale logit semantics; the kernel adds bias *after*
        # the QK*scale.
        combined_bias = matrix_bd * scale  # (B, H, T_q, T_kv_max)
        # Pad T_kv_max up to a multiple of the kernel's N_BLOCK tile so that
        # the kernel's per-n-tile block_table reads stay in-bounds. The
        # kernel reads ``mBlockTable[b, n_block * (N_BLOCK/bs) + i]`` for
        # ``i in [0, N_BLOCK/bs)`` and ``n_block in [0, ceil(seqlen_k/N_BLOCK))``,
        # so block_table must have at least
        # ``ceil(seqlen_k/N_BLOCK) * (N_BLOCK/bs)`` columns. Rounding
        # T_kv_max up to a multiple of N_BLOCK satisfies that and keeps the
        # bias / block_table shapes consistent (kernel sees
        # ``t_kv_logical = block_table.shape[1] * bs``; bounds-checked bias
        # tolerates the padding).
        bs = cache.block_size
        N_BLOCK = 64  # matches FmhaForwardSm80._n_block_size default
        T_kv_padded = ((T_kv_max + N_BLOCK - 1) // N_BLOCK) * N_BLOCK
        if T_kv_padded > T_kv_max:
            pad_cols = T_kv_padded - T_kv_max
            tail = torch.zeros(
                B, self.h, T_q, pad_cols,
                dtype=combined_bias.dtype, device=combined_bias.device,
            )
            combined_bias = torch.cat([combined_bias, tail], dim=-1)

        block_table_view = cache.block_table
        # Caller may have allocated a wider block_table than needed; the
        # kernel walks the full T_kv_padded extent.
        max_blocks_needed = T_kv_padded // bs
        if block_table_view.size(1) > max_blocks_needed:
            block_table_view = block_table_view[:, :max_blocks_needed]

        # Local import to avoid a circular import oasr->layers->attention.
        from oasr.attention import fmha
        out = fmha(
            q_with_bias_u, cache.k_cache, cache.v_cache,
            softmax_scale=scale,
            attn_bias=combined_bias,
            cache_seqlens=total_kv_lens,
            block_table=block_table_view,
        )  # (B, H, T_q, d_k)

        # 4. Output projection.
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
        """Rel-pos attention over already-prepared Q/K/V.

        ``mask`` is the additive padding bias (``0`` valid, ``-inf``
        padding). Combined with ``matrix_bd`` and scaled by
        ``1/sqrt(d_k)``, then routed through :func:`oasr.fmha`
        which selects the SDPA fallback or the SM120 fused kernel based
        on ``OASR_ATTN_BACKEND`` and the active GPU. Algebra is identical
        to the legacy SDPA path: ``S = (Q @ K^T) * scale + (matrix_bd + mask) * scale``.
        """
        n_batch_pos = pos_emb.size(0)
        T_pos = pos_emb.size(1)
        p = self.linear_pos(pos_emb).view(n_batch_pos, T_pos, self.h, self.d_k)
        p = p.transpose(1, 2)  # (B_pos, H, T_pos, d_k)

        q_t = q.transpose(1, 2)
        q_with_bias_u = (q_t + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q_t + self.pos_bias_v).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_k)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # (B, H, T_q, T_pos)
        if mask.numel() > 0:
            combined_bias = (matrix_bd + mask.unsqueeze(1)) * scale
        else:
            combined_bias = matrix_bd * scale

        # GQA broadcast happens inside fmha (kernel handles head fan-out),
        # so we pass k/v unexpanded.
        # Local import to avoid a circular import via oasr -> layers -> attention.
        from oasr.attention import fmha
        out = fmha(
            q_with_bias_u, k, v,
            softmax_scale=scale,
            attn_bias=combined_bias,
        )  # (B, H, T_q, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.h * self.d_k)
        return self.linear_out(out)


__all__ = [
    "RelPositionMultiHeadedAttention",
    "T_CACHE",
]
