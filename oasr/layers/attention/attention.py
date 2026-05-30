# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer-style attention layer for OASR streaming inference.

A single inference-only attention class is provided:
:class:`RelPositionMultiHeadedAttention`. The actual attention compute
goes through :func:`oasr.attention.fmha` (CuteDSL on supported GPUs,
SDPA fallback otherwise) for two cache modes:

* offline (``cache=None``)
* paged streaming (``cache=PagedKVCache``)

The Transformer-XL rel-pos bias ``matrix_bd`` is combined with the
padding / per-stream length mask and passed in via ``attn_bias``.

The dense streaming path (``cache=(K_cached, V_cached)``) has been removed;
streaming is paged-only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from torch import nn

from oasr.cache.paged_kv import PagedKVCache

if TYPE_CHECKING:
    from oasr.models.conformer.packing import PackedLayout


# ---------------------------------------------------------------------------
# Conformer rel-pos multi-head attention
# ---------------------------------------------------------------------------


class RelPositionMultiHeadedAttention(nn.Module):
    """Self-attention with Transformer-XL relative-position bias.

    Reference: https://arxiv.org/abs/1901.02860.

    Inference-only, self-attention only.  Q/K/V come from one fused linear
    projection on the same input; the actual attention compute goes through
    :func:`oasr.fmha` and the ``matrix_bd`` rel-pos bias is combined with the
    padding / per-stream length mask and passed in as ``attn_bias``.

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
        Whether the corresponding section of the fused QKV projection uses a
        bias term.  If any of the three is ``True`` the fused linear carries
        a bias; sections whose flag was ``False`` are zero-initialised.
    n_kv_head : int, optional
        Number of K/V heads; defaults to ``n_head``. Requires ``head_dim``
        when set explicitly.
    head_dim : int, optional
        Per-head dimension; defaults to ``n_feat // n_head``.
    """

    # Bumped from the default (1) when QKV was fused.  Legacy state-dicts
    # (separate ``linear_q/k/v``) have ``version < 2`` and get migrated by
    # :meth:`_load_from_state_dict`.
    _version = 2

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

        # Output sizes used by the head-split below and by the load-time
        # legacy state-dict migration.
        self._qkv_split = (self.inner_dim, self.inner_kv_dim, self.inner_kv_dim)
        self._qkv_bias_flags = (query_bias, key_bias, value_bias)
        qkv_bias = any(self._qkv_bias_flags)

        self.linear_qkv = nn.Linear(n_feat, sum(self._qkv_split), bias=qkv_bias)
        self.linear_out = nn.Linear(self.inner_dim, n_feat, bias=query_bias)

        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    # ------------------------------------------------------------------
    # Checkpoint migration
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate legacy 3-projection checkpoints (``linear_q/k/v``) to fused."""
        version = local_metadata.get("version", 1)
        if version < 2:
            q_w = state_dict.pop(prefix + "linear_q.weight", None)
            k_w = state_dict.pop(prefix + "linear_k.weight", None)
            v_w = state_dict.pop(prefix + "linear_v.weight", None)
            if q_w is not None and k_w is not None and v_w is not None:
                state_dict[prefix + "linear_qkv.weight"] = torch.cat(
                    [q_w, k_w, v_w], dim=0)

                if self.linear_qkv.bias is not None:
                    biases = []
                    for legacy_name, dim in zip(
                        ("linear_q.bias", "linear_k.bias", "linear_v.bias"),
                        self._qkv_split,
                    ):
                        b = state_dict.pop(prefix + legacy_name, None)
                        biases.append(
                            b if b is not None
                            else torch.zeros(dim, dtype=q_w.dtype)
                        )
                    state_dict[prefix + "linear_qkv.bias"] = torch.cat(
                        biases, dim=0)
                else:
                    # Drop any legacy biases we don't carry anymore.
                    for legacy_name in (
                        "linear_q.bias", "linear_k.bias", "linear_v.bias",
                    ):
                        state_dict.pop(prefix + legacy_name, None)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = torch.zeros((0, 0, 0)),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: Union[PagedKVCache, None] = None,
        layout: "PackedLayout | None" = None,
    ) -> Tuple[torch.Tensor, Union[PagedKVCache, None]]:
        """Compute Conformer rel-pos self-attention.

        Three modes are supported:

        * **Offline** (``cache is None``, ``layout is None``).  Full-sequence
          self-attention; ``mask`` is the additive padding bias ``(B, 1, T)``.
        * **Paged streaming** (:class:`PagedKVCache`).  K/V for this chunk
          are written into the shared block pool; attention spans the full
          per-stream context ``[0, cache_seqlens[b] + T_q)``.
        * **Packed offline** (``layout`` set).  ``x`` is a single gapless
          packed row ``(1, T_total, n_feat)``; each segment is scattered into a
          batched grid and attends only to itself (dense per-segment fmha with
          ``cache_seqlens`` masking).

        Returns
        -------
        out : Tensor
            ``(B, T_q, n_feat)``.
        cache : same type as input
            Updated cache; ``None`` for the offline / packed paths, the same
            descriptor (pool was updated in place) for paged.
        """
        B, T_q, _ = x.shape

        # Single fused QKV GEMM.  The packed path scatters the *fused* qkv in
        # one shot (one index_copy instead of three), so branch before the
        # per-modality split.
        qkv = self.linear_qkv(x)

        if layout is not None:
            return self._forward_packed(qkv, pos_emb, layout)

        q, k, v = qkv.split(self._qkv_split, dim=-1)
        q = q.view(B, T_q, self.h,    self.d_k).transpose(1, 2)  # (B, H,    T_q, D)
        k = k.view(B, T_q, self.h_kv, self.d_k).transpose(1, 2)  # (B, H_kv, T_q, D)
        v = v.view(B, T_q, self.h_kv, self.d_k).transpose(1, 2)

        if isinstance(cache, PagedKVCache):
            return self._forward_paged(q, k, v, pos_emb, cache, B, T_q)

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

        # pos_bias_{u,v}: (H, d_k) → (H, 1, d_k) broadcasts over T_q without
        # round-tripping q through a (B, T_q, H, d_k) layout.
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)  # (B, H, T_q, d_k)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)

        matrix_bd = torch.matmul(q_with_bias_v, p.permute(0, 2, 3, 1))  # (B, H, T_q, T_kv_max)

        # 3. The fmha kernel reads paged K/V directly via the block table
        #    and enforces the per-stream length mask from ``cache_seqlens``
        #    (no host-side pad_bias add).  ``attn_bias`` is the pre-scaled
        #    Transformer-XL ``matrix_bd``; ``oasr.fmha`` rounds it up to the
        #    cute kernel's CTA tile internally.
        scale = 1.0 / math.sqrt(self.d_k)
        combined_bias = matrix_bd * scale  # (B, H, T_q, T_kv_max)
        total_kv_lens = cache.cache_seqlens + T_q  # (B,) on GPU

        # Local import to avoid a circular import oasr->layers->attention.
        from oasr.attention import fmha
        out = fmha(
            q_with_bias_u, cache.k_cache, cache.v_cache,
            softmax_scale=scale,
            attn_bias=combined_bias,
            cache_seqlens=total_kv_lens,
            block_table=cache.block_table,
        )  # (B, H, T_q, d_k)

        # 4. Output projection.
        out = out.transpose(1, 2).reshape(B, T_q, self.h * self.d_k)
        return self.linear_out(out), cache

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
        n_batch_pos = pos_emb.size(0)
        T_pos = pos_emb.size(1)
        p = self.linear_pos(pos_emb).view(n_batch_pos, T_pos, self.h, self.d_k)

        q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)  # (B, H, T_q, d_k)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)

        scale = 1.0 / math.sqrt(self.d_k)
        matrix_bd = torch.matmul(q_with_bias_v, p.permute(0, 2, 3, 1))  # (B, H, T_q, T_pos)
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

        out = out.transpose(1, 2).reshape(B, T_q, self.h * self.d_k)
        return self.linear_out(out), None

    def _forward_packed(
        self,
        qkv: torch.Tensor,
        pos_emb: torch.Tensor,
        layout: "PackedLayout",
    ) -> Tuple[torch.Tensor, None]:
        """Packed self-attention: each segment attends only to itself.

        ``qkv`` is the *fused* projection ``(1, T_total, Dq+Dk+Dv)`` over the
        gapless packed row.  Two backends, picked by ``layout.use_varlen``:

        * **varlen** (sequence-packing mode) — the gapless row feeds the cute
          varlen kernel directly via ``cu_seqlens`` with a packed
          block-diagonal rel-pos bias.  Zero attention padding.
        * **batched-dense** (length-bucketing mode) — the row is scattered into
          a ``(S, T_max_seg, ·)`` grid and a single dense ``fmha`` with
          per-segment ``cache_seqlens`` masks each segment to its own length.

        Both are bit-exact to ``B=1`` inference; the conv / FFN / norm work and
        the surrounding pack/unpack are identical either way.
        """
        assert qkv.size(0) == 1, "packed attention expects a single packed row (B=1)"
        if layout.use_varlen:
            return self._forward_packed_varlen(qkv, pos_emb, layout)
        return self._forward_packed_dense(qkv, pos_emb, layout)

    def _forward_packed_dense(
        self,
        qkv: torch.Tensor,
        pos_emb: torch.Tensor,
        layout: "PackedLayout",
    ) -> Tuple[torch.Tensor, None]:
        """Batched-per-segment dense attention (length-bucketing backend).

        The whole fused ``qkv`` is scattered into a batched ``(S, T_max_seg,
        Dqkv)`` grid in **one** ``index_copy_`` (vs. a per-modality scatter),
        then split into heads.  Length-sorted buckets keep the ``T_max_seg``
        padding small.  The rel-pos bias is one batched matmul against
        ``pos_emb[:, :T_max_seg]`` (positions begin at 0 per segment), and a
        single dense ``fmha`` with per-segment ``cache_seqlens`` masks each
        segment to its own length.  One ``index_select`` gathers valid rows
        back into the gapless packed row.  Two index ops per layer, no host
        syncs.
        """
        from oasr.attention import fmha

        scale = 1.0 / math.sqrt(self.d_k)
        S, Tm = layout.num_segs, layout.max_seg_len
        Dqkv = qkv.size(-1)

        # ONE scatter: gapless (T_total, Dqkv) → batched (S*Tm, Dqkv).
        qkv_b = qkv.new_zeros(S * Tm, Dqkv)
        qkv_b.index_copy_(0, layout.conv_batched_idx, qkv.squeeze(0))
        q, k, v = qkv_b.view(S, Tm, Dqkv).split(self._qkv_split, dim=-1)
        q = q.reshape(S, Tm, self.h,    self.d_k).transpose(1, 2)  # (S, H,    Tm, D)
        k = k.reshape(S, Tm, self.h_kv, self.d_k).transpose(1, 2)  # (S, H_kv, Tm, D)
        v = v.reshape(S, Tm, self.h_kv, self.d_k).transpose(1, 2)

        q_u = q + self.pos_bias_u.unsqueeze(1)                  # (S, H, Tm, D)
        q_v = q + self.pos_bias_v.unsqueeze(1)
        p = self.linear_pos(pos_emb[:, :Tm, :]).view(1, Tm, self.h, self.d_k)
        matrix_bd = torch.matmul(q_v, p.permute(0, 2, 3, 1)) * scale  # (S,H,Tm,Tm)

        out = fmha(
            q_u, k, v,
            softmax_scale=scale,
            attn_bias=matrix_bd,
            cache_seqlens=layout.seg_lengths,
        )                                                       # (S, H, Tm, D)

        # ONE gather: batched (S*Tm, H*D) → gapless (T_total, H*D).
        out = out.transpose(1, 2).reshape(S * Tm, self.h * self.d_k)
        out = out.index_select(0, layout.conv_batched_idx).unsqueeze(0)
        return self.linear_out(out), None

    def _forward_packed_varlen(
        self,
        qkv: torch.Tensor,
        pos_emb: torch.Tensor,
        layout: "PackedLayout",
    ) -> Tuple[torch.Tensor, None]:
        """Gapless varlen attention (sequence-packing backend).

        ``q_u``/``k``/``v`` are reshaped to packed ``(T_total, heads, d_k)`` and
        fed straight to the cute varlen kernel, which restricts each query to
        its own ``cu_seqlens`` segment — zero attention padding.  Only the
        block-diagonal rel-pos bias is assembled host-side: a batched
        per-segment matmul against ``pos_emb[:, :T_max_seg]`` then one gather
        (via ``layout.bias_gather_idx`` / ``bias_offsets``) into the flat packed
        buffer.  On non-cute archs ``fmha_varlen`` falls back to the per-segment
        SDPA reference.
        """
        from oasr.attention import fmha_varlen

        scale = 1.0 / math.sqrt(self.d_k)
        S, Tm = layout.num_segs, layout.max_seg_len
        T_total = qkv.size(1)

        q, k, v = qkv.split(self._qkv_split, dim=-1)
        q = q.view(1, T_total, self.h,    self.d_k).transpose(1, 2)  # (1, H,    T_total, D)
        k = k.view(1, T_total, self.h_kv, self.d_k).transpose(1, 2)
        v = v.view(1, T_total, self.h_kv, self.d_k).transpose(1, 2)

        # Packed (T_total, heads, d_k) — fed to the varlen kernel directly.
        q_u = (q + self.pos_bias_u.unsqueeze(1)).squeeze(0).transpose(0, 1)
        k_p = k.squeeze(0).transpose(0, 1).contiguous()
        v_p = v.squeeze(0).transpose(0, 1).contiguous()

        # Packed block-diagonal rel-pos bias: batched per-segment matmul then
        # gather each segment's valid (H, T_s, T_s) block into the flat buffer.
        q_v = q + self.pos_bias_v.unsqueeze(1)                  # (1, H, T_total, d_k)
        q_v_b = q_v.new_zeros(S * Tm, self.h * self.d_k)
        q_v_b.index_copy_(
            0, layout.conv_batched_idx,
            q_v.squeeze(0).transpose(0, 1).reshape(T_total, self.h * self.d_k),
        )
        q_v_b = q_v_b.view(S, Tm, self.h, self.d_k).permute(0, 2, 1, 3)
        p = self.linear_pos(pos_emb[:, :Tm, :]).view(1, Tm, self.h, self.d_k)
        matrix_bd = torch.matmul(q_v_b, p.permute(0, 2, 3, 1)) * scale  # (S,H,Tm,Tm)
        packed_bias = matrix_bd.reshape(-1).index_select(0, layout.bias_gather_idx)

        out = fmha_varlen(
            q_u.contiguous(), k_p, v_p,
            softmax_scale=scale,
            cu_seqlens_q=layout.cu_seqlens, cu_seqlens_k=layout.cu_seqlens,
            max_seqlen_q=Tm, max_seqlen_k=Tm,
            attn_bias=packed_bias, bias_offsets=layout.bias_offsets,
        )                                                       # (T_total, H, d_k)

        out = out.reshape(T_total, self.h * self.d_k).unsqueeze(0)
        return self.linear_out(out), None


__all__ = [
    "RelPositionMultiHeadedAttention",
]
