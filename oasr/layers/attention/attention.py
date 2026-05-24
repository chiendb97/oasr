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
from typing import Optional, Tuple, Union

import torch
from torch import nn

from oasr.cache.paged_kv import PagedKVCache


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
    ) -> Tuple[torch.Tensor, Union[PagedKVCache, None]]:
        """Compute Conformer rel-pos self-attention.

        Two cache modes are supported:

        * **Offline** (``cache is None``).  Full-sequence self-attention;
          ``mask`` is the additive padding bias of shape ``(B, 1, T)``.
        * **Paged streaming** (:class:`PagedKVCache`).  K/V for this chunk
          are written into the shared block pool; attention spans the full
          per-stream context ``[0, cache_seqlens[b] + T_q)``.

        Returns
        -------
        out : Tensor
            ``(B, T_q, n_feat)``.
        cache : same type as input
            Updated cache; ``None`` for the offline path, the same descriptor
            (pool was updated in place) for paged.
        """
        B, T_q, _ = x.shape

        # Single fused QKV GEMM → split into per-modality head-major tensors.
        qkv = self.linear_qkv(x)
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


__all__ = [
    "RelPositionMultiHeadedAttention",
]
