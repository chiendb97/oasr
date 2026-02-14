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
from typing import Optional, Tuple

import torch
from torch import nn

from oasr.layers.rotary_embedding import get_apply_rotary_emb

T_CACHE = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

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
        dropout_rate: Dropout rate applied to attention weights.
        query_bias: Whether to include bias in the query projection.
        key_bias: Whether to include bias in the key projection.
        value_bias: Whether to include bias in the value projection.
        use_sdpa: If True, use ``torch.nn.functional.scaled_dot_product_attention``
            when available (PyTorch 2.0+).
        n_kv_head: Number of key/value heads (for MQA/GQA). If ``None``,
            defaults to ``n_head`` (standard multi-head).
        head_dim: Per-head dimension. If ``None``, inferred as
            ``n_feat // n_head``.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        use_sdpa: bool = False,
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
        self.dropout = nn.Dropout(p=dropout_rate)

        self.use_sdpa = use_sdpa
        self.dropout_rate = dropout_rate

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

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, ..., head, time1, d_k)
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
        cache: T_CACHE = (
            torch.zeros(0, 0, 0, 0),
            torch.zeros(0, 0, 0, 0),
        ),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape ``(batch, time1, size)``.
            key: Key tensor of shape ``(batch, time2, size)``.
            value: Value tensor of shape ``(batch, time2, size)``.
            mask: Mask tensor of shape ``(batch, 1, time2)`` or
                ``(batch, time1, time2)``.
            pos_emb: Positional embedding (unused in base class).
            cache: KV cache used for streaming decoding.

        Returns:
            A tuple of:

            - Output tensor ``(batch, time1, d_model)``.
            - Updated cache tensor pair.
        """

        del pos_emb  # Not used in standard attention

        q, k, v = self.forward_qkv(query, key, value)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache

        # SDPA path
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(1),
            dropout_p=self.dropout_rate if self.training else 0.0,
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
        dropout_rate: float,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        use_sdpa: bool = False,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        """Construct a RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head,
            n_feat,
            dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
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
            ones = torch.ones((x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (
            torch.zeros((0, 0, 0, 0)),
            torch.zeros((0, 0, 0, 0)),
        ),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """Compute scaled dot-product attention with relative positional encoding."""

        q, k, v = self.forward_qkv(query, key, value)
        # q: (batch, head, time1, d_k)
        q = q.transpose(1, 2)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # Compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        if not self.use_sdpa:
            # Compute matrix a and c as in the paper
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache

        # SDPA path: matrix_bd becomes part of the mask bias
        assert mask.dtype != torch.bool
        mask = mask.unsqueeze(1)
        mask = (matrix_bd + mask) / math.sqrt(self.d_k)
        output = torch.nn.functional.scaled_dot_product_attention(
            q_with_bias_u,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
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
        dropout_rate: float,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        use_sdpa: bool = False,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        super().__init__(
            n_head,
            n_feat,
            dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
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

        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            output = self.forward_attention(v, scores, mask)
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.unsqueeze(1),
                dropout_p=self.dropout_rate if self.training else 0.0,
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
        dropout_rate: float,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        use_sdpa: bool = False,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        # n_kv_head / head_dim not used here; use standard multi-head config.
        del n_kv_head, head_dim
        super().__init__(
            n_head,
            n_feat,
            dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
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
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
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
        rel_k = rel_k[-q.size(2) :]
        # rel_att_weights: (batch, head, time1, time2)
        rel_att_weights = torch.einsum("bhld,lrd->bhlr", q, rel_k)

        if not self.use_sdpa:
            scores = (torch.matmul(q, k.transpose(-2, -1)) + rel_att_weights) / math.sqrt(
                self.d_k
            )
            return self.forward_attention(v, scores, mask), new_cache

        # SDPA path: rel_att_weights is used as bias in the mask.
        assert mask.dtype != torch.bool
        mask = mask.unsqueeze(1)
        mask = (rel_att_weights + mask) / math.sqrt(self.d_k)
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
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
        dropout_rate: float,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        use_sdpa: bool = False,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        style: str = "google",
    ):
        super().__init__(
            n_head,
            n_feat,
            dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
        )
        self.style = style
        self._apply_rotary_emb = get_apply_rotary_emb(style)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
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

        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(1),
            dropout_p=self.dropout_rate if self.training else 0.0,
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
]

