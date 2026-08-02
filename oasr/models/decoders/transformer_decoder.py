# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transformer decoder for attention-based (AED) decode families.

WeNet/ESPnet-compatible transformer decoder used two ways:

* **U2++ attention rescoring** (Phase 2b) — one *teacher-forced* batched
  :meth:`TransformerDecoder.forward` over the CTC n-best; no autoregression.
* **AR generation** (Phase 2c, AED/Whisper-style) — incremental
  :meth:`TransformerDecoder.forward_one_step` driven by the ``aed`` decode
  strategy with a decoder-side KV cache.

Module/parameter names mirror the WeNet checkpoint layout exactly
(``embed.0.weight``, ``decoders.N.self_attn.linear_q.*``, ``after_norm``,
``output_layer``) so U2++ ``decoder.*`` weights load 1:1 with no name mapping.
:class:`BiTransformerDecoder` composes a left-to-right decoder with an optional
right-to-left branch (``r_num_blocks > 0``) for the U2++ reverse-scoring pass.

Projections, norms, the embedding and the FFN go through ``oasr.layers`` (the
waist); attention runs on :class:`oasr.layers.Attention`, which picks the fused
kernel or SDPA per mask shape.  The math matches WeNet's explicit
softmax/masked_fill within fp tolerance (verified by
``tests/test_transformer_decoder.py`` against the upstream reference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from oasr.layers import (
    TORCH_EPS,
    Attention,
    ColumnParallelLinear,
    Embedding,
    FeedForward,
    LayerNorm,
    Linear,
    RowParallelLinear,
)

from .base import BaseDecoder, DecoderState

__all__ = [
    "TransformerDecoderConfig",
    "TransformerDecoder",
    "BiTransformerDecoder",
    "add_sos_eos",
    "reverse_pad_list",
]


@dataclass
class TransformerDecoderConfig:
    """Hyperparameters for a (bi-)transformer AED decoder.

    ``vocab_size`` is the **raw** (unpadded) vocabulary — the decoder's
    embedding/output layers are plain torch linears, not the 8-aligned CTC GEMM
    kernel, so no padding is applied.  ``r_num_blocks > 0`` selects the U2++
    bitransformer (an extra right-to-left decoder scored with
    ``reverse_weight``); ``0`` is a conventional left-to-right decoder.
    ``sos_id`` / ``eos_id`` / ``reverse_weight`` travel here because they are
    properties of the trained decoder, not engine choices.
    """

    vocab_size: int = 0
    encoder_output_size: int = 256
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 6
    r_num_blocks: int = 0
    sos_id: int = -1
    eos_id: int = -1
    reverse_weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "encoder_output_size": self.encoder_output_size,
            "attention_heads": self.attention_heads,
            "linear_units": self.linear_units,
            "num_blocks": self.num_blocks,
            "r_num_blocks": self.r_num_blocks,
            "sos_id": self.sos_id,
            "eos_id": self.eos_id,
            "reverse_weight": self.reverse_weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformerDecoderConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ----------------------------------------------------------------------------
# Input helpers (WeNet ``add_sos_eos`` / ``reverse_pad_list`` semantics)
# ----------------------------------------------------------------------------


def add_sos_eos(
    ys_pad: torch.Tensor, sos: int, eos: int, ignore_id: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepend ``sos`` / append ``eos`` to a padded label batch.

    Returns ``(ys_in, ys_out)``: ``ys_in`` is ``[sos] + ys`` with padding
    positions replaced by ``eos`` (safe embedding input); ``ys_out`` is
    ``ys + [eos]`` with ``ignore_id`` padding (loss/score target layout).
    Matches WeNet's ``add_sos_eos`` exactly.
    """
    B, L = ys_pad.shape
    device = ys_pad.device
    pad = ys_pad.eq(ignore_id)  # (B, L)
    lens = (~pad).sum(dim=1)  # (B,)

    ys_in = torch.full((B, L + 1), eos, dtype=ys_pad.dtype, device=device)
    ys_in[:, 0] = sos
    ys_in[:, 1:] = ys_pad.masked_fill(pad, eos)

    ys_out = torch.full((B, L + 1), ignore_id, dtype=ys_pad.dtype, device=device)
    ys_out[:, :L] = ys_pad
    ys_out[torch.arange(B, device=device), lens] = eos
    return ys_in, ys_out


def reverse_pad_list(
    ys_pad: torch.Tensor, ys_lens: torch.Tensor, pad_value: int = -1
) -> torch.Tensor:
    """Reverse each row's valid prefix, keeping right padding in place.

    ``[[1,2,3],[4,5,P]] → [[3,2,1],[5,4,P]]`` — the input to the U2++
    right-to-left decoder.  Matches WeNet's ``reverse_pad_list``.
    """
    B, L = ys_pad.shape
    device = ys_pad.device
    idx = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
    lens = ys_lens.to(device).unsqueeze(1)  # (B, 1)
    rev_idx = (lens - 1 - idx).clamp(min=0)  # position of the mirrored token
    out = ys_pad.gather(1, rev_idx)
    return out.masked_fill(idx >= lens, pad_value)


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """``x * sqrt(d_model) + PE`` (WeNet ``PositionalEncoding``, eval mode).

    The table is a non-persistent buffer (recomputed on construction, never
    serialized) and grows lazily if a sequence exceeds ``max_len``.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.register_buffer("pe", self._build_table(max_len), persistent=False)

    def _build_table(self, length: int) -> torch.Tensor:
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        end = offset + x.size(1)
        if end > self.pe.size(1):
            self.pe = self._build_table(end).to(device=x.device, dtype=self.pe.dtype)
        return x * self.xscale + self.pe[:, offset:end].to(dtype=x.dtype, device=x.device)


class _DecoderAttention(nn.Module):
    """Multi-head attention with WeNet parameter names (``linear_q/k/v/out``).

    Projections keep the checkpoint's names; the compute is the shared
    :class:`oasr.layers.Attention`.
    """

    def __init__(self, n_head: int, d_model: int) -> None:
        super().__init__()
        assert d_model % n_head == 0, f"d_model={d_model} not divisible by heads={n_head}"
        self.h = n_head
        self.d_k = d_model // n_head
        self.linear_q = ColumnParallelLinear(d_model, d_model)
        self.linear_k = ColumnParallelLinear(d_model, d_model)
        self.linear_v = ColumnParallelLinear(d_model, d_model)
        self.linear_out = RowParallelLinear(d_model, d_model)
        self.attn = Attention(n_head, self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``mask``: bool ``(B, L_q or 1, L_k)`` — True = attend; None = full.

        ``kv_lens`` ``(B,)`` says the same thing for the cross-attention case
        (keys ``[0, len)`` valid) in the form the fused kernel can enforce
        without a materialized mask, so pass that instead where it applies.
        """
        q = self.attn.split_heads(self.linear_q(query))
        k = self.attn.split_kv_heads(self.linear_k(key))
        v = self.attn.split_kv_heads(self.linear_v(value))
        attn_mask = mask.unsqueeze(1) if mask is not None else None  # (B, 1, L_q, L_k)
        x = self.attn(q, k, v, attn_mask=attn_mask, kv_lens=kv_lens)
        return self.linear_out(self.attn.merge_heads(x))


class DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer (self-attn → cross-attn → FFN)."""

    def __init__(self, d_model: int, n_head: int, linear_units: int) -> None:
        super().__init__()
        self.self_attn = _DecoderAttention(n_head, d_model)
        self.src_attn = _DecoderAttention(n_head, d_model)
        # WeNet's ``PositionwiseFeedForward``: w_1 → ReLU → w_2.  The names are
        # the checkpoint's; ReLU folds into the GEMM epilogue on the CUDA path.
        self.feed_forward = FeedForward(
            d_model, linear_units, activation="relu", names=("w_1", "w_2")
        )
        self.norm1 = LayerNorm(d_model, eps=TORCH_EPS)
        self.norm2 = LayerNorm(d_model, eps=TORCH_EPS)
        self.norm3 = LayerNorm(d_model, eps=TORCH_EPS)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory: torch.Tensor,
        memory_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, x, x, tgt_mask)
        residual = x
        x = self.norm2(x)
        x = residual + self.src_attn(x, memory, memory, kv_lens=memory_lens)
        residual = x
        x = self.norm3(x)
        return residual + self.feed_forward(x)


# ----------------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------------


class TransformerDecoder(nn.Module):
    """Left-to-right transformer decoder (one direction of the bitransformer).

    Teacher-forced :meth:`forward` scores a whole padded label batch in one
    pass (rescoring); :meth:`forward_one_step` is the incremental AR entry.
    """

    def __init__(self, config: TransformerDecoderConfig, num_blocks: Optional[int] = None) -> None:
        super().__init__()
        d_model = config.encoder_output_size
        blocks = config.num_blocks if num_blocks is None else num_blocks
        # nn.Sequential only for the checkpoint key layout (``embed.0.weight``);
        # forward() indexes the parts explicitly.
        self.embed = nn.Sequential(
            Embedding(config.vocab_size, d_model),
            SinusoidalPositionalEncoding(d_model),
        )
        self.decoders = nn.ModuleList(
            [
                DecoderLayer(d_model, config.attention_heads, config.linear_units)
                for _ in range(blocks)
            ]
        )
        self.after_norm = LayerNorm(d_model, eps=TORCH_EPS)
        self.output_layer = Linear(d_model, config.vocab_size)

    def forward(
        self,
        memory: torch.Tensor,
        memory_lens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced decode → logits ``(B, L, V)``.

        Parameters
        ----------
        memory : Tensor
            Encoder output ``(B, T_enc, D)``.
        memory_lens : Tensor
            ``(B,)`` valid encoder lengths (cross-attention key padding).
        ys_in_pad : Tensor
            ``(B, L)`` int64 decoder input (``[sos] + hyp``, eos-padded).
        ys_in_lens : Tensor
            ``(B,)`` valid lengths of ``ys_in_pad`` (hyp length + 1).
        """
        B, L = ys_in_pad.shape
        device = ys_in_pad.device
        pos = torch.arange(L, device=device)
        # (B, L, L): causal AND key-not-padded.  Padded *query* rows keep valid
        # keys (position 0 is always sos), so no all-masked softmax rows.
        tgt_mask = (pos.unsqueeze(0) <= pos.unsqueeze(1)).unsqueeze(0) & (
            pos.unsqueeze(0) < ys_in_lens.to(device).unsqueeze(1)
        ).unsqueeze(1)
        memory_lens = memory_lens.to(device)

        x = self.embed[1](self.embed[0](ys_in_pad))
        for layer in self.decoders:
            x = layer(x, tgt_mask, memory, memory_lens)
        x = self.after_norm(x)
        return self.output_layer(x)

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_lens: torch.Tensor,
        tokens: torch.Tensor,
        offset: int,
        caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """One AR step over the last token → ``(logits (B, V), new_caches)``.

        ``caches`` holds one ``(B, offset, D)`` tensor of pre-norm layer inputs
        per layer (WeNet ``forward_one_step`` cache layout: keys/values are
        recomputed from the cached hidden, keeping the module KV-format-free —
        the paged-KV fast path lands with the ``DecoderKVCacheManager``
        integration).
        """
        device = tokens.device
        memory_lens = memory_lens.to(device)

        x = self.embed[1](self.embed[0](tokens.unsqueeze(1)), offset=offset)  # (B, 1, D)
        new_caches = []
        for i, layer in enumerate(self.decoders):
            full = x if caches is None else torch.cat([caches[i], x], dim=1)
            new_caches.append(full)
            residual = x
            q = layer.norm1(x)
            kv = layer.norm1(full)
            x = residual + layer.self_attn(q, kv, kv, None)
            residual = x
            x = layer.norm2(x)
            x = residual + layer.src_attn(x, memory, memory, kv_lens=memory_lens)
            residual = x
            x = layer.norm3(x)
            x = residual + layer.feed_forward(x)
        x = self.after_norm(x[:, -1])
        return self.output_layer(x), new_caches


class BiTransformerDecoder(BaseDecoder):
    """U2++ bitransformer: left-to-right decoder + optional right-to-left branch.

    Registered on :class:`~oasr.models.conformer.ConformerModel` as
    ``self.decoder`` so WeNet ``decoder.left_decoder.*`` /
    ``decoder.right_decoder.*`` keys map 1:1.  Driven teacher-forced by the
    ``ctc_aed_rescoring`` strategy; the AR path (Phase 2c) uses
    ``left_decoder.forward_one_step``.
    """

    decode_type = "aed"

    def __init__(self, config: TransformerDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.left_decoder = TransformerDecoder(config)
        self.right_decoder = (
            TransformerDecoder(config, num_blocks=config.r_num_blocks)
            if config.r_num_blocks > 0
            else None
        )

    @property
    def has_reverse(self) -> bool:
        return self.right_decoder is not None

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> DecoderState:
        """Rescoring is stateless (one teacher-forced pass); AR state arrives
        with the Phase 2c incremental protocol."""
        del batch_size, device, dtype
        return None

    def forward(
        self,
        memory: torch.Tensor,
        memory_lens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Teacher-forced both directions → ``(l_logits, r_logits | None)``."""
        l_x = self.left_decoder(memory, memory_lens, ys_in_pad, ys_in_lens)
        r_x = None
        if self.right_decoder is not None and r_ys_in_pad is not None:
            r_x = self.right_decoder(memory, memory_lens, r_ys_in_pad, ys_in_lens)
        return l_x, r_x
