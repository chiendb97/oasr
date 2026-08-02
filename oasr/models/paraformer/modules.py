# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SANM building blocks shared by the Paraformer encoder and NAR decoder.

Faithful ports of FunASR's ``funasr.models.sanm`` modules (inference paths
only; dropout layers are identity at eval time and are omitted).  Parameter
names and shapes match the FunASR checkpoints 1:1 so ``load_weights`` needs no
name mapping:

* :class:`SanmSelfAttention`   — ``MultiHeadedAttentionSANM`` (fused
  ``linear_q_k_v`` + FSMN memory block added to the attention output);
* :class:`FsmnBlock`           — ``MultiHeadedAttentionSANMDecoder`` (the NAR
  decoder's "self-attention" is a depthwise-conv FSMN memory block only);
* :class:`SanmCrossAttention`  — ``MultiHeadedAttentionCrossAtt`` (fused
  ``linear_k_v`` over the encoder memory);
* :class:`EncoderFeedForward`  — ``PositionwiseFeedForward`` (w_1/ReLU/w_2);
* :class:`DecoderFeedForward`  — ``PositionwiseFeedForwardDecoderSANM``
  (w_1/ReLU/LayerNorm/w_2, ``w_2`` bias-free);
* :func:`sinusoidal_position_encoding` — FunASR's ``SinusoidalPositionEncoder``
  (positions start at 1, ``[sin | cos]`` concatenation).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from oasr.layers import (
    ESPNET_EPS,
    Attention,
    ColumnParallelLinear,
    FeedForward,
    RowParallelLinear,
)

#: FunASR (ESPnet) LayerNorm epsilon — NOT PyTorch's 1e-5 default.  The CIF
#: acoustic embeddings have tiny per-row variance, where the eps choice shifts
#: normalized values by whole percent.  The value lives in ``oasr.layers.norm``
#: with the other ecosystem conventions; this alias keeps the local spelling.
LAYER_NORM_EPS = ESPNET_EPS


#: Cache of computed PE tables, keyed by ``(depth, device_str)``.  Values are
#: fp32 ``(1, L, depth)`` tensors, grown (never shrunk) as longer audio arrives.
_PE_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}
#: Rounding for the cached PE length, so a few distinct audio lengths do not
#: each trigger a rebuild.
_PE_GROWTH = 512


def _build_pe_fp32(length: int, depth: int, device: torch.device) -> torch.Tensor:
    """The FunASR PE table, always computed in fp32."""
    positions = torch.arange(1, length + 1, device=device, dtype=torch.float32)
    half = depth // 2
    log_timescale_increment = math.log(10000.0) / (depth / 2 - 1)
    inv_timescales = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32) * -log_timescale_increment
    )
    scaled_time = positions.unsqueeze(1) * inv_timescales.unsqueeze(0)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).unsqueeze(0)


def sinusoidal_position_encoding(
    length: int, depth: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """FunASR sinusoidal PE: ``(1, length, depth)``, positions ``1..length``.

    ``pe[:, t] = [sin(t·inv), cos(t·inv)]`` with ``inv_i = exp(-i·ln(10000) /
    (depth/2 - 1))`` — note the concatenated (not interleaved) sin/cos halves.

    Two things this fixes over recomputing inline per forward (N1):

    * **Built in fp32, cast on the way out.**  Under fp16 the integer position
      ladder ``arange(1, L+1)`` is exact only to 2048; past that consecutive
      positions start colliding, and every trig value derived from them is
      wrong.  FunASR computes in fp32, so matching it is also a parity
      requirement, not only a precision nicety.
    * **Cached per (depth, device).**  It is a pure function of those plus the
      length, and it was rebuilt — arange, exp, outer product, two trig passes,
      a cat — on *every* encoder forward.
    """
    key = (int(depth), str(device))
    cached = _PE_CACHE.get(key)
    if cached is None or cached.size(1) < length:
        grown = ((length + _PE_GROWTH - 1) // _PE_GROWTH) * _PE_GROWTH
        cached = _build_pe_fp32(max(grown, length), depth, device)
        _PE_CACHE[key] = cached
    return cached[:, :length].to(dtype=dtype)


class FsmnBlock(nn.Module):
    """Depthwise-conv FSMN memory block (``MultiHeadedAttentionSANMDecoder``).

    ``x`` is masked, convolved with symmetric zero padding (shifted left by
    ``sanm_shift`` when positive), residual-added, and re-masked.
    """

    def __init__(self, n_feat: int, kernel_size: int, sanm_shift: int = 0) -> None:
        super().__init__()
        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``inputs (B, T, D)``, ``mask (B, T, 1)`` float → ``(B, T, D)``."""
        inputs = inputs * mask
        x = self.pad_fn(inputs.transpose(1, 2))
        x = self.fsmn_block(x).transpose(1, 2)
        x = x + inputs
        return x * mask


class SanmSelfAttention(nn.Module):
    """SANM self-attention: multi-head attention + FSMN memory over ``v``."""

    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        kernel_size: int,
        sanm_shift: int = 0,
    ) -> None:
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q_k_v = ColumnParallelLinear(in_feat, n_feat * 3)
        self.linear_out = RowParallelLinear(n_feat, n_feat)
        # ``q`` is pre-scaled below (FunASR's convention), hence scale 1.0.
        self.attn = Attention(n_head, self.d_k, softmax_scale=1.0)
        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def _forward_fsmn(self, v: torch.Tensor, mask_btd: torch.Tensor) -> torch.Tensor:
        v = v * mask_btd
        x = self.pad_fn(v.transpose(1, 2))
        x = self.fsmn_block(x).transpose(1, 2)
        x = x + v
        return x * mask_btd

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, kv_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``x (B, T, in_feat)``, ``mask (B, 1, T)`` bool → ``(B, T, n_feat)``.

        Attention goes through the shared core rather than an explicit
        ``matmul → masked_fill → softmax → masked_fill → matmul``.  Measured on
        `paraformer-zh` (50 SANM blocks), ``B=8``, paired A/B: **1.28x** on ~5 s
        utterances and **1.17x** on ~25 s ones.

        Note the short case wins *more*, so the dominant effect is **kernel
        count**, not the ``(B, h, T, T)`` transients: the explicit form is five
        kernels per block — 250 launches across the encoder — where the fused
        call is one, and at ``T≈80`` the encoder is launch-bound.  The avoided
        transients (~3 GiB at ``B=8`` / ``T=500``) are what keeps the win from
        shrinking to nothing on long audio.

        ``mask`` is still needed for the FSMN memory branch, which is
        multiplicative; ``kv_lens`` says the same thing to attention in the
        form the fused kernel takes.  Both describe right padding, so a
        non-empty utterance always has a valid key and no row can softmax over
        all ``-inf``.

        Not bit-exact vs the explicit form — different reduction order — so the
        FunASR oracle in ``tests/test_paraformer.py`` is the gate (encoder ≤2e-5,
        CIF fires bit-exact, transcript exact).
        """
        b, t, _ = x.shape
        q, k, v = torch.split(self.linear_q_k_v(x), self.h * self.d_k, dim=-1)
        mask_btd = mask.reshape(b, t, 1).to(v.dtype)
        fsmn_memory = self._forward_fsmn(v, mask_btd)

        q_h = self.attn.split_heads(q) * self.d_k ** (-0.5)
        k_h = self.attn.split_heads(k)
        v_h = self.attn.split_heads(v)

        if kv_lens is None:
            att = self.attn(q_h, k_h, v_h, attn_mask=mask.unsqueeze(1))
        else:
            att = self.attn(q_h, k_h, v_h, kv_lens=kv_lens)
        return self.linear_out(self.attn.merge_heads(att)) + fsmn_memory


class SanmCrossAttention(nn.Module):
    """Cross-attention with fused ``linear_k_v`` (``MultiHeadedAttentionCrossAtt``)."""

    def __init__(self, n_head: int, n_feat: int) -> None:
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = ColumnParallelLinear(n_feat, n_feat)
        self.linear_k_v = ColumnParallelLinear(n_feat, n_feat * 2)
        self.linear_out = RowParallelLinear(n_feat, n_feat)
        self.attn = Attention(n_head, self.d_k)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, memory_lens: torch.Tensor
    ) -> torch.Tensor:
        """``x (B, U, D)``, ``memory (B, T, D)``, ``memory_lens (B,)``.

        The memory mask used to arrive as a ``(B, 1, T)`` tensor cast to the
        activation dtype and handed to SDPA as ``attn_mask``.  SDPA reads a
        non-bool mask as **additive**, so that ``{0.0, 1.0}`` tensor added
        ``+1.0`` to every valid logit and left padded keys fully attendable
        instead of masking them — invisible at ``B=1`` or with equal-length
        rows (a constant shift cancels in softmax), wrong for a mixed-length
        batch.  Passing the lengths and letting the attention core build the
        mask fixes it and reaches the fused kernel; FunASR masks with ``-inf``,
        which is what this now matches.  Gated by
        ``tests/test_paraformer.py::TestCrossAttentionPadding``.
        """
        q_h = self.attn.split_heads(self.linear_q(x))
        k, v = torch.split(self.linear_k_v(memory), self.h * self.d_k, dim=-1)
        k_h = self.attn.split_heads(k)
        v_h = self.attn.split_heads(v)

        # Default ``1/sqrt(d_k)`` scale — what the explicit form applied.
        att = self.attn(q_h, k_h, v_h, kv_lens=memory_lens)
        return self.linear_out(self.attn.merge_heads(att))


def EncoderFeedForward(idim: int, hidden_units: int) -> FeedForward:
    """``PositionwiseFeedForward``: w_1 → ReLU → w_2."""
    return FeedForward(idim, hidden_units, activation="relu", names=("w_1", "w_2"))


def DecoderFeedForward(idim: int, hidden_units: int) -> FeedForward:
    """``PositionwiseFeedForwardDecoderSANM``: w_1 → ReLU → LayerNorm → w_2 (no bias)."""
    return FeedForward(
        idim,
        hidden_units,
        activation="relu",
        names=("w_1", "w_2"),
        out_bias=False,
        inner_norm_eps=LAYER_NORM_EPS,
    )
