# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer model: encoder, layers, and submodules (WeNet algorithm, vLLM-style layout)."""

from __future__ import annotations

import logging
import math
from typing import List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

import oasr
from oasr.layers.linear import Linear, LinearActivation
from oasr.layers.conv import PointwiseConv1d, DepthwiseConv1d, Conv2dActivation
from oasr.layers.norm import LayerNorm, GlobalCMVN
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.slot_cnn import SlotCnnCache
from oasr.layers.attention.attention import RelPositionMultiHeadedAttention
from oasr.utils import get_norm, get_norm_activation
from ..base import BaseAsrModel, BaseEncoder
from ..heads.ctc import CTCHead
from .config import ConformerEncoderConfig, ConformerModelConfig
from .packing import PackedLayout, build_packed_layout, pack_hidden, unpack_hidden

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e10
    return mask


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) lengths -> (B, max_len) bool, True where valid (not padded)."""
    row = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    return row.unsqueeze(0) < lengths.unsqueeze(1)


# Backwards-compatible alias; the CTC head now lives in oasr.models.heads.ctc.
CTC = CTCHead


# -----------------------------------------------------------------------------
# Position-wise feed-forward
# -----------------------------------------------------------------------------


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward module: Linear -> activation -> Linear (WeNet)."""

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        activation_type: str,
        bias: bool = True,
    ):
        super().__init__()

        self.w_1 = LinearActivation(idim, hidden_units, bias=bias, activation_type=activation_type)
        self.w_2 = Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.w_1(xs))


# -----------------------------------------------------------------------------
# Convolution module
# -----------------------------------------------------------------------------


class ConvolutionModule(nn.Module):
    """Convolution module: pointwise -> GLU -> depthwise -> norm -> activation -> pointwise (WeNet)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation_type: str = "swish",
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
        norm_eps: float = 1e-5,
        conv_inner_factor: int = 2,
    ):
        super().__init__()
        self.pointwise_conv1 = PointwiseConv1d(
            channels,
            conv_inner_factor * channels,
        )
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0, "kernel_size should be odd for non-causal"
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        inner_channels = conv_inner_factor * channels // 2
        self.depthwise_conv = DepthwiseConv1d(
            channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )

        assert norm in ("batch_norm", "layer_norm", "rms_norm")

        self.norm_activation = get_norm_activation(norm)(
            inner_channels, eps=norm_eps, activation=activation_type
        )

        self.pointwise_conv2 = PointwiseConv1d(inner_channels, channels, bias=bias)

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
        # Remap old "norm.*" keys to "norm_activation.*"
        old_prefix = prefix + "norm."
        new_prefix = prefix + "norm_activation."
        keys_to_remap = [k for k in state_dict if k.startswith(old_prefix)]
        for old_key in keys_to_remap:
            state_dict[new_prefix + old_key[len(old_prefix) :]] = state_dict.pop(old_key)
        # Drop parameterless activation module keys (e.g. from nn.SiLU)
        act_prefix = prefix + "activation."
        for k in [k for k in state_dict if k.startswith(act_prefix)]:
            state_dict.pop(k)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C], mask_pad: [B, 1, T]
        # Stay in [B, T, C] throughout; reshape mask for broadcasting
        if mask_pad.size(2) > 0:
            # [B, 1, T] -> [B, T, 1] to broadcast over C dimension
            mask_btc = mask_pad.transpose(1, 2)
            x = x.masked_fill(~mask_btc, 0.0)
        if self.lorder > 0:
            if cache.size(2) == 0:
                # Pad along T dimension: (0, 0) for C, (lorder, 0) for T
                x = nn.functional.pad(x, (0, 0, self.lorder, 0), mode="constant", value=0.0)
            else:
                # cache is [B, T, C], same layout as x
                assert cache.size(0) == x.size(0) and cache.size(2) == x.size(2)
                x = torch.cat((cache, x), dim=1)
            new_cache = x[:, -self.lorder :, :].contiguous()
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        x = self.pointwise_conv1(x)
        x = oasr.glu(x)
        x = self.depthwise_conv(x)
        x = self.norm_activation(x)
        x = self.pointwise_conv2(x)
        if mask_pad.size(2) > 0:
            x = x.masked_fill(~mask_btc, 0.0)
        return x, new_cache

    def forward_packed(
        self,
        x: torch.Tensor,
        layout: PackedLayout,
    ) -> torch.Tensor:
        """Packed (sequence-packing) convolution forward over ``(1, T_total, C)``.

        Two segment-isolation strategies, picked by the conv's causality:

        * **Non-causal** (``lorder == 0``): the symmetric depthwise conv mixes
          both sides, so its post-GLU input is scattered into a *gapped* layout
          (``(K-1)//2`` zero frames between segments), run through one conv, and
          gathered back.  Zero gap frames give every segment clean
          conv-internal-zero boundaries — bit-exact to B=1 inference.
        * **Causal** (``lorder > 0``): the depthwise conv only reads left
          context, so a batched-per-segment form (each segment an independent
          ``(num_segs, max_seg_len)`` row) has no right-context contamination.
          Reuses the standard :meth:`forward` (left-pad + mask) verbatim.
        """
        if self.lorder == 0:
            return self._forward_packed_noncausal(x, layout)
        return self._forward_packed_causal(x, layout)

    def _forward_packed_noncausal(self, x: torch.Tensor, layout: PackedLayout) -> torch.Tensor:
        x = self.pointwise_conv1(x)
        x = oasr.glu(x)  # (1, T_total, C_inner)
        c_inner = x.size(2)
        gapped = x.new_zeros(1, layout.gapped_len, c_inner)
        gapped[0].index_copy_(0, layout.conv_gather_idx, x[0])
        gapped = self.depthwise_conv(gapped)  # (1, gapped_len, C_inner)
        x = gapped[0].index_select(0, layout.conv_gather_idx).unsqueeze(0)
        x = self.norm_activation(x)
        x = self.pointwise_conv2(x)
        return x

    def _forward_packed_causal(self, x: torch.Tensor, layout: PackedLayout) -> torch.Tensor:
        S, Tm, C = layout.num_segs, layout.max_seg_len, x.size(2)
        xb = x.new_zeros(S * Tm, C)
        xb.index_copy_(0, layout.conv_batched_idx, x[0])
        xb = xb.reshape(S, Tm, C)
        mask_pad = layout.seg_valid_mask.unsqueeze(1)  # (S, 1, Tm)
        yb, _ = self.forward(xb, mask_pad)  # standard causal forward
        out = yb.reshape(S * Tm, C).index_select(0, layout.conv_batched_idx)
        return out.unsqueeze(0)


# -----------------------------------------------------------------------------
# Positional encoding and subsampling
# -----------------------------------------------------------------------------


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding module (WeNet). Returns (x * xscale, pos_emb)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x * self.xscale
        size = x.size(1)
        if isinstance(offset, int):
            pos_emb = self.pe[:, offset : offset + size]
        else:
            index = offset.unsqueeze(1) + torch.arange(0, size, device=offset.device)
            index = index.clamp(min=0)
            pos_emb = torch.nn.functional.embedding(index, self.pe[0])
        return x, pos_emb

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        if isinstance(offset, int):
            return self.pe[:, offset : offset + size]
        index = offset.unsqueeze(1) + torch.arange(0, size, device=offset.device)
        index = index.clamp(min=0)
        return torch.nn.functional.embedding(index, self.pe[0])


class Conv2dSubsampling(nn.Module):
    """Conv2d subsampling module (e.g. 4x) + linear + positional encoding (WeNet Conv2dSubsampling4)."""

    # v2 = ``self.out[0].weight`` columns reordered from WeNet's ``(C, F)``
    # c-major flatten to NHWC's natural ``(F, C)`` f-major flatten so the
    # forward path can skip ``permute(0, 1, 3, 2).contiguous()``.
    _version = 2

    def __init__(
        self,
        idim: int,
        odim: int,
        subsampling_rate: int = 4,
        embed_layer_norm: bool = True,
    ):
        super().__init__()
        self.pos_enc = RelPositionalEncoding(
            odim,
            max_len=5000,
        )

        self.subsampling_rate = subsampling_rate
        if subsampling_rate == 4:
            # Both convs use fused conv + ReLU via CUTLASS Ampere Tensor Core
            # Implicit GEMM (NHWC layout). IC=1 uses kAnalytic iterator.
            self.conv1 = Conv2dActivation(1, odim, 3, stride=2, activation_type="relu")
            self.conv2 = Conv2dActivation(odim, odim, 3, stride=2, activation_type="relu")
            self.linear_dim = odim * (((idim - 1) // 2 - 1) // 2)
        elif subsampling_rate == 1:
            self.conv1 = None
            self.conv2 = None
            self.linear_dim = idim
        else:
            raise NotImplementedError(f"subsampling_rate={subsampling_rate}")
        layers = [Linear(self.linear_dim, odim)]
        if embed_layer_norm:
            layers.append(LayerNorm(odim, eps=1e-5))
        self.out = nn.Sequential(*layers)
        self.right_context = 6 if subsampling_rate == 4 else 0

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
        """Remap legacy keys and reorder the embed-linear weight.

        * Old ``nn.Sequential`` layout:
          ``conv.0.{weight,bias}`` → ``conv1.{weight,bias}``;
          ``conv.2.{weight,bias}`` → ``conv2.{weight,bias}``.
          (Conv2dActivation's own ``_load_from_state_dict`` handles the
          ``[K, IC, R, S] → [K, R, S, IC]`` weight permutation.)
        * v1 → v2: the embed linear ``self.out[0].weight`` was trained
          against a ``(C, F)`` c-major flatten of the NHWC conv output.
          We now flatten in NHWC-natural ``(F, C)`` order at runtime, so
          we permute the weight's input axis on load.  Driven by
          ``local_metadata['version']`` (``< 2`` ⇒ legacy).
        """
        remap = {
            prefix + "conv.0.weight": prefix + "conv1.weight",
            prefix + "conv.0.bias": prefix + "conv1.bias",
            prefix + "conv.2.weight": prefix + "conv2.weight",
            prefix + "conv.2.bias": prefix + "conv2.bias",
        }
        for old_key, new_key in remap.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        version = local_metadata.get("version", 1)
        if version < 2 and self.conv1 is not None:
            out_w_key = prefix + "out.0.weight"
            if out_w_key in state_dict:
                w = state_dict[out_w_key]  # (odim, C*F)
                odim, in_dim = w.shape
                C = self.out[0].out_features  # conv output channels
                assert in_dim % C == 0, (
                    f"Conv2dSubsampling.out[0].weight in_dim={in_dim} not " f"divisible by C={C}"
                )
                F = in_dim // C
                # Old col index: c*F + f  →  new col index: f*C + c.
                state_dict[out_w_key] = (
                    w.view(odim, C, F).transpose(1, 2).reshape(odim, F * C).contiguous()
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conv1 is not None:
            x = x.unsqueeze(-1)  # [N, T,   F,  1   ] NHWC
            x = self.conv1(x)  # [N, T',  F', odim] NHWC (fused ReLU)
            x = self.conv2(x)  # [N, T'', F'', odim] NHWC (fused ReLU)
            b, t, f, c = x.size()
            # NHWC-natural flatten: inner index is (f, c).  The embed
            # linear's input axis is permuted at load time to match.
            x = x.reshape(b, t, f * c)
            x_mask = x_mask[:, :, 2::2][:, :, 2::2]
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


# -----------------------------------------------------------------------------
# Encoder layer
# -----------------------------------------------------------------------------


class ConformerEncoderLayer(nn.Module):
    """Conformer encoder layer: macaron FFN -> MHA (rel_pos) -> conv -> FFN (WeNet)."""

    def __init__(
        self,
        size: int,
        normalize_before: bool = True,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        activation_type: str = "swish",
        macaron_style: bool = False,
        linear_units: int = 2048,
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        cnn_module_norm: str = "layer_norm",
        causal: bool = False,
        conv_bias: bool = True,
        conv_norm_eps: float = 1e-5,
        conv_inner_factor: int = 2,
        attention_heads: int = 4,
        n_kv_head: int = 4,
        head_dim: int = 64,
    ):
        super().__init__()
        self.size = size
        self.normalize_before = normalize_before
        self.norm_ff = get_norm(layer_norm_type)(size, eps=norm_eps)
        self.norm_mha = get_norm(layer_norm_type)(size, eps=norm_eps)

        # Always define optional submodules so that forward can safely
        # check for None regardless of the configuration flags.
        self.feed_forward = PositionwiseFeedForward(
            size,
            linear_units,
            activation_type=activation_type,
            bias=True,
        )
        self.feed_forward_macaron: Optional[PositionwiseFeedForward]
        self.norm_ff_macaron: Optional[nn.Module]
        if macaron_style:
            self.feed_forward_macaron = PositionwiseFeedForward(
                size,
                linear_units,
                activation_type=activation_type,
                bias=True,
            )

            self.norm_ff_macaron = get_norm(layer_norm_type)(size, eps=norm_eps)
            self.ff_scale = 0.5
        else:
            self.feed_forward_macaron = None
            self.norm_ff_macaron = None
            self.ff_scale = 1.0

        self.self_attn = RelPositionMultiHeadedAttention(
            attention_heads,
            size,
            query_bias=True,
            key_bias=True,
            value_bias=True,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )

        self.conv_module: Optional[ConvolutionModule]
        self.norm_conv: Optional[nn.Module]
        self.norm_final: Optional[nn.Module]
        if use_cnn_module:
            self.conv_module = ConvolutionModule(
                size,
                cnn_module_kernel,
                activation_type=activation_type,
                norm=cnn_module_norm,
                causal=causal,
                bias=conv_bias,
                norm_eps=conv_norm_eps,
                conv_inner_factor=conv_inner_factor,
            )

            self.norm_conv = get_norm(cnn_module_norm)(size, eps=norm_eps)
            self.norm_final = get_norm(layer_norm_type)(size, eps=norm_eps)
        else:
            self.conv_module = None
            self.norm_conv = None
            self.norm_final = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: Union[PagedKVCache, None] = None,
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0),
    ) -> Tuple[torch.Tensor, Union[PagedKVCache, None], torch.Tensor]:
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.feed_forward_macaron(x)
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, mask, pos_emb, att_cache)
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm_mha(x)
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + x
            if not self.normalize_before:
                x = self.norm_conv(x)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm_ff(x)
        if self.conv_module is not None:
            x = self.norm_final(x)
        return x, new_att_cache, new_cnn_cache

    def forward_packed(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        layout: PackedLayout,
    ) -> torch.Tensor:
        """Sequence-packing layer forward over a gapless packed row.

        ``x`` is ``(1, T_total, D)``.  Attention is restricted to each
        ``cu_seqlens`` segment (varlen); the conv module uses the per-segment
        isolated path.  Macaron FFN / FFN / norms are per-token and run
        unchanged.
        """
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.feed_forward_macaron(x)
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, _ = self.self_attn(x, pos_emb=pos_emb, layout=layout)
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm_mha(x)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = self.conv_module.forward_packed(x, layout)
            x = residual + x
            if not self.normalize_before:
                x = self.norm_conv(x)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm_ff(x)
        if self.conv_module is not None:
            x = self.norm_final(x)
        return x


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    supports_packing = True
    supports_paged_streaming = True

    def __init__(
        self,
        config: ConformerEncoderConfig,
        global_cmvn: Optional[GlobalCMVN] = None,
    ):
        super().__init__()
        self.global_cmvn = global_cmvn
        self._output_size = config.output_size
        # Depthwise-conv left-context kernel for streaming; 1 == no CNN cache.
        self._conv_kernel_size = config.cnn_module_kernel if config.use_cnn_module else 1
        if config.input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                config.input_size,
                config.output_size,
                subsampling_rate=4,
                embed_layer_norm=config.embed_layer_norm,
            )
        else:
            raise NotImplementedError(f"input_layer={config.input_layer}")
        self.normalize_before = config.normalize_before
        self.final_norm = config.final_norm
        self.after_norm = get_norm(config.layer_norm_type)(config.output_size, eps=config.norm_eps)

        self.encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    config.output_size,
                    normalize_before=config.normalize_before,
                    layer_norm_type=config.layer_norm_type,
                    norm_eps=config.norm_eps,
                    activation_type=config.activation_type,
                    macaron_style=config.macaron_style,
                    linear_units=config.linear_units,
                    use_cnn_module=config.use_cnn_module,
                    cnn_module_kernel=config.cnn_module_kernel,
                    cnn_module_norm=config.cnn_module_norm,
                    causal=config.causal,
                    conv_bias=config.conv_bias,
                    conv_norm_eps=config.conv_norm_eps,
                    conv_inner_factor=config.conv_inner_factor,
                    attention_heads=config.attention_heads,
                    n_kv_head=config.n_kv_head,
                    head_dim=config.head_dim,
                )
                for _ in range(config.num_blocks)
            ]
        )

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        T = xs.size(1)
        masks = make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        attn_masks = mask_to_bias(masks, xs.dtype)
        for layer in self.encoders:
            xs, _, _ = layer(xs, attn_masks, pos_emb, masks)
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_packed(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-packing encoder forward (gapless varlen attention).

        Subsampling (``embed``) runs in normal batched mode so the Conv2d
        receptive field never crosses an utterance boundary; the post-
        subsampling hidden states are then packed into one gapless row and the
        expensive conformer layers run once with per-segment varlen attention
        (``cu_seqlens`` + packed block-diagonal rel-pos bias, zero attention
        padding) + conv isolation.  Returns the same ``(B, T_out, D)`` hidden +
        ``(B, 1, T_out)`` mask as :meth:`forward` so the CTC head / decode path
        are unchanged.  Bit-exact to ``B=1`` inference.
        """
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        T = xs.size(1)
        masks = make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)  # (B, Tp, D)
        D = xs.size(2)

        cnn_kernel = (
            self.encoders[0].conv_module.depthwise_conv.kernel_size
            if self.encoders[0].conv_module is not None
            else 1
        )
        num_heads = self.encoders[0].self_attn.h
        layout = build_packed_layout(masks.squeeze(1), cnn_kernel, num_heads=num_heads)

        packed = pack_hidden(xs, layout)  # (1, T_total, D)
        for layer in self.encoders:
            packed = layer.forward_packed(packed, pos_emb, layout)
        if self.normalize_before and self.final_norm:
            packed = self.after_norm(packed)

        xs = unpack_hidden(packed, layout, D)  # (B, Tp, D)
        return xs, masks

    # ------------------------------------------------------------------
    # Properties (introspect model dims without importing config)
    # ------------------------------------------------------------------

    @property
    def num_encoder_layers(self) -> int:
        """Number of conformer encoder layers."""
        return len(self.encoders)

    @property
    def n_kv_head(self) -> int:
        """Number of KV attention heads per layer."""
        return self.encoders[0].self_attn.h_kv  # type: ignore[attr-defined]

    @property
    def head_dim(self) -> int:
        """Per-head key/value dimension."""
        return self.encoders[0].self_attn.d_k  # type: ignore[attr-defined]

    @property
    def output_size(self) -> int:
        """Encoder hidden dimension."""
        return self._output_size

    @property
    def conv_kernel_size(self) -> int:
        """Depthwise-conv kernel for streaming left-context; 1 == no CNN cache."""
        return self._conv_kernel_size

    # ------------------------------------------------------------------
    # Paged-cache forward
    # ------------------------------------------------------------------

    def forward_chunk_paged(
        self,
        xs: torch.Tensor,
        offset: Union[int, torch.Tensor],
        att_caches: List[PagedKVCache],
        cnn_cache: SlotCnnCache,
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
        cache_t1: int = -1,
    ) -> torch.Tensor:
        """Forward one chunk using paged KV and slot-indexed CNN caches.

        Both cache families are written **in-place** during this call:

        * K/V for the current chunk are written into the shared block pool by
          :class:`~oasr.layers.attention.RelPositionMultiHeadedAttention` (via
          :meth:`~oasr.cache.PagedKVCache.write_kv_chunk`).
        * CNN left-context is gathered from ``cnn_cache.buffer[slot_ids]`` at
          the top of the encoder and scattered back at the end, so the
          persistent ``CnnCacheManager`` buffer is updated in place.

        The caller is responsible for allocating the next physical KV block
        **before** this call (via ``AttentionCacheManager.prepare_chunk*``)
        and for updating ``cache_seqlens`` **after**.

        Parameters
        ----------
        xs : Tensor
            Chunk input ``(B, chunk_input_frames, mel_dim)``.  ``B`` is the
            number of streams batched in this call.
        offset : int or Tensor
            Encoder-output frame offset(s).  ``int`` is the legacy
            homogeneous case (every stream at the same offset); a 1-D
            ``int Tensor`` of shape ``(B,)`` enables the heterogeneous
            cohort-relaxed batched forward (each stream has its own
            offset and pos_emb).
        att_caches : list[PagedKVCache]
            One :class:`~oasr.layers.attention.PagedKVCache` per encoder
            layer. All items share the **same** ``block_table`` and
            ``cache_seqlens`` tensors; only ``k_cache`` / ``v_cache``
            differ per layer.
        cnn_cache : SlotCnnCache
            Descriptor over the persistent CNN buffer plus the ``(B,)`` int64
            ``slot_ids`` for this batch. The encoder gathers the per-stream
            left-context at the top and scatters the new tails back into the
            buffer at the bottom.
        att_mask : Tensor
            Unused with paged attention (per-stream length is enforced by
            the FlexAttention block-mask built from ``cache.cache_seqlens``).
            Kept for signature compatibility.
        cache_t1 : int
            Maximum cached frames across the batch as a host-side int.
            Required (must be ``>= 0``): sizes the rel-pos embedding
            without a D2H sync on ``cache_seqlens``.

        Returns
        -------
        xs : Tensor
            Encoder output ``(B, chunk_size, hidden_dim)``.
        """
        del att_mask  # paged path uses cache.cache_seqlens for masking

        B = xs.size(0)
        tmp_masks = torch.ones(B, xs.size(1), device=xs.device, dtype=torch.bool).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, _, _ = self.embed(xs, tmp_masks, offset)

        chunk_size = xs.size(1)
        assert cache_t1 >= 0, (
            "forward_chunk_paged requires cache_t1 (max cached frames across "
            "the batch) to be provided host-side; the engine tracks each "
            "stream's offset already, so passing -1 would force a D2H sync."
        )
        attention_key_size = cache_t1 + chunk_size

        # Per-stream pos_emb: positions begin at ``offset - cache_seqlens``
        # for each stream, which is non-zero only when the cache has been
        # evicted.  When ``offset`` is an int we know all streams share
        # the same offset → the broadcast (1, T_pos, D) form is used.
        if isinstance(offset, int):
            pos_start = offset - cache_t1
        else:
            # offset: Tensor (B,) on GPU. cache_seqlens: Tensor (B,).
            pos_start = offset - att_caches[0].cache_seqlens.to(offset.dtype)
        pos_emb = self.embed.pos_enc.position_encoding(
            offset=pos_start,
            size=attention_key_size,
        )

        # Gather the per-stream left-context once at the top, then scatter
        # the per-layer tails back at the end. One batched index_select +
        # one batched index_copy_ per chunk, regardless of layer count.
        cnn_in = cnn_cache.gather()  # (L, B, K-1, H)
        new_cnn_layers: List[torch.Tensor] = []
        for i, layer in enumerate(self.encoders):
            xs, _, new_cnn = layer(
                xs,
                # Padding mask is unused inside the paged self-attn path; pass
                # a zero-dim placeholder.
                torch.zeros((0, 0, 0), dtype=xs.dtype, device=xs.device),
                pos_emb,
                att_cache=att_caches[i],
                cnn_cache=cnn_in[i],
            )
            new_cnn_layers.append(new_cnn)
        cnn_cache.scatter(torch.stack(new_cnn_layers, dim=0))

        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)

        return xs


class ConformerModel(BaseAsrModel):
    """Conformer model: encoder + CTC head.

    The offline (:meth:`forward_offline`), sequence-packing
    (:meth:`forward_offline_packed`) and streaming
    (:meth:`forward_chunk_paged`) entry points the engine calls are inherited
    from :class:`~oasr.models.base.BaseAsrModel`; this class only supplies the
    Conformer-specific construction and weight loading.
    """

    def __init__(
        self,
        config: Optional[ConformerModelConfig] = None,
        global_cmvn: Optional[GlobalCMVN] = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = ConformerEncoder(config.encoder, global_cmvn=global_cmvn)
        # Registered under ``ctc`` (not ``head``) so WeNet checkpoint keys
        # ``ctc.ctc_lo.*`` map directly; ``head`` is exposed as a property.
        self.ctc = CTCHead(
            config.vocab_size,
            config.encoder.output_size,
        )

    @property
    def head(self) -> CTCHead:
        """The decode-side head (alias for ``self.ctc``)."""
        return self.ctc

    @classmethod
    def from_config(
        cls,
        config: ConformerModelConfig,
        *,
        global_cmvn: Optional[GlobalCMVN] = None,
    ) -> "ConformerModel":
        """Build a (random-weight) ConformerModel from its config."""
        return cls(config, global_cmvn=global_cmvn)

    def forward(
        self,
        input_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, _ = self.encoder(input_features, lengths)
        probs = self.ctc(hidden_states)
        return probs

    def load_weights(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = False,
    ) -> None:
        """Load a WeNet-format state-dict into this model.

        Keeps only the ``encoder.*`` parameters and the ``ctc.ctc_lo.*``
        projection, zero-padding the CTC weight/bias up to this model's
        (8-aligned) vocab when the checkpoint's vocab is smaller.  In-module
        reshaping (fused QKV, Conv2d reorder) is handled by the layers'
        ``_load_from_state_dict`` hooks.  ``encoder.embed.pos_enc.pe`` is a
        computed buffer and is expected to be absent from the checkpoint.
        """
        sd = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}

        target_vocab = self.ctc.ctc_lo.weight.shape[0]
        ctc_w = state_dict["ctc.ctc_lo.weight"]
        ctc_b = state_dict["ctc.ctc_lo.bias"]
        pad = target_vocab - ctc_w.shape[0]
        if pad > 0:
            ctc_w = F.pad(ctc_w, (0, 0, 0, pad))
            ctc_b = F.pad(ctc_b, (0, pad))
        sd["ctc.ctc_lo.weight"] = ctc_w
        sd["ctc.ctc_lo.bias"] = ctc_b

        missing, unexpected = self.load_state_dict(sd, strict=strict)
        expected_missing = {"encoder.embed.pos_enc.pe"}
        real_missing = [k for k in missing if k not in expected_missing]
        if real_missing:
            logger.warning("Unexpected missing keys: %s", real_missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)
