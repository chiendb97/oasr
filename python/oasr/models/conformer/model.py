# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer model: encoder, layers, and submodules (WeNet algorithm, vLLM-style layout)."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

from oasr.layers.linear import Linear, LinearActivation
from oasr.layers.conv import PointwiseConv1d, DepthwiseConv1d
from oasr.layers.norm import LayerNorm, RMSNorm
from oasr.layers.attention.attention import RelPositionMultiHeadedAttention, T_CACHE
from oasr.utils import get_activation, get_norm
from .config import ConformerEncoderConfig, ConformerModelConfig


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
    mask = (1.0 - mask) * -1.0e+10
    return mask


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) lengths -> (B, max_len) bool, True where valid (not padded)."""
    row = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    return row.unsqueeze(0) < lengths.unsqueeze(1)


# -----------------------------------------------------------------------------
# Position-wise feed-forward
# -----------------------------------------------------------------------------


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward: Linear -> activation -> Linear (WeNet)."""

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        activation_type: str,
        bias: bool = True,
    ):
        super().__init__()

        self.w_1 = LinearActivation(
            idim, hidden_units, bias=bias, activation_type=activation_type)
        self.w_2 = Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.w_1(xs))


# -----------------------------------------------------------------------------
# Convolution module
# -----------------------------------------------------------------------------


class ConvolutionModule(nn.Module):
    """Conformer conv: pointwise -> GLU -> depthwise -> norm -> activation -> pointwise (WeNet)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation_type: str = "swish",
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
        norm_eps: float = 1e-5,
        conv_inner_factor: int = 2
    ):
        super().__init__()
        self.pointwise_conv1 = PointwiseConv1d(
            channels, conv_inner_factor * channels,
        )
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size -
                    1) % 2 == 0, "kernel_size should be odd for non-causal"
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

        self.norm = get_norm(norm)(inner_channels, eps=norm_eps)

        self.use_layer_norm = norm != "batch_norm"

        self.pointwise_conv2 = PointwiseConv1d(
            inner_channels, channels, bias=bias)
        self.activation = get_activation(activation_type)

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
                x = nn.functional.pad(
                    x, (0, 0, self.lorder, 0), mode="constant", value=0.0)
            else:
                # cache is [B, T, C], same layout as x
                assert cache.size(0) == x.size(
                    0) and cache.size(2) == x.size(2)
                x = torch.cat((cache, x), dim=1)
            new_cache = x[:, -self.lorder:, :].contiguous()
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=-1)
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = self.activation(self.norm(x))
        else:
            # BatchNorm1d requires [B, C, T]
            x = x.transpose(1, 2)
            x = self.activation(self.norm(x))
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        if mask_pad.size(2) > 0:
            x = x.masked_fill(~mask_btc, 0.0)
        return x, new_cache


# -----------------------------------------------------------------------------
# Positional encoding and subsampling
# -----------------------------------------------------------------------------


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding for Conformer (WeNet). Returns (x * xscale, pos_emb)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -
            (math.log(10000.0) / d_model)
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
            pos_emb = self.pe[:, offset: offset + size]
        else:
            index = offset.unsqueeze(
                1) + torch.arange(0, size, device=offset.device)
            index = index.clamp(min=0)
            pos_emb = torch.nn.functional.embedding(index, self.pe[0])
        return x, pos_emb

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        if isinstance(offset, int):
            return self.pe[:, offset: offset + size]
        index = offset.unsqueeze(
            1) + torch.arange(0, size, device=offset.device)
        index = index.clamp(min=0)
        return torch.nn.functional.embedding(index, self.pe[0])


class Conv2dSubsampling(nn.Module):
    """Conv2d subsampling (e.g. 4x) + linear + positional encoding (WeNet Conv2dSubsampling4)."""

    def __init__(
        self,
        idim: int,
        odim: int,
        subsampling_rate: int = 4,
    ):
        super().__init__()
        self.pos_enc = RelPositionalEncoding(
            odim,
            max_len=5000,
        )

        self.subsampling_rate = subsampling_rate
        if subsampling_rate == 4:
            self.conv = nn.Sequential(
                nn.Conv2d(1, odim, 3, 2),
                nn.ReLU(),
                nn.Conv2d(odim, odim, 3, 2),
                nn.ReLU(),
            )
            self.linear_dim = odim * (((idim - 1) // 2 - 1) // 2)
        elif subsampling_rate == 1:
            self.conv = None
            self.linear_dim = idim
        else:
            raise NotImplementedError(f"subsampling_rate={subsampling_rate}")
        self.out = nn.Sequential(
            Linear(self.linear_dim, odim),
            LayerNorm(odim, eps=1e-5),
        )
        self.right_context = 6 if subsampling_rate == 4 else 0

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conv is not None:
            x = x.unsqueeze(1)
            x = self.conv(x)
            b, c, t, f = x.size()
            x = x.transpose(1, 2).contiguous().view(b, t, c * f)
            x_mask = x_mask[:, :, 2::2][:, :, 2::2]
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


# -----------------------------------------------------------------------------
# Encoder layer
# -----------------------------------------------------------------------------


class ConformerEncoderLayer(nn.Module):
    """Single Conformer block: macaron FFN -> MHA (rel_pos) -> conv -> FFN (WeNet)."""

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

            self.norm_ff_macaron = get_norm(
                layer_norm_type)(size, eps=norm_eps)
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
        att_cache: T_CACHE = (torch.zeros(0, 0, 0, 0),
                              torch.zeros(0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        # Attention always uses SDPA-based kernels. WeNet expects the
        # attention mask to be an additive bias (float/bfloat16/half), not a
        # bool tensor, so convert boolean masks to a large negative bias via

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
        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb, att_cache)
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


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------


class ConformerEncoder(nn.Module):
    """Conformer encoder: subsampling + pos enc + N x ConformerEncoderLayer + final norm."""

    def __init__(self, config: ConformerEncoderConfig):
        super().__init__()
        if config.input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                config.input_size,
                config.output_size,
                subsampling_rate=4,
            )
        else:
            raise NotImplementedError(f"input_layer={config.input_layer}")
        self.normalize_before = config.normalize_before
        self.final_norm = config.final_norm
        self.after_norm = get_norm(config.layer_norm_type)(
            config.output_size, eps=config.norm_eps)

        self.encoders = nn.ModuleList([
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
        ])

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        attn_masks = mask_to_bias(masks, xs.dtype)
        for layer in self.encoders:
            xs, _, _ = layer(xs, attn_masks, pos_emb, masks)
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)
        return xs, masks


# -----------------------------------------------------------------------------
# Top-level model
# -----------------------------------------------------------------------------


class ConformerModel(nn.Module):

    def __init__(
        self,
        config: Optional[ConformerModelConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = ConformerEncoder(config.encoder)

    def forward(
        self,
        input_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input_features, lengths)
