# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer model: encoder, layers, and submodules (WeNet algorithm, vLLM-style layout)."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

from oasr.layers.conv import PointwiseConv1d, DepthwiseConv1d
from oasr.layers.norm import LayerNorm, RMSNorm
from oasr.layers.attention.attention import RelPositionMultiHeadedAttention, T_CACHE
from oasr.utils import torch_dtype_to_oasr_dtype, str_activation_to_oasr_activation

from .config import ConformerEncoderConfig, ConformerModelConfig


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) lengths -> (B, max_len) bool, True where valid (not padded)."""
    row = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    return row.unsqueeze(0) < lengths.unsqueeze(1)


def _get_activation(activation_type: str) -> nn.Module:
    if activation_type == "swish":
        return nn.SiLU()
    if activation_type == "relu":
        return nn.ReLU()
    if activation_type == "gelu":
        return nn.GELU()
    return nn.SiLU()


# -----------------------------------------------------------------------------
# Position-wise feed-forward
# -----------------------------------------------------------------------------


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward: Linear -> activation -> Linear (WeNet)."""

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        activation: nn.Module | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units, bias=bias)
        self.activation = activation if activation is not None else nn.SiLU()
        self.w_2 = nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(xs)))


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
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(inner_channels, eps=norm_eps)
        else:
            self.use_layer_norm = True
            self.norm = (
                LayerNorm(inner_channels, eps=norm_eps)
                if norm == "layer_norm"
                else RMSNorm(inner_channels, eps=norm_eps)
            )

        self.pointwise_conv2 = PointwiseConv1d(
            inner_channels, channels, bias=bias)
        self.activation = _get_activation(activation_type)

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
        pos_enc: nn.Module,
        subsampling_rate: int = 4,
    ):
        super().__init__()
        self.subsampling_rate = subsampling_rate
        self.pos_enc = pos_enc
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
            nn.Linear(self.linear_dim, odim),
            nn.LayerNorm(odim, eps=1e-5),
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

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


# -----------------------------------------------------------------------------
# Encoder layer
# -----------------------------------------------------------------------------


class ConformerEncoderLayer(nn.Module):
    """Single Conformer block: macaron FFN -> MHA (rel_pos) -> conv -> FFN (WeNet)."""

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[ConvolutionModule] = None,
        normalize_before: bool = True,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.size = size
        self.normalize_before = normalize_before
        norm_class = nn.LayerNorm if layer_norm_type == "layer_norm" else _rms_norm_class()
        norm_class = LayerNorm if layer_norm_type == "layer_norm" else RMSNorm
        self.norm_ff = norm_class(size, eps=norm_eps)
        self.norm_mha = norm_class(size, eps=norm_eps)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = norm_class(size, eps=norm_eps)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if conv_module is not None:
            self.norm_conv = norm_class(size, eps=norm_eps)
            self.norm_final = norm_class(size, eps=norm_eps)

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
        return x, mask, new_att_cache, new_cnn_cache


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------


class ConformerEncoder(nn.Module):
    """Conformer encoder: subsampling + pos enc + N x ConformerEncoderLayer + final norm."""

    def __init__(self, config: ConformerEncoderConfig):
        super().__init__()
        self.config = config
        self._output_size = config.output_size
        pos_enc = RelPositionalEncoding(
            config.output_size,
            max_len=5000,
        )
        if config.input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                config.input_size,
                config.output_size,
                pos_enc,
                subsampling_rate=4,
            )
        else:
            raise NotImplementedError(f"input_layer={config.input_layer}")
        self.normalize_before = config.normalize_before
        self.final_norm = config.final_norm
        norm_class = LayerNorm if config.layer_norm_type == "layer_norm" else RMSNorm
        self.after_norm = norm_class(config.output_size, eps=config.norm_eps)
        activation = _get_activation(config.activation_type)

        def make_ff() -> PositionwiseFeedForward:
            return PositionwiseFeedForward(
                config.output_size,
                config.linear_units,
                activation=activation,
                bias=True,
            )

        layers_list = []
        for _ in range(config.num_blocks):
            self_attn = RelPositionMultiHeadedAttention(
                config.attention_heads,
                config.output_size,
                query_bias=True,
                key_bias=True,
                value_bias=True,
                use_sdpa=False,
                n_kv_head=config.n_kv_head,
                head_dim=config.head_dim,
            )
            ff = make_ff()
            ff_macaron = make_ff() if config.macaron_style else None
            conv = None
            if config.use_cnn_module:
                conv = ConvolutionModule(
                    config.output_size,
                    config.cnn_module_kernel,
                    activation_type=config.activation_type,
                    norm=config.cnn_module_norm,
                    causal=config.causal,
                    bias=config.conv_bias,
                    norm_eps=config.conv_norm_eps,
                    conv_inner_factor=config.conv_inner_factor,
                )
            layers_list.append(
                ConformerEncoderLayer(
                    config.output_size,
                    self_attn=self_attn,
                    feed_forward=ff,
                    feed_forward_macaron=ff_macaron,
                    conv_module=conv,
                    normalize_before=config.normalize_before,
                    layer_norm_type=config.layer_norm_type,
                    norm_eps=config.norm_eps,
                )
            )
        self.encoders = nn.ModuleList(layers_list)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = _make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        for layer in self.encoders:
            xs, masks, _, _ = layer(xs, masks, pos_emb, mask_pad)
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)
        return xs, masks


# -----------------------------------------------------------------------------
# Top-level model
# -----------------------------------------------------------------------------


class ConformerModel(nn.Module):
    """Conformer encoder-only model (vLLM-style interface)."""

    def __init__(
        self,
        config: Optional[ConformerModelConfig] = None,
        encoder_config: Optional[ConformerEncoderConfig] = None,
    ):
        super().__init__()
        if config is not None:
            self.config = config
            encoder_config = config.encoder
        elif encoder_config is not None:
            self.config = ConformerModelConfig(encoder=encoder_config)
            encoder_config = encoder_config
        else:
            self.config = ConformerModelConfig()
            encoder_config = self.config.encoder
        self.encoder = ConformerEncoder(encoder_config)

    @property
    def output_size(self) -> int:
        return self.encoder.output_size

    def forward(
        self,
        input_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input_features, lengths)
