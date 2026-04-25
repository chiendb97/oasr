# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer model: encoder, layers, and submodules (WeNet algorithm, vLLM-style layout)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from oasr.layers.linear import Linear, LinearActivation
from oasr.layers.conv import PointwiseConv1d, DepthwiseConv1d, Conv2dActivation
from oasr.layers.norm import LayerNorm, GlobalCMVN
from oasr.layers.attention.attention import (
    PagedKVCache,
    RelPositionMultiHeadedAttention,
    T_CACHE,
)
from oasr.utils import get_norm, get_norm_activation
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


class CTC(torch.nn.Module):
    """CTC module: Linear -> log_softmax"""

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
    ):
        """ Construct CTC module
        Args:
            vocab_size: number of output classes
            encoder_output_size: number of encoder projection units
        """
        super().__init__()
        self.ctc_lo = Linear(encoder_output_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hidden_states: batch of hidden state sequences (B, T, D)
        """
        return F.log_softmax(self.ctc_lo(hidden_states), dim=2)


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

        self.w_1 = LinearActivation(
            idim, hidden_units, bias=bias, activation_type=activation_type)
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

        self.norm_activation = get_norm_activation(norm)(
            inner_channels, eps=norm_eps, activation=activation_type)

        self.pointwise_conv2 = PointwiseConv1d(
            inner_channels, channels, bias=bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Remap old "norm.*" keys to "norm_activation.*"
        old_prefix = prefix + "norm."
        new_prefix = prefix + "norm_activation."
        keys_to_remap = [k for k in state_dict if k.startswith(old_prefix)]
        for old_key in keys_to_remap:
            state_dict[new_prefix +
                       old_key[len(old_prefix):]] = state_dict.pop(old_key)
        # Drop parameterless activation module keys (e.g. from nn.SiLU)
        act_prefix = prefix + "activation."
        for k in [k for k in state_dict if k.startswith(act_prefix)]:
            state_dict.pop(k)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
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
        x = self.norm_activation(x)
        x = self.pointwise_conv2(x)
        if mask_pad.size(2) > 0:
            x = x.masked_fill(~mask_btc, 0.0)
        return x, new_cache


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
    """Conv2d subsampling module (e.g. 4x) + linear + positional encoding (WeNet Conv2dSubsampling4)."""

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
            self.conv1 = Conv2dActivation(
                1, odim, 3, stride=2, activation_type="relu")
            self.conv2 = Conv2dActivation(
                odim, odim, 3, stride=2, activation_type="relu")
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
        """Remap old nn.Sequential checkpoint keys to the new attribute names.

        Old layout (nn.Sequential):
          conv.0.weight / conv.0.bias  →  conv1.weight / conv1.bias
          conv.2.weight / conv.2.bias  →  conv2.weight / conv2.bias

        Conv2dActivation's own _load_from_state_dict handles the
        [K, IC, R, S] → [K, R, S, IC] weight permutation for both convs.
        """
        remap = {
            prefix + "conv.0.weight": prefix + "conv1.weight",
            prefix + "conv.0.bias":   prefix + "conv1.bias",
            prefix + "conv.2.weight": prefix + "conv2.weight",
            prefix + "conv.2.bias":   prefix + "conv2.bias",
        }
        for old_key, new_key in remap.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conv1 is not None:
            x = x.unsqueeze(-1)       # [N, T,   F,  1   ] NHWC
            x = self.conv1(x)         # [N, T',  F', odim] NHWC (fused ReLU)
            x = self.conv2(x)         # [N, T'', F'', odim] NHWC (fused ReLU)
            b, t, f, c = x.size()
            # Flatten in c-major order to match WeNet's NCHW-based convention
            # (transpose(1,2).view on [b,c,t,f] == permute(0,1,3,2).view on [b,t,f,c])
            x = x.permute(0, 1, 3, 2).contiguous().view(
                b, t, c * f)  # [N, T'', odim*F'']
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
        att_cache: Union[T_CACHE, PagedKVCache] = (
            torch.zeros(0, 0, 0, 0),
            torch.zeros(0, 0, 0, 0),
        ),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0),
    ) -> Tuple[torch.Tensor, Union[T_CACHE, PagedKVCache], torch.Tensor]:
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
    """Conformer encoder module."""

    def __init__(
        self,
        config: ConformerEncoderConfig,
        global_cmvn: Optional[GlobalCMVN] = None,
    ):
        super().__init__()
        self.global_cmvn = global_cmvn
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

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                shape (elayers, b=1, cache_t2, hidden-dim), where
                `cache_t2 == cnn.lorder`; per-layer layout is (B, T, C).
            att_mask (torch.Tensor): additive attention bias (float dtype), shape
                (1, chunk_size, attention_key_size) with attention_key_size =
                cache_t1 + chunk_size. Use zeros for no masking. Empty placeholder
                (0,0,0) is replaced internally by a zero mask when chunk_size > 0.

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, shape
                (elayers, 1, cache_t2, hidden-dim), same as input cnn_cache.

        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        if att_mask.size(-1) == 0 and attention_key_size > 0:
            att_mask = torch.zeros(
                (1, chunk_size, attention_key_size),
                dtype=xs.dtype,
                device=xs.device,
            )
        elif att_mask.dtype == torch.bool:
            att_mask = mask_to_bias(att_mask, xs.dtype)
        pos_emb = self.embed.pos_enc.position_encoding(offset=offset - cache_t1,
                                                       size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            if elayers == 0:
                kv_cache = (att_cache, att_cache)
            else:
                i_kv_cache = att_cache[i:i + 1]
                size = att_cache.size(-1) // 2
                kv_cache = (i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :,
                                                                   size:])
            xs, new_kv_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=kv_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)

        r_att_cache = torch.cat(r_att_cache, dim=0)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

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

    # ------------------------------------------------------------------
    # Paged-cache forward
    # ------------------------------------------------------------------

    def forward_chunk_paged(
        self,
        xs: torch.Tensor,
        offset: int,
        att_caches: List[PagedKVCache],
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
        cache_t1: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward one chunk using paged KV cache.

        K/V for the current chunk are written directly into the shared block
        pool by :class:`~oasr.layers.attention.RelPositionMultiHeadedAttention`
        (via :func:`~oasr.layers.attention.attention._paged_write_kv`).  The
        caller is responsible for allocating the necessary physical blocks
        **before** this call and for updating ``cache_seqlens`` **after**.

        Parameters
        ----------
        xs : Tensor
            Chunk input ``(1, chunk_input_frames, mel_dim)``.
        offset : int
            Current encoder-output frame offset (used for positional encoding).
        att_caches : list[PagedKVCache]
            One :class:`~oasr.layers.attention.PagedKVCache` per encoder layer.
            All items share the **same** ``block_table`` and ``cache_seqlens``
            tensors; only ``k_cache`` / ``v_cache`` differ per layer.
        cnn_cache : Tensor
            CNN cache ``(num_layers, 1, cnn_cache_frames, hidden_dim)``.
            Pass a ``(0,0,0,0)`` tensor on the first chunk.
        att_mask : Tensor
            Additive attention bias ``(1, chunk_size, attention_key_size)``.
            A ``(0,0,0)`` placeholder is replaced internally by a zero mask.
        cache_t1 : int
            Number of frames already in the KV cache, as a host-side int.
            When non-negative, skips the ``cache_seqlens[0].item()`` D2H
            sync — the scheduler already tracks this via
            ``AttentionCacheState.num_committed_frames``.  Passing ``-1``
            (the legacy default) falls back to reading from the GPU tensor
            for callers that don't track cache length on the host.

        Returns
        -------
        xs : Tensor
            Encoder output ``(1, chunk_size, hidden_dim)``.
        r_cnn_cache : Tensor
            Updated CNN cache ``(num_layers, 1, cnn_cache_frames, hidden_dim)``.
        """
        B = xs.size(0)
        tmp_masks = torch.ones(
            B, xs.size(1), device=xs.device, dtype=torch.bool
        ).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)

        chunk_size = xs.size(1)
        if cache_t1 < 0:
            cache_t1 = int(att_caches[0].cache_seqlens[0].item())
        attention_key_size = cache_t1 + chunk_size

        if att_mask.size(-1) == 0 and attention_key_size > 0:
            att_mask = torch.zeros(
                (1, chunk_size, attention_key_size),
                dtype=xs.dtype,
                device=xs.device,
            )
        elif att_mask.dtype == torch.bool:
            att_mask = mask_to_bias(att_mask, xs.dtype)

        pos_emb = self.embed.pos_enc.position_encoding(
            offset=offset - cache_t1, size=attention_key_size
        )

        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, _, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=att_caches[i],
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache,
            )
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))

        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)

        return xs, torch.cat(r_cnn_cache, dim=0)

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> torch.Tensor:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (logits, att_cache, cnn_cache) = self.forward_chunk(chunk_xs, offset,
                                                                required_cache_size, att_cache,
                                                                cnn_cache)
            outputs.append(logits)
            offset += logits.size(1)
        outputs = torch.cat(outputs, 1)
        return outputs


class ConformerModel(nn.Module):
    """Conformer model: encoder + CTC."""

    def __init__(
        self,
        config: Optional[ConformerModelConfig] = None,
        global_cmvn: Optional[GlobalCMVN] = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = ConformerEncoder(
            config.encoder, global_cmvn=global_cmvn)
        self.ctc = CTC(
            config.vocab_size,
            config.encoder.output_size,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, _ = self.encoder(input_features, lengths)
        probs = self.ctc(hidden_states)
        return probs

    def forward_chunk(
        self,
        input_features: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, att_cache, cnn_cache = self.encoder.forward_chunk(
            input_features, offset, required_cache_size, att_cache, cnn_cache, att_mask)

        probs = self.ctc(hidden_states)
        return probs, att_cache, cnn_cache

    def forward_chunk_paged(
        self,
        input_features: torch.Tensor,
        offset: int,
        att_caches: List[PagedKVCache],
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
        cache_t1: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward one chunk with paged KV cache; returns ``(probs, r_cnn_cache)``.

        See :meth:`ConformerEncoder.forward_chunk_paged` for parameter and
        return-value documentation.
        """
        hidden_states, r_cnn_cache = self.encoder.forward_chunk_paged(
            input_features, offset, att_caches, cnn_cache, att_mask, cache_t1,
        )
        probs = self.ctc(hidden_states)
        return probs, r_cnn_cache

    def forward_chunk_by_chunk(
        self,
        input_features: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> torch.Tensor:
        outputs = self.encoder.forward_chunk_by_chunk(
            input_features, decoding_chunk_size, num_decoding_left_chunks)
        probs = self.ctc(outputs)
        return probs
