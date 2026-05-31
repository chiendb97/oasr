# Copyright    2022-2023  Xiaomi Corp.   (authors: Daniel Povey, Zengwei Yao)
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only port of the Zipformer2 encoder modules from icefall.

Faithful to icefall's ``zipformer.py`` forward / ``streaming_forward`` paths,
with training-only machinery (Balancer, Whiten, dropout, layer-skip, ScaledAdam
scaling, diagnostics) removed.  Module + parameter names mirror icefall exactly
so an icefall checkpoint loads with a 1:1 key mapping.

Tensors are time-first ``(seq_len, batch, dim)`` throughout, as in icefall.

Reference:
https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .scaling import (
    ActivationDropoutAndLinear,
    BiasNorm,
    ChunkCausalDepthwiseConv1d,
    convert_num_channels,
)


def _to_tuple(x, length: int):
    """Convert an int or 1-tuple to a tuple of the given length."""
    if isinstance(x, int):
        x = (x,)
    if len(x) == 1:
        x = x * length
    else:
        assert len(x) == length, (len(x), length)
    return tuple(x)


class CompactRelPositionalEncoding(nn.Module):
    """Compact relative positional encoding (parameter-free; ``pe`` is recomputed)."""

    def __init__(self, embed_dim: int, length_factor: float = 1.0, max_len: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0, embed_dim
        assert length_factor >= 1.0, length_factor
        self.length_factor = length_factor
        self.pe: Optional[Tensor] = None
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        T = x.size(0) + left_context_len
        if self.pe is not None and self.pe.size(0) >= T * 2 - 1:
            self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return

        x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)
        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)
        compression_length = self.embed_dim**0.5
        x_compressed = (
            compression_length
            * x.sign()
            * ((x.abs() + compression_length).log() - math.log(compression_length))
        )
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)
        x_atan = (x_compressed / length_scale).atan()
        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()
        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.
        self.pe = pe.to(dtype=torch.get_default_dtype())

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        self.extend_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        pos_emb = self.pe[
            self.pe.size(0) // 2 - x_size_left + 1 : self.pe.size(0) // 2 + x.size(0),
            :,
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return pos_emb.to(dtype=x.dtype)


class RelPositionMultiheadAttentionWeights(nn.Module):
    """Computes shared multi-head attention weights with relative position encoding."""

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads
        self.in_proj = nn.Linear(embed_dim, in_proj_dim, bias=True)
        self.linear_pos = nn.Linear(pos_dim, num_heads * pos_head_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads
        seq_len, batch_size, _ = x.shape
        query_dim = query_head_dim * num_heads

        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        p = x[..., 2 * query_dim :]

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1
        pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
        # (head, {1 or batch}, pos_dim, seq_len2)
        pos_scores = torch.matmul(p, pos_emb)
        pos_scores = pos_scores.as_strided(
            (num_heads, batch_size, seq_len, seq_len),
            (
                pos_scores.stride(0),
                pos_scores.stride(1),
                pos_scores.stride(2) - pos_scores.stride(3),
                pos_scores.stride(3),
            ),
            storage_offset=pos_scores.stride(3) * (seq_len - 1),
        )
        attn_scores = attn_scores + pos_scores

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, seq_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1), -1000)

        return attn_scores.softmax(dim=-1)

    def streaming_forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        left_context_len: int,
        key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads
        seq_len, batch_size, _ = x.shape
        query_dim = query_head_dim * num_heads

        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        p = x[..., 2 * query_dim :]

        assert cached_key.shape[0] == left_context_len, (cached_key.shape[0], left_context_len)
        k = torch.cat([cached_key, k], dim=0)
        cached_key = k[-left_context_len:, ...]
        k_len = k.shape[0]

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(k_len, batch_size, num_heads, query_head_dim)

        q = q.permute(2, 1, 0, 3)
        p = p.permute(2, 1, 0, 3)
        k = k.permute(2, 1, 3, 0)

        attn_scores = torch.matmul(q, k)

        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1 + left_context_len
        pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
        pos_scores = torch.matmul(p, pos_emb)
        pos_scores = pos_scores.as_strided(
            (num_heads, batch_size, seq_len, k_len),
            (
                pos_scores.stride(0),
                pos_scores.stride(1),
                pos_scores.stride(2) - pos_scores.stride(3),
                pos_scores.stride(3),
            ),
            storage_offset=pos_scores.stride(3) * (seq_len - 1),
        )
        attn_scores = attn_scores + pos_scores

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, k_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1), -1000)

        return attn_scores.softmax(dim=-1), cached_key


class SelfAttention(nn.Module):
    """Applies precomputed attention weights to a value projection."""

    def __init__(self, embed_dim: int, num_heads: int, value_head_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * value_head_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, attn_weights: Tensor) -> Tensor:
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        x = self.in_proj(x)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        x = torch.matmul(attn_weights, x)
        x = x.permute(2, 1, 0, 3).contiguous().view(seq_len, batch_size, -1)
        return self.out_proj(x)

    def streaming_forward(
        self, x: Tensor, attn_weights: Tensor, cached_val: Tensor, left_context_len: int
    ) -> Tuple[Tensor, Tensor]:
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        seq_len2 = seq_len + left_context_len
        x = self.in_proj(x)
        assert cached_val.shape[0] == left_context_len, (cached_val.shape[0], left_context_len)
        x = torch.cat([cached_val, x], dim=0)
        cached_val = x[-left_context_len:, ...]
        x = x.reshape(seq_len2, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        x = torch.matmul(attn_weights, x)
        x = x.permute(2, 1, 0, 3).contiguous().view(seq_len, batch_size, -1)
        return self.out_proj(x), cached_val


class FeedforwardModule(nn.Module):
    def __init__(self, embed_dim: int, feedforward_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim, embed_dim, activation="SwooshL", bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        x = self.out_proj(x)
        return x


class NonlinAttention(nn.Module):
    """Like ConvolutionModule but uses attention weights in place of convolution."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)
        self.tanh = nn.Tanh()
        self.out_proj = nn.Linear(hidden_channels, channels, bias=True)

    def forward(self, x: Tensor, attn_weights: Tensor) -> Tensor:
        x = self.in_proj(x)
        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels
        s, x, y = x.chunk(3, dim=2)
        s = self.tanh(s)
        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = x * s

        num_heads = attn_weights.shape[0]
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        x = torch.matmul(attn_weights, x)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        x = x * y
        return self.out_proj(x)

    def streaming_forward(
        self, x: Tensor, attn_weights: Tensor, cached_x: Tensor, left_context_len: int
    ) -> Tuple[Tensor, Tensor]:
        x = self.in_proj(x)
        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels
        s, x, y = x.chunk(3, dim=2)
        s = self.tanh(s)
        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = x * s

        num_heads = attn_weights.shape[0]
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        assert cached_x.shape[2] == left_context_len, (cached_x.shape[2], left_context_len)
        x_pad = torch.cat([cached_x, x], dim=2)
        cached_x = x_pad[:, :, -left_context_len:, :]
        x = torch.matmul(attn_weights, x_pad)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        x = x * y
        return self.out_proj(x), cached_x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 (GLU gating + depthwise conv + SwooshR linear)."""

    def __init__(self, channels: int, kernel_size: int, causal: bool):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(channels, 2 * bottleneck_dim)
        self.sigmoid = nn.Sigmoid()

        if causal:
            self.depthwise_conv = ChunkCausalDepthwiseConv1d(
                channels=bottleneck_dim, kernel_size=kernel_size
            )
        else:
            self.depthwise_conv = nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim, channels, activation="SwooshR"
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        chunk_size: int = -1,
    ) -> Tensor:
        x = self.in_proj(x)  # (time, batch, 2*channels)
        x, s = x.chunk(2, dim=2)
        s = self.sigmoid(s)
        x = x * s
        x = x.permute(1, 2, 0)  # (batch, channels, time)
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
        if chunk_size >= 0:
            assert self.causal, "Must initialize with causal=True to use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)
        x = x.permute(2, 0, 1)  # (time, batch, channels)
        x = self.out_proj(x)
        return x

    def streaming_forward(
        self, x: Tensor, cache: Tensor, src_key_padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        x = self.in_proj(x)
        x, s = x.chunk(2, dim=2)
        s = self.sigmoid(s)
        x = x * s
        x = x.permute(1, 2, 0)  # (batch, channels, time)
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
        x, cache = self.depthwise_conv.streaming_forward(x, cache=cache)
        x = x.permute(2, 0, 1)  # (time, batch, channels)
        x = self.out_proj(x)
        return x, cache


class BypassModule(nn.Module):
    """Learnable per-channel bypass scale: ``src_orig + (src - src_orig) * scale``."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

    def forward(self, src_orig: Tensor, src: Tensor) -> Tensor:
        return src_orig + (src - src_orig) * self.bypass_scale


class SimpleDownsample(nn.Module):
    """Downsampling by a learnable weighted average over a window of ``downsample`` frames."""

    def __init__(self, channels: int, downsample: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(downsample))
        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds
        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        assert src.shape[0] == d_seq_len * ds, (src.shape, d_seq_len, ds)
        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
        weights = self.bias.softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        return (src * weights).sum(dim=1)


class SimpleUpsample(nn.Module):
    """Upsampling by repeating each frame ``upsample`` times (parameter-free)."""

    def __init__(self, num_channels: int, upsample: int):
        super().__init__()
        self.upsample = upsample

    def forward(self, src: Tensor) -> Tensor:
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        return src.reshape(seq_len * upsample, batch_size, num_channels)


class Zipformer2EncoderLayer(nn.Module):
    """One Zipformer2 encoder layer (attention weights shared across submodules)."""

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        cnn_module_kernel: int = 31,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.bypass = BypassModule(embed_dim)
        self.bypass_mid = BypassModule(embed_dim)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, pos_dim=pos_dim, num_heads=num_heads,
            query_head_dim=query_head_dim, pos_head_dim=pos_head_dim,
        )
        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4)
        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim)
        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4)

        self.nonlin_attention = NonlinAttention(embed_dim, hidden_channels=3 * embed_dim // 4)

        self.conv_module1 = ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)
        self.conv_module2 = ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)

        # Present in icefall checkpoints (unused in forward); kept for key parity.
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.norm = BiasNorm(embed_dim)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_orig = src
        attn_weights = self.self_attn_weights(
            src, pos_emb=pos_emb, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.feed_forward1(src)

        selected_attn_weights = attn_weights[0:1]
        na = self.nonlin_attention(src, selected_attn_weights)
        src = src + na

        src = src + self.self_attn1(src, attn_weights)
        src = src + self.conv_module1(
            src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask
        )
        src = src + self.feed_forward2(src)

        src = self.bypass_mid(src_orig, src)

        src = src + self.self_attn2(src, attn_weights)
        src = src + self.conv_module2(
            src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask
        )
        src = src + self.feed_forward3(src)

        src = self.norm(src)
        src = self.bypass(src_orig, src)
        return src

    def streaming_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        cached_nonlin_attn: Tensor,
        cached_val1: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        src_orig = src
        attn_weights, cached_key = self.self_attn_weights.streaming_forward(
            src, pos_emb=pos_emb, cached_key=cached_key,
            left_context_len=left_context_len, key_padding_mask=src_key_padding_mask,
        )
        src = src + self.feed_forward1(src)

        na, cached_nonlin_attn = self.nonlin_attention.streaming_forward(
            src, attn_weights[0:1], cached_x=cached_nonlin_attn, left_context_len=left_context_len
        )
        src = src + na

        self_attn, cached_val1 = self.self_attn1.streaming_forward(
            src, attn_weights=attn_weights, cached_val=cached_val1, left_context_len=left_context_len
        )
        src = src + self_attn

        src_conv, cached_conv1 = self.conv_module1.streaming_forward(
            src, cache=cached_conv1, src_key_padding_mask=src_key_padding_mask[:, left_context_len:]
        )
        src = src + src_conv

        src = src + self.feed_forward2(src)
        src = self.bypass_mid(src_orig, src)

        self_attn, cached_val2 = self.self_attn2.streaming_forward(
            src, attn_weights=attn_weights, cached_val=cached_val2, left_context_len=left_context_len
        )
        src = src + self_attn

        src_conv, cached_conv2 = self.conv_module2.streaming_forward(
            src, cache=cached_conv2, src_key_padding_mask=src_key_padding_mask[:, left_context_len:]
        )
        src = src + src_conv

        src = src + self.feed_forward3(src)
        src = self.norm(src)
        src = self.bypass(src_orig, src)
        return (
            src, cached_key, cached_nonlin_attn, cached_val1, cached_val2,
            cached_conv1, cached_conv2,
        )


class Zipformer2Encoder(nn.Module):
    """A stack of ``num_layers`` Zipformer2EncoderLayers sharing one positional encoding."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int, pos_dim: int):
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, length_factor=1.0)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        pos_emb = self.encoder_pos(src)
        output = src
        for mod in self.layers:
            output = mod(
                output, pos_emb, chunk_size=chunk_size,
                attn_mask=attn_mask, src_key_padding_mask=src_key_padding_mask,
            )
        return output

    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        pos_emb = self.encoder_pos(src, left_context_len)
        output = src
        new_states: List[Tensor] = []
        for i, mod in enumerate(self.layers):
            (ck, cna, cv1, cv2, cc1, cc2) = states[i * 6 : (i + 1) * 6]
            (output, nck, ncna, ncv1, ncv2, ncc1, ncc2) = mod.streaming_forward(
                output, pos_emb,
                cached_key=ck, cached_nonlin_attn=cna, cached_val1=cv1, cached_val2=cv2,
                cached_conv1=cc1, cached_conv2=cc2,
                left_context_len=left_context_len, src_key_padding_mask=src_key_padding_mask,
            )
            new_states += [nck, ncna, ncv1, ncv2, ncc1, ncc2]
        return output, new_states


class DownsampledZipformer2Encoder(nn.Module):
    """Runs an inner encoder at a reduced frame rate, then upsamples and bypasses."""

    def __init__(self, encoder: nn.Module, dim: int, downsample: int):
        super().__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim, downsample)
        self.num_layers = encoder.num_layers
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = BypassModule(dim)

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]
        src = self.encoder(
            src, chunk_size=chunk_size // ds, attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        src = src[: src_orig.shape[0]]
        return self.out_combiner(src_orig, src)

    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        src_orig = src
        src = self.downsample(src)
        src, new_states = self.encoder.streaming_forward(
            src, states=states, left_context_len=left_context_len,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        src = src[: src_orig.shape[0]]
        return self.out_combiner(src_orig, src), new_states


class Zipformer2(nn.Module):
    """The Zipformer2 acoustic encoder (multi-rate U-net of encoder stacks)."""

    def __init__(
        self,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int, ...] = (2, 4),
        encoder_dim: Union[int, Tuple[int, ...]] = 384,
        num_encoder_layers: Union[int, Tuple[int, ...]] = 4,
        query_head_dim: Union[int, Tuple[int, ...]] = 24,
        pos_head_dim: Union[int, Tuple[int, ...]] = 4,
        value_head_dim: Union[int, Tuple[int, ...]] = 12,
        num_heads: Union[int, Tuple[int, ...]] = 8,
        feedforward_dim: Union[int, Tuple[int, ...]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int, ...]] = 31,
        pos_dim: int = 192,
        causal: bool = False,
        chunk_size: Tuple[int, ...] = (-1,),
        left_context_frames: Tuple[int, ...] = (-1,),
    ):
        super().__init__()
        n = len(downsampling_factor)
        self.output_downsampling_factor = output_downsampling_factor
        self.downsampling_factor = downsampling_factor
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim, n)
        self.num_encoder_layers = num_encoder_layers = _to_tuple(num_encoder_layers, n)
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim, n)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim, n)
        pos_head_dim = _to_tuple(pos_head_dim, n)
        self.num_heads = num_heads = _to_tuple(num_heads, n)
        feedforward_dim = _to_tuple(feedforward_dim, n)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel, n)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        encoders = []
        for i in range(n):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )
            encoder = Zipformer2Encoder(encoder_layer, num_encoder_layers[i], pos_dim=pos_dim)
            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder, dim=encoder_dim[i], downsample=downsampling_factor[i]
                )
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        self.downsample_output = SimpleDownsample(
            max(encoder_dim), downsample=output_downsampling_factor
        )

    def _get_full_dim_output(self, outputs: List[Tensor]) -> Tensor:
        num_encoders = len(self.encoder_dim)
        output_dim = max(self.encoder_dim)
        output_pieces = [outputs[-1]]
        cur_dim = self.encoder_dim[-1]
        for i in range(num_encoders - 2, -1, -1):
            d = self.encoder_dim[i]
            if d > cur_dim:
                output_pieces.append(outputs[i][..., cur_dim:d])
                cur_dim = d
        assert cur_dim == output_dim
        return torch.cat(output_pieces, dim=-1)

    def forward(
        self, x: Tensor, x_lens: Tensor, src_key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """x: (seq_len, batch, feature_dim) -> (out_seq_len, batch, max(encoder_dim))."""
        outputs = []
        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])
            x = module(
                x,
                chunk_size=-1,
                src_key_padding_mask=(
                    None if src_key_padding_mask is None else src_key_padding_mask[..., ::ds]
                ),
                attn_mask=None,
            )
            outputs.append(x)

        x = self._get_full_dim_output(outputs)
        x = self.downsample_output(x)
        assert self.output_downsampling_factor == 2
        lengths = (x_lens + 1) // 2
        return x, lengths

    def streaming_forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        states: List[Tensor],
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        outputs = []
        new_states: List[Tensor] = []
        layer_offset = 0
        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])
            x, new_layer_states = module.streaming_forward(
                x,
                states=states[layer_offset * 6 : (layer_offset + num_layers) * 6],
                left_context_len=self.left_context_frames[0] // ds,
                src_key_padding_mask=src_key_padding_mask[..., ::ds],
            )
            layer_offset += num_layers
            outputs.append(x)
            new_states += new_layer_states

        x = self._get_full_dim_output(outputs)
        x = self.downsample_output(x)
        assert self.output_downsampling_factor == 2
        lengths = (x_lens + 1) // 2
        return x, lengths, new_states

    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> List[Tensor]:
        """Per-layer cache: [cached_key, cached_nonlin_attn, cached_val1, cached_val2,
        cached_conv1, cached_conv2] x num_layers, flattened."""
        num_encoder_layers = _to_tuple(self.num_encoder_layers, len(self.encoders))
        num_heads_t = _to_tuple(self.num_heads, len(self.encoders))
        query_head_dim_t = _to_tuple(self.query_head_dim, len(self.encoders))
        value_head_dim_t = _to_tuple(self.value_head_dim, len(self.encoders))
        states: List[Tensor] = []

        def z(*shape):
            return torch.zeros(*shape, device=device, dtype=dtype)

        for i in range(len(self.encoders)):
            num_layers = num_encoder_layers[i]
            embed_dim = self.encoder_dim[i]
            ds = self.downsampling_factor[i]
            num_heads = num_heads_t[i]
            key_dim = query_head_dim_t[i] * num_heads
            value_dim = value_head_dim_t[i] * num_heads
            downsample_left = self.left_context_frames[0] // ds
            nonlin_attn_head_dim = 3 * embed_dim // 4
            conv_left_pad = self.cnn_module_kernel[i] // 2
            for _ in range(num_layers):
                states += [
                    z(downsample_left, batch_size, key_dim),
                    z(1, batch_size, downsample_left, nonlin_attn_head_dim),
                    z(downsample_left, batch_size, value_dim),
                    z(downsample_left, batch_size, value_dim),
                    z(batch_size, embed_dim, conv_left_pad),
                    z(batch_size, embed_dim, conv_left_pad),
                ]
        return states
