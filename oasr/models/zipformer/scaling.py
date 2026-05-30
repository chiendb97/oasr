# Copyright    2023  Xiaomi Corp.        (authors: Daniel Povey, Zengwei Yao)
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only port of the Zipformer ``scaling.py`` primitives from icefall.

Only the modules that carry parameters or affect the *forward* (inference) path
are kept; training-only machinery (ScaledAdam scaling, Balancer, Whiten,
ScaleGrad, Dropout, ScheduledFloat, custom autograd) is dropped or reduced to an
identity, since at eval time those are no-ops.  Module + parameter names mirror
icefall exactly so that an icefall checkpoint loads with a 1:1 key mapping.

Reference:
https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/scaling.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def SwooshLForward(x: Tensor) -> Tensor:
    """``swoosh_l(x) = log(1 + exp(x - 4)) - 0.08 x - 0.035``."""
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035


def SwooshRForward(x: Tensor) -> Tensor:
    """``swoosh_r(x) = log(1 + exp(x - 1)) - 0.08 x - 0.313261687``."""
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


class SwooshL(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return SwooshLForward(x)


class SwooshR(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return SwooshRForward(x)


class BiasNorm(nn.Module):
    """A cheaper replacement for LayerNorm with a learnable bias + scalar scale.

    ``scales = mean((x - bias)^2, dim=channel_dim) ** -0.5 * exp(log_scale)``;
    output is ``x * scales``.  At inference ``log_scale`` is used as-is (the
    training-time clamping is a no-op once the parameter is within range).
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        log_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(num_channels).normal_(mean=0, std=1e-4))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        channel_dim = self.channel_dim
        if channel_dim < 0:
            channel_dim += x.ndim
        bias = self.bias
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * self.log_scale.exp()
        return x * scales


class ChunkCausalDepthwiseConv1d(nn.Module):
    """Depthwise 1d conv that is causal in a chunkwise way (causal Zipformer).

    Implemented as a half-width causal conv plus a within-chunk conv scaled by a
    learnable position-in-chunk correction.  Faithful port of the icefall module
    (inference + streaming).  Parameter names (``causal_conv``, ``chunkwise_conv``,
    ``chunkwise_conv_scale``) match icefall.
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        half_kernel_size = (kernel_size + 1) // 2
        self.causal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=half_kernel_size,
            padding=0,
            bias=True,
        )
        self.chunkwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.chunkwise_conv_scale = nn.Parameter(torch.zeros(2, channels, kernel_size))
        self.kernel_size = kernel_size

    def forward(self, x: Tensor, chunk_size: int = -1) -> Tensor:
        (batch_size, num_channels, seq_len) = x.shape
        left_pad = self.kernel_size // 2
        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len
        right_pad = -seq_len % chunk_size

        x = torch.nn.functional.pad(x, (left_pad, right_pad))

        x_causal = self.causal_conv(x[..., : left_pad + seq_len])
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        num_chunks = x_chunk.shape[2] // chunk_size
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks, chunk_size)
        x_chunk = x_chunk.permute(0, 2, 1, 3).reshape(
            batch_size * num_chunks, num_channels, chunk_size
        )
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size)
        x_chunk = x_chunk * chunk_scale
        x_chunk = x_chunk.reshape(
            batch_size, num_chunks, num_channels, chunk_size
        ).permute(0, 2, 1, 3)
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks * chunk_size)[
            ..., :seq_len
        ]
        return x_chunk + x_causal

    def _get_chunk_scale(self, chunk_size: int) -> Tensor:
        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = torch.zeros(channels, t, device=left_edge.device, dtype=left_edge.dtype)
            left_edge = torch.cat((left_edge, pad), dim=-1)
            right_edge = torch.cat((pad, right_edge), dim=-1)
        return 1.0 + (left_edge + right_edge)

    def streaming_forward(self, x: Tensor, cache: Tensor):
        (batch_size, num_channels, seq_len) = x.shape
        left_pad = self.kernel_size // 2
        assert cache.shape[-1] == left_pad, (cache.shape[-1], left_pad)
        x = torch.cat([cache, x], dim=2)
        cache = x[..., -left_pad:]

        x_causal = self.causal_conv(x)
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        x_chunk = self.chunkwise_conv(x_chunk)
        chunk_scale = self._get_chunk_scale(chunk_size=seq_len)
        x_chunk = x_chunk * chunk_scale
        return x_chunk + x_causal, cache


class ActivationDropoutAndLinear(nn.Module):
    """Swoosh activation followed by a linear layer (dropout is a no-op at eval).

    Stores ``weight`` and ``bias`` directly (matching icefall), so checkpoint
    keys map without a ``.l.`` prefix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
    ):
        super().__init__()
        l = nn.Linear(in_channels, out_channels, bias=bias)
        self.weight = l.weight
        self.register_parameter("bias", l.bias)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "SwooshL":
            x = SwooshLForward(x)
        elif self.activation == "SwooshR":
            x = SwooshRForward(x)
        else:
            raise ValueError(self.activation)
        return torch.nn.functional.linear(x, self.weight, self.bias)


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    """Pad (with zeros) or truncate the last dim of ``x`` to ``num_channels``."""
    if num_channels <= x.shape[-1]:
        return x[..., :num_channels]
    shape = list(x.shape)
    shape[-1] = num_channels - shape[-1]
    zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, zeros), dim=-1)
