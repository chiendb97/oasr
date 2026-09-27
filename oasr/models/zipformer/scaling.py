# Copyright    2023  Xiaomi Corp.        (authors: Daniel Povey, Zengwei Yao)
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only port of the Zipformer ``scaling.py`` primitives from icefall.

Only the modules that carry parameters or affect the *forward* (inference) path
are kept; training-only machinery (ScaledAdam scaling, Balancer, Whiten,
ScaleGrad, Dropout, ScheduledFloat, custom autograd) is dropped or reduced to an
identity, since at eval time those are no-ops.  Module + parameter names mirror
icefall exactly so that an icefall checkpoint loads with a 1:1 key mapping.

Compute goes through OASR CUDA kernels: ``oasr.swoosh_l`` / ``oasr.swoosh_r``
(Swoosh activations), ``oasr.bias_norm`` (BiasNorm), ``oasr.depthwise_conv1d``
(the causal depthwise convs), and ``oasr.gemm`` (the fused activation+linear).
The module is therefore CUDA-only and runs in FP16 / BF16.

Reference:
https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/scaling.py
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import oasr
from oasr.layers._backend import use_gemm_kernel
from oasr.layers.conv import DepthwiseConv1d

# Re-export the shared BiasNorm under the checkpoint-compatible import path and
# parameter names (``bias``, ``log_scale``).
from oasr.layers.norm import BiasNorm as BiasNorm


class SwooshL(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return oasr.swoosh_l(x)


class SwooshR(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return oasr.swoosh_r(x)


class ChunkCausalDepthwiseConv1d(nn.Module):
    """Depthwise 1d conv that is causal in a chunkwise way (causal Zipformer).

    Implemented as a half-width causal conv plus a within-chunk conv scaled by a
    learnable position-in-chunk correction.  Faithful port of the icefall module
    (inference + streaming).  Parameter names (``causal_conv``, ``chunkwise_conv``,
    ``chunkwise_conv_scale``) match icefall.

    Unlike icefall this works in ``(N, T, C)``, the ``oasr.depthwise_conv1d``
    kernel's own layout, so neither conv needs a transpose: the causal left
    context is a padding argument offline and the cache when streaming, and
    splitting the time axis into chunks is a free view rather than a permute.
    The streaming cache is ``(N, left_pad, C)`` accordingly.
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        half_kernel_size = (kernel_size + 1) // 2
        # causal_conv: a "valid" (padding=0) conv over its left context -- the
        # cache when streaming, a ``padding=(left_pad, 0)`` override offline.
        # chunkwise_conv: symmetric padding keeps the length.
        self.causal_conv = DepthwiseConv1d(
            channels=channels, kernel_size=half_kernel_size, padding=0, bias=True
        )
        self.chunkwise_conv = DepthwiseConv1d(
            channels=channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias
        )
        self.chunkwise_conv_scale = nn.Parameter(torch.zeros(2, channels, kernel_size))
        self.kernel_size = kernel_size

    def forward(self, x: Tensor, chunk_size: int = -1) -> Tensor:
        """``x``: contiguous ``(N, T, C)`` -> ``(N, T, C)``."""
        batch_size, seq_len, num_channels = x.shape
        left_pad = self.kernel_size // 2
        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len
        right_pad = -seq_len % chunk_size

        x_causal = self.causal_conv(x, padding=(left_pad, 0))

        x_chunk = F.pad(x, (0, 0, 0, right_pad)) if right_pad else x
        num_chunks = x_chunk.shape[1] // chunk_size
        x_chunk = self.chunkwise_conv(
            x_chunk.view(batch_size * num_chunks, chunk_size, num_channels)
        )  # does not change shape
        chunk_scale = self._get_chunk_scale(chunk_size)  # (chunk_size, C)
        if right_pad:
            x_chunk = (x_chunk * chunk_scale).view(batch_size, -1, num_channels)[:, :seq_len]
            return x_chunk + x_causal
        # x_causal + x_chunk * chunk_scale, one kernel.
        shape4 = (batch_size, num_chunks, chunk_size, num_channels)
        return torch.addcmul(x_causal.view(shape4), x_chunk.view(shape4), chunk_scale).view(
            batch_size, seq_len, num_channels
        )

    def _get_chunk_scale(self, chunk_size: int) -> Tensor:
        """The ``(chunk_size, C)`` position-in-chunk scale (a transposed view)."""
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
        return (1.0 + (left_edge + right_edge)).t()

    def streaming_forward(self, x: Tensor, cache: Tensor) -> Tuple[Tensor, Tensor]:
        """``x``: contiguous ``(N, T, C)``; ``cache``: ``(N, left_pad, C)``."""
        left_pad = self.kernel_size // 2
        assert cache.shape[1] == left_pad, (cache.shape[1], left_pad)
        x_pad = torch.cat([cache, x], dim=1)
        cache = x_pad[:, -left_pad:]

        x_causal = self.causal_conv(x_pad)
        # The within-chunk conv sees only this chunk -- which is ``x`` itself,
        # already contiguous, not the cache-prefixed buffer.
        x_chunk = self.chunkwise_conv(x)
        chunk_scale = self._get_chunk_scale(chunk_size=x.shape[1])
        return torch.addcmul(x_causal, x_chunk, chunk_scale), cache


class ActivationDropoutAndLinear(nn.Module):
    """Swoosh activation followed by a linear layer (dropout is a no-op at eval).

    Stores ``weight`` and ``bias`` directly (matching icefall), so checkpoint
    keys map without a ``.l.`` prefix.  The Swoosh runs on the OASR activation
    kernel and the linear on ``oasr.gemm``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
    ):
        super().__init__()
        linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = linear.weight
        self.register_parameter("bias", linear.bias)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "SwooshL":
            x = oasr.swoosh_l(x)
        elif self.activation == "SwooshR":
            x = oasr.swoosh_r(x)
        else:
            raise ValueError(self.activation)
        # Holds bare ``weight``/``bias`` (icefall's key layout has no ``.l.``
        # level), so it cannot *be* an ``oasr.layers.Linear`` — but it goes
        # through the same backend decision.
        if use_gemm_kernel(x, self.in_channels, self.out_channels):
            return oasr.gemm(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    """Pad (with zeros) or truncate the last dim of ``x`` to ``num_channels``."""
    if num_channels <= x.shape[-1]:
        return x[..., :num_channels]
    shape = list(x.shape)
    shape[-1] = num_channels - shape[-1]
    zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, zeros), dim=-1)
