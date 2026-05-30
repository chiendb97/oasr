# Copyright    2023  Xiaomi Corp.        (authors: Daniel Povey, Zengwei Yao)
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only port of the Zipformer ``subsampling.py`` (encoder_embed) from icefall.

Module/parameter names mirror icefall so a checkpoint loads 1:1.  Training-only
modules (Balancer / Whiten / ScaleGrad / Dropout3) are replaced by identities;
crucially the ``nn.Sequential`` slots are preserved so the conv layers keep their
checkpoint indices (``conv.0`` / ``conv.4`` / ``conv.7``).

Reference:
https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/subsampling.py
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from .scaling import BiasNorm, SwooshL, SwooshR


class _Identity(nn.Module):
    """Placeholder for icefall's parameter-free train-time modules (keeps Sequential indices)."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt block used inside the Zipformer encoder_embed."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: int = 3,
        kernel_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        hidden_channels = channels * hidden_ratio

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=self.padding,
        )
        self.pointwise_conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.activation = SwooshL()
        self.pointwise_conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        bypass = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return bypass + x

    def streaming_forward(self, x: Tensor, cached_left_pad: Tensor) -> Tuple[Tensor, Tensor]:
        padding = self.padding
        T = x.size(2) - padding[0]
        bypass = x[:, :, :T, :]
        assert cached_left_pad.size(2) == padding[0], (cached_left_pad.size(2), padding[0])
        x = torch.cat([cached_left_pad, x], dim=2)
        cached_left_pad = x[:, :, T : padding[0] + T, :]

        x = torch.nn.functional.conv2d(
            x,
            weight=self.depthwise_conv.weight,
            bias=self.depthwise_conv.bias,
            padding=(0, padding[1]),
            groups=self.depthwise_conv.groups,
        )
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = bypass + x
        return x, cached_left_pad


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling to 1/2 length: (N, T, idim) -> (N, (T-7)//2, odim)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 8,
        layer2_channels: int = 32,
        layer3_channels: int = 128,
    ) -> None:
        assert in_channels >= 7
        super().__init__()

        # Sequential slots match icefall so conv params keep indices 0 / 4 / 7.
        self.conv = nn.Sequential(
            nn.Conv2d(1, layer1_channels, kernel_size=3, padding=(0, 1)),  # 0
            _Identity(),  # 1: ScaleGrad
            _Identity(),  # 2: Balancer
            SwooshR(),  # 3
            nn.Conv2d(layer1_channels, layer2_channels, kernel_size=3, stride=2),  # 4
            _Identity(),  # 5: Balancer
            SwooshR(),  # 6
            nn.Conv2d(layer2_channels, layer3_channels, kernel_size=3, stride=(1, 2)),  # 7
            _Identity(),  # 8: Balancer
            SwooshR(),  # 9
        )

        self.convnext = ConvNeXt(layer3_channels, kernel_size=(7, 7))

        self.out_width = (((in_channels - 1) // 2) - 1) // 2
        self.layer3_channels = layer3_channels

        self.out = nn.Linear(self.out_width * layer3_channels, out_channels)
        self.out_norm = BiasNorm(out_channels)

    def forward(self, x: Tensor, x_lens: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.unsqueeze(1)  # (N, 1, T, idim)
        x = self.conv(x)
        x = self.convnext(x)

        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, c * f)
        x = self.out(x)
        x = self.out_norm(x)

        x_lens = (x_lens - 7) // 2
        assert x.size(1) == x_lens.max().item(), (x.size(1), x_lens.max())
        return x, x_lens

    def streaming_forward(
        self, x: Tensor, x_lens: Tensor, cached_left_pad: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x, cached_left_pad = self.convnext.streaming_forward(x, cached_left_pad=cached_left_pad)

        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, c * f)
        x = self.out(x)
        x = self.out_norm(x)

        assert self.convnext.padding[0] == 3
        x_lens = (x_lens - 7) // 2 - 3
        assert x.size(1) == x_lens.max().item(), (x.shape, x_lens.max())
        return x, x_lens, cached_left_pad

    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Cached left padding for the ConvNeXt module: (N, C, left_pad, freq)."""
        left_pad = self.convnext.padding[0]
        freq = self.out_width
        channels = self.layer3_channels
        return torch.zeros(batch_size, channels, left_pad, freq, device=device, dtype=dtype)
