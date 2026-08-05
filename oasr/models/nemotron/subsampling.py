# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron causal depthwise-separable Conv2d subsampling (8x).

Three stride-2 stages over the ``(time, frequency)`` plane, all **causally**
padded on the time axis (``left = kernel - 1``, ``right = stride - 1``) so the
same weights serve streaming, and symmetrically-ish padded on the frequency
axis.  The stem is a dense ``1 -> 256`` convolution; the two stages after it are
depthwise + pointwise, which is what makes an 8x front-end cheap enough for a
streaming model.

Everything here stays in **NHWC** ``(B, T, F, C)``, the layout OASR's conv2d
kernel wants, rather than upstream's NCHW.  Two consequences:

* the flattened width handed to the projection is ordered ``(f, c)`` — the
  natural NHWC flatten — while upstream's ``transpose(1, 2).reshape`` produces
  ``(c, f)``.  :meth:`NemotronModel.load_weights` permutes the projection's
  input axis once at load time, so the forward path pays nothing.  This is the
  same trick (and the same reason) as ``Conv2dSubsampling``'s ``_version = 2``
  migration on the Conformer side;
* the ``(B, T, F, 1)`` stem input is just ``mel.unsqueeze(-1)``: no copy.

Per-stage time masking is not optional.  The mel features are zero past each
row's length, but a convolution with a bias turns zeros into a *nonzero*
constant, and the ``stride - 1`` right pad means the last valid output frame of a
short row reads one frame beyond it.  Re-zeroing after every stage is what keeps
a mixed-length batch from feeding the encoder invented energy.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from oasr.layers import Conv2d, Conv2dActivation, Linear

from .config import NemotronEncoderConfig


def _mask_time(x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero NHWC frames at or past each row's valid length."""
    if lengths is None:
        return x
    keep = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    return x * keep[:, :, None, None]


class _CausalPad:
    """Padding geometry shared by every stage: causal in time, both-sides in freq."""

    def __init__(self, kernel: int, stride: int) -> None:
        self.kernel = kernel
        self.stride = stride
        #: ``(left, right)`` on the time axis — NeMo's ``CausalConv2D``.
        self.time = (kernel - 1, stride - 1)
        #: ``(left, right)`` on the frequency axis.
        self.freq = (kernel - 1, stride - 1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Pad NHWC ``(B, T, F, C)``: ``F.pad`` counts dims from the last."""
        return F.pad(x, (0, 0, self.freq[0], self.freq[1], self.time[0], self.time[1]))

    def out_length(self, lengths: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if lengths is None:
            return None
        return (lengths + self.time[0] + self.time[1] - self.kernel) // self.stride + 1


class NemotronSubsampling(nn.Module):
    """``(B, T, n_mels)`` mel → ``(B, T // 8 + k, hidden_size)`` embeddings."""

    def __init__(self, config: NemotronEncoderConfig) -> None:
        super().__init__()
        channels = config.subsampling_conv_channels
        kernel = config.subsampling_conv_kernel_size
        stride = config.subsampling_conv_stride
        self._pad = _CausalPad(kernel, stride)

        # ReLU folds into the stem's epilogue: the per-stage mask multiplies by
        # 0/1 and ``relu(0) == 0``, so masking before or after the activation is
        # the same function and the fusion is free.
        self.conv_in = Conv2dActivation(
            1, channels, kernel_size=kernel, stride=stride, activation_type="relu"
        )
        self.layers = nn.ModuleList(
            _SubsamplingStage(channels, kernel, stride)
            for _ in range(config.num_subsampling_layers - 1)
        )
        self.linear = Linear(config.subsampling_out_hidden_size, config.hidden_size)

    def forward(
        self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``features (B, T, n_mels)`` → ``(embeddings (B, T', hidden), lengths)``."""
        x = features.unsqueeze(-1)  # (B, T, F, 1) NHWC, no copy
        x = self.conv_in(self._pad.apply(x))
        lengths = self._pad.out_length(lengths)
        x = _mask_time(x, lengths)
        for stage in self.layers:
            x = stage(self._pad.apply(x))
            lengths = self._pad.out_length(lengths)
            x = F.relu(_mask_time(x, lengths))
        b, t, f, c = x.shape
        return self.linear(x.reshape(b, t, f * c)), lengths

    def flatten_order(self) -> Tuple[int, int]:
        """``(channels, freq_bins)`` of the pre-projection flatten.

        Exposed so ``load_weights`` can convert upstream's ``(c, f)`` column
        order into this module's ``(f, c)`` one without re-deriving the shapes.
        """
        channels = self.conv_in.out_channels
        return channels, self.linear.in_features // channels


class _SubsamplingStage(nn.Module):
    """Depthwise stride-2 Conv2d + 1x1 pointwise Conv2d (NHWC).

    The depthwise convolution has **no kernel**: ``csrc/conv2d.cu`` takes no
    ``groups`` argument, so ``oasr.layers.Conv2d`` records the ``conv2d-groups``
    gap and serves it with ``F.conv2d``.  Reaching for ``nn.Conv2d`` here instead
    would work identically and count nothing — see ``.artifacts/kernel_coverage.md``
    §0 for why that distinction is the point.
    """

    def __init__(self, channels: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.depthwise_conv = Conv2d(
            channels, channels, kernel_size=kernel, stride=stride, groups=channels
        )
        self.pointwise_conv = Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.pointwise_conv(self.depthwise_conv(x))
        return out


__all__: List[str] = ["NemotronSubsampling"]
