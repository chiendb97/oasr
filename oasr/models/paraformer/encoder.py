# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SANM encoder (FunASR ``SANMEncoder``) for Paraformer.

Offline-only (``streaming_kind == "none"``).  The module tree matches the
FunASR checkpoint key space exactly (``encoders0.0`` — the 560→512 first
layer, ``encoders.{0..N-2}``, ``after_norm``), plus two OASR-side CMVN buffers
(``cmvn_shift`` / ``cmvn_scale``) the converter fills from ``am.mvn`` — FunASR
applies CMVN in its frontend as ``(x + shift) * scale`` on the 560-dim LFR
features; here it is the first op of the encoder so it travels with the model
(and round-trips through the native format as ordinary buffers).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from oasr.models.base import BaseEncoder

from .config import ParaformerModelConfig
from .modules import (
    LAYER_NORM_EPS,
    EncoderFeedForward,
    SanmSelfAttention,
    sinusoidal_position_encoding,
)


class EncoderLayerSANM(nn.Module):
    """Pre-norm SANM encoder layer.

    The first layer projects ``in_size`` (560) → ``size`` (512) inside its
    fused QKV; per FunASR, its attention sublayer has **no residual** when
    ``in_size != size`` (the shapes wouldn't match).
    """

    def __init__(self, in_size: int, size: int, config: ParaformerModelConfig) -> None:
        super().__init__()
        self.in_size = in_size
        self.size = size
        self.self_attn = SanmSelfAttention(
            config.encoder_attention_heads,
            in_size,
            size,
            config.encoder_kernel_size,
            config.encoder_sanm_shift,
        )
        self.feed_forward = EncoderFeedForward(size, config.encoder_linear_units)
        self.norm1 = nn.LayerNorm(in_size, eps=LAYER_NORM_EPS)
        self.norm2 = nn.LayerNorm(size, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        if self.in_size == self.size:
            x = residual + x

        residual = x
        x = self.norm2(x)
        return residual + self.feed_forward(x)


class SANMEncoder(BaseEncoder):
    """FunASR SANM encoder: CMVN → ·√d → sinusoidal PE → SANM layers → LayerNorm."""

    def __init__(self, config: ParaformerModelConfig) -> None:
        super().__init__()
        self._config = config
        d = config.encoder_output_size
        self.register_buffer("cmvn_shift", torch.zeros(config.input_size))
        self.register_buffer("cmvn_scale", torch.ones(config.input_size))
        self.encoders0 = nn.ModuleList([EncoderLayerSANM(config.input_size, d, config)])
        self.encoders = nn.ModuleList(
            [EncoderLayerSANM(d, d, config) for _ in range(config.encoder_num_blocks - 1)]
        )
        self.after_norm = nn.LayerNorm(d, eps=LAYER_NORM_EPS)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, T, input_size)`` LFR features → ``(hidden (B, T, D), masks (B, 1, T))``."""
        B, T, _ = xs.shape
        masks = (
            torch.arange(T, device=xs.device).unsqueeze(0) < xs_lens.to(xs.device).unsqueeze(1)
        ).unsqueeze(
            1
        )  # (B, 1, T) bool

        xs = (xs + self.cmvn_shift) * self.cmvn_scale
        xs = xs * self._config.encoder_output_size**0.5
        xs = xs + sinusoidal_position_encoding(T, xs.size(-1), xs.device, xs.dtype)

        xs = self.encoders0[0](xs, masks)
        for layer in self.encoders:
            xs = layer(xs, masks)
        return self.after_norm(xs), masks

    # -- BaseEncoder introspection ------------------------------------------
    @property
    def num_encoder_layers(self) -> int:
        return self._config.encoder_num_blocks

    @property
    def n_kv_head(self) -> int:
        return self._config.encoder_attention_heads

    @property
    def head_dim(self) -> int:
        return self._config.encoder_output_size // self._config.encoder_attention_heads

    @property
    def output_size(self) -> int:
        return self._config.encoder_output_size

    @property
    def subsampling_rate(self) -> int:
        # No subsampling inside the encoder — the 6× LFR decimation happens in
        # feature extraction.
        return 1
