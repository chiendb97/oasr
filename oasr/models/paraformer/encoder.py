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

from typing import Optional, Tuple, cast

import torch
from torch import nn

from oasr.layers import LayerNorm
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
        self.norm1 = LayerNorm(in_size, eps=LAYER_NORM_EPS)
        self.norm2 = LayerNorm(size, eps=LAYER_NORM_EPS)

    def forward(
        self,
        h: torch.Tensor,
        residual: Optional[torch.Tensor],
        mask: torch.Tensor,
        lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one layer from an already-normalized input.

        Returns the FFN output separately from the updated residual so the
        parent encoder can fold their addition into the following ``norm1``.
        The width-changing first layer passes ``residual=None`` because its
        attention branch has no shape-compatible residual.
        """
        attn = self.self_attn(h, mask, kv_lens=lens)
        if self.in_size == self.size:
            if residual is None:
                raise RuntimeError("same-width EncoderLayerSANM requires a residual stream")
            h, residual = self.norm2.forward_add_residual(attn, residual)
        else:
            residual = attn
            h = self.norm2(attn)

        return self.feed_forward(h), residual


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
        self.after_norm = LayerNorm(d, eps=LAYER_NORM_EPS)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, T, input_size)`` LFR features → ``(hidden (B, T, D), masks (B, 1, T))``."""
        B, T, _ = xs.shape
        lens = xs_lens.to(xs.device)
        masks = (torch.arange(T, device=xs.device).unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(
            1
        )  # (B, 1, T) bool

        xs = (xs + self.cmvn_shift) * self.cmvn_scale
        xs = xs * self._config.encoder_output_size**0.5
        xs = xs + sinusoidal_position_encoding(T, xs.size(-1), xs.device, xs.dtype)

        # The width-changing first SANM attention has no residual. Its FFN
        # residual can still fold into the next layer's first norm (or the
        # encoder's final norm when this is a one-layer configuration).
        first = cast(EncoderLayerSANM, self.encoders0[0])
        h = first.norm1(xs)
        ff, residual = first(h, None, masks, lens)

        encoders = [cast(EncoderLayerSANM, layer) for layer in self.encoders]
        if not encoders:
            return self.after_norm.forward_add(ff, residual), masks

        h, residual = encoders[0].norm1.forward_add_residual(ff, residual)
        for i, layer in enumerate(encoders):
            ff, residual = layer(h, residual, masks, lens)

            if i + 1 < len(encoders):
                h, residual = encoders[i + 1].norm1.forward_add_residual(ff, residual)
            else:
                xs = self.after_norm.forward_add(ff, residual)
        return xs, masks

    # -- BaseEncoder introspection ------------------------------------------
    @property
    def num_encoder_layers(self) -> int:
        return self._config.encoder_num_blocks

    @property
    def output_size(self) -> int:
        return self._config.encoder_output_size

    @property
    def subsampling_rate(self) -> int:
        # No subsampling inside the encoder — the 6× LFR decimation happens in
        # feature extraction.
        return 1
