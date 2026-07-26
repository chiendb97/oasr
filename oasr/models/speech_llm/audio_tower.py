# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Qwen2-Audio audio tower (Whisper-geometry encoder + ×2 average pooling).

Mirrors HF ``Qwen2AudioEncoder`` exactly: conv ×2 subsampling into the fixed
30 s window (3000 mel frames → 1500 positions), materialized sinusoidal
position table, pre-norm transformer layers with a **key-padding** attention
mask (queries are unrestricted — pad-position outputs are garbage but finite
and fall outside the valid length), then ``AvgPool1d(2)`` down to 750 frames
and a final LayerNorm.  Valid lengths follow HF's two-stage formula:
``feat = (mel - 1) // 2 + 1`` (conv) and ``out = (feat - 2) // 2 + 1`` (pool),
computed from the *real* per-row mel frame counts, so shorter utterances yield
proportionally fewer audio embeddings.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseEncoder
from .config import SpeechLlmModelConfig


class _TowerAttention(nn.Module):
    """HF Whisper-style MHA (``k_proj`` bias-free) with a key-padding mask."""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.h = n_head
        self.d_k = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """``x (B, T, D)``, ``key_mask (B, 1, 1, T)`` additive float mask."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.out_proj(out)


class _TowerLayer(nn.Module):
    def __init__(self, cfg: SpeechLlmModelConfig) -> None:
        super().__init__()
        d = cfg.audio_d_model
        self.self_attn = _TowerAttention(d, cfg.audio_encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.audio_encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.audio_encoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = residual + self.self_attn(x, key_mask)
        residual = x
        x = self.final_layer_norm(x)
        return residual + self.fc2(F.gelu(self.fc1(x)))


class Qwen2AudioTower(BaseEncoder):
    """Whisper-geometry encoder + avg-pool; parameter names mirror HF 1:1."""

    supports_packing = False
    supports_paged_streaming = False

    def __init__(self, cfg: SpeechLlmModelConfig) -> None:
        super().__init__()
        self._cfg = cfg
        d = cfg.audio_d_model
        self.conv1 = nn.Conv1d(cfg.audio_num_mel_bins, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        # HF materializes the sinusoidal table as a real (frozen) weight.
        self.embed_positions = nn.Embedding(cfg.audio_max_source_positions, d)
        self.layers = nn.ModuleList([_TowerLayer(cfg) for _ in range(cfg.audio_encoder_layers)])
        self.layer_norm = nn.LayerNorm(d)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

    @staticmethod
    def feat_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
        """Real mel frames → post-conv positions (HF two-stage formula, part 1)."""
        return (mel_lengths - 1) // 2 + 1

    @staticmethod
    def output_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
        """Real mel frames → post-pool audio-embedding count (part 2)."""
        return (Qwen2AudioTower.feat_lengths(mel_lengths) - 2) // 2 + 1

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, 3000, n_mels)`` log-mel + real frame counts → ``(hidden (B, 750, D),
        mask (B, 1, 750))``.

        The 30 s window is consumed whole (like Whisper); ``xs_lens`` carries
        the *real* per-row mel frame counts and drives the key-padding mask +
        the valid output lengths.
        """
        expected = self._cfg.audio_max_source_positions * 2
        if xs.size(1) != expected:
            raise ValueError(
                f"speech_llm expects {expected} mel frames (padded 30 s window, "
                f"feature_type='whisper_logmel'), got {xs.size(1)}"
            )
        x = xs.transpose(1, 2)  # (B, n_mels, T)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T/2, D)
        T = x.size(1)
        x = x + self.embed_positions.weight[:T].to(x.dtype)

        lens = xs_lens.to(xs.device)
        feat_lens = self.feat_lengths(lens)
        valid = torch.arange(T, device=xs.device).unsqueeze(0) < feat_lens.unsqueeze(1)
        # Additive key-padding mask (queries unrestricted — HF semantics).
        key_mask = torch.zeros(x.size(0), 1, 1, T, dtype=x.dtype, device=x.device)
        key_mask.masked_fill_(~valid.view(-1, 1, 1, T), float("-inf"))

        for layer in self.layers:
            x = layer(x, key_mask)

        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)

        out_lens = self.output_lengths(lens).clamp(min=0)
        T_out = x.size(1)
        masks = torch.arange(T_out, device=x.device).unsqueeze(0) < out_lens.unsqueeze(1)
        return x, masks.unsqueeze(1)

    # -- BaseEncoder introspection -----------------------------------------
    @property
    def num_encoder_layers(self) -> int:
        return self._cfg.audio_encoder_layers

    @property
    def output_size(self) -> int:
        return self._cfg.audio_d_model

    @property
    def subsampling_rate(self) -> int:
        return 4  # conv ×2 then avg-pool ×2
