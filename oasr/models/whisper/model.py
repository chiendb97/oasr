# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Whisper encoder-decoder ASR model (HF-checkpoint-compatible, pure PyTorch).

Module/parameter names mirror the HF ``WhisperModel`` layout with the
``model.`` prefix stripped (``encoder.conv1``, ``encoder.layers.N.self_attn.
k_proj``, ``decoder.embed_tokens``, …) so ``load_weights`` is a 1:1 copy.

Offline-only (``streaming_kind == "none"``): every utterance is a padded 30 s
log-mel window (see :func:`oasr.features.whisper.batched_whisper_logmel`) and
the encoder geometry is fixed at ``max_source_positions`` output frames.  The
decoder exposes the *batched incremental* surface the ``aed`` strategy drives:
:meth:`WhisperDecoder.prefill` (SOT prompt + cross-attention KV, computed
once) and :meth:`WhisperDecoder.step` (one token per active request), with a
dense per-layer KV cache carried in an opaque state dict.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseAsrModel, BaseEncoder, LoadReport
from ..decoders.base import BaseDecoder, DecoderState
from .config import WhisperModelConfig

logger = logging.getLogger(__name__)


class _WhisperAttention(nn.Module):
    """HF-compatible MHA (``k_proj`` bias-free, SDPA math)."""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.h = n_head
        self.d_k = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)  # (B, h, T, d_k)

    def kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project keys/values only (cross-attention prefill / cache append)."""
        return self._shape(self.k_proj(x)), self._shape(self.v_proj(x))

    def forward(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """``query (B, T_q, D)`` × pre-projected ``k``/``v`` ``(B, h, T_k, d_k)``."""
        B, T_q, _ = query.shape
        q = self._shape(self.q_proj(query))
        x = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        x = x.transpose(1, 2).contiguous().view(B, T_q, self.h * self.d_k)
        return self.out_proj(x)


class _EncoderLayer(nn.Module):
    def __init__(self, cfg: WhisperModelConfig) -> None:
        super().__init__()
        self.self_attn = _WhisperAttention(cfg.d_model, cfg.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        k, v = self.self_attn.kv(x)
        x = residual + self.self_attn(x, k, v)
        residual = x
        x = self.final_layer_norm(x)
        return residual + self.fc2(F.gelu(self.fc1(x)))


class WhisperEncoder(BaseEncoder):
    """Conv subsampling (×2) + sinusoidal positions + transformer stack."""

    supports_packing = False
    supports_paged_streaming = False

    def __init__(self, cfg: WhisperModelConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, cfg.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1)
        # HF materializes the sinusoidal table as a real (frozen) weight.
        self.embed_positions = nn.Embedding(cfg.max_source_positions, cfg.d_model)
        self.layers = nn.ModuleList([_EncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, 3000, n_mels)`` log-mel → ``(hidden (B, 1500, D), mask (B, 1, 1500))``.

        Whisper consumes the fixed 30 s window as real input — the mask is
        always full (padding is part of the recipe, not attention masking).
        """
        del xs_lens
        x = xs.transpose(1, 2)  # (B, n_mels, T)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T/2, D)
        T = x.size(1)
        if T > self._cfg.max_source_positions:
            raise ValueError(
                f"input produces {T} encoder frames > max_source_positions="
                f"{self._cfg.max_source_positions}; audio must be padded/trimmed "
                "to the 30 s Whisper window (feature_type='whisper_logmel')"
            )
        x = x + self.embed_positions.weight[:T].to(x.dtype)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        masks = torch.ones(x.size(0), 1, T, dtype=torch.bool, device=x.device)
        return x, masks

    # -- BaseEncoder introspection -----------------------------------------
    @property
    def num_encoder_layers(self) -> int:
        return self._cfg.encoder_layers

    @property
    def n_kv_head(self) -> int:
        return self._cfg.encoder_attention_heads

    @property
    def head_dim(self) -> int:
        return self._cfg.head_dim

    @property
    def output_size(self) -> int:
        return self._cfg.d_model

    @property
    def subsampling_rate(self) -> int:
        return 2


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: WhisperModelConfig) -> None:
        super().__init__()
        self.self_attn = _WhisperAttention(cfg.d_model, cfg.decoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.encoder_attn = _WhisperAttention(cfg.d_model, cfg.decoder_attention_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.decoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.decoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)


class WhisperDecoder(BaseDecoder):
    """Whisper text decoder with a batched incremental (prefill/step) surface.

    The KV state is a plain dict of dense per-layer tensors —
    ``self_k``/``self_v``: ``List[(B, h, t, d_k)]`` growing one position per
    step; ``cross_k``/``cross_v``: fixed, computed once at prefill.  Rows are
    dropped with :meth:`select` as requests finish (continuous batching).
    Paged-KV storage (``DecoderKVCacheManager``) is the planned optimization;
    the state layout is already per-request-row so the swap is local.
    """

    decode_type = "aed"

    def __init__(self, cfg: WhisperModelConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, cfg.d_model)
        self.layers = nn.ModuleList([_DecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> DecoderState:
        del batch_size, device, dtype
        return None  # state is created by prefill()

    # ------------------------------------------------------------------
    # Incremental decode surface (driven by the ``aed`` strategy)
    # ------------------------------------------------------------------

    def _forward_tokens(
        self,
        ids: torch.Tensor,
        offset: int,
        state: Dict[str, Any],
        is_prefill: bool,
    ) -> torch.Tensor:
        """Shared prefill/step forward over ``ids (B, T)`` starting at ``offset``."""
        B, T = ids.shape
        pos = torch.arange(offset, offset + T, device=ids.device)
        x = self.embed_tokens(ids) + self.embed_positions(pos).to(self.embed_tokens.weight.dtype)
        for i, layer in enumerate(self.layers):
            residual = x
            h = layer.self_attn_layer_norm(x)
            k_new, v_new = layer.self_attn.kv(h)
            if state["self_k"][i] is not None:
                k = torch.cat([state["self_k"][i], k_new], dim=2)
                v = torch.cat([state["self_v"][i], v_new], dim=2)
            else:
                k, v = k_new, v_new
            state["self_k"][i], state["self_v"][i] = k, v
            # Prefill attends causally within the prompt; a single-token step
            # attends the full cache (its row is the last query position).
            x = residual + layer.self_attn(h, k, v, is_causal=is_prefill and T > 1)
            residual = x
            h = layer.encoder_attn_layer_norm(x)
            x = residual + layer.encoder_attn(h, state["cross_k"][i], state["cross_v"][i])
            residual = x
            h = layer.final_layer_norm(x)
            x = residual + layer.fc2(F.gelu(layer.fc1(h)))
        x = self.layer_norm(x)
        return x @ self.embed_tokens.weight.t()  # tied projection → (B, T, V)

    def prefill(
        self, enc_out: torch.Tensor, prompt_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Start generation: cross-KV once + prompt forward.

        ``enc_out (B, T_enc, D)``, ``prompt_ids (B, P)`` (identical P across
        the batch — the SOT sequence) → ``(logits (B, V) at the last prompt
        position, state)``.
        """
        n = len(self.layers)
        state: Dict[str, Any] = {
            "self_k": [None] * n,
            "self_v": [None] * n,
            "cross_k": [None] * n,
            "cross_v": [None] * n,
            "pos": 0,
        }
        # Cross K/V project the *raw* encoder output (the decoder layer's
        # encoder_attn_layer_norm applies to the query side only).
        for i, layer in enumerate(self.layers):
            state["cross_k"][i], state["cross_v"][i] = layer.encoder_attn.kv(enc_out)
        logits = self._forward_tokens(prompt_ids, offset=0, state=state, is_prefill=True)
        state["pos"] = prompt_ids.size(1)
        return logits[:, -1], state

    def step(
        self, tokens: torch.Tensor, state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One generation step: ``tokens (B,)`` → ``(logits (B, V), state)``."""
        logits = self._forward_tokens(
            tokens.unsqueeze(1), offset=state["pos"], state=state, is_prefill=False
        )
        state["pos"] += 1
        return logits[:, -1], state

    @staticmethod
    def select(state: Dict[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
        """Drop finished rows: index-select every cached tensor along batch."""
        out: Dict[str, Any] = {"pos": state["pos"]}
        for key in ("self_k", "self_v", "cross_k", "cross_v"):
            out[key] = [t.index_select(0, keep) for t in state[key]]
        return out


class WhisperModel(BaseAsrModel):
    """Whisper for OASR: offline AED decoding via the incremental protocol."""

    def __init__(self, config: WhisperModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)

    @classmethod
    def from_config(cls, config: WhisperModelConfig, **aux: Any) -> "WhisperModel":
        del aux
        return cls(config)

    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> LoadReport:
        """Map an HF Whisper state-dict (``model.encoder.*`` / ``model.decoder.*``).

        ``proj_out.weight`` is tied to ``decoder.embed_tokens.weight`` and is a
        declared drop when the checkpoint materializes it.
        """
        sd = {}
        dropped = []
        for k, v in state_dict.items():
            key = k[len("model.") :] if k.startswith("model.") else k
            if key.startswith(("encoder.", "decoder.")):
                sd[key] = v
            else:
                dropped.append(k)
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        if unexpected:
            logger.warning("Unexpected keys in Whisper checkpoint: %s", unexpected[:8])
        if missing:
            logger.warning("Whisper model keys not filled: %s", missing[:8])
        mapped = [k for k in sd if k not in unexpected]
        return LoadReport(mapped=mapped, dropped=dropped, missing=list(missing))
