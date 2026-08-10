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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from oasr.layers import (
    TORCH_EPS,
    Attention,
    ColumnParallelLinear,
    Conv1d,
    Embedding,
    LayerNorm,
    RowParallelLinear,
)

from ..base import BaseAsrModel, BaseEncoder, LoadReport
from ..decoders.base import BaseDecoder, DecoderState
from .config import WhisperModelConfig

logger = logging.getLogger(__name__)


class _WhisperAttention(nn.Module):
    """HF-layout MHA (``k_proj`` bias-free) over the shared attention core.

    Projections keep HF's names so the checkpoint loads 1:1; the compute is
    :class:`oasr.layers.Attention`, shared with every other architecture.
    """

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.h = n_head
        self.d_k = d_model // n_head
        self.q_proj = ColumnParallelLinear(d_model, d_model)
        self.k_proj = ColumnParallelLinear(d_model, d_model, bias=False)
        self.v_proj = ColumnParallelLinear(d_model, d_model)
        self.out_proj = RowParallelLinear(d_model, d_model)
        # Whisper attention is never masked (the 30 s window is real input and
        # generation is causal), so the shared core routes it to SDPA — see the
        # measurement table in ``oasr/layers/attention/core.py``.
        self.attn = Attention(n_head, self.d_k)

    def kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project keys/values only (cross-attention prefill / cache append)."""
        return self.attn.split_heads(self.k_proj(x)), self.attn.split_heads(self.v_proj(x))

    def forward(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """``query (B, T_q, D)`` × pre-projected ``k``/``v`` ``(B, h, T_k, d_k)``."""
        q = self.attn.split_heads(self.q_proj(query))
        x = self.attn(q, k, v, is_causal=is_causal)
        return self.out_proj(self.attn.merge_heads(x))


class _EncoderLayer(nn.Module):
    """``fc1``/``fc2`` stay flat rather than becoming a ``FeedForward``: HF puts
    them directly on the layer, and nesting them would add a level to every
    checkpoint key.  GELU is the exact erf form (HF's ``activation_function:
    gelu``), which is why it is not folded into the GEMM epilogue — the OASR
    fused epilogue implements the tanh approximation."""

    def __init__(self, cfg: WhisperModelConfig) -> None:
        super().__init__()
        self.self_attn = _WhisperAttention(cfg.d_model, cfg.encoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(cfg.d_model, eps=TORCH_EPS)
        self.fc1 = ColumnParallelLinear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = RowParallelLinear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = LayerNorm(cfg.d_model, eps=TORCH_EPS)

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
        self.conv1 = Conv1d(cfg.num_mel_bins, cfg.d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1)
        # HF materializes the sinusoidal table as a real (frozen) weight.
        self.embed_positions = Embedding(cfg.max_source_positions, cfg.d_model)
        self.layers = nn.ModuleList([_EncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        self.layer_norm = LayerNorm(cfg.d_model, eps=TORCH_EPS)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, 3000, n_mels)`` log-mel → ``(hidden (B, 1500, D), mask (B, 1, 1500))``.

        Whisper consumes the fixed 30 s window as real input — the mask is
        always full (padding is part of the recipe, not attention masking).
        """
        del xs_lens
        x = F.gelu(self.conv1(xs))
        x = F.gelu(self.conv2(x))
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
    def output_size(self) -> int:
        return self._cfg.d_model

    @property
    def subsampling_rate(self) -> int:
        return 2


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: WhisperModelConfig) -> None:
        super().__init__()
        self.self_attn = _WhisperAttention(cfg.d_model, cfg.decoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(cfg.d_model, eps=TORCH_EPS)
        self.encoder_attn = _WhisperAttention(cfg.d_model, cfg.decoder_attention_heads)
        self.encoder_attn_layer_norm = LayerNorm(cfg.d_model, eps=TORCH_EPS)
        self.fc1 = ColumnParallelLinear(cfg.d_model, cfg.decoder_ffn_dim)
        self.fc2 = RowParallelLinear(cfg.decoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = LayerNorm(cfg.d_model, eps=TORCH_EPS)


class _CrossAttnCollector:
    """Materialises the cross-attention of a declared ``(layer, head)`` set.

    Slices the query to the requested heads **before** the score matmul and
    truncates the key axis to the real audio right after the softmax, so the
    only ``(heads, tokens, frames)`` tensors that ever exist are the ones the
    DTW will read.  Collecting whole layers instead would cost ~200 MB of
    transient on ``large`` with the all-heads fallback, for data that is then
    thrown away.
    """

    def __init__(self, heads: Sequence[Tuple[int, int]], max_frames: Optional[int] = None) -> None:
        self._order = [(int(layer), int(head)) for layer, head in heads]
        self._by_layer: Dict[int, List[int]] = {}
        for layer, head in self._order:
            self._by_layer.setdefault(layer, []).append(head)
        self._max_frames = max_frames
        self._got: Dict[Tuple[int, int], torch.Tensor] = {}

    def capture(
        self,
        layer_idx: int,
        attn: "_WhisperAttention",
        query: torch.Tensor,
        cross_k: torch.Tensor,
    ) -> None:
        wanted = self._by_layer.get(layer_idx)
        if not wanted:
            return
        idx = torch.tensor(sorted(set(wanted)), device=query.device)
        q = attn.attn.split_heads(attn.q_proj(query)).index_select(1, idx)  # (B, n, T, d_k)
        k = cross_k.index_select(1, idx).to(q.dtype)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * attn.attn.softmax_scale
        # Softmax over **all** keys — that is the distribution the model used —
        # then keep only the frames that are real audio rather than 30 s padding.
        probs = scores.softmax(dim=-1)
        if self._max_frames is not None:
            probs = probs[..., : max(1, int(self._max_frames))]
        for slot, head in enumerate(sorted(set(wanted))):
            self._got[(layer_idx, head)] = probs[:, slot]

    def stacked(self) -> torch.Tensor:
        """``(B, len(heads), T_tok, F)`` in the order the heads were requested."""
        return torch.stack([self._got[pair] for pair in self._order], dim=1)


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
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_positions = Embedding(cfg.max_target_positions, cfg.d_model)
        self.layers = nn.ModuleList([_DecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.layer_norm = LayerNorm(cfg.d_model, eps=TORCH_EPS)

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
        collect: Optional["_CrossAttnCollector"] = None,
    ) -> torch.Tensor:
        """Shared prefill/step forward over ``ids (B, T)`` starting at ``offset``.

        ``collect`` (word timestamps only) additionally materialises the
        cross-attention probabilities of a declared set of heads.  It is checked
        once per layer and is ``None`` for every decode step, so the generation
        path is unchanged; the alignment pass is a separate teacher-forced
        forward run after a row finishes.
        """
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
            if collect is not None:
                # ``nn.Module.__getattr__`` types every submodule as
                # ``Tensor | Module``; the cast is the same one the transducer
                # strategy's ``_surface`` makes, for the same reason.
                collect.capture(
                    i, cast(_WhisperAttention, layer.encoder_attn), h, state["cross_k"][i]
                )
            x = residual + layer.encoder_attn(h, state["cross_k"][i], state["cross_v"][i])
            residual = x
            h = layer.final_layer_norm(x)
            x = residual + layer.fc2(F.gelu(layer.fc1(h)))
        x = self.layer_norm(x)
        return x @ self.embed_tokens.weight.t()  # tied projection → (B, T, V)

    @torch.no_grad()
    def cross_attention(
        self,
        enc_out: torch.Tensor,
        token_ids: torch.Tensor,
        heads: Sequence[Tuple[int, int]],
        max_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced pass returning the alignment heads' attention + logits.

        ``(B, len(heads), T_tok, F)`` cross-attention probabilities in the order
        ``heads`` was given, plus the ``(B, T_tok, V)`` logits of the same pass —
        which is where the per-token posteriors for ``confidence`` come from, at
        no extra cost.

        A **second forward** rather than a hook on generation: the decode step is
        the engine's hottest AR path and a request that wants timings is the
        exception, so the cost lands on that request instead of on every step of
        every request.  One prompt-length forward next to the N steps that
        produced the transcript is a small fraction of the work already done.
        """
        n = len(self.layers)
        state: Dict[str, Any] = {
            "self_k": [None] * n,
            "self_v": [None] * n,
            "cross_k": [None] * n,
            "cross_v": [None] * n,
            "pos": 0,
        }
        for i, layer in enumerate(self.layers):
            attn = cast(_WhisperAttention, layer.encoder_attn)
            state["cross_k"][i], state["cross_v"][i] = attn.kv(enc_out)
        collector = _CrossAttnCollector(heads, max_frames)
        logits = self._forward_tokens(token_ids, 0, state, is_prefill=True, collect=collector)
        return collector.stacked(), logits

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

    @property
    def default_decode_type(self) -> str:
        return "aed"

    @property
    def capabilities(self) -> frozenset:
        """Declared, not derived: the conformance test in
        ``tests/test_model_contract.py`` checks every registered architecture's
        advertised capabilities against ``oasr.models.interfaces.CAPABILITIES``,
        and can only do that without building the model when it is a constant."""
        return frozenset({"aed"})

    def __init__(self, config: WhisperModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)

    @classmethod
    def from_config(cls, config: WhisperModelConfig, **aux: Any) -> "WhisperModel":
        del aux
        return cls(config)

    @property
    def decoder_cache_spec(self):
        """Per-layer KV geometry of the **decoder**, for admission budgeting (C3).

        Distinct from ``cache_spec``, which describes the *encoder* paged-KV
        layout the streaming backend sizes.  Whisper is offline-only, so it has
        no encoder cache spec at all — but its AR decoder still allocates KV per
        generated token, and that is what bounds how many rows can be in flight.

        Self-attention only: cross-attention KV is computed once from the
        encoder output and does not grow per token, so it does not belong in a
        per-token rate.
        """
        from oasr.models.base import CacheSpec

        cfg = self.config
        return CacheSpec(
            num_layers=int(cfg.decoder_layers),
            n_kv_head=int(cfg.decoder_attention_heads),
            head_dim=int(cfg.d_model) // int(cfg.decoder_attention_heads),
            hidden_dim=int(cfg.d_model),
        )

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
        return LoadReport.build(sd, missing, unexpected, dropped)
