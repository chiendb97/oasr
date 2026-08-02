# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Qwen2 decoder-only language model with a batched incremental surface.

Faithful port of HF ``Qwen2ForCausalLM`` inference math — RMSNorm with fp32
variance, rotary embeddings (GPT-NeoX half rotation, fp32 tables), GQA
(``num_key_value_heads <= num_attention_heads``), biased QKV / bias-free
output projections, SiLU gate-up-down MLP — exposing the same
``prefill`` / ``step`` / ``select`` surface as
:class:`~oasr.models.whisper.model.WhisperDecoder`, extended for
**variable-length, left-padded** prompts:

* :meth:`prefill` takes ``inputs_embeds`` (audio embeddings are spliced into
  the prompt *before* the LM — there is no cross-attention) plus a validity
  mask; per-row rotary positions follow HF's masked-generate convention
  (``cumsum(mask) - 1``, left pads clamped to 0);
* :meth:`step` appends one token per active row at that row's own next
  position, attending the full cache through a key-padding mask.

The KV state is a plain dict of dense per-layer tensors; rows are dropped with
:meth:`select` as requests finish (continuous batching).  When the caller
passes ``capacity`` to :meth:`prefill` (the ``llm`` strategy does — prompt
length + the batch's generation cap), the per-layer K/V buffers are
**preallocated** to that capacity and each step writes its one token slot in
place — removing the per-step ``torch.cat`` that re-copies the whole cache
(measured ~10% of a 7B decode step at B=4, growing with B).  Without
``capacity`` the legacy cat-growth path is used (direct ``prefill``/``step``
callers, tests).  Paged-KV storage (``DecoderKVCacheManager`` + the paged
FMHA) remains blocked on the CuteDSL masked-tile fix — left-padded prompts
are exactly the heavily key-padded batch shape that kernel currently NaNs on,
so attention stays on SDPA.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from oasr.layers import (
    Attention,
    ColumnParallelLinear,
    Embedding,
    GatedMLP,
    NeoxRotaryEmbedding,
    RMSNorm,
    RowParallelLinear,
    apply_rotary_pos_emb,
)

from ..decoders.base import BaseDecoder, DecoderState
from .config import SpeechLlmModelConfig


class _Qwen2Attention(nn.Module):
    def __init__(self, cfg: SpeechLlmModelConfig) -> None:
        super().__init__()
        d = cfg.text_hidden_size
        self.h = cfg.text_num_attention_heads
        self.h_kv = cfg.text_num_key_value_heads
        self.d_k = cfg.text_head_dim
        self.q_proj = ColumnParallelLinear(d, self.h * self.d_k, bias=True)
        self.k_proj = ColumnParallelLinear(d, self.h_kv * self.d_k, bias=True)
        self.v_proj = ColumnParallelLinear(d, self.h_kv * self.d_k, bias=True)
        self.o_proj = RowParallelLinear(self.h * self.d_k, d, bias=False)
        # Every mask this decoder builds is either causal or left-padded (valid
        # keys are ``[P - len, P)``, which is not a length), so the shared core
        # resolves to SDPA on every call.  It is here for the head bookkeeping,
        # the GQA handling and so a future paged/fused path is one edit.
        self.attn = Attention(self.h, self.d_k, num_kv_heads=self.h_kv)

    def qkv(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project + rotate → ``q (B, h, T, d)``, ``k``/``v`` ``(B, h_kv, T, d)``."""
        q = self.attn.split_heads(self.q_proj(x))
        k = self.attn.split_kv_heads(self.k_proj(x))
        v = self.attn.split_kv_heads(self.v_proj(x))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention over the (possibly grouped) KV → ``(B, T_q, D)``.

        Grouped KV is expanded **inside** the backend rather than by
        materialising ``k.repeat_interleave(h // h_kv, dim=1)``: that copy is the
        whole cache, per layer, per token, so it scales with ``B × context`` and
        would undo the in-place capacity-preallocated KV writes above it.  (The
        published Qwen2-Audio-7B has ``h_kv == h``, so this only bites a
        grouped-KV speech-LLM — but that is the common shape elsewhere.)
        """
        out = self.attn(q, k, v, attn_mask=attn_mask)
        return self.o_proj(self.attn.merge_heads(out))


class _Qwen2Layer(nn.Module):
    def __init__(self, cfg: SpeechLlmModelConfig) -> None:
        super().__init__()
        self.self_attn = _Qwen2Attention(cfg)
        self.mlp = GatedMLP(
            cfg.text_hidden_size, cfg.text_intermediate_size, activation="silu", bias=False
        )
        self.input_layernorm = RMSNorm(cfg.text_hidden_size, eps=cfg.text_rms_norm_eps, bias=False)
        self.post_attention_layernorm = RMSNorm(
            cfg.text_hidden_size, eps=cfg.text_rms_norm_eps, bias=False
        )


class Qwen2Lm(BaseDecoder):
    """Qwen2 causal LM (HF parameter names under ``layers.N.*`` / ``norm`` /
    ``embed_tokens`` / ``lm_head``) with the incremental decode surface."""

    decode_type = "llm"

    def __init__(self, cfg: SpeechLlmModelConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.text_hidden_size)
        self.layers = nn.ModuleList([_Qwen2Layer(cfg) for _ in range(cfg.text_num_hidden_layers)])
        self.norm = RMSNorm(cfg.text_hidden_size, eps=cfg.text_rms_norm_eps, bias=False)
        self.lm_head = ColumnParallelLinear(cfg.text_hidden_size, cfg.vocab_size, bias=False)
        if cfg.text_tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rotary = NeoxRotaryEmbedding(cfg.text_head_dim, cfg.text_rope_theta)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> DecoderState:
        del batch_size, device, dtype
        return None  # state is created by prefill()

    # ------------------------------------------------------------------
    # Incremental decode surface (driven by the ``llm`` strategy)
    # ------------------------------------------------------------------

    def _forward_layers(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        state: Dict[str, Any],
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Shared prefill/step trunk: run every layer, appending to the KV state."""
        t_prev = state["len"]
        t_new = x.size(1)
        for i, layer in enumerate(self.layers):
            residual = x
            h = layer.input_layernorm(x)
            q, k_new, v_new = layer.self_attn.qkv(h, cos, sin)
            k, v = self._append_kv(state, i, t_prev, k_new, v_new)
            x = residual + layer.self_attn.attend(q, k, v, attn_mask)
            residual = x
            h = layer.post_attention_layernorm(x)
            x = residual + layer.mlp(h)
        state["len"] = t_prev + t_new
        return x

    @staticmethod
    def _append_kv(
        state: Dict[str, Any],
        i: int,
        t_prev: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append layer ``i``'s new K/V and return the full cache views.

        Preallocated mode (``state["cap"]`` set and roomy): write the new
        tokens into their slots in place, return ``[:t]`` views — no copy of
        the existing cache.  Legacy mode (no capacity, or overflow): cat-grow.
        """
        t_new = k_new.size(2)
        cap = state["cap"]
        k_buf, v_buf = state["k"][i], state["v"][i]
        if cap is not None and k_buf is None:
            # First append (prefill): allocate the full-capacity buffers.
            B, h_kv, _, d = k_new.shape
            k_buf = k_new.new_empty(B, h_kv, cap, d)
            v_buf = v_new.new_empty(B, h_kv, cap, d)
            state["k"][i], state["v"][i] = k_buf, v_buf
        if cap is not None and t_prev + t_new <= cap:
            k_buf[:, :, t_prev : t_prev + t_new] = k_new
            v_buf[:, :, t_prev : t_prev + t_new] = v_new
            return k_buf[:, :, : t_prev + t_new], v_buf[:, :, : t_prev + t_new]
        # Legacy growth (no capacity hint, or capacity overflow).  An
        # overflowing preallocated buffer degrades to cat of the valid slice.
        if k_buf is not None:
            k = torch.cat([k_buf[:, :, :t_prev], k_new], dim=2) if t_prev else k_new
            v = torch.cat([v_buf[:, :, :t_prev], v_new], dim=2) if t_prev else v_new
        else:
            k, v = k_new, v_new
        state["k"][i], state["v"][i] = k, v
        state["cap"] = None  # buffers are now exact-length; stay on cat-growth
        return k, v

    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        valid: torch.Tensor,
        capacity: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Start generation over a **left-padded** embedded prompt.

        ``inputs_embeds (B, P, D)`` (text embeddings with audio embeddings
        spliced in), ``valid (B, P)`` bool (False = left pad) →
        ``(logits (B, V) at the last position, state)``.  With left padding
        the last position is the newest real token for every row.

        ``capacity`` (optional): total KV length this generation may reach
        (prompt + generation cap).  When given, the per-layer K/V buffers are
        preallocated once and each :meth:`step` writes its token slot in
        place instead of re-copying the cache via ``torch.cat``.
        """
        B, P, _ = inputs_embeds.shape
        device = inputs_embeds.device
        # HF masked-generate positions: cumsum - 1, pads clamped to 0.
        position_ids = (valid.long().cumsum(dim=1) - 1).clamp(min=0)
        cos, sin = self.rotary(position_ids)  # (B, P, d)

        # Causal + key-padding mask; the diagonal stays open so fully-padded
        # query rows attend themselves instead of softmaxing over -inf only.
        causal = torch.ones(P, P, dtype=torch.bool, device=device).tril()
        allowed = causal.unsqueeze(0) & valid.unsqueeze(1)
        eye = torch.eye(P, dtype=torch.bool, device=device)
        allowed = allowed | eye.unsqueeze(0)
        attn_mask = torch.zeros(B, 1, P, P, dtype=inputs_embeds.dtype, device=device)
        attn_mask.masked_fill_(~allowed.unsqueeze(1), float("-inf"))

        n = len(self.layers)
        state: Dict[str, Any] = {
            "k": [None] * n,
            "v": [None] * n,
            "key_valid": valid,
            "pos": valid.long().sum(dim=1),  # per-row next rotary position
            "len": 0,  # tokens cached so far (uniform across layers)
            "cap": None if capacity is None else max(int(capacity), P),
        }
        x = self._forward_layers(inputs_embeds, cos, sin, state, attn_mask)
        logits = self.lm_head(self.norm(x[:, -1:]))
        return logits[:, -1], state

    def step(
        self, tokens: torch.Tensor, state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One generation step: ``tokens (B,)`` → ``(logits (B, V), state)``."""
        x = self.embed_tokens(tokens.unsqueeze(1))  # (B, 1, D)
        cos, sin = self.rotary(state["pos"].unsqueeze(1))  # (B, 1, d)

        B = x.size(0)
        key_valid = torch.cat(
            [state["key_valid"], torch.ones(B, 1, dtype=torch.bool, device=x.device)], dim=1
        )
        state["key_valid"] = key_valid
        attn_mask = torch.zeros(B, 1, 1, key_valid.size(1), dtype=x.dtype, device=x.device)
        attn_mask.masked_fill_(~key_valid.view(B, 1, 1, -1), float("-inf"))

        x = self._forward_layers(x, cos, sin, state, attn_mask)
        logits = self.lm_head(self.norm(x))
        state["pos"] = state["pos"] + 1
        return logits[:, -1], state

    @staticmethod
    def select(state: Dict[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
        """Drop finished rows: index-select every cached tensor along batch."""
        out: Dict[str, Any] = {
            "key_valid": state["key_valid"].index_select(0, keep),
            "pos": state["pos"].index_select(0, keep),
            "len": state["len"],
            "cap": state["cap"],
        }
        for key in ("k", "v"):
            out[key] = [t.index_select(0, keep) for t in state[key]]
        return out
