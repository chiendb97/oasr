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

The KV state is a :class:`~oasr.cache.decoder_state.DecoderKv` — one
capacity buffer per layer plus **per-row** write offsets — shared with the AED
decoder; rows are dropped with :meth:`select` as requests finish (continuous
batching) and two prefilled states are joined with :meth:`merge`, which is what
lets a trickle of arrivals still generate in one forward.  When the caller
passes ``capacity`` to :meth:`prefill` (the ``llm`` strategy does — prompt
length + the batch's generation cap), the per-layer K/V buffers are
**preallocated** to that capacity and each step writes its one token slot in
place — removing the per-step ``torch.cat`` that re-copies the whole cache
(measured ~10% of a 7B decode step at B=4, growing with B).  Without
``capacity`` the legacy cat-growth path is used (direct ``prefill``/``step``
callers, tests); it cannot express per-row offsets and therefore cannot merge.

Attention splits by call: :meth:`prefill` hands the shared core its causal +
left-padded window as ``kv_lens``/``kv_starts``/``is_causal`` and lands on the
fused kernel — 1.8-3.3x on the attention op over the materialized-mask SDPA it
used to build, 1.03-1.05x over the whole prefill, which is GEMM-dominated — while
:meth:`step` stays on SDPA because its K/V are capacity-buffer slices the fused
kernel would have to copy whole, once per layer per step.  Paged-KV storage
(``DecoderKVCacheManager``) is still the open optimization; it is what would let
``step`` reach the kernel too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

import torch
from torch import nn

from oasr.cache import DecoderKv, build_kv
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from oasr.cache.decoder_kv import DecoderKVCacheManager


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
        # Prefill's causal + left-padded window is now a form the shared core
        # can fuse (two length vectors, no materialized mask); step's is not,
        # for a cache-stride reason documented at its call site.
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
        mask_kwargs: Dict[str, Any],
        kv_extent: Optional[int],
    ) -> torch.Tensor:
        """Attention over the (possibly grouped) KV → ``(B, T_q, D)``.

        Grouped KV is expanded **inside** the backend rather than by
        materialising ``k.repeat_interleave(h // h_kv, dim=1)``: that copy is the
        whole cache, per layer, per token, so it scales with ``B × context`` and
        would undo the in-place capacity-preallocated KV writes above it.  (The
        published Qwen2-Audio-7B has ``h_kv == h``, so this only bites a
        grouped-KV speech-LLM — but that is the common shape elsewhere.)
        """
        out = self.attn(q, k, v, kv_extent=kv_extent, **mask_kwargs)
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

    def forward(
        self,
        h: torch.Tensor,
        residual: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: DecoderKv,
        layer_idx: int,
        mask_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one LM layer and return its MLP output separately."""
        q, k_new, v_new = self.self_attn.qkv(h, cos, sin)
        k, v, kv_extent = kv.append(layer_idx, k_new, v_new)
        attn = self.self_attn.attend(q, k, v, mask_kwargs, kv_extent)
        h, residual = self.post_attention_layernorm.forward_add_residual(attn, residual)
        return self.mlp(h), residual


class Qwen2Lm(BaseDecoder):
    """Qwen2 causal LM (HF parameter names under ``layers.N.*`` / ``norm`` /
    ``embed_tokens`` / ``lm_head``) with the incremental decode surface."""

    decode_type = "llm"
    supports_paged_kv = True
    #: Nothing outside the paged pool survives between steps here — a
    #: decoder-only LM has no cross-attention cache — so a step is capturable
    #: given paged KV.
    supports_step_graphs = True

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
        kv: DecoderKv,
        mask_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Shared prefill/step trunk: run every layer, appending to the KV state."""
        t_new = x.size(1)
        layers = [cast(_Qwen2Layer, layer) for layer in self.layers]
        if not layers:
            kv.commit(t_new)
            return x

        residual = x
        h = layers[0].input_layernorm(x)
        for i, layer in enumerate(layers):
            mlp, residual = layer(h, residual, cos, sin, kv, i, mask_kwargs)
            if i + 1 < len(layers):
                h, residual = layers[i + 1].input_layernorm.forward_add_residual(mlp, residual)
            else:
                # Prefill normalizes only the final token below, so doing the
                # last RMSNorm here would add work over the whole prompt.
                x = residual + mlp
        kv.commit(t_new)
        return x

    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        valid: torch.Tensor,
        capacity: Optional[int] = None,
        kv_manager: Optional["DecoderKVCacheManager"] = None,
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

        ``kv_manager`` (optional, requires ``capacity``): page the KV out of a
        shared pool instead of preallocating a per-group buffer.
        """
        B, P, _ = inputs_embeds.shape
        n = len(self.layers)
        kv = build_kv(
            n,
            B,
            inputs_embeds.device,
            prefill_len=P,
            cap=None if capacity is None else max(int(capacity), P),
            manager=kv_manager,
            # Left padding *is* the per-row start offset: HF's masked-generate
            # convention puts every row's real prompt flush against the right of
            # the ``(B, P)`` grid, so the valid window is ``[P - len, len)``.
            starts=(P - valid.sum(dim=1)).to(torch.int32),
        )
        # HF masked-generate positions: cumsum - 1, pads clamped to 0.  For the
        # real (right-flush) region that is exactly ``lens - starts`` counted from
        # 0, which is what ``DecoderKv.position_ids`` derives; the pads' ids are
        # never read because their query rows are masked out.
        position_ids = (valid.long().cumsum(dim=1) - 1).clamp(min=0)
        cos, sin = self.rotary(position_ids)  # (B, P, d)

        # Causal + key padding.  Left padding is contiguous by construction (the
        # strategy left-pads the batch, HF's masked-generate convention), so the
        # whole mask is the window ``[P - len, P)`` intersected with the causal
        # triangle — two length vectors rather than a ``(B, 1, P, P)`` tensor.
        # That form is what the fused kernel takes, and materializing it instead
        # costs SDPA its flash path: 1.8-3.3x on the attention op at the 7B
        # prefill shapes, 1.03-1.05x over the whole prefill.
        # ``Attention`` decides which backend actually runs it, and rebuilds the
        # explicit mask itself on the SDPA side (diagonal kept open, so a fully
        # padded query row cannot come back NaN and poison real rows downstream).
        mask_kwargs: Dict[str, Any] = dict(kv.mask_kwargs(P))
        mask_kwargs["is_causal"] = True

        state: Dict[str, Any] = {"kv": kv}
        x = self._forward_layers(inputs_embeds, cos, sin, kv, mask_kwargs)
        logits = self.lm_head(self.norm(x[:, -1:]))
        return logits[:, -1], state

    def step(
        self, tokens: torch.Tensor, state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One generation step: ``tokens (B,)`` → ``(logits (B, V), state)``."""
        kv: DecoderKv = state["kv"]
        x = self.embed_tokens(tokens.unsqueeze(1))  # (B, 1, D)
        cos, sin = self.rotary(kv.position_ids(1))  # (B, 1, d)

        # Key padding only -- no causal component, since the single query row
        # attends the whole cache.  Left padding is contiguous per row, so the
        # window form reaches the fused kernel; ``DecoderKv.append`` hands over
        # the capacity buffer plus its length rather than a stride-gapped slice,
        # which is what used to make the kernel copy the whole cache per layer per
        # step and kept this call on SDPA.  With that gone the kernel is
        # 1.45-1.88x faster here even at ``T_q == 1``.
        x = self._forward_layers(x, cos, sin, kv, dict(kv.mask_kwargs(1)))
        logits = self.lm_head(self.norm(x))
        return logits[:, -1], state

    @staticmethod
    def select(state: Dict[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
        """Drop finished rows: index-select every cached tensor along batch."""
        return {"kv": state["kv"].select(keep)}

    @staticmethod
    def can_merge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Whether two prefilled states can generate in one forward."""
        return bool(a["kv"].can_merge(b["kv"]))

    @staticmethod
    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Concatenate ``b``'s rows after ``a``'s into one generating state."""
        return {"kv": a["kv"].merge(b["kv"])}
