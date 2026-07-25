# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""LLM-based ASR decode strategy — incremental greedy with streaming partials.

Everything about driving a label-synchronous decoder within the engine's tick
budget lives in :class:`~oasr.engine.decode.incremental.IncrementalArStrategy`;
this module is only what makes a speech-LLM different from an AED decoder:

* the prompt is the checkpoint's ChatML template with the projected audio
  embeddings spliced in, so it is **variable-length per row** — prefill
  left-pads with a validity mask, matching HF's masked-generate convention;
* the per-row generation cap is bounded by that row's own prompt length;
* EOS is a *set* of ids;
* partials are emitted, so the serving layer streams tokens over the existing
  ``Event::Partial`` wire with no protocol change.

Per-request :class:`~oasr.engine.request.DecodingOptions` carry a user-prompt
override, a generation cap, and sampling knobs; the default remains greedy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Dict, List, Set, Tuple

import torch

from ..request import Request
from .base import register_decode_strategy
from .incremental import IncrementalArStrategy, Prefill

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer

logger = logging.getLogger(__name__)

# Cap on the per-prompt suffix-ids memo — custom prompts are typically a
# handful of deployment-fixed strings; a runaway per-request-unique prompt
# stream must not grow the cache unboundedly.
_PROMPT_CACHE_MAX = 64


@register_decode_strategy("llm")
class LlmDecodeStrategy(IncrementalArStrategy):
    """Incremental greedy LLM decoding (Qwen2-Audio-style speech LLM)."""

    decode_type: ClassVar[str] = "llm"
    emit_partials: ClassVar[bool] = True

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        super().__init__(config, detok, model)
        decoder = getattr(model, "decoder", None) if model is not None else None
        mcfg = getattr(model, "config", None)
        if (
            decoder is None
            or not hasattr(decoder, "prefill")
            or not hasattr(decoder, "embed_tokens")
            or not hasattr(mcfg, "prompt_prefix")
        ):
            raise ValueError(
                "decode_method='llm' needs a speech-LLM model whose decoder "
                "exposes the batched incremental surface (prefill/step/select) "
                "plus prompt-template config — e.g. the 'speech_llm' architecture."
            )
        tokenizer = detok.tokenizer
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise ValueError(
                "decode_method='llm' needs the checkpoint's tokenizer to encode "
                "the prompt template; the loaded checkpoint carries no usable "
                "TokenizerSpec (install `tokenizers` / re-convert the checkpoint)."
            )
        self._mcfg = mcfg
        self._tokenizer = tokenizer

        user_prompt = config.llm_prompt or mcfg.default_user_prompt
        suffix_text = mcfg.prompt_suffix.format(prompt=user_prompt)
        self._prefix_ids: List[int] = list(tokenizer.encode(mcfg.prompt_prefix))
        self._default_suffix_ids: List[int] = list(tokenizer.encode(suffix_text))
        # Per-prompt suffix-ids memo for DecodingOptions.prompt overrides.
        self._suffix_cache: Dict[str, List[int]] = {}
        self._eos: Set[int] = set(int(t) for t in mcfg.eos_token_ids)

    def _suffix_ids_for(self, request: Request) -> List[int]:
        """Suffix token ids for one request's chat template.

        ``DecodingOptions.prompt`` swaps the user text inside the checkpoint's
        template; the surrounding ChatML structure never changes.
        """
        opts = getattr(request, "decoding", None)
        prompt = opts.prompt if opts is not None else None
        if not prompt:
            return self._default_suffix_ids
        ids = self._suffix_cache.get(prompt)
        if ids is None:
            if len(self._suffix_cache) >= _PROMPT_CACHE_MAX:
                self._suffix_cache.clear()
            ids = list(self._tokenizer.encode(self._mcfg.prompt_suffix.format(prompt=prompt)))
            self._suffix_cache[prompt] = ids
        return ids

    # ------------------------------------------------------------------
    # IncrementalArStrategy hooks
    # ------------------------------------------------------------------

    def _prefill(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> Prefill:
        """Splice audio embeddings into the prompt, left-pad, prefill."""
        inputs, valid, totals = self._build_prompt(requests, enc_out, enc_lengths)
        # Per-row generation cap: the request's DecodingOptions.max_new_tokens
        # (else the engine default), clamped by the LM's position capacity for
        # that row's actual prompt length.
        max_pos = int(self._mcfg.text_max_position_embeddings)
        max_new = [self._row_cap(req, max_pos - total - 1) for req, total in zip(requests, totals)]
        with torch.no_grad():
            # capacity → the LM preallocates its KV buffers and steps write
            # in place (no per-step cache re-copy).
            logits, state = self._decoder().prefill(
                inputs, valid, capacity=inputs.size(1) + max(max_new) + 1
            )
        return Prefill(state=state, logits=logits, max_new=max_new)

    def _build_prompt(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """``(inputs (B, P, D), valid (B, P) bool, per-row prompt lengths)``.

        Rows are **left**-padded: the audio-embedding count follows the utterance
        length, so prompts differ per row, and the LM's positions are derived from
        ``cumsum(valid) - 1`` (HF's masked-generate convention).
        """
        device, dtype = enc_out.device, enc_out.dtype
        embed = self._decoder().embed_tokens
        prefix = embed(torch.tensor(self._prefix_ids, dtype=torch.long, device=device)).to(dtype)
        # Per-row suffix: DecodingOptions.prompt swaps the user text.  Embed each
        # distinct suffix once per batch.
        suffix_ids_rows = [self._suffix_ids_for(r) for r in requests]
        suffix_embeds: Dict[tuple, torch.Tensor] = {}
        for ids in suffix_ids_rows:
            key = tuple(ids)
            if key not in suffix_embeds:
                suffix_embeds[key] = embed(torch.tensor(ids, dtype=torch.long, device=device)).to(
                    dtype
                )

        B = enc_out.size(0)
        audio_lens = enc_lengths.to(torch.long).tolist()
        totals = [
            len(self._prefix_ids) + al + len(ids) for al, ids in zip(audio_lens, suffix_ids_rows)
        ]
        P = max(totals)
        inputs = torch.zeros(B, P, enc_out.size(2), dtype=dtype, device=device)
        valid = torch.zeros(B, P, dtype=torch.bool, device=device)
        for i, (al, total) in enumerate(zip(audio_lens, totals)):
            row = torch.cat(
                [prefix, enc_out[i, :al], suffix_embeds[tuple(suffix_ids_rows[i])]], dim=0
            )
            inputs[i, P - total :] = row
            valid[i, P - total :] = True
        return inputs, valid, totals

    def _is_eos(self, token: int) -> bool:
        return token in self._eos
