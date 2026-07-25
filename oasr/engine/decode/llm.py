# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""LLM-based ASR decode strategy — incremental greedy with streaming partials.

Drives a speech-LLM (audio tower + projector + causal LM — Qwen2-Audio today)
over the incremental protocol: the encoder phase produced per-utterance audio
embeddings in LLM space (``consumes="hidden"``), :meth:`begin_offline` splices
them into the checkpoint's ChatML prompt around the audio slot, and
:meth:`advance` runs at most ``EngineConfig.decode_steps_per_tick`` batched
decoder steps per engine tick.  Unlike the ``aed`` strategy, prompts are
**variable-length** (the audio embedding count follows the utterance length),
so each micro-batch prefills left-padded with a validity mask — the LM surface
(:meth:`~oasr.models.speech_llm.Qwen2Lm.prefill`) implements HF's
masked-generate convention exactly.

Token-streaming partials: every tick that advances a request emits a
``finished=False`` :class:`RequestOutput` with the transcript so far, so the
serving layer streams tokens over the existing ``Event::Partial`` wire with no
protocol change.  Per-request :class:`~oasr.engine.request.DecodingOptions`
carry a user-prompt override, a generation cap, and sampling knobs
(``temperature`` / ``top_k`` / ``top_p``); the default remains greedy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import torch

from ..generation import select_next_tokens
from ..request import DecodingOptions, Request, RequestOutput
from .base import DecodeStrategy, register_decode_strategy

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from ..generation import StepBudget
    from .detokenize import Detokenizer

logger = logging.getLogger(__name__)

# Cap on the per-prompt suffix-ids memo — custom prompts are typically a
# handful of deployment-fixed strings; a runaway per-request-unique prompt
# stream must not grow the cache unboundedly.
_PROMPT_CACHE_MAX = 64


class _Group:
    """One encoded micro-batch generating together (continuous batching unit)."""

    def __init__(
        self,
        requests: List[Request],
        state: Dict[str, Any],
        last_logits: torch.Tensor,
        max_new_rows: List[int],
        opts_rows: List[Optional[DecodingOptions]],
    ) -> None:
        self.requests = list(requests)
        self.state = state
        self.last_logits = last_logits  # (B_active, V) — pending selection input
        self.tokens: List[List[int]] = [[] for _ in requests]
        self.max_new = list(max_new_rows)  # per-row generation cap
        self.opts = list(opts_rows)  # per-row DecodingOptions (None = defaults)


@register_decode_strategy("llm")
class LlmDecodeStrategy(DecodeStrategy):
    """Incremental greedy LLM decoding (Qwen2-Audio-style speech LLM)."""

    decode_type = "llm"
    consumes = "hidden"
    incremental = True

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
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
        self._config = config
        self._detok = detok
        self._decoder = decoder
        self._mcfg = mcfg
        self._tokenizer = tokenizer

        user_prompt = config.llm_prompt or mcfg.default_user_prompt
        suffix_text = mcfg.prompt_suffix.format(prompt=user_prompt)
        self._prefix_ids: List[int] = list(tokenizer.encode(mcfg.prompt_prefix))
        self._default_suffix_ids: List[int] = list(tokenizer.encode(suffix_text))
        # Per-prompt suffix-ids memo for DecodingOptions.prompt overrides.
        self._suffix_cache: Dict[str, List[int]] = {}
        self._eos: Set[int] = set(int(t) for t in mcfg.eos_token_ids)
        self._groups: List[_Group] = []

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
    # Incremental protocol
    # ------------------------------------------------------------------

    def begin_offline(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> None:
        """Splice audio embeddings into the prompt, left-pad, prefill."""
        device = enc_out.device
        dtype = enc_out.dtype
        embed = self._decoder.embed_tokens
        prefix = embed(torch.tensor(self._prefix_ids, dtype=torch.long, device=device)).to(dtype)
        # Per-row suffix: DecodingOptions.prompt swaps the user text.  Embed
        # each distinct suffix once per batch.
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

        # Per-row generation cap: the request's DecodingOptions.max_new_tokens
        # (else the engine default), clamped by the LM's position capacity for
        # that row's actual prompt length.
        max_pos = int(self._mcfg.text_max_position_embeddings)
        opts_rows: List[Optional[DecodingOptions]] = [
            getattr(r, "decoding", None) for r in requests
        ]
        max_new_rows: List[int] = []
        for opts, total in zip(opts_rows, totals):
            requested = (
                opts.max_new_tokens
                if opts is not None and opts.max_new_tokens is not None
                else int(self._config.max_new_tokens)
            )
            max_new_rows.append(max(1, min(int(requested), max_pos - total - 1)))
        with torch.no_grad():
            # capacity → the LM preallocates its KV buffers and steps write
            # in place (no per-step cache re-copy).
            logits, state = self._decoder.prefill(inputs, valid, capacity=P + max(max_new_rows) + 1)
        self._groups.append(_Group(requests, state, logits, max_new_rows, opts_rows))

    def has_pending(self) -> bool:
        return bool(self._groups)

    def advance(self, budget: "StepBudget") -> List[RequestOutput]:
        outputs: List[RequestOutput] = []
        advanced: List[_Group] = []
        while self._groups and budget.take():
            # Round-robin: pop the front group, advance it one batched step,
            # re-queue it at the back if it still has active rows.
            group = self._groups.pop(0)
            outputs.extend(self._advance_group(group))
            if group.requests:
                self._groups.append(group)
                if group not in advanced:
                    advanced.append(group)
        # Token-streaming partials: one per still-active request that moved
        # this tick (the serving layer forwards these as Event::Partial).
        for group in advanced:
            for row, req in enumerate(group.requests):
                tokens = group.tokens[row]
                if tokens:
                    outputs.append(
                        RequestOutput(
                            request_id=req.request_id,
                            text=self._detok.detokenize(tokens),
                            tokens=[list(tokens)],
                            finished=False,
                        )
                    )
        return outputs

    def _advance_group(self, group: _Group) -> List[RequestOutput]:
        next_tokens = select_next_tokens(group.last_logits.float(), group.opts)  # (B_active,)

        toks = next_tokens.cpu().tolist()
        finished_rows: List[int] = []
        reasons: Dict[int, str] = {}
        for row, tok in enumerate(toks):
            if tok in self._eos:
                finished_rows.append(row)
                reasons[row] = "stop"
            else:
                group.tokens[row].append(int(tok))
                if len(group.tokens[row]) >= group.max_new[row]:
                    finished_rows.append(row)
                    reasons[row] = "length"

        outputs = [self._finalize_row(group, row, reasons[row]) for row in finished_rows]
        if finished_rows:
            keep = [r for r in range(len(group.requests)) if r not in finished_rows]
            group.requests = [group.requests[r] for r in keep]
            group.tokens = [group.tokens[r] for r in keep]
            group.max_new = [group.max_new[r] for r in keep]
            group.opts = [group.opts[r] for r in keep]
            if keep:
                keep_idx = torch.tensor(keep, dtype=torch.long, device=next_tokens.device)
                group.state = self._decoder.select(group.state, keep_idx)
                next_tokens = next_tokens.index_select(0, keep_idx)

        if group.requests:
            with torch.no_grad():
                group.last_logits, group.state = self._decoder.step(next_tokens, group.state)
        return outputs

    def _finalize_row(self, group: _Group, row: int, reason: str) -> RequestOutput:
        tokens = group.tokens[row]
        return RequestOutput(
            request_id=group.requests[row].request_id,
            text=self._detok.detokenize(tokens),
            tokens=[tokens],
            finished=True,
            finish_reason=reason,
        )

    # ------------------------------------------------------------------
    # Session cleanup (abort path)
    # ------------------------------------------------------------------

    def free_session(self, request: Request) -> None:
        for group in self._groups:
            if request in group.requests:
                row = group.requests.index(request)
                keep = [r for r in range(len(group.requests)) if r != row]
                group.requests.pop(row)
                group.tokens.pop(row)
                group.max_new.pop(row)
                group.opts.pop(row)
                if keep:
                    keep_idx = torch.tensor(keep, dtype=torch.long, device=group.last_logits.device)
                    group.state = self._decoder.select(group.state, keep_idx)
                    group.last_logits = group.last_logits.index_select(0, keep_idx)
                break
        self._groups = [g for g in self._groups if g.requests]

    # ------------------------------------------------------------------
    # One-shot / streaming surfaces (not applicable)
    # ------------------------------------------------------------------

    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        raise NotImplementedError(
            "llm is an incremental strategy; the executor drives it via "
            "begin_offline/advance, not one-shot decode_offline"
        )

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        raise NotImplementedError("llm decoding is offline-only (not genuinely streamable)")

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        raise NotImplementedError("llm decoding is offline-only (not genuinely streamable)")

    def finalize(self, request: Request) -> RequestOutput:
        raise NotImplementedError("llm decoding is offline-only (not genuinely streamable)")
