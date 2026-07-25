# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention encoder-decoder (AED) decode strategy — incremental greedy.

The first ``incremental = True`` strategy over the K2 protocol: label-
synchronous generation that runs **at most** ``EngineConfig.decode_steps_per_tick``
batched decoder steps per engine tick, so one tick never stalls the serving
dispatcher regardless of transcript length.

Drives a decoder exposing the batched incremental surface
(:meth:`~oasr.models.whisper.WhisperDecoder.prefill` / ``step`` / ``select``)
plus a model config carrying the generation control ids (SOT prompt,
EOS, suppress lists) — Whisper today; any AED checkpoint with the same surface
plugs in unchanged.

Batching model: each encoded micro-batch becomes one *group* (uniform prompt
length and — for Whisper's fixed 30 s window — uniform encoder length), and
each ``advance`` round-robins one batched decoder step across groups until the
tick budget is spent.  Rows leave their group as they emit EOS or hit the
length cap; finished requests produce final outputs (greedy emits finals only
— token-streaming partials land with the LLM phase).  Per-request
:class:`~oasr.engine.request.DecodingOptions` carry a generation cap and
sampling knobs; ``prompt`` is ignored (Whisper's SOT sequence is
checkpoint-fixed).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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


@register_decode_strategy("aed")
class AedDecodeStrategy(DecodeStrategy):
    """Incremental greedy AED decoding (Whisper-style)."""

    decode_type = "aed"
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
        if decoder is None or not hasattr(decoder, "prefill"):
            raise ValueError(
                "decode_method='aed' needs a model whose decoder exposes the "
                "batched incremental surface (prefill/step/select) — e.g. the "
                "'whisper' architecture."
            )
        self._config = config
        self._detok = detok
        self._decoder = decoder
        self._prompt = list(mcfg.sot_sequence())
        self._eos = int(mcfg.eos_token_id)
        self._suppress = sorted(set(int(t) for t in mcfg.suppress_tokens))
        self._begin_suppress = sorted(set(int(t) for t in mcfg.begin_suppress_tokens))
        self._cap = int(mcfg.max_target_positions) - len(self._prompt) - 1
        self._max_new_tokens = max(1, min(int(config.max_new_tokens), self._cap))
        self._groups: List[_Group] = []

    # ------------------------------------------------------------------
    # Incremental protocol
    # ------------------------------------------------------------------

    def begin_offline(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> None:
        del enc_lengths  # Whisper: fixed-window encoder output, no padding mask
        device = enc_out.device
        prompt = torch.tensor(self._prompt, dtype=torch.long, device=device)
        prompt_ids = prompt.unsqueeze(0).expand(enc_out.size(0), -1).contiguous()
        # Per-row generation cap: DecodingOptions.max_new_tokens (else the
        # engine default) clamped by the decoder's position capacity
        # (``self._max_new_tokens`` already folds the config cap + capacity).
        opts_rows: List[Optional[DecodingOptions]] = [
            getattr(r, "decoding", None) for r in requests
        ]
        max_new_rows: List[int] = []
        for opts in opts_rows:
            if opts is not None and opts.max_new_tokens is not None:
                max_new_rows.append(max(1, min(int(opts.max_new_tokens), self._cap)))
            else:
                max_new_rows.append(self._max_new_tokens)
        with torch.no_grad():
            logits, state = self._decoder.prefill(enc_out, prompt_ids)
        self._groups.append(_Group(requests, state, logits, max_new_rows, opts_rows))

    def has_pending(self) -> bool:
        return bool(self._groups)

    def advance(self, budget: "StepBudget") -> List[RequestOutput]:
        outputs: List[RequestOutput] = []
        while self._groups and budget.take():
            # Round-robin: pop the front group, advance it one batched step,
            # re-queue it at the back if it still has active rows.
            group = self._groups.pop(0)
            outputs.extend(self._advance_group(group))
            if group.requests:
                self._groups.append(group)
        return outputs

    def _advance_group(self, group: _Group) -> List[RequestOutput]:
        logits = group.last_logits.float()
        first_step = not group.tokens[0] if group.tokens else True
        if self._suppress:
            logits[:, self._suppress] = float("-inf")
        if first_step and self._begin_suppress:
            logits[:, self._begin_suppress] = float("-inf")
        next_tokens = select_next_tokens(logits, group.opts)  # (B_active,)

        toks = next_tokens.cpu().tolist()
        finished_rows: List[int] = []
        reasons: Dict[int, str] = {}
        for row, tok in enumerate(toks):
            if tok == self._eos:
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
            "aed is an incremental strategy; the executor drives it via "
            "begin_offline/advance, not one-shot decode_offline"
        )

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        raise NotImplementedError("aed decoding is offline-only (not genuinely streamable)")

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        raise NotImplementedError("aed decoding is offline-only (not genuinely streamable)")

    def finalize(self, request: Request) -> RequestOutput:
        raise NotImplementedError("aed decoding is offline-only (not genuinely streamable)")
