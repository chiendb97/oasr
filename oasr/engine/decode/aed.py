# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention encoder-decoder (AED) decode strategy — incremental greedy.

Everything about driving a label-synchronous decoder within the engine's tick
budget lives in :class:`~oasr.engine.decode.incremental.IncrementalArStrategy`;
this module is only what makes an AED decoder different from the speech-LLM one:

* the prompt is the checkpoint's fixed SOT sequence, identical for every row, so
  the prefill is a plain expand (no per-row padding);
* Whisper's ``suppress_tokens`` / ``begin_suppress_tokens`` are applied to the
  logits before selection;
* EOS is a single token id;
* greedy emits finals only — token-streaming partials land with the LLM family.

Drives a decoder exposing the batched incremental surface
(:meth:`~oasr.models.whisper.WhisperDecoder.prefill` / ``step`` / ``select``)
plus a model config carrying the generation-control ids — Whisper today; any AED
checkpoint with the same surface plugs in unchanged.  Per-request
:class:`~oasr.engine.request.DecodingOptions` carry a generation cap and sampling
knobs; ``prompt`` is ignored (the free-text prompt has no slot in the SOT
sequence), while ``task`` and ``language`` *do* apply — they substitute the
corresponding SOT slot, which is what makes ``POST /v1/audio/translations`` and
per-request language selection possible on a checkpoint whose
``forced_decoder_ids`` froze both at conversion time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import torch

from ..request import DecodingOptions, Request
from .base import register_decode_strategy
from .incremental import ArGroup, IncrementalArStrategy, Prefill

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer

logger = logging.getLogger(__name__)


@register_decode_strategy("aed")
class AedDecodeStrategy(IncrementalArStrategy):
    """Incremental greedy AED decoding (Whisper-style)."""

    decode_type: ClassVar[str] = "aed"
    emit_partials: ClassVar[bool] = False
    #: Both slots exist in the SOT sequence, so both can be set per request.
    selective_options: ClassVar[Tuple[str, ...]] = ("task", "language")

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        super().__init__(config, detok, model)
        # Surface validation lives in ``build_decode_strategy`` via
        # ``oasr.models.interfaces.CAPABILITIES["aed"]`` — one table, one message.
        # Typed ``Any`` deliberately: the members below are a *duck-typed* AED
        # config surface, declared in that table and checked once at
        # construction, so naming a concrete class here would tie the family to
        # Whisper for nothing (and ``nn.Module.__getattr__`` types every
        # attribute as ``Tensor | Module`` anyway).
        mcfg: Any = model.config
        self._mcfg = mcfg
        self._prompt = list(mcfg.sot_sequence())
        # ``(task, language) -> prompt ids``.  Substituting a slot never changes
        # the prompt's length, so a batch mixing tasks still prefills as one
        # rectangular tensor — the whole reason this is a substitution rather
        # than a re-templating.
        self._prompt_cache: Dict[Tuple[Optional[str], Optional[str]], List[int]] = {
            (None, None): self._prompt
        }
        self._eos = int(mcfg.eos_token_id)
        self._suppress = sorted({int(t) for t in mcfg.suppress_tokens})
        self._begin_suppress = sorted({int(t) for t in mcfg.begin_suppress_tokens})
        # Remaining decoder positions once the prompt is in place.
        self._cap = int(mcfg.max_target_positions) - len(self._prompt) - 1
        # Suppress-id index tensors, materialised per device on first use: an
        # ``index_fill`` with a cached tensor beats re-building an index from a
        # Python list on every step.
        self._suppress_idx: Dict[torch.device, torch.Tensor] = {}
        self._begin_suppress_idx: Dict[torch.device, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # IncrementalArStrategy hooks
    # ------------------------------------------------------------------

    def _prefill(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> Prefill:
        del enc_lengths  # Whisper: fixed-window encoder output, no padding mask
        device = enc_out.device
        rows = [self._prompt_for(r) for r in requests]
        if all(row is self._prompt for row in rows):
            # The common case — no request overrode task or language — stays a
            # single expand rather than a per-row stack.
            prompt = torch.tensor(self._prompt, dtype=torch.long, device=device)
            prompt_ids = prompt.unsqueeze(0).expand(enc_out.size(0), -1).contiguous()
        else:
            prompt_ids = torch.tensor(rows, dtype=torch.long, device=device)
        max_new = [self._row_cap(r, self._cap) for r in requests]
        with torch.no_grad():
            logits, state = self._decoder().prefill(enc_out, prompt_ids)
        return Prefill(state=state, logits=logits, max_new=max_new)

    def _prompt_for(self, request: Request) -> List[int]:
        """This request's SOT sequence, with its task / language slots applied.

        Validated at admission (:meth:`validate_options`), so a lookup failure
        here would be an engine bug rather than a client error.
        """
        opts = request.decoding
        key = (
            getattr(opts, "task", None) if opts else None,
            getattr(opts, "language", None) if opts else None,
        )
        if key == (None, None):
            return self._prompt
        cached = self._prompt_cache.get(key)
        if cached is None:
            cached = list(self._mcfg.sot_sequence(task=key[0], language=key[1]))
            if len(cached) != len(self._prompt):
                # Substitution must be length-preserving; anything else would
                # need per-row prefill padding, which this path does not do.
                raise ValueError(
                    f"decoder prompt for task={key[0]!r} language={key[1]!r} has "
                    f"{len(cached)} tokens, expected {len(self._prompt)}"
                )
            self._prompt_cache[key] = cached
        return cached

    def validate_options(self, options: Optional[DecodingOptions]) -> None:
        """Resolve ``task`` / ``language`` against *this checkpoint* now.

        The base class only asks whether the family understands the option at
        all.  Whether this Whisper snapshot knows ``<|yue|>`` is a checkpoint
        question, and answering it at admission is what turns "unknown
        language" into a 400 for that request instead of a raise from inside a
        prefill shared with unrelated requests.
        """
        super().validate_options(options)
        if options is None or (options.task is None and options.language is None):
            return
        self._mcfg.sot_sequence(task=options.task, language=options.language)

    def _is_eos(self, token: int) -> bool:
        return token == self._eos

    def _process_logits(self, logits: torch.Tensor, group: ArGroup) -> torch.Tensor:
        """Apply Whisper's suppress lists.

        ``begin_suppress_tokens`` applies only to the first generated position.
        The group answers that via ``first_generation_step`` rather than this
        method inspecting ``group.tokens`` — a beam group's ``tokens`` is ``B x k``
        *nested* lists, so ``not tokens[0]`` would be ``False`` from step one and
        the begin-suppress list would never be applied.

        Out-of-place (``index_fill``) because ``logits`` may alias
        ``group.last_logits``.
        """
        if self._suppress:
            logits = logits.index_fill(
                1, self._ids(self._suppress_idx, self._suppress, logits), -torch.inf
            )
        if group.first_generation_step and self._begin_suppress:
            logits = logits.index_fill(
                1, self._ids(self._begin_suppress_idx, self._begin_suppress, logits), -torch.inf
            )
        return logits

    @staticmethod
    def _ids(
        cache: Dict[torch.device, torch.Tensor], ids: List[int], like: torch.Tensor
    ) -> torch.Tensor:
        idx = cache.get(like.device)
        if idx is None:
            idx = torch.tensor(ids, dtype=torch.long, device=like.device)
            cache[like.device] = idx
        return idx
