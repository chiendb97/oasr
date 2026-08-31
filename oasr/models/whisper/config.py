# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Whisper model configuration (mirrors the HF ``WhisperConfig`` fields OASR uses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..base import BaseModelConfig


@dataclass
class WhisperModelConfig(BaseModelConfig):
    """Encoder-decoder Whisper hyperparameters + generation control ids.

    The generation fields (``decoder_start_token_id`` / ``forced_decoder_ids``
    / suppress lists) travel here because they are checkpoint properties (from
    ``config.json`` / ``generation_config.json``), not engine choices — the
    ``aed`` decode strategy reads them to build the SOT prompt and the logit
    suppression masks.
    """

    model_type: str = "whisper"
    # vocab_size inherited from BaseModelConfig (51865 for multilingual tiny).
    d_model: int = 384
    encoder_layers: int = 4
    decoder_layers: int = 4
    encoder_attention_heads: int = 6
    decoder_attention_heads: int = 6
    encoder_ffn_dim: int = 1536
    decoder_ffn_dim: int = 1536
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    max_target_positions: int = 448
    activation_function: str = "gelu"

    # -- generation control (checkpoint-derived) ---------------------------
    decoder_start_token_id: int = 50258  # <|startoftranscript|>
    eos_token_id: int = 50257  # <|endoftext|>
    # [(position, token_id)] forced after SOT — language / task / notimestamps.
    forced_decoder_ids: List[Tuple[int, int]] = field(default_factory=list)
    # Token ids whose logits are set to -inf at every generation step.
    suppress_tokens: List[int] = field(default_factory=list)
    # Token ids suppressed only at the first *generated* step.
    begin_suppress_tokens: List[int] = field(default_factory=list)
    # ``{"transcribe": id, "translate": id}`` and ``{"en": id, ...}``, read off
    # the checkpoint's tokenizer by the converter.  They exist so the task and
    # language slots of the SOT prompt can be set **per request** instead of
    # being frozen at conversion time — which is what blocked
    # ``POST /v1/audio/translations``.  Empty on a checkpoint converted before
    # they existed: the per-request options then fail loudly rather than
    # decoding under the checkpoint's own task.
    task_token_ids: Dict[str, int] = field(default_factory=dict)
    language_token_ids: Dict[str, int] = field(default_factory=dict)
    # ``[[layer, head], ...]`` — the cross-attention heads that were found to
    # align with the audio, from ``generation_config.json``.  Word timestamps
    # average exactly these; see
    # :mod:`oasr.engine.decode.attention_align` for why averaging *all* heads
    # produces a near-diagonal path instead of an alignment.  Empty falls back
    # to the upper half of the decoder stack, with a warning.
    alignment_heads: List[Tuple[int, int]] = field(default_factory=list)
    # ``<|nospeech|>`` (a.k.a. ``<|nocaptions|>``), read off the checkpoint's own
    # tokenizer.  Whisper is trained to predict it at the first generated
    # position when a window carries no speech, which makes it the only
    # speech-activity signal this family has.  ``None`` on a snapshot without
    # the token (English-only builds, and anything converted before this field
    # existed): the per-request VAD options then fail loudly rather than
    # reporting a probability derived from some other token's logit.
    no_speech_token_id: Optional[int] = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads

    def sot_sequence(self, task: Optional[str] = None, language: Optional[str] = None) -> List[int]:
        """The decoder prompt: ``<|startoftranscript|>`` + forced ids in
        position order (language, task, ``<|notimestamps|>``).

        ``task`` / ``language`` substitute the corresponding slot.  A slot is
        identified by the forced token *being* one of the known task/language
        ids, so this works for any Whisper-family checkpoint without hardcoding
        an id or a position — those move between multilingual releases.

        Raises :class:`ValueError` when the checkpoint has no such slot (an
        English-only Whisper has no language token at all) or does not know the
        requested value.  Returning the default prompt instead would transcribe
        in the wrong language, or transcribe when translation was asked for,
        with nothing in the response to say so.
        """
        prompt = [self.decoder_start_token_id]
        task_ids = set(self.task_token_ids.values())
        lang_ids = set(self.language_token_ids.values())
        want_task = self._resolve(task, self.task_token_ids, "task")
        want_lang = self._resolve(language, self.language_token_ids, "language")
        placed_task = placed_lang = False
        for _pos, tok in sorted(self.forced_decoder_ids, key=lambda pt: pt[0]):
            tok = int(tok)
            if want_lang is not None and tok in lang_ids:
                tok, placed_lang = want_lang, True
            elif want_task is not None and tok in task_ids:
                tok, placed_task = want_task, True
            prompt.append(tok)
        if want_task is not None and not placed_task:
            raise ValueError(
                f"this checkpoint's decoder prompt has no task slot, so task={task!r} "
                "cannot be applied (an English-only Whisper is transcribe-only)"
            )
        if want_lang is not None and not placed_lang:
            raise ValueError(
                f"this checkpoint's decoder prompt has no language slot, so "
                f"language={language!r} cannot be applied (an English-only Whisper "
                "has no language token)"
            )
        return prompt

    @staticmethod
    def _resolve(value: Optional[str], table: Dict[str, int], what: str) -> Optional[int]:
        """``value`` → its token id, or a message naming what is available."""
        if value is None:
            return None
        if not table:
            raise ValueError(
                f"this checkpoint carries no {what} token table, so {what}={value!r} "
                "cannot be applied; re-run `oasr-convert` to record it"
            )
        try:
            return int(table[value])
        except KeyError:
            known = sorted(table)
            shown = known if len(known) <= 12 else known[:12] + ["..."]
            raise ValueError(
                f"unknown {what} {value!r} for this checkpoint; known: {shown}"
            ) from None
