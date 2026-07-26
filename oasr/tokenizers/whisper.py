# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Whisper tokenizer (GPT-2-style BPE from HF ``tokenizer.json``).

A thin specialization of :class:`~oasr.tokenizers.huggingface.HuggingFaceTokenizer`:
decoding additionally strips *every* Whisper control id — the special-token
block starting at ``<|endoftext|>`` covers the SOT sequence, language tags,
task tags, ``<|notimestamps|>`` and all timestamp tokens, so text never leaks
control markup even if a decode strategy hands over raw hypothesis ids.

Spec layout (emitted by the HF Whisper converter)::

    TokenizerSpec(
        kind="whisper",
        files={"tokenizer": ".../tokenizer.json"},
        options={"eot_id": 50257},   # first control id; everything >= is special
    )
"""

from __future__ import annotations

from typing import List, Sequence

from .base import TokenizerSpec
from .huggingface import HuggingFaceTokenizer
from .registry import register_tokenizer


class WhisperTokenizer(HuggingFaceTokenizer):
    """HF fast tokenizer + Whisper control-token stripping."""

    def __init__(self, tokenizer_json_path: str, eot_id: int) -> None:
        super().__init__(tokenizer_json_path, special_ids=frozenset({int(eot_id)}))
        self._eot_id = int(eot_id)
        # ``decode`` keeps only ids below ``eot_id``, so *everything* at or above
        # it is stripped — the language / task / timestamp control block, ~1500
        # ids, not just ``<|endoftext|>``.  ``special_ids`` has to say so: a
        # caller filtering a hypothesis by it would otherwise leak control markup
        # that ``decode`` removes.
        self._special_ids = frozenset(range(self._eot_id, self.vocab_size))

    @classmethod
    def from_spec(cls, spec: TokenizerSpec) -> "WhisperTokenizer":
        return cls(spec.files["tokenizer"], eot_id=int(spec.options.get("eot_id", 50257)))

    @property
    def eot_id(self) -> int:
        return self._eot_id

    @property
    def special_ids(self):
        return self._special_ids

    def decode(self, ids: Sequence[int]) -> str:
        text_ids: List[int] = [int(t) for t in ids if t < self._eot_id]
        if not text_ids:
            return ""
        return self._tok.decode(text_ids, skip_special_tokens=True).strip()


register_tokenizer("whisper", WhisperTokenizer.from_spec)
