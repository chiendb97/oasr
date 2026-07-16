# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace ``tokenizer.json`` tokenizer (AED / LLM checkpoints).

Wraps the ``tokenizers`` fast-tokenizer runtime directly (no ``transformers``
dependency).  Emitted by HF-format converters (Whisper / speech-LLM) in later
phases; registered here so the axis is complete from day one.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence

from .base import Tokenizer, TokenizerSpec
from .registry import register_tokenizer


class HuggingFaceTokenizer(Tokenizer):
    """Wraps a ``tokenizers.Tokenizer`` loaded from a ``tokenizer.json``."""

    def __init__(
        self, tokenizer_json_path: str, special_ids: Optional[FrozenSet[int]] = None
    ) -> None:
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "the `tokenizers` package is required for "
                "TokenizerSpec(kind='huggingface'); install it with "
                "`pip install oasr[tokenizers]` or `pip install tokenizers`"
            ) from exc
        self._tok = HFTokenizer.from_file(tokenizer_json_path)
        # HF tokenizers know their own added special tokens; the explicit set is
        # an additional strip list on top of skip_special_tokens=True.
        self._special_ids = frozenset() if special_ids is None else special_ids

    @classmethod
    def from_spec(cls, spec: TokenizerSpec) -> "HuggingFaceTokenizer":
        ids = spec.options.get("special_ids")
        return cls(
            spec.files["tokenizer"],
            special_ids=None if ids is None else frozenset(int(i) for i in ids),
        )

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size())

    @property
    def special_ids(self) -> FrozenSet[int]:
        return self._special_ids

    def decode(self, ids: Sequence[int]) -> str:
        filtered = [int(t) for t in ids if t not in self._special_ids]
        if not filtered:
            return ""
        return self._tok.decode(filtered, skip_special_tokens=True)

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)


register_tokenizer("huggingface", HuggingFaceTokenizer.from_spec)
