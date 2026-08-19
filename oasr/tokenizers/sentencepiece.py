# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SentencePiece tokenizer (icefall ``bpe.model``-style checkpoints).

Only valid when the checkpoint's output ids **are** the SentencePiece piece ids
(true for icefall BPE recipes, where ``tokens.txt`` is generated from the
model).  WeNet CTC unit ids differ from the piece ids, which is why WeNet
checkpoints carry a ``symbol_table`` spec even when a ``.model`` file sits in
the directory.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence

from .base import DEFAULT_SPECIAL_IDS, Tokenizer, TokenizerSpec
from .registry import register_tokenizer


class SentencePieceTokenizer(Tokenizer):
    """Wraps a ``sentencepiece.SentencePieceProcessor`` (optional dependency)."""

    def __init__(self, model_path: str, special_ids: Optional[FrozenSet[int]] = None) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "sentencepiece is required for TokenizerSpec(kind='sentencepiece'); "
                "install it with `pip install oasr[tokenizers]` or `pip install sentencepiece`"
            ) from exc
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self._special_ids = DEFAULT_SPECIAL_IDS if special_ids is None else special_ids
        # Fixed for the model's lifetime, and read on every decode().
        self._vocab_size = int(self._sp.GetPieceSize())

    @classmethod
    def from_spec(cls, spec: TokenizerSpec) -> "SentencePieceTokenizer":
        return cls(spec.files["model"], special_ids=spec.special_ids())

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_ids(self) -> FrozenSet[int]:
        return self._special_ids

    def decode(self, ids: Sequence[int]) -> str:
        # Padded output heads may emit ids beyond the tokenizer vocabulary on
        # degenerate input; discard them instead of failing the request.
        n = self.vocab_size
        filtered = [int(t) for t in ids if t not in self._special_ids and 0 <= int(t) < n]
        if not filtered:
            return ""
        return self._sp.DecodeIds(filtered)

    def encode(self, text: str) -> List[int]:
        return list(self._sp.EncodeAsIds(text))


register_tokenizer("sentencepiece", SentencePieceTokenizer.from_spec)
