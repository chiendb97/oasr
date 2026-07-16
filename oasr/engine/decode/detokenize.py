# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Token-id → text detokenization, shared by every decode strategy.

Since the tokenizer axis landed (:mod:`oasr.tokenizers`), this class is a thin
backward-compatible adapter: when the engine has a converter-emitted
:class:`~oasr.tokenizers.TokenizerSpec` it injects the built
:class:`~oasr.tokenizers.Tokenizer` here; the legacy path (``unit_table`` /
``sentencepiece_model`` file paths, engine-side sniffing) builds the same
``symbol_table`` tokenizer and is decode-for-decode identical to the historical
behavior: strip special ids {0, 1, 2}, join ``units.txt`` pieces, treat ``▁``
(U+2581) as a word boundary.  The SentencePiece model is loaded when available
but **not** used for decoding — its internal piece ids differ from the CTC
output ids (which come from ``units.txt``); it is kept only for callers that
want the processor object.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from oasr.tokenizers import SymbolTableTokenizer, Tokenizer

logger = logging.getLogger(__name__)

# Token IDs to strip during detokenization: <blank>, <unk>, <sos/eos>.
SPECIAL_IDS = frozenset([0, 1, 2])


class Detokenizer:
    """Maps decoded token-id sequences to text via a :class:`Tokenizer`."""

    def __init__(
        self,
        sentencepiece_model: Optional[str] = None,
        unit_table: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self._sp = self._load_sentencepiece(sentencepiece_model)
        self._tokenizer: Optional[Tokenizer] = tokenizer
        if self._tokenizer is None and unit_table is not None:
            self._tokenizer = SymbolTableTokenizer(unit_table, special_ids=SPECIAL_IDS)

    @property
    def tokenizer(self) -> Optional[Tokenizer]:
        """The underlying tokenizer (``None`` when only the id-join fallback exists)."""
        return self._tokenizer

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to text.

        Delegates to the tokenizer when one is available; otherwise strips the
        default special ids and joins the raw ids as a last resort.
        """
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids)

        # Last resort: join as-is
        filtered = [t for t in token_ids if t not in SPECIAL_IDS]
        return " ".join(str(t) for t in filtered)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sentencepiece(path: Optional[str]):
        if path is None:
            return None
        try:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.Load(path)
            return sp
        except Exception as exc:
            logger.warning("Could not load SentencePiece model %s: %s", path, exc)
            return None
