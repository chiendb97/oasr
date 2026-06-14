# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Token-id → text detokenization, shared by every decode strategy.

Strips special tokens (blank / unk / sos-eos), looks up BPE piece strings via
``units.txt``, and joins them treating ``▁`` (U+2581) as a word boundary.  The
SentencePiece model is loaded when available but **not** used for decoding —
its internal piece ids differ from the CTC output ids (which come from
``units.txt``); it is kept only for callers that want the processor object.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Token IDs to strip during detokenization: <blank>, <unk>, <sos/eos>.
SPECIAL_IDS = frozenset([0, 1, 2])


class Detokenizer:
    """Maps decoded token-id sequences to text using ``units.txt`` pieces."""

    def __init__(
        self,
        sentencepiece_model: Optional[str] = None,
        unit_table: Optional[str] = None,
    ) -> None:
        self._sp = self._load_sentencepiece(sentencepiece_model)
        self._vocab: Optional[Dict[int, str]] = None
        if unit_table is not None:
            self._vocab = self._load_unit_table(unit_table)

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to text.

        Strips special tokens (blank, unk, sos/eos), looks up ``units.txt``
        piece strings, then joins treating ``▁`` (U+2581) as a word boundary.
        """
        filtered = [t for t in token_ids if t not in SPECIAL_IDS]
        if not filtered:
            return ""

        if self._vocab is not None:
            pieces = [self._vocab.get(t, "") for t in filtered]
            text = "".join(pieces)
            return text.replace("▁", " ").strip()

        # Last resort: join as-is
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

    @staticmethod
    def _load_unit_table(path: str) -> Dict[int, str]:
        vocab: Dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    token, idx = parts[0], int(parts[1])
                    vocab[idx] = token
        return vocab
