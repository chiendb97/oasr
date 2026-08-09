# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Symbol-table tokenizer (WeNet ``units.txt`` / icefall ``tokens.txt``).

Decode-side it is bit-compatible with the legacy
:class:`~oasr.engine.decode.detokenize.Detokenizer`: strip special ids, join
piece strings, treat ``▁`` (U+2581) as a word boundary.  ``encode`` is not
supported — a flat id table cannot segment raw text; checkpoints that need
prompt encoding carry a ``sentencepiece`` / ``huggingface`` spec instead.
"""

from __future__ import annotations

from typing import ClassVar, Dict, FrozenSet, List, Optional, Sequence

from .base import DEFAULT_SPECIAL_IDS, Tokenizer, TokenizerSpec
from .registry import register_tokenizer


def load_symbol_table(path: str) -> Dict[int, str]:
    """Parse a ``<piece> <id>`` table (``units.txt`` / ``tokens.txt`` layout)."""
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


class SymbolTableTokenizer(Tokenizer):
    """Id → piece lookup over a ``units.txt`` / ``tokens.txt`` file."""

    #: Decode-only: the table is id→piece, so there is no reverse map.
    supports_encode: ClassVar[bool] = False

    def __init__(self, table_path: str, special_ids: Optional[FrozenSet[int]] = None) -> None:
        self._table = load_symbol_table(table_path)
        self._special_ids = DEFAULT_SPECIAL_IDS if special_ids is None else special_ids
        # C++ rendering for :meth:`token_pieces`, built once here rather than
        # per call.  Construction tolerates a build without the extension
        # because ``decode`` — the common case, and every CPU test — does not
        # need it; :meth:`token_pieces` does, and simply uses it.
        self._pieces_cpp = self._build_cpp()

    def _build_cpp(self):
        try:
            from oasr import _C  # type: ignore[attr-defined]

            pieces_cls = _C.alignment.SymbolTablePieces
        except (ImportError, AttributeError):  # pragma: no cover - no extension built
            return None
        return pieces_cls(self._table, set(self._special_ids))

    @classmethod
    def from_spec(cls, spec: TokenizerSpec) -> "SymbolTableTokenizer":
        return cls(spec.files["table"], special_ids=spec.special_ids())

    @property
    def vocab_size(self) -> int:
        return max(self._table) + 1 if self._table else 0

    @property
    def special_ids(self) -> FrozenSet[int]:
        return self._special_ids

    def decode(self, ids: Sequence[int]) -> str:
        filtered = [t for t in ids if t not in self._special_ids]
        if not filtered:
            return ""
        pieces = [self._table.get(t, "") for t in filtered]
        return "".join(pieces).replace("▁", " ").strip()

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError(
            "SymbolTableTokenizer cannot encode text (a flat id table has no "
            "segmentation model); use a 'sentencepiece' or 'huggingface' "
            "tokenizer spec for checkpoints that need prompt encoding"
        )

    def decode_incremental(self, new_ids, state):
        """Truly incremental: this kind's rendering is piece-local.

        ``decode`` is a table lookup, a join, a ``▁`` → space substitution and a
        ``strip``.  Every step but the strip is per-piece, so appending tokens
        cannot change how earlier ones render — the only coupling is the leading
        / trailing whitespace that ``strip`` removes.  Track the unstripped join
        and strip on the way out, and the cost is O(new tokens) rather than
        O(whole prefix) per partial.
        """
        raw = state.get("raw", "")
        ids = state.setdefault("ids", [])
        for t in new_ids:
            t = int(t)
            ids.append(t)
            if t not in self._special_ids:
                raw += self._table.get(t, "")
        state["raw"] = raw
        full = raw.replace("▁", " ").strip()
        prev = state.get("text", "")
        state["text"] = full
        # Leading-space stripping means a delta can start mid-word only on the
        # very first emission; after that the prefix is stable.
        return full[len(prev) :] if full.startswith(prev) else full

    def token_pieces(self, ids):
        """One pass, for the same reason :meth:`decode_incremental` is O(new).

        Rendering is piece-local here, so a piece is a table lookup and a ``▁``
        substitution; only ``decode``'s outer ``strip`` couples the ends, and
        the pieces have to lose exactly the characters it would remove or they
        would no longer concatenate to it.  The base class's generic version
        re-renders the whole prefix per token, which is quadratic and lands on
        the CTC decode path — the one place this is called per request.

        Implemented in C++ (``csrc/tokenizers/symbol_table.cc``) and nowhere
        else, like the alignment pass that is its only caller: a Python twin
        here would be a second rendering of the same table, checked by nothing
        at runtime, on the request path where the difference would show up as a
        word boundary in the wrong place.
        """
        # No ``int()`` pass: pybind11's caster does the conversion in C.
        return self._pieces_cpp.pieces(ids)


register_tokenizer("symbol_table", SymbolTableTokenizer.from_spec)
