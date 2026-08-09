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
(U+2581) as a word boundary.

``sentencepiece_model`` is accepted and ignored.  It used to be eagerly loaded
here — file I/O plus a resident model at engine init for every checkpoint dir
containing a ``*.model``, which the engine auto-sniffed — but nothing ever read
it: SentencePiece piece ids differ from the CTC output ids (which come from
``units.txt``), so it could not have been used for decoding.  The parameter
stays for callers passing it positionally.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

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
    # Incremental detokenization (T3)
    # ------------------------------------------------------------------
    #
    # For the append-only families — AR generation and transducer greedy — a
    # partial extends the previous hypothesis rather than replacing it, so
    # re-decoding the whole prefix every tick is Θ(n²) for no new information.
    # These two methods let such a strategy feed only what it just produced.
    #
    # CTC prefix beam search deliberately does **not** use them: its best
    # hypothesis can be re-ranked between chunks, so the prefix is not monotone
    # and "what's new" is undefined.

    def new_state(self) -> Dict[str, Any]:
        """Fresh per-request incremental-decode state."""
        if self._tokenizer is not None:
            return self._tokenizer.new_decode_state()
        return {"ids": [], "text": ""}

    def detokenize_incremental(self, new_ids: Sequence[int], state: Dict[str, Any]) -> str:
        """Extend a hypothesis by ``new_ids``; return the text delta.

        ``state["text"]`` carries the full transcript so far.  Concatenating
        every delta equals :meth:`detokenize` over the accumulated ids.
        """
        if self._tokenizer is not None:
            return self._tokenizer.decode_incremental(new_ids, state)
        ids = state.setdefault("ids", [])
        ids.extend(int(i) for i in new_ids)
        full = self.detokenize(ids)
        prev = state.get("text", "")
        state["text"] = full
        return full[len(prev) :] if full.startswith(prev) else full

    def token_pieces(self, token_ids: Sequence[int]) -> List[str]:
        """Per-token text contributions, concatenating to :meth:`detokenize`.

        What the word grouping needs (see
        :mod:`oasr.engine.decode.alignment`): the character range each token
        contributed to the rendered transcript.  Delegates to the tokenizer,
        which can answer it in one pass where its rendering is piece-local;
        the id-join fallback splits on its own separator.
        """
        if self._tokenizer is not None:
            return self._tokenizer.token_pieces(token_ids)
        pieces: List[str] = []
        seen = False
        for tok in token_ids:
            if int(tok) in SPECIAL_IDS:
                pieces.append("")  # stripped by ``detokenize``; owns no text
                continue
            pieces.append(f" {int(tok)}" if seen else str(int(tok)))
            seen = True
        return pieces
