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

from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .base import DEFAULT_SPECIAL_IDS, TokenizerSpec, WindowedIncrementalTokenizer
from .registry import register_tokenizer


class SentencePieceTokenizer(WindowedIncrementalTokenizer):
    """Wraps a ``sentencepiece.SentencePieceProcessor`` (optional dependency).

    Incremental decoding re-renders a bounded window
    (:class:`~oasr.tokenizers.base.WindowedIncrementalTokenizer`).  Inheriting
    the base class's whole-prefix re-decode instead made every streaming
    partial linear in the transcript -- 5.5 ms at 12k tokens, per stream, per
    tick, under the GIL -- and :meth:`token_pieces` quadratic: 0.9 s at 2k
    tokens (audit #27).
    """

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
        self._closes_bos: Dict[int, bool] = {}
        self._primer, self._probe = self._leading_space_pieces()

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

    # ------------------------------------------------------------------
    # The one coupling a window cannot see: the leading-space rule
    # ------------------------------------------------------------------
    #
    # SentencePiece drops the ``▁`` of the first visible piece of whatever it
    # decodes (``▁THE`` renders ``THE`` there and `` THE`` after a word), and it
    # stops doing so only once a visible, non-byte piece has rendered.  That is
    # state carried from the very start of the text, not from the last few
    # tokens, so a window decoded on its own renders its first word without the
    # space whenever the window's other tokens render nothing.  The window is
    # therefore rendered behind a *primer* -- a visible piece that ends the rule
    # -- once some earlier piece has ended it in the real text, and on its own
    # before that.  Which ids end it is asked of SentencePiece itself, per id,
    # rather than re-derived from the piece types.

    def _leading_space_pieces(self) -> Tuple[Optional[Tuple[int, str]], Optional[Tuple[int, str]]]:
        """``(primer, probe)``, each ``(id, rendering at the start of a text)``.

        The primer renders the same whatever follows it and ends the rule.  The
        probe renders differently at the start than after the primer; ``None``
        means no piece does, so the model has no rule and a plain window is
        exact.  A model with no primer keeps every window anchored at the start:
        correct, at the old cost.
        """
        sp = self._sp

        def candidates():  # lazily: both searches stop at their first match
            for i in range(self._vocab_size):
                if i in self._special_ids or sp.IsControl(i) or sp.IsUnknown(i):
                    continue
                if not (sp.IsUnused(i) or sp.IsByte(i)):
                    yield i

        primer = None
        for i in candidates():
            text = self.decode([i])
            if text and not any(c.isspace() for c in text) and self.decode([i, i]) == text * 2:
                primer = (i, text)
                break
        if primer is None:
            return None, None
        p_id, p_text = primer
        for i in candidates():
            alone = self.decode([i])
            if alone and self.decode([p_id, i]) != p_text + alone:
                return primer, (i, alone)
        return primer, None

    def _ends_leading_space_rule(self, token: int) -> bool:
        """Whether ``token``, decoded first, leaves the next ``▁`` rendered as a space."""
        known = self._closes_bos.get(token)
        if known is None:
            assert self._probe is not None
            q_id, q_text = self._probe
            known = self.decode([token, q_id]) != self.decode([token]) + q_text
            self._closes_bos[token] = known
        return known

    def _rule_ended_before(self, ids: List[int], anchor: int, state: Dict[str, Any]) -> bool:
        """Whether an id before ``anchor`` ended the rule.  Scans each id once."""
        ended_at = state.get("bos_ended_at")
        if ended_at is None:
            i = state.get("bos_scanned", 0)
            while i < len(ids):
                if self._ends_leading_space_rule(ids[i]):
                    ended_at = state["bos_ended_at"] = i
                    break
                i += 1
            state["bos_scanned"] = i
        return ended_at is not None and ended_at < anchor

    def _render_window(self, ids: List[int], anchor: int, state: Dict[str, Any]) -> str:
        window = ids[anchor:]
        if not anchor or self._probe is None or not self._rule_ended_before(ids, anchor, state):
            return self.decode(window)
        assert self._primer is not None
        p_id, p_text = self._primer
        return self.decode([p_id] + window)[len(p_text) :]

    def _may_reanchor(self, ids: List[int], anchor: int, state: Dict[str, Any]) -> bool:
        return self._primer is not None


register_tokenizer("sentencepiece", SentencePieceTokenizer.from_spec)
