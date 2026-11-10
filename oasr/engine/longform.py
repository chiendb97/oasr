# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Parallel long-form decoding for fixed-window frontends.

Long requests are split into ordinary batched requests and merged at the engine
boundary. Independent windows preserve batching at the cost of context across a
cut, so the engine cuts in pauses (``OfflineVadSegmenter.pieces``) and the
windows share no audio.  :class:`LongFormTracker` still stitches windows that
do overlap, dropping the words the later one repeats, wherever a caller
registers such windows.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .request import RequestOutput

logger = logging.getLogger(__name__)

#: Words compared when looking for the overlap between two adjacent windows.
#: Bounded so the search cost does not grow with transcript length; an overlap
#: longer than this many words means ``overlap_seconds`` is set absurdly high.
_MAX_MERGE_WORDS = 40

_WORD = re.compile(r"\S+")

#: ``"".join(token_pieces(ids)) == decode(ids)``: each token's text, in order
#: (:meth:`~oasr.engine.decode.detokenize.Detokenizer.token_pieces`).
TokenPieces = Callable[[Sequence[int]], List[str]]


def split_windows(
    audio: torch.Tensor,
    window_samples: int,
    overlap_samples: int = 0,
) -> List[torch.Tensor]:
    """Split ``audio`` into windows of at most ``window_samples``.

    Consecutive windows advance by ``window_samples - overlap_samples``, so
    ``overlap_samples`` of audio is shared with the previous window.  The last
    window is short (the frontend pads it) and never empty.
    """
    n = int(audio.numel())
    if window_samples <= 0:
        raise ValueError(f"window_samples must be positive, got {window_samples}")
    overlap = max(0, min(int(overlap_samples), window_samples - 1))
    stride = window_samples - overlap
    if n <= window_samples:
        return [audio]
    out: List[torch.Tensor] = []
    start = 0
    while start < n:
        out.append(audio[start : start + window_samples])
        if start + window_samples >= n:
            break
        start += stride
    return out


def repeated_words(previous: Sequence[str], words: Sequence[str]) -> int:
    """How many leading ``words`` repeat the tail of ``previous``: the longest match.

    The seam rule for two windows that share audio.  Word-level rather than
    character-level on purpose: the two windows saw *different* audio context so
    they will not agree character-for-character even where they agree on words,
    and a character-level match would join mid-word.  ``previous`` is the whole
    transcript of the window before, not what survived of it: the shared audio
    is that window's tail, whatever its own head lost to the window before it.
    """
    limit = min(len(previous), len(words), _MAX_MERGE_WORDS)
    for k in range(limit, 0, -1):
        if [w.lower() for w in previous[-k:]] == [w.lower() for w in words[:k]]:
            return k
    return 0


def merge_texts(pieces: Sequence[str]) -> str:
    """Join the transcripts of windows that each overlap the one before.

    Drops the head of each piece that repeats the tail of the previous one
    (:func:`repeated_words`).  An empty piece shares nothing with the next:
    windows overlap by at most half their width, so the windows on either
    side of it share no audio.
    """
    merged: List[str] = []
    previous: List[str] = []
    for piece in pieces:
        words = piece.split()
        merged.extend(words[repeated_words(previous, words) :])
        previous = words
    return " ".join(merged)


def _offset_of_word(text: str, index: int) -> int:
    """Character offset of ``text``'s ``index``-th word, or ``len(text)`` past the last."""
    if index == 0:
        return 0
    for n, m in enumerate(_WORD.finditer(text)):
        if n == index:
            return m.start()
    return len(text)


@dataclass
class _Pending:
    """One long-form parent request awaiting its windows."""

    parent_id: str
    child_ids: List[str]
    window_starts_s: List[float]
    window_ends_s: List[float]
    results: Dict[str, RequestOutput] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return len(self.results) == len(self.child_ids)


class LongFormTracker:
    """Fan-out bookkeeping: child window outputs → one stitched parent output.

    Held by :class:`~oasr.engine.ASREngine`; ``register`` at admission and
    ``absorb`` on every step's outputs.  Thread safety comes from the engine's
    own lock — every entry point that touches this already holds it.
    """

    #: Prefix for generated child ids.  Namespaced so a child can never collide
    #: with a caller-supplied request id, and recognisable in logs.
    CHILD_PREFIX = "__lf__"

    def __init__(self, token_pieces: Optional[TokenPieces] = None) -> None:
        """``token_pieces`` maps a window's tokens onto its text, so a seam that
        drops repeated words drops their tokens too.  Without it, or for a
        window whose text it does not reproduce, the tokens stay whole."""
        self._pending: Dict[str, _Pending] = {}
        self._parent_of: Dict[str, str] = {}
        self._token_pieces = token_pieces
        self._warned_unmapped = False

    def __bool__(self) -> bool:
        return bool(self._pending)

    def child_id(self, parent_id: str, index: int) -> str:
        return f"{self.CHILD_PREFIX}{parent_id}#{index}"

    def register(
        self,
        parent_id: str,
        child_ids: List[str],
        window_starts_s: List[float],
        window_ends_s: List[float],
    ) -> None:
        """Expect ``child_ids``, covering ``[start, end)`` of the file each.

        Two windows overlap where one ends after the next starts; only there is
        any of the next window's transcript a repeat.  A seam that shares no
        audio (a VAD cut in silence) keeps every word on both sides, since a
        phrase said again after a pause is not a duplicate.
        """
        self._pending[parent_id] = _Pending(
            parent_id, list(child_ids), list(window_starts_s), list(window_ends_s)
        )
        for cid in child_ids:
            self._parent_of[cid] = parent_id

    def owns(self, request_id: str) -> bool:
        return request_id in self._parent_of

    def absorb(self, outputs: List[RequestOutput]) -> List[RequestOutput]:
        """Replace child outputs with stitched parent outputs where complete.

        Non-long-form outputs pass through untouched, and interim child partials
        are dropped: a partial for window 3 of 8 is not a partial of the parent
        transcript, and emitting it would have the client render the file's
        transcript out of order.
        """
        if not self._pending:
            return outputs
        passthrough: List[RequestOutput] = []
        for out in outputs:
            parent_id = self._parent_of.get(out.request_id)
            if parent_id is None:
                passthrough.append(out)
                continue
            if not out.finished:
                continue
            entry = self._pending.get(parent_id)
            if entry is None:  # already stitched (duplicate final)
                continue
            entry.results[out.request_id] = out
            if entry.complete:
                passthrough.append(self._stitch(entry))
                for cid in entry.child_ids:
                    self._parent_of.pop(cid, None)
                self._pending.pop(parent_id, None)
        return passthrough

    def abandon(self, parent_id: str) -> List[str]:
        """Drop a parent (abort path); returns the child ids to abort."""
        entry = self._pending.pop(parent_id, None)
        if entry is None:
            return []
        for cid in entry.child_ids:
            self._parent_of.pop(cid, None)
        return entry.child_ids

    def parent_for_child(self, child_id: str) -> Optional[str]:
        return self._parent_of.get(child_id)

    def _stitch(self, entry: _Pending) -> RequestOutput:
        """One output for the whole file, windows in submission order.

        Where two windows overlap, the head of the later one that repeats the
        earlier one's tail is dropped from every field, not only the text: one
        decision per seam (:func:`repeated_words`) cuts the window's text at a
        character offset, its words where they sit in that text, and its tokens
        and per-token timestamps where they render it.  Speech segments, which
        no text owns, are cut at the middle of the shared audio instead.
        """
        ordered = [entry.results[cid] for cid in entry.child_ids]
        starts, ends = entry.window_starts_s, entry.window_ends_s
        # ``seams[i]`` sits between windows ``i`` and ``i + 1``: the middle of the
        # audio they share, or ``None`` where they share none.
        seams: List[Optional[float]] = [
            (starts[i + 1] + ends[i]) / 2.0 if ends[i] > starts[i + 1] else None
            for i in range(len(ordered) - 1)
        ]

        text: List[str] = []
        previous: List[str] = []
        tokens: List[int] = []
        stamps: List[Tuple[float, float]] = []
        words: List = []
        segments: List = []
        detected = False
        for i, (out, start) in enumerate(zip(ordered, starts)):
            own = out.text.split()
            drop = repeated_words(previous, own) if i and seams[i - 1] is not None else 0
            previous = own
            text.extend(own[drop:])
            cut = _offset_of_word(out.text, drop)

            row = out.tokens[0] if out.tokens else []
            skip = self._tokens_before(out.text, row, cut) if cut else 0
            tokens.extend(row[skip:])
            if out.timestamps:
                # Aligned with ``row``, so they lose the same tokens.
                aligned = len(out.timestamps) == len(row)
                stamps.extend(
                    (s + start, e + start) for s, e in out.timestamps[skip if aligned else 0 :]
                )
            # Word timings shift into file time; ``_replace`` keeps the type
            # rather than rebuilding it here, so a field added to ``WordTiming``
            # survives the merge without an edit.
            words.extend(
                w._replace(start=w.start + start, end=w.end + start) for w in _words_from(out, cut)
            )
            if out.segments is not None:
                detected = True
                left = seams[i - 1] if i else None
                right = seams[i] if i < len(seams) else None
                for seg in out.segments:
                    _add_segment(segments, seg, start, left, right)

        # ``finish_reason``: any window that hit its generation cap means the
        # transcript is incomplete, and that has to survive the merge — it is the
        # only signal the client gets that asking for more tokens would help.
        reasons = [o.finish_reason for o in ordered]
        reason: Optional[str] = None
        if any(r == "error" for r in reasons):
            reason = "error"
        elif any(r == "length" for r in reasons):
            reason = "length"
        elif any(r is not None for r in reasons):
            reason = "stop"

        # n-best is **not** merged: alternatives of independent windows do not
        # compose into alternatives of the file, and inventing a cross product
        # would be worse than reporting one hypothesis.
        # ``confidence`` is a mean over tokens, so each window's counts for the
        # tokens it was taken over: VAD segments are anything but equal-length.
        weighted = 0.0
        weight = 0
        overflow = 0  # any window the search could not complete leaves the file short
        for out in ordered:
            if out.confidence is not None and out.tokens:
                weighted += out.confidence * len(out.tokens[0])
                weight += len(out.tokens[0])
            overflow |= out.search_overflow or 0
        # The file has no speech only if none of its windows has any, so the
        # least of their no-speech probabilities bounds the file's from above.
        no_speech = [o.no_speech_prob for o in ordered if o.no_speech_prob is not None]

        return RequestOutput(
            request_id=entry.parent_id,
            text=" ".join(text),
            tokens=[tokens],
            finished=True,
            finish_reason=reason,
            error_stage=next((o.error_stage for o in ordered if o.finish_reason == "error"), None),
            # Per token of ``tokens``, so only when every window had them.
            timestamps=stamps if stamps and len(stamps) == len(tokens) else None,
            words=words or None,
            confidence=weighted / weight if weight else None,
            segments=segments if detected else None,
            no_speech_prob=min(no_speech) if no_speech else None,
            search_overflow=overflow or None,
        )

    def _tokens_before(self, text: str, row: Sequence[int], cut: int) -> int:
        """How many leading tokens of ``row`` render only ``text[:cut]``.

        A token whose piece is only whitespace, or nothing (a special id),
        goes with the run it sits in.  ``0`` -- keep them all -- when the
        tokens cannot be mapped onto ``text``.
        """
        if not row:
            return 0
        pieces = self._token_pieces(row) if self._token_pieces is not None else None
        if pieces is None or "".join(pieces) != text:
            if not self._warned_unmapped:
                self._warned_unmapped = True
                logger.warning(
                    "long-form: a window's tokens do not render its text, so tokens "
                    "repeated at a window seam are kept (the text is still merged)"
                )
            return 0
        at = 0
        for k, piece in enumerate(pieces):
            if piece.strip() and at + len(piece.rstrip()) > cut:
                return k
            at += len(piece)
        return len(pieces)


def _words_from(out: RequestOutput, cut: int) -> List:
    """``out.words`` that sit at or after ``out.text[cut:]``.

    A word is a literal substring of the text, in order (see
    :mod:`oasr.engine.decode.alignment`), which is what places it.
    """
    if not out.words or cut == 0:
        return list(out.words or ())
    kept = []
    at = 0
    for w in out.words:
        found = out.text.find(w.word, at)
        if found < 0:  # not the contract: nothing to place them by
            return list(out.words)
        at = found + len(w.word)
        if found >= cut:
            kept.append(w)
    return kept


def _add_segment(
    segments: List,
    seg,
    offset: float,
    left: Optional[float],
    right: Optional[float],
) -> None:
    """Append one window's speech segment in file time, inside the window's share.

    ``left`` and ``right`` are the cuts at the window's overlapping seams, so
    each stretch of shared audio is reported by one window: the earlier one up
    to the cut, the later one after.  A span the cut divided is rejoined.
    """
    start, end = seg.start + offset, seg.end + offset
    if left is not None:
        start = max(start, left)
    if right is not None:
        end = min(end, right)
    if end <= start:
        return
    prob = seg.speech_prob
    if segments and left is not None and start == left and segments[-1].end == left:
        before = segments.pop()
        length = end - before.start
        prob = (before.speech_prob * (left - before.start) + prob * (end - left)) / length
        start = before.start
    segments.append(dataclasses.replace(seg, start=start, end=end, speech_prob=prob))


__all__ = ["LongFormTracker", "merge_texts", "repeated_words", "split_windows"]
