# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Parallel long-form decoding for fixed-window frontends.

Long requests are split into ordinary batched requests and merged at the engine
boundary. Independent windows preserve batching but may lose or duplicate text
at cuts; overlap and text deduplication reduce that boundary error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .request import RequestOutput

logger = logging.getLogger(__name__)

#: Words compared when looking for the overlap between two adjacent windows.
#: Bounded so the search cost does not grow with transcript length; an overlap
#: longer than this many words means ``overlap_seconds`` is set absurdly high.
_MAX_MERGE_WORDS = 40


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


def merge_texts(pieces: Sequence[str]) -> str:
    """Join per-window transcripts, dropping text the overlap duplicated.

    Compares the tail words of the accumulated text against the head words of
    the next piece and skips the longest match.  Word-level rather than
    character-level on purpose: the two windows saw *different* audio context so
    they will not agree character-for-character even where they agree on words,
    and a character-level match would join mid-word.
    """
    merged: List[str] = []
    for piece in pieces:
        words = piece.split()
        if not words:
            continue
        if not merged:
            merged = words
            continue
        limit = min(len(merged), len(words), _MAX_MERGE_WORDS)
        overlap = 0
        for k in range(limit, 0, -1):
            if [w.lower() for w in merged[-k:]] == [w.lower() for w in words[:k]]:
                overlap = k
                break
        merged.extend(words[overlap:])
    return " ".join(merged)


@dataclass
class _Pending:
    """One long-form parent request awaiting its windows."""

    parent_id: str
    child_ids: List[str]
    window_starts_s: List[float]
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

    def __init__(self) -> None:
        self._pending: Dict[str, _Pending] = {}
        self._parent_of: Dict[str, str] = {}

    def __bool__(self) -> bool:
        return bool(self._pending)

    def child_id(self, parent_id: str, index: int) -> str:
        return f"{self.CHILD_PREFIX}{parent_id}#{index}"

    def register(self, parent_id: str, child_ids: List[str], window_starts_s: List[float]) -> None:
        self._pending[parent_id] = _Pending(parent_id, list(child_ids), list(window_starts_s))
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
        """One output for the whole file, windows in submission order."""
        ordered = [entry.results[cid] for cid in entry.child_ids]
        text = merge_texts([o.text for o in ordered])

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

        # Timestamps shift into file time; families without them (AED) report
        # none.  Tokens are concatenated so a caller can still see the ids, but
        # n-best is **not** merged: alternatives of independent windows do not
        # compose into alternatives of the file, and inventing a cross product
        # would be worse than reporting one hypothesis.
        stamps: List = []
        words: List = []
        for out, start in zip(ordered, entry.window_starts_s):
            if out.timestamps:
                stamps.extend((s + start, e + start) for s, e in out.timestamps)
            for w in out.words or ():
                # Word timings shift the same way; ``_replace`` keeps the type
                # rather than rebuilding it here, so a field added to
                # ``WordTiming`` survives the merge without an edit.
                words.append(w._replace(start=w.start + start, end=w.end + start))
        tokens: List[int] = []
        confidences: List[float] = []
        for out in ordered:
            if out.tokens:
                tokens.extend(out.tokens[0])
            if out.confidence is not None:
                confidences.append(out.confidence)

        return RequestOutput(
            request_id=entry.parent_id,
            text=text,
            tokens=[tokens],
            finished=True,
            finish_reason=reason,
            timestamps=stamps or None,
            words=words or None,
            # A mean of per-window means, not of per-token posteriors: the
            # windows are equal-length by construction, so the two agree except
            # at the tail, and this needs no per-window token count.
            confidence=(sum(confidences) / len(confidences)) if confidences else None,
        )


__all__ = ["LongFormTracker", "merge_texts", "split_windows"]
