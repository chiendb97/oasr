# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Word- and character-error-rate measurement.

The repo has thorough *numerical parity* testing and, until this module, no
end-to-end accuracy measurement at all.  Parity oracles structurally cannot
catch a frontend-convention bug: they feed identical features to both sides, so
an error in how audio becomes features cancels on both.  The ``audio_scale``
defect is the sharpest example — it shipped, survived a full tensor-parity
suite, and was caught only when somebody eyeballed a transcript against ground
truth.  An empty transcript is 100% WER and a dropped leading token is a
visible delta; both are trivial for this module and invisible to parity.

Two properties are worth stating because getting either wrong quietly changes
the number:

**Corpus WER, not the mean of per-utterance WERs.**  The definition is
``total_edits / total_reference_words`` over the whole set.  Averaging
per-utterance rates weights a three-word utterance the same as a thirty-word
one and is not comparable to any published figure.  :func:`compute` returns the
corpus rate; per-utterance rates come along for debugging, not for averaging.

**Normalization is explicit and never silently degrades.**  ``english`` is
Whisper's ``EnglishTextNormalizer`` (numbers, contractions, British/American
spelling), which is what published WER tables use.  It lives in
``transformers``; if that is not installed, asking for it *raises* rather than
falling back to something weaker, because a silent fallback would move every
number in `ci/wer-reference.json` without failing anything.

    from oasr.testing.wer import compute, normalizer

    r = compute(refs, hyps, normalizer=normalizer("english"))
    print(r.rate, r.substitutions, r.deletions, r.insertions)
    for line in r.worst(5):
        print(line)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

__all__ = ["ErrorCounts", "Result", "Utterance", "compute", "normalizer"]

#: A text -> text callable applied to both reference and hypothesis.
Normalizer = Callable[[str], str]


def normalizer(kind: str = "english") -> Normalizer:
    """Return a text normalizer by name.

    ``english``
        Whisper's ``EnglishTextNormalizer`` — the one published WER tables use.
        Requires ``transformers`` (``pip install oasr[tokenizers]`` also pulls
        it in via the model extras).
    ``basic``
        Whisper's ``BasicTextNormalizer``: case-folding, punctuation removal,
        whitespace collapse.  Correct for CJK, where the English rules
        (number words, contractions) are meaningless.
    ``none``
        Identity.  Only for callers that have already normalized.
    """
    kind = kind.lower()
    if kind == "none":
        return lambda s: s
    try:
        from transformers.models.whisper.english_normalizer import (
            BasicTextNormalizer,
            EnglishTextNormalizer,
        )
    except ImportError as exc:  # pragma: no cover - exercised by the error path test
        raise ImportError(
            f"normalizer({kind!r}) needs `transformers` for Whisper's text normalizer. "
            "Install it, or pass normalizer='none' and normalize upstream — but note "
            "that changing normalization changes every WER number, including the "
            "recorded references in ci/wer-reference.json."
        ) from exc
    # `transformers` ships no stubs for these, so they arrive as `Any`.  Binding
    # through a typed local turns that into an explicit assertion at the
    # third-party boundary — "we require a str -> str callable" — which is worth
    # more than a `# type: ignore[no-any-return]` saying only "be quiet".
    norm: Normalizer
    if kind == "english":
        # The dict is Whisper's extra spelling map; empty means "no extra rules",
        # which is what the reference implementation defaults to.
        norm = EnglishTextNormalizer({})
    elif kind == "basic":
        norm = BasicTextNormalizer()
    else:
        raise ValueError(f"unknown normalizer {kind!r}; expected english / basic / none")
    return norm


@dataclass(frozen=True)
class ErrorCounts:
    """Edit operations between one reference and one hypothesis."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def ref_len(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def rate(self) -> float:
        """Per-utterance error rate; ``0.0`` for an empty reference with no output."""
        if self.ref_len == 0:
            return 0.0 if self.insertions == 0 else float("inf")
        return self.errors / self.ref_len

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.hits + other.hits,
        )


def _align(ref: Sequence[str], hyp: Sequence[str]) -> ErrorCounts:
    """Levenshtein alignment counts, unit cost for substitute/insert/delete.

    Two rolling rows rather than the full ``(n+1) x (m+1)`` table: a long-form
    transcript can be thousands of tokens, and only the counts are wanted.  Each
    cell carries its own running ``ErrorCounts`` so the operation breakdown
    comes out of the same pass as the distance.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return ErrorCounts(insertions=m)
    if m == 0:
        return ErrorCounts(deletions=n)

    # prev[j] = best counts for ref[:0] vs hyp[:j]  ->  j insertions
    prev: List[ErrorCounts] = [ErrorCounts(insertions=j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur: List[ErrorCounts] = [ErrorCounts(deletions=i)]
        r = ref[i - 1]
        for j in range(1, m + 1):
            if r == hyp[j - 1]:
                cur.append(prev[j - 1] + ErrorCounts(hits=1))
                continue
            sub = prev[j - 1] + ErrorCounts(substitutions=1)
            dele = prev[j] + ErrorCounts(deletions=1)
            ins = cur[j - 1] + ErrorCounts(insertions=1)
            cur.append(min((sub, dele, ins), key=lambda c: c.errors))
        prev = cur
    return prev[m]


@dataclass
class Utterance:
    """One reference/hypothesis pair and its counts."""

    uid: str
    reference: str
    hypothesis: str
    counts: ErrorCounts

    @property
    def rate(self) -> float:
        return self.counts.rate


@dataclass
class Result:
    """Corpus-level error rate plus the per-utterance detail."""

    unit: str  # "word" or "char"
    counts: ErrorCounts = field(default_factory=ErrorCounts)
    utterances: List[Utterance] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """Corpus rate: total edits / total reference units.

        Not the mean of :attr:`Utterance.rate` — see the module docstring.
        """
        return self.counts.rate

    @property
    def percent(self) -> float:
        return 100.0 * self.rate

    @property
    def substitutions(self) -> int:
        return self.counts.substitutions

    @property
    def deletions(self) -> int:
        return self.counts.deletions

    @property
    def insertions(self) -> int:
        return self.counts.insertions

    def summary(self) -> str:
        c = self.counts
        label = "WER" if self.unit == "word" else "CER"
        return (
            f"{label} {self.percent:.2f}%  "
            f"({c.errors}/{c.ref_len} = {c.substitutions}S {c.deletions}D {c.insertions}I, "
            f"{len(self.utterances)} utt)"
        )

    def worst(self, n: int = 5) -> List[str]:
        """The *n* worst utterances, rendered ref/hyp for eyeballing.

        This is the part that turns "WER rose 0.4 points" into a diagnosis; a
        regression gate that only prints a number sends you back to the shell.
        """
        ranked = sorted(
            (u for u in self.utterances if u.counts.ref_len),
            key=lambda u: (-u.rate, u.uid),
        )[:n]
        out = []
        for u in ranked:
            c = u.counts
            out.append(
                f"[{u.uid}] {100 * u.rate:.1f}% "
                f"({c.substitutions}S {c.deletions}D {c.insertions}I)\n"
                f"    ref: {u.reference}\n"
                f"    hyp: {u.hypothesis}"
            )
        return out


def _tokens(text: str, unit: str) -> List[str]:
    if unit == "word":
        return text.split()
    # Character units: whitespace is not a symbol, matching the usual CER
    # convention for CJK where the reference has no word boundaries anyway.
    return [ch for ch in text if not ch.isspace()]


def compute(
    references: Iterable[str],
    hypotheses: Iterable[str],
    *,
    unit: str = "word",
    normalizer: Optional[Normalizer] = None,  # noqa: A002  — mirrors the factory's name
    uids: Optional[Sequence[str]] = None,
) -> Result:
    """Corpus error rate over paired references and hypotheses.

    Parameters
    ----------
    unit
        ``"word"`` for WER, ``"char"`` for CER.
    normalizer
        Applied to both sides before tokenizing.  ``None`` means no
        normalization, which is almost never what you want for a comparable
        number — see :func:`normalizer`.
    uids
        Optional per-utterance ids, used only in :meth:`Result.worst` output.
    """
    if unit not in ("word", "char"):
        raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")
    refs, hyps = list(references), list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} reference(s) but {len(hyps)} hypothesis(es)")
    if uids is not None and len(uids) != len(refs):
        raise ValueError(f"{len(uids)} uid(s) for {len(refs)} utterance(s)")

    norm = normalizer or (lambda s: s)
    result = Result(unit=unit)
    total = ErrorCounts()
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        nref, nhyp = norm(ref or ""), norm(hyp or "")
        counts = _align(_tokens(nref, unit), _tokens(nhyp, unit))
        total = total + counts
        result.utterances.append(
            Utterance(
                uid=uids[i] if uids is not None else str(i),
                reference=nref,
                hypothesis=nhyp,
                counts=counts,
            )
        )
    result.counts = total
    return result
