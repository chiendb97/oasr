# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert family-specific token alignments into shared word timings.

Strategies report encoder-frame spans; :class:`FrameClock` converts them to
time. Words are sliced from rendered text so they remain literal substrings.
Grouping runs only in the compiled extension to keep per-token Python off the
decode path; tests contain the independent Python oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from oasr.features.config import FeatureConfig

    from .detokenize import Detokenizer

__all__ = [
    "TokenAlignment",
    "WordTiming",
    "FrameClock",
    "AlignmentFields",
    "alignment_fields",
    "emission_fields",
    "wants_word_timings",
    "word_timings",
]


#: Optional at import time so CPU-only environments can import the package.
try:
    from oasr import _C  # type: ignore[attr-defined]

    _CPP = _C.alignment
except (ImportError, AttributeError):  # pragma: no cover - no extension built
    _CPP = None


class AlignmentFields(NamedTuple):
    """What one alignment pass produces, in the order it is written out.

    ``None`` throughout rather than empty containers: an unresolvable clock, an
    empty hypothesis and "words were not asked for" all have to be
    distinguishable from a genuinely silent utterance.

    ``confidence`` is the **mean** per-token posterior, not the product: a joint
    sequence probability decays geometrically with length, so it ranks a long
    correct transcript below a short uncertain one and is unusable as the
    ``[0, 1]`` score every serving API asks for.  The same aggregation is
    applied per word.  The C++ side accumulates it in a plain loop rather than
    with any compensated summation, because CPython grew Neumaier compensation
    for ``sum()`` in 3.12 and the published value must not depend on the
    interpreter the oracle happens to run under.
    """

    words: Optional[List["WordTiming"]]
    timestamps: Optional[List[Tuple[float, float]]]
    confidence: Optional[float]


def _words(raw: Sequence[Tuple[str, float, float, float]]) -> List["WordTiming"]:
    """C++ 4-tuples → ``WordTiming``.

    ``_make`` rather than the constructor, and ``map`` rather than a
    comprehension, so the only Python-level work per word is the ``NamedTuple``
    allocation itself.  Returning the C++ struct as a bound class instead would
    save that, at the cost of a type whose ``_replace`` (the long-form merge)
    and tuple-unpacking would have to be reimplemented.
    """
    return list(map(WordTiming._make, raw))


def wants_word_timings(request: object) -> bool:
    """Whether this request asked for word timings.

    Reads through ``request.decoding`` so a strategy never has to know that
    ``decoding`` is optional, and so "no options" and "options that did not ask"
    are the same answer.
    """
    opts = getattr(request, "decoding", None)
    return bool(opts is not None and getattr(opts, "word_timestamps", False))


class TokenAlignment(NamedTuple):
    """One decoded token's acoustic span, in **encoder frames**.

    A ``NamedTuple`` rather than a frozen dataclass for the same immutability
    at a third of the cost: one of these exists per decoded *token*, built and
    then read four times by the word grouping, all on the decode path — and a
    frozen dataclass pays ``object.__setattr__`` per field on construction.

    Attributes
    ----------
    token : int
        The token id, as it appears in ``RequestOutput.tokens[0]``.
    start_frame : float
        Index of the first encoder frame the token covers.
    end_frame : float
        Index one past the last frame it covers, so ``end - start`` is a
        duration and a token occupying frame ``t`` alone reports ``(t, t + 1)``.
        Fractional values are allowed — Paraformer's CIF boundaries are not
        integers.
    confidence : float
        Posterior probability in ``[0, 1]`` for this token under the family's
        own model.  ``1.0`` is the honest value only for a family that cannot
        compute one; prefer reporting what the decoder actually held.
    """

    token: int
    start_frame: float
    end_frame: float
    confidence: float = 1.0


class WordTiming(NamedTuple):
    """One word of the transcript with its span in seconds and its confidence.

    ``word`` is a substring of ``RequestOutput.text`` — see the module
    docstring for why that is a contract rather than an implementation detail.

    Also a ``NamedTuple``: it is read by attribute everywhere it crosses a
    boundary (the PyO3 marshaller, the Python client, the long-form merge), so
    the change is invisible there, and it unpacks for a caller that wants
    ``for word, start, end, conf in output.words``.
    """

    word: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class FrameClock:
    """Encoder-frame index → seconds of audio.

    One encoder frame spans ``frame_shift_ms × lfr_n × subsampling_rate``
    seconds: the feature hop, times the low-frame-rate decimation the frontend
    applies (FunASR stacks 7 frames and advances 6), times the encoder's own
    temporal subsampling.  Every architecture declares all three, so this holds
    for all of them: Conformer 10 ms × 1 × 4 = 40 ms, Whisper 10 × 1 × 2 = 20 ms,
    Paraformer 10 × 6 × 1 = 60 ms, Nemotron 10 × 1 × 8 = 80 ms.
    """

    seconds_per_frame: float

    def seconds(self, frame: float) -> float:
        return float(frame) * self.seconds_per_frame

    def span(self, start_frame: float, end_frame: float) -> Tuple[float, float]:
        return self.seconds(start_frame), self.seconds(end_frame)

    @classmethod
    def resolve(
        cls, feature_config: Optional["FeatureConfig"], model: object
    ) -> Optional["FrameClock"]:
        """Build the clock for a running engine, or ``None`` if it cannot be.

        ``None`` is returned rather than a guessed rate: a wrong seconds-per-frame
        produces timings that look entirely plausible and are uniformly wrong by
        a constant factor, which is the hardest kind of error to notice.  A
        strategy holding ``None`` emits no timings at all, and
        :meth:`DecodeStrategy.validate_options` refuses the request that asked
        for them.
        """
        if feature_config is None or model is None:
            return None
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            return None
        try:
            hop = float(feature_config.frame_shift_ms) / 1000.0
            lfr = int(getattr(feature_config, "lfr_n", 1) or 1)
            rate = int(getattr(encoder, "subsampling_rate", 1) or 1)
        except (TypeError, ValueError):
            return None
        spf = hop * lfr * rate
        if spf <= 0.0:
            return None
        return cls(spf)


# ---------------------------------------------------------------------------
# Token → word grouping
# ---------------------------------------------------------------------------


def _token_pieces(tokens: Sequence[int], detok: "Detokenizer") -> List[str]:
    """Render each token to the text *it* contributed, in order.

    Delegates to :meth:`Detokenizer.token_pieces`, whose contract is that the
    pieces concatenate to exactly what ``detokenize`` returns.  A token that
    renders to nothing — a special id, or a byte-BPE fragment absorbed into its
    neighbour — yields ``""`` and simply owns no characters.
    """
    return detok.token_pieces(tokens)


def alignment_fields(
    alignments: Sequence[TokenAlignment],
    detok: "Detokenizer",
    clock: Optional[FrameClock],
    *,
    offset: float = 0.0,
    want_words: bool = True,
) -> "AlignmentFields":
    """Per-token spans → everything ``RequestOutput`` publishes about timing.

    The generic entry, for the families whose spans are not emission frames
    (Paraformer's CIF boundaries, Whisper's DTW).  A :class:`TokenAlignment` is
    a ``NamedTuple``, so the whole sequence crosses into C++ as tuples with no
    Python-level unpacking.
    """
    if clock is None or not alignments:
        return AlignmentFields(None, None, None)
    pieces = _token_pieces([a.token for a in alignments], detok) if want_words else ()
    raw, timestamps, confidence = _CPP.align_spans(
        alignments, pieces, clock.seconds_per_frame, offset, want_words
    )
    return AlignmentFields(_words(raw) if want_words else None, timestamps, confidence)


def emission_fields(
    tokens: Sequence[int],
    frames: Sequence[int],
    confidences: Optional[Sequence[float]],
    detok: "Detokenizer",
    clock: Optional[FrameClock],
    *,
    offset: float = 0.0,
    frame_offset: int = 0,
    want_words: bool = True,
) -> "AlignmentFields":
    """The fused entry for the frame-synchronous families.

    Emission frames straight to words, timestamps and the utterance confidence,
    with **no per-token Python object at any point** — which is the whole reason
    this is separate from :func:`alignment_fields`.

    Both CTC and the transducer report *when they decided*, not how long the
    sound lasted: token ``k`` was emitted having consumed encoder frames up to
    ``t_k``, with the previous decision at ``t_{k-1}``, so the token owns
    ``(t_{k-1} + 1, t_k + 1)`` and the spans tile the timeline without gaps.
    The first token is the exception — it starts at its own frame rather than at
    zero, because everything before it is emission latency plus whatever silence
    preceded the utterance, and attributing seconds of leading silence to the
    first word is worse than reporting a short one.  Several tokens emitted at
    the same frame (a transducer's ``max_sym_per_frame``) each report that
    frame, which is exactly what happened.  ``frame_offset`` rebases a streaming
    chunk's local frame onto the stream; the rule itself is
    ``oasr::alignment::emission_spans``.

    ``tokens`` and ``frames`` must be the same length — a caller that cannot
    guarantee that has a bug rather than a short list, and silently truncating
    would emit a transcript timed against the wrong frames.
    """
    if clock is None or not tokens:
        return AlignmentFields(None, None, None)
    if len(frames) != len(tokens):
        raise ValueError(
            f"emission_fields got {len(tokens)} tokens and {len(frames)} frames; "
            "they index each other, so a mismatch is a decoder bug, not a short list"
        )
    pieces = _token_pieces(tokens, detok) if want_words else ()
    raw, timestamps, confidence = _CPP.align_emissions(
        frames,
        confidences if confidences is not None else (),
        pieces,
        clock.seconds_per_frame,
        frame_offset,
        offset,
        want_words,
    )
    return AlignmentFields(_words(raw) if want_words else None, timestamps, confidence)


def word_timings(
    alignments: Sequence[TokenAlignment],
    detok: "Detokenizer",
    clock: Optional[FrameClock],
    *,
    offset: float = 0.0,
) -> List[WordTiming]:
    """Group per-token alignments into per-word timings.

    ``offset`` shifts every span, for a strategy decoding a window of a longer
    recording (the long-form fan-out).  Returns ``[]`` when there is nothing to
    align or no clock to convert with — never a partially-timed list, because a
    consumer cannot tell a genuinely silent word from a missing one.
    """
    return alignment_fields(alignments, detok, clock, offset=offset).words or []
