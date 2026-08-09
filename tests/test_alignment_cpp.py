# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The C++ alignment pass, checked against a Python statement of the same rule.

``csrc/alignment/`` and ``csrc/tokenizers/`` are the *only* implementations the
engine has: the token→word grouping, the emission tiling and the symbol-table
rendering run in C++ or not at all.  A rule that exists once still has to be
checkable, so the Python version lives **here**, as an oracle — independently
written from the C++, exercised only by this file, and impossible for a
deployment to land on.

Three classes of divergence are possible and each is pinned below:

1. **Classification.**  Whitespace is Python's ``str.isspace()`` — 29 code
   points, not ASCII and not ``std::isspace`` — and "space-less script" is six
   Unicode ranges.  Both are checked against CPython over the *whole plane*,
   because a single wrong code point silently merges or splits one word in one
   language.
2. **Grouping and arithmetic.**  Randomised piece sets through both, compared
   exactly: same words, same spans, same confidences, same timestamps.  Exact
   rather than approximate — both do the same operations in the same order, so
   a difference is a bug and not a rounding artefact.  (This is what caught the
   oracle using ``sum()``, which CPython 3.12 compensates and 3.10 does not.)
3. **UTF-8.**  The C++ side works in bytes and the oracle in code points.  They
   partition identically only because a piece is a whole number of code points;
   the CJK/emoji/combining cases here are what would catch it if that stopped
   being true.

Changing the rule in C++ therefore means changing the oracle below in the same
commit — that is the cost of having the check at all, and it is much cheaper
than the alternative these tests exist to prevent.
"""

from __future__ import annotations

import random
import re
from bisect import bisect_right

import pytest
import torch

from oasr.engine.decode.alignment import (
    _CPP as _ALIGNMENT,
    AlignmentFields,
    FrameClock,
    TokenAlignment,
    WordTiming,
)

pytestmark = pytest.mark.skipif(
    _ALIGNMENT is None,
    reason="oasr._C.alignment is not built (test-cpu.yml builds no extension)",
)


@pytest.fixture(scope="module")
def cpp():
    from oasr import _C

    return _C.alignment


# ---------------------------------------------------------------------------
# The oracle: the same rule as csrc/alignment/word_timings.cc, in Python
# ---------------------------------------------------------------------------

# Restated from ``kSpacelessRanges`` rather than shared with it, so a typo on
# one side is not copied into the check for the other.
_SPACELESS_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F),  # CJK Extensions B-F + Compatibility Supplement
)

#: One word is either a single space-less character or a maximal run of ordinary
#: non-space characters.  Whitespace matches neither alternative, so ``finditer``
#: walks past it and every match is a word — "你好world" yields 你 / 好 / world.
_SPACELESS_CLASS = "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _SPACELESS_RANGES)
_WORD_RE = re.compile(f"[{_SPACELESS_CLASS}]|[^\\s{_SPACELESS_CLASS}]+")


def _reference_spans(frames, confidences, frame_offset):
    """``oasr::alignment::emission_spans``: emission frames → per-token spans.

    Token ``k`` owns the frames since the previous decision,
    ``(t_{k-1} + 1, t_k + 1)``; the first starts at its own frame.
    """
    spans = []
    prev = 0
    for i, frame in enumerate(frames):
        frame = int(frame)
        start = frame if i == 0 else min(prev + 1, frame)
        conf = float(confidences[i]) if confidences is not None and i < len(confidences) else 1.0
        spans.append(
            TokenAlignment(i, float(start + frame_offset), float(frame + frame_offset + 1), conf)
        )
        prev = frame
    return spans


def _reference_fields(spans, pieces, spf, offset, want_words):
    """``oasr::alignment::align``: spans + rendered pieces → what is published."""
    if not spans:
        return AlignmentFields(None, None, None)
    timestamps = [
        (
            a.start_frame * spf + offset,
            max(a.end_frame * spf + offset, a.start_frame * spf + offset),
        )
        for a in spans
    ]
    # Explicit accumulation, never ``sum()`` — see the module docstring.
    total = 0.0
    for a in spans:
        total += a.confidence
    confidence = total / len(spans)
    if not want_words:
        return AlignmentFields(None, timestamps, confidence)

    # Character ownership as two parallel arrays: ``ends[j]`` is one past the
    # last character piece ``j`` contributed, ``owner[j]`` the token that
    # produced it.  Ownership is monotone, so a word's member tokens are two
    # bisections rather than a set built over its characters.
    text_parts, ends, owner, pos = [], [], [], 0
    for idx, piece in enumerate(pieces[: len(spans)]):
        if not piece:
            continue
        pos += len(piece)
        text_parts.append(piece)
        ends.append(pos)
        owner.append(idx)
    text = "".join(text_parts)

    words = []
    for match in _WORD_RE.finditer(text):
        a, b = match.span()
        # ``b - 1`` rather than ``b`` keeps a word ending exactly on a piece
        # boundary from claiming the next one.
        first = bisect_right(ends, a)
        last = bisect_right(ends, b - 1)
        al = spans[owner[first]]
        start, end, conf = al.start_frame, al.end_frame, al.confidence
        for j in range(first + 1, last + 1):
            al = spans[owner[j]]
            start = min(start, al.start_frame)
            end = max(end, al.end_frame)
            conf += al.confidence
        t0 = start * spf + offset
        words.append(
            WordTiming(
                word=match.group(),
                start=t0,
                end=max(end * spf + offset, t0),
                confidence=conf / (last - first + 1),
            )
        )
    return AlignmentFields(words, timestamps, confidence)


def _reference_pieces(table, special, ids):
    """``oasr::tokenizers::SymbolTablePieces::pieces``.

    A piece is a table lookup and a ``▁`` substitution; only ``decode``'s outer
    ``strip`` couples the ends, so the pieces have to lose exactly the
    characters it would remove.
    """
    pieces = ["" if int(t) in special else table.get(int(t), "").replace("▁", " ") for t in ids]
    for i, piece in enumerate(pieces):  # what ``strip`` takes off the front
        if piece:
            pieces[i] = piece.lstrip()
            if pieces[i]:
                break
    for i in range(len(pieces) - 1, -1, -1):  # ... and off the back
        if pieces[i]:
            pieces[i] = pieces[i].rstrip()
            if pieces[i]:
                break
    return pieces


class _Detok:
    """Hands back a fixed piece list — both sides must see identical rendering."""

    def __init__(self, pieces):
        self._pieces = list(pieces)

    def token_pieces(self, ids):
        return list(self._pieces)


class TestClassification:
    def test_whitespace_is_pythons_own_over_the_whole_plane(self, cpp):
        bad = [cp for cp in range(0x110000) if cpp.is_space(cp) != chr(cp).isspace()]
        assert bad == [], f"{len(bad)} code points disagree, first: {bad[:8]}"

    def test_spaceless_matches_the_declared_ranges(self, cpp):
        def want(cp):
            return any(lo <= cp <= hi for lo, hi in _SPACELESS_RANGES)

        bad = [cp for cp in range(0x110000) if cpp.is_spaceless(cp) != want(cp)]
        assert bad == [], f"{len(bad)} code points disagree, first: {bad[:8]}"

    def test_ideographic_space_is_whitespace_not_a_word(self, cpp):
        """U+3000 sits one block below the kana range and is a real trap."""
        assert cpp.is_space(0x3000) and not cpp.is_spaceless(0x3000)
        words, _, _ = cpp.align_emissions([0, 1, 2], [], ["　", "今", "天"], 0.04)
        assert [w[0] for w in words] == ["今", "天"]


# Alphabets chosen to exercise every branch of the rule: word-initial spaces
# (sentencepiece), mid-word pieces (BPE), pieces that render to nothing,
# space-less scripts adjacent to Latin, multi-byte characters of every UTF-8
# width, and pieces that are pure whitespace.
_ALPHABETS = [
    ["the", " quick", " brown", " fox", "es", ""],
    ["你", "好", "世", "界"],
    ["hel", "lo", " 你", "好", " world"],
    ["", "a", "", " b", ""],
    ["  ", "spaced", "  ", "out ", " ", "　"],
    ["안", "녕", "hello", "こん", "にちは"],
    ["caf", "é", " naïve", " речь"],
    ["\U00020000", "\U0002fa1d", " tail", "\U0001f600"],  # 4-byte CJK ext + emoji
]


def _random_case(rng):
    alpha = rng.choice(_ALPHABETS)
    n = rng.randrange(0, 14)
    pieces = [rng.choice(alpha) for _ in range(n)]
    frames, f = [], 0
    for _ in range(n):
        f += rng.choice([0, 0, 1, 2, 7])
        frames.append(f)
    kind = rng.randrange(3)
    confs = [] if kind == 0 else [rng.random() for _ in range(max(0, n - rng.randrange(0, 3)))]
    return pieces, frames, confs


def _compare(got, want, ctx):
    """``(words, timestamps, confidence)`` from both sides, exactly."""
    g_words, g_stamps, g_conf = got
    assert list(g_stamps or []) == [tuple(t) for t in (want.timestamps or [])], ctx
    assert g_conf == want.confidence, ctx
    want_words = [(w.word, w.start, w.end, w.confidence) for w in (want.words or [])]
    assert [tuple(w) for w in g_words] == want_words, ctx


class TestAgreesWithTheOracle:
    @pytest.mark.parametrize("want_words", [True, False])
    def test_emissions(self, cpp, want_words):
        rng = random.Random(11)
        for trial in range(4000):
            pieces, frames, confs = _random_case(rng)
            spf = rng.choice([0.04, 0.02, 0.06])
            offset = rng.choice([0.0, 1.25])
            frame_offset = rng.choice([0, 5])
            got = cpp.align_emissions(frames, confs, pieces, spf, frame_offset, offset, want_words)
            want = _reference_fields(
                _reference_spans(frames, confs, frame_offset), pieces, spf, offset, want_words
            )
            _compare(got, want, f"trial {trial}: {pieces} {frames} {confs}")

    def test_spans(self, cpp):
        """The generic route — Paraformer's CIF and Whisper's DTW, whose spans
        are fractional and need not tile."""
        rng = random.Random(23)
        for trial in range(4000):
            pieces, _frames, _confs = _random_case(rng)
            n = len(pieces)
            spans = []
            edge = 0.0
            for i in range(n):
                edge += rng.random() * 3
                spans.append(TokenAlignment(i, edge, edge + rng.random() * 3, rng.random()))
            spf = rng.choice([0.04, 0.06])
            offset = rng.choice([0.0, 2.5])
            got = cpp.align_spans(spans, pieces, spf, offset, True)
            want = _reference_fields(spans, pieces, spf, offset, True)
            _compare(got, want, f"trial {trial}: {pieces}")

    def test_a_word_is_always_a_substring_of_the_rendered_transcript(self, cpp):
        """The load-bearing invariant, checked on the C++ side directly."""
        rng = random.Random(31)
        for _ in range(2000):
            pieces, frames, confs = _random_case(rng)
            words, _, _ = cpp.align_emissions(frames, confs, pieces, 0.04)
            text = "".join(pieces[: len(frames)])
            pos = 0
            for word, *_rest in words:
                idx = text.find(word, pos)
                assert idx >= 0, f"{word!r} not in {text!r} at/after {pos}"
                pos = idx + len(word)

    def test_a_mismatched_piece_count_does_not_read_past_the_spans(self, cpp):
        """Fewer spans than pieces: the extra pieces own nothing, both sides."""
        pieces = ["a", " b", " c", " d"]
        got = cpp.align_emissions([0, 1], [], pieces, 0.04)
        want = _reference_fields(_reference_spans([0, 1], None, 0), pieces, 0.04, 0.0, True)
        _compare(got, want, "short spans")
        assert [w[0] for w in got[0]] == ["a", "b"]


class TestBeamReadBack:
    """``extract_beam_tokens`` replaces two tensor ops per (row, beam)."""

    @staticmethod
    def _reference(values, lengths, beams):
        out = []
        for b in range(values.size(0)):
            rows = []
            for k in range(beams if beams >= 0 else values.size(1)):
                length = max(0, min(int(lengths[b, k]), values.size(2)))
                rows.append(values[b, k, :length].tolist())
            out.append(rows)
        return out

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_matches_the_tensor_slicing_reference(self, cpp, dtype):
        rng = torch.Generator().manual_seed(5)
        for _ in range(50):
            b, k, ln = (int(torch.randint(1, 6, (1,), generator=rng)) for _ in range(3))
            values = torch.randint(0, 5000, (b, k, ln), generator=rng).to(dtype)
            lengths = torch.randint(0, ln + 1, (b, k), generator=rng).to(dtype)
            assert cpp.extract_beam_tokens(values, lengths, -1) == self._reference(
                values, lengths, -1
            )
            take = max(1, k // 2)
            assert cpp.extract_beam_tokens(values, lengths, take) == self._reference(
                values, lengths, take
            )
            for bi in range(b):
                for ki in range(k):
                    assert (
                        cpp.extract_beam_row(values, lengths, bi, ki)
                        == self._reference(values, lengths, -1)[bi][ki]
                    )

    def test_a_length_beyond_the_buffer_is_clamped_not_read(self, cpp):
        """A decoder that overran its cap must not turn into an OOB read."""
        values = torch.arange(6, dtype=torch.int32).reshape(1, 2, 3)
        lengths = torch.tensor([[99, -4]], dtype=torch.int32)
        assert cpp.extract_beam_tokens(values, lengths, -1) == [[[0, 1, 2], []]]

    def test_a_device_tensor_is_refused_rather_than_copied(self, cpp):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        values = torch.zeros(1, 1, 2, dtype=torch.int32, device="cuda")
        lengths = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
        with pytest.raises(RuntimeError, match="host"):
            cpp.extract_beam_tokens(values, lengths, -1)


class TestSymbolTablePieces:
    """The tokenizer half: C++ rendering must equal the oracle.

    ``token_pieces`` is the one tokenizer method the word grouping calls, and
    its contract — the pieces concatenate to ``decode(ids)`` — is what makes
    every emitted word a literal substring of the transcript.  Both halves of
    that are checked here.
    """

    @pytest.fixture
    def tokenizer(self, tmp_path):
        from oasr.tokenizers import SymbolTableTokenizer

        pieces = [
            "<blank>",
            "<unk>",
            "<sos/eos>",
            "▁THE",
            "RE",
            "▁QUICK",
            "▁",
            "FOX",
            "▁A",
            "▁▁",
            "你",
            "好",
            "▁\u3000",  # U+3000, ideographic space
            "X▁Y",
        ]
        table = tmp_path / "units.txt"
        table.write_text("\n".join(f"{p} {i}" for i, p in enumerate(pieces)), encoding="utf-8")
        tok = SymbolTableTokenizer(str(table))
        assert tok._pieces_cpp is not None, "the C++ renderer should be live here"
        return tok

    def test_cpp_equals_the_oracle_and_joins_to_decode(self, tokenizer):
        rng = random.Random(59)
        n_ids = 14
        for _ in range(20000):
            ids = [rng.randrange(0, n_ids) for _ in range(rng.randrange(0, 10))]
            got = tokenizer.token_pieces(ids)
            want = _reference_pieces(tokenizer._table, tokenizer._special_ids, ids)
            assert got == want, ids
            assert "".join(got) == tokenizer.decode(ids), ids

    def test_a_whitespace_only_first_piece_strips_through_to_the_next(self, tokenizer):
        """``decode`` strips the *joined* text, so an all-space leading piece
        cannot simply be left alone once it is emptied."""
        ids = [6, 9, 3]  # "▁", "▁▁", "▁THE"  ->  "" , "", "THE"
        assert tokenizer.token_pieces(ids) == ["", "", "THE"]
        assert "".join(tokenizer.token_pieces(ids)) == tokenizer.decode(ids) == "THE"


class TestNothingIsAskedOfTheExtensionUntilItIsNeeded:
    """The no-op cases are answered before the C++ is reached.

    ``pip install -e .`` always builds ``alignment``, so no entry defends
    against its absence — but a family that produces no spans, or one with no
    resolvable clock, must still return cleanly rather than calling into it
    with nothing.
    """

    @pytest.fixture
    def no_cpp(self, monkeypatch):
        """Absence stands in for "the call must not happen": with ``_CPP`` set
        to ``None``, reaching the extension is an ``AttributeError``."""
        import oasr.engine.decode.alignment as alignment

        monkeypatch.setattr(alignment, "_CPP", None)

    def test_an_empty_hypothesis_and_a_missing_clock_short_circuit(self, no_cpp):
        from oasr.engine.decode.alignment import alignment_fields, emission_fields

        detok = _Detok([])
        assert emission_fields([], [], None, detok, FrameClock(0.04)) == (None, None, None)
        assert alignment_fields([TokenAlignment(0, 0.0, 1.0, 1.0)], detok, None) == (
            None,
            None,
            None,
        )

    def test_a_token_frame_mismatch_is_refused_before_the_call(self, no_cpp):
        """A decoder bug, not a short list — silently truncating would time the
        transcript against the wrong frames."""
        from oasr.engine.decode.alignment import emission_fields

        with pytest.raises(ValueError, match="decoder bug"):
            emission_fields([0, 1, 2], [0, 1], None, _Detok([]), FrameClock(0.04))


class TestEndToEndParity:
    """The public Python entry must agree with the oracle end to end."""

    def test_word_timings_matches_the_oracle(self):
        from oasr.engine.decode.alignment import word_timings

        rng = random.Random(47)
        for _ in range(2000):
            pieces, _f, _c = _random_case(rng)
            spans = [
                TokenAlignment(i, float(i), float(i) + 1.5, 0.5 + (i % 3) / 10)
                for i in range(len(pieces))
            ]
            detok = _Detok(pieces)
            got = word_timings(spans, detok, FrameClock(0.04), offset=0.5)
            want = _reference_fields(spans, pieces, 0.04, 0.5, True).words or []
            assert got == want
