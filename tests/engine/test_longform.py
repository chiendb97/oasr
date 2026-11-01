# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Stitching long-form windows back into one output (``oasr.engine.longform``).

Two windows that share audio both transcribe it, so the later one's head
repeats the earlier one's tail.  The merge used to drop that repeat from the
text alone: the words, tokens and per-token timestamps kept it, so a 79 s
Whisper request came back with 198 words of text and 201 word timings, three
of them a second copy of "the 15th century".  It also dropped repeats where
the windows share nothing (a VAD cut in silence), and lost ``segments``,
``no_speech_prob`` and ``error_stage`` altogether.

The windows here are built the way the real alignment pass builds them: from
token pieces, so ``text == "".join(pieces)`` and every word is a literal
substring of the text.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from oasr.engine.decode.alignment import WordTiming
from oasr.engine.longform import LongFormTracker, merge_texts
from oasr.engine.request import RequestOutput
from oasr.vad.segmenter import SpeechSegment

_PIECES: Dict[int, str] = {}


def _id(piece: str) -> int:
    for k, v in _PIECES.items():
        if v == piece:
            return k
    _PIECES[len(_PIECES) + 1000] = piece
    return len(_PIECES) + 999


def _pieces(ids: Sequence[int]) -> List[str]:
    return [_PIECES[i] for i in ids]


Token = Tuple[str, float, float]


def _tokens(text: str, t0: float = 0.0, step: float = 1.0) -> List[Token]:
    """One token per word of ``text``, ``step`` seconds each from ``t0``."""
    return [
        (w if k == 0 else " " + w, t0 + k * step, t0 + (k + 1) * step)
        for k, w in enumerate(text.split())
    ]


def _window(rid: str, toks: Sequence[Token], **fields) -> RequestOutput:
    """A finished window whose text, words and timestamps come from ``toks``.

    A piece that starts with a space starts a word, and an empty piece (a
    special id) belongs to none -- the grouping ``oasr._C.alignment`` does.
    """
    words: List[WordTiming] = []
    current: Optional[List] = None
    for piece, start, end in toks:
        if not piece:
            continue
        if current is None or piece[0].isspace():
            if current is not None:
                words.append(WordTiming(current[0], current[1], current[2], 0.9))
            current = [piece.strip(), start, end]
        else:
            current[0] += piece
            current[2] = end
    if current is not None:
        words.append(WordTiming(current[0], current[1], current[2], 0.9))
    fields.setdefault("words", words)
    fields.setdefault("timestamps", [(s, e) for _, s, e in toks])
    return RequestOutput(
        request_id=rid,
        text="".join(p for p, _, _ in toks),
        tokens=[[_id(p) for p, _, _ in toks]],
        finished=True,
        **fields,
    )


def _stitch(windows, spans, pieces=_pieces) -> RequestOutput:
    tracker = LongFormTracker(pieces)
    tracker.register(
        "file", [w.request_id for w in windows], [s for s, _ in spans], [e for _, e in spans]
    )
    (out,) = tracker.absorb(list(windows))
    assert out.request_id == "file"
    return out


def _assert_words_are_the_text(out: RequestOutput) -> None:
    """Every word a literal substring of the text, in order, none left over."""
    at = 0
    for w in out.words or ():
        found = out.text.find(w.word, at)
        assert found >= 0, (w.word, out.text[at:])
        at = found + len(w.word)
    assert [w.word for w in out.words or ()] == out.text.split()


class TestAnOverlapIsDroppedFromEveryField:
    """Windows ``[0, 30)`` and ``[29, 59)``: both heard 29-30 s."""

    SPANS = [(0.0, 30.0), (29.0, 59.0)]

    def test_words_tokens_and_timestamps_lose_the_repeat_with_the_text(self):
        first = _window("w0", _tokens("a b the 15th century", t0=25.0))
        second = _window("w1", _tokens("the 15th century may be"))
        out = _stitch([first, second], self.SPANS)

        assert out.text == "a b the 15th century may be"
        _assert_words_are_the_text(out)
        assert [(w.word, w.start) for w in out.words[-3:]] == [
            ("century", 29.0),
            ("may", 32.0),
            ("be", 33.0),
        ]
        assert out.tokens == [first.tokens[0] + second.tokens[0][3:]]
        assert out.timestamps == first.timestamps + [(32.0, 33.0), (33.0, 34.0)]

    def test_a_word_of_several_tokens_drops_all_of_them(self):
        """Whisper splits "15th" into " 15" + "th"; a special id renders nothing."""
        second = _window(
            "w1",
            [
                ("", 0.0, 0.0),
                ("the", 0.0, 0.7),
                (" 15", 0.7, 1.1),
                ("th", 1.1, 1.4),
                (" century", 1.4, 1.9),
                (" may", 1.9, 2.2),
            ],
        )
        out = _stitch([_window("w0", _tokens("of the 15th century", t0=26.0)), second], self.SPANS)

        assert out.text == "of the 15th century may"
        _assert_words_are_the_text(out)
        assert out.tokens[0][-1:] == [_id(" may")]
        assert out.timestamps[-1:] == [(29.0 + 1.9, 29.0 + 2.2)]
        assert len(out.timestamps) == len(out.tokens[0]) == 5

    def test_tokens_stay_whole_when_they_cannot_be_mapped_onto_the_text(self, caplog):
        """No ``token_pieces``: the text and words still merge, and it is said once."""
        windows = [
            _window("w0", _tokens("a b c", t0=27.0)),
            _window("w1", _tokens("c d")),
        ]
        tracker = LongFormTracker()
        with caplog.at_level(logging.WARNING, logger="oasr.engine.longform"):
            for parent in ("f1", "f2"):
                ids = [f"{parent}{w.request_id}" for w in windows]
                tracker.register(parent, ids, [0.0, 29.0], [30.0, 59.0])
                renamed = [
                    RequestOutput(**{**vars(w), "request_id": i}) for w, i in zip(windows, ids)
                ]
                (out,) = tracker.absorb(renamed)
                assert out.text == "a b c d"
                assert [w.word for w in out.words] == ["a", "b", "c", "d"]
                assert len(out.tokens[0]) == 5
        assert sum("do not render" in r.getMessage() for r in caplog.records) == 1

    def test_a_whole_window_repeated_still_dedups_the_one_after(self):
        """The repeat is matched against the previous window's own transcript,
        not what survived of it."""
        spans = [(0.0, 30.0), (29.0, 30.5), (30.0, 60.0)]
        windows = [_window(f"w{i}", _tokens(t)) for i, t in enumerate(["a b c", "c", "c d"])]
        assert _stitch(windows, spans).text == "a b c d"


class TestASeamWithoutOverlapKeepsEveryWord:
    def test_a_phrase_said_again_after_a_pause_is_kept(self):
        """A VAD cut in silence: nothing is shared, so nothing repeats."""
        windows = [
            _window("w0", _tokens("thank you")),
            _window("w1", _tokens("thank you very much")),
        ]
        out = _stitch(windows, [(0.0, 5.0), (8.0, 12.0)])
        assert out.text == "thank you thank you very much"
        _assert_words_are_the_text(out)
        assert len(out.tokens[0]) == len(out.timestamps) == 6

    def test_windows_that_only_touch_share_nothing(self):
        windows = [_window("w0", _tokens("a b")), _window("w1", _tokens("b c"))]
        assert _stitch(windows, [(0.0, 30.0), (30.0, 60.0)]).text == "a b b c"

    def test_an_empty_window_shares_nothing_with_the_next(self):
        """Windows overlap by at most half their width, so the windows on either
        side of an empty one share no audio, and its silence is not a seam."""
        spans = [(0.0, 30.0), (29.0, 59.0), (58.0, 88.0)]
        windows = [_window(f"w{i}", _tokens(t)) for i, t in enumerate(["a b", "", "b c"])]
        assert _stitch(windows, spans).text == "a b b c"
        assert merge_texts(["a b", "", "b c"]) == "a b b c"


class TestFieldsTheMergeUsedToDrop:
    def test_segments_are_cut_at_the_middle_of_the_shared_audio(self):
        """Each stretch of shared audio is reported once, and a span the cut
        divides is one span again."""
        first = _window(
            "w0",
            _tokens("a"),
            segments=[SpeechSegment(2.0, 4.0, 0.9), SpeechSegment(25.0, 30.0, 0.8)],
        )
        second = _window(
            "w1",
            _tokens("b"),
            segments=[SpeechSegment(0.0, 0.3, 0.5), SpeechSegment(0.4, 3.0, 0.6)],
        )
        out = _stitch([first, second], [(0.0, 30.0), (29.0, 59.0)])
        assert [(s.start, s.end) for s in out.segments] == [(2.0, 4.0), (25.0, 32.0)]
        assert out.segments[1].speech_prob == pytest.approx((0.8 * 4.5 + 0.6 * 2.5) / 7.0)

    def test_segments_across_a_gap_are_shifted_into_file_time(self):
        windows = [
            _window("w0", _tokens("a"), segments=[SpeechSegment(0.1, 2.0, 0.9)]),
            _window("w1", _tokens("b"), segments=[SpeechSegment(0.0, 1.5, 0.7)]),
        ]
        out = _stitch(windows, [(1.0, 3.0), (6.0, 7.5)])
        assert [(s.start, s.end, s.speech_prob) for s in out.segments] == [
            pytest.approx((1.1, 3.0, 0.9)),
            pytest.approx((6.0, 7.5, 0.7)),
        ]

    def test_no_detector_is_none_and_no_speech_found_is_empty(self):
        spans = [(0.0, 5.0), (8.0, 12.0)]
        plain = [_window("w0", _tokens("a")), _window("w1", _tokens("b"))]
        assert _stitch(plain, spans).segments is None
        silent = [
            _window("w0", _tokens("a"), segments=[]),
            _window("w1", _tokens("b"), segments=[]),
        ]
        assert _stitch(silent, spans).segments == []

    def test_no_speech_prob_is_the_least_of_the_windows(self):
        spans = [(0.0, 30.0), (29.0, 59.0), (58.0, 70.0)]
        windows = [
            _window("w0", _tokens("a"), no_speech_prob=0.9),
            _window("w1", _tokens("b"), no_speech_prob=0.2),
            _window("w2", _tokens("c")),
        ]
        assert _stitch(windows, spans).no_speech_prob == pytest.approx(0.2)
        plain = [_window("w0", _tokens("a")), _window("w1", _tokens("b"))]
        assert _stitch(plain, spans[:2]).no_speech_prob is None

    def test_the_failed_window_says_where_it_failed(self):
        windows = [
            _window("w0", _tokens("a"), finish_reason="stop"),
            _window("w1", _tokens("b"), finish_reason="error", error_stage="decode_oom"),
        ]
        out = _stitch(windows, [(0.0, 30.0), (29.0, 59.0)])
        assert (out.finish_reason, out.error_stage) == ("error", "decode_oom")

    def test_confidence_is_a_mean_over_tokens(self):
        """A one-word VAD segment does not weigh as much as a long one."""
        windows = [
            _window("w0", _tokens("a b c d e f g h i"), confidence=0.9),
            _window("w1", _tokens("j"), confidence=0.5),
        ]
        out = _stitch(windows, [(0.0, 9.0), (12.0, 13.0)])
        assert out.confidence == pytest.approx((0.9 * 9 + 0.5) / 10)
