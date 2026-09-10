# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The policy half of voice activity: hysteresis and endpointing.

Deliberately free of torch, of any model, and of any external asset.  Splitting
the detector (which varies per model) from the segmenter and endpointer (which
do not) is what makes this file possible: every rule below is exercised against a
hand-written probability trace, so the half of the axis most likely to carry an
off-by-one is tested without a GPU or a checkpoint.

Every trace is written at 10 ms per frame, so a frame index reads directly as
centiseconds.
"""

from __future__ import annotations

import pytest

from oasr.vad import (
    SPEECH_STARTED,
    SPEECH_STOPPED,
    Endpointer,
    EndpointRule,
    SpeechSegmenter,
    VadConfig,
)

SPF = 0.01


def cfg(**kw) -> VadConfig:
    """A resolved config with the hysteresis knobs pinned, not preset-derived."""
    base = {
        "backend": "energy",
        "mode": "observe",
        "threshold": 0.5,
        "neg_threshold": 0.35,
        "min_speech_ms": 0,
        "min_silence_ms": 100,
        "speech_pad_ms": 0,
        "max_speech_s": None,
    }
    base.update(kw)
    return VadConfig(**base).resolve("offline")


def trace(*runs) -> list:
    """``(value, frames)`` pairs → a flat probability trace."""
    out = []
    for value, count in runs:
        out.extend([value] * count)
    return out


def spans(segments):
    return [(round(s.start, 3), round(s.end, 3)) for s in segments]


class TestHysteresis:
    def test_enters_at_threshold_and_leaves_at_neg_threshold(self):
        # 0.4 is above neg_threshold but below threshold: it must not start a
        # run, and once running it must not end one.  That gap is the whole
        # point of having two thresholds.
        s = SpeechSegmenter(cfg(), SPF)
        got = s.run(trace((0.4, 50), (0.9, 50), (0.4, 50), (0.1, 50)))
        assert spans(got) == [(0.5, 1.5)]

    def test_a_short_dip_does_not_split_a_run(self):
        s = SpeechSegmenter(cfg(min_silence_ms=100), SPF)
        got = s.run(trace((0.9, 50), (0.0, 5), (0.9, 50)))
        assert spans(got) == [(0.0, 1.05)]

    def test_a_long_dip_does_split_a_run(self):
        s = SpeechSegmenter(cfg(min_silence_ms=100), SPF)
        got = s.run(trace((0.9, 50), (0.0, 20), (0.9, 50)))
        assert spans(got) == [(0.0, 0.5), (0.7, 1.2)]

    def test_the_boundary_is_the_first_silent_frame_not_the_last(self):
        """A closed run ends where the silence *began*, so the reported span
        does not include the silence that proved it ended."""
        s = SpeechSegmenter(cfg(min_silence_ms=100), SPF)
        got = s.run(trace((0.9, 30), (0.0, 30)))
        assert spans(got) == [(0.0, 0.3)]


class TestDurationFilters:
    def test_a_blip_shorter_than_min_speech_is_dropped(self):
        s = SpeechSegmenter(cfg(min_speech_ms=250), SPF)
        assert s.run(trace((0.0, 20), (0.9, 10), (0.0, 40))) == []

    def test_a_run_at_exactly_min_speech_survives(self):
        s = SpeechSegmenter(cfg(min_speech_ms=250), SPF)
        got = s.run(trace((0.0, 20), (0.9, 25), (0.0, 40)))
        assert spans(got) == [(0.2, 0.45)]

    def test_max_speech_cuts_a_run_that_never_pauses(self):
        s = SpeechSegmenter(cfg(max_speech_s=0.3), SPF)
        got = s.run(trace((0.9, 100)))
        assert len(got) > 1
        assert all(seg.duration <= 0.3 + 1e-9 for seg in got)


class TestPadding:
    def test_padding_extends_both_edges(self):
        s = SpeechSegmenter(cfg(speech_pad_ms=100), SPF)
        got = s.run(trace((0.0, 50), (0.9, 50), (0.0, 50)), total_seconds=1.5)
        assert spans(got) == [(0.4, 1.1)]

    def test_padding_never_goes_below_zero(self):
        s = SpeechSegmenter(cfg(speech_pad_ms=500), SPF)
        got = s.run(trace((0.9, 30), (0.0, 30)), total_seconds=0.6)
        assert got[0].start == 0.0

    def test_padding_is_clamped_to_the_audio(self):
        s = SpeechSegmenter(cfg(speech_pad_ms=500), SPF)
        got = s.run(trace((0.0, 10), (0.9, 30)), total_seconds=0.4)
        assert got[0].end == pytest.approx(0.4)

    def test_neighbours_meet_in_the_middle_instead_of_overlapping(self):
        """Two overlapping segments handed to the offline fan-out would
        transcribe the same audio twice and duplicate words at the seam."""
        s = SpeechSegmenter(cfg(min_silence_ms=100, speech_pad_ms=300), SPF)
        got = s.run(trace((0.9, 30), (0.0, 20), (0.9, 30)), total_seconds=0.8)
        assert len(got) == 2
        assert got[0].end <= got[1].start
        # The gap runs 0.30 -> 0.50, so the meeting point is its midpoint.
        assert got[0].end == pytest.approx(0.4)
        assert got[1].start == pytest.approx(0.4)


class TestEvents:
    def test_start_and_stop_bracket_the_run(self):
        s = SpeechSegmenter(cfg(min_silence_ms=100), SPF)
        events = s.push(trace((0.0, 20), (0.9, 40), (0.0, 30)))
        assert [(e.kind, round(e.time, 2)) for e in events] == [
            (SPEECH_STARTED, 0.2),
            (SPEECH_STOPPED, 0.6),
        ]

    def test_the_start_event_is_stamped_at_the_onset_not_at_its_confirmation(self):
        """``min_speech_ms`` delays the event's *arrival*, never its time.

        Silero's own VADIterator has no such filter and fires on the first frame
        over threshold, so a blip produces an event that cannot be retracted.
        Waiting costs latency; misreporting the time would cost correctness.
        """
        s = SpeechSegmenter(cfg(min_speech_ms=200), SPF)
        events = s.push(trace((0.0, 10), (0.9, 40)))
        assert len(events) == 1
        assert events[0].kind == SPEECH_STARTED
        assert events[0].time == pytest.approx(0.1)

    def test_a_dropped_blip_produces_no_events_at_all(self):
        s = SpeechSegmenter(cfg(min_speech_ms=250), SPF)
        events = s.push(trace((0.0, 10), (0.9, 5), (0.0, 40)))
        assert events == []

    def test_flush_closes_a_run_still_open_at_end_of_audio(self):
        s = SpeechSegmenter(cfg(), SPF)
        assert [e.kind for e in s.push(trace((0.9, 40)))] == [SPEECH_STARTED]
        assert [e.kind for e in s.flush()] == [SPEECH_STOPPED]
        assert spans(s.segments()) == [(0.0, 0.4)]

    def test_a_stream_never_ends_with_a_dangling_start(self):
        """A client whose turn opened and never closed waits forever."""
        s = SpeechSegmenter(cfg(), SPF)
        kinds = [e.kind for e in s.push(trace((0.0, 10), (0.9, 20)))]
        kinds += [e.kind for e in s.flush()]
        assert kinds.count(SPEECH_STARTED) == kinds.count(SPEECH_STOPPED)


class TestStreamingMatchesOffline:
    """One machine drives both flows, so the two must not disagree."""

    @pytest.mark.parametrize("chunk", [1, 3, 7, 16, 64], ids=lambda n: f"chunk{n}")
    def test_pushing_in_chunks_gives_the_same_segments(self, chunk):
        probs = trace((0.0, 23), (0.9, 41), (0.2, 17), (0.8, 29), (0.0, 33))
        want = SpeechSegmenter(cfg(min_silence_ms=100, speech_pad_ms=50), SPF).run(probs)
        s = SpeechSegmenter(cfg(min_silence_ms=100, speech_pad_ms=50), SPF)
        for i in range(0, len(probs), chunk):
            s.push(probs[i : i + chunk])
        s.flush()
        assert spans(s.segments()) == spans(want)


class TestReportingClock:
    def test_time_offset_rebases_every_span(self):
        """The reporting clock a streaming turn reset carries forward.

        Resetting the model's frame counter without this makes every turn after
        the first report timestamps starting at zero — plausible, and wrong.
        """
        s = SpeechSegmenter(cfg(), SPF, time_offset=12.5)
        got = s.run(trace((0.0, 10), (0.9, 20), (0.0, 30)))
        assert spans(got) == [(12.6, 12.8)]


class TestEndpointer:
    def rules(self):
        return (
            EndpointRule(must_contain_nonsilence=False, min_trailing_silence_s=5.0),
            EndpointRule(must_contain_nonsilence=True, min_trailing_silence_s=1.0),
            EndpointRule(
                must_contain_nonsilence=False,
                min_trailing_silence_s=0.0,
                min_utterance_length_s=20.0,
            ),
        )

    def ep(self, **kw):
        return Endpointer(cfg(endpoint_rules=self.rules(), **kw), SPF)

    def test_rule2_fires_one_second_after_speech_stops(self):
        e = self.ep()
        assert e.push(trace((0.9, 100))) is None
        assert e.push(trace((0.0, 60))) is None
        decision = e.push(trace((0.0, 60)))
        assert decision is not None
        assert decision.reason == "rule2"
        assert decision.terminal is False

    def test_rule1_fires_on_silence_alone_after_five_seconds(self):
        e = self.ep()
        assert e.push(trace((0.0, 400))) is None
        decision = e.push(trace((0.0, 150)))
        assert decision is not None and decision.reason == "rule1"

    def test_rule3_caps_a_turn_that_never_pauses(self):
        e = self.ep()
        assert e.push(trace((0.9, 1500))) is None
        decision = e.push(trace((0.9, 600)))
        assert decision is not None and decision.reason == "rule3"

    def test_one_spurious_frame_does_not_reset_the_silence_counter(self):
        """The property Riva's windowed test buys over a plain run-length one.

        WeNet and sherpa-onnx count trailing blanks with a counter that any
        non-blank frame zeroes, so a single bad frame in a pause costs a whole
        endpoint.  Here the window ending at that frame is not active, so it
        cannot qualify.
        """
        e = self.ep()
        e.push(trace((0.9, 100)))
        e.push(trace((0.0, 50)))
        e.push([0.9])  # one lone frame, mid-pause
        decision = e.push(trace((0.0, 60)))
        assert decision is not None, "a single frame reset the counter"
        assert decision.reason == "rule2"

    def test_sustained_speech_does_reset_the_counter(self):
        """The other half: real speech must still reset it, or a turn could
        never be extended once a pause started."""
        e = self.ep()
        e.push(trace((0.9, 100)))
        e.push(trace((0.0, 50)))
        e.push(trace((0.9, 40)))  # the speaker resumed
        assert e.push(trace((0.0, 60))) is None

    def test_decoded_any_overrides_the_derived_answer(self):
        """A frame-synchronous family knows whether it emitted a token, which is
        better evidence than Kaldi's length comparison."""
        e = self.ep()
        assert e.push(trace((0.0, 200)), decoded_any=False) is None
        e2 = self.ep()
        decision = e2.push(trace((0.0, 200)), decoded_any=True)
        assert decision is not None and decision.reason == "rule2"

    def test_it_fires_at_most_once_per_turn(self):
        e = self.ep()
        e.push(trace((0.9, 100)))
        assert e.push(trace((0.0, 200))) is not None
        assert e.push(trace((0.0, 200))) is None

    def test_start_timeout_closes_a_stream_that_never_spoke(self):
        e = self.ep(speech_start_timeout_s=2.0)
        decision = e.push(trace((0.0, 250)))
        assert decision is not None
        assert decision.reason == "speech_start_timeout"
        assert decision.terminal is True, "a timeout closes the stream, not the turn"

    def test_the_start_timeout_is_cancelled_once_speech_begins(self):
        """Google's specified behaviour: cancelled for the rest of the stream,
        not merely for the current turn, so a pause between turns cannot close
        a live session."""
        e = self.ep(speech_start_timeout_s=2.0)
        e.note_speech_started()
        assert e.push(trace((0.0, 250))) is None

    def test_end_timeout_is_terminal_where_a_rule_is_not(self):
        # Below rule2's own 1.0 s, or the rule would fire first and the timeout
        # could never be reached — which is the correct precedence, and is what
        # the next test pins.
        e = self.ep(speech_end_timeout_s=0.5)
        e.note_speech_started()
        e.push(trace((0.9, 50)))
        decision = e.push(trace((0.0, 200)))
        assert decision is not None
        assert decision.reason == "speech_end_timeout"
        assert decision.terminal is True, "a timeout closes the stream, not the turn"

    def test_a_rule_tighter_than_the_end_timeout_wins(self):
        """Whichever bound is tighter decides, and the turn-ending rule being
        tighter is the ordinary case: a client that set a long stream timeout
        still gets its turns ended on time."""
        e = self.ep(speech_end_timeout_s=5.0)
        e.note_speech_started()
        e.push(trace((0.9, 50)))
        decision = e.push(trace((0.0, 200)))
        assert decision is not None
        assert decision.reason == "rule2"
        assert decision.terminal is False
