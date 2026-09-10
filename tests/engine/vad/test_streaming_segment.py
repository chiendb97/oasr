# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Streaming ``vad.mode="segment"``: the gate, the turn reset, and the two clocks.

This is the only VAD mode that changes what the model sees, so the properties
worth pinning are the ones whose failures stay plausible:

* a window is skipped **only** when the detector has already judged past it, so
  an undecided window is encoded rather than dropped;
* skipping is paired with resetting the encoder's cache *and* its frame position,
  because advancing past frames the encoder never saw is otherwise a splice
  (AGENTS.md rule 13) — and a splice produces a transcript, just the wrong one;
* the model clock restarts at zero with the encoder while the reporting clock
  keeps accumulating, so a word in the third turn is still timed from the start
  of the stream;
* the transcript grows across a turn boundary rather than restarting, which is
  what a client accumulating partials sees.
"""

from __future__ import annotations

import math

import pytest
import torch

from oasr.engine.vad_stage import StreamingVadStage
from oasr.vad import SpeechSegmenter, VadConfig

SR = 16000


def tone(seconds: float, amp: float = 0.3, freq: float = 220.0) -> torch.Tensor:
    n = int(SR * seconds)
    t = torch.arange(n, dtype=torch.float32) / SR
    env = 0.5 + 0.5 * torch.sin(2 * math.pi * 4 * t)
    return (
        amp
        * env
        * (torch.sin(2 * math.pi * freq * t) + 0.5 * torch.sin(2 * math.pi * 3 * freq * t))
    )


def hiss(seconds: float, amp: float = 1e-4) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    return amp * torch.randn(int(SR * seconds), generator=g)


class _Req:
    """The two attributes the stage reads off a request."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id


def stage(**kw) -> StreamingVadStage:
    cfg = VadConfig(backend="energy", mode="segment", sample_rate=SR, **kw).resolve("streaming")
    return StreamingVadStage(cfg, seconds_per_frame=cfg.hop_ms / 1000.0, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Segmenter queries the gate is built on
# ---------------------------------------------------------------------------


class TestSegmenterGateQueries:
    def segmenter(self, **kw):
        cfg = VadConfig(mode="segment", **kw).resolve("streaming")
        return SpeechSegmenter(cfg, seconds_per_frame=0.01)

    def trace(self, seg, *runs):
        """``(value, seconds)`` pairs pushed as frames."""
        for value, seconds in runs:
            seg.push([value] * int(round(seconds / 0.01)))
        return seg

    def test_an_open_run_covers_everything_up_to_now(self):
        """A pause shorter than ``min_silence_ms`` keeps the run open, and audio
        inside an unconfirmed pause must still be encoded."""
        seg = self.trace(self.segmenter(), (0.0, 1.0), (0.9, 1.0), (0.0, 0.5))
        assert seg.open_run_start == pytest.approx(1.0, abs=0.02)
        assert seg.last_segment_end is None, "the pause is not confirmed yet"
        assert seg.overlaps_speech(2.2, 2.4), "an open run is speech until it closes"

    def test_a_confirmed_silence_closes_the_run_and_frees_the_tail(self):
        seg = self.trace(self.segmenter(), (0.0, 1.0), (0.9, 1.0), (0.0, 3.0))
        assert seg.open_run_start is None
        assert seg.last_segment_end == pytest.approx(2.0, abs=0.02)
        assert seg.overlaps_speech(1.5, 1.8)
        assert not seg.overlaps_speech(2.5, 3.0)

    def test_an_empty_or_inverted_span_is_never_speech(self):
        seg = self.trace(self.segmenter(), (0.9, 2.0), (0.0, 3.0))
        assert not seg.overlaps_speech(1.0, 1.0)
        assert not seg.overlaps_speech(1.5, 1.0)


# ---------------------------------------------------------------------------
# The encoder gate
# ---------------------------------------------------------------------------


class TestEncoderGate:
    def fed(self, wav, request_id="r", **kw):
        st = stage(**kw)
        st.open(request_id)
        st.feed_audio(request_id, wav)
        st.advance_audio([_Req(request_id)])
        return st

    def test_an_unclassified_window_is_encoded(self):
        """The gate's bias, stated as a test: encoding silence costs time and
        dropping speech costs words, so anything undecided is encoded."""
        st = self.fed(hiss(1.0))
        # Far past what the detector has seen.
        assert st.should_encode("r", 30.0, 30.6)
        # And a stream with no state at all — an abort mid-tick, say.
        assert st.should_encode("gone", 0.0, 0.6)

    def test_a_confirmed_silence_is_skippable_but_its_padding_is_not(self):
        st = self.fed(torch.cat([tone(2.0), hiss(6.0), tone(2.0)]))
        pad = st.pad_seconds
        assert pad > 0
        # Deep inside the gap: skippable.
        assert not st.should_encode("r", 4.5, 5.1)
        # Straddling the speech that precedes it: not.
        assert st.should_encode("r", 1.8, 2.4)
        # Inside the padding that follows the speech: not.
        assert st.should_encode("r", 2.0 + pad / 2, 2.0 + pad)

    def test_the_turn_boundary_needs_a_confirmed_run(self):
        st = self.fed(torch.cat([tone(2.0), hiss(0.5)]))
        assert st.turn_boundary("r") is None, "the pause is still open"
        st.feed_audio("r", hiss(5.0))
        st.advance_audio([_Req("r")])
        boundary = st.turn_boundary("r")
        assert boundary is not None
        assert boundary == pytest.approx(2.0 + st.pad_seconds, abs=0.15)

    def test_the_cap_does_not_close_a_turn_mid_speech(self):
        """``max_speech_s`` closes a run at the current frame rather than at a
        silence, and a turn closed there resets the encoder in the middle of a
        word — which streaming, unlike the offline fan-out, cannot recover by
        re-reading the audio.  A cap is a length decision, not a turn decision:
        measured at +0.88 WER on long-form audio the detector reads as one run.
        """
        st = self.fed(tone(6.0), max_speech_s=1.0)
        seg = st.state("r").segmenter
        assert seg.last_segment_end is not None, "max_speech_s did not fire"
        assert seg.open_run_start is not None, "speech did not resume"
        assert st.turn_boundary("r") is None, "the turn closed on a gapless cut"

    def test_a_confirmed_silence_still_closes_the_turn_under_a_cap(self):
        """The guard must not disable turn closing wherever a cap is set."""
        st = self.fed(torch.cat([tone(3.0), hiss(6.0)]), max_speech_s=1.0)
        assert st.turn_boundary("r") is not None


class TestIncrementalAudioFeed:
    def test_ragged_chunks_do_not_drift_against_a_single_call(self):
        """The carry is what keeps the detector's clock on the encoder's.

        Feeding at a size that is not a whole number of analysis frames is the
        normal case — a 20 ms network chunk is not a multiple of a 25 ms window
        — and dropping the remainder would shorten the stream by up to one frame
        per chunk, which is a drift, not an offset.
        """
        wav = torch.cat([tone(1.0), hiss(3.0), tone(1.0)])
        whole = stage()
        whole.open("w")
        whole.feed_audio("w", wav)
        whole.advance_audio([_Req("w")])

        ragged = stage()
        ragged.open("r")
        step = 3333  # deliberately coprime with the 160-sample hop
        for start in range(0, wav.numel(), step):
            ragged.feed_audio("r", wav[start : start + step])
            ragged.advance_audio([_Req("r")])

        assert ragged.classified_until("r") == pytest.approx(whole.classified_until("w"), abs=1e-9)
        a = [(round(s.start, 3), round(s.end, 3)) for s in whole.state("w").segmenter.raw_segments]
        b = [(round(s.start, 3), round(s.end, 3)) for s in ragged.state("r").segmenter.raw_segments]
        assert a == b and a, "ragged chunking changed the segmentation"

    def test_one_call_serves_the_whole_pool(self):
        """Streams are classified together, in one detector call, even when
        their buffers differ in length — the padded rows are masked by the
        per-row frame counts, not left to the widest member's tail."""
        st = stage()
        for rid, wav in (("a", tone(2.0)), ("b", torch.cat([tone(0.5), hiss(0.7)]))):
            st.open(rid)
            st.feed_audio(rid, wav)
        st.advance_audio([_Req("a"), _Req("b")])
        # Each row is classified to its own length, not the widest member's: the
        # short one must not inherit the long one's padding as speech.
        framing = st.detector.framing
        expect = lambda n: framing.frames_for(n) * framing.seconds_per_frame(SR)  # noqa: E731
        assert st.classified_until("a") == pytest.approx(expect(int(SR * 2.0)), abs=1e-9)
        assert st.classified_until("b") == pytest.approx(expect(int(SR * 1.2)), abs=1e-9)

    def test_closing_a_stream_releases_its_queued_audio(self):
        st = stage()
        st.open("a")
        st.feed_audio("a", tone(1.0))
        assert st._audio.get("a")
        st.close("a")
        assert "a" not in st._audio and st.state("a") is None


class TestNbestRefusal:
    """The refusal itself, without a checkpoint.

    Copied members rather than a stub, the way ``tests/test_word_timings.py``
    does it: the thing under test *is* the base implementation.
    """

    def strategy(self, mode):
        from types import SimpleNamespace

        from oasr.engine.decode.base import DecodeStrategy

        class _Strategy:
            decode_type = "ctc"
            selective_options = ()
            word_timing_modes = ()
            _clock = None
            _SELECTIVE_UNSET = DecodeStrategy._SELECTIVE_UNSET
            _SPEECH_ACTIVITY_OPTIONS = DecodeStrategy._SPEECH_ACTIVITY_OPTIONS
            validate_options = DecodeStrategy.validate_options
            _require_word_timings = DecodeStrategy._require_word_timings
            _require_speech_activity = DecodeStrategy._require_speech_activity
            _reject_nbest_across_turns = DecodeStrategy._reject_nbest_across_turns

        obj = _Strategy()
        obj._config = SimpleNamespace(vad=VadConfig(mode=mode, backend="energy"))
        return obj

    def test_segment_mode_refuses_it(self):
        from oasr.engine.request import DecodingOptions

        with pytest.raises(ValueError, match="n_best > 1 cannot be served"):
            self.strategy("segment").validate_options(DecodingOptions(n_best=3), streaming=True)

    @pytest.mark.parametrize(
        "mode,streaming",
        [
            ("segment", False),  # offline segments are separate requests, not turns
            ("observe", True),  # nothing is cut, so nothing is lost
            ("off", True),
        ],
    )
    def test_everything_else_still_accepts_it(self, mode, streaming):
        from oasr.engine.request import DecodingOptions

        self.strategy(mode).validate_options(DecodingOptions(n_best=3), streaming=streaming)


# ---------------------------------------------------------------------------
# Engine level — the real thing, on a real checkpoint
# ---------------------------------------------------------------------------


def _corpus(gap_s: float, wav_dir, n: int = 3):
    """``n`` utterances separated by ``gap_s`` of digital silence."""
    import pathlib

    import numpy as np
    import soundfile as sf

    paths = sorted(pathlib.Path(wav_dir).glob("*.wav"))[:n]
    if len(paths) < n:
        pytest.skip(f"need {n} wav files")
    parts, spans, t = [], [], 0.0
    for i, path in enumerate(paths):
        data, sr = sf.read(str(path), dtype="float32")
        assert sr == SR
        if i:
            parts.append(np.zeros(int(gap_s * SR), dtype="float32"))
            t += gap_s
        spans.append((t, t + len(data) / SR))
        parts.append(data)
        t += len(data) / SR
    return torch.from_numpy(np.concatenate(parts)), t, spans


class TestStreamingSegmentEngine:
    def engine(self, ckpt_dir, vad):
        from oasr.engine import ASREngine, EngineConfig

        return ASREngine(
            EngineConfig(
                ckpt_dir=ckpt_dir,
                service_mode="streaming",
                max_batch_size=2,
                max_num_blocks=1024,
                vad=vad,
            )
        )

    def drive(self, engine, wav, chunk=SR // 5, decoding=None):
        """Feed a stream at a live-ish cadence; collect every output it makes."""
        rid = engine.add_streaming_request(decoding=decoding)
        pos, produced, final = 0, [], None
        n = int(wav.numel())
        while final is None:
            if pos < n:
                end = min(pos + chunk, n)
                engine.feed_chunk(rid, wav[pos:end], is_last=(end >= n))
                pos = end
            for out in engine.step():
                produced.append(out)
                if out.finished:
                    final = out
        return final, produced

    @pytest.mark.requires_assets("CKPT_DIR", "WAV_DIR")
    def test_turns_close_and_the_transcript_keeps_growing(self, ckpt_dir, wav_dir):
        """A client accumulating partials must never see the transcript shrink.

        The decoder is reset at every turn boundary, so without the carry the
        partial after a boundary would report only the new turn — which a client
        cannot tell apart from the recogniser changing its mind.
        """
        wav, _total, spans = _corpus(5.0, wav_dir)
        engine = self.engine(ckpt_dir, {"mode": "segment", "backend": "energy"})
        try:
            final, produced = self.drive(engine, wav, decoding={"word_timestamps": True})
        finally:
            engine.shutdown()

        boundaries = [o for o in produced if o.endpoint_reason == "vad_segment"]
        assert len(boundaries) == len(spans) - 1, "one turn boundary per confirmed gap"
        lengths = [len(o.text.split()) for o in produced if o.text]
        assert lengths == sorted(lengths), "the transcript went backwards at a boundary"
        assert len(final.text.split()) >= lengths[-1]
        assert final.segments and len(final.segments) == len(spans)

    @pytest.mark.requires_assets("CKPT_DIR", "WAV_DIR")
    def test_timings_stay_in_session_seconds_across_a_reset(self, ckpt_dir, wav_dir):
        """The model clock restarts with the encoder; the reporting clock does not.

        Getting these two the wrong way round makes every turn after the first
        report timings that start again from zero — monotone within a turn, and
        wrong for the stream.
        """
        wav, total, spans = _corpus(5.0, wav_dir)
        engine = self.engine(ckpt_dir, {"mode": "segment", "backend": "energy"})
        try:
            final, _ = self.drive(engine, wav, decoding={"word_timestamps": True})
        finally:
            engine.shutdown()

        words = final.words or []
        assert len(words) > 10
        assert all(a.start <= b.start + 1e-6 for a, b in zip(words, words[1:]))
        # The last word must land in the last utterance, not back at the start.
        assert words[-1].end > spans[-1][0]
        assert words[-1].end <= total + 0.5
        # Every word is still a literal substring of the stitched transcript.
        cursor = 0
        for word in words:
            found = final.text.find(word.word, cursor)
            assert found >= 0, f"{word.word!r} is not a substring of the transcript"
            cursor = found + len(word.word)

    @pytest.mark.requires_assets("CKPT_DIR", "WAV_DIR")
    def test_a_backlogged_stream_skips_the_silence_it_can_prove(self, ckpt_dir, wav_dir):
        """The compute win, and its shape.

        Skipping needs the detector to be ahead of the encoder, which it is
        exactly when the stream is backlogged — the case where saving encoder
        work is worth something.  What survives the gate is the padding plus one
        window's clearance at each edge, so the fraction skipped grows with the
        gap rather than being a constant.
        """
        engine_vad = {"mode": "segment", "backend": "energy"}
        skipped = {}
        for gap in (3.0, 12.0):
            wav, total, spans = _corpus(gap, wav_dir)
            engine = self.engine(ckpt_dir, engine_vad)
            try:
                engine.transcribe_outputs([wav], streaming=True)
                counters = engine.metrics_snapshot().get("counters", {})
                skipped[gap] = counters.get("oasr_engine_audio_seconds_skipped_total", 0.0)
            finally:
                engine.shutdown()
            silence = gap * (len(spans) - 1)
            assert 0.0 < skipped[gap] < silence + 1e-6
        assert skipped[12.0] > 3 * skipped[3.0], "skipping did not scale with the silence"

    @pytest.mark.requires_assets("CKPT_DIR", "WAV_DIR")
    def test_the_transcript_survives_the_cuts(self, ckpt_dir, wav_dir):
        """Segmentation must not cost words.

        Compared against the same stream with the VAD off — a resetting encoder
        starts each turn on a clean context, so the two are not required to be
        byte-identical, but they are required to say the same thing.
        """
        import difflib

        wav, _total, _spans = _corpus(6.0, wav_dir)
        outs = {}
        for tag, vad in (("off", None), ("segment", {"mode": "segment", "backend": "energy"})):
            engine = self.engine(ckpt_dir, vad)
            try:
                outs[tag] = engine.transcribe_outputs([wav], streaming=True)[0].text
            finally:
                engine.shutdown()
        ratio = difflib.SequenceMatcher(a=outs["off"].split(), b=outs["segment"].split()).ratio()
        assert ratio > 0.93, f"segmentation changed the transcript: {ratio:.3f}"

    @pytest.mark.requires_assets("CKPT_DIR", "WAV_DIR")
    def test_nbest_is_refused_rather_than_answered_with_one_turn(self, ckpt_dir, wav_dir):
        """Alternatives of separate turns do not compose into alternatives of the
        stream — the same reason ``longform.py`` refuses to merge them."""
        wav, _total, _spans = _corpus(4.0, wav_dir, n=2)
        engine = self.engine(ckpt_dir, {"mode": "segment", "backend": "energy"})
        try:
            with pytest.raises(ValueError, match="n_best > 1 cannot be served"):
                engine.add_streaming_request(decoding={"n_best": 3})
        finally:
            engine.shutdown()
        del wav

    @pytest.mark.slow
    @pytest.mark.requires_assets("CKPT_DIR", "WAV_DIR")
    def test_turn_boundaries_under_pool_pressure_do_not_corrupt_neighbours(self, ckpt_dir, wav_dir):
        """A turn boundary hands KV blocks back to a pool other streams are using.

        A solo stream cannot see a missing cross-stream ordering here — it is the
        only claimant on the pool, so a block returned too early goes straight
        back to the stream that just stopped reading it.  So this congests the
        pool on purpose: several streams at once, a block budget too small for
        all of them to keep unlimited history, and gap lengths chosen so their
        turn boundaries do **not** line up.

        Two things make the oracle exact rather than approximate.  The reference
        streams are driven through the *same* chunked feed loop, because pacing
        decides how far ahead of the encoder the detector runs and therefore how
        much gets skipped — comparing a bulk-fed reference against a live-fed run
        would measure that, not co-tenancy.  And both phases run on **one**
        engine: building a second one in the same process is its own hazard,
        unrelated to voice activity, and would put a pre-existing failure inside
        this test's blast radius.
        """
        gaps = (4.0, 6.0, 5.0, 7.0)
        waves = [_corpus(gap, wav_dir, n=3)[0] for gap in gaps]

        from oasr.engine import ASREngine, EngineConfig

        engine = ASREngine(
            EngineConfig(
                ckpt_dir=ckpt_dir,
                service_mode="streaming",
                max_batch_size=len(waves),
                max_num_blocks=256,  # deliberately tight: boundaries must recycle
                vad={"mode": "segment", "backend": "energy"},
            )
        )
        try:
            solo = [self.drive(engine, wav)[0].text for wav in waves]

            rids = [engine.add_streaming_request() for _ in waves]
            pos, done = [0] * len(waves), {}
            chunk = SR // 5
            while len(done) < len(waves):
                for i, wav in enumerate(waves):
                    n = int(wav.numel())
                    if pos[i] < n:
                        end = min(pos[i] + chunk, n)
                        engine.feed_chunk(rids[i], wav[pos[i] : end], is_last=(end >= n))
                        pos[i] = end
                for out in engine.step():
                    if out.finished:
                        done[out.request_id] = out
            turns = engine.metrics_snapshot()["counters"]["oasr_engine_vad_segments_total"]
        finally:
            engine.shutdown()

        assert turns >= 2 * len(waves), "no turn boundary ran, so nothing was under test"
        for i, rid in enumerate(rids):
            assert done[rid].text == solo[i], f"stream {i} diverged under co-tenant load"
