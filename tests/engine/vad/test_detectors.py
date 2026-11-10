# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The detector half of voice activity, and how the engine resolves it.

The load-bearing test in this file is
:class:`TestVadOffChangesNothing`: with ``vad.mode="off"`` every transcript must
be byte-identical to what it was before the axis existed.  Everything else is a
feature; that one is the contract the whole axis rests on.
"""

from __future__ import annotations

import math

import assets
import pytest
import torch
from helpers.audio import SR, hiss, tone

from oasr.vad import (
    ROLES,
    SpeechSegmenter,
    VadConfig,
    VadFraming,
    VadSpec,
    build_detector,
    get_vad_spec,
    list_vad,
    register_vad,
)


class TestRegistry:
    def test_the_built_ins_are_registered(self):
        assert set(list_vad()) >= {
            "energy",
            "ctc_blank",
            "transducer_blank",
            "cif_alpha",
            "aed_no_speech",
        }

    def test_an_unknown_backend_names_the_registered_ones(self):
        with pytest.raises(NotImplementedError, match="ctc_blank"):
            get_vad_spec("nope")

    def test_an_asr_derived_detector_cannot_claim_presegment(self):
        """The declaration that keeps *"it cannot run before the encoder"* a fact
        of the type rather than a comment — and it is checked at registration,
        because discovering it on the first request means discovering it in
        production."""
        with pytest.raises(ValueError, match="cannot run before the encoder"):
            register_vad(
                VadSpec(
                    kind="_bad_presegment",
                    factory=lambda *a, **k: None,
                    consumes="asr_log_probs",
                    modes=("presegment",),
                )
            )

    def test_a_waveform_detector_must_declare_a_framing(self):
        with pytest.raises(ValueError, match="declares no framing"):
            register_vad(
                VadSpec(
                    kind="_bad_framing",
                    factory=lambda *a, **k: None,
                    consumes="waveform",
                    modes=("presegment",),
                )
            )

    def test_every_registered_role_is_a_known_role(self):
        for kind in list_vad():
            assert set(get_vad_spec(kind).modes) <= set(ROLES), kind

    def test_asr_derived_kinds_never_claim_presegment(self):
        for kind in list_vad():
            spec = get_vad_spec(kind)
            if spec.is_asr_derived:
                assert "presegment" not in spec.modes, kind


class TestVadFraming:
    def test_frames_for_matches_the_declared_grid(self):
        f = VadFraming(span=400, hop=160)
        assert f.frames_for(0) == 0
        assert f.frames_for(399) == 0
        assert f.frames_for(400) == 1
        assert f.frames_for(560) == 2
        assert f.min_samples == 400

    def test_seconds_per_frame_is_the_hop(self):
        assert VadFraming(span=400, hop=160).seconds_per_frame(16000) == pytest.approx(0.01)


class TestConfig:
    def test_the_preset_follows_the_service_mode(self):
        """Silero's own numbers for turn-taking; faster-whisper's for
        pre-segmentation.  One default set would be wrong for one of them."""
        streaming = VadConfig(backend="energy", mode="observe").resolve("streaming")
        offline = VadConfig(backend="energy", mode="observe").resolve("offline")
        assert streaming.preset == "turn"
        assert offline.preset == "segment"
        assert streaming.min_silence_ms == 100
        assert offline.min_silence_ms == 2000
        assert offline.speech_pad_ms > streaming.speech_pad_ms

    def test_neg_threshold_is_derived_but_never_non_positive(self):
        assert VadConfig(threshold=0.5).resolve("offline").neg_threshold == pytest.approx(0.35)
        # A run that can never exit is a segment that never closes.
        assert VadConfig(threshold=0.05).resolve("offline").neg_threshold > 0.0

    def test_an_explicit_knob_survives_the_preset(self):
        got = VadConfig(backend="energy", min_silence_ms=42).resolve("offline")
        assert got.min_silence_ms == 42

    def test_an_unknown_backend_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="vad backend must be one of"):
            VadConfig(backend="silero_but_not_yet")

    def test_string_options_are_typed_from_the_declared_field(self):
        got = VadConfig.coerce({"mode": "observe", "min_silence_ms": "1500"})
        assert got.min_silence_ms == 1500 and isinstance(got.min_silence_ms, int)

    def test_an_unknown_key_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="unknown vad config keys"):
            VadConfig.coerce({"mode": "observe", "treshold": 0.5})

    @pytest.mark.parametrize("bad", [0.1, 61.0])
    def test_timeouts_are_bounded(self, bad):
        with pytest.raises(ValueError, match=r"\[0.5, 60\]"):
            VadConfig(speech_end_timeout_s=bad)


class TestEnergyDetector:
    def detector(self, **kw):
        cfg = VadConfig(backend="energy", mode="segment", sample_rate=SR, **kw).resolve("offline")
        return cfg, build_detector(cfg, device=torch.device("cpu"))

    def test_speech_scores_above_silence(self):
        _cfg, det = self.detector()
        wav = torch.cat([hiss(1.0), tone(1.0), hiss(1.0)]).unsqueeze(0)
        probs, lens = det.detect(wav, torch.tensor([wav.size(1)]))
        row = probs[0, : int(lens[0])]
        n = row.numel() // 3
        assert row[n : 2 * n].mean() > 0.8
        assert row[:n].mean() < 0.2

    def test_digital_silence_is_not_speech(self):
        """The peak-relative rule inverts on an all-zero row unless it is
        guarded: the peak *is* the floor, so every frame sits at the peak.
        ``finalize_silence_pad`` appends exactly such a run to every closed
        stream, so this is not a hypothetical input."""
        _cfg, det = self.detector()
        wav = torch.zeros(1, SR * 2)
        probs, lens = det.detect(wav, torch.tensor([wav.size(1)]))
        assert float(probs[0, : int(lens[0])].max()) == 0.0

    def test_it_is_invariant_to_the_waveform_scale(self):
        """``audio_scale`` is a per-framework convention (WeNet 32768,
        icefall 1.0), so an absolute energy threshold would be a silent,
        scale-dependent failure.  Same audio, two conventions, same answer."""
        cfg, det = self.detector()
        wav = torch.cat([hiss(0.5), tone(1.0), hiss(0.5)])
        seg = SpeechSegmenter(cfg, det.seconds_per_frame)
        out = []
        for scale in (1.0, 32768.0):
            probs, lens = det.detect((wav * scale).unsqueeze(0), torch.tensor([wav.numel()]))
            out.append(seg.run(probs[0, : int(lens[0])].tolist()))
        assert [(round(s.start, 3), round(s.end, 3)) for s in out[0]] == [
            (round(s.start, 3), round(s.end, 3)) for s in out[1]
        ]

    def test_padding_past_a_row_length_is_never_speech(self):
        """A padded row's tail is whatever the batch's widest member made it;
        a detector that leaves it alone reports the *padding* as speech, and
        that becomes a segment, and then a transcript request for silence."""
        _cfg, det = self.detector()
        batch = torch.stack([torch.cat([tone(1.0), torch.zeros(SR)]), tone(2.0)])
        lengths = torch.tensor([SR, SR * 2])
        probs, frame_lengths = det.detect(batch, lengths)
        tail = probs[0, int(frame_lengths[0]) :]
        assert float(tail.abs().max() if tail.numel() else torch.zeros(1).max()) == 0.0

    def build(self, **kw):
        return self.detector(**kw)[1]

    def test_it_declares_every_role(self):
        """It runs on the audio, so it can precede the encoder in either flow —
        and it carries state, so it can do that incrementally."""
        spec = get_vad_spec("energy")
        assert spec.can("presegment") and spec.can("stream") and spec.can("posthoc")
        assert spec.stateful, "the running peak has to survive a chunk boundary"

    def test_streaming_matches_one_shot_once_the_peak_is_established(self):
        """The carried state is what makes the two flows agree.

        Without it each chunk is normalised against *its own* loudest frame, so a
        chunk of room tone reads as a chunk of speech — the failure a single-chunk
        test cannot see, because with one chunk the two forms are identical.
        """
        det = self.build()
        torch.manual_seed(0)
        wav = torch.cat([torch.randn(16000) * 0.5, torch.randn(16000) * 1e-4])
        one_shot, lengths = det.detect(wav.unsqueeze(0), torch.tensor([wav.numel()]))
        expect = one_shot[0].tolist()

        hop = det.framing.hop
        state, got, buf = None, [], torch.zeros(0)
        for start in range(0, wav.numel(), 4000):  # a chunk size that is not a frame multiple
            buf = torch.cat([buf, wav[start : start + 4000]])
            probs, frames, state = det.detect_streaming(
                buf.unsqueeze(0), torch.tensor([buf.numel()]), state
            )
            n = int(frames[0])
            got.extend(probs[0, :n].tolist())
            buf = buf[n * hop :]

        assert len(got) == len(expect) == int(lengths[0])
        # The tail, where the running peak has caught up with the utterance peak.
        assert max(abs(a - b) for a, b in zip(expect[100:], got[100:])) < 1e-5
        # And the quiet half really is quiet, in both forms.
        assert max(got[-50:]) < 0.05

    def test_a_chunk_of_pure_silence_does_not_reset_the_reference(self):
        """The regression the carried peak exists to prevent."""
        det = self.build()
        torch.manual_seed(1)
        loud = (torch.randn(8000) * 0.5).unsqueeze(0)
        _, _, state = det.detect_streaming(loud, torch.tensor([8000]), None)
        quiet = (torch.randn(8000) * 1e-4).unsqueeze(0)
        probs, frames, _ = det.detect_streaming(quiet, torch.tensor([8000]), state)
        assert probs[0, : int(frames[0])].max().item() < 0.05


class TestAsrDerivedDetectors:
    def build(self, kind, **kw):
        cfg = VadConfig(backend=kind, mode="observe").resolve("streaming")
        return build_detector(cfg, seconds_per_frame=0.04, **kw)

    def test_ctc_blank_is_one_minus_the_blank_posterior(self):
        det = self.build("ctc_blank", blank_id=0, dilate_s=0.0)
        # Frame 0 is confidently blank, frame 1 confidently not.
        log_probs = torch.log(
            torch.tensor([[[0.99, 0.005, 0.005], [0.01, 0.98, 0.01]]], dtype=torch.float32)
        )
        probs, _lens = det.detect_from_asr(log_probs, torch.tensor([2]))
        assert probs[0, 0] == pytest.approx(0.01, abs=1e-3)
        assert probs[0, 1] == pytest.approx(0.99, abs=1e-3)

    def test_ctc_blank_dilation_closes_in_word_gaps(self):
        """CTC is peaky: measured on read speech at 40 ms frames only ~15 % of
        frames clear 0.5, so a raw trace splits inside words."""
        det = self.build("ctc_blank", blank_id=0, dilate_s=0.1)
        spikes = torch.full((1, 24, 2), math.log(0.99))
        emissions = list(range(4, 20, 4))  # a non-blank every 4th frame, 4..16
        for f in emissions:
            spikes[0, f, 0] = math.log(0.01)
            spikes[0, f, 1] = math.log(0.99)
        probs, _ = det.detect_from_asr(spikes, torch.tensor([24]))
        gaps = probs[0, emissions[0] : emissions[-1] + 1]
        assert float(gaps.min()) > 0.5, "dilation did not bridge the in-word gaps"
        # And it stops: ±0.1 s at a 40 ms frame is two frames, so the far tail
        # stays blank.  A dilation wide enough to bridge a real pause would smear
        # every boundary by half its width, which is what the declared
        # ``min_silence_floor_ms`` exists to avoid having to do.
        assert float(probs[0, -1]) < 0.5, "dilation smeared past the emissions"

    def test_transducer_emission_spikes_become_a_run(self):
        det = self.build("transducer_blank", dilate_s=0.2)
        indicator = torch.zeros(1, 60)
        for f in (10, 17, 24, 33, 40):
            indicator[0, f] = 1.0
        probs, _ = det.detect_from_asr(indicator, torch.tensor([60]))
        run = probs[0, 10:41]
        assert float(run.min()) == pytest.approx(1.0)
        assert float(probs[0, 58]) == 0.0, "dilation reached past the emissions"

    def test_aed_no_speech_reads_the_named_token(self):
        det = self.build("aed_no_speech", no_speech_token_id=2)
        logits = torch.tensor([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]], dtype=torch.float32)
        probs, lens = det.detect_from_asr(logits, torch.tensor([1, 1]))
        assert probs.shape == (2, 1) and lens.tolist() == [1, 1]
        assert float(probs[0, 0]) < 0.01, "a confident <|nospeech|> must read as no speech"
        assert float(probs[1, 0]) > 0.99

    def test_a_token_id_outside_the_vocabulary_is_refused(self):
        det = self.build("aed_no_speech", no_speech_token_id=99)
        with pytest.raises(ValueError, match="outside the vocabulary"):
            det.detect_from_asr(torch.zeros(1, 3), torch.tensor([1]))

    def test_a_waveform_call_on_an_asr_detector_says_what_is_wrong(self):
        det = self.build("ctc_blank", blank_id=0)
        with pytest.raises(NotImplementedError, match="does not consume a waveform"):
            det.detect(torch.zeros(1, 16000), torch.tensor([16000]))


class TestOfflineSegmenter:
    def segmenter(self, **kw):
        from oasr.engine.vad_stage import OfflineVadSegmenter

        cfg = VadConfig(backend="energy", mode="segment", sample_rate=SR, **kw).resolve("offline")
        return OfflineVadSegmenter(cfg, torch.device("cpu"))

    def test_it_finds_the_speech_and_drops_the_gap(self):
        seg = self.segmenter()
        wav = torch.cat([hiss(1.0), tone(2.5), hiss(3.0), tone(1.8), hiss(0.7)])
        spans = seg.spans(wav)
        assert spans is not None and len(spans) == 2
        kept = sum(b - a for a, b in spans) / SR
        assert kept < wav.numel() / SR, "nothing was skipped"

    @pytest.mark.parametrize(
        "wav,why",
        [
            (torch.zeros(SR * 3), "digital silence"),
            (None, "all noise"),
        ],
    )
    def test_no_speech_does_not_fan_out(self, wav, why):
        """Fanning out to zero children would leave the parent request waiting
        for outputs that never arrive; decoding the whole thing is the safe
        answer, and produces whatever it produces."""
        seg = self.segmenter()
        audio = hiss(3.0) if wav is None else wav
        assert seg.spans(audio) is None, why

    def test_one_span_covering_everything_does_not_fan_out(self):
        seg = self.segmenter(speech_pad_ms=2000)
        assert seg.spans(tone(2.0)) is None

    @staticmethod
    def phrases(pauses, phrase_s=0.7):
        """Tone phrases separated by the given pauses (seconds), and where each
        pause lies, in samples.  The pauses are all shorter than the preset's
        ``min_silence_ms``, so the segmenter reads one continuous run."""
        parts, where, at = [tone(phrase_s)], [], int(phrase_s * SR)
        for pause in pauses:
            parts.append(hiss(pause))
            where.append((at, at + int(pause * SR)))
            at += int(pause * SR)
            parts.append(tone(phrase_s))
            at += int(phrase_s * SR)
        return torch.cat(parts), where

    def test_the_cap_is_not_cut_at_the_frame_it_fires(self):
        """``max_speech_s`` closes a run *at the current frame*, not at a
        silence, so the two segments it leaves touch.  Fanning those out decodes
        the word on the boundary as two half-words with no shared context and no
        overlap to dedup against -- measured at +0.94 WER on long-form audio a
        detector reads as one continuous run.  So the touching segments are
        merged, and the merged span is cut again in a pause (below)."""
        seg = self.segmenter(max_speech_s=1.0)
        wav, pauses = self.phrases([0.3] * 6)
        segments = seg.segments(wav)
        assert len(segments) > 1, "max_speech_s did not fire"
        fired = [int(round(s.end * SR)) for s in segments[:-1]]
        cuts = [a for a, _ in seg.spans(wav)[1:]]
        assert not set(cuts) & set(fired), "a span was cut where the cap fired"

    def test_no_span_is_longer_than_the_cap(self):
        """The merge alone left a span unbounded.  A 900 s chapter the detector
        read as one run came back as one span, and a Zipformer handed the whole
        file ran out of memory (2026-11-09 report, D7)."""
        seg = self.segmenter(max_speech_s=2.0)
        wav, _ = self.phrases([0.3] * 10)
        spans = seg.spans(wav)
        assert spans is not None and len(spans) > 1
        assert all(b - a <= 2 * SR for a, b in spans)
        # The pieces touch: no sample is dropped or decoded twice.
        assert all(spans[k][1] == spans[k + 1][0] for k in range(len(spans) - 1))
        assert spans[0][0] == 0 and spans[-1][1] == wav.numel()

    def test_the_cut_lands_in_a_pause(self):
        seg = self.segmenter(max_speech_s=2.0)
        wav, pauses = self.phrases([0.3] * 10)
        for cut, _ in seg.spans(wav)[1:]:
            assert any(a < cut < b for a, b in pauses), f"cut at {cut / SR:.3f}s is in speech"

    def test_the_longest_pause_in_reach_wins(self):
        seg = self.segmenter(max_speech_s=2.5)
        wav, pauses = self.phrases([0.15, 0.45, 0.15])
        cuts = [a for a, _ in seg.spans(wav)[1:]]
        assert cuts, "nothing was cut"
        long_a, long_b = pauses[1]
        assert long_a < cuts[0] < long_b

    def test_a_long_pause_keeps_only_the_padding_on_each_side(self):
        """Cut in its middle, a 1.9 s pause left about a second of silence at a
        piece's edge, and Zipformer CTC filled it with words nobody said ("he
        said", +0.16 WER on the 150 s long-form set).  A pause longer than twice
        the padding is cut like a segment boundary instead."""
        seg = self.segmenter(max_speech_s=3.0)
        wav, pauses = self.phrases([0.3, 1.5, 0.3], phrase_s=0.9)
        spans = seg.spans(wav)
        gaps = [(spans[k][1], spans[k + 1][0]) for k in range(len(spans) - 1)]
        dropped = [(a, b) for a, b in gaps if a < b]
        assert len(dropped) == 1, f"expected one trimmed cut, got {gaps}"
        (left_end, right_start), (pause_a, pause_b) = dropped[0], pauses[1]
        pad = VadConfig(mode="segment").resolve("offline").speech_pad_seconds
        frame = 0.05 * SR  # the energy detector's edges move by a few hops
        assert abs(left_end - (pause_a + pad * SR)) <= frame
        assert abs(right_start - (pause_b - pad * SR)) <= frame

    def test_audio_with_no_pause_is_still_bounded(self):
        """With nothing quieter than speech, the bound still holds: the cut goes
        in the least speech-like frame.  Memory does not wait for a pause."""
        seg = self.segmenter(max_speech_s=1.0)
        spans = seg.spans(tone(5.0))
        assert spans is not None and all(b - a <= SR for a, b in spans)

    def test_a_span_within_the_cap_is_left_whole(self):
        seg = self.segmenter(max_speech_s=30.0)
        wav, _ = self.phrases([0.3] * 4)
        assert seg.spans(wav) is None, "one span covering the audio fanned out"

    def test_a_real_gap_still_cuts_after_the_merge(self):
        """The merge must not swallow the cuts the mode exists for."""
        seg = self.segmenter(max_speech_s=1.0)
        spans = seg.spans(torch.cat([tone(2.0), hiss(4.0), tone(2.0)]))
        assert spans is not None
        middle = 4 * SR  # inside the 4 s gap, past either side's padding
        assert not any(a <= middle < b for a, b in spans), "the gap was not dropped"
        assert any(b <= 2.5 * SR for _, b in spans) and any(a >= 5.5 * SR for a, _ in spans)


class TestLongFormPieces:
    """``OfflineVadSegmenter.pieces``: where ``long_form`` cuts a file.

    D8 (2026-11-09 report): ``long_form`` cut every 30 s wherever that fell, so
    windows started and ended in the middle of words, and Whisper answered some
    of them with a word or two.  Whisper large-v3 lost 1,000 of 13,500 words on
    150 s recordings, and whisper-tiny 1,365.
    """

    def segmenter(self, **kw):
        from oasr.engine.vad_stage import OfflineVadSegmenter

        cfg = VadConfig(backend="energy", mode="segment", sample_rate=SR, **kw).resolve("offline")
        return OfflineVadSegmenter(cfg, torch.device("cpu"))

    phrases = staticmethod(TestOfflineSegmenter.phrases)

    def test_audio_that_fits_is_one_piece(self):
        seg = self.segmenter(max_speech_s=30.0)
        wav, _ = self.phrases([0.3] * 4)
        assert seg.pieces(wav) == [(0, wav.numel())]

    def test_nothing_is_dropped_as_non_speech(self):
        """``long_form`` decodes the whole file.  The gaps :meth:`spans` drops
        stay in, unless a cut lands in one."""
        seg = self.segmenter(max_speech_s=30.0)
        wav = torch.cat([hiss(1.0), tone(2.5), hiss(3.0), tone(1.8), hiss(0.7)])
        assert seg.spans(wav) is not None, "the detector found no gap to drop"
        assert seg.pieces(wav) == [(0, wav.numel())]

    def test_every_piece_fits_and_every_cut_lands_in_a_pause(self):
        seg = self.segmenter(max_speech_s=2.0)
        wav, pauses = self.phrases([0.3] * 10)
        pieces = seg.pieces(wav)
        assert len(pieces) > 1 and all(b - a <= 2 * SR for a, b in pieces)
        assert pieces[0][0] == 0 and pieces[-1][1] == wav.numel()
        for (_, end), (start, _) in zip(pieces, pieces[1:]):
            assert end == start, "a short pause is cut in its middle: the pieces touch"
            assert any(a < end < b for a, b in pauses), f"cut at {end / SR:.3f}s is in speech"

    def test_only_the_middle_of_a_long_pause_is_left_out(self):
        seg = self.segmenter(max_speech_s=3.0)
        wav, pauses = self.phrases([0.3, 1.5, 0.3], phrase_s=0.9)
        pieces = seg.pieces(wav)
        gaps = [(pieces[k][1], pieces[k + 1][0]) for k in range(len(pieces) - 1)]
        dropped = [(a, b) for a, b in gaps if a < b]
        assert len(dropped) == 1 and pauses[1][0] < dropped[0][0] < dropped[0][1] < pauses[1][1]

    def test_sound_with_no_pause_is_still_cut_to_fit(self):
        seg = self.segmenter(max_speech_s=1.0)
        pieces = seg.pieces(tone(5.0))
        assert len(pieces) > 1 and all(b - a <= SR for a, b in pieces)
        assert pieces[0][0] == 0 and pieces[-1][1] == 5 * SR


@pytest.mark.cuda
@pytest.mark.requires_assets("WHISPER_CKPT")
class TestFixedWindowResplit:
    """A span the detector could not divide, on a frontend that pads to a window.

    ``max_speech_s`` is pinned to ``window - 2 * speech_pad`` so every segment
    fits, but touching segments are merged (nothing is dropped between them), so
    a merged span can cross the window again and has to be re-split.  The width
    that re-split uses is the trap: splitting at the *full* window leaves the
    frontend no padding, and whisper-tiny then drops the tail of every window —
    +2.2 WER on long-form audio, all of it deletions at the seams.
    """

    def test_the_resplit_width_keeps_the_frontends_headroom(self):
        from oasr.engine import ASREngine, EngineConfig

        engine = ASREngine(
            EngineConfig(
                ckpt_dir=assets.require("WHISPER_CKPT"),
                device="cuda",
                dtype=torch.float16,
                service_mode="offline",
                max_batch_size=2,
                vad={"mode": "segment", "backend": "energy"},
            )
        )
        window_s = engine._config.feature_config.fixed_window_seconds
        assert window_s is not None, "this checkpoint has no fixed window to test"
        window = int(engine.sample_rate * window_s)
        assert 0 < engine._vad_resplit_samples < window
        assert engine._vad_resplit_samples == int(
            engine.sample_rate * engine._config.vad.max_speech_s
        )


class TestSpanBoundEveryEncoder:
    """D7 (2026-11-09 report): only fixed-window frontends had a span bound, so
    a Zipformer was handed a 900 s chapter the detector read as one run."""

    def test_the_default_cap_is_no_longer_than_the_probed_input(self):
        """The engine's activation probe runs ``max_batch_size`` rows of
        ``PROBE_AUDIO_SECONDS`` for a frontend without a window.  A default cap
        above it would let a full batch of spans exceed the shape it measured."""
        from oasr.engine.memory import PROBE_AUDIO_SECONDS

        resolved = VadConfig(mode="segment", backend="energy").resolve("offline")
        assert resolved.max_speech_s is not None
        assert resolved.max_speech_s <= PROBE_AUDIO_SECONDS

    @pytest.mark.cuda
    def test_an_encoder_without_a_window_gets_the_same_bound(self, ckpt_dir):
        from oasr.engine import ASREngine, EngineConfig

        engine = ASREngine(
            EngineConfig(
                ckpt_dir=ckpt_dir,
                device="cuda",
                dtype=torch.float16,
                service_mode="offline",
                max_batch_size=2,
                vad={"mode": "segment", "backend": "energy"},
            )
        )
        assert engine._config.feature_config.fixed_window_seconds is None
        cap = engine._config.vad.resolve("offline").max_speech_s
        assert engine._vad_resplit_samples == int(engine.sample_rate * cap) > 0
        assert engine._vad_splitter.max_span_samples == engine._vad_resplit_samples


@pytest.mark.cuda
class TestEngineResolution:
    """What the engine refuses, and when it refuses it."""

    def config(self, ckpt_dir, **kw):
        from oasr.engine import EngineConfig

        base = {
            "ckpt_dir": ckpt_dir,
            "service_mode": "offline",
            "max_batch_size": 2,
            "dtype": torch.float16,
        }
        base.update(kw)
        return EngineConfig(**base)

    def test_auto_resolves_to_the_familys_own_detector(self, ckpt_dir):
        from oasr.engine import ASREngine

        cfg = self.config(ckpt_dir, vad={"mode": "observe"})
        ASREngine(cfg)
        assert cfg.vad.backend == "ctc_blank"

    def test_segment_mode_refuses_an_asr_derived_detector(self, ckpt_dir):
        """It reads what the encoder produced, so it cannot cut audio before the
        encoder sees it.  Degrading to one whole-file segment would be
        indistinguishable, to a client, from audio that really was continuous."""
        from oasr.engine import ASREngine

        cfg = self.config(ckpt_dir, vad={"mode": "segment", "backend": "ctc_blank"})
        with pytest.raises(ValueError, match="cannot decide what the encoder sees"):
            ASREngine(cfg)

    def test_endpoint_mode_is_refused_offline(self, ckpt_dir):
        from oasr.engine import ASREngine

        cfg = self.config(ckpt_dir, vad={"mode": "endpoint"})
        with pytest.raises(ValueError, match="streaming control"):
            ASREngine(cfg)

    def test_a_mismatched_asr_backend_is_refused(self, ckpt_dir):
        from oasr.engine import ASREngine

        cfg = self.config(ckpt_dir, vad={"mode": "observe", "backend": "cif_alpha"})
        with pytest.raises(ValueError, match="does not produce"):
            ASREngine(cfg)

    def test_a_peaky_detector_raises_the_minimum_silence(self, ckpt_dir):
        """The `turn` preset's 100 ms cannot be resolved from a CTC trace whose
        in-word blank runs reach 840 ms."""
        from oasr.engine import ASREngine

        cfg = self.config(
            ckpt_dir, service_mode="streaming", max_num_blocks=256, vad={"mode": "endpoint"}
        )
        ASREngine(cfg)
        assert cfg.vad.min_silence_ms == get_vad_spec("ctc_blank").min_silence_floor_ms

    def test_streaming_segment_mode_refuses_the_family_detector(self, ckpt_dir):
        """ "auto" resolves to the family's own signal, which is produced by the
        stage it would have to precede.  The refusal names the alternative."""
        from oasr.engine import ASREngine

        cfg = self.config(
            ckpt_dir, service_mode="streaming", max_num_blocks=256, vad={"mode": "segment"}
        )
        with pytest.raises(ValueError, match="cannot decide what the encoder sees"):
            ASREngine(cfg)

    def test_streaming_segment_mode_accepts_a_waveform_detector(self, ckpt_dir):
        """The P4 capability: a detector that runs ahead of the encoder can gate it."""
        from oasr.engine import ASREngine

        cfg = self.config(
            ckpt_dir,
            service_mode="streaming",
            max_num_blocks=256,
            vad={"mode": "segment", "backend": "energy"},
        )
        engine = ASREngine(cfg)
        try:
            stage = engine._vad_stage
            assert stage is not None and stage.needs_audio and stage.gates_encoder
            # The segmentation preset, not the turn-taking one: this is the mode
            # that drops audio, so it wants faster-whisper's padding.
            assert cfg.vad.resolve("streaming").preset == "segment"
            # And on the host, where the audio already is.
            assert stage._device.type == "cpu"
        finally:
            engine.shutdown()

    def test_streaming_segment_mode_refuses_overlapped_partial_readback(self, ckpt_dir):
        """A partial issued against the closed turn would be collected against
        the next one, under the same stream id."""
        from oasr.engine import ASREngine

        cfg = self.config(
            ckpt_dir,
            service_mode="streaming",
            max_num_blocks=256,
            overlap_partial_readback=True,
            vad={"mode": "segment", "backend": "energy"},
        )
        with pytest.raises(ValueError, match="overlap_partial_readback"):
            ASREngine(cfg)

    def test_a_waveform_detector_offline_without_segment_mode_is_refused(self, ckpt_dir):
        """The combination that would otherwise resolve cleanly and then do
        nothing: the post-hoc path reads the *decode family's* signal, so a
        waveform detector configured for `observe` never runs."""
        from oasr.engine import ASREngine

        cfg = self.config(ckpt_dir, vad={"mode": "observe", "backend": "energy"})
        with pytest.raises(ValueError, match="only runs one for vad.mode='segment'"):
            ASREngine(cfg)

    def test_a_request_asking_for_vad_on_an_engine_without_it_is_refused(self, ckpt_dir):
        from oasr.engine import ASREngine
        from oasr.engine.request import DecodingOptions

        engine = ASREngine(self.config(ckpt_dir))
        with pytest.raises(ValueError, match="switched off"):
            engine.add_request(
                torch.zeros(SR), streaming=False, decoding=DecodingOptions(vad_events=True)
            )

    def test_single_utterance_is_refused_for_an_offline_request(self, ckpt_dir):
        from oasr.engine import ASREngine
        from oasr.engine.request import DecodingOptions

        engine = ASREngine(self.config(ckpt_dir, vad={"mode": "observe"}))
        with pytest.raises(ValueError, match="streaming control"):
            engine.add_request(
                torch.zeros(SR), streaming=False, decoding=DecodingOptions(single_utterance=True)
            )


@pytest.mark.cuda
class TestVadOffChangesNothing:
    """The negative control the whole axis rests on.

    Every other test here checks that a feature works.  This one checks that
    turning it off leaves the engine exactly as it was — which is the property
    that makes the axis safe to merge at all.
    """

    def transcripts(self, ckpt_dir, wav_paths, vad):
        import soundfile as sf

        from oasr.engine import ASREngine, EngineConfig

        engine = ASREngine(
            EngineConfig(
                ckpt_dir=ckpt_dir,
                service_mode="offline",
                max_batch_size=4,
                dtype=torch.float16,
                vad=vad,
            )
        )
        for path in wav_paths:
            data, _sr = sf.read(str(path), dtype="float32")
            engine.add_request(torch.from_numpy(data), streaming=False)
        outs = engine.run()
        return {o.request_id: o.text for o in outs}

    def test_transcripts_are_byte_identical_with_vad_off(self, ckpt_dir, wav_dir):
        import pathlib

        # Four when there are four, but two are enough: counted against WAV_DIR.
        assets.require_wavs(2)
        wavs = sorted(pathlib.Path(wav_dir).glob("*.wav"))[:4]
        # `None` is what every engine had before the axis existed; `mode="off"`
        # is the explicit spelling of the same thing.  They must agree with each
        # other *and* with the observing engine's transcripts, since observing
        # must not change what is decoded.
        base = self.transcripts(ckpt_dir, wavs, None)
        off = self.transcripts(ckpt_dir, wavs, {"mode": "off"})
        observing = self.transcripts(ckpt_dir, wavs, {"mode": "observe"})
        assert sorted(base.values()) == sorted(off.values())
        assert sorted(base.values()) == sorted(observing.values())

    def test_no_segments_are_attached_unless_the_request_asked(self, ckpt_dir, wav_dir):
        import soundfile as sf

        from oasr.engine import ASREngine, EngineConfig

        wav = assets.require_wavs(1)[0]
        data, _sr = sf.read(str(wav), dtype="float32")
        engine = ASREngine(
            EngineConfig(
                ckpt_dir=ckpt_dir,
                service_mode="offline",
                max_batch_size=2,
                dtype=torch.float16,
                vad={"mode": "observe"},
            )
        )
        engine.add_request(torch.from_numpy(data), streaming=False)
        out = engine.run()[0]
        # `None`, never `[]`: an empty list reads as "this audio had no speech".
        assert out.segments is None
