# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Word-level timestamps and confidences, and the C++ pass under them (H7).

Two halves of one feature, and AGENTS.md requires a rule change to land in
both: the per-family aligners and the shared frames-to-words pass
(``oasr/engine/decode/{alignment,ctc_align,attention_align}.py``), and the
randomized oracle that keeps the C++ implementation and its Python spec
byte-identical.  Splitting them across two files is how the two drift.

Four properties are worth pinning, and they are what these tests are organised
around:

1. **Words are cut out of the transcript.**  Every emitted ``word`` is a literal
   substring of ``RequestOutput.text``, in order and non-overlapping.  That is
   what a caption renderer, a redaction pass and a subtitle muxer all depend on,
   and reassembling words from tokenizer pieces breaks it for every kind in the
   tree.
2. **The CTC times come from the beam and are right.**  The kernel records the
   frame it emitted each token at; those frames are checked against the
   forced-alignment oracle in ``ctc_align.py``, which is itself checked
   bit-for-bit against ``torchaudio.functional.forced_align`` — an external
   reference rather than a self-consistency check.
3. **A family that cannot align says so.**  The rejection is per (family, mode),
   at admission, and names which modes do work.
4. **Nothing is paid for unless it is asked for.**  The alignment does not run,
   and the fields stay absent, without ``word_timestamps``.
"""

from __future__ import annotations

import random
import re
from bisect import bisect_right

import numpy as np
import pytest
import torch

from oasr.engine.decode.alignment import (
    _CPP as _ALIGNMENT,
    AlignmentFields,
    FrameClock,
    TokenAlignment,
    WordTiming,
    alignment_fields,
    emission_fields,
    wants_word_timings,
    word_timings,
)
from oasr.engine.decode.attention_align import dtw, resolve_alignment_heads, token_frame_spans
from oasr.engine.decode.ctc_align import align_hypotheses
from oasr.engine.request import DecodingOptions, Request

BLANK = 0

#: The grouping and the emission tiling are C++ only — there is no Python
#: implementation to fall back to, so the classes that exercise them need the
#: built extension.  Everything else in this file (the clock, the CTC oracle,
#: the DTW, the per-family contract) is pure Python and runs anywhere.
requires_cpp = pytest.mark.skipif(
    _ALIGNMENT is None,
    reason="oasr._C.alignment is not built (test-cpu.yml builds no extension)",
)


class _Detok:
    """A tokenizer stand-in: id → the text that token contributes.

    The real grouping only ever calls ``token_pieces``, whose contract is that
    the pieces concatenate to ``decode(all_ids)`` — so a table of pieces is a
    faithful stand-in for every kind in the tree, and the tests can express
    sentencepiece's ``▁``, byte-BPE's ``Ġ`` and FunASR's CJK joining as the
    strings each actually produces.  ``detokenize_incremental`` is kept
    alongside it, equivalent by construction, because a real tokenizer's
    default ``token_pieces`` *is* that method driven one id at a time.
    """

    def __init__(self, table):
        self.table = table

    def new_state(self):
        return {"ids": [], "text": ""}

    def detokenize_incremental(self, ids, state):
        out = "".join(self.table.get(int(i), "") for i in ids)
        state["text"] += out
        return out

    def token_pieces(self, ids):
        return [self.table.get(int(i), "") for i in ids]


def _align(tokens, detok, spf=0.04, conf=1.0):
    al = [TokenAlignment(t, i, i + 1, conf) for i, t in enumerate(tokens)]
    return word_timings(al, detok, FrameClock(spf)), al


# ---------------------------------------------------------------------------
# 1. Grouping
# ---------------------------------------------------------------------------


@requires_cpp
class TestWordGrouping:
    def test_pieces_join_into_words_at_whitespace(self):
        detok = _Detok({1: "HE", 2: "LLO", 3: " WOR", 4: "LD"})
        words, _ = _align([1, 2, 3, 4], detok)
        assert [w.word for w in words] == ["HELLO", "WORLD"]
        assert (words[0].start, words[0].end) == (0.0, pytest.approx(0.08))
        assert (words[1].start, words[1].end) == (pytest.approx(0.08), pytest.approx(0.16))

    def test_every_word_is_a_substring_of_the_transcript_in_order(self):
        """The load-bearing invariant — see the module docstring."""
        detok = _Detok({1: "the", 2: " quick", 3: " brown", 4: " fox"})
        words, _ = _align([1, 2, 3, 4], detok)
        text = "the quick brown fox"
        pos = 0
        for w in words:
            idx = text.find(w.word, pos)
            assert idx >= 0, f"{w.word!r} is not a substring at/after {pos}"
            pos = idx + len(w.word)

    def test_spaceless_scripts_split_per_character(self):
        """CJK has no whitespace, so whitespace splitting alone returns a clause."""
        detok = _Detok({10: "今", 11: "天", 12: "world", 13: "好"})
        words, _ = _align([10, 11, 12, 13], detok)
        assert [w.word for w in words] == ["今", "天", "world", "好"]

    def test_tokens_that_render_to_nothing_own_no_word(self):
        """Special ids and absorbed byte-BPE fragments decode to ``""``."""
        detok = _Detok({0: "", 1: "hi", 2: " there", 3: ""})
        words, _ = _align([0, 1, 2, 3], detok)
        assert [w.word for w in words] == ["hi", "there"]
        assert words[0].start == pytest.approx(0.04), "the empty leading token owns no time"

    def test_word_confidence_is_the_mean_over_its_tokens(self):
        detok = _Detok({1: "HE", 2: "LLO"})
        al = [TokenAlignment(1, 0, 1, 0.4), TokenAlignment(2, 1, 2, 0.8)]
        (word,) = word_timings(al, detok, FrameClock(0.04))
        assert word.confidence == pytest.approx(0.6)
        # The utterance score is the same aggregation over every token: the mean
        # and not the product, which would decay geometrically with length and
        # rank a long correct transcript below a short uncertain one.
        assert alignment_fields(al, detok, FrameClock(0.04)).confidence == pytest.approx(0.6)

    def test_no_clock_means_no_timings_rather_than_zeros(self):
        """A guessed frame rate produces plausible spans that are all wrong."""
        detok = _Detok({1: "hi"})
        assert word_timings([TokenAlignment(1, 0, 1, 1.0)], detok, None) == []

    def test_offset_shifts_every_span(self):
        detok = _Detok({1: "hi"})
        (w,) = word_timings([TokenAlignment(1, 0, 1, 1.0)], detok, FrameClock(0.04), offset=2.5)
        assert (w.start, w.end) == (pytest.approx(2.5), pytest.approx(2.54))


class TestFrameClock:
    @pytest.mark.parametrize(
        "hop_ms,lfr_n,subsampling,want",
        [
            (10.0, 1, 4, 0.04),  # Conformer / Zipformer
            (10.0, 1, 2, 0.02),  # Whisper
            (10.0, 6, 1, 0.06),  # Paraformer (LFR in the frontend)
            (10.0, 1, 8, 0.08),  # Nemotron FastConformer
        ],
    )
    def test_resolves_from_the_declared_geometry(self, hop_ms, lfr_n, subsampling, want):
        from types import SimpleNamespace

        fcfg = SimpleNamespace(frame_shift_ms=hop_ms, lfr_n=lfr_n)
        model = SimpleNamespace(encoder=SimpleNamespace(subsampling_rate=subsampling))
        clock = FrameClock.resolve(fcfg, model)
        assert clock is not None and clock.seconds_per_frame == pytest.approx(want)

    def test_unresolvable_geometry_returns_none(self):
        assert FrameClock.resolve(None, object()) is None
        assert FrameClock.resolve(object(), None) is None


# ---------------------------------------------------------------------------
# 2. CTC forced alignment
# ---------------------------------------------------------------------------


def _log_probs(rng, b, t, v):
    x = torch.tensor(rng.normal(size=(b, t, v)), dtype=torch.float32)
    return torch.log_softmax(x, dim=-1)


class TestCtcForcedAlignment:
    def test_matches_torchaudio_forced_align(self):
        """The external oracle: same algorithm, same spans, over random inputs.

        ``torchaudio.functional.forced_align`` returns a frame-level label
        sequence; grouping its non-blank runs gives the same per-token spans
        this module reports, and any disagreement is a real difference in the
        CTC transition rules rather than a formatting one.
        """
        ta = pytest.importorskip("torchaudio.functional")
        rng = np.random.default_rng(0)
        vocab = 12
        for trial in range(25):
            t_len = int(rng.integers(20, 80))
            n = int(rng.integers(2, min(10, t_len // 3)))
            hyp = [int(rng.integers(1, vocab)) for _ in range(n)]
            lp = _log_probs(rng, 1, t_len, vocab)

            ours = align_hypotheses(lp, torch.tensor([t_len]), [hyp], BLANK)[0]
            assert ours is not None and len(ours) == n

            ali, _ = ta.forced_align(lp, torch.tensor([hyp], dtype=torch.int32), blank=BLANK)
            runs, prev_end = [], -1
            for step, label in enumerate(ali[0].tolist()):
                if label == BLANK:
                    continue
                if runs and step == prev_end and label == runs[-1][0]:
                    runs[-1] = (label, runs[-1][1], step + 1)
                else:
                    runs.append((label, step, step + 1))
                prev_end = step + 1
            assert len(runs) == n, f"trial {trial}: reference grouping disagrees"
            for k, (a, r) in enumerate(zip(ours, runs)):
                assert (int(a.start_frame), int(a.end_frame)) == (
                    r[1],
                    r[2],
                ), f"trial {trial} token {k}"

    def test_batched_matches_per_row_with_mixed_lengths(self):
        """Padded short rows must freeze rather than keep accumulating."""
        rng = np.random.default_rng(7)
        lp = _log_probs(rng, 4, 60, 10)
        lens = torch.tensor([60, 41, 25, 13])
        hyps = [[1, 2, 3, 4, 5], [6, 7], [3, 3, 4], [9]]
        batched = align_hypotheses(lp, lens, hyps, BLANK)
        for b in range(4):
            solo = align_hypotheses(lp[b : b + 1, : lens[b]], lens[b : b + 1], [hyps[b]], BLANK)[0]
            assert batched[b] == solo, f"row {b} differs when batched"

    def test_adjacent_repeats_need_a_blank_between_them(self):
        """The one CTC-specific transition rule, tested where it is decisive.

        ``[3, 3]`` needs **three** frames — ``3, blank, 3`` — because a CTC path
        collapses two adjacent identical labels into one.  In two frames it is
        infeasible, and in three there is exactly one path.  Random log-probs do
        not pin this: dropping the rule only changes the answer when the illegal
        path happens to score higher, so a probabilistic test passes against a
        decoder that has forgotten it (this one did, once).
        """
        rng = np.random.default_rng(3)
        assert (
            align_hypotheses(_log_probs(rng, 1, 2, 8), torch.tensor([2]), [[3, 3]], BLANK)[0]
            is None
        ), "two frames cannot spell a repeated label"

        (row,) = align_hypotheses(_log_probs(rng, 1, 3, 8), torch.tensor([3]), [[3, 3]], BLANK)
        assert row is not None
        assert (row[0].start_frame, row[0].end_frame) == (0.0, 1.0)
        assert (row[1].start_frame, row[1].end_frame) == (2.0, 3.0), "frame 1 must be the blank"

        # A *different* neighbour may share a frame boundary — the rule is about
        # equality, not adjacency.
        (row,) = align_hypotheses(_log_probs(rng, 1, 2, 8), torch.tensor([2]), [[3, 4]], BLANK)
        assert row is not None and len(row) == 2

    def test_a_hypothesis_the_audio_cannot_express_returns_none(self):
        """A partially-timed list is worse than none: the caller cannot tell."""
        rng = np.random.default_rng(11)
        lp = _log_probs(rng, 1, 3, 8)
        assert align_hypotheses(lp, torch.tensor([3]), [[1, 1, 1, 1, 1]], BLANK)[0] is None

    def test_an_empty_hypothesis_aligns_to_nothing(self):
        rng = np.random.default_rng(13)
        lp = _log_probs(rng, 1, 5, 8)
        assert align_hypotheses(lp, torch.tensor([5]), [[]], BLANK)[0] == []

    def test_spans_are_monotone_and_confidences_are_probabilities(self):
        rng = np.random.default_rng(5)
        lp = _log_probs(rng, 2, 50, 9)
        rows = align_hypotheses(lp, torch.tensor([50, 50]), [[1, 2, 3], [4, 5]], BLANK)
        for row in rows:
            assert row is not None
            assert all(row[i].end_frame <= row[i + 1].start_frame for i in range(len(row) - 1))
            assert all(a.end_frame > a.start_frame for a in row)
            assert all(0.0 <= a.confidence <= 1.0 for a in row)

    @pytest.mark.cuda
    def test_cuda_and_cpu_agree(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        rng = np.random.default_rng(17)
        lp = _log_probs(rng, 2, 40, 10)
        lens = torch.tensor([40, 31])
        hyps = [[1, 2, 3], [4, 5]]
        assert align_hypotheses(lp, lens, hyps, BLANK) == align_hypotheses(
            lp.cuda(), lens.cuda(), hyps, BLANK
        )


# ---------------------------------------------------------------------------
# 2b. The beam's own emission frames (the production CTC path)
# ---------------------------------------------------------------------------


def _peaky(fires, t_len, vocab, blank=BLANK):
    """Log-probs where ``{frame: token}`` each fire hard and blank wins elsewhere."""
    logits = np.full((1, t_len, vocab), -8.0, dtype=np.float32)
    logits[0, :, blank] = 4.0
    for t, c in fires.items():
        logits[0, t, c] = 6.0
        logits[0, t, blank] = -4.0
    return torch.log_softmax(torch.from_numpy(logits), dim=-1)


@pytest.mark.cuda
class TestKernelEmissionFrames:
    """The CTC beam records *when* it emitted each token, so timing is a read.

    These need a GPU because the recording happens in the decoder kernel — that
    is the whole point: nothing here re-derives the frame on the host.
    """

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

    @pytest.mark.parametrize("paged", [False, True])
    def test_recorded_frames_are_the_frames_the_labels_fired_at(self, paged):
        from oasr.functionals.ctc_decode import ctc_beam_search_decode

        # 12 and 14 fire the same label with a blank between them: two tokens.
        fires = {5: 3, 12: 7, 14: 7, 25: 4, 33: 9}
        lp = _peaky(fires, 40, 10).cuda()
        r = ctc_beam_search_decode(
            lp,
            torch.tensor([40], device="cuda"),
            beam_size=5,
            blank_id=BLANK,
            use_paged_memory=paged,
            page_size=16,
            want_times=True,
        )
        assert r.tokens[0][0] == [3, 7, 7, 4, 9]
        assert r.times[0][0] == sorted(fires)

    def test_times_are_not_copied_back_unless_asked(self):
        """The frames are always recorded; the device→host copy is the opt-in."""
        from oasr.functionals.ctc_decode import ctc_beam_search_decode

        lp = _peaky({5: 3, 20: 4}, 30, 10).cuda()
        args = {"beam_size": 5, "blank_id": BLANK}
        plain = ctc_beam_search_decode(lp, torch.tensor([30], device="cuda"), **args)
        timed = ctc_beam_search_decode(
            lp, torch.tensor([30], device="cuda"), want_times=True, **args
        )
        assert plain.times == []
        assert timed.times and timed.tokens == plain.tokens

    @pytest.mark.parametrize("fused", [True, False])
    @pytest.mark.parametrize("paged", [False, True])
    def test_streaming_frames_are_stream_absolute_past_the_ring(self, paged, fused, monkeypatch):
        """``select_seqs`` is a ring of width ``max_seq_len``; a long stream
        decodes more frames than its output-token cap, so a recorded time read
        back from that ring would silently wrap.  Frame 28 with a cap of 8 is
        the case that catches it.

        Parametrised over ``OASR_CTC_FUSED`` because the two step
        implementations learn the frame differently — the fused one carries it
        through the chunk loop, the legacy one has to be handed the device
        counter — so only the fused path being right is not the property.
        """
        from oasr.functionals.ctc_decode import GpuDecoderConfig, GpuStreamingDecoder

        monkeypatch.setenv("OASR_CTC_FUSED", "1" if fused else "0")
        cfg = GpuDecoderConfig(
            beam_size=5, blank_id=BLANK, max_seq_len=8, use_paged_memory=paged, page_size=16
        )
        dec = GpuStreamingDecoder(cfg, use_cuda_graphs=False)
        state = dec.create_state(batch=1, vocab_size=10, device=torch.device("cuda"))
        want = []
        for chunk in range(3):
            t = 2 + chunk * 3
            dec.decode_chunk(_peaky({t: 1 + chunk}, 10, 10).cuda(), state=state)
            want.append(chunk * 10 + t)
        r = dec.finalize_stream(state=state, want_times=True)
        assert r.tokens[0][0] == [1, 2, 3]
        assert r.times[0][0] == want

    def test_the_frames_agree_with_the_forced_alignment_oracle(self):
        """Two independent answers to the same question.

        ``ctc_align.forced_align`` is checked against ``torchaudio`` above; this
        checks the *kernel's* frames against it, so the production path inherits
        that external reference without running a DP per request.
        """
        from oasr.functionals.ctc_decode import ctc_beam_search_decode

        rng = np.random.default_rng(4)
        for _ in range(8):
            t_len = int(rng.integers(30, 90))
            fires = {}
            t = int(rng.integers(2, 6))
            while t < t_len - 2 and len(fires) < 8:
                fires[t] = int(rng.integers(1, 10))
                t += int(rng.integers(3, 8))
            lp = _peaky(fires, t_len, 10).cuda()
            r = ctc_beam_search_decode(
                lp,
                torch.tensor([t_len], device="cuda"),
                beam_size=5,
                blank_id=BLANK,
                want_times=True,
            )
            toks, frames = r.tokens[0][0], r.times[0][0]
            aligned = align_hypotheses(lp, torch.tensor([t_len], device="cuda"), [toks], BLANK)[0]
            assert aligned is not None
            # The beam commits at the leading edge of a peak, so it is on the
            # oracle's frame or at most one before it — never later.
            for frame, span in zip(frames, aligned):
                assert 0 <= span.start_frame - frame <= 1, (frames, aligned)


# ---------------------------------------------------------------------------
# 3. Cross-attention DTW (AED)
# ---------------------------------------------------------------------------


class TestAttentionDtw:
    def test_recovers_a_known_diagonal_alignment(self):
        n_tok, n_frames = 6, 24
        w = np.zeros((3, n_tok, n_frames))
        for k in range(n_tok):
            w[:, k, 4 * k : 4 * k + 4] = 1.0
        w += np.random.default_rng(0).normal(0, 0.02, w.shape)
        spans = token_frame_spans(w, num_frames=n_frames, medfilt_width=3)
        assert spans == [(4 * k, 4 * k + 4) for k in range(n_tok)]

    def test_the_padded_window_is_excluded(self):
        """Whisper pads to 30 s; the DTW must not walk into the silence."""
        n_tok, n_frames = 4, 16
        w = np.zeros((2, n_tok, n_frames))
        for k in range(n_tok):
            w[:, k, 4 * k : 4 * k + 4] = 1.0
        padded = np.concatenate([w, np.zeros((2, n_tok, 200))], axis=2)
        assert token_frame_spans(padded, num_frames=n_frames, medfilt_width=3) == token_frame_spans(
            w, num_frames=n_frames, medfilt_width=3
        )

    def test_the_path_is_monotone(self):
        rng = np.random.default_rng(1)
        w = rng.random((4, 12, 60))
        spans = token_frame_spans(w, num_frames=60)
        assert spans is not None
        assert all(spans[i][0] <= spans[i + 1][0] for i in range(len(spans) - 1))

    def test_dtw_follows_the_cheap_diagonal(self):
        cost = np.full((3, 3), 5.0)
        np.fill_diagonal(cost, 0.0)
        text_idx, time_idx = dtw(cost)
        assert list(text_idx) == [0, 1, 2] and list(time_idx) == [0, 1, 2]

    def test_declared_heads_win_and_bad_pairs_are_dropped(self):
        pairs, declared = resolve_alignment_heads([[2, 1], [3, 0], [99, 0]], 4, 4)
        assert declared and pairs == [(2, 1), (3, 0)]

    def test_the_fallback_is_the_upper_half_of_the_stack(self):
        """Averaging every layer makes the matrix diagonal, not aligned."""
        pairs, declared = resolve_alignment_heads(None, 4, 2)
        assert not declared
        assert pairs == [(2, 0), (2, 1), (3, 0), (3, 1)]

    def test_nothing_to_align_returns_none(self):
        assert token_frame_spans(np.zeros((2, 0, 10)), num_frames=10) is None
        assert token_frame_spans(np.zeros((2, 3, 1)), num_frames=1) is None


# ---------------------------------------------------------------------------
# 4. Transducer emission spans
# ---------------------------------------------------------------------------


@requires_cpp
class TestEmissionSpans:
    """Shared by both frame-synchronous families — CTC and the transducer.

    Read through :func:`emission_fields` with a one-second clock, so a published
    timestamp *is* the frame index.  The tiling rule itself is
    ``oasr::alignment::emission_spans``; this is the shape it has to produce.
    """

    @staticmethod
    def _spans(frames, confidences=None, frame_offset=0):
        return emission_fields(
            list(range(len(frames))),
            frames,
            confidences,
            _Detok({}),
            FrameClock(1.0),
            frame_offset=frame_offset,
            want_words=False,
        ).timestamps

    def test_tokens_own_the_frames_since_the_previous_decision(self):
        spans = self._spans([3, 10], [0.9, 0.7])
        assert spans[0] == (3.0, 4.0), "no leading-silence attribution"
        assert spans[1] == (4.0, 11.0), "spans tile without gaps"

    def test_several_emissions_at_one_frame_all_report_it(self):
        assert self._spans([3, 3, 3], [0.9, 0.8, 0.7]) == [(3.0, 4.0)] * 3

    def test_offset_rebases_a_streaming_chunk(self):
        assert self._spans([3], [0.5], frame_offset=100) == [(103.0, 104.0)]


# ---------------------------------------------------------------------------
# 5. The per-family, per-mode contract
# ---------------------------------------------------------------------------


class _Strategy:
    """The real ``validate_options`` over a declared mode set.

    Copied members rather than a stub: the point under test *is* the base
    implementation, and a stub that never rejects would test nothing.
    """

    from oasr.engine.decode.base import DecodeStrategy as _Base

    decode_type = "ctc"
    selective_options = ()
    # No engine config, so voice activity is off and the turn-aware checks are
    # inert — this stub is here for the alignment half of ``validate_options``.
    _config = None
    _SELECTIVE_UNSET = _Base._SELECTIVE_UNSET
    validate_options = _Base.validate_options
    _require_word_timings = _Base._require_word_timings
    _reject_nbest_across_turns = _Base._reject_nbest_across_turns

    def __init__(self, modes=(), clock=FrameClock(0.04)):
        self.word_timing_modes = modes
        self._clock = clock


class TestWordTimestampContract:
    def test_a_family_with_no_alignment_refuses_and_names_the_gap(self):
        with pytest.raises(ValueError, match="cannot produce word timestamps"):
            _Strategy().validate_options(DecodingOptions(word_timestamps=True))

    def test_offline_only_families_refuse_streaming_and_name_what_works(self):
        s = _Strategy(modes=("offline",))
        s.validate_options(DecodingOptions(word_timestamps=True), streaming=False)
        with pytest.raises(ValueError, match=r"streaming request \(supported: offline\)"):
            s.validate_options(DecodingOptions(word_timestamps=True), streaming=True)

    def test_an_unresolved_frame_rate_refuses_rather_than_guessing(self):
        s = _Strategy(modes=("offline",), clock=None)
        with pytest.raises(ValueError, match="encoder frame rate"):
            s.validate_options(DecodingOptions(word_timestamps=True))

    def test_not_asking_is_never_rejected(self):
        """``False`` is the unset value; a hardcoded ``is None`` test accepts it
        everywhere, which is how the same class of bug got shipped in H5."""
        _Strategy().validate_options(DecodingOptions())
        _Strategy().validate_options(DecodingOptions(word_timestamps=False))
        _Strategy().validate_options(None)

    def test_wants_word_timings_reads_through_an_absent_options_object(self):
        assert not wants_word_timings(Request(request_id="a"))
        assert not wants_word_timings(Request(request_id="b", decoding=DecodingOptions()))
        assert wants_word_timings(
            Request(request_id="c", decoding=DecodingOptions(word_timestamps=True))
        )


class TestRegisteredFamiliesDeclareTheirModes:
    """Every registered family answers the question, and answers it plausibly."""

    def _families(self):
        import oasr.engine.decode  # noqa: F401  (registers the built-ins)
        from oasr.engine.decode.base import _REGISTRY

        return sorted(_REGISTRY.items())

    def test_the_declaration_is_a_property_so_it_can_depend_on_the_config(self):
        """A class attribute cannot say "greedy yes, beam no" for one family."""
        for name, cls in self._families():
            assert isinstance(
                cls.word_timing_modes, property
            ), f"{name} declares word_timing_modes as a class attribute"

    def test_declared_modes_are_a_subset_of_the_two_that_exist(self):
        from oasr.engine.config import EngineConfig
        from oasr.engine.decode.base import DecodeStrategy

        del EngineConfig  # only the base default is reachable without a model
        assert set(DecodeStrategy.word_timing_modes.fget(object())) == set()

    def test_ctc_times_both_modes_because_the_beam_records_the_frame(self):
        """The beam writes the emission frame beside the token as it decodes, so
        the transcript carries its own timing — including for a stream, whose
        log-probs are long gone by the time it is final."""
        from oasr.engine.decode.ctc_gpu import CtcGpuDecodeStrategy

        modes = CtcGpuDecodeStrategy.word_timing_modes.fget(object())
        assert modes == ("offline", "streaming")

    @pytest.mark.parametrize("beam,want", [(1, ("offline", "streaming")), (4, ())])
    def test_a_transducer_under_beam_search_declares_nothing(self, beam, want):
        """The beam's ``(B, k, cap)`` buffer carries labels, not frames."""
        from types import SimpleNamespace

        from oasr.engine.config import EngineConfig
        from oasr.engine.decode.detokenize import Detokenizer
        from oasr.engine.decode.transducer import TransducerDecodeStrategy

        predictor = SimpleNamespace(
            label_window_state=True,
            init_state=lambda *a, **k: None,
            predict=lambda *a, **k: None,
            advance=lambda *a, **k: None,
        )
        model = SimpleNamespace(
            decoder=predictor,
            joiner=SimpleNamespace(encoder_proj=1, decoder_proj=1),
            encode_offline=1,
            blank_id=0,
            encoder=SimpleNamespace(subsampling_rate=4),
        )
        strat = TransducerDecodeStrategy(
            EngineConfig(ckpt_dir="x", decode_options={"beam_size": beam}), Detokenizer(), model
        )
        assert tuple(strat.word_timing_modes) == want

    @pytest.mark.parametrize(
        "beam,has_attn,want", [(1, True, ("offline",)), (4, True, ()), (1, False, ())]
    )
    def test_aed_declares_from_the_decoder_and_the_beam_width(self, beam, has_attn, want):
        """Two independent reasons an AED engine cannot time its output, and
        both have to reach the declaration: a decoder with no cross-attention
        surface, and beam search — whose group retains no encoder row to
        re-forward against and would silently produce nothing."""
        from types import SimpleNamespace

        from oasr.engine.config import EngineConfig
        from oasr.engine.decode.aed import AedDecodeStrategy
        from oasr.engine.decode.detokenize import Detokenizer

        decoder = SimpleNamespace(prefill=1, step=1, select=1)
        if has_attn:
            decoder.cross_attention = lambda *a, **k: None
        mcfg = SimpleNamespace(
            sot_sequence=lambda **k: [1, 2, 3],
            eos_token_id=2,
            suppress_tokens=[],
            begin_suppress_tokens=[],
            max_target_positions=100,
            decoder_layers=4,
            decoder_attention_heads=6,
            alignment_heads=[],
        )
        model = SimpleNamespace(
            config=mcfg,
            decoder=decoder,
            encode_offline=1,
            encoder=SimpleNamespace(subsampling_rate=2),
        )
        strat = AedDecodeStrategy(
            EngineConfig(ckpt_dir="x", decode_options={"beam_size": beam}), Detokenizer(), model
        )
        assert tuple(strat.word_timing_modes) == want


# ---------------------------------------------------------------------------
# 6. Nothing is paid for unless asked for
# ---------------------------------------------------------------------------


class TestOptInIsRespected:
    def test_the_facade_drops_the_requests_when_no_row_asked(self):
        """The strategy must see ``None`` — that is what keeps a batch of
        ordinary requests on exactly the path it had before H7."""
        from oasr.engine.output_processor import OutputProcessor

        seen = {}

        class _Spy:
            def decode_offline(self, lp, lengths, requests=None):
                seen["requests"] = requests
                return []

        op = OutputProcessor.__new__(OutputProcessor)
        op._strategy = _Spy()

        plain = [Request(request_id="a"), Request(request_id="b")]
        op.decode_offline(torch.zeros(2, 1, 1), torch.tensor([1, 1]), plain)
        assert seen["requests"] is None

        asked = plain + [
            Request(request_id="c", decoding=DecodingOptions(word_timestamps=True)),
        ]
        op.decode_offline(torch.zeros(3, 1, 1), torch.tensor([1, 1, 1]), asked)
        assert seen["requests"] is asked

    def test_longform_merge_shifts_words_into_file_time(self):
        from oasr.engine.decode.alignment import WordTiming
        from oasr.engine.longform import LongFormTracker
        from oasr.engine.request import RequestOutput

        tracker = LongFormTracker()
        tracker.register("parent", ["c0", "c1"], [0.0, 30.0])
        tracker.absorb(
            [
                RequestOutput(
                    request_id="c0",
                    text="hello",
                    tokens=[[1]],
                    finished=True,
                    words=[WordTiming("hello", 1.0, 1.5, 0.9)],
                    confidence=0.9,
                ),
            ]
        )
        (merged,) = tracker.absorb(
            [
                RequestOutput(
                    request_id="c1",
                    text="world",
                    tokens=[[2]],
                    finished=True,
                    words=[WordTiming("world", 2.0, 2.5, 0.7)],
                    confidence=0.7,
                ),
            ]
        )
        assert [(w.word, w.start) for w in merged.words] == [("hello", 1.0), ("world", 32.0)]
        assert merged.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 7. Against a real checkpoint
# ---------------------------------------------------------------------------


def _engine(ckpt, **kw):
    """``(engine, sample_rate)`` — the rate comes from the config the engine
    resolved against the checkpoint's FeatureSpec, which is the only rate it
    accepts (the engine does not resample; see C2)."""
    from oasr.engine import ASREngine, EngineConfig

    cfg = EngineConfig(ckpt_dir=ckpt, service_mode="offline", max_batch_size=4, **kw)
    engine = ASREngine(cfg)
    assert cfg.feature_config is not None
    return engine, cfg.feature_config.sample_rate


def _read(path, rate):
    sf = pytest.importorskip("soundfile")
    import numpy as np_

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np_.ascontiguousarray(data.mean(axis=1)))
    if sr != rate:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, rate)
    return wav, len(wav) / rate


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.requires_assets("CKPT_DIR", "AUDIO_PATH")
class TestRealCheckpoint:
    """The invariants a unit test structurally cannot check.

    Synthetic log-probs cannot say whether a word lands on the audio it names;
    only a trained model on real speech can, and that is the failure this whole
    feature would otherwise ship with — plausible numbers, uniformly wrong.
    """

    def test_words_land_on_the_audio_and_tile_the_transcript(self, ckpt_dir, audio_path):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        engine, rate = _engine(ckpt_dir)
        try:
            wav, duration = _read(audio_path, rate)
            (out,) = engine.transcribe_outputs(
                [wav], streaming=False, decoding=DecodingOptions(word_timestamps=True)
            )
        finally:
            engine.shutdown()

        assert out.words, "a real checkpoint produced no words"
        assert out.confidence is not None and 0.0 < out.confidence <= 1.0

        pos = 0
        for w in out.words:
            idx = out.text.find(w.word, pos)
            assert idx >= 0, f"{w.word!r} is not a substring of the transcript at/after {pos}"
            pos = idx + len(w.word)

        for a, b in zip(out.words, out.words[1:]):
            assert a.start <= b.start and a.end <= b.end, f"non-monotone: {a} then {b}"
        assert out.words[0].start >= 0.0
        assert out.words[-1].end <= duration + 0.2, "a word ends after the audio does"

    def test_streaming_and_offline_agree_to_within_a_frame(self, ckpt_dir, audio_path):
        """The capability recording the frames unlocked, and its own oracle.

        A streaming decode keeps no log-probs, so this cannot come from an
        alignment; it comes from the frames the beam wrote as it went.  Offline
        decoding of the same audio is the reference — the two run entirely
        different kernels (batched offline vs the chunked streaming step) over
        the same model, so agreement is a real check rather than a tautology.
        """
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        opts = DecodingOptions(word_timestamps=True)
        engine, rate = _engine(ckpt_dir)
        try:
            wav, duration = _read(audio_path, rate)
            (offline,) = engine.transcribe_outputs([wav], streaming=False, decoding=opts)
        finally:
            engine.shutdown()

        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=ckpt_dir, service_mode="streaming", max_batch_size=4)
        engine = ASREngine(cfg)
        try:
            (stream,) = engine.transcribe_outputs([wav], streaming=True, decoding=opts)
        finally:
            engine.shutdown()

        assert stream.words, "streaming produced no words"
        assert stream.words[-1].end <= duration + 0.2
        assert [w.word for w in stream.words] == [w.word for w in offline.words]
        # Positionally, never by word text — a transcript repeats words, and
        # keying by text compares the wrong occurrences.
        drift = [abs(a.start - b.start) for a, b in zip(offline.words, stream.words)]
        assert max(drift) <= 0.05, f"streaming/offline starts diverge: {drift}"

    def test_not_asking_costs_nothing_and_returns_nothing(self, ckpt_dir, audio_path):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        engine, rate = _engine(ckpt_dir)
        try:
            wav, _ = _read(audio_path, rate)
            (plain,) = engine.transcribe_outputs([wav], streaming=False)
            (timed,) = engine.transcribe_outputs(
                [wav], streaming=False, decoding=DecodingOptions(word_timestamps=True)
            )
        finally:
            engine.shutdown()
        assert plain.words is None and plain.timestamps is None
        # ...and asking must not change *what* was decoded, only what is
        # reported: the alignment reads the log-probs, it does not steer them.
        assert plain.text == timed.text


# ---------------------------------------------------------------------------
# The compiled alignment pass against an independent Python oracle
#
# The pass is C++ only and Python raises rather than falling back, so this
# oracle is the only thing that can see a rule change land in one and not the
# other.  Neither side may use ``std::isspace`` or ``sum()``: both differ
# across implementations and the difference reaches the published output.
# ---------------------------------------------------------------------------


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


class _OracleDetok:
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

        detok = _OracleDetok([])
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
            emission_fields([0, 1, 2], [0, 1], None, _OracleDetok([]), FrameClock(0.04))


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
            detok = _OracleDetok(pieces)
            got = word_timings(spans, detok, FrameClock(0.04), offset=0.5)
            want = _reference_fields(spans, pieces, 0.04, 0.5, True).words or []
            assert got == want
