# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-frame probabilities → speech segments and speech-activity events.

Pure policy: no torch, no model, no device.  That is the point of splitting it
out — every detector in the tree, from a log-energy baseline to a CTC blank
posterior to a neural VAD, produces the same trace and goes through this same
machine, so they share segment semantics, tuning knobs and event names rather
than each inventing their own.  It also means the half of the axis most likely
to have an off-by-one is testable against synthetic traces with no GPU, no
checkpoint and no asset gate.

The state machine is Silero's, which is also NeMo's ``binarization`` and
pyannote's ``Binarize`` under different names:

* enter speech at ``threshold``, leave it at ``neg_threshold`` — the gap is what
  stops a trace hovering at one level from chattering;
* a silence shorter than ``min_silence_ms`` does not end a run;
* a run shorter than ``min_speech_ms`` is dropped entirely;
* a run longer than ``max_speech_s`` is cut anyway;
* emitted segments are padded by ``speech_pad_ms`` on each side, and two
  segments whose padding would overlap meet in the middle of the gap instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

from .config import VadConfig

__all__ = [
    "SpeechSegment",
    "VadEvent",
    "SpeechSegmenter",
    "SPEECH_STARTED",
    "SPEECH_STOPPED",
]

#: Event names.  Deliberately OpenAI's, because the realtime surface emits them
#: verbatim as ``input_audio_buffer.speech_started`` / ``.speech_stopped`` and a
#: second internal vocabulary would only be a translation table to get wrong.
SPEECH_STARTED = "speech_started"
SPEECH_STOPPED = "speech_stopped"


@dataclass(frozen=True)
class SpeechSegment:
    """One detected span of speech, in **seconds of audio**, request-relative.

    The same time base as :class:`~oasr.engine.decode.alignment.WordTiming` and
    ``RequestOutput.timestamps``, so a segment and a word are directly
    comparable without a conversion — and never wall-clock, which would make
    every span depend on transmission jitter.
    """

    start: float
    end: float
    speech_prob: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class VadEvent:
    """A speech-activity transition, in the same time base as a segment."""

    kind: str
    time: float


class SpeechSegmenter:
    """Hysteresis over a probability sequence.

    Drives both flows from one state machine: :meth:`push` for streaming (returns
    events as they become decidable) and :meth:`run` for a whole utterance
    (returns segments).

    Parameters
    ----------
    config : VadConfig
        Must already be **resolved** (:meth:`VadConfig.resolve`), so every
        optional knob carries a number.  Passing an unresolved config raises
        rather than silently substituting a default that the service mode should
        have chosen.
    seconds_per_frame : float
        The detector's frame rate.  Every span this class emits is a frame index
        multiplied by this, so a wrong value scales every boundary by a constant
        — plausible-looking and uniformly wrong.
    time_offset : float
        Added to every emitted time.  This is the reporting clock: after a
        streaming turn reset the model's frame counter restarts at zero while
        this keeps accumulating, so timestamps stay session-relative.
    """

    def __init__(
        self,
        config: VadConfig,
        seconds_per_frame: float,
        *,
        time_offset: float = 0.0,
    ) -> None:
        if config.threshold is None or config.neg_threshold is None:
            raise ValueError(
                "SpeechSegmenter needs a resolved VadConfig; call "
                "VadConfig.resolve(service_mode) first so the preset is applied"
            )
        if seconds_per_frame <= 0.0:
            raise ValueError(f"seconds_per_frame must be > 0, got {seconds_per_frame!r}")
        self._cfg = config
        self._spf = float(seconds_per_frame)
        self._offset = float(time_offset)

        self._threshold = float(config.threshold)
        self._neg_threshold = float(config.neg_threshold)
        self._min_speech = self._frames(config.min_speech_ms)
        self._min_silence = self._frames(config.min_silence_ms)
        self._pad = float(config.speech_pad_ms or 0) / 1000.0
        self._max_speech = (
            None
            if config.max_speech_s is None
            else max(1, int(round(float(config.max_speech_s) / self._spf)))
        )
        self.reset()

    def _frames(self, milliseconds: Optional[int]) -> int:
        """Milliseconds → whole frames, rounded to nearest, never negative."""
        return max(0, int(round(float(milliseconds or 0) / 1000.0 / self._spf)))

    # -- lifecycle ----------------------------------------------------------

    def reset(self, *, time_offset: Optional[float] = None) -> None:
        """Drop all state, optionally rebasing the reporting clock.

        Called at a streaming turn boundary: the detector's frame counter goes
        back to zero with the encoder's, and ``time_offset`` carries the session
        clock forward so the next turn's spans are not reported from zero.
        """
        if time_offset is not None:
            self._offset = float(time_offset)
        self._i = 0
        self._run_start: Optional[int] = None
        self._sil_start: Optional[int] = None
        self._start_emitted = False
        self._sum_p = 0.0
        self._n_p = 0
        self._raw: List[SpeechSegment] = []

    # -- introspection -------------------------------------------------------

    @property
    def triggered(self) -> bool:
        """Whether a speech run is currently open."""
        return self._run_start is not None

    @property
    def consumed_frames(self) -> int:
        return self._i

    @property
    def elapsed(self) -> float:
        """Audio seconds consumed since the last reset, plus the offset."""
        return self._offset + self._i * self._spf

    @property
    def raw_segments(self) -> List[SpeechSegment]:
        """Closed segments, **unpadded**, in order."""
        return list(self._raw)

    def segments(self, total_seconds: Optional[float] = None) -> List[SpeechSegment]:
        """Closed segments with padding applied and overlaps resolved."""
        return self._pad_segments(self._raw, total_seconds)

    # -- the machine ---------------------------------------------------------

    def push(self, probs: Sequence[float]) -> List[VadEvent]:
        """Consume the next frames; return whatever became decidable."""
        events: List[VadEvent] = []
        for p in probs:
            self._step(float(p), events)
            self._i += 1
        return events

    def _step(self, p: float, events: List[VadEvent]) -> None:
        i = self._i
        if self._run_start is None:
            if p < self._threshold:
                return
            self._run_start = i
            self._sil_start = None
            self._start_emitted = False
            self._sum_p = 0.0
            self._n_p = 0

        self._sum_p += p
        self._n_p += 1

        if p < self._neg_threshold:
            if self._sil_start is None:
                self._sil_start = i
        else:
            self._sil_start = None

        # The start event waits for ``min_speech_ms`` of speech so a blip never
        # produces an event that would have to be retracted — but it is stamped
        # at the true onset, so only its *arrival* is delayed, not its time.
        # Silero's own VADIterator has no such filter and emits on the first
        # frame over threshold; this is the one place we improve on it.
        if not self._start_emitted and (i - self._run_start + 1) >= self._min_speech:
            self._start_emitted = True
            events.append(VadEvent(SPEECH_STARTED, self._time(self._run_start, pad=-self._pad)))

        if self._sil_start is not None and (i - self._sil_start + 1) >= self._min_silence:
            self._close(self._sil_start, events)
        elif self._max_speech is not None and (i - self._run_start + 1) >= self._max_speech:
            # Cut at the current frame rather than searching backwards for the
            # longest silence in the run, which is what Silero does. The search
            # needs the whole run buffered, which a streaming segmenter does not
            # have; taking the same branch in both flows keeps offline and
            # streaming boundaries identical for the same trace.
            self._close(i + 1, events)

    def _close(self, end_frame: int, events: List[VadEvent]) -> None:
        assert self._run_start is not None
        start_frame = self._run_start
        duration_frames = end_frame - start_frame
        if duration_frames >= self._min_speech and duration_frames > 0:
            mean_p = self._sum_p / self._n_p if self._n_p else 0.0
            self._raw.append(
                SpeechSegment(
                    start=self._time(start_frame),
                    end=self._time(end_frame),
                    speech_prob=mean_p,
                )
            )
            if not self._start_emitted:
                events.append(VadEvent(SPEECH_STARTED, self._time(start_frame, pad=-self._pad)))
            events.append(VadEvent(SPEECH_STOPPED, self._time(end_frame, pad=self._pad)))
        elif self._start_emitted:
            # A start was announced and the run then failed the length test.
            # Unreachable for a sane config (the start fires exactly when the
            # test passes), but a config with min_speech > max_speech can get
            # here, and a dangling "started" with no "stopped" would leave a
            # client's turn open forever.
            events.append(VadEvent(SPEECH_STOPPED, self._time(end_frame, pad=self._pad)))
        self._run_start = None
        self._sil_start = None
        self._start_emitted = False
        self._sum_p = 0.0
        self._n_p = 0

    def flush(self) -> List[VadEvent]:
        """Close an open run at end of audio."""
        events: List[VadEvent] = []
        if self._run_start is not None:
            self._close(self._i, events)
        return events

    def run(
        self, probs: Sequence[float], total_seconds: Optional[float] = None
    ) -> List[SpeechSegment]:
        """One-shot: reset, consume everything, close, return padded segments."""
        self.reset(time_offset=self._offset)
        self.push(probs)
        self.flush()
        return self.segments(total_seconds)

    # -- helpers -------------------------------------------------------------

    def _time(self, frame: int, pad: float = 0.0) -> float:
        return max(0.0, self._offset + frame * self._spf + pad)

    def _pad_segments(
        self, segments: Sequence[SpeechSegment], total_seconds: Optional[float]
    ) -> List[SpeechSegment]:
        """Apply ``speech_pad_ms``, letting neighbours meet in the middle.

        Two segments separated by less than twice the padding would otherwise
        overlap, and overlapping segments handed to the offline fan-out would
        transcribe the same audio twice and duplicate words at the seam. Silero
        splits the gap for the same reason.
        """
        if not segments:
            return []
        out: List[SpeechSegment] = []
        n = len(segments)
        for k, seg in enumerate(segments):
            lo = seg.start - self._pad
            hi = seg.end + self._pad
            if k > 0:
                lo = max(lo, (segments[k - 1].end + seg.start) / 2.0)
            if k + 1 < n:
                hi = min(hi, (seg.end + segments[k + 1].start) / 2.0)
            lo = max(self._offset, lo, 0.0)
            if total_seconds is not None:
                hi = min(hi, self._offset + float(total_seconds))
            if hi > lo:
                out.append(replace(seg, start=lo, end=hi))
        return out
