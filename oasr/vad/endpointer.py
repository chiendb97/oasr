# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""When has the speaker finished — Kaldi's rule disjunction over a robust silence count.

Endpointing is not segmentation.  Kaldi's own header is explicit about it: *"by
endpointing in this context we mean deciding when to stop decoding, and not
generic speech/silence segmentation"*, and it ships a separate energy VAD for the
other job that it warns is unsuitable for ASR.  Two problems, two components —
this module is the first one, :mod:`oasr.vad.segmenter` is the second.

The rule structure is Kaldi's ``OnlineEndpointConfig``, which WeNet, sherpa,
sherpa-onnx and Vosk all copy: an OR of clauses, each an AND of conditions over
trailing silence, turn length, and whether anything was decoded.
``max_relative_cost`` is absent by design — see :class:`~oasr.vad.config.EndpointRule`.

What is *not* Kaldi's is how trailing silence is counted.  A plain run-length
counter, which is what WeNet and sherpa-onnx use, is reset to zero by a single
spurious non-blank frame in the middle of a pause, so one bad frame costs a whole
endpoint.  Riva instead asks what fraction of a trailing window is active.  This
takes Riva's test and uses it to *qualify* the run-length counter: a frame resets
the counter only if the window ending at it is genuinely active.  A lone frame in
silence cannot qualify, and a real last-word frame always does — so the counter
keeps Kaldi's exact semantics (silence measured from the last real speech) while
gaining Riva's robustness, instead of trading one for the other.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence

from .config import VadConfig

__all__ = ["Endpointer", "EndpointDecision", "TURN_RULES", "START_TIMEOUT", "END_TIMEOUT"]

#: Reason prefix for a fired :class:`~oasr.vad.config.EndpointRule`.
TURN_RULES = "rule"
#: Reasons that close the stream rather than ending a turn.
START_TIMEOUT = "speech_start_timeout"
END_TIMEOUT = "speech_end_timeout"


@dataclass(frozen=True)
class EndpointDecision:
    """Why, and when, the endpointer fired.

    Attributes
    ----------
    reason : str
        ``"rule1"``..``"ruleN"`` for a turn boundary, or one of
        :data:`START_TIMEOUT` / :data:`END_TIMEOUT`.  Carried through to
        ``oasr_engine_endpoints_total{reason}``, which is why it is a small fixed
        vocabulary rather than free text.
    time : float
        Seconds of audio at which it fired, in the segmenter's time base.
    terminal : bool
        ``True`` closes the stream (a timeout expired); ``False`` ends the turn
        and leaves the connection open.  Conflating the two would have a
        conversational client's second turn silently dropped.
    """

    reason: str
    time: float
    terminal: bool


class Endpointer:
    """Turn-boundary policy over a per-frame speech probability stream.

    Parameters
    ----------
    config : VadConfig
        Must be resolved (:meth:`VadConfig.resolve`).
    seconds_per_frame : float
        The detector's frame rate.
    time_offset : float
        Reporting-clock base, matching the segmenter's.
    """

    def __init__(
        self,
        config: VadConfig,
        seconds_per_frame: float,
        *,
        time_offset: float = 0.0,
    ) -> None:
        if config.threshold is None:
            raise ValueError(
                "Endpointer needs a resolved VadConfig; call VadConfig.resolve(service_mode)"
            )
        if seconds_per_frame <= 0.0:
            raise ValueError(f"seconds_per_frame must be > 0, got {seconds_per_frame!r}")
        self._cfg = config
        self._spf = float(seconds_per_frame)
        self._offset = float(time_offset)
        self._threshold = float(config.threshold)
        self._rules = tuple(config.endpoint_rules or ())
        self._window_cap = max(1, int(round(float(config.activity_window_ms) / 1000.0 / self._spf)))
        self._activity_threshold = float(config.activity_threshold)
        self._start_timeout = config.speech_start_timeout_s
        self._end_timeout = config.speech_end_timeout_s
        self._speech_started_ever = False
        self.reset()

    # -- lifecycle ----------------------------------------------------------

    def reset(self, *, time_offset: Optional[float] = None) -> None:
        """Drop per-turn state.

        ``_speech_started_ever`` deliberately survives: Google specifies that the
        speech-start timeout is cancelled *for the rest of the stream* once the
        first speech begins, not merely for the current turn, and a client that
        pauses between turns must not have the stream closed under it.
        """
        if time_offset is not None:
            self._offset = float(time_offset)
        self._window: Deque[bool] = deque(maxlen=self._window_cap)
        self._nonsilent_in_window = 0
        self._frames = 0
        self._last_active_frame: Optional[int] = None
        self._fired = False

    def note_speech_started(self) -> None:
        """Record that the segmenter announced speech.

        Driven from the segmenter rather than from this class's own activity gate
        so ``speech_started`` means one thing across the event stream and the
        timeout, instead of two subtly different things that agree most of the
        time.
        """
        self._speech_started_ever = True

    # -- introspection -------------------------------------------------------

    @property
    def trailing_silence(self) -> float:
        """Seconds since the last frame that qualified as real speech."""
        if self._frames == 0:
            return 0.0
        if self._last_active_frame is None:
            return self._frames * self._spf
        return (self._frames - 1 - self._last_active_frame) * self._spf

    @property
    def utterance_length(self) -> float:
        """Seconds of audio consumed in this turn."""
        return self._frames * self._spf

    @property
    def contains_nonsilence(self) -> bool:
        """Kaldi's derivation: something was decoded iff the turn is longer than its tail."""
        return self.utterance_length > self.trailing_silence

    # -- the machine ---------------------------------------------------------

    def push(
        self, probs: Sequence[float], *, decoded_any: Optional[bool] = None
    ) -> Optional[EndpointDecision]:
        """Consume this chunk's frames; return a decision the first time one fires.

        ``decoded_any`` overrides the derived ``contains_nonsilence`` when the
        decode strategy actually knows — a frame-synchronous family knows whether
        it has emitted a token, and that is strictly better evidence than a
        length comparison.  ``None`` keeps Kaldi's derivation.

        Returns at most one decision per turn; :meth:`reset` re-arms it.
        """
        for p in probs:
            self._step(float(p))
            decision = self._decide(decoded_any)
            if decision is not None:
                return decision
        return None

    def _step(self, p: float) -> None:
        nonsilent = p >= self._threshold
        if len(self._window) == self._window.maxlen and self._window[0]:
            self._nonsilent_in_window -= 1
        self._window.append(nonsilent)
        if nonsilent:
            self._nonsilent_in_window += 1
        # A frame resets the trailing-silence counter only if the window ending
        # at it is genuinely active, so an isolated non-silent frame in a pause
        # cannot.  ``ceil`` rather than a bare product: with a 7-frame window at
        # threshold 0.2 the bar is 2 frames, not 1.4 rounded down to 1, which
        # would make the test a no-op.
        needed = max(1, math.ceil(self._activity_threshold * len(self._window)))
        if nonsilent and self._nonsilent_in_window >= needed:
            self._last_active_frame = self._frames
        self._frames += 1

    def _decide(self, decoded_any: Optional[bool]) -> Optional[EndpointDecision]:
        if self._fired:
            return None
        now = self._offset + self._frames * self._spf

        # Timeouts first: they close the stream, and a stream that should be
        # closed must not instead report a turn boundary and keep running.
        if (
            self._start_timeout is not None
            and not self._speech_started_ever
            and self.utterance_length >= float(self._start_timeout)
        ):
            return self._fire(START_TIMEOUT, now, terminal=True)
        if (
            self._end_timeout is not None
            and self._speech_started_ever
            and self.trailing_silence >= float(self._end_timeout)
        ):
            return self._fire(END_TIMEOUT, now, terminal=True)

        nonsilence = self.contains_nonsilence if decoded_any is None else bool(decoded_any)
        trailing = self.trailing_silence
        length = self.utterance_length
        for index, rule in enumerate(self._rules, start=1):
            if rule.must_contain_nonsilence and not nonsilence:
                continue
            if trailing < rule.min_trailing_silence_s:
                continue
            if length < rule.min_utterance_length_s:
                continue
            return self._fire(f"{TURN_RULES}{index}", now, terminal=False)
        return None

    def _fire(self, reason: str, time: float, *, terminal: bool) -> EndpointDecision:
        self._fired = True
        return EndpointDecision(reason=reason, time=time, terminal=terminal)


def rule_names(config: VadConfig) -> List[str]:
    """Every reason this config can produce, for metric-label pre-registration."""
    names = [f"{TURN_RULES}{i}" for i in range(1, len(config.endpoint_rules or ()) + 1)]
    return names + [START_TIMEOUT, END_TIMEOUT]
