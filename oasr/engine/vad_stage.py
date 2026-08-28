# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Where voice activity meets the engine: one offline splitter, one streaming stage.

Two entry points, because the two flows want different things from the same
three components:

* **offline** — :class:`OfflineVadSegmenter` turns a whole waveform into speech
  spans, which the engine fans out through the machinery ``longform.py`` already
  has.  Only the *splitter* changes; ``LongFormTracker.register`` has always
  taken arbitrary per-child start offsets, and ``_stitch`` has always shifted
  token timestamps and word timings into file time, so a VAD cut needs none of
  that rewritten.
* **streaming** — :class:`StreamingVadStage` holds one segmenter and one
  endpointer per live stream and advances them from whatever per-frame signal
  the tick produced.

The offline splitter runs on **CPU by default**.  The detector that can
pre-segment today is a few pooling ops over the waveform, so a device round trip
buys nothing and costs a synchronisation on the admitting thread — which is the
dispatcher thread, holding the GIL for every other in-flight request.  Set
``VadConfig.device`` to move it once a detector exists that is worth the trip.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from oasr.vad import (
    EndpointDecision,
    Endpointer,
    SpeechSegment,
    SpeechSegmenter,
    VadConfig,
    VadEvent,
    build_detector,
)
from oasr.vad.detector import as_rows

logger = logging.getLogger(__name__)

__all__ = ["OfflineVadSegmenter", "StreamingVadStage", "StreamVadState"]


class OfflineVadSegmenter:
    """Whole waveform → speech spans, for the offline fan-out.

    Parameters
    ----------
    config : VadConfig
        Already resolved for the offline service mode.
    device : torch.device
        Where the detector runs.
    """

    def __init__(self, config: VadConfig, device: torch.device) -> None:
        self._cfg = config
        self._device = device
        self._detector = build_detector(config, device=device)

    @property
    def detector(self):
        return self._detector

    def segments(self, waveform: torch.Tensor) -> List[SpeechSegment]:
        """Speech spans in seconds, request-relative."""
        wav = waveform.reshape(1, -1).to(device=self._device, dtype=torch.float32)
        lengths = torch.tensor([wav.size(1)], dtype=torch.int64, device=self._device)
        probs, frame_lengths = self._detector.detect(wav, lengths)
        rows = as_rows(probs, frame_lengths)
        if not rows or not rows[0]:
            return []
        segmenter = SpeechSegmenter(self._cfg, self._detector.seconds_per_frame)
        total = wav.size(1) / float(self._cfg.sample_rate)
        return segmenter.run(rows[0], total_seconds=total)

    def spans(self, waveform: torch.Tensor) -> Optional[List[Tuple[int, int]]]:
        """Speech spans as ``[start_sample, end_sample)`` pairs, or ``None``.

        ``None`` means *do not fan out* and is returned for the two cases where
        splitting would be worse than not splitting: no speech was found at all
        (the request should still decode, and produce whatever it produces
        rather than silently returning nothing), and a single span that already
        covers the whole waveform (fanning out to one child would add a hop for
        no cut).
        """
        total_samples = int(waveform.numel())
        if total_samples <= 0:
            return None
        rate = float(self._cfg.sample_rate)
        pairs: List[Tuple[int, int]] = []
        for segment in self.segments(waveform):
            start = max(0, int(round(segment.start * rate)))
            end = min(total_samples, int(round(segment.end * rate)))
            if end > start:
                pairs.append((start, end))
        if not pairs:
            return None
        if len(pairs) == 1 and pairs[0] == (0, total_samples):
            return None
        return pairs


class StreamVadState:
    """One live stream's segmenter, endpointer and reporting clock."""

    __slots__ = ("segmenter", "endpointer", "detector_state", "pending", "decision", "frames")

    def __init__(self, segmenter: SpeechSegmenter, endpointer: Optional[Endpointer]) -> None:
        self.segmenter = segmenter
        self.endpointer = endpointer
        self.detector_state = None
        #: Events produced this tick, drained by the executor into the output.
        self.pending: List[VadEvent] = []
        #: The endpoint decision this stream has reached, if any.
        self.decision: Optional[EndpointDecision] = None
        self.frames = 0


class StreamingVadStage:
    """Per-tick speech activity across the whole live pool.

    Built once per engine and driven from the streaming executor.  The detector
    call is **batched across streams** — never one call per stream — for the same
    reason the feature path packs the pool into a single fbank: at streaming
    cadence what a launch costs is the host issuing it, and 146 per-stream copies
    per step once accounted for 21 % of the streaming step's wall clock before
    they became a single multi-tensor op.

    Only ASR-derived detectors are wired here today, and they need no batching of
    their own: the ASR has already produced one tensor for the whole cohort, so
    this stage slices rather than gathers.
    """

    def __init__(
        self,
        config: VadConfig,
        *,
        seconds_per_frame: float,
        device: torch.device,
        detector_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._cfg = config
        self._spf = float(seconds_per_frame)
        self._device = device
        self._detector = build_detector(
            config, device=device, seconds_per_frame=seconds_per_frame, **(detector_kwargs or {})
        )
        self._states: Dict[str, StreamVadState] = {}

    @property
    def detector(self):
        return self._detector

    @property
    def seconds_per_frame(self) -> float:
        return self._detector.seconds_per_frame

    # -- lifecycle ----------------------------------------------------------

    def open(
        self,
        request_id: str,
        *,
        endpoint: bool = False,
        endpoint_silence_ms: Optional[int] = None,
    ) -> StreamVadState:
        """Allocate this stream's policy state.

        ``endpoint`` is per stream rather than per engine so a request can ask
        for ``single_utterance`` on an engine whose default mode only observes —
        which is how every provider exposes it, as a per-request flag.

        ``endpoint_silence_ms`` overrides the trailing silence by **rewriting**
        the rule that carries one, not by appending a rule.  An extra clause can
        only ever make the disjunction fire *sooner*, so a caller asking for a
        longer silence would have been silently ignored — precisely the failure
        this option exists to prevent.
        """
        cfg = self._cfg
        if endpoint_silence_ms is not None:
            cfg = self._with_silence_override(cfg, float(endpoint_silence_ms) / 1000.0)
        segmenter = SpeechSegmenter(cfg, self.seconds_per_frame)
        wants_endpoint = endpoint or cfg.endpoints
        endpointer = Endpointer(cfg, self.seconds_per_frame) if wants_endpoint else None
        state = StreamVadState(segmenter, endpointer)
        self._states[request_id] = state
        return state

    def close(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def state(self, request_id: str) -> Optional[StreamVadState]:
        return self._states.get(request_id)

    @staticmethod
    def _with_silence_override(cfg: VadConfig, seconds: float) -> VadConfig:
        from dataclasses import replace

        rules = []
        for rule in cfg.endpoint_rules or ():
            if rule.min_trailing_silence_s > 0.0 and rule.must_contain_nonsilence:
                rules.append(replace(rule, min_trailing_silence_s=seconds))
            else:
                rules.append(rule)
        return replace(cfg, endpoint_rules=tuple(rules))

    # -- per tick ------------------------------------------------------------

    def advance(
        self,
        request_ids: Sequence[str],
        tensor: torch.Tensor,
        lengths: torch.Tensor,
        *,
        decoded_any: Optional[Sequence[bool]] = None,
    ) -> None:
        """Feed one cohort's new frames into every member's policy state.

        ``tensor`` is whatever the detector's ``consumes`` names, batched in the
        same row order as ``request_ids``.  One detector call, one device→host
        transfer, then pure Python over a few hundred floats per stream.
        """
        if not request_ids:
            return
        probs, frame_lengths = self._detector.detect_from_asr(tensor, lengths)
        rows = as_rows(probs, frame_lengths)
        for index, request_id in enumerate(request_ids):
            state = self._states.get(request_id)
            if state is None or index >= len(rows):
                continue
            row = rows[index]
            if not row:
                continue
            state.frames += len(row)
            events = state.segmenter.push(row)
            if events:
                state.pending.extend(events)
                if state.endpointer is not None:
                    for event in events:
                        if event.kind == "speech_started":
                            state.endpointer.note_speech_started()
            if state.endpointer is not None and state.decision is None:
                flag = None
                if decoded_any is not None and index < len(decoded_any):
                    flag = bool(decoded_any[index])
                state.decision = state.endpointer.push(row, decoded_any=flag)

    def advance_from_map(
        self,
        requests: Sequence[object],
        tensor_map: Dict[str, torch.Tensor],
    ) -> None:
        """Advance every stream from this tick's per-request chunk tensors.

        The tensors arrive one per request — that is the streaming backend's
        output shape — so they are grouped by chunk width and concatenated into
        one call per width.  In steady state every stream is fed the same chunk,
        so there is exactly one group and exactly one detector launch for the
        whole pool.

        The concatenation copies the cohort's log-probs: at 64 streams, a 16
        frame chunk and a 5 000-token vocabulary that is 20 MB per tick, which
        is well under a millisecond of bandwidth and is the price of keeping the
        detector's declared input shape honest.  It is also the *safe* read:
        a captured-graph backend hands out one output buffer per shape key, so a
        tensor from this map is live only until the next replay.
        """
        groups: Dict[int, List[str]] = {}
        for request in requests:
            request_id = getattr(request, "request_id", None)
            tensor = tensor_map.get(request_id) if request_id is not None else None
            if tensor is None or request_id not in self._states:
                continue
            groups.setdefault(int(tensor.size(-2)), []).append(request_id)
        for width, ids in groups.items():
            if not ids:
                continue
            batch = torch.cat([tensor_map[rid] for rid in ids], dim=0)
            lengths = torch.full((len(ids),), width, dtype=torch.int64, device=batch.device)
            self.advance(ids, batch, lengths)

    def endpointed(self, request_id: str) -> Optional[EndpointDecision]:
        """This stream's endpoint decision, if it has reached one."""
        state = self._states.get(request_id)
        return state.decision if state is not None else None

    def drain_events(self, request_id: str) -> Optional[List[VadEvent]]:
        """Take this stream's events since the last drain, or ``None``."""
        state = self._states.get(request_id)
        if state is None or not state.pending:
            return None
        events = state.pending
        state.pending = []
        return events

    def finish(
        self, request_id: str
    ) -> Tuple[Optional[List[VadEvent]], Optional[List[SpeechSegment]]]:
        """Close an open run at end of audio and hand back the final view."""
        state = self._states.get(request_id)
        if state is None:
            return None, None
        events = state.pending + state.segmenter.flush()
        state.pending = []
        segments = state.segmenter.segments()
        return (events or None), (segments or None)
