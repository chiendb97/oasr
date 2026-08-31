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
  the tick produced.  Which signal that is decides where the stage sits: an
  ASR-derived detector is fed the encoder's output and therefore runs *behind*
  the encoder, so it can label audio and end a turn but never decide what gets
  encoded; a waveform detector is fed the audio as it arrives and runs *ahead* of
  it, which is what makes :meth:`StreamingVadStage.should_encode` a gate rather
  than a report.  ``vad.mode="segment"`` is the mode that uses the second.

Waveform detectors run on **CPU by default**, in both flows.  The one that can
pre-segment today is a few pooling ops over audio that is already on the host, so
a device round trip buys nothing and costs a synchronisation — on the admitting
(dispatcher, GIL-holding) thread offline, and inside the step loop streaming.
Set ``VadConfig.device`` to move it once a detector exists that is worth the trip.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

from oasr.vad import (
    SPEECH_STARTED,
    EndpointDecision,
    Endpointer,
    SpeechSegment,
    SpeechSegmenter,
    VadConfig,
    VadEvent,
    build_detector,
    get_vad_spec,
)
from oasr.vad.detector import SpeechDetector, VadState, as_rows

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .request import Request

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
    def detector(self) -> SpeechDetector:
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

        Spans that **touch** are merged first, and that is load-bearing rather
        than tidying: the fan-out decodes each span as an independent request
        with no shared context and no overlap to dedup against, which is sound
        for a cut that lands in a silence and wrong for one that does not.  A
        boundary with no gap on either side drops no audio, so it buys the
        encoder nothing and costs the word that straddles it.  ``max_speech_s``
        is the way in — :class:`~oasr.vad.SpeechSegmenter` closes a run that
        reaches the cap *at the current frame* rather than at a silence — and a
        real gap shorter than twice ``speech_pad_ms`` is the other, since the
        padding then meets in the middle.
        """
        total_samples = int(waveform.numel())
        if total_samples <= 0:
            return None
        rate = float(self._cfg.sample_rate)
        pairs: List[Tuple[int, int]] = []
        for segment in self.segments(waveform):
            start = max(0, int(round(segment.start * rate)))
            end = min(total_samples, int(round(segment.end * rate)))
            if end <= start:
                continue
            if pairs and start <= pairs[-1][1]:
                pairs[-1] = (pairs[-1][0], max(pairs[-1][1], end))
                continue
            pairs.append((start, end))
        if not pairs:
            return None
        if len(pairs) == 1 and pairs[0] == (0, total_samples):
            return None
        return pairs


class StreamVadState:
    """One live stream's segmenter, endpointer and reporting clock."""

    __slots__ = (
        "segmenter",
        "endpointer",
        "detector_state",
        "pending",
        "decision",
        "frames",
        "carry",
    )

    def __init__(self, segmenter: SpeechSegmenter, endpointer: Optional[Endpointer]) -> None:
        self.segmenter = segmenter
        self.endpointer = endpointer
        self.detector_state: Optional[VadState] = None
        #: Events produced this tick, drained by the executor into the output.
        self.pending: List[VadEvent] = []
        #: The endpoint decision this stream has reached, if any.
        self.decision: Optional[EndpointDecision] = None
        self.frames = 0
        #: Waveform detectors only: the samples of the last chunk that did not
        #: fill a whole analysis frame.  Dropping them instead would shorten the
        #: stream by up to one frame per chunk, so a detector's clock would drift
        #: away from the encoder's a few milliseconds at a time — the class of
        #: error whose output stays entirely plausible.
        self.carry: Optional[torch.Tensor] = None


class StreamingVadStage:
    """Per-tick speech activity across the whole live pool.

    Built once per engine and driven from the streaming executor.  The detector
    call is **batched across streams** — never one call per stream — for the same
    reason the feature path packs the pool into a single fbank: at streaming
    cadence what a launch costs is the host issuing it, and 146 per-stream copies
    per step once accounted for 21 % of the streaming step's wall clock before
    they became a single multi-tensor op.

    Two feeds, chosen by the configured detector's declared ``consumes``:

    * an **ASR-derived** detector is advanced from the tick's encoder output
      (:meth:`advance_from_map`).  It needs no batching of its own — the ASR has
      already produced one tensor for the whole cohort, so this stage slices
      rather than gathers.  It runs *after* the encoder, so it can label audio
      and end a turn but can never decide what the encoder sees.
    * a **waveform** detector is advanced from the audio as it is fed
      (:meth:`advance_audio`), which puts it *ahead* of the encoder and is what
      makes :meth:`should_encode` a real gate.  That is the second half of the
      pairing every production system ships: a cheap pre-ASR detector deciding
      what to encode, and an ASR-derived one deciding when a turn ended.
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
        #: Audio fed but not yet classified, per stream.  Keyed independently of
        #: ``_states`` because a whole-waveform streaming request is fed at
        #: ``admit`` time, which is before the scheduler promotes it and
        #: therefore before :meth:`open` runs.
        self._audio: Dict[str, List[torch.Tensor]] = {}
        spec = get_vad_spec(str(config.backend))
        self._consumes = spec.consumes
        self._stateful = spec.stateful
        self._pad_s = config.speech_pad_seconds
        if self.needs_audio:
            framing = self._detector.framing
            if framing is None:
                raise ValueError(
                    f"vad backend={config.backend!r} consumes a waveform but reports no "
                    "framing, so the stage cannot tell how many samples one frame "
                    "consumed and would drift against the encoder's clock"
                )
            if framing.history or framing.prefill:
                # The carry rule below keeps ``buf[frames * hop:]``, which is only
                # the untouched remainder when frames start at multiples of the
                # hop from the buffer's front.  A detector with leading context
                # needs a different rule, and guessing one would misalign every
                # boundary it reports.
                raise NotImplementedError(
                    f"vad backend={config.backend!r} declares history/prefill framing, "
                    "which the incremental waveform feed does not carry yet"
                )
            self._hop = int(framing.hop)
            self._min_samples = int(framing.min_samples)
        else:
            self._hop = 0
            self._min_samples = 0

    @property
    def detector(self) -> SpeechDetector:
        return self._detector

    @property
    def seconds_per_frame(self) -> float:
        return self._detector.seconds_per_frame

    @property
    def consumes(self) -> str:
        """What this stage's detector is fed — ``"waveform"`` or an ASR tensor."""
        return self._consumes

    @property
    def needs_audio(self) -> bool:
        """Whether the executor must tee the fed waveform into this stage."""
        return self._consumes == "waveform"

    @property
    def gates_encoder(self) -> bool:
        """Whether this stage decides which encoder windows actually run."""
        return self._cfg.gates_encoder

    @property
    def pad_seconds(self) -> float:
        """``speech_pad_ms`` in seconds — the gate's margin on both sides."""
        return self._pad_s

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
        if self._stateful:
            state.detector_state = self._detector.new_state(1)
        self._states[request_id] = state
        return state

    def close(self, request_id: str) -> None:
        self._states.pop(request_id, None)
        self._audio.pop(request_id, None)

    def state(self, request_id: str) -> Optional[StreamVadState]:
        return self._states.get(request_id)

    @staticmethod
    def _with_silence_override(cfg: VadConfig, seconds: float) -> VadConfig:
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
            flag = None
            if decoded_any is not None and index < len(decoded_any):
                flag = bool(decoded_any[index])
            self._consume_row(state, row, flag)

    def advance_from_map(
        self,
        requests: Sequence["Request"],
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
            request_id = request.request_id
            tensor = tensor_map.get(request_id)
            if tensor is None or request_id not in self._states:
                continue
            groups.setdefault(int(tensor.size(-2)), []).append(request_id)
        for width, ids in groups.items():
            if not ids:
                continue
            batch = torch.cat([tensor_map[rid] for rid in ids], dim=0)
            lengths = torch.full((len(ids),), width, dtype=torch.int64, device=batch.device)
            self.advance(ids, batch, lengths)

    # -- waveform feed (the pre-encoder gate) --------------------------------

    def feed_audio(self, request_id: str, chunk: Any) -> None:
        """Tee one fed audio chunk to this stage.

        Called from the executor's two ingestion points rather than from the
        input processor, which has no business knowing about voice activity.
        Holds a **reference**, not a copy: the same tensor is already queued for
        feature extraction, and it is released one tick later when
        :meth:`advance_audio` consumes it.
        """
        if not self.needs_audio:
            return
        wave = torch.as_tensor(chunk, dtype=torch.float32).reshape(-1)
        if wave.numel() == 0:
            return
        self._audio.setdefault(request_id, []).append(wave.to("cpu", copy=False))

    def advance_audio(self, requests: Sequence["Request"]) -> None:
        """Classify every stream's newly fed audio in one batched detector call.

        Runs at the top of the tick, ahead of the encoder, so the gate has a
        verdict for the window the encoder is about to be handed.  Streams whose
        buffer does not yet hold a whole analysis frame keep it as carry rather
        than being called with nothing — a zero-frame row still costs a column in
        the padded batch.
        """
        if not self.needs_audio:
            return
        ids: List[str] = []
        states: List[StreamVadState] = []
        buffers: List[torch.Tensor] = []
        for request in requests:
            request_id = request.request_id
            state = self._states.get(request_id)
            if state is None:
                continue
            chunks = self._audio.pop(request_id, None)
            parts: List[torch.Tensor] = []
            if state.carry is not None and state.carry.numel():
                parts.append(state.carry)
            if chunks:
                parts.extend(chunks)
            if not parts:
                continue
            buf = parts[0] if len(parts) == 1 else torch.cat(parts)
            state.carry = buf
            if buf.numel() < self._min_samples:
                continue
            ids.append(request_id)
            states.append(state)
            buffers.append(buf)
        if not ids:
            return

        sizes = [int(b.numel()) for b in buffers]
        widest = max(sizes)
        if all(size == widest for size in sizes):
            # The steady state: every stream was fed the same chunk, so there is
            # nothing to pad and one op replaces a Python-level copy per stream.
            batch = torch.stack(buffers)
        else:
            batch = torch.zeros(len(buffers), widest, dtype=torch.float32)
            for slot, buf in enumerate(buffers):
                batch[slot, : buf.numel()] = buf
        lengths = torch.tensor(sizes, dtype=torch.int64)
        batch = batch.to(self._device)
        lengths = lengths.to(self._device)

        stacked = self._detector.stack_states([st.detector_state for st in states])
        probs, frame_lengths, new_state = self._detector.detect_streaming(batch, lengths, stacked)
        rows = as_rows(probs, frame_lengths)
        per_stream = self._detector.unstack_states(new_state, len(states))

        for index, state in enumerate(states):
            state.detector_state = per_stream[index]
            row = rows[index] if index < len(rows) else []
            consumed = len(row) * self._hop
            buf = buffers[index]
            # ``clone``, not a view: the remainder is under one analysis window
            # wide, but a slice of ``buf`` keeps the *whole* concatenated buffer's
            # storage alive — which for a stream fed a complete waveform at
            # admission is the entire utterance, held for the life of the stream.
            state.carry = buf[consumed:].clone() if consumed else buf
            if row:
                self._consume_row(state, row)

    @staticmethod
    def _consume_row(
        state: StreamVadState, row: Sequence[float], decoded_any: Optional[bool] = None
    ) -> None:
        """Push one stream's new probabilities through its policy state.

        Shared by both feeds, so an ASR-derived trace and a waveform trace reach
        the same state machine by the same route — which is the whole reason the
        segmenter and the endpointer are detector-agnostic.
        """
        state.frames += len(row)
        events = state.segmenter.push(row)
        if events:
            state.pending.extend(events)
            if state.endpointer is not None:
                for event in events:
                    if event.kind == SPEECH_STARTED:
                        state.endpointer.note_speech_started()
        if state.endpointer is not None and state.decision is None:
            state.decision = state.endpointer.push(row, decoded_any=decoded_any)

    def classified_until(self, request_id: str) -> float:
        """Audio seconds of this stream the detector has actually judged."""
        state = self._states.get(request_id)
        return 0.0 if state is None else state.segmenter.elapsed

    def turn_boundary(self, request_id: str) -> Optional[float]:
        """Session time after which audio belongs to a **new** turn, or ``None``.

        The end of the last confirmed speech run plus the segment padding —
        unless speech has already resumed by then, in which case there is no
        silence to cut at and this returns ``None``.  It is
        deliberately not the same question :meth:`should_encode` answers: a turn
        boundary is a fact about audio the detector has already judged, so it can
        be acted on however far behind the encoder happens to be, while a skip
        needs the verdict *before* the encoder is dispatched.  Separating them is
        what makes ``segment`` mode do something useful on a real-time stream,
        where the encoder keeps pace with the audio and there is never enough
        lookahead to skip: the silence still gets encoded, but the turn still
        closes, so the KV cache stops growing and the next turn starts clean.
        """
        state = self._states.get(request_id)
        if state is None:
            return None
        end = state.segmenter.last_segment_end
        if end is None:
            return None
        boundary = end + self._pad_s
        run_start = state.segmenter.open_run_start
        if run_start is not None and run_start <= boundary:
            # Speech resumes at or before the boundary, so there is no silence
            # to cut at: closing the turn here would reset the encoder in the
            # middle of a word, and streaming cannot replay the audio to recover
            # it.  ``max_speech_s`` is how a run closes without a silence
            # (:class:`~oasr.vad.SpeechSegmenter` cuts at the current frame), and
            # a cap is a length decision, not a turn decision.
            return None
        return boundary

    def should_encode(self, request_id: str, start_s: float, end_s: float) -> bool:
        """Whether the encoder window covering ``[start_s, end_s)`` must run.

        Two conditions, and the order matters.  The window is encoded unless the
        detector has already classified past its far edge *and* nothing in it —
        widened by ``speech_pad_ms`` on both sides — is speech.  An unclassified
        window is therefore always encoded, which is the only safe direction for
        this gate: encoding silence costs time, dropping speech costs words, and
        the two mistakes are not comparable.
        """
        state = self._states.get(request_id)
        if state is None:
            return True
        pad = self._pad_s
        if state.segmenter.elapsed < end_s + pad:
            return True
        return state.segmenter.overlaps_speech(start_s - pad, end_s + pad)

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
