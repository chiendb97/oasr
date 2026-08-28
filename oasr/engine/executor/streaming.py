# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Chunk-by-chunk streaming executor with paged KV cache."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Union

import numpy as np
import torch

from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .. import metrics as m
from ..config import EngineConfig
from ..input_processor import InputProcessor
from ..model_runner import ModelRunner
from ..output_processor import OutputProcessor
from ..request import Request, RequestOutput, RequestState
from ..scheduler import Scheduler
from .base import Executor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from oasr.vad import EndpointDecision

    from ..vad_stage import StreamingVadStage

logger = logging.getLogger(__name__)

#: ``RequestOutput.endpoint_reason`` for a ``vad.mode="segment"`` turn boundary.
#: Distinct from the endpointer's ``"rule<N>"`` values on purpose: those name a
#: clause of the Kaldi-shaped disjunction that ended the *request*, and this one
#: says the stream simply crossed a confirmed silence and carried on.
_TURN_BOUNDARY = "vad_segment"


class StreamingExecutor(Executor):
    """Streaming inference with one encoder chunk per active stream per tick.

    See :class:`oasr.engine.executor.base.Executor` for the per-tick
    protocol.  Each :meth:`step`:

    1. Admits waiting streams up to ``max_batch_size`` and allocates their
       paged KV / CNN / CTC caches.
    2. Runs one batched GPU fbank across every active stream that has
       pending audio (one kernel call for the whole pool, on a dedicated
       feat stream so it overlaps with the encoder forward).
    3. Runs ``forward_streaming_step`` against every stream whose feature
       buffer holds a full encoder window, then decodes a partial per
       result.
    4. Finalises streams the client has explicitly closed
       (``audio_final``) once both the audio deque and the feature
       buffer are drained, freeing their cache slots.

    Steps 2 and 3 run in one of two orders, selected by
    ``streaming_feature_lookahead``:

    * **pipelined** (default, :meth:`_step_pipelined`) — forward first, then
      extract, so the feature pack's host work runs while the GPU is busy with
      the encoder.  Features extracted in step *N* are forwarded by step *N+1*.
    * **serial** (:meth:`_step_serial`) — extract then forward, both in the same
      step.  The order this executor had before; kept as the A/B arm and for a
      deployment that would rather not spend one step of pipeline depth.
    """

    streaming: ClassVar[bool] = True

    #: Per-engine speech-activity stage, or ``None`` when VAD is off.  A class
    #: attribute with a ``None`` default, not just an ``__init__`` assignment,
    #: for the same reason ``Executor._metrics`` is one: the isolation tests
    #: build executors minimally, and an attribute that only exists after a full
    #: construction turns "VAD is off" into an AttributeError on the failure
    #: path — which is exactly the path those tests exercise.
    _vad: Optional["StreamingVadStage"] = None

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        input_processor: InputProcessor,
        model_runner: ModelRunner,
        output_processor: OutputProcessor,
        config: EngineConfig,
        device: torch.device,
        metrics: Optional[m.EngineMetrics] = None,
        vad_stage: Optional["StreamingVadStage"] = None,
    ) -> None:
        if metrics is not None:
            self._metrics = metrics
        self._vad = vad_stage
        self._scheduler = scheduler
        self._inp = input_processor
        self._mr = model_runner
        self._op = output_processor
        self._config = config
        self._device = device

        # Dedicated CUDA stream for the streaming fbank kernel.  Lets the
        # H2D waveform copy + batched fbank for the current step overlap
        # with the encoder forward of the previous step's tail (the
        # encoder forward is async, so once dispatched the GPU can run
        # both kernels concurrently when they live on different streams).
        self._feat_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        # Extract the *next* step's features after issuing this step's encoder
        # forward rather than before it — see
        # :meth:`_step_pipelined`.  Pointless without a second stream to put the
        # feature work on, so it follows ``_feat_stream``.
        self._lookahead: bool = bool(
            getattr(config, "streaming_feature_lookahead", True) and self._feat_stream is not None
        )

    # ------------------------------------------------------------------
    # Executor ABC
    # ------------------------------------------------------------------

    def admit(self, request: Request) -> None:
        """Register a streaming request and enqueue it for admission.

        Streaming is chunk-by-chunk: :meth:`InputProcessor.prepare_streaming`
        registers the request with an empty audio queue, then audio arrives
        via :meth:`feed_chunk`.  Two entry shapes feed this:

        * ``add_streaming_request`` — ``request.audio is None``; the caller
          pushes chunks itself (the real-time / serving path).
        * ``transcribe(waveform, streaming=True)`` — a full waveform is
          attached up front; we split it into per-step audio chunks and
          enqueue them here (the last marked final) so the step loop windows
          it exactly like a live feed would.

        The engine is waveform-only.
        """
        self._inp.prepare_streaming(request)
        if request.audio is not None:
            wav = torch.as_tensor(request.audio, dtype=torch.float32, device="cpu").reshape(-1)
            chunk_samples = self._inp.streaming_audio_chunk_samples
            n = int(wav.numel())
            if request.sample_rate:
                self._metrics.incr(m.AUDIO_SECONDS, n / request.sample_rate)
            self._tee_audio(request.request_id, wav)
            if n == 0:
                request.audio_final = True
            else:
                starts = range(0, n, chunk_samples)
                last = n - (n % chunk_samples or chunk_samples)
                for s in starts:
                    self._inp.append_streaming_chunk(
                        request, wav[s : s + chunk_samples], is_last=(s == last)
                    )
        self._scheduler.add_request(request)

    def feed_chunk(
        self,
        request_id: str,
        chunk: Union[torch.Tensor, "np.ndarray"],
        is_last: bool = False,
    ) -> None:
        req = self._scheduler.find_request(request_id)
        if req is None:
            raise KeyError(f"feed_chunk: unknown or finished request_id {request_id!r}")
        # Counted per chunk, not per request: a live stream has no total
        # duration until it closes, and a stream that is abandoned mid-flight
        # would otherwise contribute nothing to the RTFx denominator despite
        # having cost the engine every chunk it did send.
        if self._metrics.enabled and req.sample_rate:
            n = chunk.shape[-1] if hasattr(chunk, "shape") else len(chunk)
            self._metrics.incr(m.AUDIO_SECONDS, int(n) / req.sample_rate)
        self._tee_audio(request_id, chunk)
        self._inp.append_streaming_chunk(req, chunk, is_last=is_last)

    def _tee_audio(self, request_id: str, chunk: Union[torch.Tensor, "np.ndarray"]) -> None:
        """Hand the fed waveform to a detector that runs *ahead* of the encoder.

        Only a waveform detector wants this, and only it can gate the encoder:
        the ASR-derived kinds read what the encoder produced, so by the time they
        have an opinion the work is already done.  Teed at the executor's two
        ingestion points rather than inside ``InputProcessor``, which has no
        business knowing about voice activity — and at *both*, because a
        whole-waveform streaming request never goes through ``feed_chunk``.

        The stage holds a reference, not a copy; the same samples are already
        queued for feature extraction and are released a tick later.
        """
        if self._vad is not None and self._vad.needs_audio:
            self._vad.feed_audio(request_id, chunk)

    def _open_vad(self, req: Request) -> None:
        """Allocate this stream's speech-activity state, if VAD is running."""
        if self._vad is None:
            return
        options = getattr(req, "decoding", None)
        self._vad.open(
            req.request_id,
            endpoint=bool(getattr(options, "single_utterance", False)),
            endpoint_silence_ms=getattr(options, "endpoint_silence_ms", None),
        )

    def _close_vad(self, request_id: str) -> None:
        if self._vad is not None:
            self._vad.close(request_id)

    def _advance_vad(self, ready: List[Request], log_probs_map: Dict[str, torch.Tensor]) -> None:
        """Feed this tick's per-frame signal into every ready stream's policy.

        Runs *after* the decode, which is the device->host boundary, so the
        detector's own small readback lands on an already-drained queue rather
        than adding a second synchronisation to the step.
        """
        if self._vad is None or not ready or self._vad.needs_audio:
            # A waveform detector was already advanced from the audio itself, at
            # the top of the tick — that is what put it *ahead* of the encoder.
            # Feeding it the encoder's output here would hand it a tensor it does
            # not consume, and it says so rather than guessing.
            return
        nvtx_push("vad_step")
        t0 = time.perf_counter()
        try:
            self._vad.advance_from_map(ready, log_probs_map)
        except Exception:  # noqa: BLE001 - speech activity must never fail a transcript
            nvtx_pop()
            logger.warning("voice activity step failed; continuing without it", exc_info=True)
            return
        nvtx_pop()
        self._metrics.observe_stage("streaming.vad", time.perf_counter() - t0)

    def _advance_audio_vad(self, running: List[Request]) -> None:
        """Classify newly fed audio before the encoder is asked to run.

        This is the half of the tick that makes ``vad.mode="segment"`` possible:
        the detector is ahead of the encoder rather than behind it, so the gate
        has a verdict for the window about to be dispatched.  Failure is
        swallowed for the same reason the post-decode advance swallows it —
        speech activity is metadata, and a detector that raises must not cost the
        stream its transcript.  What it *does* cost is the gate's verdict, and an
        absent verdict means "encode", so the stream degrades to plain streaming
        rather than to dropped audio.
        """
        if self._vad is None or not self._vad.needs_audio or not running:
            return
        nvtx_push("vad_audio")
        t0 = time.perf_counter()
        try:
            self._vad.advance_audio(running)
        except Exception:  # noqa: BLE001 - speech activity must never fail a transcript
            nvtx_pop()
            logger.warning("voice activity audio step failed; continuing", exc_info=True)
            return
        nvtx_pop()
        self._metrics.observe_stage("streaming.vad", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Encoder gating and turn boundaries (vad.mode="segment")
    # ------------------------------------------------------------------

    @property
    def _input_frame_seconds(self) -> float:
        """Audio seconds one **input feature frame** covers.

        The hop, times the frontend's low-frame-rate stacking — the same two
        factors ``FrameClock`` uses, minus the encoder's subsampling, because the
        gate reasons about the buffer the encoder is *fed*, not what it emits.
        """
        fc = self._config.feature_config
        if fc is None:
            return 0.0
        return float(fc.frame_shift_ms) / 1000.0 * max(1, int(getattr(fc, "lfr_n", 1) or 1))

    def _gate(self, ready: List[Request], outputs: List[RequestOutput]) -> List[Request]:
        """Drop the encoder windows the detector says are silence.

        The one mode that changes what the model sees, so the bias is explicit:
        a window is skipped only when the detector has already classified past
        its far edge *and* nothing within it — widened by ``speech_pad_ms`` — is
        speech.  Anything else is encoded.  Encoding silence costs time; dropping
        speech costs words.

        A skip implies a **turn boundary**, and that is not a policy choice: the
        next window the encoder does see is no longer the one after the last it
        saw, so its cache and its position have to go back to zero together
        (AGENTS.md rule 13).  Resetting the encoder cuts the decoder's context
        with it, so the turn has to be closed and its transcript folded into the
        stream's running one.

        Skipping and closing are nonetheless asked *separately*, because they
        need different things from the detector.  A skip needs a verdict for a
        window that has not been dispatched yet, so it only ever happens when the
        detector is running ahead of the encoder — which is to say when the
        stream is backlogged, exactly the case where saving encoder work is worth
        something.  On a stream arriving at real time the encoder keeps pace and
        there is no lookahead to skip with; the silence is encoded, but the turn
        still closes behind it, so the KV cache stops growing and the next turn
        starts on a clean context.  Making the skip a precondition for the close
        would have made ``segment`` mode inert on exactly the streams it is
        advertised for.
        """
        if not ready or self._vad is None or not self._vad.gates_encoder:
            return ready
        shift = self._input_frame_seconds
        if shift <= 0.0:
            return ready
        window = self._mr.decoding_window
        stride = self._mr.stride
        keep: List[Request] = []
        for req in ready:
            skipped = 0
            while req.has_ready_encoder_chunk(window):
                start = req.feature_base + req.feature_cursor
                if self._vad.should_encode(req.request_id, start * shift, (start + window) * shift):
                    break
                available = req.feature_frames - req.feature_cursor
                advanced = min(stride, available)
                if advanced <= 0:
                    break
                req.feature_cursor += advanced
                skipped += advanced
            if skipped:
                # Encoder work that never happened, in the unit an operator can
                # compare against ``oasr_engine_audio_seconds_total``.
                self._metrics.incr(m.AUDIO_SECONDS_SKIPPED, skipped * shift)
            start_s = (req.feature_base + req.feature_cursor) * shift
            self._maybe_close_turn(req, start_s, outputs)
            if not req.has_ready_encoder_chunk(window):
                continue
            if req.offset == 0:
                # First window of a turn: pin the reporting clock to where this
                # turn actually starts, so the model clock can restart at zero.
                req.stream_time_offset = start_s
            keep.append(req)
        return keep

    def _maybe_close_turn(
        self, req: Request, window_start_s: float, outputs: List[RequestOutput]
    ) -> None:
        """Close the open turn if the encoder has crossed a confirmed silence.

        ``stream_time_offset`` is what stops this firing twice for one boundary:
        it is rebased to the new turn's start, which is at or past the boundary,
        so the test only passes again once a *later* run has closed.
        """
        if req.offset <= 0 or self._vad is None:
            return
        boundary = self._vad.turn_boundary(req.request_id)
        if boundary is None:
            return
        if window_start_s < boundary or req.stream_time_offset >= boundary:
            return
        self._close_turn(req, outputs)

    def _close_turn(self, req: Request, outputs: List[RequestOutput]) -> None:
        """Finalize the open turn, fold it into the stream, and rewind the stream.

        Order is load-bearing.  The decoder is finalized *before* the encoder
        cache is reset, because finalizing reads the beam the encoder's output
        built; resetting first would finalize against a stream that no longer has
        the frames its hypotheses came from.
        """
        final = self._op.finalize_streaming(req)
        self._commit_turn(req, final)
        self._op.free_session(req)
        self._op.create_session(req)
        self._mr.reset_stream(req)  # encoder cache + req.offset, together
        req.turn_index += 1
        self._metrics.incr(m.VAD_SEGMENTS, 1.0)

        # Publish the closed turn now rather than waiting for the next forward:
        # a stream sitting out a long pause produces no output at all otherwise,
        # and the transcript the client is holding would stay a chunk short of
        # what the engine already knows for the length of the silence.
        turn = RequestOutput(
            request_id=req.request_id,
            text=req.committed_text,
            tokens=[list(req.committed_tokens)],
            finished=False,
            endpoint_reason=_TURN_BOUNDARY,
        )
        if req.committed_timestamps:
            turn.timestamps = list(req.committed_timestamps)
        if req.committed_words:
            turn.words = list(req.committed_words)
        self._attach_vad_events([turn])
        if turn.text or turn.speech_events:
            outputs.append(turn)

    def _commit_turn(self, req: Request, final: RequestOutput) -> None:
        """Fold one closed turn's transcript into the stream's running one.

        Everything time-valued is shifted from turn-local seconds into session
        seconds here, at the one moment both clocks are in hand.  A plain join,
        deliberately not ``longform.merge_texts``: that one drops text an
        *overlap* duplicated, and turns share no audio — so its word-overlap
        dedup could only ever delete a word the speaker really did repeat across
        the pause.
        """
        offset = req.stream_time_offset
        text = (final.text or "").strip()
        if text:
            req.committed_text = f"{req.committed_text} {text}" if req.committed_text else text
        if final.tokens and final.tokens[0]:
            req.committed_tokens.extend(final.tokens[0])
        for start, end in final.timestamps or ():
            req.committed_timestamps.append((start + offset, end + offset))
        for word in final.words or ():
            req.committed_words.append(
                word._replace(start=word.start + offset, end=word.end + offset)
            )
        if final.confidence is not None:
            req.committed_confidences.append(float(final.confidence))

    def _apply_turn_carry(self, req: Request, out: RequestOutput) -> None:
        """Put one output back into session time, and in front of it the turns already closed.

        Applies to partials as well as to the final, because a client that saw a
        transcript shrink at a turn boundary would have no way to tell that from
        the recogniser changing its mind.  ``stream_time_offset`` alone is enough
        to need the shift: a stream whose *leading* silence was skipped has
        closed no turn yet and still decodes from a non-zero second.
        """
        offset = req.stream_time_offset
        if offset:
            if out.timestamps:
                out.timestamps = [(s + offset, e + offset) for s, e in out.timestamps]
            if out.words:
                out.words = [
                    w._replace(start=w.start + offset, end=w.end + offset) for w in out.words
                ]
        if not req.has_committed_turns:
            return
        parts = [p for p in (req.committed_text, out.text) if p]
        out.text = " ".join(parts)
        # One row: n-best across turns is refused at admission, so collapsing to
        # the best hypothesis loses nothing a caller could have asked for.
        tail = out.tokens[0] if out.tokens else []
        out.tokens = [list(req.committed_tokens) + list(tail)]
        if req.committed_timestamps or out.timestamps:
            out.timestamps = list(req.committed_timestamps) + list(out.timestamps or ())
        if req.committed_words or out.words:
            out.words = list(req.committed_words) + list(out.words or ())
        scores = list(req.committed_confidences)
        if out.confidence is not None:
            scores.append(float(out.confidence))
        if scores:
            out.confidence = sum(scores) / len(scores)

    def _apply_turn_carry_batch(
        self, requests: List[Request], outputs: List[RequestOutput]
    ) -> None:
        """:meth:`_apply_turn_carry` over a tick's worth of outputs."""
        if self._vad is None or not self._vad.gates_encoder or not outputs:
            return
        by_id = {r.request_id: r for r in requests}
        for out in outputs:
            req = by_id.get(out.request_id)
            if req is not None:
                self._apply_turn_carry(req, out)

    def _attach_vad_events(self, outputs: List[RequestOutput]) -> None:
        """Hang each stream's new events on the output it already produces.

        Deliberately not synthesised as a separate output: a ``RequestOutput``
        carries the transcript *so far*, and an event-only one would have to
        carry an empty ``text``, which a caption client renders by blanking the
        line.  At the default ``partial_decode_interval`` of 1 every active
        stream produces an output every tick, so events ride out immediately;
        at a coarser interval they are held until the next one, and the flush at
        finalize guarantees none are dropped.
        """
        if self._vad is None:
            return
        for out in outputs:
            events = self._vad.drain_events(out.request_id)
            if events:
                out.speech_events = events

    def abort(self, request_id: str) -> None:
        """Remove a streaming request, freeing its cache slot if any."""
        self._close_vad(request_id)
        req = self._scheduler.abort_request(request_id)
        # ``stream_id`` is the admission marker (set by the scheduler when a
        # stream is promoted to RUNNING) — present for both paged and stateful
        # backends, whereas ``stream_context`` is ``None`` for stateful streams.
        if req is not None and req.stream_id is not None:
            self._op.free_session(req)
            self._mr.free_stream(req)

    def _fail_cohort(
        self, cohort: List[Request], exc: BaseException, stage: str
    ) -> List[RequestOutput]:
        """Finalize a failed cohort with an error and free its caches.

        The batched forward has no per-stream boundary — one call covers every
        ready stream — so a failure inside it cannot be attributed to one of
        them after the fact.  Retrying the cohort one stream at a time would
        attribute it, but a partially-applied batched commit means the streams
        that *did* advance would have their chunk committed twice, silently
        corrupting KV.  Failing the cohort is the honest option.

        What this still buys is the point of the exercise: streams outside the
        cohort keep running, and the engine keeps ticking.  Before this, the
        exception escaped ``step()`` and the serving dispatcher turned it into
        an INTERNAL error for *every* in-flight request — a few such ticks and
        the process drained.
        """
        logger.warning(
            "streaming %s failed for %d stream(s) (%s: %s); finalizing them with an "
            "error and leaving the rest of the pool running",
            stage,
            len(cohort),
            type(exc).__name__,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        outputs: List[RequestOutput] = []
        for req in cohort:
            out = RequestOutput(
                request_id=req.request_id,
                text="",
                tokens=[[]],
                finished=True,
                finish_reason="error",
                error_stage=stage,
            )
            req.output = out
            req.state = RequestState.FINISHED
            outputs.append(out)
            self._close_vad(req.request_id)
            # Free unconditionally: the cache state after a failed forward is
            # unknown, and leaking a slot per failure exhausts the pool in a
            # way that looks like a capacity bug rather than an error path.
            for release in (self._op.free_session, self._mr.free_stream):
                try:
                    release(req)
                except Exception:  # noqa: BLE001 — best effort during teardown
                    logger.debug("failed releasing %s for %s", release, req.request_id)
            self._scheduler.finish_request(req.request_id)
        return outputs

    # ------------------------------------------------------------------
    # Per-step stages
    # ------------------------------------------------------------------

    def _extract_features(
        self, running: List[Request], outputs: List[RequestOutput]
    ) -> List[Request]:
        """Batched GPU fbank across every stream with pending audio.

        One kernel call for the whole active pool rather than N sequential fbank
        calls, issued on the dedicated feat stream when running on CUDA.  The
        producer->consumer ordering lives inside ``extract_streaming_batch`` (it
        is the one that appends into ``feature_buffer`` on the current stream);
        the wait here is a redundant belt for our own read of ``feature_buffer``,
        not the protection — putting the only wait here raced the append.

        Returns the surviving ``running`` set: a failure fails the cohort that
        was being extracted and leaves the rest of the pool alone.
        """
        needs_feat = [r for r in running if r.has_pending_audio]
        if not needs_feat:
            return running
        nvtx_push("extract_fbank")
        t0 = time.perf_counter()
        try:
            self._inp.extract_streaming_batch(needs_feat, cuda_stream=self._feat_stream)
            if self._feat_stream is not None:
                torch.cuda.current_stream(self._device).wait_stream(self._feat_stream)
        except Exception as exc:  # noqa: BLE001 — see _fail_cohort
            nvtx_pop()
            outputs.extend(self._fail_cohort(needs_feat, exc, "streaming_features"))
            return [r for r in running if r.state is not RequestState.FINISHED]
        nvtx_pop()
        self._metrics.observe_stage("streaming.features", time.perf_counter() - t0)
        return running

    def _forward(
        self, ready: List[Request], outputs: List[RequestOutput]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Issue one encoder chunk per ready stream.  Enqueued, not synchronised.

        ``None`` on failure, after the cohort has been failed — the caller drops
        the decode and re-filters its running set.
        """
        self._metrics.observe_batch(m.MODE_STREAMING, len(ready))
        nvtx_push("forward_streaming")
        t0 = time.perf_counter()
        try:
            log_probs_map = self._mr.forward_streaming_step(ready)
        except Exception as exc:  # noqa: BLE001 — see _fail_cohort
            nvtx_pop()
            outputs.extend(self._fail_cohort(ready, exc, "streaming_forward"))
            return None
        nvtx_pop()
        self._metrics.observe_stage("streaming.encode", time.perf_counter() - t0)
        return log_probs_map

    def _decode(
        self,
        ready: List[Request],
        log_probs_map: Dict[str, torch.Tensor],
        outputs: List[RequestOutput],
    ) -> bool:
        """Decode one partial per forwarded stream — the device->host boundary.

        Returns ``False`` if the cohort failed (the caller re-filters ``running``).
        """
        nvtx_push("decode_streaming")
        t0 = time.perf_counter()
        try:
            produced = self._op.decode_streaming_batch(ready, log_probs_map)
        except Exception as exc:  # noqa: BLE001 — see _fail_cohort
            nvtx_pop()
            outputs.extend(self._fail_cohort(ready, exc, "streaming_forward"))
            return False
        nvtx_pop()
        self._metrics.observe_stage("streaming.decode", time.perf_counter() - t0)
        self._advance_vad(ready, log_probs_map)
        self._apply_turn_carry_batch(ready, produced)
        self._attach_vad_events(produced)
        outputs.extend(produced)
        return True

    def _step_serial(
        self, running: List[Request], window: int, outputs: List[RequestOutput]
    ) -> List[Request]:
        """Extract -> forward -> decode, all within this step.

        Every stream's chunk is forwarded in the step that extracted it, so the
        pack's host cost sits in front of the encoder with an idle GPU.  That is
        the cost ``_step_pipelined`` removes; this order is kept as its A/B arm.
        """
        running = self._extract_features(running, outputs)
        if not running:
            return running
        ready = self._gate([r for r in running if r.has_ready_encoder_chunk(window)], outputs)
        if not ready:
            return running
        log_probs_map = self._forward(ready, outputs)
        if log_probs_map is None or not self._decode(ready, log_probs_map, outputs):
            return [r for r in running if r.state is not RequestState.FINISHED]
        return running

    def _step_pipelined(
        self, running: List[Request], window: int, outputs: List[RequestOutput]
    ) -> List[Request]:
        """Forward -> extract -> decode: the pack runs against the encoder.

        The streams forwarded here are the ones whose features the *previous*
        step extracted, which is what frees this step's extraction to run behind
        the encoder instead of in front of it.  The host cost it hides — the
        per-stream concat + scale + write into pinned staging — was the single
        largest block of GPU-idle in a streaming step, and removing it from the
        critical path measures **1.25x** end to end (256 utterances, pool 64,
        chunk 16, 5 interleaved rounds), transcripts byte-identical.  In the
        profile the step's GPU-idle goes 7.15 -> 5.87 ms and its idle gaps over
        0.2 ms total 168.0 -> 96.4 ms; both understate the win, because CUPTI
        inflates the very host work being hidden.

        Three orderings are load-bearing:

        * The extract must land **after** the forward is issued and **before**
          the decode.  The decode is the device->host readback, so anything
          after it is host work with a drained queue and overlaps nothing.
        * ``extract_streaming_batch`` appends into ``feature_buffer`` on the
          current stream, so the append is ordered behind the encoder kernels
          that read the window this step consumed — including the realloc-copy
          when a buffer grows or drops its consumed prefix.
        * A stream the extract fails is freed by ``_fail_cohort``, so the decode
          re-filters ``ready`` rather than reading a released session.  Such a
          stream still gets the partial for the chunk that *was* forwarded,
          followed by its terminal error — the same pair the serial order
          produces when a feature failure follows a successful step.
        """
        ready = self._gate([r for r in running if r.has_ready_encoder_chunk(window)], outputs)
        log_probs_map = self._forward(ready, outputs) if ready else None
        if ready and log_probs_map is None:
            ready = []
            running = [r for r in running if r.state is not RequestState.FINISHED]

        running = self._extract_features(running, outputs)

        if ready and log_probs_map is not None:
            alive = [r for r in ready if r.state is not RequestState.FINISHED]
            if alive and not self._decode(alive, log_probs_map, outputs):
                running = [r for r in running if r.state is not RequestState.FINISHED]
        return running

    def step(self) -> List[RequestOutput]:
        nvtx_push("streaming.schedule")
        t0 = time.perf_counter()
        newly_admitted, running = self._scheduler.schedule_streaming()
        nvtx_pop()
        self._metrics.observe_stage("streaming.schedule", time.perf_counter() - t0)

        outputs: List[RequestOutput] = []

        if newly_admitted:
            self._metrics.observe_queue_wait(m.MODE_STREAMING, newly_admitted)
            nvtx_push("allocate_stream")
            t0 = time.perf_counter()
            for req in newly_admitted:
                self._mr.allocate_stream(req)  # encoder KV + CNN cache
                self._op.create_session(req)  # decode-side beam state
                self._open_vad(req)  # speech-activity segmenter + endpointer
            nvtx_pop()
            self._metrics.observe_stage("streaming.allocate", time.perf_counter() - t0)

        if not running:
            return outputs

        # Ahead of the encoder, so the gate has a verdict for the window that is
        # about to be dispatched.  Newly admitted streams already have their
        # policy state (``_open_vad`` ran above), and a stream fed at ``admit``
        # has audio queued from before that.
        self._advance_audio_vad(running)

        window = self._config.decoding_window
        if self._lookahead:
            running = self._step_pipelined(running, window, outputs)
        else:
            running = self._step_serial(running, window, outputs)
        if not running:
            return outputs

        # Finalize closed streams after their ready features are consumed.
        # Cache-exhausted streams cannot progress, so return their partial
        # transcript with ``finish_reason="length"`` instead of holding a slot.
        nvtx_push("finalize_streams")
        t0 = time.perf_counter()
        for req in list(running):
            drained = (
                req.audio_final
                and (not req.has_pending_audio)
                and (not req.has_ready_encoder_chunk(window))
            )
            # An endpoint ends the turn with audio still arriving, which is the
            # whole point: the client keeps its socket open and the server stops
            # recognizing.  That is Google's `single_utterance` contract, and the
            # event it maps to has been declared in the proto and never emitted
            # since the Google-shaped surface landed.
            endpoint = self._vad.endpointed(req.request_id) if self._vad is not None else None
            if drained or req.cache_exhausted or endpoint is not None:
                final = self._op.finalize_streaming(req)
                self._op.fill_nbest_texts(req, final)
                self._apply_turn_carry(req, final)
                self._finish_vad(req, final, endpoint)
                if req.cache_exhausted:
                    # A truncated transcript, counted where it is decided.  The
                    # allocator itself cannot report this: the capacity gate
                    # exists precisely so the pool is never asked for a block
                    # it cannot give, so there is no failed allocation to see.
                    self._metrics.incr(m.KV_EXHAUSTED)
                    if final.finish_reason is None:
                        final.finish_reason = "length"
                req.output = final
                outputs.append(final)
                self._op.free_session(req)  # decode-side beam state
                self._mr.free_stream(req)  # encoder KV + CNN cache
                self._close_vad(req.request_id)
                self._scheduler.finish_request(req.request_id)
        nvtx_pop()
        self._metrics.observe_stage("streaming.finalize", time.perf_counter() - t0)

        return outputs

    def _finish_vad(
        self,
        req: Request,
        final: RequestOutput,
        endpoint: Optional["EndpointDecision"],
    ) -> None:
        """Flush this stream's speech activity onto its final output.

        The flush is what closes a run that was still open when the audio ended,
        so a stream that stops mid-word still reports a segment rather than
        dropping it.  ``endpoint_reason`` says *which* rule ended the turn; a
        turn that ended because the audio did leaves it ``None``, and the two are
        genuinely different things for a client deciding whether to keep the
        socket open.
        """
        if self._vad is None:
            return
        events, segments = self._vad.finish(req.request_id)
        if events:
            final.speech_events = (final.speech_events or []) + events
        if segments:
            final.segments = segments
        if endpoint is not None:
            final.endpoint_reason = endpoint.reason
            if final.finish_reason is None:
                final.finish_reason = "stop"
            self._metrics.incr(m.ENDPOINTS, 1.0)

    def record_gauges(self, metrics) -> None:
        """Report paged block-pool occupancy.

        ``None`` when the streaming backend keeps fixed per-slot state rather
        than a growing paged pool (``StatefulStreamingBackend``): there is no
        pool to report, and emitting zeros would make a healthy stateful engine
        look like one whose pool never fills.
        """
        pool = getattr(self._mr, "_block_pool", None)
        if pool is None:
            return
        total = pool.num_total_blocks
        metrics.set_gauge(m.KV_BLOCKS_USED, float(total - pool.num_free_blocks))
        metrics.set_gauge(m.KV_BLOCKS_CAPACITY, float(total))

    def has_pending(self) -> bool:
        return self._scheduler.num_waiting_streaming > 0 or self._scheduler.num_running > 0

    def num_running(self) -> int:
        return self._scheduler.num_running

    def num_waiting(self) -> int:
        return self._scheduler.num_waiting_streaming

    def find_request(self, request_id: str) -> Optional[Request]:
        req = self._scheduler.find_request(request_id)
        # Only surface streaming-mode requests to the engine.  An offline
        # request that happens to share the scheduler instance would
        # otherwise be visible here and confuse routing.
        if req is not None and not req.streaming:
            return None
        return req
