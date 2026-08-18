# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Executor for offline batch inference.

Runs a scheduled offline batch to completion: the scheduler partitions the
batch into encoder micro-batches (:meth:`Scheduler.split_offline_batch` — plain
length-bucketed chunks, padded-frame chunks, or gapless sequence-packed rows),
and this executor runs each one on the default stream — batched GPU fbank
(:func:`oasr.features.batched.batched_fbank` / ``mfcc``) → encoder forward → CTC
decode → finalise.

Feature extraction is GPU-only, so there is no CPU prep to hide behind GPU
compute; micro-batches run sequentially with no cross-step overlap.  Software-
pipelining them — enqueueing micro-batch *i+1*'s collation and forward before
finishing *i* on the host — was built and measured at **0.999x**, because in
practice a tick never has more than one micro-batch to pipeline: the frame and
count budgets are applied by ``schedule_offline`` when it selects the batch, so
``split_offline_batch`` returns it whole.  The GPU-idle stretch a profile shows
at the micro-batch boundary is really the *tick* boundary — scheduling and
admission between ``step()`` calls — and ``offline.finalize`` is instrumented
here to keep that attributable (it measures ~20 us, not the ~3 ms the boundary
costs).

Batch *selection* and *partitioning* both live in the scheduler; this class owns
only execution.  Sequence packing flips the forward call from the padded
``forward_offline`` to the gapless varlen ``forward_offline_packed`` — the
decode + finalise tail is identical for both.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.cache import DecoderKvExhausted
from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .. import metrics as m
from ..decode.base import EncodeOutput
from ..request import Request, RequestOutput, RequestState
from ..scheduler import Scheduler
from .base import Executor

logger = logging.getLogger(__name__)

#: Consecutive ticks that may skip admission because the decode budget was spent.
#: Bounds how long a saturated decode pool can defer new prefills; past this the
#: next tick admits regardless, so admission cannot starve indefinitely.
_MAX_SKIPPED_ADMITS = 8


class OfflineExecutor(Executor):
    """Execute scheduled offline batches as sequential micro-batches.

    See :class:`oasr.engine.executor.base.Executor` for the per-tick
    protocol.  Each ``step()`` pulls a batch from the scheduler, asks it to
    partition the batch into micro-batches, and runs them back-to-back
    (fbank → forward → decode → finalise), returning one final output per
    request.  There is no per-request running state and no cross-step
    overlap.

    With ``enable_packing=True`` the scheduler partitions into packed rows and
    the forward runs the gapless varlen attention — zero attention padding,
    bit-exact to ``B=1`` inference.
    """

    streaming: ClassVar[bool] = False

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        input_processor,
        model_runner,
        output_processor,
        device: torch.device,
        enable_packing: bool = False,
        decode_steps_per_tick: int = 32,
        max_decode_slots: Optional[int] = None,
        decode_kv_budget_gib: Optional[float] = None,
        max_tick_ms: float = 0.0,
        decode_admit_window_ms: float = 0.0,
        max_batch_size: int = 32,
        metrics: Optional[m.EngineMetrics] = None,
    ) -> None:
        if metrics is not None:
            self._metrics = metrics
        self._scheduler = scheduler
        self._inp = input_processor
        self._mr = model_runner
        self._op = output_processor
        self._device = device
        # Selects the forward variant in :meth:`_run_stage`; the scheduler's
        # partitioner reads the same flag off ``EngineConfig`` to emit packed
        # rows, so the two stay consistent for the engine's lifetime.
        self._enable_packing = bool(enable_packing)
        # Incremental (label-synchronous AR) decode support: requests begun
        # via ``strategy.begin_offline`` park here in state RUNNING and are
        # driven by ``strategy.advance(StepBudget)`` — at most
        # ``decode_steps_per_tick`` batched decoder steps **and** at most
        # ``max_tick_ms`` of wall clock per engine tick, so one tick always does
        # bounded work *in time* (the serving dispatcher's contract; a step count
        # alone does not bound it, since step cost is model-dependent).
        # ``max_decode_slots`` gates new-batch admission while the pending pool
        # is full.  All three are inert for one-shot strategies.
        self._decode_steps_per_tick = int(decode_steps_per_tick)
        self._max_decode_slots = max_decode_slots
        self._decode_kv_budget_gib = decode_kv_budget_gib
        self._max_tick_ms = float(max_tick_ms)
        # AR admission coalescing: hold a thin waiting queue briefly so
        # near-simultaneous arrivals prefill as one decode batch (see
        # :meth:`_batch_wide_enough`).  ``0`` disables.
        self._decode_admit_window_ms = float(decode_admit_window_ms)
        self._max_batch_size = int(max_batch_size)
        self._pending: Dict[str, Request] = {}
        # A tick that spent its whole budget advancing does not also prefill a new
        # micro-batch — prefill is the largest single blob in a tick (audio tower
        # + projector + an LM forward over the whole prompt), and stacking it on
        # top of a full decode budget defeats the point of bounding the tick.
        # Counted so admission can never starve: after ``_MAX_SKIPPED_ADMITS``
        # consecutive skips the next tick admits regardless.
        self._skipped_admits = 0

    # ------------------------------------------------------------------
    # Executor ABC
    # ------------------------------------------------------------------

    def admit(self, request: Request) -> None:
        """Prepare an offline request and enqueue it for batching."""
        self._inp.prepare_offline(request)
        if request.audio is not None and request.sample_rate:
            # After ``prepare_offline`` this is a 1-D float32 CPU waveform, so
            # the duration is exact rather than derived from a frame estimate.
            # ``.shape[-1]`` rather than ``.numel()``: the field is declared as
            # tensor-or-ndarray and only one of the two has ``numel``.
            n = int(request.audio.shape[-1])
            self._metrics.incr(m.AUDIO_SECONDS, n / request.sample_rate)
        self._scheduler.add_request(request)

    def feed_chunk(
        self,
        request_id: str,
        chunk: Union[torch.Tensor, "np.ndarray"],
        is_last: bool = False,
    ) -> None:
        raise NotImplementedError(
            "OfflineExecutor does not accept streaming audio chunks. "
            "Set service_mode='streaming' if you need feed_chunk."
        )

    def abort(self, request_id: str) -> None:
        """Drop a request — from the waiting queue, or (for incremental
        strategies) from the in-flight pending pool, releasing its decoder
        state via ``free_session``.

        One-shot offline requests don't allocate a per-stream cache, so there
        is nothing to free beyond the scheduler entry itself.
        """
        req = self._pending.pop(request_id, None)
        if req is not None:
            self._op.strategy.free_session(req)
            req.state = RequestState.FINISHED
            return
        self._scheduler.abort_request(request_id)

    def shutdown(self) -> None:
        """Release in-flight incremental decode state.

        Requests parked by an ``incremental`` strategy hold decoder-KV buffers
        (dense, capacity-preallocated for the speech-LLM path), so dropping the
        pool without telling the strategy leaks them until the next GC pass.
        One-shot strategies never park and this is a no-op for them.
        """
        strategy = self._op.strategy
        for req in list(self._pending.values()):
            try:
                strategy.free_session(req)
            except Exception:  # pragma: no cover - defensive
                pass
            req.state = RequestState.FINISHED
        self._pending.clear()

    def step(self) -> List[RequestOutput]:
        """One engine tick, always bounded work.

        One-shot strategies (CTC / WFST / transducer / rescoring): pull a
        batch and run it to completion — unchanged.  Incremental strategies
        (AED / LLM): first advance every pending request within the tick budget
        (``decode_steps_per_tick`` steps *and* ``max_tick_ms`` of wall clock),
        then — if the budget wasn't spent and decode slots remain — admit +
        encode + prefill one new batch.
        """
        outputs: List[RequestOutput] = []
        strategy = self._op.strategy
        budget_spent = False
        if strategy.incremental and strategy.has_pending():
            outputs, budget_spent = self._advance_pending()
        if self._may_admit(budget_spent) and self._batch_wide_enough() and self._admission_open():
            # Cap the batch at the decode slots actually free.  ``_admission_open``
            # only answers "is there *a* slot"; without this limit a tick with one
            # free slot would still pull a full ``max_batch_size`` batch and
            # prefill all of it, overshooting ``max_decode_slots`` by up to
            # ``max_batch_size - 1`` requests' worth of decoder KV — an OOM path,
            # not a slowdown, since prefill preallocates per row.
            batch = self._scheduler.schedule_offline(limit=self._admission_limit())
            outputs.extend(self.run(batch))
        return outputs

    def _reject(self, request: Request, reason: str, stage: str = "unknown") -> RequestOutput:
        """Terminal output for a request the executor could not run.

        Marked ``finished`` with ``finish_reason="error"`` so the caller sees a
        result instead of waiting on a request that will never run; the serving
        layer maps the empty transcript + reason onto its error envelope, and
        ``stage`` becomes the label on ``oasr_requests_failed_total``.
        """
        request.state = RequestState.FINISHED
        out = RequestOutput(
            request_id=request.request_id,
            text="",
            tokens=[[]],
            finished=True,
            finish_reason="error",
            error_stage=stage,
        )
        request.output = out
        logger.debug("rejected request %s at %s: %s", request.request_id, stage, reason)
        return out

    def _admission_limit(self) -> Optional[int]:
        """Decode slots free right now; ``None`` for one-shot strategies.

        One-shot families finalise within the tick that admits them, so they hold
        no slots and are bounded by ``max_batch_size`` alone.
        """
        strategy = self._op.strategy
        if not strategy.incremental:
            return None
        limits = []
        if self._max_decode_slots is not None:
            limits.append(max(0, int(self._max_decode_slots) - len(self._pending)))
        rows_by_bytes = self._rows_within_kv_budget(strategy)
        if rows_by_bytes is not None:
            limits.append(rows_by_bytes)
        return min(limits) if limits else None

    def _rows_within_kv_budget(self, strategy) -> Optional[int]:
        """Rows still affordable under ``decode_kv_budget_gib``, or ``None``.

        Budgeting by request count does not bound decoder-KV memory: the
        footprint of a row is its position budget times the model's per-token
        rate, and prefill preallocates all of it up front.  Returns ``None``
        when either the budget or the model's per-row footprint is unknown, so
        an unmeasurable model keeps today's slot-only behaviour rather than
        being throttled by a guess.
        """
        budget_gib = self._decode_kv_budget_gib
        if not budget_gib:
            return None
        per_row = strategy.kv_bytes_per_row()
        if not per_row:
            return None
        total = int(budget_gib * (1024**3))
        in_flight = len(self._pending) * per_row
        return max(0, (total - in_flight) // per_row)

    def _batch_wide_enough(self) -> bool:
        """Whether the waiting queue should be prefilled now, or held to widen.

        Only for incremental strategies, and only when
        ``EngineConfig.decode_admit_window_ms`` is set.  An AR decoder step is
        weight-read bound, so two decode groups cost roughly twice one group of
        the same total rows — total forwards is the sum over groups of each
        group's step count.  The strategy now merges groups after the fact
        (``IncrementalArStrategy._absorb``), so this window is no longer what
        saves those forwards; what it still saves is the merge's copy and one
        encoder + prefill pass per extra arrival.

        Holds until either the queue reaches ``max_batch_size`` or the oldest
        waiting request has waited out the window.
        """
        window_ms = self._decode_admit_window_ms
        if window_ms <= 0 or not self._op.strategy.incremental:
            return True
        n_waiting = self._scheduler.num_waiting_offline
        if n_waiting == 0:
            return True  # nothing to hold
        if n_waiting >= self._max_batch_size:
            return True  # as wide as the engine will ever forward
        oldest = self._scheduler.oldest_offline_wait()
        return oldest is None or (oldest * 1000.0) >= window_ms

    def _may_admit(self, budget_spent: bool) -> bool:
        """Whether this tick should also prefill a new batch.

        Prefill is unbudgeted and the largest single blob in a tick, so a tick
        that already spent its decode budget skips it — otherwise the tick bound
        is ``budget + prefill`` and the deadline buys nothing.  Bounded skipping:
        after ``_MAX_SKIPPED_ADMITS`` consecutive skips a batch is admitted
        regardless, so a steady stream of long generations cannot starve
        admission indefinitely.
        """
        if not budget_spent:
            self._skipped_admits = 0
            return True
        if self._skipped_admits >= _MAX_SKIPPED_ADMITS:
            self._skipped_admits = 0
            return True
        self._skipped_admits += 1
        return False

    def has_pending(self) -> bool:
        return self._scheduler.num_waiting_offline > 0 or bool(self._pending)

    def num_running(self) -> int:
        """One-shot offline requests admit-and-finalise within a single
        ``step()`` and never park; incremental requests park in the pending
        pool until their strategy finishes them."""
        return len(self._pending)

    def num_waiting(self) -> int:
        return self._scheduler.num_waiting_offline

    def record_gauges(self, metrics) -> None:
        """Report decode-slot occupancy.

        A capacity of ``0`` means "no ceiling applies" — either the family is
        one-shot and parks nothing, or no ceiling was configured.  Reported as
        zero rather than omitted so the series exists from startup: a missing
        series and a scrape failure look identical on a dashboard, and a flat
        zero does not.
        """
        capacity = 0.0
        if self._op.strategy.incremental and self._max_decode_slots is not None:
            capacity = float(self._max_decode_slots)
        metrics.set_gauge(m.DECODE_SLOTS_IN_USE, float(len(self._pending)))
        metrics.set_gauge(m.DECODE_SLOTS_CAPACITY, capacity)

    def find_request(self, request_id: str) -> Optional[Request]:
        req = self._pending.get(request_id)
        if req is not None:
            return req
        return self._scheduler.find_request(request_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, batch: List[Request]) -> List[RequestOutput]:
        """Process ``batch`` and return outputs in the original input order.

        The scheduler partitions ``batch`` into micro-batches (and length-sorts
        them); this restores the original arrival order before returning.
        For incremental strategies this encodes + prefills only — the batch
        parks in the pending pool and outputs arrive from later ``step()``
        ticks, so the return is empty.
        """
        if not batch:
            return []

        # Queue wait is stamped here, at the one moment a request stops
        # waiting: the scheduler has just selected this batch, and nothing
        # between here and the forward can send a row back to the queue.
        self._metrics.observe_queue_wait(m.MODE_OFFLINE, batch)

        chunks, orig_indices = self._scheduler.split_offline_batch(batch)

        outputs: List[RequestOutput] = []
        for c in chunks:
            nvtx_push(f"offline.micro_batch[B={len(c)}]")
            outputs.extend(self._run_micro_batch(c))
            nvtx_pop()  # offline.micro_batch

        # Restore original arrival order (the length sort changed positions).
        # Incremental prefill produces no outputs — nothing to restore.
        if orig_indices is None or not outputs:
            return outputs
        restored: List[Optional[RequestOutput]] = [None] * len(outputs)
        for pos, orig in enumerate(orig_indices):
            restored[orig] = outputs[pos]
        return [r for r in restored if r is not None]

    # ------------------------------------------------------------------
    # Incremental decode (label-synchronous AR strategies)
    # ------------------------------------------------------------------

    def _admission_open(self) -> bool:
        """Whether this tick may admit a new offline batch.

        Always true for one-shot strategies.  For incremental strategies the
        pending pool gates admission: a full pool means the tick spends its
        budget advancing in-flight requests instead of prefilling new ones.
        """
        if not self._op.strategy.incremental:
            return True
        if self._max_decode_slots is None:
            return not self._pending
        return len(self._pending) < int(self._max_decode_slots)

    def _advance_pending(self) -> Tuple[List[RequestOutput], bool]:
        """Run one budgeted ``strategy.advance`` pass and finalise what it finished.

        Returns ``(outputs, budget_spent)``; ``budget_spent`` tells :meth:`step`
        whether this tick still has room to prefill a new batch.
        """
        from ..generation import StepBudget

        budget = StepBudget.for_tick(self._decode_steps_per_tick, self._max_tick_ms)
        t0 = time.perf_counter()
        outputs = self._op.strategy.advance(budget)
        self._metrics.observe_stage("offline.advance", time.perf_counter() - t0)
        for out in outputs:
            if not out.finished:
                continue
            req = self._pending.pop(out.request_id, None)
            if req is not None:
                # Same finalisation the one-shot path does.  ``fill_nbest_texts``
                # was missing here: it only mattered once an AR family could emit
                # more than one hypothesis, which beam search is exactly when —
                # before that every incremental final carried a single row and the
                # call would have been a no-op.
                self._op.fill_nbest_texts(req, out)
                req.output = out
                req.state = RequestState.FINISHED
        return outputs, budget.exhausted()

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _collate(
        self,
        chunk: List[Request],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build device-side (features, lengths) for one micro-batch.

        Runs the batched GPU fbank/mfcc over the whole micro-batch in one
        shot (see :meth:`InputProcessor.collate`).
        """
        return self._inp.collate(chunk)

    def _run_micro_batch(self, chunk: List[Request]) -> List[RequestOutput]:
        """Collate + run one micro-batch, isolating a failure to its cause.

        Without this, any exception in feature collation, the encoder forward,
        or CTC decode escapes ``step()`` — and the serving dispatcher turns a
        failed step into an INTERNAL error for *every* in-flight request.  One
        pathological input (a zero-length waveform, a NaN, an out-of-range
        vocab id) was a multi-tenant outage.

        On failure with more than one member the batch is re-run one request at
        a time, so the peers that were only guilty of sharing a tick still get
        their transcripts and the bad one is named.  That costs an extra pass
        over a batch that already failed, and only over that batch.  OOM is the
        exception: retrying under memory pressure is how a single over-large
        request turns into a cascade, so it rejects the batch outright.

        **Where** the failure happened decides how the retry runs.  ``collate``
        releases ``request.audio`` once the GPU feature tensor owns the batch, so
        a failure *after* it cannot re-run from the top — the re-collate dies on
        ``NoneType.size`` and that, not the real cause, is what every request in
        the batch is told.  It is not hypothetical: a conv kernel that failed on
        an over-wide batch reported an ``AttributeError`` about waveforms, and
        the shape that actually broke never reached a log line.  Past collate the
        isolation pass therefore re-runs each row against the features already
        built, which needs no waveform.
        """
        features: Optional[torch.Tensor] = None
        lengths: Optional[torch.Tensor] = None
        try:
            self._record_batch_shape(chunk)
            nvtx_push("offline.collate")
            t0 = time.perf_counter()
            try:
                features, lengths = self._collate(chunk)
            finally:
                # Balanced even when _collate raises: an unpaired push
                # mis-nests every range for the rest of a profiling session.
                nvtx_pop()
            self._metrics.observe_stage("offline.collate", time.perf_counter() - t0)
            return self._run_stage(chunk, features, lengths)
        except Exception as exc:  # noqa: BLE001 — one bad request must not take the tick
            return self._isolate_failure(chunk, exc, features, lengths)

    def _isolate_failure(
        self,
        chunk: List[Request],
        exc: BaseException,
        features: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
    ) -> List[RequestOutput]:
        """Turn one micro-batch's failure into per-request terminal outputs.

        ``features`` is what the failing micro-batch had already collated, or
        ``None`` if collation is where it died — which is what decides whether
        the isolation pass can start from the waveforms or has to start from the
        features (see :meth:`_run_micro_batch`).
        """
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            logger.warning(
                "offline micro-batch of %d ran out of memory; rejecting it "
                "(lower max_batch_size or the padded-waste cap): %s",
                len(chunk),
                exc,
            )
            return [self._reject(req, "out of memory", stage="offline_oom") for req in chunk]
        if len(chunk) == 1:
            logger.warning(
                "offline request %s failed: %s: %s",
                chunk[0].request_id,
                type(exc).__name__,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return [self._reject(chunk[0], f"{type(exc).__name__}: {exc}", stage="offline_forward")]
        outputs: List[RequestOutput] = []
        if features is None or lengths is None:
            # Collation itself failed, so the waveforms are still there and
            # the whole pipeline can be re-run per request.
            logger.warning(
                "offline micro-batch of %d failed collating (%s: %s); re-running one "
                "at a time to isolate the request responsible",
                len(chunk),
                type(exc).__name__,
                exc,
            )
            for req in chunk:
                outputs.extend(self._run_micro_batch([req]))
            return outputs
        if self._enable_packing:
            # Packed rows are a gapless concatenation, not one row per request,
            # so there is nothing to slice per request.  Reject with the real
            # error rather than a misleading one.
            logger.warning(
                "packed offline micro-batch of %d failed (%s: %s); rejecting it — "
                "a packed row cannot be split per request to isolate the cause",
                len(chunk),
                type(exc).__name__,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return [
                self._reject(req, f"{type(exc).__name__}: {exc}", stage="offline_forward")
                for req in chunk
            ]
        logger.warning(
            "offline micro-batch of %d failed after collation (%s: %s); re-running "
            "one row at a time over the features already built",
            len(chunk),
            type(exc).__name__,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        for i, req in enumerate(chunk):
            outputs.extend(self._run_collated_single(req, features[i : i + 1], lengths[i : i + 1]))
        return outputs

    def _run_collated_single(
        self,
        request: Request,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        """Re-run one already-collated row, isolating its failure to itself."""
        try:
            return self._run_stage([request], features, lengths)
        except torch.cuda.OutOfMemoryError as exc:
            logger.warning(
                "offline request %s ran out of memory on its own: %s", request.request_id, exc
            )
            return [self._reject(request, "out of memory", stage="offline_oom")]
        except Exception as exc:  # noqa: BLE001 — the whole point is to name this one
            logger.warning(
                "offline request %s failed: %s: %s",
                request.request_id,
                type(exc).__name__,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return [self._reject(request, f"{type(exc).__name__}: {exc}", stage="offline_forward")]

    def _record_batch_shape(self, chunk: List[Request]) -> None:
        """Record this micro-batch's width and how much of it is padding.

        Read off the **host-side** waveform lengths, before :meth:`_collate`
        releases them, for two reasons.  The device ``lengths`` tensor would
        need a ``.item()`` to sum, putting a device-to-host synchronisation on
        the collate path — a metric that slows down what it measures.  And the
        waveforms are gone afterwards: ``collate`` clears ``request.audio``
        once the GPU feature tensor owns the batch, which is also why the
        single-request retry path (where a peer's audio is already released)
        reports width but not padding rather than reporting a wrong ratio.
        """
        if not self._metrics.enabled:
            return
        self._metrics.observe_batch(m.MODE_OFFLINE, len(chunk))
        lengths = [int(r.audio.shape[-1]) for r in chunk if r.audio is not None]
        if len(lengths) == len(chunk):
            self._metrics.observe_padding(lengths)

    def _run_stage(
        self,
        chunk: List[Request],
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        """Forward + CTC decode + finalise on the default stream.

        Packing mode runs the gapless varlen ``forward_offline_packed`` over
        the one packed row; otherwise the padded ``forward_offline``.  Decode
        and finalisation are identical for both.
        """
        enc_out, output_lengths = self._encode_stage(chunk, features, lengths)
        if self._op.strategy.incremental:
            return self._prefill_stage(chunk, enc_out, output_lengths)
        return self._finalise_decoded(chunk, self._decode_encoded(chunk, enc_out, output_lengths))

    def _encode_stage(
        self,
        chunk: List[Request],
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor]:
        """Encoder forward for one micro-batch — enqueued, never synchronised.

        Kept separate from the decode so :meth:`_run_micro_batches_pipelined` can
        issue the next micro-batch's GPU work while the host is still finishing
        the previous one.
        """
        nvtx_push("offline.forward")
        t0 = time.perf_counter()
        consumes = self._op.strategy.consumes
        if consumes == "hidden":
            # Autoregressive families (transducer/AED/LLM) consume raw encoder
            # hidden states and own their decoder; CTC consumes the fused-head
            # log-probs (the CUDA-graph fast path).
            enc_out, output_lengths = self._mr.encode_offline(features, lengths)
        elif consumes == "both":
            # CTC+AED rescoring: one encoder pass, then the head applied on the
            # side — the strategy needs the CTC log-probs (n-best) *and* the
            # hidden states (decoder cross-attention memory).
            hidden, output_lengths = self._mr.encode_offline(features, lengths)
            enc_out = EncodeOutput(hidden=hidden, log_probs=self._mr.apply_head(hidden))
        elif self._enable_packing:
            enc_out, output_lengths = self._mr.forward_offline_packed(features, lengths)
        else:
            enc_out, output_lengths = self._mr.forward_offline(features, lengths)
        nvtx_pop()
        self._metrics.observe_stage("offline.encode", time.perf_counter() - t0)
        return enc_out, output_lengths

    def _prefill_stage(
        self,
        chunk: List[Request],
        enc_out,
        output_lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        """Label-synchronous AR (AED/LLM) prefill: park the batch, emit nothing.

        The requests sit in the pending pool in state RUNNING; ``step()`` drives
        them via budgeted ``advance`` calls until the strategy finishes each one.
        """
        strategy = self._op.strategy
        nvtx_push("offline.prefill")
        t_prefill = time.perf_counter()
        try:
            strategy.begin_offline(chunk, enc_out, output_lengths)
        except (torch.cuda.OutOfMemoryError, DecoderKvExhausted) as exc:
            # Prefill reserves this micro-batch's decoder KV — a capacity buffer
            # per group, or one paged slot per row — so it is where an
            # over-committed pool actually fails.  Reject *this batch* with an
            # attributable error rather than letting the exception escape
            # ``step()`` — the serving dispatcher turns a failed step into an
            # INTERNAL error for every in-flight request, so one over-large batch
            # would take down its peers.
            nvtx_pop()
            logger.warning(
                "decoder-KV prefill could not reserve memory for %d request(s); "
                "rejecting the batch (lower max_decode_slots / "
                "max_new_tokens, or raise the memory budget): %s",
                len(chunk),
                exc,
            )
            return [
                self._reject(req, "prefill out of memory", stage="prefill_oom") for req in chunk
            ]
        for req in chunk:
            req.state = RequestState.RUNNING
            self._pending[req.request_id] = req
        nvtx_pop()
        self._metrics.observe_stage("offline.prefill", time.perf_counter() - t_prefill)
        return []

    def _decode_encoded(
        self,
        chunk: List[Request],
        enc_out,
        output_lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        """Decode one micro-batch's encoder output — the device→host boundary."""
        nvtx_push("offline.decode")
        t_decode = time.perf_counter()
        # The micro-batch rides along in row order: a family that can time its
        # own output needs to know, before it decodes, which rows asked.
        outputs = self._op.decode_offline(enc_out, output_lengths, chunk)
        nvtx_pop()
        self._metrics.observe_stage("offline.decode", time.perf_counter() - t_decode)
        return outputs

    def _finalise_decoded(
        self,
        chunk: List[Request],
        outputs: List[RequestOutput],
    ) -> List[RequestOutput]:
        """Attach ids, render the n-best texts, mark the requests finished.

        Pure host work, and the largest single piece of it in an offline tick:
        detokenising a micro-batch measured ~3 ms with **no** GPU operation
        issued for its whole duration.  :meth:`_run_micro_batches_pipelined` runs
        it against the next micro-batch's forward for exactly that reason.
        """
        nvtx_push("offline.finalize")
        t0 = time.perf_counter()
        for req, out in zip(chunk, outputs):
            out.request_id = req.request_id
            out.finished = True
            self._op.fill_nbest_texts(req, out)
            req.output = out
            req.state = RequestState.FINISHED
        nvtx_pop()
        self._metrics.observe_stage("offline.finalize", time.perf_counter() - t0)
        return outputs
