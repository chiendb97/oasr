# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Sequential executor for offline batch inference.

Runs a scheduled offline batch to completion: the scheduler partitions the
batch into encoder micro-batches (:meth:`Scheduler.split_offline_batch` — plain
length-bucketed chunks, padded-frame chunks, or gapless sequence-packed rows),
and this executor runs each one back-to-back on the default stream — batched
GPU fbank (:func:`oasr.features.batched.batched_fbank` / ``mfcc``) → encoder
forward → CTC decode → finalise.  Feature extraction is GPU-only, so there is no
CPU prep to hide behind GPU compute; micro-batches run sequentially with no
cross-step overlap.

Batch *selection* and *partitioning* both live in the scheduler; this class owns
only execution.  Sequence packing flips the forward call from the padded
``forward_offline`` to the gapless varlen ``forward_offline_packed`` — the
decode + finalise tail is identical for both.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.utils.nvtx import nvtx_pop, nvtx_push

from ..decode.base import EncodeOutput
from ..request import Request, RequestOutput, RequestState
from ..scheduler import Scheduler
from .base import Executor

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
        max_tick_ms: float = 0.0,
    ) -> None:
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
        self._max_tick_ms = float(max_tick_ms)
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
        if self._may_admit(budget_spent) and self._admission_open():
            batch = self._scheduler.schedule_offline()
            outputs.extend(self.run(batch))
        return outputs

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

        chunks, orig_indices = self._scheduler.split_offline_batch(batch)

        outputs: List[RequestOutput] = []
        for c in chunks:
            nvtx_push(f"offline.micro_batch[B={len(c)}]")
            nvtx_push("offline.collate")
            features, lengths = self._collate(c)
            nvtx_pop()
            outputs.extend(self._run_stage(c, features, lengths))
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
        outputs = self._op.strategy.advance(budget)
        for out in outputs:
            if not out.finished:
                continue
            req = self._pending.pop(out.request_id, None)
            if req is not None:
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
        nvtx_push("offline.forward")
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

        strategy = self._op.strategy
        if strategy.incremental:
            # Label-synchronous AR (AED/LLM): prefill only.  The requests park
            # in the pending pool in state RUNNING; ``step()`` drives them via
            # budgeted ``advance`` calls until the strategy finishes each one.
            nvtx_push("offline.prefill")
            strategy.begin_offline(chunk, enc_out, output_lengths)
            for req in chunk:
                req.state = RequestState.RUNNING
                self._pending[req.request_id] = req
            nvtx_pop()
            return []

        nvtx_push("offline.decode")
        outputs = self._op.decode_offline(enc_out, output_lengths)
        nvtx_pop()

        for req, out in zip(chunk, outputs):
            out.request_id = req.request_id
            out.finished = True
            self._op.fill_nbest_texts(req, out)
            req.output = out
            req.state = RequestState.FINISHED
        return outputs
