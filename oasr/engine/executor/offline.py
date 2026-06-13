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

from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.utils.nvtx import nvtx_pop, nvtx_push

from ..request import Request, RequestOutput, RequestState
from ..scheduler import Scheduler
from .base import Executor


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
        """Drop a request from the offline waiting queue.

        Offline requests don't allocate a per-stream cache, so there is
        nothing to free beyond the scheduler entry itself.
        """
        self._scheduler.abort_request(request_id)

    def step(self) -> List[RequestOutput]:
        """One engine tick: pull a batch from the scheduler and run it to
        completion (split → fbank → forward → decode → finalise)."""
        batch = self._scheduler.schedule_offline()
        return self.run(batch)

    def has_pending(self) -> bool:
        return self._scheduler.num_waiting_offline > 0

    def num_running(self) -> int:
        """Offline requests admit-and-finalise within a single ``step()``;
        they never park in a running pool, so this is always ``0``."""
        return 0

    def num_waiting(self) -> int:
        return self._scheduler.num_waiting_offline

    def find_request(self, request_id: str) -> Optional[Request]:
        return self._scheduler.find_request(request_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, batch: List[Request]) -> List[RequestOutput]:
        """Process ``batch`` and return outputs in the original input order.

        The scheduler partitions ``batch`` into micro-batches (and length-sorts
        them); this restores the original arrival order before returning.
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
        if orig_indices is None:
            return outputs
        restored: List[Optional[RequestOutput]] = [None] * len(outputs)
        for pos, orig in enumerate(orig_indices):
            restored[orig] = outputs[pos]
        return [r for r in restored if r is not None]

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
        if self._op.strategy.consumes == "hidden":
            # Autoregressive families (transducer/AED/LLM) consume raw encoder
            # hidden states and own their decoder; CTC consumes the fused-head
            # log-probs (the CUDA-graph fast path).
            enc_out, output_lengths = self._mr.encode_offline(features, lengths)
        elif self._enable_packing:
            enc_out, output_lengths = self._mr.forward_offline_packed(features, lengths)
        else:
            enc_out, output_lengths = self._mr.forward_offline(features, lengths)
        nvtx_pop()
        nvtx_push("offline.decode")
        outputs = self._op.decode_offline(enc_out, output_lengths)
        nvtx_pop()

        for req, out in zip(chunk, outputs):
            out.request_id = req.request_id
            out.finished = True
            req.output = out
            req.state = RequestState.FINISHED
        return outputs
