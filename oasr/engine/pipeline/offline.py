# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Sequential pipeline for offline batch inference.

Splits a scheduled offline batch into length-bucketed micro-batches and
runs each one to completion before the next: batched GPU fbank
(:func:`oasr.features.batched.batched_fbank` / ``mfcc``) → encoder forward
→ CTC decode → finalise.  Feature extraction is GPU-only, so there is no
CPU prep to hide behind GPU compute and the old multi-micro-batch overlap
(producer thread + dedicated feature stream) is gone — micro-batches run
back-to-back on the default stream.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..request import Request, RequestOutput, RequestState
from ..scheduler import Scheduler
from .base import Pipeline


class OfflinePipeline(Pipeline):
    """Execute offline batches as sequential length-bucketed micro-batches.

    See :class:`oasr.engine.pipeline.base.Pipeline` for the per-tick
    protocol.  Each ``step()`` pulls a batch from the scheduler, splits it
    into length-bucketed micro-batches, and runs them back-to-back
    (fbank → forward → decode → finalise), returning one final output per
    request.  There is no per-request running state and no cross-step
    overlap.
    """

    streaming: ClassVar[bool] = False

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        input_processor,
        model_runner,
        output_processor,
        micro_batch_size: int,
        device: torch.device,
        sort_by_length: bool = True,
        preferred_sizes: Optional[Sequence[int]] = None,
        max_batch_frames: Optional[int] = None,
    ) -> None:
        self._scheduler = scheduler
        self._inp = input_processor
        self._mr = model_runner
        self._op = output_processor
        self._micro_batch_size = max(1, int(micro_batch_size))
        self._device = device
        self._sort_by_length = sort_by_length
        # Padded-frame budget per micro-batch (``max_len * count`` in
        # pre-subsampling feature frames).  When set, ``_split_chunks`` packs
        # length-sorted requests into chunks bounded by this budget instead of
        # by a fixed count — the non-packing length-aware batching path.
        self._max_batch_frames: Optional[int] = (
            int(max_batch_frames) if max_batch_frames is not None else None
        )
        self._dtype: torch.dtype = input_processor._config.dtype
        # Sorted ascending list of preferred chunk widths.  When set,
        # ``_split_chunks`` greedily peels off chunks of the largest
        # preferred value ``<= remaining``; the tail (always smaller than
        # ``min(preferred_sizes)``, rare when the scheduler also snaps)
        # ships as one final odd chunk.  ``None`` keeps the legacy
        # balanced even-split behaviour.
        self._preferred_sizes: Optional[List[int]] = (
            sorted({int(v) for v in preferred_sizes if int(v) >= 1})
            if preferred_sizes else None
        )

    # ------------------------------------------------------------------
    # Pipeline ABC
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
            "OfflinePipeline does not accept streaming audio chunks. "
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
        """Process ``batch`` and return outputs in the original input order."""
        if not batch:
            return []

        chunks, orig_indices = self._split_chunks(batch)

        outputs: List[RequestOutput] = []
        for c in chunks:
            features, lengths = self._collate(c)
            outputs.extend(self._gpu_stage(c, features, lengths))

        # Restore original arrival order (sort-by-length changes positions).
        if orig_indices is None:
            return outputs
        restored: List[Optional[RequestOutput]] = [None] * len(outputs)
        for pos, orig in enumerate(orig_indices):
            restored[orig] = outputs[pos]
        return [r for r in restored if r is not None]

    # ------------------------------------------------------------------
    # Micro-batching
    # ------------------------------------------------------------------

    def _split_chunks(
        self, batch: List[Request]
    ) -> Tuple[List[List[Request]], Optional[List[int]]]:
        """Split ``batch`` into length-bucketed micro-batches.

        Returns ``(chunks, orig_indices)``.  ``orig_indices[pos]`` is the
        input index of the request at flat output position ``pos``; it is
        ``None`` when no reordering was applied.
        """
        n = len(batch)
        mb = self._micro_batch_size

        # Frame-budget split takes precedence when configured: bound padded
        # compute (``max_len * count``) per micro-batch regardless of count,
        # so a mixed short/long pool never forms an over-padded forward.
        if self._max_batch_frames is not None:
            return self._split_by_frames(batch)

        # Skip the short-circuit when preferred sizes are configured —
        # we still want the greedy snap (e.g. n=7, mb=8, PBS=[4] should
        # yield [4, 3] not [7]) below.
        if n <= mb and not self._preferred_sizes:
            return [list(batch)], None

        if self._sort_by_length:
            enumerated = sorted(enumerate(batch), key=lambda p: p[1].num_frames)
            ordered = [r for _, r in enumerated]
            orig_indices = [i for i, _ in enumerated]
        else:
            ordered = list(batch)
            orig_indices = None

        chunks: List[List[Request]] = []
        if self._preferred_sizes:
            # Greedy snap: each chunk is the largest preferred value that
            # fits the remaining count, capped by ``mb``.  Scheduler
            # snapping makes the tail rare; when present it ships as one
            # final odd chunk.
            idx = 0
            while idx < n:
                remaining = n - idx
                size = self._snap_to_preferred(min(remaining, mb))
                if size == 0:
                    size = remaining  # tail < min(preferred); one odd chunk.
                chunks.append(ordered[idx: idx + size])
                idx += size
        else:
            # Balance: avoid a tiny trailing micro-batch that wastes launch overhead.
            nchunks = (n + mb - 1) // mb
            base, rem = divmod(n, nchunks)
            idx = 0
            for i in range(nchunks):
                size = base + (1 if i < rem else 0)
                chunks.append(ordered[idx: idx + size])
                idx += size
        return chunks, orig_indices

    def _split_by_frames(
        self, batch: List[Request]
    ) -> Tuple[List[List[Request]], Optional[List[int]]]:
        """Split ``batch`` into micro-batches bounded by a padded-frame budget.

        Length-sorts (when enabled) then greedily accumulates requests into a
        chunk until adding the next would push the padded width
        ``max_len * (count + 1)`` over ``max_batch_frames`` — or the count over
        ``micro_batch_size``.  A single request always ships even if it alone
        exceeds the budget (it can't be split in non-packing mode).  Sorting
        ascending keeps each chunk length-tight, minimising padding.
        """
        budget = self._max_batch_frames
        assert budget is not None
        mb = self._micro_batch_size

        if self._sort_by_length:
            enumerated = sorted(enumerate(batch), key=lambda p: p[1].num_frames)
            ordered = [r for _, r in enumerated]
            orig_indices: Optional[List[int]] = [i for i, _ in enumerated]
        else:
            ordered = list(batch)
            orig_indices = None

        chunks: List[List[Request]] = []
        cur: List[Request] = []
        cur_max = 0
        for r in ordered:
            rlen = max(1, r.num_frames)
            new_max = max(cur_max, rlen)
            if cur and (new_max * (len(cur) + 1) > budget or len(cur) >= mb):
                chunks.append(cur)
                cur = [r]
                cur_max = rlen
            else:
                cur.append(r)
                cur_max = new_max
        if cur:
            chunks.append(cur)
        return chunks, orig_indices

    def _snap_to_preferred(self, candidate: int) -> int:
        """Largest preferred chunk size ``<= candidate``; 0 if none fits."""
        pbs = self._preferred_sizes
        if not pbs or candidate < pbs[0]:
            return 0
        snapped = 0
        for v in pbs:
            if v <= candidate:
                snapped = v
            else:
                break
        return snapped

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _collate(
        self,
        chunk: List[Request],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build device-side (features, lengths) for one micro-batch.

        Runs the batched GPU fbank/mfcc over the whole micro-batch in one
        shot (see :meth:`InputProcessor.collate_gpu`).
        """
        return self._inp.collate_gpu(chunk)

    def _gpu_stage(
        self,
        chunk: List[Request],
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        """Forward + CTC decode + finalise on the default stream."""
        log_probs, output_lengths = self._mr.forward_offline(features, lengths)
        outputs = self._op.decode_offline(log_probs, output_lengths)

        for req, out in zip(chunk, outputs):
            out.request_id = req.request_id
            out.finished = True
            req.output = out
            req.state = RequestState.FINISHED
        return outputs
