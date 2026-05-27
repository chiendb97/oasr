# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Producer/consumer pipeline for offline batch inference.

Splits a scheduled offline batch into length-bucketed micro-batches and
runs them so that the GPU forward+decode on one micro-batch overlaps with
CPU prep (or GPU fbank) on the next.

Two collate modes are supported:

* **GPU fbank** (``gpu_feature_extraction=True``) — the batched Kaldi
  fbank / mfcc from :mod:`oasr.features.batched` runs on a dedicated
  CUDA stream.  It yields ~10× the feature-extraction throughput of a
  per-utterance ``torchaudio.compliance.kaldi.{fbank,mfcc}`` loop
  because it fuses the whole micro-batch into a single sequence of
  kernels with no Python per-utt overhead.  The feat stream runs ahead
  of the default-stream forward+decode, overlapping feature extraction
  of chunk ``k+1`` with encoder work on chunk ``k``.
* **CPU pool** (``gpu_feature_extraction=False``) — pre-submits the
  whole batch's fbank to the shared :class:`ThreadPoolExecutor` in
  chunk order, then harvests each micro-batch's futures on the producer
  thread.  The hardware copy engine gives free H2D / compute overlap.

The queue depth controls how many micro-batches may be in flight
simultaneously.  ``depth=1`` degenerates to sequential execution (no
producer thread, no queue).  ``depth=3`` is a good default — more than
that rarely helps because one micro-batch of GPU fbank is typically
enough to hide one micro-batch of encoder forward.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .request import Request, RequestOutput, RequestState


class OfflinePipeline:
    """Execute a list of offline requests with CPU/GPU overlap."""

    def __init__(
        self,
        *,
        input_processor,
        model_runner,
        output_processor,
        micro_batch_size: int,
        depth: int,
        device: torch.device,
        sort_by_length: bool = True,
        gpu_feature_extraction: bool = True,
        preferred_sizes: Optional[Sequence[int]] = None,
        persistent_producer: bool = True,
    ) -> None:
        self._inp = input_processor
        self._mr = model_runner
        self._op = output_processor
        self._micro_batch_size = max(1, int(micro_batch_size))
        self._depth = max(1, int(depth))
        self._device = device
        self._sort_by_length = sort_by_length
        self._gpu_feat = bool(gpu_feature_extraction) and device.type == "cuda"
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
        # Dedicated stream so GPU-side fbank of chunk k+1 can overlap with
        # the encoder forward + CTC decode of chunk k on the default stream.
        # Only created in GPU-feature mode; in CPU-feature mode the H2D
        # copy engine gives us free overlap without an extra stream.
        self._feat_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device)
            if self._gpu_feat else None
        )

        # ---- Persistent cross-step pipeline ----
        # When enabled, a long-lived producer thread pulls submitted batches
        # from `_incoming`, splits them into micro-batches, runs fbank /
        # collate, and parks the result on `_ready` for the engine's step
        # loop to drain (via :meth:`drain_ready`) on a later tick.  This
        # lets `_collate(batch[k+1])` overlap with `_gpu_stage(batch[k])`
        # **across** step boundaries — the within-batch pipelining in
        # :meth:`run` only overlaps when one batch is split into multiple
        # chunks, which doesn't happen when the scheduler hands out 10-20
        # req batches under HTTP-driven trickle admission.
        # Cross-step async pipeline state — see :meth:`submit` /
        # :meth:`drain_ready`.  Each ``submit`` spawns a short-lived
        # collate worker; outputs land on ``_ready`` and are consumed by
        # ``drain_ready`` on the engine's step loop.
        self._persistent_enabled = bool(persistent_producer)
        if self._persistent_enabled:
            self._ready: "queue.Queue[Optional[Tuple]]" = queue.Queue(
                maxsize=max(2, self._depth)
            )
            self._stop_event = threading.Event()
            self._in_flight = 0
            self._in_flight_lock = threading.Lock()
            self._producer_error: List[BaseException] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, batch: List[Request]) -> List[RequestOutput]:
        """Process ``batch`` and return outputs in the original input order."""
        if not batch:
            return []

        chunks, orig_indices = self._split_chunks(batch)

        # CPU-side fbank mode pre-submits all extractions to the shared
        # worker pool in chunk order so chunk-0 finishes first.  GPU-side
        # mode skips the pool — fbank runs on the feat stream inside
        # ``_collate`` and the pool is left idle.
        if self._gpu_feat:
            futures: Dict[str, "Future[torch.Tensor]"] = {}
        else:
            ordered: List[Request] = [r for c in chunks for r in c]
            futures = self._inp.prefetch_features(ordered)

        if self._depth == 1 or len(chunks) == 1:
            outputs = self._run_sequential(chunks, futures)
        else:
            outputs = self._run_pipelined(chunks, futures)

        # Restore original arrival order (sort-by-length changes positions).
        if orig_indices is None:
            return outputs
        restored: List[Optional[RequestOutput]] = [None] * len(outputs)
        for pos, orig in enumerate(orig_indices):
            restored[orig] = outputs[pos]
        return [r for r in restored if r is not None]

    # ------------------------------------------------------------------
    # Persistent cross-step pipeline
    # ------------------------------------------------------------------

    def submit(self, batch: List[Request]) -> None:
        """Hand ``batch`` off for async collation.

        Spawns a short-lived worker thread that splits ``batch`` into
        micro-batches, runs fbank / collate on the side stream, and parks
        each result on the internal ready queue.  Call :meth:`drain_ready`
        on this or a subsequent step to run ``_gpu_stage`` against whatever
        the worker has prepared.

        Spawning a fresh thread per submission matches the proven
        per-batch :meth:`_run_pipelined` pattern (which got engine-bench
        to 1830 utts/s) and avoids the GIL-starvation behavior of a single
        long-lived producer thread that empirically pays 100× per-collate
        overhead under sustained dispatcher ticking.

        When the persistent producer is disabled (``persistent_producer=False``
        in the constructor) this falls back to a synchronous :meth:`run`
        and buffers the outputs for the next ``drain_ready``.
        """
        if not batch:
            return
        if not self._persistent_enabled:
            outputs = self.run(batch)
            if outputs:
                if not hasattr(self, "_sync_buffered"):
                    self._sync_buffered: List[RequestOutput] = []
                self._sync_buffered.extend(outputs)
            return
        with self._in_flight_lock:
            self._in_flight += len(batch)
        batch_ref = batch  # capture by ref for the worker

        def worker() -> None:
            try:
                if self._device.type == "cuda":
                    idx = (
                        self._device.index
                        if self._device.index is not None
                        else torch.cuda.current_device()
                    )
                    torch.cuda.set_device(int(idx))
                chunks, _orig_indices = self._split_chunks(batch_ref)
                if self._gpu_feat:
                    futures: Dict[str, "Future[torch.Tensor]"] = {}
                else:
                    ordered: List[Request] = [r for c in chunks for r in c]
                    futures = self._inp.prefetch_features(ordered)
                for c in chunks:
                    if self._stop_event.is_set():
                        return
                    features, lengths, event = self._collate(c, futures)
                    # Blocking put — backpressures the worker when the
                    # consumer (engine.step) hasn't drained recent items.
                    self._ready.put((c, features, lengths, event))
            except BaseException as e:  # noqa: BLE001 — surface via drain
                self._producer_error.append(e)
                with self._in_flight_lock:
                    self._in_flight -= len(batch_ref)

        t = threading.Thread(
            target=worker,
            daemon=True,
            name="oasr-offline-collate",
        )
        t.start()

    def drain_ready(self, wait_first: float = 0.005) -> List[RequestOutput]:
        """Run ``_gpu_stage`` for every collated micro-batch currently in
        the ready queue.

        When ``wait_first > 0`` and there is in-flight pipeline work but
        nothing immediately ready, blocks up to ``wait_first`` seconds
        waiting for the first item.  Required for cross-step pipelining:
        without it the dispatcher's busy-loop (step → empty drain → step)
        starves the collate worker of GIL and per-collate latency balloons
        from ~5 ms to ~500 ms.  The blocking sleep parks the engine thread,
        letting the worker thread acquire the GIL and do its Python work.
        After the first item arrives (or the timeout expires) the rest of
        the queue is drained non-blocking.

        Surfaces producer-thread errors as exceptions on the caller, one
        at a time per call.
        """
        if not self._persistent_enabled:
            buffered = getattr(self, "_sync_buffered", None)
            if buffered:
                outputs = list(buffered)
                buffered.clear()
                return outputs
            return []
        if self._producer_error:
            raise self._producer_error.pop(0)
        outputs: List[RequestOutput] = []
        first = True
        while True:
            try:
                if first and wait_first > 0 and self.in_flight() > 0:
                    item = self._ready.get(timeout=wait_first)
                else:
                    item = self._ready.get_nowait()
            except queue.Empty:
                break
            first = False
            chunk, features, lengths, event = item
            chunk_outs = self._gpu_stage(chunk, features, lengths, event)
            outputs.extend(chunk_outs)
            with self._in_flight_lock:
                self._in_flight -= len(chunk_outs)
        return outputs

    def in_flight(self) -> int:
        """Requests currently submitted but not yet returned by
        :meth:`drain_ready`.  Used by the engine to know whether to keep
        looping ``step()`` after the scheduler queues are empty."""
        if not self._persistent_enabled:
            return len(getattr(self, "_sync_buffered", []))
        with self._in_flight_lock:
            return self._in_flight

    def shutdown(self) -> None:
        """Signal in-flight collate workers to stop ASAP.  Best-effort;
        already-collated items remain in :attr:`_ready` and are drained
        by the next :meth:`drain_ready`."""
        if not self._persistent_enabled:
            return
        self._stop_event.set()

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
    # Execution modes
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        chunks: List[List[Request]],
        futures: Dict[str, "Future[torch.Tensor]"],
    ) -> List[RequestOutput]:
        outputs: List[RequestOutput] = []
        for c in chunks:
            features, lengths, event = self._collate(c, futures)
            outputs.extend(self._gpu_stage(c, features, lengths, event))
        return outputs

    def _run_pipelined(
        self,
        chunks: List[List[Request]],
        futures: Dict[str, "Future[torch.Tensor]"],
    ) -> List[RequestOutput]:
        Item = Tuple[
            List[Request], torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]
        ]
        q: "queue.Queue[Optional[Item]]" = queue.Queue(maxsize=self._depth)
        error: List[BaseException] = []

        def producer() -> None:
            try:
                for c in chunks:
                    features, lengths, event = self._collate(c, futures)
                    q.put((c, features, lengths, event))
            except BaseException as e:  # noqa: BLE001 — propagate to consumer
                error.append(e)
            finally:
                q.put(None)

        prod = threading.Thread(
            target=producer,
            daemon=True,
            name="oasr-offline-producer",
        )
        prod.start()

        outputs: List[RequestOutput] = []
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                c, features, lengths, event = item
                outputs.extend(self._gpu_stage(c, features, lengths, event))
        finally:
            prod.join()

        if error:
            raise error[0]
        return outputs

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _collate(
        self,
        chunk: List[Request],
        futures: Dict[str, "Future[torch.Tensor]"],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]]:
        """Build device-side (features, lengths) for one micro-batch.

        GPU-feature mode runs fbank on ``_feat_stream`` and records an
        event so the default-stream consumer knows when the data is
        ready.  CPU mode pads+pins on the producer thread and issues the
        pinned H2D; the hardware copy engine already overlaps that with
        default-stream compute without an explicit stream wrapper.
        """
        if self._gpu_feat:
            assert self._feat_stream is not None
            with torch.cuda.stream(self._feat_stream):
                features, lengths = self._inp.collate_gpu(chunk)
                event = torch.cuda.Event()
                event.record(self._feat_stream)
            return features, lengths, event

        padded_cpu, lengths_cpu = self._inp.collate_cpu(chunk, futures)
        non_blocking = self._device.type == "cuda"
        features = padded_cpu.to(
            device=self._device, dtype=self._dtype, non_blocking=non_blocking,
        )
        lengths = lengths_cpu.to(device=self._device, non_blocking=non_blocking)
        return features, lengths, None

    def _gpu_stage(
        self,
        chunk: List[Request],
        features: torch.Tensor,
        lengths: torch.Tensor,
        event: Optional[torch.cuda.Event],
    ) -> List[RequestOutput]:
        """Forward + CTC decode + finalise on the default stream."""
        if event is not None:
            event.wait(torch.cuda.current_stream(self._device))

        log_probs, output_lengths = self._mr.forward_offline(features, lengths)
        outputs = self._op.decode_offline(log_probs, output_lengths)

        for req, out in zip(chunk, outputs):
            out.request_id = req.request_id
            out.finished = True
            req.output = out
            req.state = RequestState.FINISHED
        return outputs
