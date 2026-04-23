# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Producer/consumer pipeline for offline batch inference.

Splits a scheduled offline batch into length-bucketed micro-batches and
runs them so that the GPU forward+decode on one micro-batch overlaps with
CPU prep (or GPU fbank) on the next.

Two collate modes are supported:

* **GPU fbank** (``gpu_feature_extraction=True``) — the batched Kaldi
  fbank from :mod:`oasr.features.gpu_fbank` runs on a dedicated CUDA
  stream.  It yields ~10× the fbank throughput of a per-utterance
  ``torchaudio.compliance.kaldi.fbank`` loop because it fuses the whole
  micro-batch into a single sequence of kernels with no Python per-utt
  overhead.  The feat stream runs ahead of the default-stream
  forward+decode, overlapping fbank of chunk ``k+1`` with encoder work
  on chunk ``k``.
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
from typing import Dict, List, Optional, Tuple

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
        # Dedicated stream so GPU-side fbank of chunk k+1 can overlap with
        # the encoder forward + CTC decode of chunk k on the default stream.
        # Only created in GPU-feature mode; in CPU-feature mode the H2D
        # copy engine gives us free overlap without an extra stream.
        self._feat_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device)
            if self._gpu_feat else None
        )

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

        if n <= mb:
            return [list(batch)], None

        if self._sort_by_length:
            enumerated = sorted(enumerate(batch), key=lambda p: p[1].num_frames)
            ordered = [r for _, r in enumerated]
            orig_indices = [i for i, _ in enumerated]
        else:
            ordered = list(batch)
            orig_indices = None

        # Balance: avoid a tiny trailing micro-batch that wastes launch overhead.
        nchunks = (n + mb - 1) // mb
        base, rem = divmod(n, nchunks)
        chunks: List[List[Request]] = []
        idx = 0
        for i in range(nchunks):
            size = base + (1 if i < rem else 0)
            chunks.append(ordered[idx: idx + size])
            idx += size
        return chunks, orig_indices

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
