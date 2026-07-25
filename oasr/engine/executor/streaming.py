# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Chunk-by-chunk streaming executor with paged KV cache."""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Union

import numpy as np
import torch

from oasr.utils.nvtx import nvtx_pop, nvtx_push

from ..config import EngineConfig
from ..input_processor import InputProcessor
from ..model_runner import ModelRunner
from ..output_processor import OutputProcessor
from ..request import Request, RequestOutput
from ..scheduler import Scheduler
from .base import Executor


class StreamingExecutor(Executor):
    """Streaming inference with one encoder chunk per active stream per tick.

    See :class:`oasr.engine.executor.base.Executor` for the per-tick
    protocol.  Each :meth:`step`:

    1. Admits waiting streams up to ``max_batch_size`` and allocates their
       paged KV / CNN / CTC caches.
    2. Runs one batched GPU fbank across every active stream that has
       pending audio (one kernel call for the whole pool, on a dedicated
       feat stream so it overlaps with the previous step's encoder
       forward).
    3. Runs ``forward_streaming_step`` against every stream whose feature
       buffer now holds a full encoder window, then decodes a partial
       per result.
    4. Finalises streams the client has explicitly closed
       (``audio_final``) once both the audio deque and the feature
       buffer are drained, freeing their cache slots.
    """

    streaming: ClassVar[bool] = True

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        input_processor: InputProcessor,
        model_runner: ModelRunner,
        output_processor: OutputProcessor,
        config: EngineConfig,
        device: torch.device,
    ) -> None:
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
        self._inp.append_streaming_chunk(req, chunk, is_last=is_last)

    def abort(self, request_id: str) -> None:
        """Remove a streaming request, freeing its cache slot if any."""
        req = self._scheduler.abort_request(request_id)
        # ``stream_id`` is the admission marker (set by the scheduler when a
        # stream is promoted to RUNNING) — present for both paged and stateful
        # backends, whereas ``stream_context`` is ``None`` for stateful streams.
        if req is not None and req.stream_id is not None:
            self._op.free_session(req)
            self._mr.free_stream(req)

    def step(self) -> List[RequestOutput]:
        nvtx_push("streaming.schedule")
        newly_admitted, running = self._scheduler.schedule_streaming()
        nvtx_pop()

        outputs: List[RequestOutput] = []

        if newly_admitted:
            nvtx_push("allocate_stream")
            for req in newly_admitted:
                self._mr.allocate_stream(req)  # encoder KV + CNN cache
                self._op.create_session(req)  # decode-side beam state
            nvtx_pop()

        if not running:
            return outputs

        # 1. Batched GPU fbank across every stream with pending audio —
        #    one kernel call for the whole active pool rather than N
        #    sequential fbank calls.  Issued on the dedicated feat stream
        #    when running on CUDA so it can overlap with the previous
        #    step's encoder forward; the default stream waits on the
        #    recorded event before reading feature_buffer.
        needs_feat = [r for r in running if r.has_pending_audio]
        if needs_feat:
            nvtx_push("extract_fbank")
            self._inp.extract_streaming_batch(
                needs_feat,
                cuda_stream=self._feat_stream,
            )
            if self._feat_stream is not None:
                torch.cuda.current_stream(self._device).wait_stream(self._feat_stream)
            nvtx_pop()

        # 2. For each stream whose feature buffer now holds at least one
        #    encoder window, run forward_chunk_paged.
        window = self._config.decoding_window
        ready = [r for r in running if r.has_ready_encoder_chunk(window)]
        if ready:
            nvtx_push("forward_streaming")
            log_probs_map: Dict[str, torch.Tensor] = self._mr.forward_streaming_step(ready)
            nvtx_pop()
            nvtx_push("decode_streaming")
            partials = self._op.decode_streaming_batch(ready, log_probs_map)
            outputs.extend(partials)
            nvtx_pop()

        # 3. Finalise streams whose audio is exhausted and whose feature
        #    buffer has been fully consumed.  A stream can reach this
        #    state in the same step it ran its last encoder chunk, so we
        #    check *after* the forward pass.  Only streams the client
        #    has explicitly closed (``audio_final``) are eligible — a
        #    freshly admitted streaming request that hasn't yet received
        #    audio is otherwise indistinguishable from a drained one and
        #    would be finalised with an empty transcript on the very
        #    first step.
        nvtx_push("finalize_streams")
        for req in list(running):
            if (
                req.audio_final
                and (not req.has_pending_audio)
                and (not req.has_ready_encoder_chunk(window))
            ):
                final = self._op.finalize_streaming(req)
                self._op.fill_nbest_texts(req, final)
                req.output = final
                outputs.append(final)
                self._op.free_session(req)  # decode-side beam state
                self._mr.free_stream(req)  # encoder KV + CNN cache
                self._scheduler.finish_request(req.request_id)
        nvtx_pop()

        return outputs

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
