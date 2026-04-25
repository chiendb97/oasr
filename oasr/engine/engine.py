# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR inference and serving engine."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from oasr.models.conformer.convert import load_wenet_checkpoint

from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .config import EngineConfig
from .input_processor import InputProcessor
from .model_runner import ModelRunner
from .output_processor import OutputProcessor
from .pipeline import OfflinePipeline
from .request import Request, RequestOutput
from .scheduler import Scheduler, SchedulerOutput

logger = logging.getLogger(__name__)


class ASREngine:
    """Unified ASR inference engine with dynamic batching.

    Handles both streaming (chunk-by-chunk with paged KV cache) and offline
    (single-pass batched) requests in one step loop.  Each step:

    1. **Schedule** — admit up to ``max_offline_batch_size`` waiting offline
       requests and admit waiting streaming requests up to
       ``max_batch_size``.
    2. **Ingest** — batched GPU fbank across every active stream's next
       pending audio chunk (one kernel call for the whole running pool,
       no stream ever sees audio beyond its own enqueued chunks).
    3. **Forward** — route the admitted offline batch through the
       pipelined :class:`OfflinePipeline` (length-bucketed micro-batches
       of size ``offline_micro_batch_size`` overlap GPU feature extraction
       with encoder forward + CTC decode) and run one encoder chunk per
       streaming request whose buffer now holds a full window.
    4. **Postprocess** — decode log-probs and finalise completed requests.

    Dynamic batching is length-aware: within the offline pipeline,
    requests are sorted by estimated feature length and split into
    similarly-sized micro-batches to minimise padded-compute waste.
    Starvation of bursty offline requests is bounded by ``max_wait_time``.

    Parameters
    ----------
    config : EngineConfig
        Fully configured engine settings.  ``ckpt_dir`` must be set.

    Examples
    --------
    Streaming transcription::

        engine = ASREngine(EngineConfig(ckpt_dir="/path/to/ckpt"))
        text = engine.transcribe("audio.wav")

    Explicit offline batch::

        ids = [engine.add_request(p, streaming=False) for p in paths]
        results = engine.run()
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        device_str = config.device
        dtype = config.dtype

        logger.info("Loading model from %s ...", config.ckpt_dir)
        model, model_config = load_wenet_checkpoint(
            config.ckpt_dir,
            config.checkpoint_name,
            device=device_str,
            dtype=dtype,
        )
        self._model = model
        config._model_config = model_config

        self._device = torch.device(device_str)
        cache_config = config.build_cache_config(model_config)

        self._input_processor = InputProcessor(config, self._device)
        self._scheduler = Scheduler(config)
        self._model_runner = ModelRunner(model, config, cache_config)
        self._output_processor = OutputProcessor(config)

        # Offline execution pipeline — handles CPU/GPU overlap and
        # length-bucketed micro-batching internally.
        micro = int(config.offline_micro_batch_size or 0)
        if micro <= 0:
            micro = min(config.max_offline_batch_size, 32)
        self._offline_pipeline = OfflinePipeline(
            input_processor=self._input_processor,
            model_runner=self._model_runner,
            output_processor=self._output_processor,
            micro_batch_size=micro,
            depth=max(1, int(config.offline_pipeline_depth)),
            device=self._device,
            gpu_feature_extraction=bool(config.offline_gpu_feature_extraction),
        )

        # Dedicated CUDA stream for the streaming fbank kernel.  Lets the
        # H2D waveform copy + batched fbank for the current step overlap
        # with the encoder forward of the previous step's tail (the
        # encoder forward is async, so once dispatched the GPU can run
        # both kernels concurrently when they live on different streams).
        self._feat_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda" else None
        )

    # ------------------------------------------------------------------
    # Request management
    # ------------------------------------------------------------------

    def add_request(
        self,
        audio: Union[str, torch.Tensor, "np.ndarray"],
        request_id: Optional[str] = None,
        sample_rate: int = 16000,
        streaming: bool = True,
        priority: int = 0,
    ) -> str:
        """Add a new request to the engine.

        Both paths defer actual feature extraction until the engine step
        loop can batch it: streaming ingests audio **chunk by chunk** inside
        ``step()`` (batched across all active streams in one GPU fbank
        call), and the offline pipeline batches fbank within each GPU
        micro-batch.  What :meth:`add_request` does synchronously is load
        the waveform, stamp a cheap but exact (Kaldi ``snip_edges``
        formula) frame count so the scheduler can bucket by length, and
        split the waveform into streaming audio chunks on the CPU.  No
        fbank runs until the request is admitted and stepped.

        Parameters
        ----------
        audio : str, Tensor, or ndarray
            Audio input (file path, waveform tensor, or NumPy array).
        request_id : str, optional
            Unique identifier.  Auto-generated if omitted.
        sample_rate : int
            Audio sample rate in Hz.
        streaming : bool, default ``True``
            ``True`` routes the request through the paged-cache streaming
            path; ``False`` routes it through the batched offline path.
        priority : int, default ``0``
            Lower values are scheduled first within each waiting queue.

        Returns
        -------
        str
            The assigned ``request_id``.
        """
        req = Request(
            audio,
            request_id=request_id,
            streaming=streaming,
            sample_rate=sample_rate,
            priority=priority,
        )
        if streaming:
            # Split the waveform into audio-sample chunks *without* running
            # fbank.  Feature extraction runs inside ``step()`` batched
            # across all active streams — realistic streaming behaviour
            # (one chunk's worth of audio produces one chunk's worth of
            # features) and a much better match for the model's forward
            # throughput than the old pre-extract-everything admission.
            self._input_processor.prepare_streaming(req)
        else:
            self._input_processor.prepare_offline(req)
        self._scheduler.add_request(req)
        return req.request_id

    def add_streaming_request(
        self,
        request_id: Optional[str] = None,
        sample_rate: int = 16000,
        priority: int = 0,
    ) -> str:
        """Open a chunk-by-chunk streaming request.

        Registers an empty streaming request with no audio attached.  Push
        audio into the engine via :meth:`feed_chunk` (one chunk per call).
        The engine starts processing chunks in the next :meth:`step` after
        the request is admitted by the scheduler — feeding chunks before
        admission just queues them on the request's audio deque.

        Parameters
        ----------
        request_id : str, optional
            Unique identifier.  Auto-generated (UUID4 hex) if omitted.
        sample_rate : int
            Sample rate of the audio that will be fed via :meth:`feed_chunk`.
        priority : int, default ``0``
            Lower values are scheduled first within the streaming queue.

        Returns
        -------
        str
            The assigned ``request_id`` — pass it to :meth:`feed_chunk`.
        """
        req = Request(
            audio=None,
            request_id=request_id,
            streaming=True,
            sample_rate=sample_rate,
            priority=priority,
        )
        self._input_processor.prepare_streaming_open(req)
        self._scheduler.add_request(req)
        return req.request_id

    def feed_chunk(
        self,
        request_id: str,
        chunk: Union[torch.Tensor, "np.ndarray"],
        is_last: bool = False,
    ) -> None:
        """Push one audio chunk into a streaming request.

        Parameters
        ----------
        request_id : str
            Id returned from :meth:`add_streaming_request`.
        chunk : Tensor or ndarray
            1-D audio samples for this chunk.  Convert to CPU float32 happens
            automatically; passing CPU tensors that are already in the right
            shape is fastest.
        is_last : bool, default ``False``
            Set ``True`` on the last chunk to flush the trailing partial
            frame and trigger finalisation as soon as the encoder drains.

        Notes
        -----
        Tolerates feeding chunks before admission (they queue on the
        request's audio deque) and after admission (they're consumed by the
        next :meth:`step`).  Raises if the request id is unknown or has
        already been finalised.
        """
        req = self._scheduler.find_request(request_id)
        if req is None:
            raise KeyError(
                f"feed_chunk: unknown or finished request_id {request_id!r}"
            )
        self._input_processor.append_streaming_chunk(req, chunk, is_last=is_last)

    def abort_request(self, request_id: str) -> None:
        """Remove a request from the engine, freeing cache if allocated."""
        req = self._scheduler.abort_request(request_id)
        if req is not None and req.stream_context is not None:
            self._model_runner.free_stream(req)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    def step(self) -> List[RequestOutput]:
        """Execute one engine step covering streaming + offline work."""
        nvtx_push("engine.step")
        nvtx_push("schedule")
        sched: SchedulerOutput = self._scheduler.schedule()
        nvtx_pop()
        outputs: List[RequestOutput] = []

        # Streaming admission: allocate cache for freshly admitted streams.
        if sched.newly_admitted:
            nvtx_push("allocate_stream")
            for req in sched.newly_admitted:
                if req.streaming:
                    self._model_runner.allocate_stream(req)
            nvtx_pop()

        # Offline batch — run the padded forward and finalise immediately.
        if sched.offline_batch:
            nvtx_push("offline_batch")
            outputs.extend(self._run_offline_batch(sched.offline_batch))
            nvtx_pop()

        running = sched.running_streams
        if running:
            # 1. Batched GPU fbank across every stream with pending audio —
            #    one kernel call for the whole active pool rather than N
            #    sequential fbank calls.  Issued on the dedicated feat
            #    stream when running on CUDA so it can overlap with the
            #    previous step's encoder forward; the default stream waits
            #    on the recorded event before reading feature_buffer.
            needs_feat = [r for r in running if r.has_pending_audio]
            if needs_feat:
                nvtx_push("extract_fbank")
                self._input_processor.extract_streaming_batch(
                    needs_feat, cuda_stream=self._feat_stream,
                )
                if self._feat_stream is not None:
                    # Synchronise default stream with feat stream so the
                    # downstream forward sees the freshly-written
                    # feature_buffer slices.
                    torch.cuda.current_stream(self._device).wait_stream(
                        self._feat_stream
                    )
                nvtx_pop()

            # 2. For each stream whose feature buffer now holds at least
            #    one encoder window, run forward_chunk_paged.
            window = self._config.decoding_window
            ready = [r for r in running if r.has_ready_encoder_chunk(window)]
            if ready:
                nvtx_push("forward_streaming")
                log_probs_map: Dict[str, torch.Tensor] = (
                    self._model_runner.forward_streaming_step(ready)
                )
                nvtx_pop()
                nvtx_push("decode_streaming")
                for req in ready:
                    lp = log_probs_map.get(req.request_id)
                    if lp is not None:
                        partial = self._output_processor.decode_streaming_chunk(req, lp)
                        outputs.append(partial)
                nvtx_pop()

            # 3. Finalise streams whose audio is exhausted and whose
            #    feature buffer has been fully consumed.  A stream can
            #    reach this state in the same step it ran its last
            #    encoder chunk, so we check *after* the forward pass.
            nvtx_push("finalize_streams")
            for req in list(running):
                if (not req.has_pending_audio) \
                        and (not req.has_ready_encoder_chunk(window)):
                    final = self._output_processor.finalize_streaming(req)
                    req.output = final
                    outputs.append(final)
                    self._model_runner.free_stream(req)
                    self._scheduler.finish_request(req.request_id)
            nvtx_pop()

        nvtx_pop()  # engine.step
        return outputs

    def run(self) -> List[RequestOutput]:
        """Run the engine until all pending requests are complete."""
        final_outputs: List[RequestOutput] = []
        while self._scheduler.has_pending():
            step_outputs = self.step()
            final_outputs.extend(o for o in step_outputs if o.finished)
        return final_outputs

    # ------------------------------------------------------------------
    # Offline batch handling
    # ------------------------------------------------------------------

    def _run_offline_batch(self, batch: List[Request]) -> List[RequestOutput]:
        """Forward one offline batch through the pipelined executor."""
        return self._offline_pipeline.run(batch)

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, List[str], torch.Tensor, "np.ndarray"],
        sample_rate: int = 16000,
        streaming: bool = True,
    ) -> Union[str, List[str]]:
        """Transcribe one or more audio inputs.

        Parameters
        ----------
        audio : str, list, Tensor, or ndarray
            Single or multiple audio inputs.
        sample_rate : int
            Sample rate of the audio (Hz).
        streaming : bool, default ``True``
            ``True`` uses the chunk-by-chunk streaming path; ``False`` uses
            the batched offline path.  Offline is strictly faster when
            real-time output isn't needed.
        """
        is_single = not isinstance(audio, list)
        audio_list: List = [audio] if is_single else audio  # type: ignore[list-item]

        request_ids = [
            self.add_request(a, sample_rate=sample_rate, streaming=streaming)
            for a in audio_list
        ]
        final = self.run()

        id_to_text: Dict[str, str] = {o.request_id: o.text for o in final}
        texts = [id_to_text.get(rid, "") for rid in request_ids]
        return texts[0] if is_single else texts

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def num_running(self) -> int:
        """Number of currently active streaming requests."""
        return self._scheduler.num_running

    @property
    def num_waiting(self) -> int:
        """Total number of requests across both waiting queues."""
        return self._scheduler.num_waiting
