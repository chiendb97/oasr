# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR inference and serving engine."""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.models.conformer.convert import load_wenet_checkpoint

from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .config import EngineConfig
from .input_processor import InputProcessor
from .model_runner import ModelRunner
from .output_processor import OutputProcessor
from .pipeline import OfflinePipeline, Pipeline, StreamingPipeline
from .request import Request, RequestOutput
from .scheduler import Scheduler

logger = logging.getLogger(__name__)


class ASREngine:
    """Single-mode ASR inference engine.

    Configured at construction (``EngineConfig.service_mode``) to handle
    either **streaming** (chunk-by-chunk with paged KV cache, partial
    outputs per tick) **or** **offline** (length-bucketed batched
    single-pass forward, one final output per request) — never both
    within the same lifecycle.

    The engine is a thin orchestrator over one :class:`Pipeline`
    instance: every public entry (``add_*``, ``feed_chunk``, ``abort``,
    ``step``, ``run``, status) routes through ``self._pipeline``.
    Throughput features land either as shared changes to
    :class:`InputProcessor` / :class:`ModelRunner` /
    :class:`OutputProcessor` (picked up by either pipeline) or as
    explicit, deliberate changes to a single pipeline implementation —
    the two modes never share dead branches.

    Parameters
    ----------
    config : EngineConfig
        Fully configured engine settings.  ``ckpt_dir`` must be set;
        ``service_mode`` defaults to ``"streaming"``.

    Examples
    --------
    Streaming::

        engine = ASREngine(EngineConfig(ckpt_dir="/path/to/ckpt"))
        text = engine.transcribe("audio.wav")

    Offline batch::

        cfg = EngineConfig(ckpt_dir="/path/to/ckpt", service_mode="offline")
        engine = ASREngine(cfg)
        texts = engine.transcribe_offline(["a.wav", "b.wav", "c.wav"])
    """

    def __init__(self, config: EngineConfig) -> None:
        # Engine-wide re-entrant lock guarding scheduler queues and per-request
        # audio mutations. Held by every public entry (add_*, feed_chunk,
        # abort_request, step, run, num_*). RLock — so run() can call step()
        # without deadlock. Uncontended cost is ~50 ns; invisible next to GPU
        # work. Single-thread callers (the default) are unaffected.
        self._lock = threading.RLock()

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

        # CUDA Graph capture: each cache type (encoder, feature extraction,
        # CTC) owns its own ``torch.cuda.graph_pool_handle()``. Sharing one
        # pool across *different* cache families turned out to cause silent
        # output aliasing — the encoder graph's intermediate allocations and
        # the feature graph's captured ``feats_out`` ended up at the same
        # device address, so a feature replay clobbered the encoder's
        # captured output buffer. Within one cache family the pool is still
        # shared across shape buckets (that's where the fragmentation win
        # lives). ``InputProcessor`` and ``ModelRunner`` each allocate their
        # own pool internally when ``graph_pool=None``.
        self._graph_pool: Optional[Tuple[int, int]] = None

        self._input_processor = InputProcessor(
            config, self._device, graph_pool=self._graph_pool
        )
        self._scheduler = Scheduler(config)
        self._model_runner = ModelRunner(
            model, config, cache_config, graph_pool=self._graph_pool
        )
        self._output_processor = OutputProcessor(config)

        # Build exactly one pipeline matching ``config.service_mode``.
        # The other mode's machinery (paged KV cache vs. persistent
        # producer thread) never wakes up, so there's no point paying
        # the construction cost or holding the dead references.  All
        # three building-block singletons (``InputProcessor`` /
        # ``ModelRunner`` / ``OutputProcessor``) are still shared so
        # throughput features land in one place regardless of mode.
        self._pipeline: Pipeline = self._build_pipeline(config)

        # Pre-warm the encoder CUDA-Graph cache at each preferred batch
        # size so the first real chunk replays instead of capturing.  Skipped
        # silently when graphs are off, the model_runner has no graph cache,
        # or no preferred sizes are configured.  Best-effort: a failure here
        # falls back to lazy capture on first traffic.
        if (
            config.preferred_batch_size
            and self._device.type == "cuda"
            and bool(config.use_cuda_graphs)
        ):
            try:
                self._model_runner.prewarm_encoder_graphs(
                    config.preferred_batch_size
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Encoder graph pre-warm failed (will capture on first "
                    "chunk instead): %s",
                    exc,
                )

        # Warm up the cute FMHA compile cache so the first request
        # doesn't pay JIT-compile latency. Skipped on CPU and on archs
        # where the cute backend isn't available (warmup_fmha is a no-op
        # in those cases).
        if self._device.type == "cuda":
            from oasr.jit.attention import warmup_fmha
            try:
                warmup_fmha(
                    n_head=model.encoder.encoders[0].self_attn.h,
                    n_kv_head=model.encoder.encoders[0].self_attn.h_kv,
                    head_dim=model.encoder.encoders[0].self_attn.d_k,
                    max_batch_size=config.max_batch_size,
                    chunk_size=config.chunk_size,
                    max_attention_key_size=config.chunk_size * 16,
                    device=self._device,
                    dtype=dtype,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "FMHA warmup failed (will compile on first call): %s",
                    exc,
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
        with self._lock:
            self._validate_mode(streaming)
            self._pipeline.admit(req)
        return req.request_id

    def add_requests_batch(self, specs: List[Dict]) -> List[str]:
        """Bulk admission — single Python entry for many requests.

        Each ``spec`` is a dict with keys:

        - ``audio``: ``None`` for a streaming-open admission (no audio yet);
          otherwise the raw waveform (``str`` / ``Tensor`` / ``ndarray``)
          that ``add_request`` would accept.
        - ``request_id``: optional pre-assigned id.
        - ``sample_rate``: int, defaults to ``16000``.
        - ``streaming``: bool, defaults to ``True``.
        - ``priority``: int, defaults to ``0``.

        Holds ``self._lock`` for the whole batch — one acquire/release pair
        instead of N — and avoids N round-trips across the PyO3 boundary
        when the Rust dispatcher coalesces a tick's worth of admits.
        Returns the assigned request ids in the same order.
        """
        request_ids: List[str] = []
        with self._lock:
            for spec in specs:
                audio = spec.get("audio")
                rid = spec.get("request_id")
                sample_rate = int(spec.get("sample_rate", 16000))
                streaming = bool(spec.get("streaming", True))
                priority = int(spec.get("priority", 0))
                req = Request(
                    audio=audio,
                    request_id=rid,
                    streaming=streaming,
                    sample_rate=sample_rate,
                    priority=priority,
                )
                self._validate_mode(streaming)
                self._pipeline.admit(req)
                request_ids.append(req.request_id)
        return request_ids

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
        with self._lock:
            self._validate_mode(True)
            self._pipeline.admit(req)
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
        with self._lock:
            # ``OfflinePipeline.feed_chunk`` raises ``NotImplementedError``,
            # so this also serves as the mode check on offline engines.
            self._pipeline.feed_chunk(request_id, chunk, is_last=is_last)

    def abort_request(self, request_id: str) -> None:
        """Remove a request from the engine, freeing cache if allocated."""
        with self._lock:
            self._pipeline.abort(request_id)

    # ------------------------------------------------------------------
    # Internal — pipeline construction and mode validation
    # ------------------------------------------------------------------

    def _build_pipeline(self, config: EngineConfig) -> Pipeline:
        """Construct the single pipeline matching ``config.service_mode``."""
        if config.service_mode == "streaming":
            return StreamingPipeline(
                scheduler=self._scheduler,
                input_processor=self._input_processor,
                model_runner=self._model_runner,
                output_processor=self._output_processor,
                config=config,
                device=self._device,
            )
        return OfflinePipeline(
            scheduler=self._scheduler,
            input_processor=self._input_processor,
            model_runner=self._model_runner,
            output_processor=self._output_processor,
            micro_batch_size=int(config.max_batch_size),
            depth=max(1, int(config.offline_pipeline_depth)),
            device=self._device,
            gpu_feature_extraction=bool(config.offline_gpu_feature_extraction),
            preferred_sizes=config.preferred_batch_size,
            persistent_producer=bool(config.offline_persistent_pipeline),
        )

    def _validate_mode(self, streaming: bool) -> None:
        """Raise ``ValueError`` when the per-request ``streaming`` flag
        doesn't match ``config.service_mode``.

        Routing a mismatched request would silently land it in the wrong
        pipeline (offline ``admit`` on a streaming engine would just
        never run; streaming ``admit`` on an offline engine would
        produce empty outputs).  Surface the error eagerly so the caller
        can re-deploy with the right ``service_mode``.
        """
        if streaming != self._pipeline.streaming:
            raise ValueError(
                f"Request streaming={streaming} does not match configured "
                f"service_mode={self._config.service_mode!r}.  The engine "
                "accepts only one mode per lifecycle; restart with the "
                "matching service_mode."
            )

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    def step(self) -> List[RequestOutput]:
        """Execute one engine step — one call into the configured pipeline."""
        with self._lock:
            nvtx_push("engine.step")
            outputs = self._pipeline.step()
            nvtx_pop()
            return outputs

    def run(self) -> List[RequestOutput]:
        """Run the engine until all pending requests are complete.

        Holds the engine lock for the entire run.  Other threads calling
        :meth:`add_request` / :meth:`feed_chunk` will block until ``run``
        returns; use :meth:`step` in a loop instead if you need concurrent
        submission while draining.
        """
        with self._lock:
            final_outputs: List[RequestOutput] = []
            while self._pipeline.has_pending():
                step_outputs = self.step()
                final_outputs.extend(o for o in step_outputs if o.finished)
            return final_outputs

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
            real-time output isn't needed.  See also
            :meth:`transcribe_offline`.
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

    def transcribe_offline(
        self,
        audio: Union[str, List[str], torch.Tensor, "np.ndarray"],
        sample_rate: int = 16000,
    ) -> Union[str, List[str]]:
        """Batch transcription convenience — :meth:`transcribe` with
        ``streaming=False``.

        Inputs flow through the dynamic-batching offline pipeline
        (length-bucketed micro-batches, CPU/GPU overlap).  Use this when
        real-time partials are not needed — it's strictly faster than the
        streaming path on the same audio.
        """
        return self.transcribe(audio, sample_rate=sample_rate, streaming=False)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def num_running(self) -> int:
        """Currently-active requests.

        For streaming mode: streams admitted to the running pool.  For
        offline mode: requests submitted to the persistent pipeline but
        not yet drained back as outputs.  In both cases the Rust
        dispatcher uses ``num_running + num_waiting`` to decide whether
        to skip a step / enter an idle wait, so the count must include
        any work still in flight.
        """
        with self._lock:
            return self._pipeline.num_running()

    @property
    def num_waiting(self) -> int:
        """Requests in the waiting queue (admission pending)."""
        with self._lock:
            return self._pipeline.num_waiting()
