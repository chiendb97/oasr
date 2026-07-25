# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR inference and serving engine."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.models import PretrainedModel, load_pretrained
from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .config import EngineConfig
from .decode import EncodeOutput, get_decode_strategy_class
from .executor import (
    Executor,
    OfflineExecutor,
    StreamingExecutor,
)
from .graph_cache import round_up_bucket
from .input_processor import InputProcessor
from .model_runner import ModelRunner
from .output_processor import OutputProcessor
from .request import DecodingOptions, Request, RequestOutput
from .scheduler import Scheduler

logger = logging.getLogger(__name__)


class ASREngine:
    """Single-mode ASR inference engine.

    Configured at construction (``EngineConfig.service_mode``) to handle
    either **streaming** (chunk-by-chunk with paged KV cache, partial
    outputs per tick) **or** **offline** (length-bucketed batched
    single-pass forward, one final output per request) — never both
    within the same lifecycle.

    The engine is a thin orchestrator over one :class:`Executor`
    instance: every public entry (``add_*``, ``feed_chunk``, ``abort``,
    ``step``, ``run``, status) routes through ``self._executor``.
    Throughput features land either as shared changes to
    :class:`InputProcessor` / :class:`ModelRunner` /
    :class:`OutputProcessor` (picked up by either executor) or as
    explicit, deliberate changes to a single executor implementation —
    the two modes never share dead branches.

    Parameters
    ----------
    config : EngineConfig
        Fully configured engine settings.  ``ckpt_dir`` must be set;
        ``service_mode`` defaults to ``"streaming"``.

    Examples
    --------
    The engine is **waveform-only**: ``audio`` arguments are waveform tensors
    / numpy arrays at the model sample rate.  Decode audio files at the entry
    point (the serving front-end, or here the harness) before admitting.

    Streaming::

        engine = ASREngine(EngineConfig(ckpt_dir="/path/to/ckpt"))
        wav, _sr = torchaudio.load("audio.wav")
        text = engine.transcribe(wav.squeeze(0))

    Offline batch::

        cfg = EngineConfig(ckpt_dir="/path/to/ckpt", service_mode="offline")
        engine = ASREngine(cfg)
        wavs = [torchaudio.load(p)[0].squeeze(0) for p in ("a.wav", "b.wav")]
        texts = engine.transcribe_offline(wavs)
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
        # ``load_pretrained`` accepts a local checkpoint dir (the common case —
        # a straight pass-through to the registry loader) or a HuggingFace Hub
        # repo id, which it downloads first.  Beyond the model it surfaces the
        # converter-emitted tokenizer / feature specs, which fill any
        # engine-config fields the caller left unset.
        loaded = load_pretrained(
            config.ckpt_dir,
            checkpoint_name=config.checkpoint_name,
            device=device_str,
            dtype=dtype,
        )
        model, model_config = loaded.model, loaded.config
        self._model = model
        config._model_config = model_config
        tokenizer = self._apply_checkpoint_specs(config, loaded)

        self._device = torch.device(device_str)
        # ``cache_spec`` is ``None`` for an offline-only encoder (Whisper,
        # Paraformer, the Qwen2-Audio tower), in which case no streaming cache is
        # built at all — previously every engine allocated the paged KV pool plus
        # the CNN-cache tensors (~0.4 GB at the defaults) even when nothing could
        # ever read them, on exactly the LLM/offline deployments where VRAM is
        # tightest (H13).
        cache_spec = model.cache_spec
        cache_config = config.build_cache_config(cache_spec) if cache_spec is not None else None

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

        self._input_processor = InputProcessor(config, self._device, graph_pool=self._graph_pool)
        self._scheduler = Scheduler(config)
        # Decode-method selection: ``config.decode_method`` picks among the
        # checkpoint's advertised capabilities; ``None`` runs the model's
        # default family (the unchanged production path).
        decode_method = config.decode_method or model.default_decode_type
        capabilities = model.capabilities
        if decode_method not in capabilities:
            raise ValueError(
                f"decode_method={decode_method!r} is not a capability of this "
                f"checkpoint (capabilities: {sorted(capabilities)}). Pick one "
                "of those, or leave decode_method=None for the model default."
            )
        self._decode_method = decode_method
        if config.service_mode == "streaming" and model.streaming_kind == "none":
            raise ValueError(
                "this checkpoint's encoder is offline-only (streaming_kind="
                "'none'); it exposes no chunk forward for the streaming "
                "runtime. Use service_mode='offline'."
            )
        # Resolve the decode strategy's declared input ("log_probs" / "hidden"
        # / "both") from the registry *class* before any component is built —
        # the streaming backends route their per-chunk forward on it, and the
        # OutputProcessor (which owns the strategy instance) is constructed
        # only after the runner (it needs the runner-derived geometry).
        strategy_cls = get_decode_strategy_class(decode_method, config)
        consumes = strategy_cls.consumes
        if consumes == "both" and config.service_mode == "streaming":
            raise ValueError(
                f"decode_method={decode_method!r} is offline-only: its strategy "
                "consumes both encoder hidden states and log-probs, which the "
                "streaming chunk forward does not produce (final-only streaming "
                "rescoring is a planned follow-up). Use service_mode='offline'."
            )
        if strategy_cls.incremental and config.service_mode == "streaming":
            raise ValueError(
                f"decode_method={decode_method!r} is offline-only: label-"
                "synchronous AR decoding (AED/LLM) is not genuinely streamable. "
                "Use service_mode='offline'."
            )
        if config.enable_sequence_packing and consumes in ("hidden", "both"):
            logger.warning(
                "enable_sequence_packing is ignored for decode_method=%r: the "
                "%s offline path runs the plain padded encode "
                "(packed hidden is a planned multi-paradigm follow-up)",
                decode_method,
                "hidden-states" if consumes == "hidden" else "hidden+log-probs",
            )
        self._model_runner = ModelRunner(
            model, config, cache_config, graph_pool=self._graph_pool, consumes=consumes
        )

        # Source the streaming geometry from the model's encoder + the streaming
        # backend so the executor / input processor window the feature buffer per
        # the actual architecture (Conformer paged vs Zipformer stateful) rather
        # than hardcoded Conformer constants.  For Conformer these equal the old
        # defaults (4 / 6 / 67 / 64) — zero behaviour change.
        config._subsampling_rate_override = model.encoder.subsampling_rate
        config._right_context_override = model.encoder.right_context
        # A backend that allocates no streaming state reports ``0`` for both (the
        # offline-only ``none`` backend, and any engine pinned to offline mode, which
        # now selects it — see ``ModelRunner``).  Leave the config's own values in
        # place rather than stamping zeros onto an engine nobody will stream through:
        # the geometry is still introspectable, and an accidental reader gets a
        # plausible window instead of a division by zero.
        if self._model_runner.decoding_window > 0:
            config._decoding_window_override = self._model_runner.decoding_window
            config._stride_override = self._model_runner.stride

        self._output_processor = OutputProcessor(
            config, decode_type=decode_method, model=model, tokenizer=tokenizer
        )

        # Build exactly one executor matching ``config.service_mode``.
        # The other mode's machinery (paged KV cache vs. persistent
        # producer thread) never wakes up, so there's no point paying
        # the construction cost or holding the dead references.  All
        # three building-block singletons (``InputProcessor`` /
        # ``ModelRunner`` / ``OutputProcessor``) are still shared so
        # throughput features land in one place regardless of mode.
        self._executor: Executor = self._build_executor(config)

        # Pre-warm the streaming encoder CUDA-Graph cache over a (B, cache_t1)
        # ladder so a live stream never pays a blocking lazy capture
        # mid-request (the interactive-latency tail: every new cache_t1 bucket
        # as the stream grows would otherwise trigger a ~cudaGraphInstantiate
        # stall).  We capture B in ``{1, max_batch_size} ∪ preferred`` (the
        # single-live-stream and full-cohort shapes) across a bucket ladder
        # covering the first ``prewarm_chunks`` encoder chunks — the
        # latency-critical stream ramp plus most short/medium utterances.
        # Deeper buckets (long utterances) and intermediate cohort-drain B
        # values still capture lazily (one-time).  Streaming-only: offline
        # never runs the streaming encoder forward.  Best-effort.
        if (
            self._device.type == "cuda"
            and bool(config.use_cuda_graphs)
            and config.service_mode == "streaming"
        ):
            try:
                pref = [int(b) for b in (config.preferred_batch_size or [])]
                batch_sizes = sorted({1, int(config.max_batch_size), *pref})
                prewarm_chunks = (
                    config.num_left_chunks
                    if config.num_left_chunks and config.num_left_chunks > 0
                    else 32
                )
                cs = int(config.chunk_size)
                buckets = sorted({round_up_bucket(cs * k) for k in range(int(prewarm_chunks) + 2)})
                self._model_runner.prewarm_encoder_graphs(batch_sizes, cache_t1_buckets=buckets)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Encoder graph pre-warm failed (will capture on first " "chunk instead): %s",
                    exc,
                )

        # Warm up the cute FMHA compile cache so the first request
        # doesn't pay JIT-compile latency. Skipped on CPU and on archs
        # where the cute backend isn't available (warmup_fmha is a no-op
        # in those cases).  Conformer-specific: requires the stacked-layer
        # layout AND the paged-FMHA attention interface (``self_attn.h_kv``);
        # encoders with their own attention (Zipformer's torch matmul, the
        # Paraformer SANM blocks) skip it.
        if (
            self._device.type == "cuda"
            and hasattr(model.encoder, "encoders")
            and len(getattr(model.encoder, "encoders", [])) > 0
            and hasattr(getattr(model.encoder.encoders[0], "self_attn", None), "h_kv")
        ):
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

        # Pre-warm the offline GPU path (fbank → encoder → CTC decode) so the
        # first real request doesn't pay one-time cuBLAS/cuDNN/CTC-workspace
        # initialisation — measured at ~3 s on the cold first ``step()``, which
        # otherwise dominates short runs and produces a bimodal latency tail.
        if self._device.type == "cuda" and config.service_mode == "offline":
            try:
                self._prewarm_offline()
            except Exception as exc:  # pragma: no cover
                logger.warning("Offline prewarm failed (first request will be slow): %s", exc)

        # Admission-prep overlap (offline only): a daemon thread runs the
        # per-request ``prepare_offline`` (waveform normalise + frame-count
        # stamp) off the caller/step thread so it overlaps the GPU ``step()``.
        # The thread is lock-free — it only prepares and hands finished
        # requests to ``step()`` via ``_prep_out``; ``step`` drains that queue
        # under the engine lock it already holds.  ``_admit_inflight`` counts
        # requests accepted but not yet in the scheduler so ``num_waiting``
        # (and the dispatcher's idle check) account for in-flight prep.
        self._overlap_admit: bool = bool(
            getattr(config, "overlap_admit", False) and config.service_mode == "offline"
        )
        self._prep_in: "queue.Queue[Optional[Request]]" = queue.Queue()
        self._prep_out: "queue.Queue[Request]" = queue.Queue()
        self._admit_inflight: int = 0
        self._admit_inflight_lock = threading.Lock()
        self._prep_thread: Optional[threading.Thread] = None
        if self._overlap_admit:
            self._prep_thread = threading.Thread(
                target=self._prep_loop, name="oasr-admit-prep", daemon=True
            )
            self._prep_thread.start()

    # ------------------------------------------------------------------
    # Request management
    # ------------------------------------------------------------------

    def add_request(
        self,
        audio: Union[torch.Tensor, "np.ndarray"],
        request_id: Optional[str] = None,
        sample_rate: int = 16000,
        streaming: bool = True,
        priority: int = 0,
        decoding: Optional[Union[DecodingOptions, Dict]] = None,
    ) -> str:
        """Add a new request to the engine.

        Both paths defer actual feature extraction until the engine step
        loop can batch it: streaming ingests audio **chunk by chunk** inside
        ``step()`` (batched across all active streams in one GPU fbank
        call), and the offline executor batches fbank within each GPU
        micro-batch.  What :meth:`add_request` does synchronously is load
        the waveform, stamp a cheap but exact (Kaldi ``snip_edges``
        formula) frame count so the scheduler can bucket by length, and
        split the waveform into streaming audio chunks on the CPU.  No
        fbank runs until the request is admitted and stepped.

        Parameters
        ----------
        audio : Tensor or ndarray
            A **waveform** at the model sample rate.  File decoding happens at
            the entry point, never in the engine — passing a file path raises
            ``TypeError``.
        request_id : str, optional
            Unique identifier.  Auto-generated if omitted.
        sample_rate : int
            Audio sample rate in Hz.
        streaming : bool, default ``True``
            ``True`` routes the request through the paged-cache streaming
            path; ``False`` routes it through the batched offline path.
        priority : int, default ``0``
            Lower values are scheduled first within each waiting queue.
        decoding : DecodingOptions or dict, optional
            Per-request decoding options (n-best, generation cap, sampling,
            prompt).  A plain dict is coerced — the PyO3 dispatcher passes
            one.  ``None`` keeps every engine default.

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
            decoding=DecodingOptions.coerce(decoding),
        )
        if self._overlap_admit:
            self._validate_mode(streaming)
            # Raise on the caller's thread: a fixed-window violation surfaced
            # from the prep thread would only be logged.
            self._input_processor.check_audio_duration(req.audio)
            with self._admit_inflight_lock:
                self._admit_inflight += 1
            self._prep_in.put(req)
            return req.request_id
        with self._lock:
            self._validate_mode(streaming)
            self._executor.admit(req)
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
        - ``decoding``: optional :class:`DecodingOptions` or plain dict of
          per-request decoding options.

        Holds ``self._lock`` for the whole batch — one acquire/release pair
        instead of N — and avoids N round-trips across the PyO3 boundary
        when the Rust dispatcher coalesces a tick's worth of admits.
        Returns the assigned request ids in the same order.

        Raises on the first invalid spec (after admitting the valid ones).
        Callers that need per-spec outcomes — the serving dispatcher, where one
        malformed request must not fail its batch-mates — should use
        :meth:`add_requests_batch_checked` instead.
        """
        results = self.add_requests_batch_checked(specs)
        for res in results:
            err = res.get("error")
            if err:
                raise ValueError(err)
        return [str(res["request_id"]) for res in results]

    def add_requests_batch_checked(self, specs: List[Dict]) -> List[Dict]:
        """:meth:`add_requests_batch` with per-spec outcomes instead of a raise.

        Returns one dict per spec, in order: ``{"request_id": str}`` on success,
        ``{"request_id": str, "error": str}`` when that spec was rejected. A
        rejected spec is never admitted and never enters the scheduler; every
        other spec in the batch is admitted normally.

        This is the entry point the PyO3 dispatcher uses. Bulk admission
        coalesces up to ``admit_threshold`` envelopes into one call, so a
        batch-wide raise would turn one client's bad ``top_p`` into an error for
        dozens of unrelated requests.
        """
        if self._overlap_admit:
            return self._admit_batch_overlapped(specs)
        results: List[Dict] = []
        with self._lock:
            for spec in specs:
                results.append(self._admit_one_checked(spec))
        return results

    def _admit_one_checked(self, spec: Dict) -> Dict:
        """Build + admit one spec, converting any rejection into a result dict.

        Caller holds ``self._lock``.  ``request_id`` is echoed even on failure so
        the caller can attribute the error without guessing.
        """
        rid = spec.get("request_id")
        try:
            streaming = bool(spec.get("streaming", True))
            req = Request(
                audio=spec.get("audio"),
                request_id=rid,
                streaming=streaming,
                sample_rate=int(spec.get("sample_rate", 16000)),
                priority=int(spec.get("priority", 0)),
                decoding=DecodingOptions.coerce(spec.get("decoding")),
            )
            self._validate_mode(streaming)
            self._executor.admit(req)
        except Exception as exc:
            return {"request_id": rid or "", "error": f"{type(exc).__name__}: {exc}"}
        return {"request_id": req.request_id}

    # ------------------------------------------------------------------
    # Admission-prep overlap (offline)
    # ------------------------------------------------------------------

    def _admit_batch_overlapped(self, specs: List[Dict]) -> List[Dict]:
        """Overlap fast-path for :meth:`add_requests_batch_checked` (offline only).

        Builds the (cheap) :class:`Request` objects on the caller's thread,
        returns their ids immediately, and hands the requests to the prep
        thread.  The ``prepare_offline`` (waveform normalise + frame stamp)
        then runs off the step thread; ``step()`` admits prepared
        requests to the scheduler.  ``_admit_inflight`` is bumped before
        queueing so :attr:`num_waiting` reflects work the scheduler can't see
        yet (otherwise the dispatcher could idle-wait past pending admits).

        Per-spec rejections are reported like the non-overlap path: an invalid
        spec is never queued for prep.
        """
        reqs: List[Request] = []
        results: List[Dict] = []
        for spec in specs:
            rid = spec.get("request_id")
            try:
                req = Request(
                    audio=spec.get("audio"),
                    request_id=rid,
                    streaming=bool(spec.get("streaming", True)),
                    sample_rate=int(spec.get("sample_rate", 16000)),
                    priority=int(spec.get("priority", 0)),
                    decoding=DecodingOptions.coerce(spec.get("decoding")),
                )
                self._validate_mode(req.streaming)  # reads immutable executor.streaming
                # Raise here, not on the prep thread, where it would only be logged.
                self._input_processor.check_audio_duration(req.audio)
            except Exception as exc:
                results.append({"request_id": rid or "", "error": f"{type(exc).__name__}: {exc}"})
                continue
            reqs.append(req)
            results.append({"request_id": req.request_id})
        if reqs:
            with self._admit_inflight_lock:
                self._admit_inflight += len(reqs)
            for req in reqs:
                self._prep_in.put(req)
        return results

    def _prep_loop(self) -> None:
        """Daemon: prepare queued offline requests off the step thread.

        Runs ``prepare_offline`` (lock-free — touches only the request) then
        hands the request to ``_prep_out`` for ``step()`` to admit.  Never
        acquires the engine lock, so it makes progress even while ``run()``
        holds it across a drain loop.  A ``None`` sentinel stops the thread.
        """
        while True:
            req = self._prep_in.get()
            if req is None:
                return
            try:
                self._input_processor.prepare_offline(req)
                self._prep_out.put(req)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "admit-prep failed for request %s", getattr(req, "request_id", "?")
                )
                with self._admit_inflight_lock:
                    self._admit_inflight -= 1

    def _drain_prepared(self) -> None:
        """Admit all prepared requests to the scheduler.  Caller holds the
        engine lock (invoked at the head of :meth:`step`)."""
        n = 0
        while True:
            try:
                req = self._prep_out.get_nowait()
            except queue.Empty:
                break
            self._scheduler.add_request(req)
            n += 1
        if n:
            with self._admit_inflight_lock:
                self._admit_inflight -= n

    def _num_admit_inflight(self) -> int:
        with self._admit_inflight_lock:
            return self._admit_inflight

    def shutdown(self) -> None:
        """Release engine-held resources (best-effort, idempotent).

        Stops the admission-prep thread and drains the executor: incremental AR
        strategies park requests with live decoder-KV buffers in the executor's
        pending pool, and without an explicit release those only go away when
        the garbage collector gets to them.
        """
        t = self._prep_thread
        if t is not None and t.is_alive():
            self._prep_in.put(None)
            t.join(timeout=2.0)
        self._prep_thread = None
        try:
            self._executor.shutdown()
        except Exception:  # pragma: no cover - defensive; shutdown must not raise
            logger.exception("executor shutdown failed")

    def _prewarm_offline(self) -> None:
        """Run one dummy offline batch per preferred size to absorb one-time
        cuBLAS/cuDNN/CTC initialisation at startup rather than on the first
        request.  Uses silent waveforms so it exercises the real
        fbank → encoder → CTC-decode path at representative shapes."""
        sizes = self._config.preferred_batch_size or [int(self._config.max_batch_size)]
        n = 16000 * 6  # ~6 s of audio — representative frame count
        for b in sorted({int(s) for s in sizes if int(s) >= 1}):
            reqs: List[Request] = []
            for _ in range(b):
                # collate reads ``request.audio`` directly (the canonical
                # waveform ``prepare_offline`` would have produced), so seed it.
                reqs.append(
                    Request(
                        audio=torch.zeros(n, dtype=torch.float32),
                        streaming=False,
                        sample_rate=16000,
                    )
                )
            feats, lengths = self._input_processor.collate(reqs)
            # Mirror OfflineExecutor._run_stage's consumes routing so the
            # prewarm exercises (and initialises) the same path production
            # requests take.
            strategy = self._output_processor.strategy
            consumes = strategy.consumes
            if consumes == "hidden":
                enc_out, out_len = self._model_runner.encode_offline(feats, lengths)
            elif consumes == "both":
                hidden, out_len = self._model_runner.encode_offline(feats, lengths)
                enc_out = EncodeOutput(
                    hidden=hidden, log_probs=self._model_runner.apply_head(hidden)
                )
            else:
                enc_out, out_len = self._model_runner.forward_offline(feats, lengths)
            # Incremental strategies have no one-shot decode; the encoder
            # forward above is the expensive warmup either way.
            if not strategy.incremental:
                self._output_processor.decode_offline(enc_out, out_len)
        torch.cuda.synchronize()

    def add_streaming_request(
        self,
        request_id: Optional[str] = None,
        sample_rate: int = 16000,
        priority: int = 0,
        decoding: Optional[Union[DecodingOptions, Dict]] = None,
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
        decoding : DecodingOptions or dict, optional
            Per-request decoding options; see :meth:`add_request`.

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
            decoding=DecodingOptions.coerce(decoding),
        )
        with self._lock:
            self._validate_mode(True)
            self._executor.admit(req)
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
            # ``OfflineExecutor.feed_chunk`` raises ``NotImplementedError``,
            # so this also serves as the mode check on offline engines.
            self._executor.feed_chunk(request_id, chunk, is_last=is_last)

    def abort_request(self, request_id: str) -> None:
        """Remove a request from the engine, freeing cache if allocated."""
        with self._lock:
            self._executor.abort(request_id)

    # ------------------------------------------------------------------
    # Internal — checkpoint-derived specs (features + tokenizer)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_checkpoint_specs(config: EngineConfig, loaded: PretrainedModel):
        """Fill engine defaults from the converter-emitted checkpoint specs.

        Explicit ``EngineConfig`` values always win, but a spec-vs-override
        mismatch logs loudly.  Returns the :class:`oasr.tokenizers.Tokenizer`
        to inject into the :class:`OutputProcessor` (``None`` → legacy sniffed
        ``sentencepiece_model`` / ``unit_table`` paths).
        """
        spec = loaded.feature_spec
        if spec is not None:
            if getattr(config, "_feature_config_explicit", False):
                diffs = spec.mismatches(config.feature_config)
                if diffs:
                    logger.warning(
                        "Explicit feature_config disagrees with the checkpoint's "
                        "FeatureSpec (%s); the explicit config wins, but this "
                        "usually degrades accuracy",
                        "; ".join(diffs),
                    )
            else:
                config.feature_config = spec.to_feature_config()
            # The waveform scale the checkpoint was trained on travels with
            # the spec too (Kaldi frontends: 32768.0 int16 scale; Whisper:
            # 1.0).  An explicit non-default engine value wins with a warning.
            if getattr(config, "_audio_scale_explicit", False):
                if float(config.audio_scale) != float(spec.audio_scale):
                    logger.warning(
                        "Explicit audio_scale %.1f differs from the checkpoint's "
                        "FeatureSpec (%.1f); the explicit value wins, but this "
                        "usually breaks recognition",
                        config.audio_scale,
                        spec.audio_scale,
                    )
            else:
                config.audio_scale = float(spec.audio_scale)

        tok_spec = loaded.tokenizer_spec
        if tok_spec is None:
            return None
        if getattr(config, "_tokenizer_paths_explicit", False):
            spec_table = tok_spec.files.get("table")
            if (
                config.unit_table is not None
                and spec_table is not None
                and Path(config.unit_table).resolve() != Path(spec_table).resolve()
            ):
                logger.warning(
                    "Explicit unit_table %s differs from the checkpoint's tokenizer "
                    "spec (%s); the explicit table wins",
                    config.unit_table,
                    spec_table,
                )
            return None
        try:
            from oasr.tokenizers import build_tokenizer

            return build_tokenizer(tok_spec)
        except Exception as exc:
            logger.warning(
                "Could not build tokenizer from checkpoint spec %r (%s); falling "
                "back to the legacy sniffed detokenizer paths",
                tok_spec.kind,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Internal — executor construction and mode validation
    # ------------------------------------------------------------------

    def _build_executor(self, config: EngineConfig) -> Executor:
        """Construct the single executor matching ``config.service_mode``."""
        if config.service_mode == "streaming":
            return StreamingExecutor(
                scheduler=self._scheduler,
                input_processor=self._input_processor,
                model_runner=self._model_runner,
                output_processor=self._output_processor,
                config=config,
                device=self._device,
            )
        # Offline: the scheduler partitions each batch into micro-batches
        # (length-bucketed, padded-frame-capped, or sequence-packed — all
        # driven by ``EngineConfig``).  The executor only needs ``enable_packing``
        # to pick the forward variant; it stays consistent with the scheduler's
        # partitioner because both read the same config flag.
        return OfflineExecutor(
            scheduler=self._scheduler,
            input_processor=self._input_processor,
            model_runner=self._model_runner,
            output_processor=self._output_processor,
            device=self._device,
            enable_packing=config.enable_sequence_packing,
            decode_steps_per_tick=config.decode_steps_per_tick,
            max_decode_slots=(
                config.max_decode_slots
                if config.max_decode_slots is not None
                else config.max_batch_size
            ),
            max_tick_ms=config.max_tick_ms,
            decode_admit_window_ms=config.decode_admit_window_ms,
            max_batch_size=config.max_batch_size,
        )

    def _validate_mode(self, streaming: bool) -> None:
        """Raise ``ValueError`` when the per-request ``streaming`` flag
        doesn't match ``config.service_mode``.

        Routing a mismatched request would silently land it in the wrong
        executor (offline ``admit`` on a streaming engine would just
        never run; streaming ``admit`` on an offline engine would
        produce empty outputs).  Surface the error eagerly so the caller
        can re-deploy with the right ``service_mode``.
        """
        if streaming != self._executor.streaming:
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
        """Execute one engine step — one call into the configured executor."""
        with self._lock:
            if self._overlap_admit:
                # Admit requests the prep thread finished preparing since the
                # last step (cheap scheduler enqueue under the lock we hold).
                self._drain_prepared()
            nvtx_push("engine.step")
            outputs = self._executor.step()
            nvtx_pop()
            return outputs

    def run(self) -> List[RequestOutput]:
        """Run the engine until all pending requests are complete.

        Holds the engine lock for the entire run.  Other threads calling
        :meth:`add_request` / :meth:`feed_chunk` will block until ``run``
        returns; use :meth:`step` in a loop instead if you need concurrent
        submission while draining.

        With ``overlap_admit`` the prep thread runs lock-free, so this loops
        per-step (releasing the lock between steps) and waits for in-flight
        admission prep to drain rather than holding the lock across the whole
        run.
        """
        if self._overlap_admit:
            return self._run_overlapped()
        with self._lock:
            final_outputs: List[RequestOutput] = []
            while self._executor.has_pending():
                step_outputs = self.step()
                final_outputs.extend(o for o in step_outputs if o.finished)
            return final_outputs

    def _run_overlapped(self) -> List[RequestOutput]:
        """``run`` variant for ``overlap_admit``: drain via per-step locking so
        the lock-free prep thread can admit concurrently; yield the GIL when
        idle so prep makes progress."""
        final_outputs: List[RequestOutput] = []
        while True:
            step_outputs = self.step()
            final_outputs.extend(o for o in step_outputs if o.finished)
            with self._lock:
                pending = self._executor.has_pending()
            if not pending and self._num_admit_inflight() == 0:
                break
            if not step_outputs and not pending:
                # Requests still being prepared by the prep thread — yield the
                # GIL briefly so it can finish before we re-step.
                time.sleep(0.0005)
        return final_outputs

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[torch.Tensor, "np.ndarray", List[Union[torch.Tensor, "np.ndarray"]]],
        sample_rate: int = 16000,
        streaming: bool = True,
    ) -> Union[str, List[str]]:
        """Transcribe one or more **waveforms**.

        Parameters
        ----------
        audio : Tensor, ndarray, or list of those
            One or more waveforms at the model sample rate.  Decode audio
            files before calling — the engine is waveform-only.
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
            self.add_request(a, sample_rate=sample_rate, streaming=streaming) for a in audio_list
        ]
        final = self.run()

        id_to_text: Dict[str, str] = {o.request_id: o.text for o in final}
        texts = [id_to_text.get(rid, "") for rid in request_ids]
        return texts[0] if is_single else texts

    def transcribe_offline(
        self,
        audio: Union[torch.Tensor, "np.ndarray", List[Union[torch.Tensor, "np.ndarray"]]],
        sample_rate: int = 16000,
    ) -> Union[str, List[str]]:
        """Batch transcription convenience — :meth:`transcribe` with
        ``streaming=False``.

        Inputs flow through the dynamic-batching offline executor
        (length-bucketed micro-batches, CPU/GPU overlap).  Use this when
        real-time partials are not needed — it's strictly faster than the
        streaming path on the same audio.
        """
        return self.transcribe(audio, sample_rate=sample_rate, streaming=False)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def service_mode(self) -> str:
        """The mode this engine was built for — ``"streaming"`` or ``"offline"``.

        The engine is the authority: several decode families are offline-only
        (AED / LLM / Paraformer / rescoring) and are rejected at construction in
        streaming mode, so a front-end that assumed the other mode would reject
        requests this engine could serve — or accept ones it cannot.
        """
        return self._config.service_mode

    @property
    def decode_method(self) -> str:
        """The resolved decode family actually running (never ``None``).

        ``EngineConfig.decode_method`` if the caller pinned one, else the
        model's ``default_decode_type``.  Distinct from
        ``EngineConfig.decoder_type``, which only selects the CTC *kernel*.
        """
        return self._decode_method

    @property
    def capabilities(self) -> List[str]:
        """Decode families this checkpoint could serve, sorted."""
        return sorted(self._model.capabilities)

    @property
    def num_running(self) -> int:
        """Currently-active requests.

        For streaming mode: streams admitted to the running pool.  For
        offline mode: requests submitted to the persistent executor but
        not yet drained back as outputs.  In both cases the Rust
        dispatcher uses ``num_running + num_waiting`` to decide whether
        to skip a step / enter an idle wait, so the count must include
        any work still in flight.
        """
        with self._lock:
            return self._executor.num_running()

    @property
    def num_waiting(self) -> int:
        """Requests in the waiting queue (admission pending).

        Includes requests accepted into the admission-prep queue but not yet
        visible to the scheduler (``overlap_admit``), so callers — notably the
        Rust dispatcher's idle check — don't treat the engine as idle while
        admission prep is still in flight.
        """
        with self._lock:
            n = self._executor.num_waiting()
        if self._overlap_admit:
            n += self._num_admit_inflight()
        return n
