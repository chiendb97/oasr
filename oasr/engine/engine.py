# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR inference and serving engine."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
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
from .longform import LongFormTracker
from .memory import (
    MIN_BLOCKS_PER_STREAM,
    PROBE_AUDIO_SECONDS,
    UNMEASURED_ACTIVATION_FRACTION,
    MemoryProfile,
    bytes_per_kv_block,
    derive_decode_kv_budget,
    derive_pool_blocks,
    measure_peak_activation,
    read_device_memory,
)
from .metrics import (
    AUDIO_SECONDS_SKIPPED,
    TOKENS_GENERATED,
    VAD_SEGMENTS,
    EngineMetrics,
    build_metrics,
)
from .model_runner import ModelRunner
from .output_processor import OutputProcessor
from .request import DecodingOptions, Request, RequestOutput
from .scheduler import Scheduler
from .streaming_backend import get_streaming_backend_class
from .vad_stage import OfflineVadSegmenter, StreamingVadStage

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

    #: Long-form fan-out tracker, ``None`` unless ``EngineConfig.long_form`` is
    #: set and the frontend has a fixed window.  Declared at class level, not
    #: only assigned in ``__init__``, because ``abort_request`` / ``step`` read it
    #: on every call and the concurrency tests drive a hand-built engine through
    #: ``__new__`` — an attribute that exists only after a full construction
    #: would make those entry points depend on construction order.
    _longform: Optional["LongFormTracker"] = None

    #: Device-memory profile taken during construction, ``None`` unless a
    #: capacity was left to derive.  Class-level for the same reason as
    #: ``_longform``: the concurrency tests drive a hand-built engine through
    #: ``__new__``, and a read must not depend on construction having run.
    _memory_profile: Optional[MemoryProfile] = None

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
            architecture=config.architecture,
        )
        model, model_config = loaded.model, loaded.config
        self._model = model
        config._model_config = model_config
        tokenizer = self._apply_checkpoint_specs(config, loaded)

        self._device = torch.device(device_str)
        # An offline-only encoder has no cache spec.  An engine pinned to offline
        # mode also selects the ``none`` streaming backend, so neither case should
        # reserve a streaming cache that no execution path can read.
        cache_spec = model.cache_spec
        # Streaming geometry the *encoder* declares (the backend-derived window /
        # stride are stamped further down, once the backend exists).  Resolved
        # here because the VRAM probe below needs the real chunk window to pick a
        # representative shape.
        config._subsampling_rate_override = model.encoder.subsampling_rate
        config._right_context_override = model.encoder.right_context

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

        # Engine-scope metric collection.  One collector per engine, not a
        # module global: a process holding an ``EnginePool`` of several engines
        # must keep their series apart, which is what the ``engine`` label on
        # the exported metric is for.
        self._metrics: EngineMetrics = build_metrics(self._device)

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
        # Size the paged KV pool from free VRAM when the operator left it to the
        # engine (``max_num_blocks=None``).  Must happen before the pool is
        # allocated, and after everything else that is already resident, so the
        # profile sees the real occupancy.
        cache_config = None
        if (
            cache_spec is not None
            and config.service_mode == "streaming"
            and get_streaming_backend_class(model.encoder.streaming_kind).allocates_paged_pool
        ):
            # Only a pool-owning runtime needs a cache config or a memory probe.
            if config.max_num_blocks is None:
                self._autosize_kv_pool(config, cache_spec, consumes)
            else:
                self._check_explicit_kv_pool(config)
            cache_config = config.build_cache_config(cache_spec)
        self._model_runner = ModelRunner(
            model, config, cache_config, graph_pool=self._graph_pool, consumes=consumes
        )

        # A backend that allocates no streaming state reports ``0`` for both (the
        # offline-only ``none`` backend, and any engine pinned to offline mode, which
        # now selects it — see ``ModelRunner``).  Leave the config's own values in
        # place rather than stamping zeros onto an engine nobody will stream through:
        # the geometry is still introspectable, and an accidental reader gets a
        # plausible window instead of a division by zero.
        if self._model_runner.decoding_window > 0:
            config._decoding_window_override = self._model_runner.decoding_window
            config._stride_override = self._model_runner.stride
            # Whether the closing silence pad has to round the stream up to a whole
            # window first — declared by the runtime, because only it knows whether
            # a sub-window tail can be forwarded.  See
            # ``StreamingEncoderBackend.finalize_align_frames``.
            # Same dynamic-override idiom as the four lines above, and the same
            # ``attr-defined`` noise: these are engine-stamped private fields, not
            # dataclass fields, because making them fields would put them in
            # ``EngineConfig``'s public surface and its round-trip.
            config._finalize_align_frames = (  # type: ignore[attr-defined]
                self._model_runner.streaming_backend.finalize_align_frames
            )

        self._output_processor = OutputProcessor(
            config, decode_type=decode_method, model=model, tokenizer=tokenizer
        )

        # Resolve "auto" VAD and refuse a combination that cannot be served, at
        # construction rather than on the first request.  Needs the strategy —
        # which ASR-derived detector exists is a property of the decode family —
        # so it lands right after the output processor.
        self._vad_stage: Optional[StreamingVadStage] = None
        self._resolve_vad(config, decode_method)

        # Ceiling on in-flight decoder KV for the AR families, derived from free
        # VRAM unless the operator supplied one (``0`` disables it).  Needs the
        # strategy — it owns the per-row footprint — so it lands after the output
        # processor and before the executor that reads the budget.
        if strategy_cls.incremental and config.decode_kv_budget_gib is None:
            self._autosize_decode_kv_budget(config, consumes)

        # Construct only the selected mode's executor; processors remain shared.
        self._executor: Executor = self._build_executor(config)

        # Best-effort prewarming covers common batch sizes and early cache buckets
        # so live streams avoid lazy graph-capture stalls. Other shapes stay lazy.
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

        # Warm the fused-attention compile cache when the encoder exposes the
        # paged-attention interface.  The helper is a no-op on unsupported paths.
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

        # Pre-warm the offline path so one-time library and workspace setup does
        # not inflate the first request's latency.
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

        # Long-form fan-out is meaningful only for a fixed-window
        # frontend: without one, a long request already decodes end to end and
        # segmenting it would only cost accuracy.
        self._longform: Optional[LongFormTracker] = None
        self._longform_window_samples = 0
        self._longform_overlap_samples = 0
        window_s = config.feature_config.fixed_window_seconds
        if getattr(config, "long_form", False):
            if window_s is None:
                logger.warning(
                    "long_form=True but the %r frontend has no fixed window; "
                    "long audio already decodes whole, so segmentation is off",
                    config.feature_config.feature_type,
                )
            else:
                sr = int(config.feature_config.sample_rate)
                self._longform = LongFormTracker()
                self._longform_window_samples = int(sr * window_s)
                self._longform_overlap_samples = int(
                    sr * min(float(config.long_form_overlap_seconds), window_s / 2.0)
                )
                logger.info(
                    "long-form decoding enabled: %.0fs windows, %.2fs overlap",
                    window_s,
                    self._longform_overlap_samples / sr,
                )

        # VAD segmentation reuses the same fan-out and the same fan-in; only the
        # splitter differs.  It *supersedes* the fixed-window splitter when both
        # are configured: a cut that lands in silence is strictly better than one
        # at an arbitrary sample count, and it needs no overlap, so the word-level
        # dedup ``merge_texts`` does at a window seam becomes unnecessary.
        self._vad_splitter: Optional[OfflineVadSegmenter] = None
        vad_cfg = config.vad
        if vad_cfg is not None and vad_cfg.mode == "segment" and config.service_mode == "offline":
            device = torch.device(vad_cfg.device or "cpu")
            self._vad_splitter = OfflineVadSegmenter(vad_cfg.resolve("offline"), device)
            if self._longform is None:
                self._longform = LongFormTracker()
            logger.info(
                "vad segmentation enabled: backend=%s on %s (supersedes the "
                "fixed-window splitter)",
                vad_cfg.backend,
                device,
            )

    # ------------------------------------------------------------------
    # VRAM-aware capacity sizing
    # ------------------------------------------------------------------

    def _check_explicit_kv_pool(self, config: EngineConfig) -> None:
        """Warn when an explicit ``max_num_blocks`` cannot serve ``max_batch_size``.

        The derived path already refuses a pool below
        ``max_batch_size * MIN_BLOCKS_PER_STREAM`` — :func:`derive_pool_blocks`
        takes that as its floor because below it every stream runs out of
        encoder cache.  The **explicit** path had no such check, and its default
        is a flat 2048 blocks *whatever* ``max_batch_size`` is, so widening the
        pool silently narrows every stream's history:

            frames per stream = max_num_blocks * block_size_frames / max_batch_size

        At the default 2048 blocks and 16 frames that is 512 frames per stream at
        ``max_batch_size=64`` and **32** at 1024 — one second of audio.  Streams
        then hit the capacity gate, finalize early with ``finish_reason="length"``
        and a truncated transcript, and the engine gets *faster*, because it is
        decoding less.  A throughput benchmark reads that as a win; this is the
        line that says otherwise.

        A warning rather than a raise: an operator may genuinely want a bounded
        history, and ``num_left_chunks >= 0`` makes a small pool correct by
        design.  What must not happen is it going unsaid.
        """
        blocks = int(config.max_num_blocks or 0)
        batch = max(1, int(config.max_batch_size))
        floor = batch * MIN_BLOCKS_PER_STREAM
        if blocks >= floor or config.num_left_chunks >= 0:
            return
        frames = blocks * int(config.block_size_frames) // batch
        logger.warning(
            "max_num_blocks=%d is below the %d blocks max_batch_size=%d needs "
            "(%d per stream): every stream gets %d encoder frames of history and "
            "will be finalized early with finish_reason='length'. Raise "
            "max_num_blocks to >= %d, lower max_batch_size, bound the history with "
            "num_left_chunks, or set max_num_blocks=None to size the pool from VRAM.",
            blocks,
            floor,
            batch,
            MIN_BLOCKS_PER_STREAM,
            frames,
            floor,
        )

    def _autosize_kv_pool(self, config: EngineConfig, cache_spec, consumes: str) -> None:
        """Resolve ``max_num_blocks=None`` into a block count that fits the card.

        Sets ``config.max_num_blocks`` in place, so everything downstream (the
        cache config, the pool, the per-stream ceiling) sees a plain number and
        needs no knowledge that it was derived.

        Raises
        ------
        ValueError
            On a non-CUDA device (nothing to measure), or when not even the
            minimum viable pool fits — see
            :func:`~oasr.engine.memory.derive_pool_blocks`.
        """
        if self._device.type != "cuda":
            raise ValueError(
                "max_num_blocks=None derives the paged KV pool from free VRAM, "
                f"which needs a CUDA device (got device={config.device!r}). Set "
                "an explicit max_num_blocks for a non-CUDA engine."
            )
        per_block = bytes_per_kv_block(
            num_layers=cache_spec.num_layers,
            block_size_frames=int(config.block_size_frames),
            n_kv_head=cache_spec.n_kv_head,
            head_dim=cache_spec.head_dim,
            dtype=config.dtype,
        )
        # Blocks one stream can hold, hence the ceiling past which the pool is
        # memory nothing can hand out.  With eviction the retained history is an
        # exact requirement (the pool has no capacity gate — see
        # ``CacheConfig.__post_init__``), so the floor and the ceiling coincide
        # and the derivation degenerates into "check that it fits".
        batch = max(1, int(config.max_batch_size))
        if config.num_left_chunks >= 0:
            frames = int(config.chunk_size) * int(config.num_left_chunks)
            per_stream = max(1, -(-frames // int(config.block_size_frames)))
            per_stream = min(per_stream, int(config.max_blocks_per_seq))
            floor_per_stream = per_stream
        else:
            per_stream = int(config.max_blocks_per_seq)
            floor_per_stream = min(per_stream, MIN_BLOCKS_PER_STREAM)
        profile = self._profile_device_memory(config, consumes)
        sizing = derive_pool_blocks(
            profile,
            per_block,
            min_blocks=batch * floor_per_stream,
            max_blocks=batch * per_stream,
        )
        config.max_num_blocks = sizing.blocks
        logger.info(
            "paged KV pool derived from VRAM: %s | %s",
            sizing.describe(),
            profile.describe(),
        )
        if sizing.limited_by == "block_table" and config.num_left_chunks < 0:
            # The card could afford more, but no stream could address it.  Say so:
            # the remaining lever is the per-stream ceiling, not the pool size, and
            # the operator has no way to tell those apart from the number alone.
            logger.info(
                "the pool is capped by max_blocks_per_seq (%d) x max_batch_size "
                "(%d), not by VRAM (%.2fGiB was available). Raise "
                "max_blocks_per_seq to spend it on a longer per-stream history.",
                config.max_blocks_per_seq,
                batch,
                profile.available_bytes / float(1024**3),
            )

    def _autosize_decode_kv_budget(self, config: EngineConfig, consumes: str) -> None:
        """Resolve ``decode_kv_budget_gib=None`` into a byte ceiling for AR decode.

        Leaves the budget off (``None``) on a non-CUDA device: there is no VRAM
        ceiling to enforce, and inventing one would throttle admission for no
        reason.  The executor reads any falsey value as "no byte budget".
        """
        if self._device.type != "cuda":
            return
        try:
            per_row = self._output_processor.strategy.kv_bytes_per_row()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("kv_bytes_per_row failed (%s); budgeting without it", exc)
            per_row = None
        profile = self._profile_device_memory(config, consumes)
        budget = derive_decode_kv_budget(profile, bytes_per_row=per_row)
        config.decode_kv_budget_gib = budget.gib
        log = logger.warning if budget.clamped_to_one_row else logger.info
        log(
            "decoder-KV budget derived from VRAM: %s | %s",
            budget.describe(),
            profile.describe(),
        )

    def _profile_device_memory(self, config: EngineConfig, consumes: str) -> MemoryProfile:
        """Measure the device: what is resident, and what one forward costs.

        Cached for the engine's lifetime — both derivations describe the same
        moment, and the probe forward is not free.
        """
        if self._memory_profile is not None:
            return self._memory_profile
        activation = 0
        measured = True
        try:
            activation = measure_peak_activation(
                lambda: self._probe_forward(config, consumes), self._device
            )
        except Exception as exc:
            measured = False
            logger.warning(
                "activation probe failed (%s); reserving %.0f%% of the budget for "
                "transients instead of a measured peak",
                exc,
                100.0 * UNMEASURED_ACTIVATION_FRACTION,
            )
        free, total = read_device_memory(self._device)
        profile = MemoryProfile(
            total_bytes=total,
            free_bytes=free,
            activation_bytes=activation,
            utilization=float(config.gpu_memory_utilization),
            activation_measured=measured,
        )
        self._memory_profile = profile
        return profile

    def _probe_forward(self, config: EngineConfig, consumes: str) -> None:
        """One representative forward at the widest shape the engine will run.

        Widest, not longest: the peak transient scales with ``max_batch_size`` x
        the per-row input length, and the engine's own ceiling on both is what
        the reserve has to cover.

        * streaming — one encoder chunk window at the full cohort width, which is
          exactly the shape ``forward_step`` issues at steady state.
        * offline — the frontend's fixed window if it has one, else
          :data:`~oasr.engine.memory.PROBE_AUDIO_SECONDS` of audio.

        Goes through ``InputProcessor.collate`` and the same ``consumes`` routing
        as production, so fbank staging and the head's ``(B, T, V)`` log-probs are
        in the measurement rather than assumed away.  Calls the **model** rather
        than ``ModelRunner``: the pool derivation runs before the runner exists
        (the runner is what allocates the pool), and the runner only delegates.
        """
        fc = config.feature_config
        assert fc is not None  # EngineConfig.__post_init__ always materialises one
        sr = int(fc.sample_rate)
        if config.service_mode == "streaming":
            # Feature frames one chunk consumes -> samples, snip_edges geometry.
            frames = int(config.decoding_window)
            n = fc.frame_length_samples + max(0, frames - 1) * fc.frame_shift_samples
        else:
            window_s = fc.fixed_window_seconds or PROBE_AUDIO_SECONDS
            n = int(sr * float(window_s))
        batch = max(1, int(config.max_batch_size))
        reqs = [
            Request(audio=torch.zeros(n, dtype=torch.float32), streaming=False, sample_rate=sr)
            for _ in range(batch)
        ]
        feats, lengths = self._input_processor.collate(reqs)
        with torch.no_grad():
            if consumes == "hidden":
                self._model.encode_offline(feats, lengths)
            elif consumes == "both":
                hidden, _ = self._model.encode_offline(feats, lengths)
                self._model.head(hidden)
            else:
                self._model.forward_offline(feats, lengths)
        if config.service_mode == "streaming":
            # ``collate`` is the *offline* ingest path; a streaming engine will
            # never touch the staging buffers it just grew, so give them back
            # rather than leaving pinned host memory resident for the process.
            self._input_processor.release_staging()

    # ------------------------------------------------------------------
    # Request management
    # ------------------------------------------------------------------

    def _resolve_vad(self, config: EngineConfig, decode_method: str) -> None:
        """Pin the VAD backend and check the mode is serviceable.

        Three things are settled here, all of them at construction:

        1. the detector's sample rate is the engine's, because the engine is
           single-rate and never resamples — a detector running at another rate
           would report spans in a different second than the transcript;
        2. ``backend=None`` ("auto") becomes the concrete ASR-derived kind the
           running decode family can feed, which is what makes *"no separate VAD
           model configured"* a real configuration rather than a degraded one;
        3. a mode the configured detector cannot serve raises **here**, naming
           the gap, instead of producing one whole-file segment that a client
           cannot distinguish from audio that really was one long utterance.
        """
        vad = config.vad
        if vad is None or not vad.enabled:
            return
        from oasr.vad import get_vad_spec

        # ``EngineConfig.__post_init__`` always materialises one; binding it here
        # keeps the rest of this method free of the Optional and gives a clear
        # failure if that invariant ever breaks.
        feature_config = config.feature_config
        assert feature_config is not None, "EngineConfig always materialises a feature_config"
        feature_rate = int(feature_config.sample_rate)
        if vad.sample_rate != feature_rate:
            if vad.sample_rate != 16000:  # 16000 is the field default, i.e. unset
                logger.warning(
                    "vad.sample_rate=%d is ignored: the engine is single-rate and "
                    "serves %d Hz, so the detector runs at that rate",
                    vad.sample_rate,
                    feature_rate,
                )
            vad.sample_rate = feature_rate

        strategy = self._output_processor.strategy
        family_kind = strategy.speech_activity_kind
        streaming = config.service_mode == "streaming"

        if vad.backend is None:
            if family_kind is None:
                raise ValueError(
                    f"vad.mode={vad.mode!r} needs a speech detector, but "
                    f"decode_method={decode_method!r} carries no per-frame "
                    "speech signal of its own and no vad.backend was configured. "
                    "Set --vad-backend (e.g. 'energy'), or run a decode family "
                    "whose output carries one."
                )
            vad.backend = family_kind

        spec = get_vad_spec(vad.backend)
        if spec.is_asr_derived and vad.backend != family_kind:
            own = repr(family_kind) if family_kind else "none"
            raise ValueError(
                f"vad.backend={vad.backend!r} reads a tensor that "
                f"decode_method={decode_method!r} does not produce (this "
                f"family's own detector is {own}). An ASR-derived detector is "
                "tied to the family whose output it reads."
            )
        if spec.is_asr_derived and not strategy.asr_speech_activity_modes:
            raise ValueError(
                f"vad.backend={vad.backend!r} is this family's detector, but the "
                "current configuration cannot supply it (a transducer under beam "
                "search records labels rather than frames, and a Whisper snapshot "
                "without a no-speech token has nothing to read). Configure a "
                "separate detector with --vad-backend, or change the decode "
                "configuration."
            )

        if (
            vad.mode == "segment"
            and streaming
            and getattr(config, "overlap_partial_readback", False)
        ):
            # The overlapped read-back emits the *previous* emit step's partial,
            # and it identifies a still-live stream by ``stream_id``.  A turn
            # boundary frees and re-creates the decode session under the same
            # stream_id, so a partial issued against the closed turn would be
            # collected against the new one and published as if it belonged to
            # it — the old turn's text, a turn late, in front of the new turn's.
            raise ValueError(
                "vad.mode='segment' cannot run with overlap_partial_readback=True: "
                "a turn boundary recreates the decode session under the same "
                "stream id, so an in-flight partial would be attributed to the "
                "turn after the one it came from. Disable one of the two."
            )
        if not streaming and not spec.is_asr_derived and vad.mode != "segment":
            # A waveform detector offline is wired for segmentation and nothing
            # else: the post-hoc labelling path reads the *decode family's* own
            # signal, so this combination would resolve cleanly and then produce
            # no segments at all — a silent no-op, which is the one outcome this
            # whole resolver exists to prevent.
            raise ValueError(
                f"vad.backend={vad.backend!r} is a waveform detector, and an "
                f"offline engine only runs one for vad.mode='segment' (got "
                f"{vad.mode!r}). Use --vad-mode segment to cut the audio at "
                "speech boundaries, or drop --vad-backend to label it with the "
                "decode family's own signal."
            )

        # ``segment`` needs a detector that can run *ahead* of the encoder, in
        # both service modes and for the same reason: it is the mode that decides
        # which audio the model sees.  Streaming needs ``stream`` on top, because
        # there it has to reach that verdict incrementally.
        if vad.mode == "segment":
            roles = ("presegment", "stream") if streaming else ("presegment",)
        else:
            roles = ("stream",) if streaming else ("posthoc",)
        for role in roles:
            if spec.can(role):
                continue
            raise ValueError(
                f"vad.backend={vad.backend!r} declares roles {list(spec.modes)}, "
                f"which does not include {role!r} — what a "
                f"{config.service_mode} engine in vad.mode={vad.mode!r} needs. "
                + (
                    "An ASR-derived detector reads what the encoder produced, so "
                    "it cannot decide what the encoder sees; configure a waveform "
                    "detector such as 'energy'."
                    if spec.is_asr_derived and role == "presegment"
                    else "Pick a detector that declares it."
                )
            )
        if vad.mode == "endpoint" and not streaming:
            raise ValueError(
                "vad.mode='endpoint' is a streaming control: an offline request "
                "already has exactly one utterance boundary, its end. Use "
                "vad.mode='segment' to cut long audio at speech boundaries, or "
                "'observe' to label it."
            )
        # A peaky detector cannot resolve a short silence from its own sparsity;
        # raise the preset to what it declares it can actually tell apart, and
        # say so, rather than letting the operator discover it from shredded
        # segments.  Only ever raises: an operator who asked for *longer* keeps
        # their value.
        floor = int(spec.min_silence_floor_ms)
        if floor > 0:
            resolved = vad.resolve(config.service_mode)
            if (resolved.min_silence_ms or 0) < floor:
                logger.info(
                    "vad.min_silence_ms raised %d -> %d ms: the %r signal is "
                    "sparse between emissions and cannot resolve a shorter gap",
                    resolved.min_silence_ms or 0,
                    floor,
                    vad.backend,
                )
                vad.min_silence_ms = floor

        window_s = feature_config.fixed_window_seconds
        if vad.mode == "segment" and window_s is not None:
            # A fixed-window frontend pads and *trims* every utterance to its
            # window, so a segment longer than it would be silently truncated —
            # and admission would reject it outright.  The padding counts: a 30 s
            # segment plus 400 ms on each side is 30.8 s, which does not fit.
            # Resolve first to read the padding the preset chose, then pin the
            # cap on the unresolved config so ``resolve`` honours it.
            resolved = vad.resolve(config.service_mode)
            budget = float(window_s) - 2.0 * (resolved.speech_pad_ms or 0) / 1000.0
            if budget <= 0:
                raise ValueError(
                    f"vad.speech_pad_ms={resolved.speech_pad_ms} leaves no room in "
                    f"the {feature_config.feature_type!r} frontend's "
                    f"{window_s:.0f}s window; lower it."
                )
            if resolved.max_speech_s is None or resolved.max_speech_s > budget:
                vad.max_speech_s = budget
                logger.info(
                    "vad.max_speech_s capped at %.2fs to fit the %r frontend's "
                    "%.0fs window with %d ms of padding",
                    budget,
                    feature_config.feature_type,
                    window_s,
                    resolved.speech_pad_ms or 0,
                )
        resolved_cfg = vad.resolve(config.service_mode)
        if streaming and vad.emits_events:
            clock = getattr(strategy, "_clock", None)
            if spec.is_asr_derived:
                if clock is None:
                    # Same refusal as word timings, for the same reason: a detector
                    # fed a guessed frame rate reports boundaries that are plausible
                    # and uniformly wrong by a constant factor.
                    raise ValueError(
                        f"vad.mode={vad.mode!r} needs the encoder frame rate, which "
                        f"decode_method={decode_method!r} cannot resolve (no feature "
                        "config or no declared subsampling). Speech-activity times "
                        "would be scaled by an unknown constant."
                    )
                stage_spf = clock.seconds_per_frame
                # An ASR-derived detector's tensor is produced on the engine's
                # device and is large; keeping the detector beside it is free.
                stage_device = torch.device(vad.device or config.device)
            else:
                stage_spf = spec.framing_for(resolved_cfg).seconds_per_frame(
                    resolved_cfg.sample_rate
                )
                # A waveform detector reads the audio, which arrives on the host.
                # Sending it to the GPU costs an H2D plus the device→host read of
                # its own answer — a synchronisation per tick, inside the step
                # loop, for a couple of pooling ops.  The offline splitter defaults
                # to CPU for the same reason; ``vad.device`` overrides both.
                stage_device = torch.device(vad.device or "cpu")
            self._vad_stage = StreamingVadStage(
                resolved_cfg,
                seconds_per_frame=stage_spf,
                device=stage_device,
                detector_kwargs=strategy.speech_activity_kwargs(),
            )
        # The frame rate is the detector's, not the config's: an ASR-derived
        # detector runs on the *encoder* grid (40 ms on a 4x-subsampled
        # Conformer), and reporting the waveform hop here would understate the
        # resolution by 4x for every one of them.
        clock = getattr(strategy, "_clock", None)
        if self._vad_stage is not None:
            resolution_ms = 1000.0 * self._vad_stage.seconds_per_frame
        elif spec.is_asr_derived and clock is not None:
            resolution_ms = 1000.0 * clock.seconds_per_frame
        else:
            resolution_ms = float(resolved_cfg.hop_ms)
        logger.info(
            "voice activity: backend=%s mode=%s preset=%s min_silence=%dms " "resolution=%.0fms",
            vad.backend,
            vad.mode,
            resolved_cfg.preset,
            resolved_cfg.min_silence_ms or 0,
            resolution_ms,
        )

    def _resolve_sample_rate(self, sample_rate: Optional[int]) -> int:
        """Validate a caller-supplied rate, or default to the model's.

        ``None`` means "whatever this checkpoint runs at" — the only rate the
        engine can serve, since every frame count comes from
        ``feature_config.sample_rate`` and nothing here resamples.  Anything else
        must match it exactly; see
        :meth:`~oasr.engine.input_processor.InputProcessor.check_sample_rate` for
        why a mismatch cannot be allowed through.

        Resolved on the *caller's* thread so the raise reaches them: under
        ``overlap_admit`` the same check inside ``prepare_offline`` runs on the
        prep thread, where it would only be logged.
        """
        if sample_rate is None:
            return int(self._config.feature_config.sample_rate)
        self._input_processor.check_sample_rate(sample_rate)
        return int(sample_rate)

    def _maybe_fan_out_longform(
        self,
        audio,
        request_id: Optional[str],
        sample_rate: int,
        priority: int,
        decoding,
    ) -> Optional[str]:
        """Split over-window audio into per-window child requests.

        Returns the parent request id when the audio was fanned out, or ``None``
        when it fits one window and should be admitted normally.  The caller sees
        one id either way; :meth:`step` merges the children back.
        """
        from .longform import split_windows

        wave = torch.as_tensor(audio, dtype=torch.float32, device="cpu").reshape(-1)

        spans: Optional[List[Tuple[int, int]]] = None
        if self._vad_splitter is not None:
            spans = self._vad_splitter.spans(wave)

        if spans is not None:
            windows = [wave[start:end] for start, end in spans]
            starts = [start / float(sample_rate) for start, _end in spans]
        else:
            # No VAD, or VAD found nothing worth cutting at — fall back to the
            # fixed-window splitter, which is a no-op for audio that already
            # fits.  Returning here rather than fanning out a single child keeps
            # "VAD found one span covering everything" identical to "VAD off".
            if (
                self._longform_window_samples <= 0
                or int(wave.numel()) <= self._longform_window_samples
            ):
                return None
            windows = split_windows(
                wave, self._longform_window_samples, self._longform_overlap_samples
            )
            stride = self._longform_window_samples - self._longform_overlap_samples
            starts = [(i * stride) / float(sample_rate) for i in range(len(windows))]

        parent_id = request_id or uuid.uuid4().hex
        tracker = self._longform
        assert tracker is not None, "the caller only fans out when a tracker exists"
        child_ids: List[str] = [tracker.child_id(parent_id, i) for i in range(len(windows))]
        tracker.register(parent_id, child_ids, starts)
        if spans is not None:
            kept = sum(int(w.numel()) for w in windows)
            self._metrics.incr(VAD_SEGMENTS, float(len(windows)))
            self._metrics.incr(
                AUDIO_SECONDS_SKIPPED,
                max(0.0, (int(wave.numel()) - kept) / float(sample_rate)),
            )
        logger.debug(
            "%s request %s: %.1fs -> %d segments (%.1fs of speech)",
            "vad-segmented" if spans is not None else "long-form",
            parent_id,
            wave.numel() / float(sample_rate),
            len(windows),
            sum(int(w.numel()) for w in windows) / float(sample_rate),
        )
        # Bulk admission so the windows land in one batch — they are independent,
        # which is exactly what makes the batched path applicable here.
        self.add_requests_batch(
            [
                {
                    "audio": w,
                    "request_id": cid,
                    "sample_rate": sample_rate,
                    "streaming": False,
                    "priority": priority,
                    "decoding": decoding,
                }
                for cid, w in zip(child_ids, windows)
            ]
        )
        return parent_id

    def add_request(
        self,
        audio: Union[torch.Tensor, "np.ndarray"],
        request_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
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
        sample_rate : int, optional
            Sample rate of ``audio`` in Hz.  Defaults to the model's own rate.
            The engine does **not** resample: any other value raises
            ``ValueError`` rather than transcribing the audio at the wrong
            speed.  Resample at the entry point (the ``oasr-server`` front-end
            does this for you).
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
        sr = self._resolve_sample_rate(sample_rate)
        if self._longform is not None and not streaming:
            fanned = self._maybe_fan_out_longform(audio, request_id, sr, priority, decoding)
            if fanned is not None:
                return fanned
        req = Request(
            audio,
            request_id=request_id,
            streaming=streaming,
            sample_rate=sr,
            priority=priority,
            decoding=DecodingOptions.coerce(decoding),
        )
        if self._overlap_admit:
            self._validate_mode(streaming)
            self._validate_decoding(req)
            # Raise on the caller's thread: a fixed-window violation surfaced
            # from the prep thread would only be logged.
            self._input_processor.check_audio_duration(req.audio)
            with self._admit_inflight_lock:
                self._admit_inflight += 1
            self._prep_in.put(req)
            return req.request_id
        with self._lock:
            self._validate_mode(streaming)
            self._validate_decoding(req)
            self._executor.admit(req)
        return req.request_id

    def _validate_decoding(self, request: Request) -> None:
        """Refuse per-request options the running decode family cannot act on.

        The batched admission paths have always done this; the single-request
        entry point did not, so the *documented* Python API
        (``transcribe(decoding=...)``) accepted ``task`` / ``language`` /
        ``word_timestamps`` on a family with no such control and silently
        returned a transcript of something else — the one failure mode
        ``DecodeStrategy.validate_options`` exists to make impossible.
        """
        self._output_processor.strategy.validate_options(
            request.decoding, streaming=request.streaming
        )

    def new_audio_buffer(self, num_samples: int) -> Optional[torch.Tensor]:
        """A page-locked float32 host buffer for one request's waveform, or ``None``.

        Offered to whoever decodes the audio — in a served process, the Rust
        front-end after the codec — so that its copy lands somewhere the engine
        can DMA from directly, and ``collate`` no longer has to pack the
        micro-batch into staging first.  Fill all ``num_samples`` elements, then
        submit the **tensor** as the request's ``audio``.

        ``None`` means "use ordinary memory": a CPU engine, or a request past
        ``EngineConfig.max_pinned_audio_seconds``.  That path is the older one
        and still correct, so a caller can treat this as a hint.  See
        :meth:`oasr.engine.input_processor.InputProcessor.new_audio_buffer`.
        """
        return self._input_processor.new_audio_buffer(int(num_samples))

    def add_requests_batch(self, specs: List[Dict]) -> List[str]:
        """Bulk admission — single Python entry for many requests.

        Each ``spec`` is a dict with keys:

        - ``audio``: ``None`` for a streaming-open admission (no audio yet);
          otherwise the raw waveform (``str`` / ``Tensor`` / ``ndarray``)
          that ``add_request`` would accept.
        - ``request_id``: optional pre-assigned id.
        - ``sample_rate``: int, defaults to the model's rate (the only other
          accepted value; see :meth:`add_request`).
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
                sample_rate=self._resolve_sample_rate(spec.get("sample_rate")),
                priority=int(spec.get("priority", 0)),
                decoding=DecodingOptions.coerce(spec.get("decoding")),
            )
            self._validate_mode(streaming)
            self._validate_decoding(req)
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
                    sample_rate=self._resolve_sample_rate(spec.get("sample_rate")),
                    priority=int(spec.get("priority", 0)),
                    decoding=DecodingOptions.coerce(spec.get("decoding")),
                )
                self._validate_mode(req.streaming)  # reads immutable executor.streaming
                self._validate_decoding(req)
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

        Stops the admission-prep thread, drains the executor, and releases the
        input processor's staging buffers: incremental AR strategies park
        requests with live decoder-KV buffers in the executor's pending pool,
        and the staging buffers hold pinned host memory (a process-global
        resource) — without an explicit release both only go away when the
        garbage collector gets to them.
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
        try:
            self._input_processor.release_staging()
        except Exception:  # pragma: no cover - defensive
            logger.exception("staging release failed")

    def _prewarm_offline(self) -> None:
        """Run one dummy offline batch per preferred size to absorb one-time
        cuBLAS/cuDNN/CTC initialisation at startup rather than on the first
        request.  Uses silent waveforms so it exercises the real
        fbank → encoder → CTC-decode path at representative shapes."""
        sizes = self._config.preferred_batch_size or [int(self._config.max_batch_size)]
        sr = int(self._config.feature_config.sample_rate)
        n = sr * 6  # ~6 s of audio — representative frame count
        for b in sorted({int(s) for s in sizes if int(s) >= 1}):
            reqs: List[Request] = []
            for _ in range(b):
                # collate reads ``request.audio`` directly (the canonical
                # waveform ``prepare_offline`` would have produced), so seed it.
                reqs.append(
                    Request(
                        audio=torch.zeros(n, dtype=torch.float32),
                        streaming=False,
                        sample_rate=sr,
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
        sample_rate: Optional[int] = None,
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
        sample_rate : int, optional
            Sample rate of the audio that will be fed via :meth:`feed_chunk`.
            Defaults to the model's own rate, and must equal it — the engine
            does not resample, and rejecting here (at open) beats discovering
            it after the client has been told the stream is live.
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
            sample_rate=self._resolve_sample_rate(sample_rate),
            priority=priority,
            decoding=DecodingOptions.coerce(decoding),
        )
        with self._lock:
            self._validate_mode(True)
            # Same check the batched and single-shot paths already make.  Without
            # it the *documented* way to open a stream accepted per-request
            # options the running family cannot act on and answered with a
            # transcript of something else — and the refusal has to land here,
            # at open, rather than after the client has been told the stream is
            # live and has started sending audio.
            self._validate_decoding(req)
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
        """Remove a request from the engine, freeing cache if allocated.

        A long-form parent id is not known to the executor — the windows are —
        so aborting one has to abort every window it fanned out to, or the
        cancelled file keeps decoding and its outputs pile up in the tracker.
        """
        with self._lock:
            if self._longform is not None:
                children = self._longform.abandon(request_id)
                if children:
                    for cid in children:
                        self._executor.abort(cid)
                    return
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
            # the spec too.  An explicit non-default engine value wins with a warning.
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
                metrics=self._metrics,
                vad_stage=self._vad_stage,
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
            decode_kv_budget_gib=config.decode_kv_budget_gib,
            max_tick_ms=config.max_tick_ms,
            decode_admit_window_ms=config.decode_admit_window_ms,
            max_batch_size=config.max_batch_size,
            collate_prefetch=config.offline_collate_prefetch,
            metrics=self._metrics,
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
            # Counted before long-form absorption, and only on finished
            # outputs.  Before, because the windows are what the decoder
            # actually generated and the stitched parent would report the same
            # work under a different total; only finished, because a streaming
            # partial carries the transcript *so far* and adding each one would
            # count every token once per tick it survived.
            if self._metrics.enabled:
                n_tokens = sum(len(o.tokens[0]) for o in outputs if o.finished and o.tokens)
                if n_tokens:
                    self._metrics.incr(TOKENS_GENERATED, n_tokens)
            if self._longform is not None and self._longform:
                # Replace per-window child outputs with one stitched parent.
                outputs = self._longform.absorb(outputs)
            return outputs

    def metrics_snapshot(self) -> Dict:
        """Drain engine-scope metrics for the serving front-end's exporter.

        Called by the Rust dispatcher on a rate-limited cadence, inside the GIL
        scope it already holds for the tick, and replayed into Prometheus
        there.  Counters and gauges come back **absolute** so the protocol is
        idempotent — a missed drain loses nothing and a repeated one
        double-counts nothing — while histogram samples are handed over and
        cleared.

        The point-in-time gauges are refreshed *here* rather than per tick:
        occupancy only has to be as fresh as the drain that exports it, and
        reading the block pool's free list takes a lock that is much better
        taken a few times a second than a few thousand.
        """
        self._executor.record_gauges(self._metrics)
        self._metrics.observe_gpu()
        return self._metrics.snapshot()

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
        sample_rate: Optional[int] = None,
        streaming: bool = True,
        decoding: Optional[Union[DecodingOptions, Dict]] = None,
    ) -> Union[str, List[str]]:
        """Transcribe one or more **waveforms**.

        Parameters
        ----------
        audio : Tensor, ndarray, or list of those
            One or more waveforms at the model sample rate.  Decode audio
            files before calling — the engine is waveform-only.
        sample_rate : int, optional
            Sample rate of the audio (Hz); defaults to the model's own and must
            equal it.  See :meth:`add_request`.
        streaming : bool, default ``True``
            ``True`` uses the chunk-by-chunk streaming path; ``False`` uses
            the batched offline path.  Offline is strictly faster when
            real-time output isn't needed.  See also
            :meth:`transcribe_offline`.
        decoding : DecodingOptions or dict, optional
            Per-request options applied to every waveform — ``task`` /
            ``language`` for the families with those controls, sampling knobs
            for the AR ones.  Without this the convenience API could not reach
            options the serving layer can, which is the kind of asymmetry that
            sends people to the HTTP surface for things the library can do.
        """
        outputs = self.transcribe_outputs(
            audio if isinstance(audio, list) else [audio],
            sample_rate=sample_rate,
            streaming=streaming,
            decoding=decoding,
        )
        texts = [o.text for o in outputs]
        return texts if isinstance(audio, list) else texts[0]

    def transcribe_outputs(
        self,
        audio: List[Union[torch.Tensor, "np.ndarray"]],
        sample_rate: Optional[int] = None,
        streaming: bool = True,
        decoding: Optional[Union[DecodingOptions, Dict]] = None,
    ) -> List[RequestOutput]:
        """:meth:`transcribe` returning the **whole** output, in input order.

        ``transcribe`` answers with strings, which is what nearly every caller
        wants and is why it exists.  Everything else a decode produces —
        ``words``, ``confidence``, ``timestamps``, ``nbest_texts``,
        ``finish_reason`` — is unreachable through it, so asking for word
        timings from the library used to mean driving ``add_request`` / ``run``
        by hand while the HTTP surface returned them in one call.  That
        asymmetry is what this closes.

        A row the engine never returned (aborted, or dropped) yields an empty
        output rather than shifting the list, so indices always line up with
        the inputs.
        """
        request_ids = [
            self.add_request(a, sample_rate=sample_rate, streaming=streaming, decoding=decoding)
            for a in audio
        ]
        by_id: Dict[str, RequestOutput] = {o.request_id: o for o in self.run()}
        return [
            by_id.get(rid, RequestOutput(request_id=rid, text="", tokens=[[]], finished=True))
            for rid in request_ids
        ]

    def transcribe_offline(
        self,
        audio: Union[torch.Tensor, "np.ndarray", List[Union[torch.Tensor, "np.ndarray"]]],
        sample_rate: Optional[int] = None,
        decoding: Optional[Union[DecodingOptions, Dict]] = None,
    ) -> Union[str, List[str]]:
        """Batch transcription convenience — :meth:`transcribe` with
        ``streaming=False``.

        Inputs flow through the dynamic-batching offline executor
        (length-bucketed micro-batches, CPU/GPU overlap).  Use this when
        real-time partials are not needed — it's strictly faster than the
        streaming path on the same audio.
        """
        return self.transcribe(audio, sample_rate=sample_rate, streaming=False, decoding=decoding)

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
    def sample_rate(self) -> int:
        """The **only** waveform sample rate this engine accepts, in Hz.

        Comes from the resolved feature config (checkpoint-derived via
        :class:`~oasr.features.FeatureSpec` unless the caller pinned one), which
        is what every frame count is computed from.  A request's own
        ``sample_rate`` is *not* used for feature extraction — it is validated
        against this and rejected on mismatch — so a front-end must resample to
        this rate before submitting.
        """
        return int(self._config.feature_config.sample_rate)

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
