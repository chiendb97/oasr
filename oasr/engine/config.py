# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR Engine configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from oasr.cache import CacheConfig
from oasr.decode import DecoderConfig
from oasr.features import FeatureConfig
from oasr.functionals.ctc_decode import GpuDecoderConfig
from oasr.models.base import BaseModelConfig, CacheSpec

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Unified configuration for the ASR inference engine.

    Parameters
    ----------
    ckpt_dir : str
        Path to a WeNet-format checkpoint directory containing ``final.pt``,
        ``train.yaml``, ``global_cmvn``, and optionally a SentencePiece model.
    checkpoint_name : str
        Filename of the model weights inside ``ckpt_dir``.
    architecture : str, optional
        Force a registered architecture instead of letting
        ``CheckpointConverter.detect`` rank the candidates.  Needed only for an
        **explicit-only** converter — one whose ``detect()`` is always False
        because its directory layout is claimed by another architecture.
        ``transducer`` is the case: an icefall pruned-RNNT export sniffs as
        ``zipformer``, and a hybrid export really does carry both branches, so
        the transducer branch has to be asked for by name.  Without this the
        only way to reach such a checkpoint from the engine was to re-convert it
        to native format first.
    device : str
        CUDA device string, e.g. ``"cuda"`` or ``"cuda:0"``.
    dtype : torch.dtype
        Floating-point precision for model and cache tensors.
    chunk_size : int
        Number of encoder output frames per streaming chunk (after 4×
        subsampling).  Must match the value used during model training for
        consistent streaming accuracy.
    num_left_chunks : int
        Number of past chunks to keep in the attention cache.
        ``-1`` means unlimited (keep all history).
    max_batch_size : int
        Encoder forward batch size.  In streaming mode it caps the running
        pool and sizes the paged KV cache; in offline mode it is the GPU
        forward width of each executor micro-batch.  The service runs in
        one mode at a time, never both.  ``preferred_batch_size`` is layered
        on top as a Triton-style soft cap (snap-to-preferred admission)
        while ``max_batch_size`` remains the hard cap on resources.
    preferred_batch_size : list[int], optional
        Triton-style preferred batch sizes.  When set, the scheduler admits
        streaming/offline batches only at these B values; ``max_wait_time``
        is the escape valve.  Drives the encoder CUDA-Graph pre-warm and
        defaults ``feature_graph_batch_buckets`` when that is unset.  Every
        value must be ``<= max_batch_size``; sorted/deduped on init.  ``None``
        keeps the legacy "admit greedily up to ``max_batch_size``" behaviour.
    max_num_blocks : int or None
        Total number of physical KV-cache blocks in the shared block pool.
        Should satisfy ``max_num_blocks >= max_batch_size * max_blocks_per_seq``.
        ``None`` means **derive it from free VRAM** at engine construction (H4)
        — see :attr:`gpu_memory_utilization`.
    gpu_memory_utilization : float
        Fraction of the device the engine may occupy in total, weights included.
        Only read when a capacity is left to derive (``max_num_blocks=None`` in
        streaming mode, ``decode_kv_budget_gib=None`` for an AR decode family).
    block_size_frames : int
        Frames per KV-cache block (page).  Setting this equal to ``chunk_size``
        means each chunk maps to exactly one block.
    max_blocks_per_seq : int
        Maximum logical blocks per stream in the block table tensor.
    feature_config : FeatureConfig, optional
        Feature extraction config.  Defaults to 80-dim log-mel FBANK at 16 kHz
        with dither disabled (``dither=0.0``) for deterministic inference.
    decoder_type : str
        Which GPU CTC decoder to use:
        ``"ctc_cuda"`` — GPU beam search via ``GpuStreamingDecoder`` (default),
        ``"ctc_wfst"`` — k2 WFST beam search (GPU, requires a k2 build).
        This is a **registry selector** (it picks the strategy), not a family
        option, so it stays here.
    decode_options : dict
        Per-family decode knobs, validated against the active strategy's
        ``options_cls``.  Adding a decode family needs no new field here — see
        ``oasr.engine.decode.options`` and ``docs/architecture.md``.
    ctc_decoder_config, wfst_decoder_config, fst_path, rescoring_ctc_weight,
    rescoring_reverse_weight, transducer_max_sym_per_frame, max_new_tokens,
    llm_prompt
        **Deprecated aliases.**  These are per-family options that now live on
        the owning strategy (``strategy.options``).  They still work — the
        public API and every ``oasr-server`` flag map onto them — and each
        carries the same default as the option it aliases, but new knobs should
        go in the family's ``options_cls`` and be set through
        ``decode_options``.
    sentencepiece_model : str, optional
        Path to a SentencePiece ``.model`` file for detokenization.
        Auto-detected from ``ckpt_dir`` if not provided.
    unit_table : str, optional
        Path to a ``units.txt`` vocabulary file used as a fallback when
        SentencePiece is unavailable.  Auto-detected from ``ckpt_dir``.
    """

    ckpt_dir: str = ""
    checkpoint_name: str = "final.pt"
    architecture: Optional[str] = None
    device: str = "cuda"
    # bfloat16 avoids overflow in wide-activation encoders while retaining a
    # served low-precision dtype.
    dtype: torch.dtype = torch.bfloat16

    # One mode per engine lifecycle.  Request admission rejects a mismatched
    # streaming flag instead of routing it to an inactive executor.
    service_mode: str = "streaming"
    # Offline-only: prepare waveform metadata on a daemon thread while the
    # previous batch runs.  Results cross into ``step()`` through a queue.
    overlap_admit: bool = False
    # Streaming-only: overlap next-step feature packing with the current encoder
    # forward.  This adds one step of pipeline latency; ``False`` restores serial order.
    streaming_feature_lookahead: bool = True
    # Offline-only: collate the next micro-batch on a side stream so host work and
    # feature extraction overlap the encoder.  Keeps one extra feature batch live.
    offline_collate_prefetch: bool = True

    # Streaming chunking
    chunk_size: int = 16
    num_left_chunks: int = -1
    # Pad the final streaming chunk with silence to complete an encoder window.
    # This preserves trailing tokens and avoids the unsupported captured-graph
    # path for partial windows; the added silence decodes to blanks.
    finalize_silence_pad: bool = True

    # Batching
    # Encoder-forward width.  It also caps the streaming pool and sizes its cache.
    max_batch_size: int = 32
    # Preferred scheduler widths, normalized to unique ascending values within
    # ``max_batch_size``.  Also seed graph pre-warm buckets; ``None`` admits greedily.
    preferred_batch_size: Optional[List[int]] = None
    # Minimum ``min_len / max_len`` within an offline batch.  ``0`` disables this
    # filter and leaves ``max_offline_pad_ratio`` as the padding guard.
    length_bucket_ratio: float = 0.0
    # Reject a candidate when ``(max_len * batch_size) / sum_len`` exceeds this.
    max_offline_pad_ratio: float = 4.0
    # Maximum ``max_len * batch_size`` pre-subsampling frames per offline batch.
    # ``None`` uses only ``max_batch_size``; sequence packing takes precedence.
    max_batch_frames: Optional[int] = None
    # Maximum time (seconds) a waiting request may sit in the queue before it
    # is flushed even if no ideal length-bucket peer has arrived.  Prevents
    # starvation of outlier-length requests under heavy load.
    max_wait_time: float = 0.2
    # Scheduling policy for offline admission:
    #   "fcfs"    — strict first-come-first-served, no bucketing
    #   "bucket"  — pick oldest, then fill batch with length-similar peers
    #   "sjf"     — shortest-job-first (best throughput, can starve long reqs;
    #               starvation is still bounded by ``max_wait_time``)
    schedule_policy: str = "bucket"
    # Admit streaming requests in lockstep cohorts to avoid fragmented offset
    # groups.  ``False`` favors immediate admission over batch width.
    streaming_cohort_admit: bool = True

    # Offline-only: pack utterances into one encoder row while isolating
    # attention, convolution and positional state at segment boundaries.
    # Subsampling remains batched so its receptive field cannot cross segments.
    enable_sequence_packing: bool = False
    # Post-subsampling frame budget per packed row, including segment gaps.
    max_packed_frames: int = 8192

    # Paged-KV block count.  ``None`` derives it from available device memory;
    # offline engines allocate no pool.
    max_num_blocks: Optional[int] = 2048
    block_size_frames: int = 16
    max_blocks_per_seq: int = 512
    # Total device-memory fraction available to derived capacities.  The
    # remainder covers capture pools, transient activations and fragmentation.
    gpu_memory_utilization: float = 0.90

    # Capture steady-state streaming forwards by ``(B_active, cache_t1_bucket)``.
    # Bias inputs are tile-padded because the fused kernel reads complete tiles.
    use_cuda_graphs: bool = True

    # Optional feature and CTC graph caches, gated by ``use_cuda_graphs``.
    # Disabled by default because replay staging can outweigh launch savings.
    use_feature_cuda_graphs: bool = False
    use_ctc_cuda_graphs: bool = False
    # Capture the transducer predictor step (embedding + recurrent layers + emit
    # masks + joint projection).  Nine launches for ~12-39 us of GPU work, and a
    # third of the greedy loop's host time; see oasr/engine/predictor_graph.py.
    use_transducer_cuda_graphs: bool = True
    # Feature-graph batch buckets; ``None`` uses powers of two up to the batch cap.
    feature_graph_batch_buckets: Optional[List[int]] = None

    # Feature extraction
    feature_config: Optional[FeatureConfig] = None

    # CTC kernel implementation; ``ctc_wfst`` requires ``fst_path``.
    decoder_type: str = "ctc_cuda"
    ctc_decoder_config: Optional[GpuDecoderConfig] = None
    wfst_decoder_config: Optional[DecoderConfig] = None
    fst_path: Optional[str] = None

    # Decode family.  ``None`` uses the model default; explicit values must be a
    # declared model capability.
    decode_method: Optional[str] = None

    # Per-family knobs validated by the active strategy's ``options_cls``.
    decode_options: Dict[str, Any] = field(default_factory=dict)

    # CTC+AED rescoring weights.  ``None`` uses the checkpoint's trained reverse
    # weight; zero skips the reverse pass.
    rescoring_ctc_weight: float = 0.5
    rescoring_reverse_weight: Optional[float] = None

    # Transducer safety cap on non-blank emissions per encoder frame.
    transducer_max_sym_per_frame: int = 10

    # Incremental-AR work and admission caps.  ``max_decode_slots=None`` uses
    # ``max_batch_size``; frame-synchronous strategies ignore both.
    decode_steps_per_tick: int = 32
    max_decode_slots: Optional[int] = None
    # Total decoder-KV budget for in-flight AR requests.  ``None`` derives it
    # from free memory; ``0`` disables byte budgeting.  A request-count cap alone
    # cannot bound KV memory because row position budgets differ.
    decode_kv_budget_gib: Optional[float] = None

    # Recycle the oldest streaming KV block at capacity instead of terminating.
    # Disabled by default because eviction shortens the attention span.
    recycle_streaming_history: bool = False

    # Split overlong fixed-window inputs and stitch parallel window results.
    # Overlap mitigates boundary errors.
    long_form: bool = False
    # Audio shared between adjacent long-form windows.  Overlapping lets the
    # stitcher drop duplicated words at a cut instead of losing one; 0 disables.
    long_form_overlap_seconds: float = 1.0
    # AR decode time cap per tick.  It prevents new steps after the deadline but
    # cannot preempt one, so the effective bound is this value plus one step.
    # ``0`` disables the time cap.
    max_tick_ms: float = 25.0
    # Hold AR arrivals briefly to share one encoder/prefill pass.  This trades up
    # to one window of first-token latency for less setup work.
    decode_admit_window_ms: float = 0.0

    # AR generation length cap (per request), read by incremental strategies.
    max_new_tokens: int = 448

    # Speech-LLM user prompt (``decode_method="llm"``): text placed in the
    # checkpoint's chat template next to the audio.  ``None`` uses the model
    # config's ``default_user_prompt``.
    llm_prompt: Optional[str] = None

    # Emit streaming partials every N decode steps.  ``<=0`` disables interim
    # readbacks without stopping decode-state advancement.
    partial_decode_interval: int = 1
    # Issue partial readbacks asynchronously and emit the previous result.  This
    # removes a blocking sync but adds one-chunk lag; final output is unchanged.
    overlap_partial_readback: bool = False

    # Detokenization
    sentencepiece_model: Optional[str] = None
    unit_table: Optional[str] = None

    # Legacy feature pipelines expect int16-scale samples, while audio loaders
    # return normalized floats.  A checkpoint FeatureSpec overrides this default
    # when its frontend was trained on normalized waveforms.
    audio_scale: float = 32768.0

    # Maximum request duration eligible for page-locked input memory.  Longer
    # requests use ordinary heap memory; ``0`` disables pinned allocation.
    max_pinned_audio_seconds: float = 300.0

    _model_config: Optional[BaseModelConfig] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.service_mode not in ("streaming", "offline"):
            raise ValueError(
                f"service_mode must be 'streaming' or 'offline', got " f"{self.service_mode!r}"
            )
        if self.decoder_type not in ("ctc_cuda", "ctc_wfst"):
            raise ValueError(
                f"decoder_type must be 'ctc_cuda' or 'ctc_wfst', got "
                f"{self.decoder_type!r}. CPU decoders are no longer exposed "
                "through the engine; use the standalone oasr.decode API for those."
            )
        if self.max_batch_frames is not None and self.max_batch_frames < 1:
            raise ValueError(
                f"max_batch_frames must be a positive int or None, got "
                f"{self.max_batch_frames!r}"
            )
        if self.transducer_max_sym_per_frame < 1:
            raise ValueError(
                f"transducer_max_sym_per_frame must be >= 1, got "
                f"{self.transducer_max_sym_per_frame!r}"
            )
        if self.decode_steps_per_tick < 1:
            raise ValueError(
                f"decode_steps_per_tick must be >= 1, got {self.decode_steps_per_tick!r}"
            )
        if self.long_form_overlap_seconds < 0:
            raise ValueError(
                "long_form_overlap_seconds must be >= 0, got " f"{self.long_form_overlap_seconds!r}"
            )
        if self.decode_kv_budget_gib is not None and self.decode_kv_budget_gib < 0:
            raise ValueError(
                "decode_kv_budget_gib must be > 0, 0 (disabled) or None "
                f"(derive from VRAM), got {self.decode_kv_budget_gib!r}"
            )
        if self.max_num_blocks is not None and self.max_num_blocks < 1:
            raise ValueError(
                "max_num_blocks must be a positive int or None (derive from "
                f"VRAM), got {self.max_num_blocks!r}"
            )
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                "gpu_memory_utilization must be in (0, 1], got " f"{self.gpu_memory_utilization!r}"
            )
        if self.max_decode_slots is not None and self.max_decode_slots < 1:
            raise ValueError(
                f"max_decode_slots must be a positive int or None, got "
                f"{self.max_decode_slots!r}"
            )
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens!r}")
        if self.max_tick_ms < 0:
            raise ValueError(f"max_tick_ms must be >= 0 (0 disables), got {self.max_tick_ms!r}")
        if self.decode_admit_window_ms < 0:
            raise ValueError(
                "decode_admit_window_ms must be >= 0 (0 disables), got "
                f"{self.decode_admit_window_ms!r}"
            )
        if self.enable_sequence_packing:
            if self.service_mode != "offline":
                raise ValueError("enable_sequence_packing requires service_mode='offline'")
            if self.max_packed_frames < 1:
                raise ValueError(
                    f"max_packed_frames must be a positive int, got " f"{self.max_packed_frames!r}"
                )
        # Track which of the checkpoint-derivable fields the caller set
        # explicitly, so a converter-emitted FeatureSpec / TokenizerSpec can
        # fill the defaults without overriding a deliberate choice (the engine
        # warns loudly when an explicit value disagrees with the spec).
        # ``audio_scale`` counts as explicit only when it differs from the
        # class default; setting the default is indistinguishable from omission.
        self._audio_scale_explicit = self.audio_scale != 32768.0
        self._feature_config_explicit = self.feature_config is not None
        self._tokenizer_paths_explicit = (
            self.sentencepiece_model is not None or self.unit_table is not None
        )
        if self.feature_config is None:
            self.feature_config = FeatureConfig(dither=0.0)
        # ``ctc_decoder_config`` / ``wfst_decoder_config`` are deliberately left
        # ``None`` here.  They are **CTC-family** options and are built by the
        # CTC strategies' ``options_cls`` factories, so other decode families do
        # not construct configs they never read.  Access them through
        # ``strategy.options.decoder_config``.
        if self.preferred_batch_size is not None:
            cleaned = sorted({int(v) for v in self.preferred_batch_size})
            if not cleaned:
                raise ValueError(
                    "preferred_batch_size must contain at least one value, " "or be None to disable"
                )
            if cleaned[0] < 1:
                raise ValueError(f"preferred_batch_size values must be >= 1, got {cleaned}")
            if cleaned[-1] > self.max_batch_size:
                raise ValueError(
                    f"preferred_batch_size values must be <= max_batch_size "
                    f"({self.max_batch_size}); got {cleaned}"
                )
            self.preferred_batch_size = cleaned
            # Default feature graph buckets to the preferred set so the two
            # caches share one ladder; explicit override still wins.
            if self.feature_graph_batch_buckets is None:
                self.feature_graph_batch_buckets = list(cleaned)
        # Auto-detect SentencePiece model and unit table from checkpoint dir.
        # Deprecated fallback: when the checkpoint conversion emits a
        # TokenizerSpec (WeNet / icefall / native all do), the engine builds the
        # tokenizer from the spec and these sniffed paths are ignored.
        if self.ckpt_dir and os.path.isdir(self.ckpt_dir):
            if self.sentencepiece_model is None:
                for fname in os.listdir(self.ckpt_dir):
                    if fname.endswith(".model"):
                        self.sentencepiece_model = os.path.join(self.ckpt_dir, fname)
                        break
            if self.unit_table is None:
                for fname in ("units.txt", "words.txt"):
                    candidate = os.path.join(self.ckpt_dir, fname)
                    if os.path.exists(candidate):
                        self.unit_table = candidate
                        break

    # ------------------------------------------------------------------
    # Subsampling constants for Conv2dSubsampling (4× with right_context=6)
    # ------------------------------------------------------------------

    # These geometry properties retain legacy defaults so a standalone
    # ``EngineConfig`` stays usable.  The
    # engine **overrides** them after loading the model (``ASREngine`` sets
    # ``_*_override`` from ``model.encoder`` / the streaming backend) so they
    # reflect the loaded architecture's streaming geometry.

    @property
    def subsampling_rate(self) -> int:
        """Total temporal subsampling factor (input frames per encoder frame)."""
        ov = getattr(self, "_subsampling_rate_override", None)
        return ov if ov is not None else 4

    @property
    def right_context(self) -> int:
        """Future input frames the subsampling needs beyond one chunk."""
        ov = getattr(self, "_right_context_override", None)
        return ov if ov is not None else 6

    @property
    def decoding_window(self) -> int:
        """Input feature frames consumed per encoder chunk.

        Default (Conformer): ``(chunk_size - 1) * subsampling_rate +
        right_context + 1``.  Overridden by the engine from the streaming
        backend (e.g. Zipformer's non-overlapping stateful window).
        """
        ov = getattr(self, "_decoding_window_override", None)
        if ov is not None:
            return ov
        return (self.chunk_size - 1) * self.subsampling_rate + self.right_context + 1

    @property
    def stride(self) -> int:
        """Feature frame stride between consecutive chunk windows."""
        ov = getattr(self, "_stride_override", None)
        if ov is not None:
            return ov
        return self.subsampling_rate * self.chunk_size

    # ------------------------------------------------------------------
    # CacheConfig builder
    # ------------------------------------------------------------------

    def build_cache_config(self, cache_spec: CacheSpec) -> CacheConfig:
        """Derive a :class:`CacheConfig` from a model's :class:`CacheSpec`.

        Parameters
        ----------
        cache_spec : CacheSpec
            Architecture-agnostic cache descriptor, e.g. ``model.cache_spec``
            (live model) or ``model_config.cache_spec`` (config object).

        Raises
        ------
        ValueError
            When ``max_num_blocks`` is still ``None``.  ``None`` means "derive
            from VRAM", which :class:`~oasr.engine.engine.ASREngine` resolves
            before it gets here; reaching this point unresolved means the caller
            asked for a pool on a device with no VRAM to measure.
        """
        if self.max_num_blocks is None:
            raise ValueError(
                "max_num_blocks=None means 'derive from free VRAM', which needs "
                "a CUDA device (ASREngine resolves it at construction). Set an "
                "explicit block count to build a cache config directly."
            )
        block_size = self.block_size_frames
        num_left_chunks = self.num_left_chunks
        window = cache_spec.fixed_attention_window
        prefill = window is not None
        if window is not None:
            # A trained fixed window is not a knob: how much history the model may
            # attend to is baked into its mask, so the engine derives the retained
            # cache from the model rather than from ``EngineConfig``.  One page per
            # chunk keeps eviction aligned with the trained chunk grid, and the
            # retained window is rounded *up* — any excess is masked by the
            # encoder's own ``chunked_limited`` bias, which it builds regardless.
            block_size = self.chunk_size
            num_left_chunks = -(-int(window) // self.chunk_size) + 1  # ceil + current
            if self.num_left_chunks != -1:
                logger.warning(
                    "num_left_chunks=%d ignored: this encoder declares a trained "
                    "attention window of %d frames, so the retained cache is "
                    "derived (%d chunks of %d frames)",
                    self.num_left_chunks,
                    window,
                    num_left_chunks,
                    self.chunk_size,
                )
        return CacheConfig(
            num_layers=cache_spec.num_layers,
            n_kv_head=cache_spec.n_kv_head,
            head_dim=cache_spec.head_dim,
            hidden_dim=cache_spec.hidden_dim,
            kernel_size=cache_spec.conv_kernel_size,
            stream_states=tuple(cache_spec.stream_states),
            prefill_kv_window=prefill,
            chunk_size=self.chunk_size,
            num_left_chunks=num_left_chunks,
            recycle_streaming_history=self.recycle_streaming_history,
            block_size_frames=block_size,
            max_num_blocks=self.max_num_blocks,
            max_blocks_per_seq=self.max_blocks_per_seq,
            max_batch_size=self.max_batch_size,
            device=torch.device(self.device),
            dtype=self.dtype,
        )
