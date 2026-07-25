# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR Engine configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from oasr.cache import CacheConfig
from oasr.ctc_decode import GpuDecoderConfig
from oasr.decode import DecoderConfig
from oasr.features import FeatureConfig
from oasr.models.base import BaseModelConfig, CacheSpec


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
    max_num_blocks : int
        Total number of physical KV-cache blocks in the shared block pool.
        Should satisfy ``max_num_blocks >= max_batch_size * max_blocks_per_seq``.
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
    ctc_decoder_config : GpuDecoderConfig, optional
        Config for ``decoder_type="ctc_cuda"``.  Defaults to
        ``GpuDecoderConfig()``.
    wfst_decoder_config : DecoderConfig, optional
        Config for ``decoder_type="ctc_wfst"``.  Defaults to
        ``DecoderConfig(search_type="wfst")``.
    fst_path : str, optional
        Path to the WFST FST file (only needed for ``"ctc_wfst"``).
    sentencepiece_model : str, optional
        Path to a SentencePiece ``.model`` file for detokenization.
        Auto-detected from ``ckpt_dir`` if not provided.
    unit_table : str, optional
        Path to a ``units.txt`` vocabulary file used as a fallback when
        SentencePiece is unavailable.  Auto-detected from ``ckpt_dir``.
    """

    ckpt_dir: str = ""
    checkpoint_name: str = "final.pt"
    device: str = "cuda"
    # bfloat16 by default: same exponent range as fp32, so wide-activation
    # models (e.g. conv2d6 subsampling, which can hit ~5e4 at the embed) do
    # not overflow the way fp16 (max 65504) does.
    dtype: torch.dtype = torch.bfloat16

    # Service mode — the engine runs in exactly one mode per lifecycle,
    # never both.  ``"streaming"`` admits chunk-by-chunk requests (paged
    # KV cache, partial outputs); ``"offline"`` admits full-audio
    # requests (length-bucketed micro-batched forward, single final
    # output).  ``ASREngine.add_request`` validates that the per-request
    # ``streaming`` flag matches this mode and raises ``ValueError`` on
    # drift — the Rust dispatcher relies on this to surface
    # misconfiguration eagerly instead of silently routing into a
    # quiescent executor.
    service_mode: str = "streaming"
    # Offline-only: overlap per-request admission prep (waveform normalise +
    # frame-count stamp — the GIL-bound CPU cost of ``add_request[s_batch]``)
    # with the GPU ``step()`` of the previous batch.  A daemon prep thread does
    # the heavy ``prepare_offline`` lock-free and hands finished requests to
    # ``step()`` via a thread-safe queue (``step`` drains it under the engine
    # lock it already holds — no extra locking, no ``run()`` deadlock).  Only
    # the cheap ``Request`` construction stays on the caller's thread.  Default
    # off; the serving front-end enables it (the GPU step is the overlap
    # window).  No effect in streaming mode.
    overlap_admit: bool = False

    # Streaming chunking
    chunk_size: int = 16
    num_left_chunks: int = -1
    # On the final streaming chunk (``is_last``), append one ``decoding_window``
    # of trailing silence so the last real-audio encoder window is FULL rather
    # than a short partial tail.  Recovers the final word the CTC decoder would
    # otherwise truncate (measured WER 8.54%→6.89% on 100 LJSpeech utts) and
    # keeps every real-audio window on the encoder CUDA-graph fast path (the
    # sub-window path is both slow-eager and graph-incorrect at B>1).  The
    # trailing silence decodes to blanks.  Set ``False`` to disable.
    finalize_silence_pad: bool = True

    # Batching
    # Encoder forward batch size — used in both modes since the service runs
    # streaming OR offline (never both at once).  In streaming mode it caps
    # the running pool, sizes the paged KV cache, and is the captured
    # CUDA-Graph B.  In offline mode it is the GPU forward width: the
    # scheduler admits one ``max_batch_size`` length-bucketed batch per
    # ``step()`` and :class:`OfflineExecutor` runs it as a single forward.
    max_batch_size: int = 32
    # Triton-style preferred batch sizes.  When set, the scheduler snaps
    # streaming admission and offline batch construction to one of these B
    # values (largest preferred ``<=`` available).  ``max_wait_time`` is the
    # escape valve when no preferred grouping is reachable.  Also drives the
    # encoder CUDA-Graph pre-warm at engine init and defaults
    # ``feature_graph_batch_buckets`` when that field is unset, so the two
    # graph caches share one bucket set.  Normalised in ``__post_init__``:
    # values are deduped, sorted ascending, and required to satisfy
    # ``1 <= v <= max_batch_size``.  ``None`` (default) preserves the legacy
    # "admit greedily up to ``max_batch_size``" behaviour.
    preferred_batch_size: Optional[List[int]] = None
    # Length-bucket tolerance for offline batching.  Requests are grouped so
    # that ``min_len / max_len >= length_bucket_ratio`` within a batch,
    # bounding padded-compute waste.  ``0`` disables this ratio entirely and
    # relies solely on ``max_offline_pad_ratio`` as the safety net.  Splitting
    # a bursty ``transcribe(list_of_N)`` call into sub-batches saves only a few
    # percent of padded GPU compute, so off-by-default is faster on real
    # datasets where adjacent-utterance length spread is moderate.
    length_bucket_ratio: float = 0.0
    # Hard cap on padded waste: reject a candidate from an offline batch when
    # adding it would push ``(max_len * batch_size) / sum_len`` above this ratio.
    # Last line of defence against mixing very short and very long clips.  The
    # default is permissive enough to admit e.g. LJSpeech (~1–10 s spread) in a
    # single batch but still guards against pathological mixes.
    max_offline_pad_ratio: float = 4.0
    # Length-aware batching: hard cap on **padded** input frames per offline
    # micro-batch, i.e. ``max_len * batch_size`` in pre-subsampling feature
    # frames (the same unit as ``Request.num_frames``).  ``None`` (default)
    # bounds each :class:`OfflineExecutor` micro-batch solely by
    # ``max_batch_size``.  When set, length-sorted requests are greedily grouped
    # into micro-batches bounded by this padded-frame budget (via
    # ``OfflineExecutor._split_by_frames``) so a mixed short/long pool never
    # forms an over-padded forward — exact-equivalent to the standard padded
    # forward, only the batch composition changes.  Independent of sequence
    # packing (``enable_sequence_packing``), which packs to a gapless varlen
    # forward instead; packing takes precedence when both are set.
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
    # Streaming cohort admission: when ``streaming_cohort_admit`` is True the
    # scheduler only admits new streaming requests when **either** the running
    # pool is empty **or** every running stream is still at ``offset == 0``
    # (i.e. has not yet run an encoder chunk).  This keeps every active
    # cohort in lockstep so that ``_forward_batched_paged`` can dispatch a
    # single ``B = max_batch_size`` encoder call instead of fragmenting into
    # many small offset groups.  The biggest streaming throughput win on
    # backlog-style workloads — at the cost of brief GPU idle time during
    # cohort transitions.  Set to ``False`` for maximally responsive
    # admission (one new request per freed slot).
    streaming_cohort_admit: bool = True

    # Sequence packing (offline only).  When ``True`` the offline executor
    # concatenates several utterances into one packed encoder forward instead
    # of padding each micro-batch to its max length.  Attention is restricted
    # to same-utterance tokens via ``cu_seqlens`` (varlen FMHA on the cute
    # path, per-segment SDPA fallback otherwise); the depthwise conv is
    # isolated per-segment with zero gap-frames; positional encoding + rel-pos
    # bias are rebuilt per segment.  Subsampling (``embed``) still runs in
    # normal batched mode so the Conv2d receptive field never crosses an
    # utterance boundary.  Mutually independent of ``max_batch_frames`` (that
    # governs the *non*-packing length-aware mode).
    enable_sequence_packing: bool = False
    # Token budget for one packed encoder row, in **post-subsampling** encoder
    # frames (≈ input_frames / 4).  The offline executor (packing mode) greedily fills a packed
    # row with whole utterances until the next one would push the summed
    # post-subsampling length (plus per-segment conv gap-frames) over this
    # budget, then spills into another row.  Sized to keep one packed forward
    # near the kernel's efficient occupancy without exhausting smem/registers.
    max_packed_frames: int = 8192

    # Paged KV cache
    max_num_blocks: int = 2048
    block_size_frames: int = 16
    max_blocks_per_seq: int = 512

    # CUDA Graph capture for the steady-state streaming encoder forward.
    # The cute DSL fmha is compiled with TVM-FFI (``--enable-tvm-ffi``)
    # and invoked with raw torch tensors, matching Flash Attention's
    # ``flash_attn/cute`` pattern. Capture + replay collapses the 12-layer
    # ~200-kernel encoder forward into a single CUDA Graph launch per
    # ``(B_active, cache_t1_bucket)`` shape; at steady state (B ==
    # max_batch_size) this is a clean win over the eager dispatch.
    #
    # The bias tile in the cute kernel is read unpredicated along T_q —
    # safe only when adjacent gmem is mapped. Eager calls always landed
    # adjacent allocations on the default pool so the over-read was
    # invisible; once graph capture started carving the address space into
    # private pools the over-read could fall into unmapped pages and trip
    # ``cudaErrorIllegalAddress``. ``RelPositionMultiHeadedAttention.
    # _forward_paged`` now pads ``combined_bias`` to the kernel's
    # ``(M_BLOCK, N_BLOCK)`` tile so every bias read stays in-bounds.
    use_cuda_graphs: bool = True

    # Sub-toggles for the two opt-in graph caches (gated by ``use_cuda_graphs``).
    # ``use_feature_cuda_graphs`` controls ``GraphedFeatureExtraction``
    # (batched fbank/mfcc capture). ``use_ctc_cuda_graphs`` controls the
    # per-state captured ``streaming_step`` graphs inside ``GpuStreamingDecoder``.
    #
    # Both default OFF. They were experimentally validated at small workloads
    # (N≈256 utts at B=64) where they shave 5-10% of wall time, but at
    # production-scale runs (N>=2000) the per-replay CPU work in the feature
    # graph (zero + copy + pin into the stable bucket buffer) and the per-non-blank
    # ``cudaMemcpy2DAsync`` of ``log_prob`` slices in the CTC graph outweigh
    # the kernel-launch savings — the captured intermediates also keep more
    # memory live in the graph pool. Eager mode is faster at scale; the
    # captured paths remain available for deployments where the trade-off
    # flips (small B / many short utterances / fixed preferred batch size).
    use_feature_cuda_graphs: bool = False
    use_ctc_cuda_graphs: bool = False
    # Optional override for the feature-graph B_active buckets. ``None``
    # selects power-of-two buckets up to ``max_batch_size``
    # (``[1, 2, 4, 8, 16, 32, ...]``). Services with a fixed preferred batch
    # size can pin a tighter list (e.g. ``[8, 32]``) to skip unused captures.
    feature_graph_batch_buckets: Optional[List[int]] = None

    # Feature extraction
    feature_config: Optional[FeatureConfig] = None

    # Decoding.  Default is the GPU CTC prefix beam — a single batched C++
    # kernel rather than an N-times Python loop (~50× faster per-utt at common
    # batch sizes, ~5 ms decode for 64 reqs).  Set to ``"ctc_wfst"`` for the
    # k2 WFST beam search (also GPU) when ``fst_path`` is provided.
    decoder_type: str = "ctc_cuda"
    ctc_decoder_config: Optional[GpuDecoderConfig] = None
    wfst_decoder_config: Optional[DecoderConfig] = None
    fst_path: Optional[str] = None

    # Decode-method selection among the model's capabilities.  ``None``
    # (default) runs ``model.default_decode_type`` — the unchanged production
    # behaviour.  Set to another capability the checkpoint advertises (e.g.
    # ``"ctc_aed_rescoring"`` on a U2++ hybrid) to opt in; the engine validates
    # the name against ``model.capabilities`` at construction.
    decode_method: Optional[str] = None

    # CTC+AED attention rescoring (``decode_method="ctc_aed_rescoring"``).
    # ``rescoring_ctc_weight`` fuses the CTC n-best score into the decoder
    # score (WeNet's decode-time ``ctc_weight``; 0.5 is the WeNet U2++ recipe
    # setting — distinct from the 0.3 *training* loss weight).
    # ``rescoring_reverse_weight`` weights the right-to-left decoder pass;
    # ``None`` uses the checkpoint's trained ``reverse_weight`` (0.0 on plain
    # transformer decoders — the reverse pass is then skipped entirely).
    # The n-best width is ``ctc_decoder_config.beam_size``.
    rescoring_ctc_weight: float = 0.5
    rescoring_reverse_weight: Optional[float] = None

    # Transducer (RNNT) greedy decode: cap on non-blank emissions per encoder
    # frame.  Safety bound against degenerate loops; applied uniformly so
    # results are deterministic.  Only read by ``decode_type == "transducer"``
    # models (see ``oasr/engine/decode/transducer.py``).
    transducer_max_sym_per_frame: int = 10

    # Incremental (label-synchronous AR) decode — AED / LLM strategies only.
    # ``decode_steps_per_tick`` caps the *batched* decoder steps one engine
    # ``step()`` runs across all pending requests, keeping per-tick work
    # bounded (the serving dispatcher's contract; see keystone K2 in
    # docs/design/multi_paradigm.md).  ``max_decode_slots`` caps how many
    # AR requests may be in flight before new-batch admission pauses;
    # ``None`` defaults to ``max_batch_size``.  Both are inert for
    # frame-synchronous strategies (CTC / transducer / rescoring).
    decode_steps_per_tick: int = 32
    max_decode_slots: Optional[int] = None
    # Wall-clock cap on one engine tick's decode phase, in milliseconds.  The
    # step cap above bounds *work*, not *time*, and one decoder step spans two
    # orders of magnitude across models (measured: ~1.5 ms for whisper-tiny at
    # B=8, ~18 ms for Qwen2-Audio-7B at B=4), so a fixed step count means a
    # ~50 ms tick on one model and a ~580 ms tick on another.  The serving
    # dispatcher holds the GIL for a whole tick, so that is the floor on cancel
    # latency, admission latency, and the gap between streaming partials.
    #
    # Whichever limit binds first wins: light models still run many steps per
    # tick, heavy models stop early and stream tokens at an interactive cadence.
    # The deadline stops *starting* steps rather than preempting one, so the real
    # bound is ``max_tick_ms + one step``.  ``0`` disables it (step cap only).
    # Inert for frame-synchronous strategies (CTC / transducer / rescoring),
    # which do not use the incremental protocol.
    max_tick_ms: float = 25.0

    # AR generation length cap (per request), read by incremental strategies.
    max_new_tokens: int = 448

    # Speech-LLM user prompt (``decode_method="llm"``): the text placed in the
    # checkpoint's chat template next to the audio (e.g. Qwen2-Audio's ASR
    # prompt).  ``None`` uses the model config's ``default_user_prompt``.
    llm_prompt: Optional[str] = None

    # Streaming interim-partial cadence.  After each streaming decode step the
    # engine reads the best-so-far hypothesis back to the host to emit a partial
    # transcript — a per-stream device→host sync that, profiled, is ~17% of
    # streaming wall time (the token bytes are trivial; the blocking
    # ``cudaStreamSynchronize`` is the cost).  ``1`` (default) emits a partial
    # every step but via one batched read-back for the whole ready set rather
    # than one ``.cpu()`` per stream.  ``N>1`` emits every N-th step (lower
    # partial cadence, less sync).  ``<=0`` disables interim partials entirely
    # (final transcript only) for throughput / non-interactive consumers — the
    # decode state still advances every step; only the read-back is skipped.
    partial_decode_interval: int = 1
    # Overlap the interim-partial read-back.  When ``False`` (default) each
    # emit step reads the beam buffer back with a blocking
    # ``cudaStreamSynchronize`` and emits *this* step's partial immediately —
    # the lowest first-token latency, best for **interactive** streaming.  When
    # ``True`` the read-back is issued non-blocking and the partial from the
    # *previous* emit step is emitted instead (a one-chunk lag), taking the
    # blocking sync off the critical path — a backlog/throughput optimization.
    # The final transcript (``finalize_streaming``) is always a blocking read,
    # so end-of-stream output is identical either way.  Off by default because
    # the engine's primary streaming target is interactive latency, which the
    # one-chunk partial lag regresses for no reliable throughput gain at low
    # concurrency.
    overlap_partial_readback: bool = False

    # Detokenization
    sentencepiece_model: Optional[str] = None
    unit_table: Optional[str] = None

    # Audio scale factor applied before feature extraction.
    # WeNet checkpoints are trained with Kaldi-style features where the audio
    # is at int16 scale (range ~[-32768, 32768]).  ``torchaudio.load`` returns
    # float32 normalized to [-1, 1], so multiply by 32768 to restore the scale.
    audio_scale: float = 32768.0

    # Set by the engine after model loading
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
        if self.max_decode_slots is not None and self.max_decode_slots < 1:
            raise ValueError(
                f"max_decode_slots must be a positive int or None, got "
                f"{self.max_decode_slots!r}"
            )
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens!r}")
        if self.max_tick_ms < 0:
            raise ValueError(f"max_tick_ms must be >= 0 (0 disables), got {self.max_tick_ms!r}")
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
        # class default (32768.0 — setting the default explicitly is a no-op);
        # Whisper checkpoints need the spec to flip it to 1.0.
        self._audio_scale_explicit = self.audio_scale != 32768.0
        self._feature_config_explicit = self.feature_config is not None
        self._tokenizer_paths_explicit = (
            self.sentencepiece_model is not None or self.unit_table is not None
        )
        if self.feature_config is None:
            self.feature_config = FeatureConfig(dither=0.0)
        if self.ctc_decoder_config is None:
            self.ctc_decoder_config = GpuDecoderConfig()
        if self.wfst_decoder_config is None:
            self.wfst_decoder_config = DecoderConfig(search_type="wfst")
        # Normalise preferred_batch_size: dedupe, sort, validate each <= cap.
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

    # These four geometry properties default to the Conformer/Conv2dSubsampling
    # values so a standalone ``EngineConfig`` (no model) stays usable.  The
    # engine **overrides** them after loading the model (``ASREngine`` sets
    # ``_*_override`` from ``model.encoder`` / the streaming backend) so they
    # reflect the actual architecture's streaming geometry — Conformer is
    # unchanged (4 / 6 / 67 / 64), Zipformer reports its stateful window.

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
        """
        return CacheConfig(
            num_layers=cache_spec.num_layers,
            n_kv_head=cache_spec.n_kv_head,
            head_dim=cache_spec.head_dim,
            hidden_dim=cache_spec.hidden_dim,
            kernel_size=cache_spec.conv_kernel_size,
            chunk_size=self.chunk_size,
            num_left_chunks=self.num_left_chunks,
            block_size_frames=self.block_size_frames,
            max_num_blocks=self.max_num_blocks,
            max_blocks_per_seq=self.max_blocks_per_seq,
            max_batch_size=self.max_batch_size,
            device=torch.device(self.device),
            dtype=self.dtype,
        )
