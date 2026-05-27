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
from oasr.models.conformer.config import ConformerModelConfig


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
        forward width of each pipeline micro-batch.  The service runs in
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
        Which CTC decoder to use:
        ``"ctc_greedy"`` — fast CPU greedy,
        ``"ctc_prefix_beam"`` — CPU prefix beam search (default),
        ``"ctc_gpu"`` — GPU beam search via ``GpuStreamingDecoder``,
        ``"ctc_wfst"`` — CPU WFST beam search (requires k2).
    gpu_decoder_config : GpuDecoderConfig, optional
        Config for ``decoder_type="ctc_gpu"``.  Defaults to
        ``GpuDecoderConfig()``.
    cpu_decoder_config : DecoderConfig, optional
        Config for CPU decoders.  Defaults to ``DecoderConfig()``.
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
    dtype: torch.dtype = torch.float16

    # Service mode — the engine runs in exactly one mode per lifecycle,
    # never both.  ``"streaming"`` admits chunk-by-chunk requests (paged
    # KV cache, partial outputs); ``"offline"`` admits full-audio
    # requests (length-bucketed micro-batched forward, single final
    # output).  ``ASREngine.add_request`` validates that the per-request
    # ``streaming`` flag matches this mode and raises ``ValueError`` on
    # drift — the Rust dispatcher relies on this to surface
    # misconfiguration eagerly instead of silently routing into a
    # quiescent pipeline.
    service_mode: str = "streaming"

    # Streaming chunking
    chunk_size: int = 16
    num_left_chunks: int = -1

    # Batching
    # Encoder forward batch size — used in both modes since the service runs
    # streaming OR offline (never both at once).  In streaming mode it caps
    # the running pool, sizes the paged KV cache, and is the captured
    # CUDA-Graph B.  In offline mode it is the GPU forward width of each
    # micro-batch in :class:`OfflinePipeline`; the scheduler admits up to
    # ``max_batch_size * offline_pipeline_depth`` requests per ``step()`` so
    # the pipeline producer always has enough work queued to keep
    # ``offline_pipeline_depth`` micro-batches in flight.
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
    # a bursty ``transcribe(list_of_N)`` call into sub-batches multiplies CPU
    # feature-extraction cost (sequential torchaudio passes) while saving only
    # a few percent of padded GPU compute, so off-by-default is faster on real
    # datasets where adjacent-utterance length spread is moderate.
    length_bucket_ratio: float = 0.0
    # Hard cap on padded waste: reject a candidate from an offline batch when
    # adding it would push ``(max_len * batch_size) / sum_len`` above this ratio.
    # Last line of defence against mixing very short and very long clips.  The
    # default is permissive enough to admit e.g. LJSpeech (~1–10 s spread) in a
    # single batch but still guards against pathological mixes.
    max_offline_pad_ratio: float = 4.0
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
    # How many offline micro-batches to keep in flight at once.  ``1`` runs
    # sequentially (no threading, no CPU/GPU overlap).  ``2+`` runs CPU
    # feature prep for micro-batch ``k+1`` on a background thread while GPU
    # forward+decode executes micro-batch ``k`` on the main thread.  Values
    # above 3 rarely help because CPU prep is usually only 1–2 micro-batches
    # ahead of the GPU in steady state.
    offline_pipeline_depth: int = 3
    # Keep the offline producer thread alive across ``step()`` calls so the
    # collate of batch ``k+1`` overlaps with the encoder-forward+decode of
    # batch ``k``.  When ``False`` each step rebuilds a fresh producer (the
    # legacy per-batch behavior), which is fine for ``transcribe(list)``
    # but loses pipelining when reqs arrive one-at-a-time over HTTP/gRPC
    # and the scheduler only hands out small per-step batches.  Defaults
    # ``True`` because the HTTP-frontend workload is the throughput-critical
    # case and the cross-batch behavior is a strict superset of legacy.
    offline_persistent_pipeline: bool = True
    # Run fbank/mfcc feature extraction on the GPU for offline batches.
    # When ``True`` (default) the pipeline pads waveforms, ships them to
    # the device with one H2D copy, and runs torchaudio's kaldi-compliant
    # fbank per-utterance on a dedicated CUDA stream.  This bypasses the
    # CPU fbank pool, which was the throughput ceiling for offline
    # batching (~550 fbanks/s with 4 workers).  When ``False`` the pipeline
    # falls back to the CPU pool path — useful on GPUs that are already
    # saturated by the model forward.
    offline_gpu_feature_extraction: bool = True

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
    # Number of parallel CPU workers used to extract fbank/mfcc features for
    # offline batches.  ``0`` = auto-detect a sensible default (``min(8,
    # nproc // 2)``).  ``1`` disables the worker pool entirely.
    # Kaldi-style fbank releases the GIL inside its C++ op, so a small thread
    # pool yields real parallelism and is typically the largest CPU-side
    # speedup for offline throughput.
    num_feature_workers: int = 0
    # Intra-op thread count for PyTorch CPU ops (``torch.set_num_threads``).
    # ``0`` = auto: ``min(16, 4 * num_feature_workers)``.  The PyTorch default
    # is ``nproc`` which heavily oversubscribes for short CPU ops like
    # per-utterance fbank and causes large slowdowns on many-core hosts.
    cpu_intra_op_threads: int = 0

    # Decoding.  Default is the GPU CTC prefix beam — same algorithm as
    # ``ctc_prefix_beam`` but the inner loop is a single batched C++ kernel
    # instead of an N-times Python loop.  Benchmarked ~50× faster per-utt
    # at common batch sizes (~5 ms decode for 64 reqs vs ~250 ms on CPU).
    # Set to ``"ctc_prefix_beam"`` for the CPU path (e.g. when running
    # without a GPU build), or ``"ctc_wfst"`` when ``fst_path`` is set.
    decoder_type: str = "ctc_gpu"
    gpu_decoder_config: Optional[GpuDecoderConfig] = None
    cpu_decoder_config: Optional[DecoderConfig] = None
    fst_path: Optional[str] = None

    # Detokenization
    sentencepiece_model: Optional[str] = None
    unit_table: Optional[str] = None

    # Audio scale factor applied before feature extraction.
    # WeNet checkpoints are trained with Kaldi-style features where the audio
    # is at int16 scale (range ~[-32768, 32768]).  ``torchaudio.load`` returns
    # float32 normalized to [-1, 1], so multiply by 32768 to restore the scale.
    audio_scale: float = 32768.0

    # Set by the engine after model loading
    _model_config: Optional[ConformerModelConfig] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.service_mode not in ("streaming", "offline"):
            raise ValueError(
                f"service_mode must be 'streaming' or 'offline', got "
                f"{self.service_mode!r}"
            )
        if self.feature_config is None:
            self.feature_config = FeatureConfig(dither=0.0)
        if self.gpu_decoder_config is None:
            self.gpu_decoder_config = GpuDecoderConfig()
        if self.cpu_decoder_config is None:
            self.cpu_decoder_config = DecoderConfig(search_type="prefix_beam")
        # Normalise preferred_batch_size: dedupe, sort, validate each <= cap.
        if self.preferred_batch_size is not None:
            cleaned = sorted({int(v) for v in self.preferred_batch_size})
            if not cleaned:
                raise ValueError(
                    "preferred_batch_size must contain at least one value, "
                    "or be None to disable"
                )
            if cleaned[0] < 1:
                raise ValueError(
                    f"preferred_batch_size values must be >= 1, got {cleaned}"
                )
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
        # Auto-detect SentencePiece model and unit table from checkpoint dir
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

    @property
    def subsampling_rate(self) -> int:
        """4× temporal subsampling from Conv2dSubsampling."""
        return 4

    @property
    def right_context(self) -> int:
        """Right context frames required by Conv2dSubsampling."""
        return 6

    @property
    def decoding_window(self) -> int:
        """Input feature frames consumed per encoder chunk.

        ``(chunk_size - 1) * subsampling_rate + right_context + 1``
        """
        return (self.chunk_size - 1) * self.subsampling_rate + self.right_context + 1

    @property
    def stride(self) -> int:
        """Feature frame stride between consecutive chunk windows."""
        return self.subsampling_rate * self.chunk_size

    # ------------------------------------------------------------------
    # CacheConfig builder
    # ------------------------------------------------------------------

    def build_cache_config(self, model_config: ConformerModelConfig) -> CacheConfig:
        """Derive a :class:`CacheConfig` from the loaded model configuration.

        Parameters
        ----------
        model_config : ConformerModelConfig
            The model configuration returned by ``load_wenet_checkpoint``.
        """
        enc = model_config.encoder
        n_kv_head = enc.n_kv_head if enc.n_kv_head is not None else enc.attention_heads
        head_dim = enc.head_dim if enc.head_dim is not None else enc.output_size // enc.attention_heads
        return CacheConfig(
            num_layers=enc.num_blocks,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            hidden_dim=enc.output_size,
            kernel_size=enc.cnn_module_kernel,
            chunk_size=self.chunk_size,
            num_left_chunks=self.num_left_chunks,
            block_size_frames=self.block_size_frames,
            max_num_blocks=self.max_num_blocks,
            max_blocks_per_seq=self.max_blocks_per_seq,
            max_batch_size=self.max_batch_size,
            device=torch.device(self.device),
            dtype=self.dtype,
        )
