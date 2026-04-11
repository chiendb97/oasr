# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR Engine configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

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
        Maximum number of concurrent streaming requests in a single step.
    max_num_blocks : int
        Total number of physical KV-cache blocks in the shared block pool.
        Should satisfy ``max_num_blocks >= max_batch_size * max_blocks_per_seq``.
    block_size_frames : int
        Frames per KV-cache block (page).  Setting this equal to ``chunk_size``
        means each chunk maps to exactly one block.
    max_blocks_per_seq : int
        Maximum logical blocks per stream in the block table tensor.
    use_paged_cache : bool
        If True (default), use ``forward_chunk_paged`` + ``BlockPool``.
        If False, use ``forward_chunk`` with dense cache tensors.
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

    # Streaming chunking
    chunk_size: int = 16
    num_left_chunks: int = -1

    # Batching
    max_batch_size: int = 32

    # Paged KV cache
    max_num_blocks: int = 2048
    block_size_frames: int = 16
    max_blocks_per_seq: int = 512
    use_paged_cache: bool = True

    # Feature extraction
    feature_config: Optional[FeatureConfig] = None

    # Decoding
    decoder_type: str = "ctc_prefix_beam"
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
        if self.feature_config is None:
            self.feature_config = FeatureConfig(dither=0.0)
        if self.gpu_decoder_config is None:
            self.gpu_decoder_config = GpuDecoderConfig()
        if self.cpu_decoder_config is None:
            self.cpu_decoder_config = DecoderConfig(search_type="prefix_beam")
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

    @property
    def required_cache_size(self) -> int:
        """Attention cache size in encoder frames for dense streaming mode."""
        return self.chunk_size * self.num_left_chunks

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
            device=torch.device(self.device),
            dtype=self.dtype,
        )
