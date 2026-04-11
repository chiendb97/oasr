# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Chunk-by-chunk feature extraction: per-stream engine and batched wrapper."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch

from .backends import _extract, _kaldifeat_fbank_opts, _kaldifeat_mfcc_opts, _to_wav_list
from .config import FeatureConfig

logger = logging.getLogger("oasr")


class _StreamingFeatureExtractor:
    """Incremental feature extraction for one audio stream (implementation detail)."""

    def __init__(self, config: FeatureConfig) -> None:
        if not config.snip_edges:
            raise ValueError(
                "Batched streaming requires snip_edges=True for correct frame "
                "alignment across chunk boundaries."
            )
        self._config = config
        self._remainder: Optional[torch.Tensor] = None
        self._num_frames_extracted: int = 0

        self._online_extractor: object = None
        self._online_frames_fetched: int = 0
        if config.backend == "kaldifeat":
            self._init_kaldifeat_online()

    def _init_kaldifeat_online(self) -> None:
        try:
            import kaldifeat
        except ImportError:
            raise ImportError(
                "kaldifeat is required for the 'kaldifeat' backend. "
                "Install with: pip install kaldifeat"
            ) from None

        if self._config.feature_type == "fbank" and hasattr(kaldifeat, "OnlineFbank"):
            opts = _kaldifeat_fbank_opts(self._config)
            self._online_extractor = kaldifeat.OnlineFbank(opts)
        elif self._config.feature_type == "mfcc" and hasattr(kaldifeat, "OnlineMfcc"):
            opts = _kaldifeat_mfcc_opts(self._config)
            self._online_extractor = kaldifeat.OnlineMfcc(opts)
        else:
            logger.debug(
                "kaldifeat online classes not available; "
                "falling back to remainder-buffer strategy"
            )

    @property
    def config(self) -> FeatureConfig:
        return self._config

    @property
    def num_frames_extracted(self) -> int:
        if self._online_extractor is not None:
            return self._online_frames_fetched
        return self._num_frames_extracted

    def process_chunk(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        if self._online_extractor is not None:
            return self._process_chunk_kaldifeat(waveform)
        return self._process_chunk_torchaudio(waveform)

    def _process_chunk_torchaudio(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        wav = waveform.squeeze(0) if waveform.dim() == 2 else waveform

        if self._remainder is not None and self._remainder.numel() > 0:
            combined = torch.cat([self._remainder, wav])
        else:
            combined = wav

        n_samples = combined.numel()
        frame_len = self._config.frame_length_samples

        if n_samples < frame_len:
            self._remainder = combined
            return None

        features = _extract(combined, self._config)
        n_frames = features.size(0)

        frame_shift = self._config.frame_shift_samples
        consumed_start = n_frames * frame_shift
        if consumed_start < n_samples:
            self._remainder = combined[consumed_start:]
        else:
            self._remainder = combined.new_empty(0)

        self._num_frames_extracted += n_frames
        return features

    def _process_chunk_kaldifeat(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        wav = waveform.squeeze(0) if waveform.dim() == 2 else waveform
        self._online_extractor.accept_waveform(  # type: ignore[union-attr]
            float(self._config.sample_rate),
            wav.to(torch.float32),
        )
        num_ready = self._online_extractor.num_frames_ready  # type: ignore[union-attr]
        if num_ready <= self._online_frames_fetched:
            return None
        frame_indices = list(range(self._online_frames_fetched, num_ready))
        features = self._online_extractor.get_frames(frame_indices)  # type: ignore[union-attr]
        self._online_frames_fetched = num_ready
        return features

    def flush(self) -> Optional[torch.Tensor]:
        if self._online_extractor is not None:
            return self._flush_kaldifeat()
        return self._flush_torchaudio()

    def _flush_torchaudio(self) -> Optional[torch.Tensor]:
        if self._remainder is None or self._remainder.numel() == 0:
            return None

        remainder = self._remainder
        self._remainder = remainder.new_empty(0)

        frame_len = self._config.frame_length_samples
        if remainder.numel() < frame_len:
            padding = remainder.new_zeros(frame_len - remainder.numel())
            remainder = torch.cat([remainder, padding])

        features = _extract(remainder, self._config)
        if features.numel() == 0:
            return None

        self._num_frames_extracted += features.size(0)
        return features

    def _flush_kaldifeat(self) -> Optional[torch.Tensor]:
        if hasattr(self._online_extractor, "input_finished"):
            self._online_extractor.input_finished()  # type: ignore[union-attr]
        num_ready = self._online_extractor.num_frames_ready  # type: ignore[union-attr]
        if num_ready <= self._online_frames_fetched:
            return None
        frame_indices = list(range(self._online_frames_fetched, num_ready))
        features = self._online_extractor.get_frames(frame_indices)  # type: ignore[union-attr]
        self._online_frames_fetched = num_ready
        return features

    def reset(self) -> None:
        self._remainder = None
        self._num_frames_extracted = 0
        self._online_frames_fetched = 0
        if self._config.backend == "kaldifeat":
            self._online_extractor = None
            self._init_kaldifeat_online()


def _collate_optional_features(
    feat_list: List[Optional[torch.Tensor]],
    output_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad per-stream feature tensors (some may be ``None``) to ``(B, max_frames, F)``."""
    B = len(feat_list)
    feat_lengths = torch.zeros(B, dtype=torch.long)
    for i, f in enumerate(feat_list):
        if f is not None:
            feat_lengths[i] = f.size(0)

    max_frames = int(feat_lengths.max().item()) if B > 0 else 0

    ref = next((f for f in feat_list if f is not None), None)
    dtype = ref.dtype if ref is not None else torch.float32
    device = ref.device if ref is not None else torch.device("cpu")

    padded = torch.zeros(B, max_frames, output_dim, dtype=dtype, device=device)
    for i, f in enumerate(feat_list):
        if f is not None and f.size(0) > 0:
            padded[i, : f.size(0)] = f

    return padded, feat_lengths


class BatchedStreamingFeatureExtractor:
    """Batched incremental (chunk-by-chunk) audio feature extractor.

    Manages *B* independent audio streams.  Each :meth:`process_chunk` call
    feeds one chunk per stream and returns padded features
    ``(B, max_new_frames, output_dim)`` suitable for batched inference.

    Usage::

        from oasr.features import FeatureConfig, BatchedStreamingFeatureExtractor

        config = FeatureConfig(num_mel_bins=80, dither=0.0)
        ext = BatchedStreamingFeatureExtractor(config, batch_size=4)

        for chunk_batch in audio_chunk_batches:
            feats, feat_lens = ext.process_chunk(chunk_batch)
            model.forward_chunk(feats, feat_lens, ...)

        final_feats, final_lens = ext.flush()
        ext.reset()

    Individual streams can be reset via ``reset(stream_indices=[i])``.

    Parameters
    ----------
    config : FeatureConfig
        Shared configuration for all streams.
    batch_size : int
        Number of parallel streams.
    """

    def __init__(self, config: FeatureConfig, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._config = config
        self._batch_size = batch_size
        self._extractors = [
            _StreamingFeatureExtractor(config) for _ in range(batch_size)
        ]

    @property
    def config(self) -> FeatureConfig:
        return self._config

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_frames_extracted(self) -> torch.Tensor:
        return torch.tensor(
            [e.num_frames_extracted for e in self._extractors],
            dtype=torch.long,
        )

    def process_chunk(
        self,
        waveforms: Union[torch.Tensor, List[torch.Tensor]],
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one chunk per stream; return padded new frames and per-stream lengths."""
        wavs = _to_wav_list(waveforms, lengths)
        if len(wavs) != self._batch_size:
            raise ValueError(
                f"Expected {self._batch_size} waveforms (one per stream), "
                f"got {len(wavs)}"
            )

        feat_list: List[Optional[torch.Tensor]] = [
            self._extractors[i].process_chunk(w) for i, w in enumerate(wavs)
        ]
        return _collate_optional_features(feat_list, self._config.output_dim)

    def flush(self) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_list: List[Optional[torch.Tensor]] = [
            e.flush() for e in self._extractors
        ]
        return _collate_optional_features(feat_list, self._config.output_dim)

    def reset(self, stream_indices: Optional[List[int]] = None) -> None:
        if stream_indices is None:
            for e in self._extractors:
                e.reset()
        else:
            for i in stream_indices:
                self._extractors[i].reset()
