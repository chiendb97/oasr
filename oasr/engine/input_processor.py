# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Audio loading and feature extraction for the ASR engine."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch

from oasr.features import FeatureConfig, extract_features_batch

from .config import EngineConfig
from .request import Request


class InputProcessor:
    """Converts raw audio into model-ready features.

    Handles audio loading (file paths, tensors, NumPy arrays), batched
    feature extraction, and splitting full-length features into streaming
    chunk windows.

    CMVN is **not** applied here — it is baked into the model as a
    ``GlobalCMVN`` layer inside ``ConformerEncoder``.

    Parameters
    ----------
    config : EngineConfig
        Engine configuration (used for chunking params and feature config).
    device : torch.device
        Target device for output tensors.
    """

    def __init__(self, config: EngineConfig, device: torch.device) -> None:
        self._config = config
        self._device = device
        self._feature_config: FeatureConfig = config.feature_config  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def load_audio(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Load and normalise audio to a 1-D float32 waveform tensor (CPU).

        Parameters
        ----------
        audio : str, Tensor, or ndarray
            File path, waveform tensor ``(T,)`` / ``(1, T)``, or NumPy array.
        sample_rate : int
            Expected sample rate.  If a file is loaded at a different rate it
            is resampled to ``sample_rate``.

        Returns
        -------
        torch.Tensor
            1-D float32 CPU tensor of shape ``(T,)``.
        """
        scale = self._config.audio_scale
        if isinstance(audio, str):
            import torchaudio

            waveform, sr = torchaudio.load(audio)  # (C, T)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
            return (waveform.squeeze(0).float() * scale)
        elif isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio)
            if wav.dtype != torch.float32:
                wav = wav.float()
            return wav.squeeze() * scale
        elif isinstance(audio, torch.Tensor):
            wav = audio.float()
            if wav.dim() == 2:
                wav = wav.squeeze(0)
            return wav.cpu() * scale
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

    # ------------------------------------------------------------------
    # Batched offline processing
    # ------------------------------------------------------------------

    def process_offline(self, requests: List[Request]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load audio and extract features for a batch of offline requests.

        Populates ``request.features`` and ``request.feature_lengths`` for
        each request, and also returns the stacked batch tensors.

        Returns
        -------
        features : Tensor
            ``(B, max_T, F)`` padded feature tensor on the engine device.
        feature_lengths : Tensor
            ``(B,)`` valid frame counts on the engine device.
        """
        waveforms: List[torch.Tensor] = []
        for req in requests:
            wav = self.load_audio(req.audio, req.sample_rate)
            waveforms.append(wav)

        features, feat_lengths = extract_features_batch(waveforms, self._feature_config)
        features = features.to(device=self._device, dtype=self._config.dtype)
        feat_lengths = feat_lengths.to(device=self._device)

        # Cache on each request
        for i, req in enumerate(requests):
            req.features = features[i : i + 1]
            req.feature_lengths = feat_lengths[i : i + 1]

        return features, feat_lengths

    # ------------------------------------------------------------------
    # Streaming chunking
    # ------------------------------------------------------------------

    def chunk_features(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Split a ``(1, T, F)`` feature tensor into overlapping chunk windows.

        The windowing replicates the logic in
        ``ConformerEncoder.forward_chunk_by_chunk``:

        * stride = ``subsampling_rate * chunk_size``
        * window = ``(chunk_size - 1) * subsampling_rate + right_context + 1``

        Returns
        -------
        List[Tensor]
            List of ``(1, window, F)`` feature windows ready for
            ``model.forward_chunk`` / ``model.forward_chunk_paged``.
        """
        cfg = self._config
        stride = cfg.stride
        window = cfg.decoding_window
        num_frames = features.size(1)
        context = cfg.right_context + 1

        chunks: List[torch.Tensor] = []
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + window, num_frames)
            chunks.append(features[:, cur:end, :])
        return chunks

    def process_for_streaming(self, request: Request) -> None:
        """Load audio, extract features, and pre-chunk for streaming.

        Populates ``request.features``, ``request.feature_lengths``, and
        ``request.chunks_remaining``.
        """
        wav = self.load_audio(request.audio, request.sample_rate)
        waveforms = [wav]
        features, feat_lengths = extract_features_batch(waveforms, self._feature_config)
        features = features.to(device=self._device, dtype=self._config.dtype)
        feat_lengths = feat_lengths.to(device=self._device)

        request.features = features  # (1, T, F)
        request.feature_lengths = feat_lengths

        request.chunks_remaining = self.chunk_features(features)
