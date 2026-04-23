# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Audio loading and feature extraction for the ASR engine."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.features import FeatureConfig, extract_features_batch
from oasr.features.backends import _extract as _extract_single
from oasr.features.gpu_fbank import batched_fbank_gpu, supports_batched_gpu_fbank

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

        # Parallel feature-extraction worker pool.  Kaldi-style fbank/mfcc release
        # the GIL inside their C++ ops, so a small Python thread pool yields real
        # parallelism on multi-core boxes.  We also cap torch's intra-op thread
        # count because the default (``nproc``) oversubscribes for these
        # short-running ops and actively hurts single-call latency.
        nproc = os.cpu_count() or 2
        nw = int(getattr(config, "num_feature_workers", 0) or 0)
        if nw <= 0:
            # Empirically, 4 concurrent fbank ops saturates the throughput of a
            # single utterance-level extract; more workers trigger torch-pool
            # oversubscription and actively hurt.  Keep it small even on
            # many-core hosts.
            nw = min(4, max(1, nproc // 4))
        self._num_workers: int = nw
        self._pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=nw, thread_name_prefix="oasr-feat")
            if nw > 1 else None
        )

        intra = int(getattr(config, "cpu_intra_op_threads", 0) or 0)
        if intra <= 0:
            # Keep total thread count (workers × intra) near nproc without
            # overshooting; for nw=4 this resolves to 2, which matches the
            # empirical sweet spot for LJSpeech-scale workloads.
            intra = max(1, min(8, nproc // max(nw, 1)))
        if torch.get_num_threads() > intra:
            try:
                torch.set_num_threads(intra)
            except RuntimeError:
                # set_num_threads can fail if a parallel section is already open;
                # harmless — we'll just use the existing value.
                pass

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass

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

        Populates ``request.features``, ``request.feature_lengths``, and
        ``request.num_frames`` for each request, and also returns the
        stacked batch tensors.

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
        lengths_cpu = feat_lengths.detach().cpu().tolist()
        for i, req in enumerate(requests):
            req.features = features[i: i + 1]
            req.feature_lengths = feat_lengths[i: i + 1]
            req.num_frames = int(lengths_cpu[i])

        return features, feat_lengths

    def _estimate_num_frames(self, num_samples: int) -> int:
        """Cheap bucketing estimate of feature-frame count from sample count.

        Uses the Kaldi ``snip_edges`` formula so scheduler bucketing matches
        the exact length the subsequent batched extraction will produce, but
        without running the actual windowing / FFT.  Bucketing never needs
        frame-perfect accuracy, so we fall back to a simple hop-based
        estimate when ``snip_edges=False``.
        """
        cfg = self._feature_config
        frame_length = cfg.frame_length_samples
        frame_shift = cfg.frame_shift_samples
        if cfg.snip_edges:
            if num_samples < frame_length:
                return 0
            return (num_samples - frame_length) // frame_shift + 1
        if num_samples <= 0:
            return 0
        return (num_samples + frame_shift // 2) // frame_shift

    def _extract_one(self, waveform: torch.Tensor) -> torch.Tensor:
        """Per-waveform fbank/mfcc; runs on a pool worker when enabled."""
        return _extract_single(waveform, self._feature_config)

    def prepare_offline(self, request: Request) -> None:
        """Register an offline request without running feature extraction.

        Loads the waveform into ``request.waveform`` (cheap when the input is
        already a tensor or numpy array) and stamps a cheap sample-count
        based ``num_frames`` estimate so the scheduler can bucket by length
        without a D2H sync.  Actual fbank/mfcc extraction is deferred to
        :meth:`collate_offline`, where it runs as part of the pipelined
        producer so that each micro-batch's per-utterance extractions
        saturate the worker pool together (rather than being interleaved
        FIFO with unrelated requests from later micro-batches).
        """

        wav = self.load_audio(request.audio, request.sample_rate)
        request.waveform = wav
        request.num_frames = self._estimate_num_frames(int(wav.numel()))
        # Clear any stale feature cache from reused Request objects.
        request.features = None
        request.feature_lengths = None

    def prefetch_features(
        self, requests: List[Request]
    ) -> Dict[str, "Future[torch.Tensor]"]:
        """Submit fbank/mfcc futures for ``requests`` in the given order.

        Returns a ``request_id -> Future`` dict.  Callers should iterate in
        the *same* order they want features to be available; the pool
        processes submissions FIFO, so the head of ``requests`` finishes
        first.  This lets the offline pipeline sort by length and get
        chunk-0's features in a bounded wall-clock window regardless of how
        many later chunks are co-pending.

        Falls back to an empty dict when the worker pool is disabled;
        :meth:`collate_offline` will synthesise features on demand in that
        case.
        """
        if self._pool is None:
            return {}
        futures: Dict[str, "Future[torch.Tensor]"] = {}
        for r in requests:
            wav = r.waveform
            if wav is None:
                wav = self.load_audio(r.audio, r.sample_rate)
                r.waveform = wav
            futures[r.request_id] = self._pool.submit(self._extract_one, wav)
        return futures

    def collate_gpu(
        self,
        requests: List[Request],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched feature extraction on the GPU.

        Pads waveforms to a single ``(B, T_max)`` tensor, ships it to the
        device with one pinned non-blocking H2D, and runs fbank over the
        whole batch in one call.

        When the feature config matches the standard Kaldi-compliant fbank
        settings (see :func:`supports_batched_gpu_fbank`), uses the truly
        batched :func:`batched_fbank_gpu` implementation that issues a
        handful of fused kernels over the entire micro-batch — ~10× faster
        than looping :func:`torchaudio.compliance.kaldi.fbank` per
        utterance because it eliminates the per-call Python overhead.

        Falls back to the per-utterance loop for unusual configs
        (non-Povey windows, MFCC, dithered, etc.) so output quality is
        preserved even when the fast path is unavailable.

        Returns
        -------
        features : Tensor
            ``(B, max_feat_frames, F)`` padded feature tensor in the
            engine's configured dtype, on ``device``.
        feat_lengths : Tensor
            ``(B,)`` int32 valid frame counts on ``device``.
        """
        assert requests, "cannot collate empty batch"

        waveforms: List[torch.Tensor] = []
        for r in requests:
            wav = r.waveform
            if wav is None:
                wav = self.load_audio(r.audio, r.sample_rate)
            waveforms.append(wav)

        wav_lengths = torch.tensor(
            [w.size(0) for w in waveforms], dtype=torch.int64
        )
        T_max = int(wav_lengths.max().item())

        # One pinned H2D for the whole micro-batch of waveforms.
        padded_wav_cpu = torch.zeros(
            len(waveforms), T_max, dtype=torch.float32
        )
        for i, w in enumerate(waveforms):
            padded_wav_cpu[i, : w.size(0)] = w
        if self._device.type == "cuda":
            padded_wav_cpu = padded_wav_cpu.pin_memory()
        wav_gpu = padded_wav_cpu.to(device=self._device, non_blocking=True)

        if supports_batched_gpu_fbank(self._feature_config):
            lengths_gpu = wav_lengths.to(
                device=self._device, non_blocking=True
            )
            features_f32, feat_lengths = batched_fbank_gpu(
                wav_gpu, lengths_gpu, self._feature_config
            )
            features = features_f32.to(dtype=self._config.dtype)
        else:
            # Fall back to per-utterance fbank on GPU (still much faster
            # than CPU pool for non-standard configs).
            feat_list: List[torch.Tensor] = []
            for i, L in enumerate(wav_lengths.tolist()):
                feat_list.append(
                    _extract_single(wav_gpu[i, :L], self._feature_config)
                )
            feat_lengths = torch.tensor(
                [f.size(0) for f in feat_list],
                dtype=torch.int32, device=self._device,
            )
            padded_feat = torch.nn.utils.rnn.pad_sequence(
                feat_list, batch_first=True, padding_value=0.0
            )
            features = padded_feat.to(dtype=self._config.dtype)

        # Release waveforms; the GPU feature tensor owns the batch now.
        for r in requests:
            r.waveform = None

        return features, feat_lengths

    def collate_cpu(
        self,
        requests: List[Request],
        futures: Optional[Dict[str, "Future[torch.Tensor]"]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract + pad fbank/mfcc features into pinned CPU tensors.

        Does *not* issue any H2D copy.  The caller is expected to push the
        returned pinned tensors to the device on whatever CUDA stream it
        wants the copy to land on.  Returned pad tensor is float32; the
        caller chooses the target dtype when it converts on device.

        If ``futures`` is provided, harvests the fbank results from it
        (submitted eagerly by :meth:`prefetch_features`); otherwise runs
        extraction inline, batching across the worker pool when available.
        """
        assert requests, "cannot collate empty batch"

        # 1. Resolve per-utterance features on CPU.
        if futures:
            feat_list: List[torch.Tensor] = [
                futures.pop(r.request_id).result() for r in requests
            ]
        elif self._pool is not None and len(requests) > 1:
            waveforms = [
                r.waveform if r.waveform is not None
                else self.load_audio(r.audio, r.sample_rate)
                for r in requests
            ]
            feat_list = list(self._pool.map(self._extract_one, waveforms))
        else:
            feat_list = []
            for r in requests:
                wav = r.waveform
                if wav is None:
                    wav = self.load_audio(r.audio, r.sample_rate)
                feat_list.append(self._extract_one(wav))

        # 2. Pad + pin in preparation for a non-blocking H2D copy.
        feat_lengths_cpu = torch.tensor(
            [f.size(0) for f in feat_list], dtype=torch.int32
        )
        padded = torch.nn.utils.rnn.pad_sequence(
            feat_list, batch_first=True, padding_value=0.0
        )
        if self._device.type == "cuda" and not padded.is_pinned():
            padded = padded.pin_memory()
            feat_lengths_cpu = feat_lengths_cpu.pin_memory()

        # Release waveforms; we no longer need them once features exist.
        for r in requests:
            r.waveform = None

        return padded, feat_lengths_cpu

    def collate_offline(
        self,
        requests: List[Request],
        futures: Optional[Dict[str, "Future[torch.Tensor]"]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather per-waveform features into one padded batch on device.

        Convenience wrapper around :meth:`collate_cpu` that immediately
        issues the H2D copy on the current CUDA stream.  Prefer
        :meth:`collate_cpu` + explicit stream-controlled H2D in the
        pipelined path.
        """
        padded, feat_lengths_cpu = self.collate_cpu(requests, futures)
        features = padded.to(
            device=self._device,
            dtype=self._config.dtype,
            non_blocking=True,
        )
        feat_lengths = feat_lengths_cpu.to(
            device=self._device, non_blocking=True
        )
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

        Uses the batched GPU fbank fast path (:func:`batched_fbank_gpu`)
        on a single-item batch when the config permits — even for one
        utterance it's ~2–3× faster than the per-call
        ``torchaudio.compliance.kaldi.fbank`` path because it skips the
        per-call mel-bank construction and Python-level option building
        that dominates single-utt runtime.  Populates
        ``request.features``, ``request.feature_lengths``,
        ``request.num_frames``, and ``request.chunks_remaining``.
        """
        wav = request.waveform if request.waveform is not None \
            else self.load_audio(request.audio, request.sample_rate)

        if self._device.type == "cuda" \
                and supports_batched_gpu_fbank(self._feature_config):
            wav_cpu = wav
            if not wav_cpu.is_pinned():
                wav_cpu = wav_cpu.pin_memory()
            wav_gpu = wav_cpu.to(self._device, non_blocking=True).unsqueeze(0)
            lengths_gpu = torch.tensor(
                [wav.size(0)], dtype=torch.int64, device=self._device,
            )
            features_f32, feat_lengths = batched_fbank_gpu(
                wav_gpu, lengths_gpu, self._feature_config,
            )
            features = features_f32.to(dtype=self._config.dtype)
            # Trim to the one valid length before chunking so the
            # downstream windowing sees only the utterance's real frames.
            T = int(feat_lengths[0].item())
            features = features[:, :T, :]
        else:
            features, feat_lengths = extract_features_batch(
                [wav], self._feature_config,
            )
            features = features.to(
                device=self._device, dtype=self._config.dtype,
            )
            feat_lengths = feat_lengths.to(device=self._device)

        request.features = features  # (1, T, F)
        request.feature_lengths = feat_lengths
        request.num_frames = int(feat_lengths.item() if feat_lengths.numel() == 1
                                  else feat_lengths[0].item())
        request.chunks_remaining = self.chunk_features(features)
        request.waveform = None
