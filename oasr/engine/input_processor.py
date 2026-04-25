# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Audio loading and feature extraction for the ASR engine."""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.features import FeatureConfig, extract_features_batch
from oasr.features.backends import _extract as _extract_single
from oasr.features.gpu_fbank import batched_fbank_gpu, supports_batched_gpu_fbank

from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .config import EngineConfig
from .request import Request


class _NullCtx:
    """No-op context manager used when fbank doesn't need a dedicated stream."""

    __slots__ = ()

    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


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

        Kept for tests and legacy use.  The live streaming path now slices
        encoder chunks directly out of ``Request.feature_buffer`` in the
        engine step loop; this helper is no longer called for admission.

        * stride = ``subsampling_rate * chunk_size``
        * window = ``(chunk_size - 1) * subsampling_rate + right_context + 1``
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

    # ------------------------------------------------------------------
    # Streaming audio ingest + per-step batched fbank
    # ------------------------------------------------------------------

    def prepare_streaming(self, request: Request) -> None:
        """Register a streaming request from a full waveform (legacy API).

        Splits the raw waveform into **audio-sample** chunks that the engine
        then feeds into the fbank path incrementally — one chunk per step per
        stream.  Only the tail of the previous chunk is re-used when
        extracting features, so the engine never looks at future audio when
        processing the current chunk.

        The default audio-chunk size corresponds to ``stride`` feature frames
        (i.e. one encoder chunk worth of new audio): at 16 kHz with a 10 ms
        hop and ``chunk_size=16``/``subsampling_rate=4`` that's 10240 samples
        ≈ 640 ms.

        Prefer :meth:`prepare_streaming_open` + :meth:`append_streaming_chunk`
        for chunk-by-chunk feeding, which models a real-time client more
        faithfully and avoids the up-front waveform load.
        """
        wav = request.waveform if request.waveform is not None \
            else self.load_audio(request.audio, request.sample_rate)

        # Keep the waveform on CPU float32 for fbank; GPU promotion is batched
        # in ``extract_streaming_batch``.
        wav_cpu = wav.detach()
        if wav_cpu.device.type != "cpu":
            wav_cpu = wav_cpu.cpu()

        chunk_samples = self.streaming_audio_chunk_samples
        chunks: "deque[torch.Tensor]" = deque()
        n = wav_cpu.numel()
        for start in range(0, n, chunk_samples):
            chunks.append(wav_cpu[start: start + chunk_samples].contiguous())

        request.audio_chunks = chunks
        request.audio_tail = wav_cpu.new_empty(0)
        request.audio_final = True  # whole-waveform API — no more audio will arrive
        request.num_frames = self._estimate_num_frames(n)
        request.feature_buffer = None
        request.feature_frames = 0
        request.feature_cursor = 0
        request.waveform = None

    def prepare_streaming_open(self, request: Request) -> None:
        """Register an empty streaming request — chunks arrive via
        :meth:`append_streaming_chunk`.

        Sets up the streaming state with an empty audio queue.  No fbank
        runs and no waveform load happens here; the engine starts processing
        as soon as the first chunk lands.
        """
        request.audio_chunks = deque()
        request.audio_tail = torch.empty(0, dtype=torch.float32)
        request.audio_final = False
        request.num_frames = 0
        request.feature_buffer = None
        request.feature_frames = 0
        request.feature_cursor = 0
        request.waveform = None

    def append_streaming_chunk(
        self,
        request: Request,
        chunk: Union[torch.Tensor, np.ndarray],
        is_last: bool = False,
    ) -> None:
        """Push one audio chunk onto an open streaming request.

        Parameters
        ----------
        request : Request
            A request previously initialised with :meth:`prepare_streaming_open`.
        chunk : Tensor or ndarray
            1-D audio samples (CPU or GPU; converted to CPU float32).
        is_last : bool
            ``True`` marks the final chunk — sets ``audio_final`` so the
            engine flushes the trailing partial frame.
        """
        if request.audio_chunks is None:
            raise RuntimeError(
                "append_streaming_chunk called on a request that was not "
                "initialised via prepare_streaming_open"
            )
        if request.audio_final:
            raise RuntimeError(
                f"feed_chunk after is_last=True for request {request.request_id}"
            )

        # Normalise to CPU float32 1-D, scaled like load_audio does.
        scale = self._config.audio_scale
        if isinstance(chunk, np.ndarray):
            wav: torch.Tensor = torch.from_numpy(chunk)
        elif isinstance(chunk, torch.Tensor):
            wav = chunk
        else:  # type: ignore[unreachable]
            raise TypeError(f"Unsupported chunk type: {type(chunk)}")
        if wav.dtype != torch.float32:
            wav = wav.float()
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        if wav.device.type != "cpu":
            wav = wav.cpu()
        if scale != 1.0:
            wav = wav * scale
        wav = wav.contiguous()

        request.audio_chunks.append(wav)
        request.samples_enqueued += wav.numel()
        if is_last:
            request.audio_final = True
        # Keep the scheduler's bucket estimate roughly in sync.  O(1) using
        # the running total instead of re-summing the deque per chunk.
        request.num_frames = self._estimate_num_frames(request.samples_enqueued)

    @property
    def streaming_audio_chunk_samples(self) -> int:
        """Default per-step audio-chunk size in samples (= ``stride`` frames)."""
        fcfg = self._feature_config
        return self._config.stride * fcfg.frame_shift_samples

    def extract_streaming_batch(
        self,
        requests: List[Request],
        cuda_stream: Optional["torch.cuda.Stream"] = None,
    ) -> None:
        """Run one batched GPU-fbank call over all queued streams.

        For each request with a pending audio chunk, this pops the next
        chunk, prepends the previous ``audio_tail``, pads all streams to the
        max combined length, ships one ``(B, T)`` waveform to the device,
        and runs :func:`batched_fbank_gpu` once for the whole batch.  The
        per-stream new frames are concatenated onto ``feature_buffer``.

        No stream is allowed to look at samples beyond its own enqueued
        chunk — we only fuse across *different* streams, never across future
        chunks of the same stream.

        Parameters
        ----------
        cuda_stream : torch.cuda.Stream, optional
            When provided (and the engine is on CUDA) the H2D copy and
            ``batched_fbank_gpu`` kernel run on this stream so they can
            overlap with the encoder forward on the default stream.  The
            caller is responsible for synchronising the default stream
            against fbank completion before reading ``feature_buffer``.
        """
        if not requests:
            return

        fcfg = self._feature_config
        frame_shift = fcfg.frame_shift_samples
        frame_len = fcfg.frame_length_samples
        feat_dim = fcfg.output_dim
        dtype = self._config.dtype
        device = self._device

        # Collect per-stream (combined waveform, is_final_flush) pairs.
        combined: List[torch.Tensor] = []
        targets: List[Request] = []
        is_flush: List[bool] = []
        for req in requests:
            if req.audio_chunks is None or req.audio_tail is None:
                continue
            if req.audio_chunks:
                chunk = req.audio_chunks.popleft()
                cat = chunk if req.audio_tail.numel() == 0 \
                    else torch.cat([req.audio_tail, chunk])
                flush = req.audio_final and not req.audio_chunks \
                    and cat.numel() >= frame_len
                # On the very last chunk pad the tail so the final partial
                # frame still gets emitted (mirrors
                # ``_StreamingFeatureExtractor._flush_torchaudio``).
                if req.audio_final and not req.audio_chunks:
                    if cat.numel() < frame_len:
                        pad = cat.new_zeros(frame_len - cat.numel())
                        cat = torch.cat([cat, pad])
                        flush = True
                combined.append(cat)
                targets.append(req)
                is_flush.append(flush)
            elif req.audio_final and req.audio_tail.numel() > 0:
                # No chunks left but tail still carries unconsumed samples
                # (can happen when the whole waveform was < chunk_samples).
                cat = req.audio_tail
                if cat.numel() < frame_len:
                    pad = cat.new_zeros(frame_len - cat.numel())
                    cat = torch.cat([cat, pad])
                combined.append(cat)
                targets.append(req)
                is_flush.append(True)

        if not targets:
            return

        # Drop streams whose combined buffer is still shorter than one
        # frame — they need more audio before fbank can produce anything.
        # Keep their tail updated and retry next step.
        fbank_inputs: List[torch.Tensor] = []
        fbank_reqs: List[Request] = []
        fbank_flush: List[bool] = []
        for req, cat, flush in zip(targets, combined, is_flush):
            if cat.numel() < frame_len and not flush:
                req.audio_tail = cat
                continue
            fbank_inputs.append(cat)
            fbank_reqs.append(req)
            fbank_flush.append(flush)

        if not fbank_reqs:
            return

        nvtx_push("pad+pin")
        lengths_cpu = torch.tensor(
            [w.numel() for w in fbank_inputs], dtype=torch.int64
        )
        T_max = int(lengths_cpu.max().item())

        padded_cpu = torch.zeros(len(fbank_inputs), T_max, dtype=torch.float32)
        for i, w in enumerate(fbank_inputs):
            padded_cpu[i, : w.numel()] = w
        if device.type == "cuda":
            padded_cpu = padded_cpu.pin_memory()
            lengths_cpu_pin = lengths_cpu.pin_memory()
        else:
            lengths_cpu_pin = lengths_cpu
        nvtx_pop()

        use_gpu_fbank = device.type == "cuda" and supports_batched_gpu_fbank(fcfg)
        # If a dedicated CUDA stream was provided, run the H2D copy and the
        # fbank kernel on it so they can overlap with the encoder forward on
        # the default stream.  The caller is responsible for inserting the
        # event-wait before reading feature_buffer.
        use_alt_stream = (
            use_gpu_fbank and cuda_stream is not None
        )
        if use_gpu_fbank:
            stream_ctx = (
                torch.cuda.stream(cuda_stream) if use_alt_stream
                else _NullCtx()
            )
            with stream_ctx:
                nvtx_push("h2d")
                wav_gpu = padded_cpu.to(device=device, non_blocking=True)
                lengths_gpu = lengths_cpu_pin.to(device=device, non_blocking=True)
                nvtx_pop()
                nvtx_push("fbank")
                feats_f32, feat_lens = batched_fbank_gpu(wav_gpu, lengths_gpu, fcfg)
                feats = feats_f32.to(dtype=dtype)
                feat_lens_cpu = feat_lens.to(device="cpu").tolist()
                nvtx_pop()
        else:
            # Fallback: per-utt CPU/torchaudio fbank.  Used by non-Povey
            # configs and CPU-only devices.
            feat_list: List[torch.Tensor] = []
            feat_lens_cpu_list: List[int] = []
            for i, w in enumerate(fbank_inputs):
                f = _extract_single(w, fcfg)
                feat_list.append(f)
                feat_lens_cpu_list.append(f.size(0))
            max_nf = max(feat_lens_cpu_list) if feat_lens_cpu_list else 0
            feats_cpu = torch.zeros(
                len(feat_list), max_nf, feat_dim, dtype=torch.float32
            )
            for i, f in enumerate(feat_list):
                feats_cpu[i, : f.size(0)] = f
            feats = feats_cpu.to(device=device, dtype=dtype, non_blocking=True)
            feat_lens_cpu = feat_lens_cpu_list

        # Distribute new frames back into per-stream buffers and update tails.
        nvtx_push("distribute")
        for i, req in enumerate(fbank_reqs):
            new_nf = int(feat_lens_cpu[i])
            if new_nf > 0:
                new_feats = feats[i, :new_nf, :]  # (new_nf, F) view on device
                self._append_features(req, new_feats, feat_dim)
            # New tail = samples beyond the last consumed frame
            consumed = new_nf * frame_shift
            cat = fbank_inputs[i]
            if fbank_flush[i]:
                req.audio_tail = cat.new_empty(0)
            elif consumed < cat.numel():
                req.audio_tail = cat[consumed:].contiguous()
            else:
                req.audio_tail = cat.new_empty(0)
        nvtx_pop()

    def _append_features(
        self,
        request: Request,
        new_frames: torch.Tensor,
        feat_dim: int,
    ) -> None:
        """Append ``new_frames`` to ``request.feature_buffer``.

        The buffer grows amortised-doubled so we never pay an O(T) copy per
        chunk at steady state.  Consumed prefix (before ``feature_cursor``)
        is compacted opportunistically so long utterances don't keep
        re-allocating.
        """
        n_new = new_frames.size(0)
        buf = request.feature_buffer
        have = request.feature_frames

        # Compact the buffer if the consumed prefix is a large share of it
        # (cheap amortised, avoids unbounded growth on long streams).
        if buf is not None and request.feature_cursor > 0 \
                and request.feature_cursor >= have // 2:
            keep = buf[request.feature_cursor: have].contiguous()
            request.feature_buffer = keep
            buf = request.feature_buffer
            request.feature_frames = keep.size(0)
            have = request.feature_frames
            request.feature_cursor = 0

        needed = have + n_new
        if buf is None or needed > buf.size(0):
            cap = max(needed, (buf.size(0) * 2) if buf is not None else max(needed, 128))
            new_buf = new_frames.new_zeros(cap, feat_dim)
            if buf is not None and have > 0:
                new_buf[:have] = buf[:have]
            request.feature_buffer = new_buf
            buf = new_buf

        buf[have: have + n_new] = new_frames
        request.feature_frames = have + n_new
