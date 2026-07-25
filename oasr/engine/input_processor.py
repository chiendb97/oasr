# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Audio loading and feature extraction for the ASR engine."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.features import FeatureConfig, build_extractor
from oasr.features.backends import _extract as _extract_single
from oasr.features.batched import supports_batched_fbank, supports_batched_mfcc
from oasr.features.lfr import apply_lfr_batch, lfr_output_length
from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .config import EngineConfig
from .graph_cache import GraphedFeatureExtraction
from .request import Request


class InputProcessor:
    """Converts raw **waveforms** into model-ready features.

    The engine is waveform-only: ``audio`` is a 1-D float32 tensor (or numpy
    array) at the model sample rate — file decoding happens at the entry point
    (the serving front-end via ``oasr-asr``, or the bench/test harness), never
    here.  Two paths share this class:

    * **offline** — :meth:`prepare_offline` then :meth:`collate` (one
      batched GPU fbank over a length-bucketed micro-batch);
    * **streaming** — :meth:`prepare_streaming`, :meth:`append_streaming_chunk`,
      :meth:`extract_streaming_batch` (per-step batched fbank across streams).

    CMVN is **not** applied here — it is baked into the model as a
    ``GlobalCMVN`` layer inside ``ConformerEncoder``.

    Parameters
    ----------
    config : EngineConfig
        Engine configuration (used for chunking params and feature config).
    device : torch.device
        Target device for output tensors.
    """

    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        *,
        graph_pool: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._config = config
        self._device = device
        self._feature_config: FeatureConfig = config.feature_config  # type: ignore[assignment]
        # Resolve the frontend once, through the feature registry, so the batch
        # path stays architecture-agnostic (F1).  Raises here — at engine
        # construction — for an unregistered ``feature_type`` rather than on the
        # first request.
        self._extractor = build_extractor(self._feature_config)

        # Offline collate staging buffers, reused across micro-batches so the
        # slow allocations (``cudaHostAlloc`` for pinned host, device malloc)
        # are paid once and grown geometrically.
        #   ``_wav_flat``   — pinned **host** 1-D buffer holding the batch's
        #                     waveforms packed end-to-end (no padding).  The
        #                     only CPU-side copy in collate; a single async
        #                     H2D ships it to the device.
        #   ``_wav_padded`` — **device** 1-D buffer, viewed as ``(B, T_max)``,
        #                     into which the packed waveforms are scattered.
        #                     Zero-padding and the audio-scale multiply both
        #                     run here on the GPU (see :meth:`collate`).
        self._wav_flat: Optional[torch.Tensor] = None
        self._wav_padded: Optional[torch.Tensor] = None

        # Shared CUDA Graph memory-pool handle injected by ``ASREngine`` so the
        # feature-extraction graph cache (added in Step 2 of the plan) shares
        # one pool with the encoder/CTC captures. ``None`` when the engine has
        # CUDA graphs disabled or is running on CPU; the eager feature path is
        # used in that case.
        self._graph_pool = graph_pool

        # CUDA-Graph cache for the steady-state streaming feature path. Lazy
        # captures keyed by ``B_active`` bucket. ``None`` disables capture
        # (CPU device, master ``use_cuda_graphs`` off, sub-toggle off, or the
        # feature config doesn't satisfy the batched fbank/mfcc backend).
        # The pool is per-cache to avoid intermediate-allocation aliasing with
        # the encoder graph cache; ``graph_pool`` from the caller is honoured
        # when non-None but the engine deliberately passes ``None`` so the
        # cache allocates its own private pool.
        self._feature_graph: Optional[GraphedFeatureExtraction] = None
        if (
            device.type == "cuda"
            and bool(getattr(config, "use_cuda_graphs", True))
            and bool(getattr(config, "use_feature_cuda_graphs", True))
            and (
                supports_batched_fbank(self._feature_config)
                or supports_batched_mfcc(self._feature_config)
            )
        ):
            self._feature_graph = GraphedFeatureExtraction(
                pool=graph_pool,
                device=device,
                feature_config=self._feature_config,
                output_dtype=config.dtype,
                chunk_samples=self.streaming_audio_chunk_samples,
                max_batch_size=int(config.max_batch_size),
                batch_buckets=getattr(config, "feature_graph_batch_buckets", None),
            )

    # ------------------------------------------------------------------
    # Batched offline processing
    # ------------------------------------------------------------------

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
            est = (num_samples - frame_length) // frame_shift + 1
        elif num_samples <= 0:
            return 0
        else:
            est = (num_samples + frame_shift // 2) // frame_shift
        if cfg.lfr_enabled:
            est = lfr_output_length(est, cfg.lfr_n)
        return est

    def check_audio_duration(self, audio) -> None:
        """Public duration guard, callable before the waveform is canonicalised.

        The engine calls this on the *admitting* thread so an over-long request
        raises back to the caller even when ``overlap_admit`` defers
        :meth:`prepare_offline` to the prep thread (where a raise would only be
        logged, leaving the client waiting for an output that never comes).
        No-op for frontends without a fixed window.
        """
        if audio is None or self._feature_config.fixed_window_seconds is None:
            return
        self._check_input_duration(int(torch.as_tensor(audio).numel()))

    def _check_input_duration(self, num_samples: int) -> None:
        """Reject audio longer than a fixed-window frontend can represent.

        Frontends with a :attr:`~oasr.features.FeatureConfig.fixed_window_seconds`
        (``whisper_logmel``: the 30 s Whisper window, shared by Qwen2-Audio) pad
        *and trim* every utterance to that window, so longer audio would be
        silently dropped and the caller would receive a plausible transcript of
        the first N seconds only.  Fail loudly at admission instead — long-form
        decoding needs windowed inference, which is a separate feature.
        """
        window_s = self._feature_config.fixed_window_seconds
        if window_s is None:
            return
        limit = int(self._feature_config.sample_rate * window_s)
        if num_samples > limit:
            got_s = num_samples / float(self._feature_config.sample_rate)
            raise ValueError(
                f"audio is {got_s:.1f}s but this checkpoint's frontend "
                f"({self._feature_config.feature_type}) is fixed to a "
                f"{window_s:.0f}s window; longer audio would be silently "
                "truncated. Segment the audio before submitting it."
            )

    def prepare_offline(self, request: Request) -> None:
        """Register an offline request without running feature extraction.

        Canonicalises ``request.audio`` **in place** (→ 1-D float32 CPU
        waveform) and stamps a cheap sample-count based ``num_frames`` estimate
        so the scheduler can bucket by length without a D2H sync.  No audio
        scaling happens here — the int16-scale multiply runs on the GPU after
        padding in :meth:`collate`, which also runs the batched fbank/mfcc.

        Raises ``ValueError`` when the audio exceeds a fixed-window frontend's
        capacity (see :meth:`_check_input_duration`).
        """
        request.audio = torch.as_tensor(request.audio, dtype=torch.float32, device="cpu").reshape(
            -1
        )
        self._check_input_duration(int(request.audio.numel()))
        request.num_frames = self._estimate_num_frames(int(request.audio.numel()))
        # Clear any stale feature cache from reused Request objects.
        request.features = None
        request.feature_lengths = None

    def _flat_host(self, n: int) -> torch.Tensor:
        """Reused pinned **host** 1-D buffer of length ``n`` (geometric growth).

        Pinned so the subsequent ``.to(cuda, non_blocking=True)`` is a true
        async H2D.  Reuse is safe on the synchronous offline path: the buffer
        is fully transferred (and the batch D→H-synced at decode) before the
        next collate overwrites it — :class:`OfflineExecutor` runs micro-batches
        back-to-back on the default stream with no producer-thread overlap.
        """
        cur = 0 if self._wav_flat is None else self._wav_flat.numel()
        if cur < n:
            self._wav_flat = torch.empty(max(n, cur * 2), dtype=torch.float32, pin_memory=True)
        return self._wav_flat[:n]

    def _padded_device(self, batch: int, t_max: int) -> torch.Tensor:
        """Reused **device** buffer viewed as ``(batch, t_max)`` (geometric
        growth).  Holds the zero-padded, scaled waveform batch: the caller
        zeroes it, scatters the packed waveforms in, then scales — all on the
        GPU.  Reuse is safe for the same reason as :meth:`_flat_host`."""
        need = batch * t_max
        cur = 0 if self._wav_padded is None else self._wav_padded.numel()
        if cur < need:
            self._wav_padded = torch.empty(
                max(need, cur * 2), dtype=torch.float32, device=self._device
            )
        return self._wav_padded[:need].view(batch, t_max)

    def collate(
        self,
        requests: List[Request],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched feature extraction for one offline micro-batch.

        Builds a padded + scaled ``(B, T_max)`` device batch from the requests'
        waveforms (:meth:`_padded_waveform_batch`), runs fbank/mfcc over it in
        one shot (:meth:`_fbank_batch`), then releases the host waveforms so the
        GPU feature tensor owns the batch.  ``prepare_offline`` has already
        canonicalised each ``request.audio`` to a 1-D float32 CPU waveform.

        Returns
        -------
        features : Tensor
            ``(B, max_feat_frames, F)`` padded features in the engine dtype,
            on ``device``.
        feat_lengths : Tensor
            ``(B,)`` valid frame counts on ``device``.
        """
        assert requests, "cannot collate empty batch"

        waveforms = [r.audio for r in requests]
        wav_lengths = torch.tensor([w.size(0) for w in waveforms], dtype=torch.int64)

        wav_device = self._padded_waveform_batch(waveforms)
        features, feat_lengths = self._fbank_batch(wav_device, wav_lengths)

        # Release the host waveforms; the GPU feature tensor owns the batch now.
        for r in requests:
            r.audio = None
        return features, feat_lengths

    def _padded_waveform_batch(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """Pad 1-D waveforms into one ``(B, T_max)`` device tensor, zero-padded
        and ``audio_scale``-scaled — with the GPU doing the heavy lifting.

        On CUDA the CPU does a single copy (packing the waveforms end-to-end
        into a reused **pinned** host buffer); one async H2D ships them over,
        then the GPU zeroes a reused ``(B, T_max)`` buffer (the padding),
        scatters each waveform into its row, and multiplies by ``audio_scale``
        *after* padding.  This keeps the per-batch ``cudaHostAlloc`` and the
        ``zero(B*T_max)`` off the CPU.
        """
        scale = self._config.audio_scale
        wav_sizes = [w.size(0) for w in waveforms]
        batch, t_max = len(waveforms), max(wav_sizes)

        if self._device.type != "cuda":
            padded = torch.zeros(batch, t_max, dtype=torch.float32)
            for i, (w, n) in enumerate(zip(waveforms, wav_sizes)):
                if n:
                    padded[i, :n] = w
            return padded.mul_(scale) if scale != 1.0 else padded

        flat = self._flat_host(sum(wav_sizes))  # pinned; sole CPU-side copy
        off = 0
        for w, n in zip(waveforms, wav_sizes):
            if n:
                flat[off : off + n] = w
            off += n
        flat_device = flat.to(self._device, non_blocking=True)  # one async H2D

        padded = self._padded_device(batch, t_max)
        padded.zero_()  # GPU zero-padding
        off = 0
        for i, n in enumerate(wav_sizes):
            if n:
                padded[i, :n] = flat_device[off : off + n]  # device-to-device
            off += n
        if scale != 1.0:
            padded.mul_(scale)  # GPU scale, after padding
        return padded

    def _fbank_batch(
        self, wav_device: torch.Tensor, wav_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the checkpoint's frontend over a padded ``(B, T_max)`` waveform batch.

        Which frontend runs is resolved once at construction through the feature
        registry (:func:`oasr.features.build_extractor`), so this stays
        architecture-agnostic: a new frontend registers an
        :class:`~oasr.features.ExtractorSpec` and needs no engine edit.  The Kaldi
        extractor internally picks the fused kernel or a per-utterance fallback
        depending on how exotic the config is.

        LFR stacking is a post-transform over any extractor's output, applied here
        rather than inside an extractor (offline only — ``prepare_streaming``
        rejects LFR configs, which cannot be windowed across chunks).
        """
        fcfg = self._feature_config
        lengths_device = wav_lengths.to(self._device, non_blocking=True)
        features_f32, feat_lengths = self._extractor(wav_device, lengths_device, fcfg)
        if fcfg.lfr_enabled:
            features_f32, feat_lengths = apply_lfr_batch(
                features_f32, feat_lengths, fcfg.lfr_m, fcfg.lfr_n
            )
        return features_f32.to(dtype=self._config.dtype), feat_lengths

    def prepare_streaming(self, request: Request) -> None:
        """Register an empty streaming request — chunks arrive via
        :meth:`append_streaming_chunk`.

        Sets up the streaming state with an empty audio queue.  No fbank
        runs and no waveform load happens here; the engine starts processing
        as soon as the first chunk lands.
        """
        if not self._extractor.supports_streaming:
            raise NotImplementedError(
                f"the {self._extractor.kind!r} frontend cannot run incrementally "
                "(it normalises over a fixed window, so it needs the whole "
                "utterance); this checkpoint is offline-only. Use "
                "service_mode='offline'."
            )
        if self._feature_config.lfr_enabled:
            raise NotImplementedError(
                "LFR feature stacking is offline-only; the streaming feature "
                "path does not window LFR frames across chunks"
            )
        request.audio_chunks = deque()
        request.audio_tail = torch.empty(0, dtype=torch.float32)
        request.audio_final = False
        request.num_frames = 0
        request.feature_buffer = None
        request.feature_frames = 0
        request.feature_cursor = 0

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
            A request previously initialised with :meth:`prepare_streaming`.
        chunk : Tensor or ndarray
            1-D audio samples (CPU or GPU; converted to CPU float32).
        is_last : bool
            ``True`` marks the final chunk — sets ``audio_final`` so the
            engine flushes the trailing partial frame.
        """
        if request.audio_chunks is None:
            raise RuntimeError(
                "append_streaming_chunk called on a request that was not "
                "initialised via prepare_streaming"
            )
        if request.audio_final:
            raise RuntimeError(f"feed_chunk after is_last=True for request {request.request_id}")

        # Normalise to a 1-D float32 CPU waveform (shared with the offline path)
        # and apply the int16 ``audio_scale``.  Streaming keeps its scale on the
        # CPU: it's a tiny per-chunk multiply feeding the chunk+tail concat, not
        # the offline batch pad (the GPU scale-after-pad in ``collate`` is
        # offline-only).
        wav = torch.as_tensor(chunk, dtype=torch.float32, device="cpu").reshape(-1)
        scale = self._config.audio_scale
        if scale != 1.0:
            wav = wav * scale
        wav = wav.contiguous()

        request.audio_chunks.append(wav)
        request.samples_enqueued += wav.numel()
        if is_last:
            # Flush the final word with trailing silence so the last
            # real-audio encoder window is a FULL window rather than a short
            # partial tail.  A partial final window (a) gives the CTC decoder
            # too few frames to emit the last word's tokens — measured to
            # truncate final words (WER 8.54%→6.89% on 100 LJSpeech utts) —
            # and (b) is the only chunk that takes the sub-window encoder path,
            # which the streaming CUDA-graph mis-encodes at B>1 (see
            # ``ModelRunner._forward_single``).  One ``decoding_window`` of
            # silence makes every real-audio window full (graph fast path) and
            # decodes the trailing silence to blanks.  Opt out via
            # ``EngineConfig.finalize_silence_pad = False``.
            if getattr(self._config, "finalize_silence_pad", True):
                pad = self._config.decoding_window * self._feature_config.frame_shift_samples
                request.audio_chunks.append(torch.zeros(pad, dtype=torch.float32))
                request.samples_enqueued += pad
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
        """Run one batched feature-extraction call over all queued streams.

        For each request with a pending audio chunk, this pops the next
        chunk, prepends the previous ``audio_tail``, pads all streams to the
        max combined length, ships one ``(B, T)`` waveform to the device,
        and runs :func:`batched_fbank` / :func:`batched_mfcc` once for the
        whole batch.  The per-stream new frames are concatenated onto
        ``feature_buffer``.

        No stream is allowed to look at samples beyond its own enqueued
        chunk — we only fuse across *different* streams, never across future
        chunks of the same stream.

        Parameters
        ----------
        cuda_stream : torch.cuda.Stream, optional
            When provided (and the engine is on CUDA) the H2D copy and
            the batched feature kernel run on this stream so they can
            overlap with the encoder forward on the default stream.  The
            caller is responsible for synchronising the default stream
            against feature completion before reading ``feature_buffer``.
        """
        if not requests:
            return
        frame_len = self._feature_config.frame_length_samples

        fbank_inputs, fbank_reqs, fbank_flush = self._collect_streaming_inputs(requests, frame_len)
        if not fbank_reqs:
            return

        feats, feat_lens_cpu = self._run_streaming_features(fbank_inputs, fbank_flush, cuda_stream)
        self._distribute_streaming_features(
            fbank_reqs, fbank_inputs, fbank_flush, feats, feat_lens_cpu
        )

    def _collect_streaming_inputs(
        self, requests: List[Request], frame_len: int
    ) -> Tuple[List[torch.Tensor], List[Request], List[bool]]:
        """Pop one pending chunk per stream, prepend its ``audio_tail``, and
        return the per-stream combined waveforms ready for fbank.

        Returns ``(fbank_inputs, fbank_reqs, fbank_flush)`` aligned by index.
        A stream whose combined buffer is still shorter than one frame (and is
        not a final flush) keeps its tail and is skipped this step.  No stream
        ever looks past its own enqueued audio — we fuse across *different*
        streams, never across future chunks of the same stream.
        """
        fbank_inputs: List[torch.Tensor] = []
        fbank_reqs: List[Request] = []
        fbank_flush: List[bool] = []
        for req in requests:
            if req.audio_chunks is None or req.audio_tail is None:
                continue
            if req.audio_chunks:
                chunk = req.audio_chunks.popleft()
                cat = chunk if req.audio_tail.numel() == 0 else torch.cat([req.audio_tail, chunk])
                flush = req.audio_final and not req.audio_chunks and cat.numel() >= frame_len
                # On the very last chunk pad the tail so the final partial
                # frame still gets emitted.
                if req.audio_final and not req.audio_chunks and cat.numel() < frame_len:
                    cat = torch.cat([cat, cat.new_zeros(frame_len - cat.numel())])
                    flush = True
            elif req.audio_final and req.audio_tail.numel() > 0:
                # No chunks left but the tail still carries unconsumed samples
                # (whole waveform was < chunk_samples).
                cat = req.audio_tail
                if cat.numel() < frame_len:
                    cat = torch.cat([cat, cat.new_zeros(frame_len - cat.numel())])
                flush = True
            else:
                continue
            # Too-short non-final buffers wait for more audio next step.
            if cat.numel() < frame_len and not flush:
                req.audio_tail = cat
                continue
            fbank_inputs.append(cat)
            fbank_reqs.append(req)
            fbank_flush.append(flush)
        return fbank_inputs, fbank_reqs, fbank_flush

    def _run_streaming_features(
        self,
        fbank_inputs: List[torch.Tensor],
        fbank_flush: List[bool],
        cuda_stream: Optional["torch.cuda.Stream"],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Pad the combined waveforms to ``(B, T_max)`` and run fbank/mfcc.

        Returns ``(feats, feat_lens_cpu)`` — ``feats`` a device tensor, and the
        host-side per-stream frame counts (Kaldi snip_edges formula, so the
        fbank-output length tensor is never D→H synced).  Prefers the captured
        feature CUDA-graph in steady state, the eager batched kernel otherwise,
        and a per-utterance CPU extraction for non-standard configs.
        """
        fcfg = self._feature_config
        frame_shift = fcfg.frame_shift_samples
        frame_len = fcfg.frame_length_samples
        feat_dim = fcfg.output_dim
        dtype = self._config.dtype
        device = self._device

        nvtx_push("pad+pin")
        sample_counts = [w.numel() for w in fbank_inputs]
        t_max = max(sample_counts)
        feat_lens_cpu: List[int] = [
            ((n - frame_len) // frame_shift + 1) if n >= frame_len else 0 for n in sample_counts
        ]
        lengths_cpu = torch.tensor(sample_counts, dtype=torch.int64)
        # Zero only the *pad tail* of each row, not the whole buffer: the
        # [0:n] region is immediately overwritten by the waveform copy, so the
        # previous full ``torch.zeros`` spent ~B*T_max of CPU fill per step on
        # bytes it then clobbered.  In steady state every row is exactly T_max
        # long (all streams fed equal chunks) so no tail zeroing runs at all;
        # only ragged / flush steps touch ``zero_``.
        padded_cpu = torch.empty(len(fbank_inputs), t_max, dtype=torch.float32)
        for i, w in enumerate(fbank_inputs):
            n = w.numel()
            padded_cpu[i, :n] = w
            if n < t_max:
                padded_cpu[i, n:].zero_()
        if device.type == "cuda":
            padded_cpu = padded_cpu.pin_memory()
            lengths_cpu = lengths_cpu.pin_memory()
        nvtx_pop()

        use_batched = device.type == "cuda" and (
            supports_batched_fbank(fcfg) or supports_batched_mfcc(fcfg)
        )
        if not use_batched:
            # Per-utterance CPU/torchaudio fallback (non-Povey configs, CPU).
            feat_list = [_extract_single(w, fcfg) for w in fbank_inputs]
            feat_lens_cpu = [f.size(0) for f in feat_list]
            max_nf = max(feat_lens_cpu) if feat_lens_cpu else 0
            feats_cpu = torch.zeros(len(feat_list), max_nf, feat_dim, dtype=torch.float32)
            for i, f in enumerate(feat_list):
                feats_cpu[i, : f.size(0)] = f
            feats = feats_cpu.to(device=device, dtype=dtype, non_blocking=True)
            return feats, feat_lens_cpu

        # A dedicated feature stream (when provided) overlaps the H2D + kernel
        # with the encoder forward on the default stream; the caller inserts the
        # event-wait before reading ``feature_buffer``.
        stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()

        # Captured-graph fast path: steady state only (no flush) and within the
        # pre-built B bucket + ``t_pad``.  Any miss falls through to eager.
        fg = self._feature_graph
        if fg is not None and not any(fbank_flush) and padded_cpu.size(1) <= fg.t_pad:
            with stream_ctx:
                nvtx_push("feature_graph_replay")
                feats_view = fg.replay(len(fbank_inputs), padded_cpu, lengths_cpu)
                nvtx_pop()
            if feats_view is not None:
                return feats_view[: len(fbank_inputs)], feat_lens_cpu

        with stream_ctx:
            nvtx_push("h2d")
            wav_device = padded_cpu.to(device=device, non_blocking=True)
            lengths_device = lengths_cpu.to(device=device, non_blocking=True)
            nvtx_pop()
            nvtx_push("feature")
            feats_f32, _ = self._extractor(wav_device, lengths_device, fcfg)
            feats = feats_f32.to(dtype=dtype)
            nvtx_pop()
        return feats, feat_lens_cpu

    def _distribute_streaming_features(
        self,
        fbank_reqs: List[Request],
        fbank_inputs: List[torch.Tensor],
        fbank_flush: List[bool],
        feats: torch.Tensor,
        feat_lens_cpu: List[int],
    ) -> None:
        """Append each stream's new feature frames to its ring buffer and reset
        its ``audio_tail`` to the samples beyond the last consumed frame."""
        frame_shift = self._feature_config.frame_shift_samples
        feat_dim = self._feature_config.output_dim
        nvtx_push("distribute")
        for i, req in enumerate(fbank_reqs):
            new_nf = int(feat_lens_cpu[i])
            if new_nf > 0:
                self._append_features(req, feats[i, :new_nf, :], feat_dim)
            consumed = new_nf * frame_shift
            cat = fbank_inputs[i]
            if fbank_flush[i] or consumed >= cat.numel():
                req.audio_tail = cat.new_empty(0)
            else:
                req.audio_tail = cat[consumed:].contiguous()
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
        if buf is not None and request.feature_cursor > 0 and request.feature_cursor >= have // 2:
            keep = buf[request.feature_cursor : have].contiguous()
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

        buf[have : have + n_new] = new_frames
        request.feature_frames = have + n_new
