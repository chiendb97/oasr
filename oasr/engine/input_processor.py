# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Audio loading and feature extraction for the ASR engine."""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from oasr.features import FeatureConfig, StreamingFraming, build_extractor
from oasr.features.batched import supports_batched_fbank, supports_batched_mfcc
from oasr.features.lfr import apply_lfr_batch, lfr_output_length
from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .config import EngineConfig
from .graph_cache import GraphedFeatureExtraction
from .request import Request

logger = logging.getLogger(__name__)

#: Streaming staging slots rotated through by :meth:`InputProcessor._next_stream_slot`.
#: Two is the pipeline depth the feature stream introduces — one pair in flight
#: while the next is filled.
_STREAM_STAGING_SLOTS = 2


@dataclass
class _StreamStagingSlot:
    """One streaming staging buffer pair, plus the event that retires it.

    The pair is *pinned host* memory read by an async H2D, so it may not be
    rewritten until that copy has actually run.  ``ready`` records the copy's
    completion on the stream it was issued to; a slot is only safe to refill
    once that event has fired.  See
    :meth:`InputProcessor._next_stream_slot` for why both the rotation and the
    event are load-bearing.
    """

    flat: Optional[torch.Tensor] = None
    lens: Optional[torch.Tensor] = None
    ready: Optional["torch.cuda.Event"] = None


@dataclass
class _StreamInput:
    """One stream's combined waveform for this step, kept as its pieces.

    ``segments`` concatenated is the buffer the frontend sees: the carry-over
    ``audio_tail``, this step's chunk, and on the final chunk a short zero pad.
    They stay separate until :meth:`InputProcessor._run_streaming_features`
    packs every stream's pieces into the staging batch with one ``cat`` — a
    per-stream concatenation here would be a full copy of the chunk, for a
    buffer that is copied again a moment later.

    ``n_samples`` is their total, carried rather than re-summed because it also
    picks the frame count off the declared grid.
    """

    request: Request
    segments: List[torch.Tensor]
    n_samples: int
    flush: bool


#: Appends a compacted feature buffer should absorb before compacting again.
#: Compaction reallocates and copies, so its cost is amortised over this many
#: ticks; 16 takes reallocation from ~91 % of appends to under 7 %.
_FEATURE_HEADROOM_APPENDS = 16

#: Absolute ceiling on that headroom, in frames.  Without it a fixed-window
#: frontend (``whisper_logmel``, a 3000-frame ``n_new``) would reserve tens of
#: thousands of frames per stream for headroom it can never use.
_FEATURE_HEADROOM_MAX = 1024


def _suffix(segments: List[torch.Tensor], start: int) -> torch.Tensor:
    """``torch.cat(segments)[start:]``, without materialising the concatenation.

    In steady state ``start`` lands inside the last segment — the chunk just
    consumed — so this is a view and costs nothing, which is the whole point of
    keeping the pieces apart.  It only concatenates when the retained tail spans
    a segment boundary, i.e. when a step consumed less than the carry-over it
    started with.
    """
    for i, seg in enumerate(segments):
        n = seg.numel()
        if start < n:
            head = seg[start:] if start else seg
            rest = segments[i + 1 :]
            return head if not rest else torch.cat([head, *rest])
        start -= n
    return segments[-1].new_empty(0)


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
        # path stays architecture-agnostic.  Raises here — at engine
        # construction — for an unregistered ``feature_type`` rather than on the
        # first request.
        self._extractor = build_extractor(self._feature_config)
        # Resolved on first streaming use, not here: a frontend's framing carries
        # preconditions (Kaldi's ``snip_edges``) that an offline-only engine has no
        # business paying.
        self._streaming_framing: Optional[StreamingFraming] = None

        # Geometrically grown offline staging: packed pinned host samples and a
        # padded device buffer. This keeps collate to one host copy and one H2D.
        self._wav_flat: Optional[torch.Tensor] = None
        self._wav_padded: Optional[torch.Tensor] = None
        # Streaming staging is event-retired because a separate stream reads it
        # asynchronously while later engine steps prepare new input.
        self._stream_slots: List[_StreamStagingSlot] = [
            _StreamStagingSlot() for _ in range(_STREAM_STAGING_SLOTS)
        ]
        self._stream_slot_idx = 0
        # Read-only zero run the ragged rows of a streaming step pad with, so a
        # short row is one more ``cat`` source rather than a separate zero fill.
        self._stream_pad: Optional[torch.Tensor] = None
        # Oversized batches allocate per call rather than retaining excess memory.
        self._max_staging_elems = int(getattr(config, "max_staging_elems", None) or (256 << 20))
        # Largest request :meth:`new_audio_buffer` will page-lock, in samples;
        # ``0`` declines everything (CPU engine, or the knob turned off).
        self._max_pinned_audio_samples = (
            int(
                float(getattr(config, "max_pinned_audio_seconds", 0.0) or 0.0)
                * self._feature_config.sample_rate
            )
            if device.type == "cuda"
            else 0
        )

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

    def check_sample_rate(self, sample_rate: Optional[int]) -> None:
        """Reject a request whose audio is not at the model's sample rate.

        The engine is **waveform-only at the model's rate**: every frame count
        here comes from :attr:`FeatureConfig.sample_rate`, and the mel filterbank
        is built for it.  ``Request.sample_rate`` is carried through the whole
        stack but feeds nothing except long-form window arithmetic, so audio at
        another rate is interpreted as if it were at the model's — 8 kHz
        telephony plays back at double speed to the frontend, 44.1 kHz media at
        a third — and the client gets a confident, wrong transcript with no
        error anywhere.

        Resampling belongs at the entry point, not here: the serving front-end
        (``oasr-asr``) converts before the waveform crosses PyO3, so nothing
        reaching this check should ever mismatch.  Callers driving the engine
        directly (benchmarks, tests, notebooks) get a loud failure instead of a
        plausible transcript.

        ``None`` is accepted as "unspecified" and treated as the model's rate.
        """
        if sample_rate is None:
            return
        want = int(self._feature_config.sample_rate)
        got = int(sample_rate)
        if got != want:
            raise ValueError(
                f"audio is declared at {got} Hz but this checkpoint's frontend "
                f"({self._feature_config.feature_type}) requires {want} Hz; the "
                "engine does not resample. Resample the waveform before "
                "submitting it (the oasr-server front-end does this for you)."
            )

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

        Raises ``ValueError`` when the audio is not at the model's sample rate
        (see :meth:`check_sample_rate`) or exceeds a fixed-window frontend's
        capacity (see :meth:`_check_input_duration`).
        """
        self.check_sample_rate(request.sample_rate)
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

        Beyond :attr:`_max_staging_elems` the buffer is **not** retained: pinned
        memory is a process-global scarce resource and geometric growth sized by
        the longest utterance ever seen never shrinks, so one outlier request
        would hold its peak for the process lifetime.  Past the cap we allocate
        per call and let it go.
        """
        # Pinning is a CUDA operation and *fails* without a usable device, so
        # it follows the engine's device rather than being unconditional — a
        # CPU engine gains nothing from page-locked host memory anyway.
        pin = self._device.type == "cuda"
        if n > self._max_staging_elems:
            return torch.empty(n, dtype=torch.float32, pin_memory=pin)
        cur = 0 if self._wav_flat is None else self._wav_flat.numel()
        if cur < n:
            self._wav_flat = torch.empty(
                min(max(n, cur * 2), self._max_staging_elems),
                dtype=torch.float32,
                pin_memory=pin,
            )
        return self._wav_flat[:n]

    def new_audio_buffer(self, num_samples: int) -> Optional[torch.Tensor]:
        """A **page-locked** 1-D float32 host buffer the caller fills, or ``None``.

        The point of the offer: whoever produces the waveform — the Rust
        front-end, after the codec — has to copy it somewhere anyway, and if
        that somewhere is pinned then :meth:`collate` can DMA each row straight
        into the padded device batch instead of packing the micro-batch into
        staging first.  One copy of the audio after decode instead of two.  The
        buffer is uninitialised; the caller must fill all ``num_samples``
        elements before handing it over as ``Request.audio``.

        Hand back the **tensor**, not a view of it (a ``numpy()`` view is fine
        to *write* through): the async H2D only stays safe because PyTorch
        records an event against the caching host allocator's block when the
        copy is issued, and it finds that block through the storage context of
        the tensor it allocated.  A ``from_numpy`` re-wrap of the same pages is
        pinned but anonymous to that bookkeeping, so nothing would stop the
        block being recycled under an in-flight DMA — one request's audio
        arriving as another's transcript.  Reasoned from the allocator's
        contract rather than observed: a probe could not force the reuse
        ordering, which is the argument for shipping the shape that cannot
        depend on it.

        Returns ``None`` — meaning "allocate ordinary memory yourself" — for a
        CPU engine, a non-positive size, or a request longer than
        ``EngineConfig.max_pinned_audio_seconds``.  Declining is not an error:
        page-locked memory is process-global and the caching host allocator
        keeps what it takes, so an unbounded offer would let one long request
        hold a permanent reservation.
        """
        if num_samples <= 0 or num_samples > self._max_pinned_audio_samples:
            return None
        try:
            return torch.empty(num_samples, dtype=torch.float32, pin_memory=True)
        except RuntimeError:
            # No usable CUDA context to page-lock against.  The caller's
            # fallback is a correct, slightly slower path, so this is a
            # capability answer rather than a failure — but it should never be
            # reached on a device the engine reported as CUDA, hence the log.
            logger.warning("cannot allocate pinned host memory; audio buffers stay on the heap")
            self._max_pinned_audio_samples = 0
            return None

    def release_staging(self) -> None:
        """Drop the reusable staging buffers (called on engine teardown / idle).

        Without this the pinned host buffer and its device twin survive as long
        as the ``InputProcessor`` does, which for a long-lived server is the
        process.
        """
        self._wav_flat = None
        self._wav_padded = None
        # Wait out any copy still reading a slot before dropping the pinned pages
        # under it — freeing host memory a queued DMA reads is worse than the
        # race this fixes.
        for slot in self._stream_slots:
            if slot.ready is not None:
                slot.ready.synchronize()
            slot.flat = None
            slot.lens = None
            slot.ready = None

    def _padded_device(self, batch: int, t_max: int) -> torch.Tensor:
        """Reused **device** buffer viewed as ``(batch, t_max)`` (geometric
        growth).  Holds the zero-padded, scaled waveform batch: the caller
        zeroes it, scatters the packed waveforms in, then scales — all on the
        GPU.  Reuse is safe for the same reason as :meth:`_flat_host`."""
        need = batch * t_max
        if need > self._max_staging_elems:
            return torch.empty(need, dtype=torch.float32, device=self._device).view(batch, t_max)
        cur = 0 if self._wav_padded is None else self._wav_padded.numel()
        if cur < need:
            self._wav_padded = torch.empty(
                min(max(need, cur * 2), self._max_staging_elems),
                dtype=torch.float32,
                device=self._device,
            )
        return self._wav_padded[:need].view(batch, t_max)

    def _next_stream_slot(self) -> _StreamStagingSlot:
        """Rotate to the next staging slot and block until it is safe to rewrite.

        The streaming H2D is async out of *pinned* host memory, so the DMA reads
        this buffer after the launch returns.  ``wait_stream`` cannot help — it
        orders GPU work, not the host's next refill.  The **event** makes reuse
        correct; the **rotation** makes it free, since a slot comes round a full
        step later and ``synchronize()`` finds the event already fired.  Depth 2,
        not 1: a step can produce features without issuing a forward, so two can
        run back-to-back with nothing draining the queue.

        Not the race that emptied streaming transcripts — that was the
        device-side hand-off in :meth:`extract_streaming_batch`; double-buffering
        here was measured to change its WER by nothing.  Pinned by
        ``TestStagingBuffers::test_consecutive_streaming_steps_get_different_buffers``.
        """
        self._stream_slot_idx = (self._stream_slot_idx + 1) % len(self._stream_slots)
        slot = self._stream_slots[self._stream_slot_idx]
        if slot.ready is not None:
            slot.ready.synchronize()
            slot.ready = None
        return slot

    def _stream_host(self, slot: _StreamStagingSlot, batch: int, t_max: int) -> torch.Tensor:
        """This slot's pinned ``(batch, t_max)`` host buffer (geometric growth).

        Same grow-once discipline as :meth:`_flat_host`; safety comes from the
        caller having taken the slot through :meth:`_next_stream_slot`.
        """
        need = batch * t_max
        cur = 0 if slot.flat is None else slot.flat.numel()
        if cur < need:
            slot.flat = torch.empty(
                max(need, cur * 2),
                dtype=torch.float32,
                pin_memory=(self._device.type == "cuda"),
            )
        assert slot.flat is not None
        return slot.flat[:need].view(batch, t_max)

    def _stream_lengths_host(self, slot: _StreamStagingSlot, batch: int) -> torch.Tensor:
        """This slot's pinned ``(batch,)`` int64 host buffer for stream lengths."""
        cur = 0 if slot.lens is None else slot.lens.numel()
        if cur < batch:
            slot.lens = torch.empty(
                max(batch, cur * 2),
                dtype=torch.int64,
                pin_memory=(self._device.type == "cuda"),
            )
        assert slot.lens is not None
        return slot.lens[:batch]

    def _retire_stream_slot(self, slot: _StreamStagingSlot) -> None:
        """Record, on the issuing stream, the event that releases *slot*.

        Called once per step **after** every device op that reads the slot's
        buffers has been enqueued, and from inside the feature-stream context so
        the event lands on the stream carrying the copy.  A CPU engine records
        nothing: there is no async copy and the buffers are not pinned.
        """
        if self._device.type != "cuda":
            return
        event = torch.cuda.Event()
        event.record()
        slot.ready = event

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
        # Pinned so :meth:`_fbank_batch`'s ``non_blocking=True`` is a real async
        # DMA.  A *pageable* source makes CUDA synchronise the stream before
        # staging the bytes, which here means waiting out the batch's 26 MB
        # waveform H2D and every scatter behind it — a drain in the middle of
        # the collate, for 8 bytes per row.
        wav_lengths = torch.tensor(
            [w.size(0) for w in waveforms],
            dtype=torch.int64,
            pin_memory=(self._device.type == "cuda"),
        )

        wav_device = self._padded_waveform_batch(waveforms)
        features, feat_lengths = self._fbank_batch(wav_device, wav_lengths)

        # Release the host waveforms; the GPU feature tensor owns the batch now.
        for r in requests:
            r.audio = None
        return features, feat_lengths

    def _padded_waveform_batch(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """Pad 1-D waveforms into one ``(B, T_max)`` device tensor, zero-padded
        and ``audio_scale``-scaled — with the GPU doing the heavy lifting.

        Two paths, chosen by where the waveforms already live:

        * **Already page-locked** (the caller took them from
          :meth:`new_audio_buffer`) — each row is DMA'd straight into its slice
          of the padded device batch.  No CPU copy at all: the pack this class
          used to do was the *second* copy of the audio after the codec, and
          removing it measures **1.12-1.18x** end-to-end offline.
        * **Anything else** — the CPU packs the waveforms end-to-end into a
          reused pinned buffer, one async H2D ships them over, and the GPU
          scatters each into its row.  Copying once into staging beats B
          separate copies out of pageable memory, each of which would
          synchronise the stream before staging itself.

        Both zero the pad region and apply ``audio_scale`` on the GPU, *after*
        padding, so the two agree bit-for-bit.
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

        # All-or-nothing rather than per row: a mixed batch would pay the pack
        # for the unpinned rows *and* B extra launches for the pinned ones, and
        # in practice a process is fed by one producer — the front-end, which
        # pins, or an in-process caller, which does not.  ``is_pinned`` is a
        # driver query (~0.7 us), so this costs ~45 us at B=64 against the ~4 ms
        # of packing it decides about.
        if all(w.is_pinned() for w in waveforms):
            padded = self._padded_device(batch, t_max)
            padded.zero_()  # GPU zero-padding
            for i, (w, n) in enumerate(zip(waveforms, wav_sizes)):
                if n:
                    padded[i, :n].copy_(w, non_blocking=True)  # async H2D, no pack
            if scale != 1.0:
                padded.mul_(scale)
            return padded

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
            # ``features_f32`` is padded to the batch's widest row, so its own
            # T bounds every per-row length — pass it so LFR need not sync to
            # read the max off the device.
            features_f32, feat_lengths = apply_lfr_batch(
                features_f32,
                feat_lengths,
                fcfg.lfr_m,
                fcfg.lfr_n,
                max_length=int(features_f32.size(1)),
            )
        return features_f32.to(dtype=self._config.dtype), feat_lengths

    def prepare_streaming(self, request: Request) -> None:
        """Register an empty streaming request — chunks arrive via
        :meth:`append_streaming_chunk`.

        Sets up the streaming state with an empty audio queue.  No fbank
        runs and no waveform load happens here; the engine starts processing
        as soon as the first chunk lands.

        Rejects a rate mismatch at *open* time (see :meth:`check_sample_rate`) —
        the alternative is discovering it on the first chunk, after the client
        has already been told the stream is live.
        """
        self.check_sample_rate(request.sample_rate)
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
        framing = self.streaming_framing
        request.audio_chunks = deque()
        # The buffer starts with the frontend's implicit left padding rather than
        # empty: a centered STFT's frame 0 begins ``n_fft // 2`` samples *before*
        # the signal.  Signal-domain pre-emphasis may require one additional
        # sample; frontends without implicit history declare ``prefill = 0``.
        request.audio_tail = torch.zeros(framing.prefill, dtype=torch.float32)
        request.audio_final = False
        request.num_frames = 0
        request.feature_buffer = None
        request.feature_frames = 0
        request.feature_cursor = 0
        request.feature_base = 0

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

        # Normalise to a 1-D float32 CPU waveform (shared with the offline path).
        # ``audio_scale`` is **not** applied here: that would be a whole extra
        # pass over the waveform plus an allocation, per chunk per stream, when
        # :meth:`_run_streaming_features` can do the entire step's batch in one
        # ``mul_`` over the packed staging buffer.  The multiply is elementwise,
        # so scaling there is bit-identical to scaling here.  Nothing between
        # the two reads the sample *values* — the scheduler and the streaming
        # backends only ask whether the queue is empty.
        wav = torch.as_tensor(chunk, dtype=torch.float32, device="cpu").reshape(-1).contiguous()

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
                pad_frames = self._finalize_pad_frames(request.samples_enqueued)
                pad = pad_frames * self._feature_config.frame_shift_samples
                request.audio_chunks.append(torch.zeros(pad, dtype=torch.float32))
                request.samples_enqueued += pad
            request.audio_final = True
        # Keep the scheduler's bucket estimate roughly in sync.  O(1) using
        # the running total instead of re-summing the deque per chunk.
        request.num_frames = self._estimate_num_frames(request.samples_enqueued)

    def _finalize_pad_frames(self, samples_enqueued: int) -> int:
        """Feature frames of silence to append when a stream is closed.

        One ``decoding_window``, plus — when the streaming runtime declares an
        alignment (``StreamingEncoderBackend.finalize_align_frames``) — however
        many frames it takes to round the stream up to a whole window first.

        Without the rounding, a runtime that skips its sub-window tail hands the
        decoder ``window - (frames % window)`` frames of flush silence, i.e. one
        frame in the worst case: the trailing subword then never gets emitted, and
        *which* utterances lose it depends on their length.  With it, every stream
        gets at least a full window, whatever its length.
        """
        window = int(self._config.decoding_window)
        align = int(getattr(self._config, "_finalize_align_frames", 0) or 0)
        if align <= 0:
            return window
        framing = self.streaming_framing
        frames = framing.frames_for(framing.prefill + int(samples_enqueued))
        return window + (-frames) % align

    @property
    def streaming_framing(self) -> StreamingFraming:
        """The frontend's declared streaming frame grid (memoised).

        Raises ``NotImplementedError`` for a frontend that declares none, or whose
        framing has a config precondition this deployment violates (Kaldi needs
        ``snip_edges=True``).
        """
        if self._streaming_framing is None:
            self._streaming_framing = self._extractor.framing_for(self._feature_config)
        return self._streaming_framing

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
            overlap with the encoder forward on the default stream.
            **This method orders the hand-off itself**: the append into
            ``feature_buffer`` happens on the current stream, so the wait on
            ``cuda_stream`` (and the ``record_stream`` that keeps the allocator
            from recycling the output) belong here, not in the caller.  A caller
            that also waits before its own read of ``feature_buffer`` is then
            merely redundant, not load-bearing.
        """
        if not requests:
            return
        # Samples one frame reads, from the frontend's declared grid.  For Kaldi
        # this is ``frame_length_samples`` (what the old hardcoded value was); for
        # a centered STFT it is ``n_fft`` plus any pre-emphasis history, and using
        # the window length there would emit a frame before its last samples had
        # arrived.
        min_samples = self.streaming_framing.min_samples

        inputs = self._collect_streaming_inputs(requests, min_samples)
        if not inputs:
            return

        feats, feat_lens_cpu = self._run_streaming_features(inputs, cuda_stream)
        # ``feats`` is produced on ``cuda_stream`` and appended on the current
        # stream, so the cross-stream read is ordered here rather than by the
        # caller: the step loop's `wait_stream` fires after this returns, which is
        # after the racing copy has been issued.  Deleting it silently feeds the
        # buffer unwritten memory — conformer streaming 3.70% -> 99.32% WER, 195
        # of 200 transcripts empty, nothing raised.  Negative control:
        # ``TestStreamingFeatureStreamHandoff``.
        #
        # ``record_stream`` is the second half: without it the allocator can hand
        # ``feats``'s block to a later feature step while the append still reads it.
        if cuda_stream is not None and self._device.type == "cuda":
            consumer = torch.cuda.current_stream(self._device)
            consumer.wait_stream(cuda_stream)
            feats.record_stream(consumer)
        self._distribute_streaming_features(inputs, feats, feat_lens_cpu)

    def _collect_streaming_inputs(
        self, requests: List[Request], frame_len: int
    ) -> List["_StreamInput"]:
        """Pop one pending chunk per stream and describe the combined waveform.

        The combined buffer is returned as its **pieces** — the carried-over
        ``audio_tail``, this step's chunk, and (on the final chunk) a short
        zero pad — not as a concatenation of them.  Concatenating here cost a
        full copy of every stream's chunk per step, for a buffer that is copied
        again into the staging batch a moment later;
        :meth:`_run_streaming_features` now packs the pieces straight into the
        staging row, so the intermediate never exists.

        A stream whose combined buffer is still shorter than one frame (and is
        not a final flush) keeps its tail and is skipped this step — that branch
        does concatenate, because the pieces have to survive until more audio
        arrives, but it runs when a chunk is smaller than a single frame.  No
        stream ever looks past its own enqueued audio: we fuse across
        *different* streams, never across future chunks of the same stream.
        """
        inputs: List[_StreamInput] = []
        for req in requests:
            if req.audio_chunks is None or req.audio_tail is None:
                continue
            tail = req.audio_tail
            if req.audio_chunks:
                chunk = req.audio_chunks.popleft()
                segments = [chunk] if tail.numel() == 0 else [tail, chunk]
                n = tail.numel() + chunk.numel()
                last = req.audio_final and not req.audio_chunks
                flush = last and n >= frame_len
                # On the very last chunk pad the tail so the final partial
                # frame still gets emitted.
                if last and n < frame_len:
                    segments.append(torch.zeros(frame_len - n, dtype=torch.float32))
                    n = frame_len
                    flush = True
            elif req.audio_final and tail.numel() > 0:
                # No chunks left but the tail still carries unconsumed samples
                # (whole waveform was < chunk_samples).
                segments = [tail]
                n = tail.numel()
                if n < frame_len:
                    segments.append(torch.zeros(frame_len - n, dtype=torch.float32))
                    n = frame_len
                flush = True
            else:
                continue
            # Too-short non-final buffers wait for more audio next step.
            if n < frame_len and not flush:
                req.audio_tail = segments[0] if len(segments) == 1 else torch.cat(segments)
                continue
            inputs.append(_StreamInput(request=req, segments=segments, n_samples=n, flush=flush))
        return inputs

    def _run_streaming_features(
        self,
        inputs: List["_StreamInput"],
        cuda_stream: Optional["torch.cuda.Stream"],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Pack the per-stream pieces into ``(B, T_max)`` and run fbank/mfcc.

        Returns ``(feats, feat_lens_cpu)`` — ``feats`` a device tensor, and the
        host-side per-stream frame counts (Kaldi snip_edges formula, so the
        fbank-output length tensor is never D->H synced).  Prefers the captured
        feature CUDA-graph in steady state, the eager batched kernel otherwise,
        and a per-utterance CPU extraction for non-standard configs.
        """
        fcfg = self._feature_config
        framing = self.streaming_framing
        dtype = self._config.dtype
        device = self._device
        batch = len(inputs)

        nvtx_push("pad+pin")
        sample_counts = [inp.n_samples for inp in inputs]
        t_max = max(sample_counts)
        # Host-side, from the declared grid, so the extractor's output-length
        # tensor is never D->H synced.
        feat_lens_cpu: List[int] = [framing.frames_for(n) for n in sample_counts]

        # Reused pinned staging: ``pin_memory()`` is a ``cudaHostAlloc`` + copy,
        # and this ran **twice per streaming step** on the default path (the
        # stable-buffer variant existed only behind ``use_feature_cuda_graphs``,
        # which is off by default).  Page-locking is a kernel-level operation
        # that also serialises against the driver, so at streaming cadence it is
        # a per-step tax for a buffer whose shape barely changes.
        # Double-buffered + event-retired — see :meth:`_next_stream_slot`.  The
        # rotation is what keeps the pinned-staging win while making the reuse
        # ordering explicit instead of accidental.
        slot = self._next_stream_slot()
        padded_cpu = self._pack_streaming_waveforms(inputs, slot, t_max)

        lengths_cpu = self._stream_lengths_host(slot, batch)
        lengths_cpu.copy_(torch.tensor(sample_counts, dtype=torch.int64))
        nvtx_pop()

        if device.type != "cuda":
            # CPU engine: run the frontend on the host buffers so nothing pays an
            # H2D for a device that has none.  Which implementation the extractor
            # picks (fused kernels, batched torch, or its own per-utterance
            # fallback for a config the kernels cannot express) is the extractor's
            # decision — this used to be a hardcoded ``_extract_single`` call,
            # which is Kaldi-only and silently produced Kaldi features for any
            # other registered frontend.
            self._scale_audio_(padded_cpu)
            feats_cpu, _ = self._extractor.extract_streaming(
                padded_cpu[:, :t_max], lengths_cpu, fcfg
            )
            feats = feats_cpu.to(device=device, dtype=dtype, non_blocking=True)
            return feats, feat_lens_cpu

        # A dedicated feature stream (when provided) overlaps the H2D + kernel
        # with the encoder forward on the default stream; the caller inserts the
        # event-wait before reading ``feature_buffer``.  That wait orders the two
        # *streams* — it says nothing about when the host may reuse ``slot``,
        # which is what :meth:`_next_stream_slot` is for.
        stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()

        # Captured-graph fast path: steady state only (no flush) and within the
        # pre-built B bucket + ``t_pad``.  Any miss falls through to eager.
        fg = self._feature_graph
        scaled = False
        if fg is not None and not any(inp.flush for inp in inputs) and t_max <= fg.t_pad:
            # The captured graph owns the H2D, so there is no device copy for the
            # scale to ride on here and it has to happen on the host.  Recorded,
            # because a bucket miss falls through to the eager path below and
            # scaling twice would be silent.
            self._scale_audio_(padded_cpu)
            scaled = True
            with stream_ctx:
                nvtx_push("feature_graph_replay")
                feats_view = fg.replay(batch, padded_cpu, lengths_cpu)
                nvtx_pop()
                # ``replay`` only host-memcpies out of ``slot`` into the graph's
                # own captured buffers, so the slot is already free here — but
                # retire it on the stream anyway rather than encoding that
                # cross-file detail as an unchecked assumption.
                self._retire_stream_slot(slot)
            if feats_view is not None:
                return feats_view[:batch], feat_lens_cpu

        with stream_ctx:
            nvtx_push("h2d")
            wav_device = padded_cpu.to(device=device, non_blocking=True)
            lengths_device = lengths_cpu.to(device=device, non_blocking=True)
            if not scaled:
                # On the device, where it is free and overlapped.  On the host it
                # is a second full pass over the packed batch — 0.65 ms per step
                # at 64 streams, which is more than the pack itself costs and
                # would swallow the whole point of batching it.
                self._scale_audio_(wav_device)
            nvtx_pop()
            # Both H2Ds are enqueued; the event that releases ``slot`` goes in
            # behind them on the same stream.  Everything below reads the device
            # copies, not the host buffers.
            self._retire_stream_slot(slot)
            nvtx_push("feature")
            feats_f32, _ = self._extractor.extract_streaming(wav_device, lengths_device, fcfg)
            feats = feats_f32.to(dtype=dtype)
            nvtx_pop()
        return feats, feat_lens_cpu

    def _pack_streaming_waveforms(
        self, inputs: List["_StreamInput"], slot: _StreamStagingSlot, t_max: int
    ) -> torch.Tensor:
        """Pack every stream's pieces into ``slot``'s pinned ``(B, t_max)`` batch.

        One ``cat`` for the whole step, not one op per stream.  Each row is its
        pieces followed by a zero pad to ``t_max``, so the concatenation in row
        order *is* the padded layout and lands directly in the pinned staging —
        no intermediate per-stream buffer, no per-row copy, no per-row zero fill.
        The pads come from one shared read-only zero run, and in steady state
        there are none: every stream is fed the same chunk size and converges on
        the same carry-over tail.

        What this replaces, measured in-engine at 64 streams x one 640 ms chunk:
        0.44 ms/step concatenating each stream's tail onto its chunk plus 0.46
        ms/step copying the results into staging, against 0.61 ms/step for the
        single ``cat`` — *two* passes over the batch collapsed into one, not
        just fewer dispatches.

        The obvious alternative — DMA each row straight to the device out of
        page-locked chunks, as :meth:`_padded_waveform_batch` does offline —
        measured **worse** here (0.57-1.04 ms for 2B launches).  A streaming row
        is one chunk, ~40 KB, so B launch overheads outweigh the pack they save;
        an offline row is a whole utterance and the trade flips.

        Samples are packed **raw**: keeping ``audio_scale`` off this pass is what
        makes it one pass, and :meth:`_scale_audio_` applies it to whichever copy
        the caller goes on to use.
        """
        padded_cpu = self._stream_host(slot, len(inputs), t_max)
        segments: List[torch.Tensor] = []
        for inp in inputs:
            segments.extend(inp.segments)
            pad = t_max - inp.n_samples
            if pad:
                segments.append(self._pad_zeros(pad))
        torch.cat(segments, out=padded_cpu.view(-1))
        return padded_cpu

    def _scale_audio_(self, waveforms: torch.Tensor) -> None:
        """Apply ``audio_scale`` in place, wherever the batch currently lives.

        The streaming pack writes raw samples so it stays *one* pass over the
        batch; the multiply then rides on the device copy in the common case.
        The two paths that consume the pinned host buffer directly — a CPU
        engine, and the captured feature graph, which owns its own H2D — have no
        device copy to ride on and scale on the host instead.  Elementwise
        either way, and fp32 multiply is correctly rounded on both, so all three
        agree bit for bit.
        """
        scale = self._config.audio_scale
        if scale != 1.0:
            waveforms.mul_(scale)

    def _pad_zeros(self, n: int) -> torch.Tensor:
        """A read-only ``n``-sample zero run, from one grow-only shared buffer.

        Only the row padding of a *ragged* step uses this, and a row's pad is
        never read back: the frame counts come from the declared grid, so
        everything past ``n_samples`` is discarded downstream.  Shared rather
        than allocated because it is never written to.
        """
        cur = 0 if self._stream_pad is None else self._stream_pad.numel()
        if cur < n:
            self._stream_pad = torch.zeros(max(n, cur * 2), dtype=torch.float32)
        assert self._stream_pad is not None
        return self._stream_pad[:n]

    def _distribute_streaming_features(
        self,
        inputs: List["_StreamInput"],
        feats: torch.Tensor,
        feat_lens_cpu: List[int],
    ) -> None:
        """Append each stream's new feature frames to its ring buffer and reset
        its ``audio_tail`` to the samples beyond the last consumed frame.

        The retained tail is the combined buffer's ``[F * hop:]`` — written in
        *buffer* coordinates, which is why it needs no adjustment for a frontend
        whose grid starts before sample 0: frame ``F`` reads from buffer offset
        ``F * hop`` whatever the absolute alignment, so the same rule keeps a
        centered grid's look-back and a pre-emphasis history sample without
        knowing about either.

        Every stream's append is *planned* first and then committed as one
        ``torch._foreach_copy_``.  Issued one at a time this was the single
        largest cost in the streaming step — 146 device-to-device copies per step
        carrying 0.09 ms of work, 21% of the wall clock, because at streaming
        cadence what a copy costs is the host submitting it, not the bytes.  The
        multi-tensor form submits one kernel for the whole ready set; the pairs
        are independent (each writes its own buffer's ``[have, have + n_new)``,
        which no other pair reads), so batching cannot reorder anything.
        """
        hop = self.streaming_framing.hop
        feat_dim = self._feature_config.output_dim
        nvtx_push("distribute")
        dsts: List[torch.Tensor] = []
        srcs: List[torch.Tensor] = []
        for i, inp in enumerate(inputs):
            req = inp.request
            new_nf = int(feat_lens_cpu[i])
            if new_nf > 0:
                self._plan_append_features(req, feats[i, :new_nf, :], feat_dim, dsts, srcs)
            consumed = new_nf * hop
            if inp.flush or consumed >= inp.n_samples:
                req.audio_tail = inp.segments[-1].new_empty(0)
            else:
                req.audio_tail = _suffix(inp.segments, consumed)
        if dsts:
            if len(dsts) == 1:
                dsts[0].copy_(srcs[0])
            else:
                torch._foreach_copy_(dsts, srcs)
        nvtx_pop()

    def _plan_append_features(
        self,
        request: Request,
        new_frames: torch.Tensor,
        feat_dim: int,
        dsts: List[torch.Tensor],
        srcs: List[torch.Tensor],
    ) -> None:
        """Grow/compact ``request.feature_buffer`` and queue the append copies.

        Everything that has to happen in order — the reallocation decision and
        the frame-count bookkeeping — happens here; the copies themselves are
        appended to ``dsts`` / ``srcs`` for the caller's single batched submit.
        The buffer grows amortised-doubled so we never pay an O(T) copy per chunk
        at steady state, and the consumed prefix (before ``feature_cursor``) is
        dropped opportunistically so long utterances don't keep re-allocating.

        Every queued pair reads either ``feats`` or *this* request's outgoing
        buffer and writes a freshly allocated one, so no pair in the batch reads
        what another writes — which is what lets them go in one
        ``torch._foreach_copy_``, whose members are unordered with respect to
        each other.  That is also why dropping the prefix is folded into the
        reallocation instead of compacting first: a standalone compaction leaves
        a buffer exactly as long as what it kept, so a grow *always* followed it,
        chaining old→keep→new — two copies of the same frames, and two copies
        that could not have shared a batch.
        """
        n_new = new_frames.size(0)
        buf = request.feature_buffer
        have = request.feature_frames
        cursor = request.feature_cursor

        # Compact only when the append would not otherwise fit.  Dropping the
        # prefix eagerly (the old ``cursor >= have // 2``) reallocated on **91 %
        # of appends** at steady state, because a stream consumes about as many
        # frames per tick as it gains: ``cursor`` tracks ``have``, the condition
        # is true almost always, and each drop then re-sized the buffer from the
        # *live* window rather than from the old capacity — to ``2 x live``,
        # which one tick refills. A self-perpetuating realloc loop, and
        # ``new_zeros`` is ~7 us, so it cost ~102 us of every 2.6 ms tick at 16
        # streams plus an extra copy pair per stream in the batched submit.
        drop_prefix = buf is not None and cursor > 0 and have + n_new > buf.size(0)
        if drop_prefix:
            keep_n = have - cursor
            src_start = cursor
            # What the old two-step form would have left to double from.
            old_cap = keep_n
        else:
            keep_n = have
            src_start = 0
            old_cap = buf.size(0) if buf is not None else 0

        needed = keep_n + n_new
        if buf is None or drop_prefix or needed > buf.size(0):
            # Leave room for several more appends, so compaction is periodic
            # rather than per-tick.  Sizing from ``old_cap * 2`` cannot do that
            # after a drop: ``old_cap`` is then the live window, and a buffer
            # twice the live window is full again on the next append.  The
            # headroom is capped in absolute frames so a fixed-window frontend,
            # whose ``n_new`` is a whole 30 s window, does not multiply it.
            headroom = min(_FEATURE_HEADROOM_APPENDS * n_new, _FEATURE_HEADROOM_MAX)
            floor = old_cap if buf is not None else 128
            cap = max(needed + headroom, floor)
            new_buf = new_frames.new_zeros(cap, feat_dim)
            if buf is not None and keep_n > 0:
                # Queued before ``request.feature_buffer`` is reassigned, so the
                # view keeps the outgoing storage alive across the batched submit
                # — otherwise the allocator could hand it straight back as some
                # later request's ``new_buf`` while this copy is still pending.
                dsts.append(new_buf[:keep_n])
                srcs.append(buf[src_start : src_start + keep_n])
            request.feature_buffer = new_buf
            buf = new_buf
            if drop_prefix:
                # The cursor is rebased, so the frames it used to count move into
                # ``feature_base``.  Their sum is the stream's absolute input-frame
                # index, which is the only thing that can answer "which seconds of
                # audio does the next encoder window cover" — the question the
                # speech-activity gate asks before deciding to skip one.
                request.feature_base += cursor
                request.feature_cursor = 0

        dsts.append(buf[keep_n : keep_n + n_new])
        srcs.append(new_frames)
        request.feature_frames = keep_n + n_new
