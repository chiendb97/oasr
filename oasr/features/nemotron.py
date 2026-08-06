# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""NeMo/Nemotron log-mel spectrogram frontend (batched, offline + streaming).

Reproduces ``NemotronAsrStreamingFeatureExtractor`` (which in turn reproduces
NeMo's ``AudioToMelSpectrogramPreprocessor`` for the Nemotron ASR releases):

* input waveforms at **[-1, 1] float scale** (``FeatureSpec.audio_scale = 1.0``
  — like the icefall/Whisper frontends and *unlike* WeNet's int16 scale);
* per-row **pre-emphasis** ``x[t] - 0.97 * x[t-1]`` with ``x[0]`` kept, applied
  before the STFT and re-zeroed past each row's valid length;
* STFT ``n_fft = 512`` (the power of two above ``win_length = 400``),
  ``hop = 160``, **non-periodic** Hann window, centered with *constant* (zero)
  padding — not reflect;
* power spectrum, slaney-scale mel filterbank (``fmin = 0``, ``fmax = sr/2``);
* ``log(mel + 2**-24)`` — a **log of the raw mel**, not ``log10`` and not
  normalized in any way.  Nemotron never applies per-feature CMVN, so unlike
  every other frontend in this package there is nothing to normalize over and
  the recipe is exactly frame-local.

Two consequences of that last point are worth naming, because they are what make
this its own extractor rather than a :class:`~oasr.features.FeatureConfig` of the
Kaldi one:

1. the log guard ``2**-24`` and the missing ``log10``/``dither``/``povey``
   window put it outside what ``oasr.features.batched`` can express;
2. the returned frame count is ``floor(L / hop)`` while the tensor is
   ``floor(L_max / hop) + 1`` frames wide.  That is not an off-by-one: it is
   HF's ``attention_mask`` convention, and the encoder's causal subsampling
   reads the mask, so a frontend that claimed the extra frame would feed the
   encoder one frame of zero-padding as if it were audio.

Two implementations, one recipe
-------------------------------
:func:`batched_nemotron_logmel` runs on **OASR kernels** when the batch is on
CUDA (``stft_frame`` → ``rfft_power`` → ``mel_log``: three launches, no
intermediate waveform copies) and on the torch reference otherwise.  The torch
path is not dead weight — it is the CPU path, the fp32 parity oracle the tests
compare the kernel against, and the "is this the kernels' fault" A/B.  Force it
with ``OASR_FEATURE_BACKEND=torch``.

Streaming
---------
The recipe is frame-local, so it streams — but the frame *grid* comes from one
``center=True`` pass over the whole utterance, so a chunked caller has to
reproduce it rather than restart it.  :func:`nemotron_streaming_framing` declares
how (:class:`~oasr.features.StreamingFraming`) and
:func:`batched_nemotron_logmel_streaming` consumes it:

* ``prefill = n_fft // 2 + 1`` zero samples start the buffer — ``n_fft // 2`` for
  the centered grid's implicit left pad, plus **one** for pre-emphasis, which
  NeMo applies to the *signal* and which therefore reaches one sample before
  each frame;
* ``history = 1`` marks that leading sample as context rather than a frame start;
* ``span = n_fft`` (not ``win_length``) is what one frame reads.

Verified bit-exact against the offline centered pass (max abs difference 0.0 in
fp64), which is also what upstream's own docstring promises: feeding
``audio[hop * frame - n_fft // 2 :]`` with ``center=False`` reproduces it
frame for frame.
"""

from __future__ import annotations

import functools
import os
from typing import Tuple

import torch

from .config import FeatureConfig
from .registry import StreamingFraming

__all__ = [
    "batched_nemotron_logmel",
    "batched_nemotron_logmel_streaming",
    "nemotron_stft_geometry",
    "nemotron_streaming_framing",
]

#: The log-zero guard NeMo uses (``2 ** -24``); reproduced exactly because it
#: sets the floor of every silent bin and therefore the encoder's input scale.
LOG_ZERO_GUARD = 2.0**-24

#: Widest FFT the OASR ``rfft`` kernel accepts (``oasr/fft.py::_validate_n_fft``).
_MAX_KERNEL_N_FFT = 2048


def nemotron_stft_geometry(cfg: FeatureConfig) -> Tuple[int, int, int]:
    """``(n_fft, hop_length, win_length)`` in samples for ``cfg``.

    ``n_fft`` is the power of two at or above ``win_length`` (NeMo's
    ``2 ** ceil(log2(win_length))``), so the released 25 ms / 10 ms geometry at
    16 kHz gives ``(512, 160, 400)``.
    """
    win_length = cfg.frame_length_samples
    hop_length = cfg.frame_shift_samples
    n_fft = 1 << max(0, (win_length - 1).bit_length())
    return n_fft, hop_length, win_length


def nemotron_streaming_framing(cfg: FeatureConfig) -> StreamingFraming:
    """How to reproduce this frontend's frame grid from a growing sample buffer."""
    n_fft, hop, _ = nemotron_stft_geometry(cfg)
    history = 1 if cfg.preemphasis_coefficient else 0
    return StreamingFraming(span=n_fft, hop=hop, history=history, prefill=n_fft // 2 + history)


@functools.lru_cache(maxsize=8)
def _hann_window(win_length: int, device_str: str) -> torch.Tensor:
    """fp32 **non-periodic** Hann window, cached per (length, device).

    ``periodic=False`` matters: it is a different window from the
    ``torch.stft`` default and the one NeMo trained with.
    """
    return torch.hann_window(
        win_length, periodic=False, device=torch.device(device_str), dtype=torch.float32
    )


@functools.lru_cache(maxsize=8)
def _padded_hann_window(win_length: int, n_fft: int, device_str: str) -> torch.Tensor:
    """The Hann window zero-padded, centered, to ``n_fft`` — what ``torch.stft`` uses.

    ``torch.stft`` pads a short ``window`` symmetrically to ``n_fft`` before
    applying it, so a kernel that windows the whole ``n_fft`` frame needs the
    padded form.  The OASR kernel instead takes the *unpadded* window plus a
    ``win_offset``, so this exists only for the torch reference's tests.
    """
    left = (n_fft - win_length) // 2
    return torch.nn.functional.pad(
        _hann_window(win_length, device_str), (left, n_fft - win_length - left)
    )


@functools.lru_cache(maxsize=8)
def _mel_filters(sample_rate: int, n_fft: int, n_mels: int, device_str: str) -> torch.Tensor:
    """Slaney-normalized slaney-scale mel filterbank ``(n_mels, n_fft // 2 + 1)``.

    ``torchaudio.functional.melscale_fbanks(norm="slaney", mel_scale="slaney")``
    is the same table ``librosa.filters.mel`` builds — the one the HF feature
    extractor uses, and which it prefers over ``transformers``'s own
    ``mel_filter_bank`` for exactly this reason.  Verified against librosa at
    ``(16 kHz, 512, 128)``: max absolute difference 2.6e-7, which is fp32
    accumulation order, not a different filter.
    """
    from torchaudio.functional import melscale_fbanks

    fb = melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )  # (n_freqs, n_mels)
    table: torch.Tensor = fb.t().contiguous().to(torch.device(device_str))
    return table


def _use_kernel(waveforms: torch.Tensor, n_fft: int) -> bool:
    """Whether the fused kernel chain can serve this call.

    CPU is out of scope for the kernels (as everywhere in OASR), and the ``rfft``
    kernel is limited to a power-of-two length in ``[8, 2048]`` — ``n_fft`` is a
    power of two by construction, so only the range needs checking.
    """
    if os.environ.get("OASR_FEATURE_BACKEND") == "torch":
        return False
    return waveforms.is_cuda and 8 <= n_fft <= _MAX_KERNEL_N_FFT


def _frame_counts(
    lengths: torch.Tensor,
    total_samples: int,
    n_fft: int,
    hop: int,
    history: int,
    center: bool,
) -> Tuple[int, torch.Tensor]:
    """``(num_frames, per_row_valid_frames)`` for one framing mode.

    Offline (``center``) matches ``torch.stft(center=True)``'s tensor width
    ``floor(T / hop) + 1`` and HF's valid count ``floor(L / hop)``.  Streaming
    frames a buffer whose first ``history`` samples are context only, so both
    counts are ``floor((n - history - n_fft) / hop) + 1``.
    """
    if center:
        num_frames = total_samples // hop + 1
        valid = torch.div(lengths, hop, rounding_mode="floor")
    else:
        num_frames = max(0, (total_samples - history - n_fft) // hop + 1)
        valid = torch.div(lengths - history - n_fft, hop, rounding_mode="floor") + 1
    valid = valid.clamp(min=0, max=num_frames).to(torch.int32)
    return num_frames, valid


def _logmel_kernel(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
    *,
    center: bool,
    history: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``stft_frame`` → ``rfft_power`` → ``mel_log``: three launches, no temporaries."""
    import oasr

    device_str = str(waveforms.device)
    n_fft, hop, win_length = nemotron_stft_geometry(cfg)
    num_frames, feat_lengths = _frame_counts(
        lengths, waveforms.size(1), n_fft, hop, history, center
    )
    if num_frames == 0:
        empty = waveforms.new_zeros(waveforms.size(0), 0, cfg.num_mel_bins)
        return empty, feat_lengths

    frames = oasr.stft_frame(
        waveforms,
        lengths,
        _hann_window(win_length, device_str),
        n_fft,
        hop,
        num_frames,
        # A centered pass starts frame 0 at ``-n_fft // 2``; a streaming buffer
        # starts it *after* the ``history`` pre-emphasis samples.
        center_offset=(n_fft // 2 if center else -history),
        preemph_coef=float(cfg.preemphasis_coefficient),
        preemph_replicate=False,  # NeMo keeps x[0]; Kaldi would replicate it
    )
    power = oasr.rfft_power(frames, n=n_fft)
    features = oasr.mel_log(
        power,
        _mel_filters(cfg.sample_rate, n_fft, cfg.num_mel_bins, device_str),
        log_floor=0.0,
        log_offset=LOG_ZERO_GUARD,
        # Zeroes the padded tail: ``log`` of a silent bin is a large negative
        # constant, not zero, so an unmasked tail is real-looking energy.
        frame_lengths=feat_lengths,
    )
    return features, feat_lengths


def _logmel_torch(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
    *,
    center: bool,
    history: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation: the upstream op sequence, op for op."""
    device = waveforms.device
    device_str = str(device)
    n_fft, hop, win_length = nemotron_stft_geometry(cfg)

    time_mask = torch.arange(waveforms.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
    wav = (waveforms * time_mask).float()

    coeff = cfg.preemphasis_coefficient
    if coeff:
        # Keep sample 0, then first-difference.  The re-mask matters: at index
        # ``len`` the difference is ``0 - coeff * x[len - 1]``, which is not zero.
        wav = torch.cat([wav[:, :1], wav[:, 1:] - coeff * wav[:, :-1]], dim=1)
        wav = wav * time_mask

    num_frames, feat_lengths = _frame_counts(
        lengths, waveforms.size(1), n_fft, hop, history, center
    )
    if num_frames == 0:
        return waveforms.new_zeros(waveforms.size(0), 0, cfg.num_mel_bins), feat_lengths

    if center:
        stft = torch.stft(
            wav,
            n_fft,
            hop_length=hop,
            win_length=win_length,
            window=_hann_window(win_length, device_str),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        parts = torch.view_as_real(stft)
        power = parts.pow(2).sum(-1).sqrt().pow(2)  # (B, n_freqs, n_frames)
    else:
        # Frame the buffer past its ``history`` context samples.  ``torch.stft``
        # cannot express that offset, so frame explicitly and reuse its padded
        # window (a short window is centered inside ``n_fft``).
        body = wav[:, history:]
        need = (num_frames - 1) * hop + n_fft
        if body.size(1) < need:
            body = torch.nn.functional.pad(body, (0, need - body.size(1)))
        frames = body.unfold(1, n_fft, hop)[:, :num_frames]
        windowed = frames * _padded_hann_window(win_length, n_fft, device_str)
        spec = torch.fft.rfft(windowed, n=n_fft)
        parts = torch.view_as_real(spec)
        power = parts.pow(2).sum(-1).sqrt().pow(2).transpose(1, 2)  # (B, n_freqs, n_frames)

    filters = _mel_filters(cfg.sample_rate, n_fft, cfg.num_mel_bins, device_str)
    mel = filters @ power  # (B, n_mels, n_frames)
    features = torch.log(mel + LOG_ZERO_GUARD).transpose(1, 2)  # (B, n_frames, n_mels)

    frame_mask = torch.arange(features.size(1), device=device).unsqueeze(
        0
    ) < feat_lengths.unsqueeze(1)
    return (features * frame_mask.unsqueeze(-1)).contiguous(), feat_lengths


def _logmel(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
    *,
    center: bool,
    history: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert waveforms.dim() == 2, "waveforms must be (B, T)"
    n_fft, _, _ = nemotron_stft_geometry(cfg)
    lengths = lengths.to(device=waveforms.device)
    runner = _logmel_kernel if _use_kernel(waveforms, n_fft) else _logmel_torch
    return runner(waveforms, lengths, cfg, center=center, history=history)


def batched_nemotron_logmel(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Nemotron log-mel features over whole utterances.

    Parameters
    ----------
    waveforms : Tensor
        ``(B, T)`` float32 waveforms at **[-1, 1]** scale.
    lengths : Tensor
        ``(B,)`` valid sample counts.  Everything past a row's length is zeroed
        *before* pre-emphasis and the resulting frames are zeroed after the
        mel, so padding contributes neither energy nor a log floor.
    cfg : FeatureConfig
        ``feature_type="nemotron_logmel"``; reads ``sample_rate``,
        ``num_mel_bins``, ``frame_length_ms``, ``frame_shift_ms`` and
        ``preemphasis_coefficient``.

    Returns
    -------
    features : Tensor
        ``(B, floor(T / hop) + 1, num_mel_bins)`` float32.
    feat_lengths : Tensor
        ``(B,)`` int32 — ``floor(len / hop)`` per row (HF's attention-mask
        count, one less than the number of STFT frames a centered transform
        produces; see the module docstring).
    """
    return _logmel(waveforms, lengths, cfg, center=True, history=0)


def batched_nemotron_logmel_streaming(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Incremental variant: frame a growing buffer, not a whole utterance.

    ``waveforms`` is each stream's carried-over buffer with the next audio chunk
    appended, laid out as :func:`nemotron_streaming_framing` describes — the
    first ``history`` samples are pre-emphasis context, frame ``f`` then spans
    ``[history + f * hop, history + f * hop + n_fft)``.  Emitting
    ``floor((n - history - n_fft) / hop) + 1`` frames and retaining ``buf[F * hop:]``
    reproduces the offline centered grid exactly.
    """
    framing = nemotron_streaming_framing(cfg)
    return _logmel(waveforms, lengths, cfg, center=False, history=framing.history)
