# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Low-level CUDA-kernel functional API for the log-mel / FBANK / MFCC pipelines.

Four building blocks, which chain through :func:`oasr.rfft_power`:

* :func:`stft_frame` -- framing + signal-domain pre-emphasis + windowing +
  zero-pad, straight off a padded waveform batch. Owns the framing, so the
  caller needs neither ``unfold`` nor ``torch.stft``; the ``center_offset`` /
  ``win_offset`` / boundary knobs cover both the centered NeMo/Whisper grid and
  Kaldi's ``snip_edges`` one.
* :func:`fbank_preprocess` -- the pre-framed alternative to the above: per-frame
  DC removal + pre-emphasis + windowing + zero-pad.
* :func:`mel_log` -- mel filterbank + log (floor and/or additive guard) over a
  power spectrum, with optional per-row frame-count masking.
* :func:`dct_lifter` -- DCT-II + (optional) cepstral lifter + (optional)
  replace ``c[0]`` with a per-frame log-energy. Used by MFCC.

:class:`oasr.layers.Fbank` / :class:`oasr.layers.Mfcc` wrap the pre-framed chain;
:func:`oasr.features.batched_nemotron_logmel` wraps the :func:`stft_frame` one.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api


@functools.cache
def _get_features_module():
    from oasr.jit.features import gen_features_module

    return gen_features_module().build_and_load()


@oasr_api
def stft_frame(
    waveform: torch.Tensor,
    lengths: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    num_frames: int,
    *,
    center_offset: int = 0,
    win_offset: Optional[int] = None,
    preemph_coef: float = 0.0,
    preemph_replicate: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Frame, pre-emphasise, window and zero-pad a padded waveform batch.

    Frame ``f`` covers signal samples ``f * hop_length - center_offset + i`` for
    ``i`` in ``[0, n_fft)``.  Everything outside ``[0, lengths[b])`` reads as
    zero — which is simultaneously the constant STFT padding and the per-row
    length mask.

    Args:
        waveform: ``(B, T)`` float32 padded waveforms (CUDA, contiguous).
        lengths: ``(B,)`` valid sample counts (cast to int32).
        window: ``(win_length,)`` float32 analysis window, ``win_length <= n_fft``.
        n_fft: Frame width in samples.
        hop_length: Samples between consecutive frames.
        num_frames: Frames to emit.  Required rather than derived: the count
            depends on how the caller pads, and only the *symmetric* case has a
            single right answer.
        center_offset: Where frame 0 starts, as a negative offset from sample 0.
            ``n_fft // 2`` reproduces ``torch.stft(center=True,
            pad_mode="constant")``; ``0`` reproduces ``center=False`` (Kaldi's
            ``snip_edges`` framing); a **negative** value skips that many leading
            samples, which is how a streaming caller hands over a buffer whose
            first samples are pre-emphasis history rather than a frame start.
        win_offset: Where the window sits inside the frame. Defaults to
            ``(n_fft - win_length) // 2``, which is what ``torch.stft`` does for
            a short window.
        preemph_coef: ``y[t] = x[t] - coef * x[t-1]`` applied in the **signal**
            domain before windowing. ``0.0`` disables.
        preemph_replicate: Boundary at ``t == 0``: ``False`` gives NeMo's
            ``y[0] = x[0]``, ``True`` gives Kaldi's ``y[0] = (1 - coef) * x[0]``.
        out: Optional pre-allocated ``(B, num_frames, n_fft)`` float32 output.

    Returns:
        Float32 ``(B, num_frames, n_fft)`` ready for :func:`oasr.rfft_power`.
    """
    if waveform.dim() != 2:
        raise ValueError(f"waveform must be (B, T), got shape {tuple(waveform.shape)}")
    if waveform.dtype != torch.float32:
        raise ValueError(f"waveform must be float32, got {waveform.dtype}")
    if window.dtype != torch.float32 or window.dim() != 1:
        raise ValueError("window must be a 1-D float32 tensor")
    win_length = int(window.shape[0])
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")

    if win_offset is None:
        win_offset = (n_fft - win_length) // 2
    if not (0 <= win_offset <= n_fft - win_length):
        raise ValueError(
            f"win_offset {win_offset} places the window outside [0, {n_fft}) "
            f"for win_length {win_length}"
        )

    B = waveform.shape[0]
    out_shape = (B, num_frames, n_fft)
    if out is None:
        out = torch.empty(out_shape, device=waveform.device, dtype=torch.float32)
    elif tuple(out.shape) != out_shape or out.dtype != torch.float32:
        raise ValueError(
            f"out must have shape {out_shape} and dtype float32, "
            f"got shape {tuple(out.shape)} dtype {out.dtype}"
        )
    if num_frames == 0:
        return out

    _get_features_module().stft_frame(
        out,
        waveform.contiguous(),
        lengths.to(device=waveform.device, dtype=torch.int32).contiguous(),
        window,
        int(hop_length),
        int(center_offset),
        int(win_offset),
        float(preemph_coef),
        bool(preemph_replicate),
    )
    return out


@oasr_api
def fbank_preprocess(
    frames: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    preemph_coef: float = 0.97,
    remove_dc_offset: bool = True,
    apply_preemph: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DC removal, pre-emphasis, windowing, and zero-padding for a batch of frames.

    Args:
        frames: ``(..., frame_length)`` float32 framed waveforms (CUDA).
        window: ``(frame_length,)`` float32 window function.
        n_fft: Output FFT length (must be ``>= frame_length``).
        preemph_coef: Pre-emphasis coefficient (Kaldi default ``0.97``).
        remove_dc_offset: Subtract per-frame mean before pre-emphasis.
        apply_preemph: Apply ``y[k] = x[k] - coef * x[k-1]`` (replicate
            boundary at ``k=0``).
        out: Optional pre-allocated output of shape ``(..., n_fft)``.

    Returns:
        Float32 tensor of shape ``(..., n_fft)`` ready for :func:`oasr.rfft_power`.
    """
    if frames.dtype != torch.float32:
        raise ValueError(f"frames must be float32, got {frames.dtype}")
    if window.dtype != torch.float32:
        raise ValueError(f"window must be float32, got {window.dtype}")
    if window.dim() != 1:
        raise ValueError("window must be 1-D")

    frame_length = frames.shape[-1]
    if window.shape[0] != frame_length:
        raise ValueError(
            f"window length ({window.shape[0]}) must equal frame_length ({frame_length})"
        )
    if n_fft < frame_length:
        raise ValueError(f"n_fft ({n_fft}) must be >= frame_length ({frame_length})")

    out_shape = frames.shape[:-1] + (n_fft,)
    if out is None:
        out = torch.empty(out_shape, device=frames.device, dtype=torch.float32)
    elif out.shape != out_shape or out.dtype != torch.float32:
        raise ValueError(
            f"out must have shape {tuple(out_shape)} and dtype float32, "
            f"got shape {tuple(out.shape)} dtype {out.dtype}"
        )

    _get_features_module().fbank_preprocess(
        out,
        frames.contiguous(),
        window,
        float(preemph_coef),
        bool(remove_dc_offset),
        bool(apply_preemph),
    )
    return out


@oasr_api
def mel_log(
    power: torch.Tensor,
    mel_mat: torch.Tensor,
    log_floor: float = 1.1754944e-38,
    out: Optional[torch.Tensor] = None,
    *,
    log_offset: float = 0.0,
    frame_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mel filterbank + log over a real-FFT power spectrum.

    Computes ``log(max(mel_mat @ power, log_floor) + log_offset)`` per frame,
    fused into one kernel.

    Args:
        power: ``(..., n_freq)`` float32 power spectrum (CUDA).
        mel_mat: ``(num_mel, n_freq)`` float32 mel filterbank.
        log_floor: Floor applied before ``log`` (default: ``float32`` tiny —
            Kaldi's convention).
        out: Optional pre-allocated ``(..., num_mel)`` output.
        log_offset: Added after the floor (NeMo's ``2 ** -24`` guard). The two
            are separate knobs because they set the value of a silent bin
            differently, and that value is the encoder's input scale.
        frame_lengths: Optional ``(B,)`` valid frame count per row; requires a
            3-D ``(B, num_frames, n_freq)`` ``power``. Frames at or past a row's
            count are written as **zero** rather than as ``log`` of a silent bin.

    Returns:
        Float32 tensor of shape ``(..., num_mel)`` holding log-mel energies.
    """
    if power.dtype != torch.float32 or mel_mat.dtype != torch.float32:
        raise ValueError("power and mel_mat must both be float32")
    if mel_mat.dim() != 2:
        raise ValueError("mel_mat must be 2-D (num_mel, n_freq)")
    if mel_mat.shape[1] != power.shape[-1]:
        raise ValueError(
            f"mel_mat second dim ({mel_mat.shape[1]}) must equal power last dim "
            f"({power.shape[-1]})"
        )
    if frame_lengths is not None and power.dim() != 3:
        raise ValueError(
            "frame_lengths requires a 3-D (B, num_frames, n_freq) power tensor, "
            f"got {power.dim()}-D"
        )

    num_mel = mel_mat.shape[0]
    out_shape = power.shape[:-1] + (num_mel,)
    if out is None:
        out = torch.empty(out_shape, device=power.device, dtype=torch.float32)
    elif out.shape != out_shape or out.dtype != torch.float32:
        raise ValueError(
            f"out must have shape {tuple(out_shape)} and dtype float32, "
            f"got shape {tuple(out.shape)} dtype {out.dtype}"
        )

    lens = (
        None
        if frame_lengths is None
        else frame_lengths.to(device=power.device, dtype=torch.int32).contiguous()
    )
    _get_features_module().mel_log(
        out,
        power.contiguous(),
        mel_mat.contiguous(),
        float(log_floor),
        float(log_offset),
        lens,
    )
    return out


@oasr_api
def dct_lifter(
    log_mel: torch.Tensor,
    dct_mat: torch.Tensor,
    lifter: Optional[torch.Tensor] = None,
    energy: Optional[torch.Tensor] = None,
    replace_c0_with_energy: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DCT-II + optional cepstral lifter + optional ``c[0] = log_energy``.

    Args:
        log_mel: ``(..., num_mel)`` float32 log-mel energies (CUDA).
        dct_mat: ``(num_ceps, num_mel)`` float32 DCT matrix.
        lifter: Optional ``(num_ceps,)`` float32 cepstral lifter weights.
        energy: Optional ``(total_frames,)`` float32 frame log-energies.
        replace_c0_with_energy: If True (and ``energy`` is provided), overwrite
            the output's first cepstral coefficient with ``energy[frame]``.
        out: Optional pre-allocated ``(..., num_ceps)`` output.

    Returns:
        Float32 tensor of shape ``(..., num_ceps)`` holding the MFCC.
    """
    if log_mel.dtype != torch.float32 or dct_mat.dtype != torch.float32:
        raise ValueError("log_mel and dct_mat must both be float32")
    if dct_mat.dim() != 2:
        raise ValueError("dct_mat must be 2-D (num_ceps, num_mel)")
    if dct_mat.shape[1] != log_mel.shape[-1]:
        raise ValueError(
            f"dct_mat second dim ({dct_mat.shape[1]}) must equal log_mel last dim "
            f"({log_mel.shape[-1]})"
        )
    if lifter is not None:
        if lifter.dtype != torch.float32 or lifter.shape != (dct_mat.shape[0],):
            raise ValueError(
                f"lifter must be float32 with shape ({dct_mat.shape[0]},), "
                f"got dtype {lifter.dtype} shape {tuple(lifter.shape)}"
            )

    num_ceps = dct_mat.shape[0]
    out_shape = log_mel.shape[:-1] + (num_ceps,)
    if out is None:
        out = torch.empty(out_shape, device=log_mel.device, dtype=torch.float32)
    elif out.shape != out_shape or out.dtype != torch.float32:
        raise ValueError(
            f"out must have shape {tuple(out_shape)} and dtype float32, "
            f"got shape {tuple(out.shape)} dtype {out.dtype}"
        )

    _get_features_module().dct_lifter(
        out,
        log_mel.contiguous(),
        dct_mat.contiguous(),
        lifter,
        energy,
        bool(replace_c0_with_energy),
    )
    return out
