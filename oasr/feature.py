# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Low-level CUDA-kernel functional API for the FBANK / MFCC pipeline.

Three building blocks back the :class:`oasr.layers.Fbank` and
:class:`oasr.layers.Mfcc` modules:

* :func:`fbank_preprocess` -- DC removal + pre-emphasis + windowing + zero-pad
  for one batch of frames.
* :func:`mel_log` -- mel filterbank + log floor over a power spectrum.
* :func:`dct_lifter` -- DCT-II + (optional) cepstral lifter + (optional)
  replace ``c[0]`` with a per-frame log-energy. Used by MFCC.

These primitives chain naturally with :func:`oasr.rfft_power`.
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
) -> torch.Tensor:
    """Mel filterbank + log over a real-FFT power spectrum.

    Computes ``log(max(mel_mat @ power, log_floor))`` per frame, fused into
    one kernel.

    Args:
        power: ``(..., n_freq)`` float32 power spectrum (CUDA).
        mel_mat: ``(num_mel, n_freq)`` float32 mel filterbank.
        log_floor: Floor to apply before ``log`` (default: ``float32`` tiny).
        out: Optional pre-allocated ``(..., num_mel)`` output.

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

    num_mel = mel_mat.shape[0]
    out_shape = power.shape[:-1] + (num_mel,)
    if out is None:
        out = torch.empty(out_shape, device=power.device, dtype=torch.float32)
    elif out.shape != out_shape or out.dtype != torch.float32:
        raise ValueError(
            f"out must have shape {tuple(out_shape)} and dtype float32, "
            f"got shape {tuple(out.shape)} dtype {out.dtype}"
        )

    _get_features_module().mel_log(
        out, power.contiguous(), mel_mat.contiguous(), float(log_floor)
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
