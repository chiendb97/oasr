# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GPU-accelerated FBANK and MFCC feature-extraction modules.

End-to-end Kaldi-compatible (matches :mod:`torchaudio.compliance.kaldi`) FBANK
and MFCC pipelines built on the OASR CUDA kernels:

    framing          (torch.unfold, zero-copy view)
    fbank_preprocess (DC removal + pre-emphasis + windowing + zero-pad)
    rfft_power       (real-FFT + power spectrum)
    mel_log          (mel filterbank + log floor)        --> FBANK output
    dct_lifter       (DCT-II + cepstral lifter)          --> MFCC output

Every per-frame transform is a single fused CUDA kernel, eliminating temporary
tensor traffic that dominates short FFTs.

Currently supports the inference profile that OASR's offline pipeline uses:
``dither=0``, ``snip_edges=True``, ``use_energy=False`` by default. Window types
povey / hanning / hamming / blackman / rectangular are all supported.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

import oasr
from oasr.feature import dct_lifter as _cuda_dct_lifter
from oasr.feature import fbank_preprocess as _cuda_fbank_preprocess
from oasr.feature import mel_log as _cuda_mel_log
from oasr.features.config import FeatureConfig

__all__ = ["Fbank", "Mfcc"]


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _build_window(window_type: str, frame_length: int) -> torch.Tensor:
    i = torch.arange(frame_length, dtype=torch.float32)
    if window_type == "povey":
        w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * i / (frame_length - 1))
        return w.pow(0.85)
    if window_type == "hanning":
        return 0.5 - 0.5 * torch.cos(2.0 * math.pi * i / (frame_length - 1))
    if window_type == "hamming":
        return 0.54 - 0.46 * torch.cos(2.0 * math.pi * i / (frame_length - 1))
    if window_type == "blackman":
        return (
            0.42
            - 0.5 * torch.cos(2.0 * math.pi * i / (frame_length - 1))
            + 0.08 * torch.cos(4.0 * math.pi * i / (frame_length - 1))
        )
    if window_type == "rectangular":
        return torch.ones(frame_length, dtype=torch.float32)
    raise ValueError(
        f"Unknown window_type {window_type!r}; expected one of "
        "'povey', 'hanning', 'hamming', 'blackman', 'rectangular'"
    )


def _build_mel_filterbank(
    num_bins: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
) -> torch.Tensor:
    """Kaldi-style triangular mel filterbank, shape ``(num_bins, n_fft//2+1)``."""
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq = nyquist + high_freq
    if not (0.0 <= low_freq < high_freq <= nyquist):
        raise ValueError(
            f"Invalid mel cutoffs: low={low_freq}, high={high_freq}, nyquist={nyquist}"
        )

    num_freq = n_fft // 2 + 1

    def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 1127.0 * torch.log(1.0 + f / 700.0)

    mel_low = _hz_to_mel(torch.tensor(low_freq))
    mel_high = _hz_to_mel(torch.tensor(high_freq))
    mel_edges = torch.linspace(mel_low.item(), mel_high.item(), num_bins + 2)

    bin_hz = torch.arange(num_freq, dtype=torch.float32) * (sample_rate / n_fft)
    bin_mel = _hz_to_mel(bin_hz)

    left = mel_edges[:-2].unsqueeze(1)
    center = mel_edges[1:-1].unsqueeze(1)
    right = mel_edges[2:].unsqueeze(1)
    bin_mel = bin_mel.unsqueeze(0)

    up = (bin_mel - left) / (center - left)
    down = (right - bin_mel) / (right - center)
    return torch.clamp(torch.minimum(up, down), min=0.0).to(torch.float32)


def _build_dct_matrix(num_ceps: int, num_mel: int) -> torch.Tensor:
    """Kaldi-style orthonormal DCT-II of shape ``(num_ceps, num_mel)``."""
    n = torch.arange(num_mel, dtype=torch.float32) + 0.5
    k = torch.arange(num_ceps, dtype=torch.float32).unsqueeze(1)
    mat = torch.cos(math.pi * k * n / num_mel)
    mat[0] *= 1.0 / math.sqrt(num_mel)
    if num_ceps > 1:
        mat[1:] *= math.sqrt(2.0 / num_mel)
    return mat


def _build_lifter(num_ceps: int, lifter_q: float) -> Optional[torch.Tensor]:
    """Kaldi cepstral lifter ``1 + (Q/2) * sin(pi * n / Q)``; ``None`` if ``Q==0``."""
    if lifter_q == 0.0:
        return None
    n = torch.arange(num_ceps, dtype=torch.float32)
    return 1.0 + (lifter_q / 2.0) * torch.sin(math.pi * n / lifter_q)


def _frame_lengths(lengths: torch.Tensor, frame_length: int, frame_shift: int) -> torch.Tensor:
    """Number of valid frames per utterance under ``snip_edges=True``."""
    return torch.clamp((lengths.long() - frame_length) // frame_shift + 1, min=0).to(
        torch.int32
    )


class Fbank(nn.Module):
    """End-to-end GPU log-mel FBANK extractor backed by OASR CUDA kernels.

    Matches :func:`torchaudio.compliance.kaldi.fbank` output for the supported
    options (``dither=0``, ``snip_edges=True``, ``use_energy=False``).

    Args:
        config: :class:`FeatureConfig` with the desired Kaldi parameters.
            ``feature_type`` is forced to ``"fbank"``.

    Example:
        >>> import torch, oasr
        >>> from oasr.features import FeatureConfig
        >>> from oasr.layers import Fbank
        >>> fb = Fbank(FeatureConfig(num_mel_bins=80)).cuda()
        >>> wav = torch.randn(4, 16000, device="cuda")
        >>> feats, feat_lens = fb(wav)  # (4, 98, 80), (4,)
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        if config.dither != 0.0:
            raise NotImplementedError("Fbank currently requires dither=0.0")
        if not config.snip_edges:
            raise NotImplementedError("Fbank currently requires snip_edges=True")

        self._config = config
        self.frame_length: int = config.frame_length_samples
        self.frame_shift: int = config.frame_shift_samples
        self.n_fft: int = _next_power_of_two(self.frame_length)
        self.preemph: float = float(config.preemphasis_coefficient)
        self.remove_dc_offset: bool = True  # Kaldi default
        self.apply_preemph: bool = self.preemph != 0.0

        if not (8 <= self.n_fft <= 2048):
            raise ValueError(
                f"frame_length={self.frame_length} samples leads to n_fft={self.n_fft}, "
                "outside the supported [8, 2048] range of the OASR FFT kernel."
            )

        window = _build_window(config.window_type, self.frame_length)
        mel_mat = _build_mel_filterbank(
            num_bins=config.num_mel_bins,
            n_fft=self.n_fft,
            sample_rate=config.sample_rate,
            low_freq=config.low_freq,
            high_freq=config.high_freq,
        )
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("mel_mat", mel_mat, persistent=False)

        # Kaldi clamps to a tiny positive epsilon; use float32 tiny for parity
        # with torchaudio's behavior under energy_floor=0.
        self._log_floor: float = float(torch.finfo(torch.float32).tiny)

    @property
    def config(self) -> FeatureConfig:
        return self._config

    @property
    def output_dim(self) -> int:
        return self._config.num_mel_bins

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the FBANK pipeline.

        Args:
            waveforms: ``(B, T)`` or ``(T,)`` float32 waveforms on CUDA.
            lengths: Optional ``(B,)`` int tensor of valid sample counts.

        Returns:
            ``(features, feat_lengths)`` where ``features`` has shape
            ``(B, max_frames, num_mel_bins)`` and ``feat_lengths`` has shape ``(B,)``.
        """
        return self._run(waveforms, lengths, fuse_postprocess=None)

    def _frame_and_preprocess(self, waveforms: torch.Tensor):
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        elif waveforms.dim() != 2:
            raise ValueError(f"waveforms must be 1-D or 2-D, got shape {tuple(waveforms.shape)}")
        if waveforms.dtype != torch.float32:
            waveforms = waveforms.to(torch.float32)
        if not waveforms.is_cuda:
            raise ValueError("Fbank requires CUDA tensors")

        B, T = waveforms.shape
        if T < self.frame_length:
            return waveforms, B, 0, None

        frames = waveforms.unfold(-1, self.frame_length, self.frame_shift)
        num_frames = frames.size(1)
        frames = frames.contiguous()  # (B, num_frames, frame_length)

        windowed = _cuda_fbank_preprocess(
            frames,
            self.window,
            n_fft=self.n_fft,
            preemph_coef=self.preemph,
            remove_dc_offset=self.remove_dc_offset,
            apply_preemph=self.apply_preemph,
        )
        return waveforms, B, num_frames, windowed

    def _run(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor],
        fuse_postprocess,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        waveforms, B, num_frames, windowed = self._frame_and_preprocess(waveforms)
        device = waveforms.device

        if num_frames == 0:
            out_dim = (
                self._config.num_ceps
                if self._config.feature_type == "mfcc"
                else self._config.num_mel_bins
            )
            return (
                torch.zeros(B, 0, out_dim, device=device, dtype=torch.float32),
                torch.zeros(B, dtype=torch.int32, device=device),
            )

        power = oasr.rfft_power(windowed, n=self.n_fft)        # (B, F, n_fft//2+1)
        log_mel = _cuda_mel_log(power, self.mel_mat, log_floor=self._log_floor)  # (B, F, M)

        if fuse_postprocess is not None:
            features = fuse_postprocess(log_mel)
        else:
            features = log_mel

        if lengths is None:
            feat_lengths = torch.full(
                (B,), num_frames, dtype=torch.int32, device=device
            )
        else:
            feat_lengths = _frame_lengths(
                lengths.to(device), self.frame_length, self.frame_shift
            )
        return features, feat_lengths


class Mfcc(Fbank):
    """End-to-end GPU MFCC extractor backed by OASR CUDA kernels.

    Matches :func:`torchaudio.compliance.kaldi.mfcc` output for the supported
    options (``dither=0``, ``snip_edges=True``, ``use_energy=False``).

    Args:
        config: :class:`FeatureConfig`. ``feature_type`` is forced to ``"mfcc"``.
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        # Build DCT + lifter on top of FBANK buffers.
        dct_mat = _build_dct_matrix(config.num_ceps, config.num_mel_bins)
        lifter = _build_lifter(config.num_ceps, float(config.cepstral_lifter))
        self.register_buffer("dct_mat", dct_mat, persistent=False)
        if lifter is not None:
            self.register_buffer("lifter", lifter, persistent=False)
        else:
            self.lifter = None  # type: ignore[assignment]

        if config.use_energy:
            raise NotImplementedError("Mfcc currently requires use_energy=False")

    @property
    def output_dim(self) -> int:
        return self._config.num_ceps

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _post(log_mel: torch.Tensor) -> torch.Tensor:
            return _cuda_dct_lifter(
                log_mel,
                self.dct_mat,
                lifter=self.lifter,
                energy=None,
                replace_c0_with_energy=False,
            )

        return self._run(waveforms, lengths, fuse_postprocess=_post)
