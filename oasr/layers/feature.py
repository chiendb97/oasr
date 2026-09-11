# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Kaldi-compatible FBANK and MFCC — the frontend's one implementation.

Four kernels, no waveform temporaries and no ``unfold``::

    stft_frame   framing + per-frame DC removal + pre-emphasis + window + zero-pad
    rfft_power   real FFT + power spectrum
    mel_log      mel filterbank + Kaldi's epsilon floor + log         --> FBANK
    dct_lifter   DCT-II + cepstral lifter                             --> MFCC

Framing is *varlen*: :func:`oasr.stft_frame` reads each row of the padded batch
under its own valid length, so a batch costs one launch per stage regardless of
how ragged it is.

Two paths, one implementation.  The kernel chain above runs on CUDA; the torch
chain below it is the CPU path, the fp32 parity oracle and the
``OASR_FEATURE_BACKEND=torch`` A/B — the same three-role split the GEMM and
layer backends have.  :mod:`oasr.features.batched` is the engine's entry into
this module, so a served request and :class:`Fbank` run the same code and
cannot drift apart.

Supported profile: ``dither=0``, ``snip_edges=True``, ``use_energy=False``,
window types povey / hanning / hamming / blackman / rectangular.  Anything else
is refused here and served by the per-utterance reference path
(:func:`oasr.features.backends._extract`).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover - the waist must not import the feature package
    # ``oasr.features`` imports this module (batched.py is its entry point), so a
    # runtime import here would be a cycle.  Only the annotations need the name.
    from oasr.features.config import FeatureConfig

__all__ = [
    "Fbank",
    "Mfcc",
    "KALDI_LOG_FLOOR",
    "WINDOW_TYPES",
    "kaldi_fbank",
    "kaldi_mfcc",
    "kaldi_feat_lengths",
]

#: Floor applied to mel energies before the log.  **Kaldi's is ``FLT_EPSILON``**
#: (``mel_energies.ApplyFloor(std::numeric_limits<float>::epsilon())``), which is
#: what ``torchaudio.compliance.kaldi`` — the per-utterance reference path this
#: module must agree with — uses as well.  ``float32`` *tiny* is a different
#: number by 31 orders of magnitude and sets every silent bin to ``-87.34``
#: instead of ``-15.94``; on an ``audio_scale=1.0`` checkpoint (icefall) that
#: reaches a fifth of real frames, and no parity oracle can see it because both
#: sides of a parity test are fed the same features.
KALDI_LOG_FLOOR: float = float(torch.finfo(torch.float32).eps)

#: Kaldi analysis windows this module can build.  A config naming anything else
#: falls to the per-utterance reference rather than being approximated here.
WINDOW_TYPES = ("povey", "hanning", "hamming", "blackman", "rectangular")


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def use_kernel_path(waveforms: torch.Tensor, n_fft: int) -> bool:
    """Whether the CUDA chain serves this call, raising on a *declared* gap.

    CPU and ``OASR_FEATURE_BACKEND=torch`` are out of the kernels' scope and take
    the torch path silently.  An ``n_fft`` the FFT kernel cannot do is a kernel
    gap, so it raises rather than rerouting to torch — a silent reroute makes a
    missing kernel invisible.
    """
    if os.environ.get("OASR_FEATURE_BACKEND", "").strip().lower() == "torch":
        return False
    if not waveforms.is_cuda:
        return False
    if not (8 <= n_fft <= 2048 and (n_fft & (n_fft - 1)) == 0):
        raise NotImplementedError(
            f"the Kaldi feature kernel requires a power-of-two n_fft in [8, 2048], got "
            f"{n_fft}; set OASR_FEATURE_BACKEND=torch for the reference path"
        )
    return True


# ---------------------------------------------------------------------------
# Kaldi tables.  Cached, so every call reuses one tensor object per (config,
# device) pair -- which is also what makes the chain safe to capture in a CUDA
# graph: the addresses the graph bakes in never move.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def kaldi_window(
    window_type: str,
    frame_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Kaldi analysis window, shape ``(frame_length,)``.

    Matches ``torchaudio.compliance.kaldi``'s window functions, which all use the
    symmetric ``N - 1`` denominator rather than numpy's periodic convention.
    """
    i = torch.arange(frame_length, device=device, dtype=dtype)
    a = 2.0 * math.pi * i / (frame_length - 1)
    if window_type == "povey":
        return (0.5 - 0.5 * torch.cos(a)).pow(0.85)
    if window_type == "hanning":
        return 0.5 - 0.5 * torch.cos(a)
    if window_type == "hamming":
        return 0.54 - 0.46 * torch.cos(a)
    if window_type == "blackman":
        return 0.42 - 0.5 * torch.cos(a) + 0.08 * torch.cos(2.0 * a)
    if window_type == "rectangular":
        return torch.ones(frame_length, device=device, dtype=dtype)
    raise ValueError(f"unsupported window_type {window_type!r}; expected one of {WINDOW_TYPES}")


@lru_cache(maxsize=32)
def kaldi_mel_banks(
    num_bins: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Kaldi triangular mel filterbank, shape ``(num_bins, n_fft // 2 + 1)``."""
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq = nyquist + high_freq  # kaldi's "0.0 means Nyquist"
    if not (0.0 <= low_freq < high_freq <= nyquist):
        raise ValueError(
            f"invalid mel cutoffs: low={low_freq}, high={high_freq}, nyquist={nyquist}"
        )

    num_bins_fft = n_fft // 2 + 1

    def _to_mel(f: torch.Tensor) -> torch.Tensor:
        return 1127.0 * torch.log(1.0 + f / 700.0)

    mel_low = _to_mel(torch.tensor(low_freq))
    mel_high = _to_mel(torch.tensor(high_freq))
    mel_edges = torch.linspace(mel_low.item(), mel_high.item(), num_bins + 2)

    # FFT bin center frequencies (Hz).
    bin_hz = torch.arange(num_bins_fft, dtype=torch.float32) * (sample_rate / n_fft)
    bin_mel = _to_mel(bin_hz).unsqueeze(0)  # (1, num_bins_fft)

    left = mel_edges[:-2].unsqueeze(1)  # (num_bins, 1)
    center = mel_edges[1:-1].unsqueeze(1)
    right = mel_edges[2:].unsqueeze(1)

    up = (bin_mel - left) / (center - left)
    down = (right - bin_mel) / (right - center)
    mel_mat = torch.clamp(torch.minimum(up, down), min=0.0)
    return mel_mat.to(device=device, dtype=dtype)


@lru_cache(maxsize=16)
def kaldi_dct_matrix(
    num_ceps: int,
    num_mel_bins: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Kaldi orthonormal DCT-II matrix, shape ``(num_ceps, num_mel_bins)``.

    ``dct[0, n] = 1/sqrt(M)``, ``dct[k, n] = sqrt(2/M) * cos(pi*(n + 0.5)*k / M)``
    — ``torchaudio.functional.create_dct(norm='ortho')`` with Kaldi's C0 weight.
    Built in float64 and rounded once, so the table does not carry the error of
    a float32 ``cos`` into every frame.
    """
    M = num_mel_bins
    n = torch.arange(M, dtype=torch.float64).unsqueeze(0)  # (1, M)
    k = torch.arange(num_ceps, dtype=torch.float64).unsqueeze(1)  # (num_ceps, 1)
    dct = torch.cos(math.pi * (n + 0.5) * k / M) * math.sqrt(2.0 / M)
    dct[0, :] = 1.0 / math.sqrt(M)
    return dct.to(device=device, dtype=dtype)


@lru_cache(maxsize=16)
def kaldi_lifter(
    num_ceps: int,
    cepstral_lifter: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Kaldi cepstral lifter ``1 + (Q/2) sin(pi k / Q)``; ``None`` when ``Q == 0``."""
    if cepstral_lifter == 0.0:
        return None
    k = torch.arange(num_ceps, dtype=torch.float64)
    lifter = 1.0 + 0.5 * cepstral_lifter * torch.sin(math.pi * k / cepstral_lifter)
    return lifter.to(device=device, dtype=dtype)


def kaldi_feat_lengths(lengths: torch.Tensor, frame_length: int, frame_shift: int) -> torch.Tensor:
    """Kaldi ``snip_edges`` frame count per row, as int32."""
    if lengths.dtype != torch.int64:
        lengths = lengths.long()
    return torch.clamp((lengths - frame_length) // frame_shift + 1, min=0).to(torch.int32)


# ---------------------------------------------------------------------------
# The log-mel body: one kernel path, one torch path.
# ---------------------------------------------------------------------------


def _log_mel_kernel(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    cfg: FeatureConfig,
    num_frames: int,
    n_fft: int,
) -> torch.Tensor:
    """``stft_frame`` -> ``rfft_power`` -> ``mel_log``: three launches, no temporaries."""
    import oasr

    device = waveforms.device
    frames = oasr.stft_frame(
        waveforms,
        lengths,
        kaldi_window(cfg.window_type, cfg.frame_length_samples, device),
        n_fft,
        cfg.frame_shift_samples,
        num_frames,
        center_offset=0,  # snip_edges: frame 0 starts at sample 0
        win_offset=0,  # ... and the window sits at the head of the frame
        preemph_coef=float(cfg.preemphasis_coefficient),
        preemph_replicate=True,  # Kaldi's y[0] = (1 - coef) * x[0]
        remove_dc_offset=True,
    )
    power = oasr.rfft_power(frames, n=n_fft)
    mel_mat = kaldi_mel_banks(
        cfg.num_mel_bins, n_fft, cfg.sample_rate, cfg.low_freq, cfg.high_freq, device
    )
    log_mel: torch.Tensor = oasr.mel_log(power, mel_mat, log_floor=KALDI_LOG_FLOOR)
    return log_mel


def _log_mel_torch(waveforms: torch.Tensor, cfg: FeatureConfig, n_fft: int) -> torch.Tensor:
    """The same chain in torch: CPU path, parity oracle, ``OASR_FEATURE_BACKEND`` A/B."""
    device = waveforms.device
    frame_length = cfg.frame_length_samples
    preemph = cfg.preemphasis_coefficient

    # snip_edges=True: drop any trailing tail that does not fit a whole frame.
    frames = waveforms.unfold(-1, frame_length, cfg.frame_shift_samples)

    # Kaldi removes the DC offset per frame, then pre-emphasises what is left —
    # so pre-emphasis is frame-local, with x[-1] replicated as x[0].
    frames = frames - frames.mean(dim=-1, keepdim=True)
    preem = torch.empty_like(frames)
    preem[..., 1:] = frames[..., 1:] - preemph * frames[..., :-1]
    preem[..., 0] = frames[..., 0] - preemph * frames[..., 0]

    windowed = preem * kaldi_window(cfg.window_type, frame_length, device)
    if n_fft > frame_length:
        windowed = torch.nn.functional.pad(windowed, (0, n_fft - frame_length))
    spectrum = torch.fft.rfft(windowed, n=n_fft)  # (B, N, n_fft/2+1) complex
    power = spectrum.real.pow(2) + spectrum.imag.pow(2)

    mel_mat = kaldi_mel_banks(
        cfg.num_mel_bins, n_fft, cfg.sample_rate, cfg.low_freq, cfg.high_freq, device
    )
    mel_energies = torch.matmul(power, mel_mat.t())  # (B, N, num_mel)
    return torch.log(mel_energies.clamp_min(KALDI_LOG_FLOOR))


def _log_mel(
    waveforms: torch.Tensor, lengths: torch.Tensor, cfg: FeatureConfig
) -> Tuple[torch.Tensor, int]:
    """``(log_mel, num_frames)``; ``num_frames == 0`` when no whole frame fits."""
    if waveforms.dim() != 2:
        raise ValueError(f"waveforms must be (B, T), got shape {tuple(waveforms.shape)}")
    B, T = waveforms.shape
    frame_length = cfg.frame_length_samples
    num_frames = max(0, (T - frame_length) // cfg.frame_shift_samples + 1)
    if num_frames == 0:
        return waveforms.new_zeros(B, 0, cfg.num_mel_bins), 0

    n_fft = _next_power_of_two(frame_length)
    if use_kernel_path(waveforms, n_fft):
        return _log_mel_kernel(waveforms, lengths, cfg, num_frames, n_fft), num_frames
    return _log_mel_torch(waveforms, cfg, n_fft), num_frames


def _mfcc_post(log_mel: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """DCT-II + cepstral liftering over ``(B, N, num_mel)`` log-mel energies."""
    device = log_mel.device
    dct = kaldi_dct_matrix(cfg.num_ceps, cfg.num_mel_bins, device)
    lifter = kaldi_lifter(cfg.num_ceps, float(cfg.cepstral_lifter), device)
    if use_kernel_path(log_mel, _next_power_of_two(cfg.frame_length_samples)):
        import oasr

        cepstra: torch.Tensor = oasr.dct_lifter(log_mel, dct, lifter=lifter)
        return cepstra
    mfcc = torch.matmul(log_mel, dct.t())
    return mfcc if lifter is None else mfcc * lifter


# ---------------------------------------------------------------------------
# Functional entry points -- what oasr/features/batched.py calls.
# ---------------------------------------------------------------------------


def kaldi_fbank(
    waveforms: torch.Tensor, lengths: torch.Tensor, cfg: FeatureConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Kaldi log-mel FBANK over a padded ``(B, T)`` waveform batch.

    Args:
        waveforms: ``(B, T)`` float32 waveforms, ``audio_scale`` already applied.
        lengths: ``(B,)`` valid sample counts.
        cfg: A config satisfying :func:`oasr.features.supports_batched_fbank`.

    Returns:
        ``(features, feat_lengths)`` — ``(B, N, num_mel_bins)`` float32 and
        ``(B,)`` int32 frame counts.  Rows shorter than the padded width carry
        real values past their own count; ``feat_lengths`` is what bounds them.
    """
    log_mel, num_frames = _log_mel(waveforms, lengths, cfg)
    if num_frames == 0:
        return log_mel, torch.zeros(waveforms.size(0), dtype=torch.int32, device=waveforms.device)
    return log_mel, kaldi_feat_lengths(lengths, cfg.frame_length_samples, cfg.frame_shift_samples)


def kaldi_mfcc(
    waveforms: torch.Tensor, lengths: torch.Tensor, cfg: FeatureConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Kaldi MFCC: :func:`kaldi_fbank`'s log-mel, then DCT-II + lifter."""
    log_mel, num_frames = _log_mel(waveforms, lengths, cfg)
    device = waveforms.device
    if num_frames == 0:
        return (
            torch.zeros(waveforms.size(0), 0, cfg.num_ceps, device=device, dtype=torch.float32),
            torch.zeros(waveforms.size(0), dtype=torch.int32, device=device),
        )
    return _mfcc_post(log_mel, cfg), kaldi_feat_lengths(
        lengths, cfg.frame_length_samples, cfg.frame_shift_samples
    )


# ---------------------------------------------------------------------------
# Waist modules
# ---------------------------------------------------------------------------


class Fbank(nn.Module):
    """Kaldi log-mel FBANK as a layer, on :func:`kaldi_fbank`.

    Matches :func:`torchaudio.compliance.kaldi.fbank` for the supported options
    (``dither=0``, ``snip_edges=True``, ``use_energy=False``).

    Args:
        config: :class:`FeatureConfig` with the desired Kaldi parameters.

    Example:
        >>> import torch
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
        if config.window_type not in WINDOW_TYPES:
            raise ValueError(
                f"unsupported window_type {config.window_type!r}; expected one of {WINDOW_TYPES}"
            )

        self._config = config
        self.frame_length: int = config.frame_length_samples
        self.frame_shift: int = config.frame_shift_samples
        self.n_fft: int = _next_power_of_two(self.frame_length)
        if not (8 <= self.n_fft <= 2048):
            raise ValueError(
                f"frame_length={self.frame_length} samples leads to n_fft={self.n_fft}, "
                "outside the supported [8, 2048] range of the OASR FFT kernel."
            )

    @property
    def config(self) -> FeatureConfig:
        return self._config

    @property
    def output_dim(self) -> int:
        return self._config.num_mel_bins

    def _prepare(
        self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        elif waveforms.dim() != 2:
            raise ValueError(f"waveforms must be 1-D or 2-D, got shape {tuple(waveforms.shape)}")
        if waveforms.dtype != torch.float32:
            waveforms = waveforms.to(torch.float32)
        if lengths is None:
            lengths = torch.full(
                (waveforms.size(0),),
                waveforms.size(1),
                dtype=torch.int64,
                device=waveforms.device,
            )
        return waveforms, lengths.to(waveforms.device)

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the FBANK pipeline.

        Args:
            waveforms: ``(B, T)`` or ``(T,)`` float32 waveforms.
            lengths: Optional ``(B,)`` valid sample counts; the padded width when
                omitted.

        Returns:
            ``(features, feat_lengths)`` of shapes ``(B, N, num_mel_bins)`` and ``(B,)``.
        """
        waveforms, lengths = self._prepare(waveforms, lengths)
        return kaldi_fbank(waveforms, lengths, self._config)


class Mfcc(Fbank):
    """Kaldi MFCC as a layer, on :func:`kaldi_mfcc`.

    Matches :func:`torchaudio.compliance.kaldi.mfcc` for the supported options
    (``dither=0``, ``snip_edges=True``, ``use_energy=False``).
    """

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
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
        waveforms, lengths = self._prepare(waveforms, lengths)
        return kaldi_mfcc(waveforms, lengths, self._config)
