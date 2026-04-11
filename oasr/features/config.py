# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Feature extraction configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for batched audio feature extraction.

    Wraps Kaldi-compatible parameters used by both ``torchaudio`` and
    ``kaldifeat`` backends.  Default values match the standard 80-dim
    log-mel FBANK configuration used by Conformer / Paraformer models.

    Parameters
    ----------
    feature_type : str
        ``"fbank"`` for log-mel filterbank or ``"mfcc"`` for MFCC.
    sample_rate : int
        Audio sample rate in Hz.
    num_mel_bins : int
        Number of mel filterbank channels.
    frame_length_ms : float
        Analysis frame length in milliseconds.
    frame_shift_ms : float
        Frame shift (hop length) in milliseconds.
    dither : float
        Dithering constant (0.0 disables dithering).  Set to ``0.0``
        for deterministic chunk-by-chunk output.
    energy_floor : float
        Floor on energy (absolute) for log computation.
    preemphasis_coefficient : float
        Pre-emphasis filter coefficient.
    window_type : str
        Window function: ``"povey"``, ``"hanning"``, ``"hamming"``,
        ``"blackman"``, or ``"rectangular"``.
    num_ceps : int
        Number of cepstral coefficients to retain (MFCC only).
    cepstral_lifter : float
        Cepstral liftering constant (MFCC only).
    use_energy : bool
        If True, replace C0 with log-energy.
    low_freq : float
        Low cutoff frequency for mel filterbank (Hz).
    high_freq : float
        High cutoff frequency for mel filterbank (Hz).
        ``0.0`` means Nyquist (``sample_rate / 2``).
    snip_edges : bool
        If True, only produce frames that fit entirely within the signal.
        Must be ``True`` for :class:`BatchedStreamingFeatureExtractor`.
    backend : str
        ``"torchaudio"`` or ``"kaldifeat"``.
    """

    feature_type: str = "fbank"
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    dither: float = 0.0
    energy_floor: float = 0.0
    preemphasis_coefficient: float = 0.97
    window_type: str = "povey"
    num_ceps: int = 13
    cepstral_lifter: float = 22.0
    use_energy: bool = False
    low_freq: float = 20.0
    high_freq: float = 0.0
    snip_edges: bool = True
    backend: str = "torchaudio"

    def __post_init__(self) -> None:
        if self.feature_type not in ("fbank", "mfcc"):
            raise ValueError(
                f"feature_type must be 'fbank' or 'mfcc', got {self.feature_type!r}"
            )
        if self.backend not in ("torchaudio", "kaldifeat"):
            raise ValueError(
                f"backend must be 'torchaudio' or 'kaldifeat', got {self.backend!r}"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.frame_length_ms <= 0 or self.frame_shift_ms <= 0:
            raise ValueError("frame_length_ms and frame_shift_ms must be positive")
        if self.frame_shift_ms > self.frame_length_ms:
            raise ValueError(
                f"frame_shift_ms ({self.frame_shift_ms}) must be <= "
                f"frame_length_ms ({self.frame_length_ms})"
            )

    @property
    def frame_length_samples(self) -> int:
        """Frame length in number of audio samples."""
        return int(self.sample_rate * self.frame_length_ms / 1000.0)

    @property
    def frame_shift_samples(self) -> int:
        """Frame shift (hop) in number of audio samples."""
        return int(self.sample_rate * self.frame_shift_ms / 1000.0)

    @property
    def output_dim(self) -> int:
        """Dimensionality of each output feature vector."""
        return self.num_ceps if self.feature_type == "mfcc" else self.num_mel_bins
