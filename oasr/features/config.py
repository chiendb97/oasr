# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Feature extraction configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
    # Low-frame-rate stacking (FunASR/Paraformer): stack ``lfr_m`` consecutive
    # frames (replicating the first/last frame at the edges) and advance by
    # ``lfr_n`` — 80-mel LFR 7/6 yields 560-dim features at a 60 ms hop.
    # ``1/1`` disables.  Offline-only: the streaming feature path rejects it.
    lfr_m: int = 1
    lfr_n: int = 1
    # Whisper log-mel only (``feature_type="whisper_logmel"``): every
    # utterance is padded/trimmed to this many seconds (30 s → 3000 frames →
    # 1500 encoder positions) and globally max-normalized, per the Whisper
    # recipe.  Kaldi fields above are ignored except ``sample_rate`` /
    # ``num_mel_bins``; the STFT geometry is fixed (n_fft 400, hop 160).
    whisper_chunk_seconds: float = 30.0

    def __post_init__(self) -> None:
        # Validated against the extractor registry, so registering an out-of-tree
        # frontend makes its ``feature_type`` legal without editing this list.
        from .registry import list_extractors

        kinds = list_extractors()
        if self.feature_type not in kinds:
            raise ValueError(f"feature_type must be one of {kinds}, got {self.feature_type!r}")
        if self.backend not in ("torchaudio", "kaldifeat"):
            raise ValueError(f"backend must be 'torchaudio' or 'kaldifeat', got {self.backend!r}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.frame_length_ms <= 0 or self.frame_shift_ms <= 0:
            raise ValueError("frame_length_ms and frame_shift_ms must be positive")
        if self.frame_shift_ms > self.frame_length_ms:
            raise ValueError(
                f"frame_shift_ms ({self.frame_shift_ms}) must be <= "
                f"frame_length_ms ({self.frame_length_ms})"
            )
        if self.lfr_m < 1 or self.lfr_n < 1:
            raise ValueError(f"lfr_m/lfr_n must be >= 1, got {self.lfr_m}/{self.lfr_n}")

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
        """Dimensionality of each output feature vector (after LFR stacking)."""
        base = self.num_ceps if self.feature_type == "mfcc" else self.num_mel_bins
        return base * self.lfr_m

    @property
    def lfr_enabled(self) -> bool:
        """Whether low-frame-rate stacking is active."""
        return self.lfr_m != 1 or self.lfr_n != 1

    @property
    def fixed_window_seconds(self) -> Optional[float]:
        """Audio window this frontend pads/trims every utterance to, if any.

        Declared by the registered extractor
        (:attr:`~oasr.features.ExtractorSpec.window_seconds_attr`), not by a name
        check here.  ``None`` for the Kaldi frontends, whose cost tracks the real
        utterance length.  ``whisper_logmel`` resolves to :attr:`whisper_chunk_seconds`: every
        row is padded *and trimmed* to that window, so (a) audio beyond it would
        be silently dropped — the engine rejects it at admission instead — and
        (b) per-row encoder cost is **constant**, which the batching policies
        need to know rather than inferring cost from frame counts.
        """
        from .registry import build_extractor

        try:
            spec = build_extractor(self)
        except NotImplementedError:
            # An unregistered ``feature_type`` is caught where it matters (the
            # engine resolves an extractor at construction).  A bare config object
            # makes no window claim.
            return None
        if spec.window_seconds_attr is None:
            return None
        return float(getattr(self, spec.window_seconds_attr))

    @property
    def fixed_window_frames(self) -> Optional[int]:
        """:attr:`fixed_window_seconds` expressed in output feature frames."""
        secs = self.fixed_window_seconds
        if secs is None:
            return None
        return int(secs * 1000.0 / self.frame_shift_ms)
