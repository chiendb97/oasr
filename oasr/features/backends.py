# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Backend implementations, dispatch, and batched offline feature extraction."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from oasr.api_logging import oasr_api

from .config import FeatureConfig


def _torchaudio_fbank(waveform: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """FBANK via ``torchaudio.compliance.kaldi``.

    Input: ``(C, T)`` or ``(T,)``.  Output: ``(T', num_mel_bins)``.
    """
    import torchaudio

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return torchaudio.compliance.kaldi.fbank(
        waveform,
        sample_frequency=float(cfg.sample_rate),
        num_mel_bins=cfg.num_mel_bins,
        frame_length=cfg.frame_length_ms,
        frame_shift=cfg.frame_shift_ms,
        dither=cfg.dither,
        energy_floor=cfg.energy_floor,
        preemphasis_coefficient=cfg.preemphasis_coefficient,
        window_type=cfg.window_type,
        use_energy=cfg.use_energy,
        low_freq=cfg.low_freq,
        high_freq=cfg.high_freq,
        snip_edges=cfg.snip_edges,
    )


def _torchaudio_mfcc(waveform: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """MFCC via ``torchaudio.compliance.kaldi``.

    Input: ``(C, T)`` or ``(T,)``.  Output: ``(T', num_ceps)``.
    """
    import torchaudio

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return torchaudio.compliance.kaldi.mfcc(
        waveform,
        sample_frequency=float(cfg.sample_rate),
        num_mel_bins=cfg.num_mel_bins,
        num_ceps=cfg.num_ceps,
        cepstral_lifter=cfg.cepstral_lifter,
        frame_length=cfg.frame_length_ms,
        frame_shift=cfg.frame_shift_ms,
        dither=cfg.dither,
        energy_floor=cfg.energy_floor,
        preemphasis_coefficient=cfg.preemphasis_coefficient,
        window_type=cfg.window_type,
        use_energy=cfg.use_energy,
        low_freq=cfg.low_freq,
        high_freq=cfg.high_freq,
        snip_edges=cfg.snip_edges,
    )


def _kaldifeat_fbank_opts(cfg: FeatureConfig):
    """Build ``kaldifeat.FbankOptions`` from a :class:`FeatureConfig`."""
    import kaldifeat

    opts = kaldifeat.FbankOptions()
    opts.frame_opts.samp_freq = cfg.sample_rate
    opts.frame_opts.frame_length_ms = cfg.frame_length_ms
    opts.frame_opts.frame_shift_ms = cfg.frame_shift_ms
    opts.frame_opts.dither = cfg.dither
    opts.frame_opts.preemph_coeff = cfg.preemphasis_coefficient
    opts.frame_opts.window_type = cfg.window_type
    opts.frame_opts.snip_edges = cfg.snip_edges
    opts.mel_opts.num_bins = cfg.num_mel_bins
    opts.mel_opts.low_freq = cfg.low_freq
    opts.mel_opts.high_freq = cfg.high_freq
    opts.energy_floor = cfg.energy_floor
    opts.use_energy = cfg.use_energy
    return opts


def _kaldifeat_mfcc_opts(cfg: FeatureConfig):
    """Build ``kaldifeat.MfccOptions`` from a :class:`FeatureConfig`."""
    import kaldifeat

    opts = kaldifeat.MfccOptions()
    opts.frame_opts.samp_freq = cfg.sample_rate
    opts.frame_opts.frame_length_ms = cfg.frame_length_ms
    opts.frame_opts.frame_shift_ms = cfg.frame_shift_ms
    opts.frame_opts.dither = cfg.dither
    opts.frame_opts.preemph_coeff = cfg.preemphasis_coefficient
    opts.frame_opts.window_type = cfg.window_type
    opts.frame_opts.snip_edges = cfg.snip_edges
    opts.mel_opts.num_bins = cfg.num_mel_bins
    opts.mel_opts.low_freq = cfg.low_freq
    opts.mel_opts.high_freq = cfg.high_freq
    opts.energy_floor = cfg.energy_floor
    opts.use_energy = cfg.use_energy
    opts.num_ceps = cfg.num_ceps
    opts.cepstral_lifter = cfg.cepstral_lifter
    return opts


def _kaldifeat_fbank(waveform: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """FBANK via ``kaldifeat.Fbank`` (GPU-capable).

    Input: ``(C, T)`` or ``(T,)``.  Output: ``(T', num_mel_bins)``.
    """
    import kaldifeat

    opts = _kaldifeat_fbank_opts(cfg)
    opts.device = waveform.device
    fbank_computer = kaldifeat.Fbank(opts)
    wav = waveform.squeeze(0) if waveform.dim() == 2 else waveform
    return fbank_computer([wav.to(torch.float32)])[0]


def _kaldifeat_mfcc(waveform: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """MFCC via ``kaldifeat.Mfcc`` (GPU-capable).

    Input: ``(C, T)`` or ``(T,)``.  Output: ``(T', num_ceps)``.
    """
    import kaldifeat

    opts = _kaldifeat_mfcc_opts(cfg)
    opts.device = waveform.device
    mfcc_computer = kaldifeat.Mfcc(opts)
    wav = waveform.squeeze(0) if waveform.dim() == 2 else waveform
    return mfcc_computer([wav.to(torch.float32)])[0]


_BACKENDS = {
    ("torchaudio", "fbank"): _torchaudio_fbank,
    ("torchaudio", "mfcc"): _torchaudio_mfcc,
    ("kaldifeat", "fbank"): _kaldifeat_fbank,
    ("kaldifeat", "mfcc"): _kaldifeat_mfcc,
}


def _extract(waveform: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """Dispatch feature extraction to the selected backend."""
    key = (cfg.backend, cfg.feature_type)
    fn = _BACKENDS.get(key)
    if fn is None:
        raise ValueError(
            f"Unsupported combination: backend={cfg.backend!r}, "
            f"feature_type={cfg.feature_type!r}. "
            f"Supported: {list(_BACKENDS.keys())}"
        )
    return fn(waveform, cfg)


# ---------------------------------------------------------------------------
# Batched offline extraction
# ---------------------------------------------------------------------------


def _to_wav_list(
    waveforms: Union[torch.Tensor, List[torch.Tensor]],
    lengths: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Normalise batched waveform input into a list of 1-D tensors."""
    if isinstance(waveforms, (list, tuple)):
        return [w.squeeze(0) if w.dim() == 2 else w for w in waveforms]
    if waveforms.dim() == 1:
        return [waveforms]
    if waveforms.dim() == 2:
        B = waveforms.size(0)
        if lengths is not None:
            return [waveforms[i, : int(lengths[i].item())] for i in range(B)]
        return [waveforms[i] for i in range(B)]
    raise ValueError(
        f"waveforms must be 1-D, 2-D or a list of tensors, got shape {waveforms.shape}"
    )


def _extract_batch_kaldifeat(
    wavs: List[torch.Tensor], cfg: FeatureConfig
) -> List[torch.Tensor]:
    """Native batched extraction via ``kaldifeat``."""
    import kaldifeat

    device = wavs[0].device
    float_wavs = [w.to(torch.float32) for w in wavs]

    if cfg.feature_type == "fbank":
        opts = _kaldifeat_fbank_opts(cfg)
        opts.device = device
        computer = kaldifeat.Fbank(opts)
    else:
        opts = _kaldifeat_mfcc_opts(cfg)
        opts.device = device
        computer = kaldifeat.Mfcc(opts)

    return computer(float_wavs)


def _extract_batch(
    waveforms: Union[torch.Tensor, List[torch.Tensor]],
    cfg: FeatureConfig,
    lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features for a batch and return a padded tensor + lengths."""
    wavs = _to_wav_list(waveforms, lengths)
    if not wavs:
        return torch.empty(0, 0, cfg.output_dim), torch.zeros(0, dtype=torch.long)

    if cfg.backend == "kaldifeat":
        feat_list = _extract_batch_kaldifeat(wavs, cfg)
    else:
        feat_list = [_extract(w, cfg) for w in wavs]

    feat_lengths = torch.tensor(
        [f.size(0) for f in feat_list], dtype=torch.long
    )
    max_frames = int(feat_lengths.max().item())
    B = len(feat_list)
    F = cfg.output_dim
    ref = feat_list[0]
    padded = torch.zeros(B, max_frames, F, dtype=ref.dtype, device=ref.device)
    for i, f in enumerate(feat_list):
        padded[i, : f.size(0)] = f
    return padded, feat_lengths


@oasr_api
def fbank_batch(
    waveforms: Union[torch.Tensor, List[torch.Tensor]],
    lengths: Optional[torch.Tensor] = None,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    dither: float = 0.0,
    energy_floor: float = 0.0,
    backend: str = "torchaudio",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract FBANK features for a batch of waveforms.

    Parameters
    ----------
    waveforms : Tensor or list of Tensor
        Either a padded batch tensor of shape ``(B, T)`` or a list of
        1-D tensors with potentially different lengths.
    lengths : Tensor, optional
        Actual sample counts per utterance, shape ``(B,)``.
        Required when *waveforms* is a padded ``(B, T)`` tensor with
        variable-length items.  Ignored for list input.
    sample_rate, num_mel_bins, frame_length_ms, frame_shift_ms,
    dither, energy_floor, backend
        Kaldi-compatible parameters (see :class:`FeatureConfig`).

    Returns
    -------
    features : Tensor
        Padded feature tensor of shape ``(B, max_frames, num_mel_bins)``.
    feature_lengths : Tensor
        Number of valid frames per utterance, shape ``(B,)``.

    Examples
    --------
    >>> wavs = torch.randn(4, 16000)  # batch of 4 one-second clips
    >>> feats, feat_lens = oasr.fbank_batch(wavs)
    >>> feats.shape
    torch.Size([4, 98, 80])
    """
    cfg = FeatureConfig(
        feature_type="fbank",
        sample_rate=sample_rate,
        num_mel_bins=num_mel_bins,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        dither=dither,
        energy_floor=energy_floor,
        backend=backend,
    )
    return _extract_batch(waveforms, cfg, lengths)


@oasr_api
def mfcc_batch(
    waveforms: Union[torch.Tensor, List[torch.Tensor]],
    lengths: Optional[torch.Tensor] = None,
    sample_rate: int = 16000,
    num_ceps: int = 13,
    num_mel_bins: int = 23,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    dither: float = 0.0,
    energy_floor: float = 0.0,
    backend: str = "torchaudio",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract MFCC features for a batch of waveforms.

    Parameters
    ----------
    waveforms : Tensor or list of Tensor
        Either a padded batch tensor of shape ``(B, T)`` or a list of
        1-D tensors with potentially different lengths.
    lengths : Tensor, optional
        Actual sample counts per utterance, shape ``(B,)``.
    sample_rate, num_ceps, num_mel_bins, frame_length_ms, frame_shift_ms,
    dither, energy_floor, backend
        Kaldi-compatible parameters (see :class:`FeatureConfig`).

    Returns
    -------
    features : Tensor
        Padded feature tensor of shape ``(B, max_frames, num_ceps)``.
    feature_lengths : Tensor
        Number of valid frames per utterance, shape ``(B,)``.
    """
    cfg = FeatureConfig(
        feature_type="mfcc",
        sample_rate=sample_rate,
        num_ceps=num_ceps,
        num_mel_bins=num_mel_bins,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        dither=dither,
        energy_floor=energy_floor,
        backend=backend,
    )
    return _extract_batch(waveforms, cfg, lengths)


@oasr_api
def extract_features_batch(
    waveforms: Union[torch.Tensor, List[torch.Tensor]],
    config: FeatureConfig,
    lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features for a batch using a :class:`FeatureConfig`.

    Parameters
    ----------
    waveforms : Tensor or list of Tensor
        Padded ``(B, T)`` tensor or list of 1-D tensors.
    config : FeatureConfig
        Complete feature extraction configuration.
    lengths : Tensor, optional
        Actual sample counts per utterance, shape ``(B,)``.

    Returns
    -------
    features : Tensor
        ``(B, max_frames, config.output_dim)`` padded feature tensor.
    feature_lengths : Tensor
        ``(B,)`` valid frame counts.
    """
    return _extract_batch(waveforms, config, lengths)
