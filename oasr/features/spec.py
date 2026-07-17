# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-derived feature-extraction spec.

A :class:`FeatureSpec` is emitted by the checkpoint converter (from
``train.yaml`` / recipe defaults / ``preprocessor_config.json``) and travels
with the :class:`~oasr.checkpoints.ConvertedCheckpoint` bundle, so the engine
extracts the features the checkpoint was trained with instead of applying an
engine-side default.  An explicit ``EngineConfig.feature_config`` still wins,
but a spec-vs-override mismatch logs loudly.

``kind`` keys the (future) extractor registry: ``"kaldi_fbank"`` /
``"kaldi_mfcc"`` map onto today's :class:`FeatureConfig` backends;
``"whisper_logmel"`` and ``"raw"`` land with their model packages.  ``lfr_m`` /
``lfr_n`` describe low-frame-rate stacking (Paraformer consumes 80×7 = 560-dim
LFR features); ``1/1`` means off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .config import FeatureConfig

_KALDI_KINDS = {"kaldi_fbank": "fbank", "kaldi_mfcc": "mfcc"}


@dataclass
class FeatureSpec:
    kind: str = "kaldi_fbank"
    sample_rate: int = 16000
    feature_dim: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    # Dither is a training-time augmentation; converters emit 0.0 so inference
    # is deterministic regardless of the training config's value.
    dither: float = 0.0
    # Low-frame-rate stacking: stack lfr_m frames, advance lfr_n (1/1 = off).
    lfr_m: int = 1
    lfr_n: int = 1
    # Feature normalization applied model-side (e.g. "global_cmvn"); None = none.
    normalize: Optional[str] = None
    # Waveform scale the checkpoint expects before feature extraction (Kaldi
    # checkpoints are trained on int16-scale audio → 32768.0 for [-1, 1] input).
    audio_scale: float = 32768.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_feature_config(self) -> FeatureConfig:
        """Materialize a :class:`FeatureConfig` for the supported kinds."""
        if self.kind == "whisper_logmel":
            return FeatureConfig(
                feature_type="whisper_logmel",
                sample_rate=self.sample_rate,
                num_mel_bins=self.feature_dim,
            )
        if self.kind not in _KALDI_KINDS:
            raise ValueError(
                f"FeatureSpec kind {self.kind!r} has no FeatureConfig mapping; "
                f"supported: {sorted(_KALDI_KINDS) + ['whisper_logmel']}"
            )
        cfg = FeatureConfig(
            feature_type=_KALDI_KINDS[self.kind],
            sample_rate=self.sample_rate,
            frame_length_ms=self.frame_length_ms,
            frame_shift_ms=self.frame_shift_ms,
            dither=self.dither,
        )
        if self.kind == "kaldi_mfcc":
            cfg.num_ceps = self.feature_dim
        else:
            cfg.num_mel_bins = self.feature_dim
        return cfg

    def mismatches(self, config: FeatureConfig) -> List[str]:
        """Fields where an explicit :class:`FeatureConfig` disagrees with this spec.

        Only the fields the spec pins are compared; returns human-readable
        ``"name: spec=... config=..."`` strings (empty list = compatible).
        """
        if self.kind == "whisper_logmel":
            diffs = []
            for name, spec_v, cfg_v in [
                ("feature_type", "whisper_logmel", config.feature_type),
                ("sample_rate", self.sample_rate, config.sample_rate),
                ("feature_dim", self.feature_dim, config.num_mel_bins),
            ]:
                if spec_v != cfg_v:
                    diffs.append(f"{name}: spec={spec_v!r} config={cfg_v!r}")
            return diffs
        if self.kind not in _KALDI_KINDS:
            return [f"kind: spec={self.kind!r} config=kaldi ({config.feature_type!r})"]
        diffs = []
        pairs = [
            ("feature_type", _KALDI_KINDS[self.kind], config.feature_type),
            ("sample_rate", self.sample_rate, config.sample_rate),
            ("feature_dim", self.feature_dim, config.output_dim),
            ("frame_length_ms", self.frame_length_ms, config.frame_length_ms),
            ("frame_shift_ms", self.frame_shift_ms, config.frame_shift_ms),
            ("dither", self.dither, config.dither),
        ]
        for name, spec_v, cfg_v in pairs:
            if spec_v != cfg_v:
                diffs.append(f"{name}: spec={spec_v!r} config={cfg_v!r}")
        return diffs
