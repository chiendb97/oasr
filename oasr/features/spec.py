# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-derived feature-extraction spec.

A :class:`FeatureSpec` is emitted by the checkpoint converter (from
``train.yaml`` / recipe defaults / ``preprocessor_config.json``) and travels
with the :class:`~oasr.checkpoints.ConvertedCheckpoint` bundle, so the engine
extracts the features the checkpoint was trained with instead of applying an
engine-side default.  An explicit ``EngineConfig.feature_config`` still wins,
but a spec-vs-override mismatch logs loudly.

``kind`` keys the extractor registry (:mod:`oasr.features.registry`):
``"kaldi_fbank"`` / ``"kaldi_mfcc"`` map onto the Kaldi backends,
``"whisper_logmel"`` onto the fixed-window Whisper recipe, ``"nemotron_logmel"``
onto the NeMo pre-emphasis + natural-log mel recipe; ``"raw"`` lands with its
model package.  ``lfr_m`` / ``lfr_n`` describe low-frame-rate stacking
(Paraformer consumes 80×7 = 560-dim LFR features); ``1/1`` means off.

Not every field applies to every kind — the ``whisper_logmel`` recipe fixes its
own frame geometry — so :meth:`FeatureSpec.mismatches` reports a spec that asks
for something a kind cannot honour rather than dropping the request silently.
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
    # Kaldi analysis window ("povey" default; FunASR frontends use "hamming").
    window_type: str = "povey"
    # Feature normalization applied model-side (e.g. "global_cmvn"); None = none.
    normalize: Optional[str] = None
    # Waveform scale the checkpoint expects before feature extraction. This is a
    # per-*framework* convention, not a global default: WeNet trains on
    # int16-scale audio (32768.0) while icefall/lhotse feed the [-1, 1] float
    # straight through (1.0). A wrong value is not noisy -- it offsets every
    # log-mel bin by a constant and costs the transcript's leading token.
    audio_scale: float = 32768.0
    # Fixed frontend window; admission and batching use the same declared value.
    window_seconds: Optional[float] = None
    # Optional signal-domain pre-emphasis; ``None`` keeps the frontend default.
    preemphasis: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSpec":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    #: Spec fields the ``whisper_logmel`` recipe fixes internally and therefore
    #: cannot honour, mapped to the value that means "not requested". A converter
    #: setting one of these is stating something that would be silently dropped,
    #: so :meth:`mismatches` reports it rather than ignoring it.
    #: Spec fields the ``nemotron_logmel`` recipe fixes internally (a non-periodic
    #: Hann window, no dither, no LFR), mapped to the value that means "not
    #: requested".  Unlike Whisper's, the *frame geometry* is honoured.
    _NEMOTRON_FIXED_FIELDS = {
        "dither": 0.0,
        "window_type": "povey",  # the recipe uses a non-periodic Hann window
        "lfr_m": 1,
        "lfr_n": 1,
    }

    _WHISPER_FIXED_FIELDS = {
        "frame_length_ms": 25.0,  # n_fft 400 @ 16 kHz
        "frame_shift_ms": 10.0,  # hop 160 @ 16 kHz
        "dither": 0.0,
        "window_type": "povey",  # the recipe uses a Hann window
        "lfr_m": 1,
        "lfr_n": 1,
    }

    def to_feature_config(self) -> FeatureConfig:
        """Materialize a :class:`FeatureConfig` for the supported kinds."""
        if self.kind == "nemotron_logmel":
            cfg = FeatureConfig(
                feature_type="nemotron_logmel",
                sample_rate=self.sample_rate,
                num_mel_bins=self.feature_dim,
                frame_length_ms=self.frame_length_ms,
                frame_shift_ms=self.frame_shift_ms,
            )
            # Not fixed-window, so the frame geometry above *is* honoured (the
            # extractor derives n_fft from ``frame_length_ms``).  Only the
            # Hann window and the ``log(x + 2**-24)`` guard are fixed.
            if self.preemphasis is not None:
                cfg.preemphasis_coefficient = self.preemphasis
            return cfg
        if self.kind == "whisper_logmel":
            cfg = FeatureConfig(
                feature_type="whisper_logmel",
                sample_rate=self.sample_rate,
                num_mel_bins=self.feature_dim,
            )
            # The Whisper recipe hardcodes its frame geometry (n_fft 400 / hop
            # 160 / Hann / slaney mels), so only the three fields
            # ``oasr.features.whisper`` actually reads are carried across:
            # sample_rate, num_mel_bins and the window. Copying the rest would
            # put numbers on the config that the extractor ignores.
            if self.window_seconds is not None:
                cfg.whisper_chunk_seconds = self.window_seconds
            return cfg
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
            window_type=self.window_type,
            lfr_m=self.lfr_m,
            lfr_n=self.lfr_n,
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
        if self.kind == "nemotron_logmel":
            diffs = []
            pairs = [
                ("feature_type", "nemotron_logmel", config.feature_type),
                ("sample_rate", self.sample_rate, config.sample_rate),
                ("feature_dim", self.feature_dim, config.num_mel_bins),
                ("frame_length_ms", self.frame_length_ms, config.frame_length_ms),
                ("frame_shift_ms", self.frame_shift_ms, config.frame_shift_ms),
            ]
            if self.preemphasis is not None:
                pairs.append(("preemphasis", self.preemphasis, config.preemphasis_coefficient))
            for name, spec_v, cfg_v in pairs:
                if spec_v != cfg_v:
                    diffs.append(f"{name}: spec={spec_v!r} config={cfg_v!r}")
            for name, inert in self._NEMOTRON_FIXED_FIELDS.items():
                asked = getattr(self, name)
                if asked != inert:
                    diffs.append(
                        f"{name}: spec={asked!r} but the nemotron_logmel recipe "
                        f"fixes it at {inert!r} and cannot honour the request"
                    )
            return diffs
        if self.kind == "whisper_logmel":
            diffs = []
            pairs = [
                ("feature_type", "whisper_logmel", config.feature_type),
                ("sample_rate", self.sample_rate, config.sample_rate),
                ("feature_dim", self.feature_dim, config.num_mel_bins),
            ]
            # The window is honoured (oasr/features/whisper.py reads
            # ``whisper_chunk_seconds``), so a disagreement is a real one. Only
            # compare when the spec pins it; ``None`` means "the frontend's
            # default", which any config value satisfies.
            if self.window_seconds is not None:
                pairs.append(("window_seconds", self.window_seconds, config.whisper_chunk_seconds))
            for name, spec_v, cfg_v in pairs:
                if spec_v != cfg_v:
                    diffs.append(f"{name}: spec={spec_v!r} config={cfg_v!r}")
            # Fields the recipe fixes internally: report a spec that asks for
            # something else, instead of dropping the request silently.
            for name, inert in self._WHISPER_FIXED_FIELDS.items():
                asked = getattr(self, name)
                if asked != inert:
                    diffs.append(
                        f"{name}: spec={asked!r} but the whisper_logmel recipe "
                        f"fixes it at {inert!r} and cannot honour the request"
                    )
            return diffs
        if self.kind not in _KALDI_KINDS:
            return [f"kind: spec={self.kind!r} config=kaldi ({config.feature_type!r})"]
        diffs = []
        # ``feature_dim`` is the pre-LFR dimension (mel bins / cepstra);
        # ``config.output_dim`` already folds the LFR stacking in.
        base_dim = config.num_ceps if self.kind == "kaldi_mfcc" else config.num_mel_bins
        pairs = [
            ("feature_type", _KALDI_KINDS[self.kind], config.feature_type),
            ("sample_rate", self.sample_rate, config.sample_rate),
            ("feature_dim", self.feature_dim, base_dim),
            ("frame_length_ms", self.frame_length_ms, config.frame_length_ms),
            ("frame_shift_ms", self.frame_shift_ms, config.frame_shift_ms),
            ("dither", self.dither, config.dither),
            ("window_type", self.window_type, config.window_type),
            ("lfr_m", self.lfr_m, config.lfr_m),
            ("lfr_n", self.lfr_n, config.lfr_n),
        ]
        for name, spec_v, cfg_v in pairs:
            if spec_v != cfg_v:
                diffs.append(f"{name}: spec={spec_v!r} config={cfg_v!r}")
        return diffs
