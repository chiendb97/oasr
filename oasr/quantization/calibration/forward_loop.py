# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Calibration audio → the model, through the checkpoint's own frontend.

An algorithm receives a ``forward_loop(model)`` callable (ModelOpt's contract)
and never sees audio, manifests or features.  :func:`asr_forward_loop` builds one:

* **the serving frontend** — features come from the checkpoint's ``FeatureSpec``
  through :func:`oasr.features.build_extractor`, with its ``audio_scale``, exactly
  as :class:`oasr.engine.input_processor.InputProcessor` computes them.
  ``audio_scale`` differs per framework (WeNet 32768, icefall 1.0) and a wrong
  one degrades recognition silently, so calibration must not re-derive it;
* **no padding** — one utterance per forward, so no padded frame reaches an
  observer;
* **reproducible** — a deterministic selection (sorted by id, evenly strided),
  eager, TF32 off; :func:`select_manifest` returns the provenance recorded in the
  checkpoint.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

#: The engine's waveform scale when a checkpoint carries no ``FeatureSpec``
#: (``EngineConfig.audio_scale``'s default).
_DEFAULT_AUDIO_SCALE = 32768.0


@dataclass(frozen=True)
class CalibrationSet:
    """The utterances chosen from a manifest, and how."""

    entries: Tuple[Dict[str, Any], ...]
    audio_root: Path
    provenance: Dict[str, Any]


def select_manifest(
    manifest: Path, audio_root: Optional[Path] = None, n: int = 128
) -> CalibrationSet:
    """*n* entries of a ``{id, audio, ...}`` jsonl manifest, evenly spread.

    Sorted by ``id``, then every ``len // n``-th entry: on LibriSpeech dev-clean
    that touches every speaker rather than the first few chapters.
    """
    manifest = Path(manifest)
    raw = manifest.read_bytes()
    entries = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"{manifest} has no entries")
    entries.sort(key=lambda e: str(e.get("id", e["audio"])))
    n = min(n, len(entries))
    stride = max(1, len(entries) // n)
    chosen = tuple(entries[::stride][:n])
    return CalibrationSet(
        entries=chosen,
        audio_root=Path(audio_root) if audio_root is not None else manifest.parent,
        provenance={
            "manifest": manifest.name,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "utterances": len(chosen),
            "selection": f"sorted by id, stride {stride}",
        },
    )


def load_waveform(path: Path, expected_rate: int) -> torch.Tensor:
    """A mono float32 waveform at *expected_rate*; refuses another rate."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "calibration reads audio with soundfile; `pip install oasr[audio]`"
        ) from exc
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != expected_rate:
        raise ValueError(
            f"{path}: {rate} Hz audio for a {expected_rate} Hz model; calibrate on audio at "
            "the model's rate (resampling here would calibrate on a different signal than "
            "serving sees)"
        )
    return torch.from_numpy(data.mean(axis=1).copy())


class CheckpointFrontend:
    """The engine's offline feature path for one unpadded waveform.

    Mirrors ``InputProcessor`` (``_padded_waveform_batch`` then ``_fbank_batch``):
    scale by ``audio_scale``, the registered extractor, then LFR stacking when the
    config asks for it — in fp32.
    """

    def __init__(self, feature_spec: Any, device: torch.device) -> None:
        from oasr.features import FeatureConfig, build_extractor

        if feature_spec is not None:
            self.config = feature_spec.to_feature_config()
            self.audio_scale = float(feature_spec.audio_scale)
            self.sample_rate = int(feature_spec.sample_rate)
        else:
            self.config = FeatureConfig(dither=0.0)
            self.audio_scale = _DEFAULT_AUDIO_SCALE
            self.sample_rate = int(getattr(self.config, "sample_rate", 16000))
        self.device = torch.device(device)
        self.extractor = build_extractor(self.config)

    def __call__(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        from oasr.features.lfr import apply_lfr_batch

        wav = waveform.to(self.device, dtype=torch.float32).reshape(1, -1)
        if self.audio_scale != 1.0:
            wav = wav * self.audio_scale
        lengths = torch.tensor([wav.shape[1]], dtype=torch.int64, device=self.device)
        feats, feat_lengths = self.extractor(wav, lengths, self.config)
        if self.config.lfr_enabled:
            feats, feat_lengths = apply_lfr_batch(
                feats,
                feat_lengths,
                self.config.lfr_m,
                self.config.lfr_n,
                max_length=int(feats.size(1)),
            )
        return feats.float(), feat_lengths


@contextlib.contextmanager
def no_tf32() -> Iterator[None]:
    """TF32 off for the duration: calibration statistics must be reproducible."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def asr_forward_loop(
    calibration: CalibrationSet, frontend: CheckpointFrontend, tokenizer: Any = None
) -> Callable[[nn.Module], None]:
    """``forward_loop(model)``: every calibration utterance through the model.

    Through ``model.calibration_forward`` -- the encoder, and whatever decoder
    the model's quantization layout covers (driven as serving drives it; a
    text prompt is encoded with *tokenizer*, the checkpoint's) -- or, for a
    model without one, ``encode_offline``.  Waveforms are read once and kept,
    so the loop can run more than once.
    """
    waveforms: List[torch.Tensor] = [
        load_waveform(_resolve(calibration.audio_root, e["audio"]), frontend.sample_rate)
        for e in calibration.entries
    ]

    def forward_loop(model: nn.Module) -> None:
        from oasr.models.base import first_compute_parameter

        param = first_compute_parameter(model)
        # ``no_grad``, not ``inference_mode``: the frontend caches derived tables
        # (the mel matrix's spans) keyed on a version counter that inference
        # tensors do not have.
        with no_tf32(), torch.no_grad():
            calibrate = getattr(model, "calibration_forward", None)
            for wav in waveforms:
                feats, lengths = frontend(wav)
                feats = feats.to(device=param.device, dtype=param.dtype)
                if calibrate is not None:
                    calibrate(feats, lengths.to(param.device), tokenizer=tokenizer)
                else:
                    model.encode_offline(feats, lengths.to(param.device))  # type: ignore[operator]

    return forward_loop


def _resolve(root: Path, audio: str) -> Path:
    path = Path(audio)
    return path if path.is_absolute() else root / path


__all__ = [
    "CalibrationSet",
    "CheckpointFrontend",
    "asr_forward_loop",
    "load_waveform",
    "no_tf32",
    "select_manifest",
]
