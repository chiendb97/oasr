# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Manifest loading and batched transcription for accuracy measurement.

Shared by ``benchmarks/bench_accuracy.py`` (the sweep CLI) and
``tests/test_accuracy.py`` (the regression gate) so the two cannot disagree
about what "run the manifest" means — a gate that measures something slightly
different from the benchmark is a gate nobody trusts.

A manifest is JSON Lines, one utterance per line::

    {"id": "LJ001-0001", "audio": "LJ001-0001.wav", "text": "printing in the ..."}

``audio`` is resolved against an audio root, so a manifest is a few tens of KB
of text that can be checked in while the corpus stays on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["Entry", "load_manifest", "load_audio", "transcribe"]


@dataclass
class Entry:
    """One manifest line."""

    uid: str
    audio: Path
    text: str


def load_manifest(
    path: Path,
    audio_root: Optional[Path] = None,
    limit: Optional[int] = None,
    *,
    check_audio: bool = True,
) -> List[Entry]:
    """Parse a manifest, resolving relative audio paths against *audio_root*.

    Raises rather than skipping when audio is missing: a manifest run that
    quietly drops half its utterances reports a WER for a different set than
    the one the reference was recorded on.  ``check_audio=False`` inspects the
    text side only, for callers validating the manifest without the corpus.
    """
    entries: List[Entry] = []
    missing: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON — {exc}") from None
            for key in ("audio", "text"):
                if key not in rec:
                    raise ValueError(f"{path}:{lineno}: manifest line has no {key!r} field")
            p = Path(rec["audio"])
            if not p.is_absolute() and audio_root is not None:
                p = audio_root / p
            if check_audio and not p.exists():
                missing.append(str(p))
            entries.append(Entry(uid=str(rec.get("id", p.stem)), audio=p, text=rec["text"]))
            if limit and len(entries) >= limit:
                break
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(entries)} manifest audio file(s) not found, e.g.\n"
            f"  {missing[0]}\n"
            f"Manifests ship without audio — point the audio root at the corpus."
        )
    return entries


def load_audio(entries: Sequence[Entry], target_sample_rate: int) -> Tuple[List, float]:
    """Load and resample the manifest's audio; returns ``(waveforms, seconds)``."""
    import torch
    import torchaudio

    waves, seconds = [], 0.0
    for e in entries:
        wav, sr = torchaudio.load(str(e.audio))
        wav = wav.mean(dim=0) if wav.dim() > 1 and wav.size(0) > 1 else wav.squeeze(0)
        if sr != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        waves.append(wav.to(torch.float32))
        seconds += wav.numel() / float(target_sample_rate)
    return waves, seconds


def transcribe(
    engine,
    waves: Sequence["torch.Tensor"],
    batch_size: int,
    *,
    warmup: bool = True,
    streaming: bool = False,
) -> Tuple[List[str], List[float]]:
    """Transcribe in batches; returns ``(hypotheses, per_utterance_ms)``.

    The warm-up call matters for the *speed* half of a row: JIT compilation and
    CUDA-graph capture happen on first use, so without it the first
    configuration of any sweep reports compile time as if it were inference.

    ``streaming`` drives ``engine.transcribe`` (chunk by chunk) instead of
    ``transcribe_offline``.  Same manifest, same denominator, so the two rates are
    directly comparable — which is what makes a streaming accuracy gate meaningful
    rather than a second number nobody can interpret.
    """
    import time

    go = engine.transcribe if streaming else engine.transcribe_offline
    if warmup and waves:
        go(list(waves[:1]))

    hyps: List[str] = []
    per_utt_ms: List[float] = []
    for i in range(0, len(waves), batch_size):
        chunk = list(waves[i : i + batch_size])
        t0 = time.perf_counter()
        out = go(chunk)
        dt_ms = 1000.0 * (time.perf_counter() - t0)
        texts = out if isinstance(out, list) else [out]
        hyps.extend(t if isinstance(t, str) else t.text for t in texts)
        # Batch latency spread across its rows: the per-request number a caller
        # of the batched API actually experiences.
        per_utt_ms.extend([dt_ms / len(chunk)] * len(chunk))
    return hyps, per_utt_ms
