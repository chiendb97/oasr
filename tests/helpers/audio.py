# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Waveforms for tests: synthetic signals and the wav corpus.

Five modules each grew their own "read a wav off ``$WAV_DIR``" helper, and they
did not agree -- some averaged a stereo file to mono, some squeezed channel 0
and would have handed the engine one side of a stereo recording.  The engine is
waveform-only, so tests decode here exactly as ``oasr-asr`` and the bench
harness do, and never hand a path to the engine.
"""

from __future__ import annotations

import glob
import math
import os
from typing import List, Tuple

import torch

SR = 16000

__all__ = [
    "SR",
    "hiss",
    "speech_corpus",
    "tone",
    "wav_path",
    "wav_paths",
    "waveform",
    "waveforms",
]


def tone(seconds: float, amp: float = 0.3, freq: float = 220.0, sr: int = SR) -> torch.Tensor:
    """Amplitude-modulated harmonic tone — the "speech" side of a VAD fixture."""
    n = int(sr * seconds)
    t = torch.arange(n, dtype=torch.float32) / sr
    env = 0.5 + 0.5 * torch.sin(2 * math.pi * 4 * t)
    return (
        amp
        * env
        * (torch.sin(2 * math.pi * freq * t) + 0.5 * torch.sin(2 * math.pi * 3 * freq * t))
    )


def hiss(seconds: float, amp: float = 1e-4, seed: int = 7, sr: int = SR) -> torch.Tensor:
    """Near-silence — the "no speech" side. Seeded, so a boundary is reproducible."""
    g = torch.Generator().manual_seed(seed)
    return amp * torch.randn(int(sr * seconds), generator=g)


def wav_paths(wav_dir: str, need: int = 0) -> List[str]:
    """Every .wav in ``wav_dir``, sorted, so an index means the same file twice.

    ``need`` gates on the count through ``tests/assets.py``. ``wav_dir`` is the
    ``WAV_DIR`` asset at every caller, and a directory with too few clips is an
    incomplete asset: the test is counted and fails under ``--strict-assets``,
    where a hand-written ``pytest.skip`` was neither (audit L12).
    """
    paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if need:
        import assets

        assets.require_contents(
            "WAV_DIR",
            len(paths) >= need,
            f"need {need} .wav file(s) under {wav_dir}, found {len(paths)}",
        )
    return paths


def wav_path(wav_dir: str, n: int = 0) -> str:
    return wav_paths(wav_dir, need=n + 1)[n]


def waveform(wav_dir: str, n: int = 0) -> torch.Tensor:
    """One wav as a 1-D float32 CPU waveform, mixed down if it is not mono."""
    import torchaudio

    wav, _sr = torchaudio.load(wav_path(wav_dir, n))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).float()


def waveforms(wav_dir: str, count: int) -> List[torch.Tensor]:
    wav_paths(wav_dir, need=count)  # gate once, before any is read
    return [waveform(wav_dir, i) for i in range(count)]


def speech_corpus(
    wav_dir: str, gap_s: float = 3.0, n: int = 3
) -> Tuple[torch.Tensor, float, List[Tuple[float, float]]]:
    """``n`` utterances separated by ``gap_s`` of digital silence.

    Returns the concatenated 1-D waveform, its duration in seconds, and each
    utterance's ``(start, end)`` span -- the spans a segmenter is supposed to
    recover.  This existed twice, once with the spans and once without, which
    is two definitions of the corpus a VAD is scored against.
    """
    import numpy as np
    import soundfile as sf

    paths = wav_paths(wav_dir, need=n)[:n]
    parts: List = []
    spans: List[Tuple[float, float]] = []
    t = 0.0
    for i, path in enumerate(paths):
        data, rate = sf.read(str(path), dtype="float32")
        assert rate == SR, f"{path}: expected {SR} Hz, got {rate}"
        if i:
            parts.append(np.zeros(int(gap_s * SR), dtype="float32"))
            t += gap_s
        spans.append((t, t + len(data) / SR))
        parts.append(data)
        t += len(data) / SR
    return torch.from_numpy(np.concatenate(parts)), t, spans
