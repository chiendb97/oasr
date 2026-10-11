# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Audio I/O for the suite: decode once, before any clock starts.

Every backend receives the same float32 mono waveform at the same sample rate,
decoded here.  Transport encodings (a WAV body for an HTTP server, a PCM16
frame for a WebSocket) are produced from that waveform by the backend's
``prepare`` hook, also before timing -- an HTTP client that encodes inside the
timed loop measures its own numpy work (the service harness once capped at
~320 req/s that way against a server sustaining several times that).
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np

#: The rate every prepared dataset is stored at and every shipped model expects.
DEFAULT_SAMPLE_RATE = 16000


def read_info(path: str | Path) -> tuple[int, int]:
    """``(frames, sample_rate)`` from the file header, without decoding."""
    import soundfile as sf

    info = sf.info(str(path))
    return int(info.frames), int(info.samplerate)


def load(
    path: str | Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    offset_s: float = 0.0,
    duration_s: Optional[float] = None,
) -> np.ndarray:
    """Decode ``[offset_s, offset_s + duration_s)`` of *path* to float32 mono.

    Multi-channel audio is averaged, never truncated to channel 0 -- one side of
    a stereo call is a different input.  Resampling happens here, once, so a
    backend never sees a rate it has to convert inside the timed region.
    """
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        sr = int(f.samplerate)
        start = int(round(offset_s * sr))
        frames = -1 if duration_s is None else int(round(duration_s * sr))
        if start:
            f.seek(start)
        data = f.read(frames=frames, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if sr != target_sr:
        mono = resample(mono, sr, target_sr)
    return mono


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resampling (scipy), falling back to torchaudio."""
    if src_sr == dst_sr:
        return x
    g = math.gcd(src_sr, dst_sr)
    try:
        from scipy.signal import resample_poly

        return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:  # pragma: no cover - scipy ships with the audio extra
        import torch
        import torchaudio

        out = torchaudio.functional.resample(torch.from_numpy(x), src_sr, dst_sr)
        return out.numpy().astype(np.float32)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    """Clip to ``[-1, 1]`` and scale to int16 (1.0 maps to 32767)."""
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def encode_wav(x: np.ndarray, sample_rate: int) -> bytes:
    """A complete PCM16 mono WAV file in memory (stdlib only)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(to_pcm16(x).tobytes())
    return buf.getvalue()


def write(path: str | Path, x: np.ndarray, sample_rate: int) -> None:
    """Write *x* as FLAC (``.flac``) or PCM16 WAV (anything else)."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "FLAC" if path.suffix.lower() == ".flac" else "WAV"
    sf.write(str(path), x, int(sample_rate), format=fmt, subtype="PCM_16")


def split_chunks(x: np.ndarray, chunk_samples: int) -> List[np.ndarray]:
    """Contiguous chunks of *chunk_samples*; the last one may be shorter."""
    if chunk_samples <= 0:
        raise ValueError(f"chunk_samples must be positive, got {chunk_samples}")
    return [np.ascontiguousarray(x[i : i + chunk_samples]) for i in range(0, len(x), chunk_samples)]
