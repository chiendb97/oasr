# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR inference and serving engine.

Provides a modular, vLLM-inspired engine for offline and streaming Conformer-CTC
inference on a single GPU with paged attention memory management.

Components
----------
:class:`EngineConfig`
    Unified configuration aggregating model, cache, feature extraction, decoding,
    and detokenization settings.
:class:`ASREngine`
    Unified engine for both streaming and offline transcription.  Pass
    ``streaming=False`` to :meth:`add_request` or :meth:`transcribe` for
    batched offline behaviour.
:class:`Request`
    A single ASR inference request.
:class:`RequestOutput`
    Transcription result for a single request.
:class:`RequestState`
    Lifecycle state enum: WAITING → RUNNING → FINISHED.

The engine is **waveform-only** — decode audio files at the entry point (the
serving front-end, or the harness) and pass waveforms in.

Quick start
-----------
Offline transcription::

    import torchaudio
    from oasr.engine import ASREngine, EngineConfig

    engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))
    wav, _sr = torchaudio.load("audio.wav")
    text = engine.transcribe(wav.squeeze(0), streaming=False)

Streaming transcription (multiple concurrent requests)::

    import torchaudio
    from oasr.engine import ASREngine, EngineConfig

    engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))
    wavs = [torchaudio.load(p)[0].squeeze(0) for p in ("a.wav", "b.wav", "c.wav")]
    texts = engine.transcribe(wavs)
"""

from .config import EngineConfig
from .engine import ASREngine
from .request import Request, RequestOutput, RequestState

__all__ = [
    "EngineConfig",
    "ASREngine",
    "Request",
    "RequestOutput",
    "RequestState",
]
