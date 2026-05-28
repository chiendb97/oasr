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

Quick start
-----------
Offline transcription::

    from oasr.engine import ASREngine, EngineConfig

    engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))
    text = engine.transcribe("audio.wav", streaming=False)

Streaming transcription (multiple concurrent requests)::

    from oasr.engine import ASREngine, EngineConfig

    engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))
    texts = engine.transcribe(["a.wav", "b.wav", "c.wav"])
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
