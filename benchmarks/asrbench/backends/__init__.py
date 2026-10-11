# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Backend registry.  Imported lazily: listing backends must not import torch.

Add a framework: subclass :class:`~benchmarks.asrbench.backends.base.Backend`,
declare its :class:`~benchmarks.asrbench.backends.base.Capabilities` honestly,
and add one line here.  ``docs/asrbench.md`` walks through it.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Type

from benchmarks.asrbench.backends.base import (
    STREAMING_MODES,
    Backend,
    BackendInfo,
    Capabilities,
    RequestFailed,
    StreamEvent,
    StreamSession,
    Transcript,
    Unsupported,
)

#: name -> "module:Class"
BACKENDS: Dict[str, str] = {
    "oasr_engine": "benchmarks.asrbench.backends.oasr_engine:OasrEngineBackend",
    "oasr_server": "benchmarks.asrbench.backends.openai_http:OasrServerBackend",
    "openai_http": "benchmarks.asrbench.backends.openai_http:OpenAIHttpBackend",
    "transformers": "benchmarks.asrbench.backends.transformers_asr:TransformersBackend",
    "mock": "benchmarks.asrbench.backends.mock:MockBackend",
}

__all__ = [
    "BACKENDS",
    "STREAMING_MODES",
    "Backend",
    "BackendInfo",
    "Capabilities",
    "RequestFailed",
    "StreamEvent",
    "StreamSession",
    "Transcript",
    "Unsupported",
    "backend_class",
    "create",
]


def backend_class(name: str) -> Type[Backend]:
    try:
        target = BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; registered: {sorted(BACKENDS)}") from None
    module, _, cls = target.partition(":")
    return getattr(importlib.import_module(module), cls)


def create(name: str, options: Optional[Dict[str, Any]] = None, mode: str = "offline") -> Backend:
    cls = backend_class(name)
    ok, why = cls.available()
    if not ok:
        raise RuntimeError(f"backend {name!r} is unavailable: {why}")
    return cls(options, mode)
