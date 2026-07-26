# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer registry: ``TokenizerSpec.kind`` → implementation factory.

Mirrors the other per-axis registries (:mod:`oasr.models.registry`,
``oasr.engine.decode``): implementations self-register at import time and
:func:`build_tokenizer` is the single generic entry point.  Adding a tokenizer
kind is a factory + one ``register_tokenizer`` call — no engine edits.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

from .base import Tokenizer, TokenizerSpec

logger = logging.getLogger(__name__)

TokenizerFactory = Callable[[TokenizerSpec], Tokenizer]

_REGISTRY: Dict[str, TokenizerFactory] = {}


def register_tokenizer(kind: str, factory: TokenizerFactory) -> None:
    """Register a tokenizer implementation under *kind* (last write wins)."""
    if kind in _REGISTRY:
        logger.debug("Overriding tokenizer registration for %r", kind)
    _REGISTRY[kind] = factory


def _ensure_builtins() -> None:
    # Import side effect runs each module's register_tokenizer call.  Lazy to
    # avoid import cycles (implementations import this module to register).
    if "symbol_table" not in _REGISTRY:
        import oasr.tokenizers.symbol_table  # noqa: F401
    if "sentencepiece" not in _REGISTRY:
        import oasr.tokenizers.sentencepiece  # noqa: F401
    if "huggingface" not in _REGISTRY:
        import oasr.tokenizers.huggingface  # noqa: F401
    if "whisper" not in _REGISTRY:
        import oasr.tokenizers.whisper  # noqa: F401
    if "funasr_char" not in _REGISTRY:
        import oasr.tokenizers.funasr_char  # noqa: F401


def list_tokenizers() -> List[str]:
    """Names of all registered tokenizer kinds."""
    _ensure_builtins()
    return sorted(_REGISTRY)


def build_tokenizer(spec: TokenizerSpec) -> Tokenizer:
    """Instantiate the tokenizer described by *spec*."""
    _ensure_builtins()
    try:
        factory = _REGISTRY[spec.kind]
    except KeyError:
        raise KeyError(
            f"Unknown tokenizer kind {spec.kind!r}; registered: {sorted(_REGISTRY)}"
        ) from None
    return factory(spec)
