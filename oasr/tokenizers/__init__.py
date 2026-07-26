# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer axis: ``TokenizerSpec`` (emitted by checkpoint converters) →
registered :class:`Tokenizer` implementations.

Implementations with optional dependencies (``sentencepiece``, ``tokenizers``)
import those lazily at construction, so importing this package is always safe.
"""

from .base import DEFAULT_SPECIAL_IDS, Tokenizer, TokenizerSpec
from .huggingface import HuggingFaceTokenizer
from .registry import build_tokenizer, list_tokenizers, register_tokenizer
from .sentencepiece import SentencePieceTokenizer
from .symbol_table import SymbolTableTokenizer, load_symbol_table

__all__ = [
    "DEFAULT_SPECIAL_IDS",
    "Tokenizer",
    "TokenizerSpec",
    "build_tokenizer",
    "list_tokenizers",
    "register_tokenizer",
    "SymbolTableTokenizer",
    "SentencePieceTokenizer",
    "HuggingFaceTokenizer",
    "load_symbol_table",
]
