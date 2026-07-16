# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Pluggable decode strategies for the ASR engine.

Importing this package registers every built-in strategy (CTC-GPU, CTC-WFST, and
the autoregressive skeletons) so :func:`build_decode_strategy` can resolve them.
Add a new decode family by subclassing :class:`DecodeStrategy` and decorating it
with :func:`register_decode_strategy`.
"""

# Import for side effects: each module registers its strategy on import.
from . import ctc_gpu, ctc_wfst, skeleton, transducer  # noqa: E402,F401
from .base import (
    DecodeStrategy,
    build_decode_strategy,
    get_decode_strategy_class,
    register_decode_strategy,
)
from .detokenize import Detokenizer

__all__ = [
    "DecodeStrategy",
    "Detokenizer",
    "build_decode_strategy",
    "get_decode_strategy_class",
    "register_decode_strategy",
]
