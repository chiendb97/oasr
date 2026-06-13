# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Decode + detokenization facade for the ASR engine.

``OutputProcessor`` is a thin facade over a pluggable
:class:`~oasr.engine.decode.DecodeStrategy` (selected by the model's
``decode_type`` + ``config.decoder_type``) and a shared
:class:`~oasr.engine.decode.Detokenizer`.  The executors call the same four
methods regardless of decode family; the strategy owns the algorithm and any
per-request decode state.  Add a decode family by registering a new
``DecodeStrategy`` — no change here.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch

from .config import EngineConfig
from .decode import Detokenizer, build_decode_strategy
from .request import Request, RequestOutput

logger = logging.getLogger(__name__)


class OutputProcessor:
    """Routes encoder output to the active :class:`DecodeStrategy`.

    Parameters
    ----------
    config : EngineConfig
        Engine configuration.
    decode_type : str
        The model's decode family (``model.decode_type``); selects the strategy
        together with ``config.decoder_type`` (``"ctc_cuda"`` / ``"ctc_wfst"``
        for CTC).  Defaults to ``"ctc"``.
    """

    def __init__(self, config: EngineConfig, decode_type: str = "ctc") -> None:
        self._config = config
        self._decode_type = decode_type
        self._detok = Detokenizer(config.sentencepiece_model, config.unit_table)
        self._strategy = build_decode_strategy(decode_type, config, self._detok)

    @property
    def strategy(self):
        """The active :class:`~oasr.engine.decode.DecodeStrategy`."""
        return self._strategy

    # ------------------------------------------------------------------
    # Offline
    # ------------------------------------------------------------------

    def decode_offline(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> List[RequestOutput]:
        """Decode a batched encoder output → one final output per row."""
        return self._strategy.decode_offline(log_probs, lengths)

    # ------------------------------------------------------------------
    # Streaming session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, request: Request) -> None:
        """Allocate the decode strategy's per-request state on admission."""
        self._strategy.create_session(request)

    def free_session(self, request: Request) -> None:
        """Release the decode strategy's per-request state on finalize/abort."""
        self._strategy.free_session(request)

    # ------------------------------------------------------------------
    # Streaming decode
    # ------------------------------------------------------------------

    def decode_streaming_batch(
        self, requests: List[Request], log_probs_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        """Advance N ready streams one chunk; return any partial outputs."""
        return self._strategy.decode_streaming_batch(requests, log_probs_map)

    def decode_streaming_chunk(self, request: Request, log_probs: torch.Tensor) -> RequestOutput:
        """Advance a single stream one chunk; return a partial output."""
        return self._strategy.decode_streaming_chunk(request, log_probs)

    def finalize_streaming(self, request: Request) -> RequestOutput:
        """Finalize streaming decoding and return the complete transcript."""
        return self._strategy.finalize(request)

    # ------------------------------------------------------------------
    # Detokenization (kept on the facade for callers/tests)
    # ------------------------------------------------------------------

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to text (see :class:`Detokenizer`)."""
        return self._detok.detokenize(token_ids)
