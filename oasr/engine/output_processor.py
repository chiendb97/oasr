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

    def __init__(
        self, config: EngineConfig, decode_type: str = "ctc", model=None, tokenizer=None
    ) -> None:
        self._config = config
        self._decode_type = decode_type
        # ``tokenizer`` (an :class:`oasr.tokenizers.Tokenizer` built from the
        # checkpoint's converter-emitted TokenizerSpec) takes precedence; the
        # config paths are the legacy sniffed fallback.
        self._detok = Detokenizer(
            config.sentencepiece_model, config.unit_table, tokenizer=tokenizer
        )
        # ``model`` is threaded to the strategy so autoregressive families can
        # reach ``model.decoder`` / ``model.joiner`` (CTC strategies ignore it).
        self._strategy = build_decode_strategy(decode_type, config, self._detok, model)

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

    def fill_nbest_texts(self, request: Request, output: RequestOutput) -> None:
        """Detokenize the top-N hypotheses into ``output.nbest_texts``.

        Applies only when the request asked for ``DecodingOptions.n_best > 1``
        and the decode family produced multiple hypothesis rows (CTC / WFST
        beams); greedy families carry a single row and are left untouched.
        Called by the executors on **final** outputs — interim partials always
        carry the best hypothesis only.
        """
        opts = request.decoding
        n = int(opts.n_best) if opts is not None else 1
        if n <= 1 or not output.tokens:
            return
        if len(output.tokens) <= 1:
            # A greedy family produced one hypothesis and always will.  Say so
            # at DEBUG rather than silently returning `max_alternatives` worth
            # of nothing: a client asking for 5 alternatives from `aed` / `llm`
            # / `transducer` / `paraformer` cannot otherwise tell the request
            # was honoured-and-empty from unsupported.  Deliberately reads only
            # `output` — this method is a pure function of (request, output),
            # which is what lets it be tested without an engine.
            logger.debug(
                "n_best=%d requested but this decode family produced a single "
                "hypothesis; returning one alternative",
                n,
            )
            return
        texts = [output.text]
        for row in output.tokens[1:n]:
            texts.append(self._detok.detokenize(row))
        output.nbest_texts = texts
        # Truncate the token rows to what was asked for, before they cross the
        # PyO3 boundary.  CTC ships its **whole** beam (default 10-16 rows) for
        # every request regardless of `n_best`, and the serving layer discards
        # all but the first `n` — so the extra rows are pure marshalling cost on
        # the GIL-holding dispatcher thread.
        if len(output.tokens) > n:
            output.tokens = output.tokens[:n]
            if output.scores is not None and len(output.scores) > n:
                output.scores = output.scores[:n]
