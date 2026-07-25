# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""WFST CTC beam-search decode strategy (in-tree GPU decoder or k2).

Wraps :class:`oasr.decode.Decoder`.  Streaming keeps a per-request decoder on
the request object (lazily created on first chunk), so the session lifecycle
methods stay no-ops (inherited from the base).

Unlike the CTC strategies, WFST decoding emits WORD ids in the decoding
graph's ``words.txt`` symbol space — not BPE unit ids — so text comes from the
word table found next to the FST (the standard k2 ``lang_*/{HLG.pt,words.txt}``
layout), joined with spaces.  The shared unit-table detokenizer is only a
fallback when no word table exists.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional

import torch

from oasr.decode import Decoder, DecoderResult

from ..request import Request, RequestOutput
from .base import DecodeStrategy, register_decode_strategy

if TYPE_CHECKING:
    from ..config import EngineConfig
    from .detokenize import Detokenizer

logger = logging.getLogger(__name__)


@register_decode_strategy("ctc_wfst")
class CtcWfstDecodeStrategy(DecodeStrategy):
    """CTC decoding via a k2 WFST beam search (GPU; requires a k2 build)."""

    decode_type: ClassVar[str] = "ctc"
    consumes: ClassVar[str] = "log_probs"

    def __init__(self, config: "EngineConfig", detok: "Detokenizer", model=None) -> None:
        super().__init__(config, detok, model)
        # Streaming decoder sizing (GPU backend): every concurrent stream borrows a
        # channel from one shared multi-channel decoder, so the pool must cover the
        # engine's concurrent stream cap. Each channel's winners ring commits only
        # while the channel is open; one 32 MiB mapping chunk (4Mi entries) per
        # channel is ample — the per-chunk GC keeps the live window at ~one chunk.
        cfg = config.wfst_decoder_config
        max_bs = int(getattr(config, "max_batch_size", 0) or 0)
        if cfg is not None and getattr(cfg, "wfst_backend", "gpu").lower() == "gpu":
            streams = max(max_bs, cfg.wfst_max_streams)
            log_entries = cfg.wfst_stream_log_entries or (4 << 20)
            if (streams, log_entries) != (cfg.wfst_max_streams, cfg.wfst_stream_log_entries):
                cfg = replace(cfg, wfst_max_streams=streams, wfst_stream_log_entries=log_entries)
        self._stream_cfg = cfg
        self._words = self._load_word_table(getattr(config, "fst_path", None))

    @staticmethod
    def _load_word_table(fst_path: Optional[str]) -> Optional[Dict[int, str]]:
        """``words.txt`` beside the FST ("WORD id" per line), or None."""
        if not fst_path:
            return None
        path = os.path.join(os.path.dirname(os.path.abspath(fst_path)), "words.txt")
        if not os.path.exists(path):
            logger.warning(
                "no words.txt next to %s — falling back to the unit-table "
                "detokenizer, which does NOT match WFST word ids",
                fst_path,
            )
            return None
        table: Dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    table[int(parts[1])] = parts[0]
        return table

    def _to_text(self, word_ids: List[int]) -> str:
        if self._words is None:
            return self._detok.detokenize(word_ids)
        return " ".join(self._words[t] for t in word_ids if t in self._words)

    # ------------------------------------------------------------------
    # Offline
    # ------------------------------------------------------------------

    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        cfg = self._config.wfst_decoder_config
        # Size the GPU offline decoder's lane pool to the engine's batch width so the
        # whole batch decodes in one GPU launch — batched throughput is the headline
        # perf lever (B=1: 1560x vs B=32: 5964x on the reference stack). No-op for the
        # k2 backend, which decodes one utterance per call.
        if getattr(cfg, "wfst_backend", "gpu").lower() == "gpu":
            lanes = max(int(self._config.max_batch_size), cfg.wfst_max_offline_lanes)
            if lanes != cfg.wfst_max_offline_lanes:
                cfg = replace(cfg, wfst_max_offline_lanes=lanes)
        decoder = Decoder(cfg, fst=self._config.fst_path)

        results: List[DecoderResult] = decoder.decode_batch(enc_out, enc_lengths)
        outputs = []
        for result in results:
            best = result.tokens[0] if result.tokens else []
            text = self._to_text(best)
            outputs.append(
                RequestOutput(
                    request_id="",
                    text=text,
                    tokens=result.tokens,
                    scores=result.scores,
                    finished=True,
                )
            )
        return outputs

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        # k2 is single-threaded per request; loop per stream.
        outputs: List[RequestOutput] = []
        for req in requests:
            lp = enc_out_map.get(req.request_id)
            if lp is not None:
                outputs.append(self.decode_streaming_chunk(req, lp))
        return outputs

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        if not hasattr(request, "_wfst_decoder"):
            request._wfst_decoder = Decoder(self._stream_cfg, fst=self._config.fst_path)
            request._wfst_decoder.init_stream()

        chunk_logp = enc_out.squeeze(0)  # (1, T, V) -> (T, V)
        result: DecoderResult = request._wfst_decoder.decode_chunk(chunk_logp)
        best = result.tokens[0] if result.tokens else []
        return RequestOutput(
            request_id=request.request_id,
            text=self._to_text(best),
            tokens=result.tokens,
            scores=result.scores,
            finished=False,
        )

    def finalize(self, request: Request) -> RequestOutput:
        wfst_dec = getattr(request, "_wfst_decoder", None)
        if wfst_dec is None:
            # No chunks were decoded (empty audio).
            return RequestOutput(
                request_id=request.request_id,
                text="",
                tokens=[],
                finished=True,
            )
        result: DecoderResult = wfst_dec.finalize_stream()
        best = result.tokens[0] if result.tokens else []
        text = self._to_text(best)
        return RequestOutput(
            request_id=request.request_id,
            text=text,
            tokens=result.tokens,
            scores=result.scores,
            finished=True,
        )
