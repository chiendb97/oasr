# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""WFST (k2) CTC beam-search decode strategy.

Wraps :class:`oasr.decode.Decoder` (k2 WFST beam search).  Streaming keeps a
per-request decoder on the request object (lazily created on first chunk), so
the session lifecycle methods stay no-ops (inherited from the base).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List

import torch

from oasr.decode import Decoder, DecoderResult

from ..request import Request, RequestOutput
from .base import DecodeStrategy, register_decode_strategy

if TYPE_CHECKING:
    from ..config import EngineConfig
    from .detokenize import Detokenizer


@register_decode_strategy("ctc_wfst")
class CtcWfstDecodeStrategy(DecodeStrategy):
    """CTC decoding via a k2 WFST beam search (GPU; requires a k2 build)."""

    decode_type: ClassVar[str] = "ctc"
    consumes: ClassVar[str] = "log_probs"

    def __init__(self, config: "EngineConfig", detok: "Detokenizer", model=None) -> None:
        self._config = config
        self._detok = detok

    # ------------------------------------------------------------------
    # Offline
    # ------------------------------------------------------------------

    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        cfg = self._config.wfst_decoder_config
        decoder = Decoder(cfg, fst=self._config.fst_path)

        lengths_list = enc_lengths.cpu().tolist()
        outputs = []
        for b in range(enc_out.size(0)):
            t = int(lengths_list[b])
            logp = enc_out[b, :t, :]  # (T, V)
            result: DecoderResult = decoder.decode(logp)
            best = result.tokens[0] if result.tokens else []
            text = self._detok.detokenize(best)
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
            request._wfst_decoder = Decoder(
                self._config.wfst_decoder_config, fst=self._config.fst_path
            )
            request._wfst_decoder.init_stream()

        chunk_logp = enc_out.squeeze(0)  # (1, T, V) -> (T, V)
        result: DecoderResult = request._wfst_decoder.decode_chunk(chunk_logp)
        best = result.tokens[0] if result.tokens else []
        return RequestOutput(
            request_id=request.request_id,
            text=self._detok.detokenize(best),
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
        text = self._detok.detokenize(best)
        return RequestOutput(
            request_id=request.request_id,
            text=text,
            tokens=result.tokens,
            scores=result.scores,
            finished=True,
        )
