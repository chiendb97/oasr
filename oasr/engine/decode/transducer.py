# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) frame-synchronous greedy decode strategy.

Consumes raw encoder hidden states (``consumes="hidden"``) and drives the
model's stateless predictor (``model.decoder``) + ``model.joiner`` directly. For
each encoder frame the joiner combines the frame with the current prediction;
``argmax`` either emits a label (update the predictor's label window, stay on the
frame, bounded by ``max_sym_per_frame``) or is blank (advance to the next frame).
The predictor projection is recomputed only when a label is emitted; the encoder
is projected once up front (the icefall greedy fast path).

Batched over the micro-batch — utterances advance their own frame pointers in
lockstep steps. Streaming transducer (state threaded across chunks) is a
follow-up; the streaming methods raise until then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List

import torch

from ..request import Request, RequestOutput
from .base import DecodeStrategy, register_decode_strategy

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer


@register_decode_strategy("transducer")
class TransducerDecodeStrategy(DecodeStrategy):
    """Greedy RNNT decoding over encoder hidden states."""

    decode_type: ClassVar[str] = "transducer"
    consumes: ClassVar[str] = "hidden"

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        self._config = config
        self._detok = detok
        self._model = model
        # Cap on non-blank emissions per frame (safety against degenerate loops;
        # the same cap is applied uniformly so results are deterministic).
        self._max_sym = int(getattr(config, "transducer_max_sym_per_frame", 10))

    # ------------------------------------------------------------------
    # Offline greedy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        model = self._model
        assert model is not None, "TransducerDecodeStrategy needs the model"
        decoder = model.decoder
        joiner = model.joiner
        blank = int(model.blank_id)
        max_sym = self._max_sym

        device = enc_out.device
        B, T, _ = enc_out.shape
        lengths = enc_lengths.to(device=device, dtype=torch.long)

        # Project the encoder output once; per step only the predictor is re-run.
        enc_proj = joiner.encoder_proj(enc_out)  # (B, T, J)
        context = decoder.init_state(B, device)  # (B, context_size) int64
        dec_proj = joiner.decoder_proj(decoder(context))  # (B, J)

        t = torch.zeros(B, dtype=torch.long, device=device)
        sym = torch.zeros(B, dtype=torch.long, device=device)
        rows = torch.arange(B, device=device)
        hyps: List[List[int]] = [[] for _ in range(B)]

        max_steps = int(T) * (max_sym + 1) + B + 1  # termination safety bound
        for _ in range(max_steps):
            active = t < lengths
            if not bool(active.any()):
                break

            enc_t = enc_proj[rows, t.clamp(max=T - 1)]  # (B, J)
            logits = joiner(enc_t, dec_proj, project_input=False)  # (B, V)
            tok = logits.argmax(dim=-1)  # (B,)

            is_blank = (tok == blank) | (sym >= max_sym)
            emit = active & ~is_blank
            advance = active & is_blank

            if bool(emit.any()):
                for b in emit.nonzero(as_tuple=False).flatten().tolist():
                    tk = int(tok[b])
                    hyps[b].append(tk)
                    context[b] = torch.roll(context[b], -1, dims=0)
                    context[b, -1] = tk
                    sym[b] += 1
                # Predictor depends only on the label window — recompute (rows
                # that didn't emit are unchanged, so this stays correct).
                dec_proj = joiner.decoder_proj(decoder(context))

            t = torch.where(advance, t + 1, t)
            sym = torch.where(advance, torch.zeros_like(sym), sym)

        return [
            RequestOutput(
                request_id="",
                text=self._detok.detokenize(hyps[b]),
                tokens=[hyps[b]],
                finished=True,
            )
            for b in range(B)
        ]

    # ------------------------------------------------------------------
    # Streaming (follow-up)
    # ------------------------------------------------------------------

    def _streaming_unsupported(self):
        raise NotImplementedError(
            "Streaming transducer decode is not implemented yet — the offline "
            "greedy path (decode_offline) is wired. Streaming needs the predictor "
            "state + frame pointer threaded across encoder chunks."
        )

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        self._streaming_unsupported()

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        self._streaming_unsupported()

    def finalize(self, request: Request) -> RequestOutput:
        self._streaming_unsupported()
