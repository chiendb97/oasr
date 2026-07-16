# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) frame-synchronous greedy decode strategy.

Consumes raw encoder hidden states (``consumes="hidden"``) and drives the
model's stateless predictor (``model.decoder``) + ``model.joiner`` directly. For
each encoder frame the joiner combines the frame with the current prediction;
``argmax`` either emits a label (update the predictor's label window, stay on the
frame, bounded by ``max_sym_per_frame``) or is blank (advance to the next frame).
The predictor projection is recomputed only on steps where at least one row
emitted; the encoder is projected once up front (the icefall greedy fast path).

One vectorized greedy core (:meth:`_greedy_loop`) serves both paths:

* **offline** — fresh predictor state per micro-batch row, loop to the row's
  encoder length;
* **streaming** — per-request :class:`_Session` (label window + predictor
  projection + accumulated hypothesis) threaded across chunks; each tick decodes
  the new chunk's frames in a batch grouped by chunk length.

The per-emit row loop is fully vectorized: label windows shift via a masked
``torch.cat``, and emitted tokens are collected as per-step snapshots read back
in one sync at loop end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple

import torch

from ..request import Request, RequestOutput
from .base import DecodeStrategy, register_decode_strategy

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer


@dataclass
class _Session:
    """Per-stream greedy state carried across chunks."""

    context: torch.Tensor  # (1, context_size) int64 label window
    dec_proj: torch.Tensor  # (1, J) predictor projection for that window
    hyp: List[int] = field(default_factory=list)
    steps: int = 0  # decoded chunks (drives the partial-emit cadence)


@register_decode_strategy("transducer")
class TransducerDecodeStrategy(DecodeStrategy):
    """Greedy RNNT decoding over encoder hidden states (offline + streaming)."""

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
        # Interim-partial cadence (shared engine knob): emit a partial every
        # N-th chunk; <= 0 disables partials (final transcript only).
        self._partial_interval = int(getattr(config, "partial_decode_interval", 1))
        # ``None`` marks a created-but-uninitialized session (state materializes
        # on the first chunk, when the encoder output's device is known).
        self._sessions: Dict[str, Optional[_Session]] = {}

    # ------------------------------------------------------------------
    # Vectorized greedy core (shared by offline + streaming)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _greedy_loop(
        self,
        enc_out: torch.Tensor,  # (B, T, D) encoder hidden
        lengths: torch.Tensor,  # (B,) valid frames per row
        context: torch.Tensor,  # (B, context_size) label windows (mutated copy returned)
        dec_proj: torch.Tensor,  # (B, J) predictor projections for those windows
    ) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """Run batched greedy over ``enc_out``; returns newly emitted tokens per
        row plus the updated ``(context, dec_proj)`` predictor state."""
        model = self._model
        joiner = model.joiner
        decoder = model.decoder
        blank = int(model.blank_id)
        max_sym = self._max_sym

        device = enc_out.device
        B, T, _ = enc_out.shape
        lengths = lengths.to(device=device, dtype=torch.long)

        # Project the encoder output once; per step only the predictor is re-run.
        enc_proj = joiner.encoder_proj(enc_out)  # (B, T, J)

        t = torch.zeros(B, dtype=torch.long, device=device)
        sym = torch.zeros(B, dtype=torch.long, device=device)
        rows = torch.arange(B, device=device)
        no_emit = torch.full((B,), -1, dtype=torch.long, device=device)
        emitted: List[torch.Tensor] = []  # per-step (B,) token snapshots, -1 = no emit

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
                # Shift the emitted label into each emitting row's window; rows
                # that didn't emit keep their window, so the batched predictor
                # recompute reproduces their previous projection exactly.
                shifted = torch.cat([context[:, 1:], tok.unsqueeze(1)], dim=1)
                context = torch.where(emit.unsqueeze(1), shifted, context)
                dec_proj = joiner.decoder_proj(decoder(context))
                emitted.append(torch.where(emit, tok, no_emit))
                sym = sym + emit.long()

            t = t + advance.long()
            sym = torch.where(advance, torch.zeros_like(sym), sym)

        if emitted:
            # One host readback for the whole loop.
            snap = torch.stack(emitted, dim=1).tolist()  # B × S
            hyps = [[tk for tk in row if tk >= 0] for row in snap]
        else:
            hyps = [[] for _ in range(B)]
        return hyps, context, dec_proj

    def _init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder = self._model.decoder
        joiner = self._model.joiner
        context = decoder.init_state(batch_size, device)  # (B, context_size) int64
        dec_proj = joiner.decoder_proj(decoder(context))  # (B, J)
        return context, dec_proj

    # ------------------------------------------------------------------
    # Offline greedy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        assert self._model is not None, "TransducerDecodeStrategy needs the model"
        B = enc_out.size(0)
        context, dec_proj = self._init_state(B, enc_out.device)
        hyps, _, _ = self._greedy_loop(enc_out, enc_lengths, context, dec_proj)
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
    # Streaming greedy (per-request predictor state across chunks)
    # ------------------------------------------------------------------

    def create_session(self, request: Request) -> None:
        """Register the stream; predictor state initializes lazily on the first
        chunk (the device/dtype come from the encoder output)."""
        self._sessions.setdefault(request.request_id, None)  # type: ignore[arg-type]

    def free_session(self, request: Request) -> None:
        self._sessions.pop(request.request_id, None)

    def _session(self, request_id: str, device: torch.device) -> _Session:
        s = self._sessions.get(request_id)
        if s is None:
            context, dec_proj = self._init_state(1, device)
            s = _Session(context=context, dec_proj=dec_proj)
            self._sessions[request_id] = s
        return s

    @torch.no_grad()
    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        assert self._model is not None, "TransducerDecodeStrategy needs the model"
        ready = [r for r in requests if r.request_id in enc_out_map]
        if not ready:
            return []

        # Group by chunk length so each group runs one batched greedy loop.
        groups: Dict[int, List[Request]] = {}
        for req in ready:
            groups.setdefault(int(enc_out_map[req.request_id].size(1)), []).append(req)

        outputs: List[RequestOutput] = []
        for T_chunk, group in groups.items():
            enc = torch.cat([enc_out_map[r.request_id] for r in group], dim=0)  # (B, T, D)
            device = enc.device
            sessions = [self._session(r.request_id, device) for r in group]
            context = torch.cat([s.context for s in sessions], dim=0)
            dec_proj = torch.cat([s.dec_proj for s in sessions], dim=0)
            lengths = torch.full((len(group),), T_chunk, dtype=torch.long, device=device)

            new_hyps, context, dec_proj = self._greedy_loop(enc, lengths, context, dec_proj)

            for b, (req, s) in enumerate(zip(group, sessions)):
                s.context = context[b : b + 1]
                s.dec_proj = dec_proj[b : b + 1]
                s.hyp.extend(new_hyps[b])
                s.steps += 1
                if self._partial_interval > 0 and s.steps % self._partial_interval == 0:
                    outputs.append(
                        RequestOutput(
                            request_id=req.request_id,
                            text=self._detok.detokenize(s.hyp),
                            tokens=[list(s.hyp)],
                            finished=False,
                        )
                    )
        return outputs

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        outs = self.decode_streaming_batch([request], {request.request_id: enc_out})
        if outs:
            return outs[0]
        # Partials disabled (partial_decode_interval <= 0): state advanced, no emit.
        s = self._sessions.get(request.request_id)
        hyp = list(s.hyp) if s is not None else []
        return RequestOutput(
            request_id=request.request_id,
            text=self._detok.detokenize(hyp),
            tokens=[hyp],
            finished=False,
        )

    def finalize(self, request: Request) -> RequestOutput:
        """Final transcript from the accumulated session hypothesis.

        The session itself is released by :meth:`free_session` (the executor
        calls it right after finalize).
        """
        s: Optional[_Session] = self._sessions.get(request.request_id)
        hyp = list(s.hyp) if s is not None else []
        return RequestOutput(
            request_id=request.request_id,
            text=self._detok.detokenize(hyp),
            tokens=[hyp],
            finished=True,
        )
