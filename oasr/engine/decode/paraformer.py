# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer non-autoregressive decode strategy.

Consumes raw encoder hidden states (``consumes="hidden"``) and drives the
model's CIF predictor + NAR decoder: ``model.predict`` integrates encoder
frames into per-token acoustic embeddings (plus token counts and CIF fire
positions), ``model.nar_decode`` produces ``(B, U, V)`` log-probs in **one
parallel pass**, and the transcript is the per-position argmax — no
autoregressive loop, so the ordinary one-shot offline path fits unchanged
(mirroring the FunASR greedy inference exactly).

The CIF fire positions double as a coarse token alignment: token *k* spans
``(fire[k-1], fire[k])`` encoder frames, converted to seconds via the LFR
frame duration (60 ms for the standard fbank 10 ms × LFR 6 frontend) and
emitted as ``RequestOutput.timestamps``.

Offline-only: Paraformer's ``streaming_kind == "none"`` — the engine rejects
streaming requests at admission, and the streaming session methods raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List, Tuple

import torch

from ..request import Request, RequestOutput
from .base import DecodeStrategy, register_decode_strategy

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer


@register_decode_strategy("paraformer")
class ParaformerDecodeStrategy(DecodeStrategy):
    """One-shot NAR decoding over encoder hidden states (offline only)."""

    decode_type: ClassVar[str] = "paraformer"
    consumes: ClassVar[str] = "hidden"

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        super().__init__(config, detok, model)
        # Surface validation lives in ``build_decode_strategy`` via
        # ``oasr.models.interfaces.CAPABILITIES["paraformer"]``.
        mcfg = model.config
        self._filtered_ids = {int(mcfg.blank_id), int(mcfg.sos_id), int(mcfg.eos_id)}
        # Seconds per encoder frame: feature hop × LFR decimation.
        fcfg = config.feature_config
        self._frame_seconds = float(fcfg.frame_shift_ms) / 1000.0 * int(getattr(fcfg, "lfr_n", 1))

    # ------------------------------------------------------------------
    # Offline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        B = enc_out.size(0)
        acoustic_embeds, token_lens, fires = self._model.predict(enc_out, enc_lengths)
        if acoustic_embeds.size(1) == 0 or int(token_lens.max().item()) < 1:
            return [self._empty_output() for _ in range(B)]

        log_probs = self._model.nar_decode(enc_out, enc_lengths, acoustic_embeds, token_lens)
        # Greedy per position; per-row score = sum of best log-probs over the
        # row's valid tokens (the FunASR greedy hypothesis score).
        best_lp, best_ids = log_probs.float().max(dim=-1)  # (B, U)

        ids_cpu = best_ids.cpu()
        lp_cpu = best_lp.cpu()
        lens_cpu = token_lens.cpu()
        fires_cpu = fires.cpu()

        outputs: List[RequestOutput] = []
        for b in range(B):
            n = int(lens_cpu[b].item())
            if n <= 0:
                outputs.append(self._empty_output())
                continue
            row_ids = ids_cpu[b, :n].tolist()
            score = float(lp_cpu[b, :n].sum().item())
            spans = self._token_spans(fires_cpu[b], n)
            kept: List[int] = []
            kept_ts: List[Tuple[float, float]] = []
            for k, tok in enumerate(row_ids):
                if tok in self._filtered_ids:
                    continue
                kept.append(tok)
                kept_ts.append(spans[k])
            text = self._detok.detokenize(kept)
            outputs.append(
                RequestOutput(
                    request_id="",
                    text=text,
                    tokens=[kept],
                    scores=[score],
                    finished=True,
                    timestamps=kept_ts,
                )
            )
        return outputs

    def _token_spans(self, fires_row: torch.Tensor, n_tokens: int) -> List[Tuple[float, float]]:
        """CIF fire positions → per-token ``(start_s, end_s)`` spans.

        Token *k* ends at the *k*-th frame whose integrated weight fired and
        starts where the previous token ended (0 for the first).  If the fire
        count disagrees with the token count (rounding at the tail), missing
        spans reuse the last boundary — timestamps stay monotonic.
        """
        fire_idx = (fires_row >= 1.0).nonzero(as_tuple=False).reshape(-1).tolist()
        spans: List[Tuple[float, float]] = []
        prev = 0.0
        for k in range(n_tokens):
            end_frame = fire_idx[k] if k < len(fire_idx) else (fire_idx[-1] if fire_idx else 0)
            end = end_frame * self._frame_seconds
            if end < prev:
                end = prev
            spans.append((prev, end))
            prev = end
        return spans

    @staticmethod
    def _empty_output() -> RequestOutput:
        return RequestOutput(
            request_id="", text="", tokens=[[]], scores=[0.0], finished=True, timestamps=[]
        )

    # ------------------------------------------------------------------
    # Streaming (unsupported — Paraformer is offline-only)
    # ------------------------------------------------------------------

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        raise RuntimeError("the paraformer decode strategy is offline-only")

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        raise RuntimeError("the paraformer decode strategy is offline-only")

    def finalize(self, request: Request) -> RequestOutput:
        raise RuntimeError("the paraformer decode strategy is offline-only")
