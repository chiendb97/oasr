# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer: non-autoregressive ASR (SANM encoder + CIF predictor + NAR decoder).

Offline-only (``streaming_kind == "none"``).  The engine drives it through the
``paraformer`` decode strategy (``consumes="hidden"``): one
:meth:`encode_offline` pass, then :meth:`predict` (CIF → per-token acoustic
embeddings + token counts + fire positions) and :meth:`nar_decode` (one
parallel decoder pass → ``(B, U, V)`` log-probs) — no autoregressive loop, so
the one-shot offline executor path fits unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Mapping, Tuple, cast

import torch
from torch import nn

from oasr.models.base import (
    BaseAsrModel,
    LoadReport,
    align_out_features,
    pad_output_projection,
)
from oasr.quantization.layout import GemmChain, QuantLayout, SmoothGroup

from .config import ParaformerModelConfig
from .decoder import DecoderLayerSANM, ParaformerSANMDecoder
from .encoder import SANMEncoder
from .predictor import CifPredictor

logger = logging.getLogger(__name__)


class ParaformerModel(BaseAsrModel):
    """FunASR Paraformer for OASR (offline NAR decoding)."""

    def __init__(self, config: ParaformerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SANMEncoder(config)
        self.predictor = CifPredictor(config)
        self.decoder = ParaformerSANMDecoder(config)

    @classmethod
    def from_config(cls, config: ParaformerModelConfig, **aux: Any) -> "ParaformerModel":
        del aux
        return cls(config)

    # -- engine-facing metadata ---------------------------------------------
    @property
    def default_decode_type(self) -> str:
        return "paraformer"

    @property
    def capabilities(self) -> frozenset:
        return frozenset({"paraformer"})

    # -- weights --------------------------------------------------------------
    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> LoadReport:
        """FunASR keys map 1:1 onto this module tree (``encoder.encoders0.*``,
        ``encoder.encoders.*``, ``decoder.decoders.*``, ``predictor.*``); the
        converter additionally injects ``encoder.cmvn_shift`` /
        ``encoder.cmvn_scale`` parsed from ``am.mvn``."""
        sd = {}
        dropped = []
        for k, v in state_dict.items():
            if k.startswith(("encoder.", "decoder.", "predictor.")):
                sd[k] = v
            else:
                dropped.append(k)
        # The vocabulary head and the CIF alpha head are allocated GEMM-aligned;
        # widen the checkpoint's rows to match (no-op for a native bundle
        # already saved padded).
        pad_output_projection(
            sd, "decoder.output_layer.", align_out_features(self.config.vocab_size)
        )
        pad_output_projection(sd, "predictor.cif_output.", align_out_features(1))
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        if unexpected:
            logger.warning("Unexpected keys in Paraformer checkpoint: %s", unexpected[:8])
        if missing:
            logger.warning("Paraformer model keys not filled: %s", missing[:8])
        return LoadReport.build(sd, missing, unexpected, dropped)

    # -- quantization -------------------------------------------------------
    def quantization_layout(self) -> QuantLayout:
        """SANM encoder and NAR decoder; the CIF predictor and vocab head stay fp.

        Read off :mod:`~oasr.models.paraformer.encoder` and ``decoder``:

        * encoder ``norm1`` feeds only the fused ``linear_q_k_v`` (the first
          layer's included: 560 LFR features in), fusable.  V does **not** fold
          into ``linear_out``: it also feeds the FSMN memory block, which is added
          to the attention output, so ``linear_out`` is smoothed at run time;
        * encoder ``norm2`` feeds only ``feed_forward.w_1``, whose ReLU is
          positively homogeneous: ``w_2``'s smoothing folds into ``w_1``'s rows,
          and ``w_1`` -> ``w_2`` is a chain;
        * decoder ``norm1`` feeds only ``feed_forward.w_1`` (FFN first, residual
          deferred), and the FFN's inner ``norm`` (after the ReLU) feeds only its
          bias-free ``w_2`` -- two ``norm_affine`` groups, both fusable; the norm in
          between rules out a chain and a ``producer_rows`` fold;
        * decoder ``norm2`` feeds the FSMN (a depthwise conv), no GEMM;
          ``norm3`` feeds only ``src_attn.linear_q``;
        * the cross-attention's fused ``linear_k_v`` reads the encoder memory and
          its V rows reach ``linear_out`` through attention alone (``v_to_o``);
        * the encoder's ``after_norm`` output is integrated *raw* by the CIF
          predictor into the decoder's input embeddings, so it cannot absorb a
          scale: every layer's ``linear_k_v`` is smoothed at run time instead.

        Kept full precision: the CIF predictor (its alphas decide the token
        count) and the vocabulary head.  Pinned batch-invariant: ``cif_output``,
        the fp GEMM whose alphas weight the acoustic embeddings every quantized
        decoder layer reads (``M = B x T``).
        """
        groups: List[SmoothGroup] = []
        chains: List[GemmChain] = []
        cfg = self.config
        for name, count in (("encoders0", 1), ("encoders", cfg.encoder_num_blocks - 1)):
            for i in range(count):
                p = f"encoder.{name}.{i}"
                ffn = f"{p}.feed_forward"
                groups += [
                    SmoothGroup(
                        f"{p}.norm1", (f"{p}.self_attn.linear_q_k_v",), "norm_affine", True
                    ),
                    SmoothGroup(None, (f"{p}.self_attn.linear_out",), "runtime"),
                    SmoothGroup(f"{p}.norm2", (f"{ffn}.w_1",), "norm_affine", True),
                    SmoothGroup(f"{ffn}.w_1", (f"{ffn}.w_2",), "producer_rows"),
                ]
                chains.append(GemmChain(f"{ffn}.w_1", f"{ffn}.w_2"))
        cross: List[str] = []
        dec = cast(ParaformerSANMDecoder, self.decoder)
        for name, layers in (("decoders", dec.decoders), ("decoders3", dec.decoders3)):
            for i, layer in enumerate(cast(Iterable[nn.Module], layers)):
                p = f"decoder.{name}.{i}"
                ffn = f"{p}.feed_forward"
                groups += [
                    SmoothGroup(f"{p}.norm1", (f"{ffn}.w_1",), "norm_affine", True),
                    SmoothGroup(f"{ffn}.norm", (f"{ffn}.w_2",), "norm_affine", True),
                ]
                src = layer.src_attn if isinstance(layer, DecoderLayerSANM) else None
                if src is None:
                    continue
                d = src.h * src.d_k
                groups += [
                    SmoothGroup(f"{p}.norm3", (f"{p}.src_attn.linear_q",), "norm_affine", True),
                    SmoothGroup(
                        f"{p}.src_attn.linear_k_v",
                        (f"{p}.src_attn.linear_out",),
                        "v_to_o",
                        v_rows=(d, 2 * d),
                        heads=(src.h, src.h, src.d_k),
                    ),
                ]
                cross.append(f"{p}.src_attn.linear_k_v")
        if cross:
            groups.append(SmoothGroup(None, tuple(cross), "runtime"))
        return QuantLayout(
            groups=tuple(groups),
            keep_full_precision=("predictor.*", "decoder.output_layer"),
            chains=tuple(chains),
            batch_invariant=("predictor.cif_output",),
        )

    def calibration_forward(
        self, features: torch.Tensor, lengths: torch.Tensor, *, tokenizer: Any = None
    ) -> None:
        """Encoder, CIF and the NAR decoder pass -- the whole offline path."""
        del tokenizer
        enc_out, enc_lens = self.encode_offline(features, lengths)
        acoustic_embeds, token_lens, _fires = self.predict(enc_out, enc_lens)
        if int(token_lens.max()) > 0:
            self.nar_decode(enc_out, enc_lens, acoustic_embeds, token_lens)

    # -- SupportsParaformer compute contract ----------------------------------
    def predict(
        self,
        enc_out: torch.Tensor,
        enc_lens: torch.Tensor,
        return_alphas: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """CIF → ``(acoustic_embeds (B, U, D) fp32, token_lens (B,) int32,
        fires (B, T+1) fp32)``, plus ``alphas (B, T+1) fp32`` when asked.

        ``alphas`` is the predictor's per-frame token weight — a sigmoid whose
        sum over an utterance is the token count.  It was computed and dropped
        here before speech activity needed it; ``return_alphas`` keeps the
        three-value contract every existing caller relies on while making the
        fourth reachable without a second predictor forward.  It is opt-in
        rather than always-on for the same reason word timings are: the caller
        pays only for what it asked for, and here that is one extra live tensor
        per batch.
        """
        T = enc_out.size(1)
        mask = (
            torch.arange(T, device=enc_out.device).unsqueeze(0)
            < enc_lens.to(enc_out.device).unsqueeze(1)
        ).float()
        acoustic_embeds, token_num, alphas, fires = self.predictor(enc_out, mask)
        token_lens = token_num.round().long().clamp(min=0).to(torch.int32)
        if return_alphas:
            return acoustic_embeds, token_lens, fires, alphas
        return acoustic_embeds, token_lens, fires

    def nar_decode(
        self,
        enc_out: torch.Tensor,
        enc_lens: torch.Tensor,
        acoustic_embeds: torch.Tensor,
        token_lens: torch.Tensor,
    ) -> torch.Tensor:
        """One parallel decoder pass → ``(B, U, V)`` log-probs.

        ``acoustic_embeds`` from :meth:`predict` is a ``[:, :U]`` view of the CIF
        output and becomes the decoder's residual stream, whose fused add +
        LayerNorm launcher wants the *same strides* as the attention output -- a
        size-1 batch dim's stride included, which ``is_contiguous()`` ignores.
        Casting to fp16 / bf16 always made a fresh, canonically strided copy; in
        fp32 (the calibration master) the cast is a no-op, so ask for the copy.
        """
        embeds = acoustic_embeds.to(
            dtype=enc_out.dtype, memory_format=torch.contiguous_format, copy=True
        )
        log_probs, _ = self.decoder(enc_out, enc_lens, embeds, token_lens)
        return log_probs
