# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer SANM NAR decoder (FunASR ``ParaformerSANMDecoder``).

Non-autoregressive: one parallel pass over the CIF acoustic embeddings
(``(B, U, D)`` continuous inputs — the token ``embed`` exists in the
checkpoint for the training-time sampler and is registered for state-dict
completeness but never used at inference).  Layer order is FunASR's
distinctive FFN-first arrangement: pre-norm FFN (residual deferred), FSMN
"self-attention" (residual = the layer input), then cross-attention over the
encoder memory.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from oasr.layers import Embedding, FeedForward, LayerNorm, Linear
from oasr.models.base import align_out_features

from .config import ParaformerModelConfig
from .modules import LAYER_NORM_EPS, DecoderFeedForward, FsmnBlock, SanmCrossAttention


class DecoderLayerSANM(nn.Module):
    """One NAR decoder layer; ``self_attn`` / ``src_attn`` may be ``None``
    (the final ``decoders3`` layer is FFN-only — and per FunASR it returns the
    FFN output *without* a residual in that case)."""

    def __init__(
        self,
        size: int,
        self_attn: Optional[FsmnBlock],
        src_attn: Optional[SanmCrossAttention],
        feed_forward: FeedForward,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size, eps=LAYER_NORM_EPS)
        if self_attn is not None:
            self.norm2 = LayerNorm(size, eps=LAYER_NORM_EPS)
        if src_attn is not None:
            self.norm3 = LayerNorm(size, eps=LAYER_NORM_EPS)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_lens: torch.Tensor,
    ) -> torch.Tensor:
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x = self.self_attn(tgt, tgt_mask)
            x = residual + x

        if self.src_attn is not None:
            residual = x
            x = self.norm3(x)
            x = residual + self.src_attn(x, memory, memory_lens)

        return x


class ParaformerSANMDecoder(nn.Module):
    """16 × (FFN + FSMN + cross-attn) + 1 FFN-only layer + norm + vocab head."""

    #: Read by ``BaseAsrModel.default_decode_type`` via the ``decoder`` slot.
    decode_type = "paraformer"

    def __init__(self, config: ParaformerModelConfig) -> None:
        super().__init__()
        d = config.encoder_output_size
        self.embed = nn.Sequential(Embedding(config.vocab_size, d))
        self.decoders = nn.ModuleList(
            [
                DecoderLayerSANM(
                    d,
                    FsmnBlock(d, config.decoder_kernel_size, config.decoder_sanm_shift),
                    SanmCrossAttention(config.decoder_attention_heads, d),
                    DecoderFeedForward(d, config.decoder_linear_units),
                )
                for _ in range(config.decoder_att_layer_num)
            ]
        )
        self.decoders3 = nn.ModuleList(
            [DecoderLayerSANM(d, None, None, DecoderFeedForward(d, config.decoder_linear_units))]
        )
        self.after_norm = LayerNorm(d, eps=LAYER_NORM_EPS)
        # The vocabulary head is widened to what the GEMM kernels can address
        # (8404 -> 8408 for paraformer-zh).  ``config.vocab_size`` stays the
        # true vocabulary — the tokenizer and the sos/eos ids are defined
        # against it — and ``ParaformerModel.load_weights`` pads the checkpoint
        # rows to match, giving the padding classes a bias far below any real
        # logit.  Without this the head is the one projection in the model that
        # can never reach a kernel.
        self.vocab_size = config.vocab_size
        self.output_layer = Linear(d, align_out_features(config.vocab_size))

    def forward(
        self,
        memory: torch.Tensor,
        memory_lens: torch.Tensor,
        acoustic_embeds: torch.Tensor,
        token_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One parallel NAR pass → ``(log_probs (B, U, V), token_lens)``."""
        B, U, _ = acoustic_embeds.shape
        device = memory.device
        tgt_mask = (
            (torch.arange(U, device=device).unsqueeze(0) < token_lens.to(device).unsqueeze(1))
            .unsqueeze(-1)
            .to(acoustic_embeds.dtype)
        )  # (B, U, 1) — multiplicative, consumed by the FSMN branch
        memory_lens = memory_lens.to(device)

        x = acoustic_embeds
        for layer in self.decoders:
            x = layer(x, tgt_mask, memory, memory_lens)
        x = self.decoders3[0](x, tgt_mask, memory, memory_lens)
        x = self.after_norm(x)
        # Drop the alignment padding before the softmax so the returned width is
        # the true vocabulary and the normalizer is exactly the unpadded one.
        logits = self.output_layer(x)[..., : self.vocab_size]
        return torch.log_softmax(logits, dim=-1), token_lens
