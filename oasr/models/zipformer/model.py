# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Zipformer CTC model: encoder_embed + Zipformer2 encoder + CTC head.

Offline (:meth:`ZipformerEncoder.forward`) and chunk-wise streaming
(:meth:`ZipformerModel.streaming_forward`) are both supported, faithfully
ported from icefall.  Streaming uses Zipformer's own per-layer cache (see
:meth:`get_streaming_init_states`); it does **not** use the engine's paged-KV /
slot-CNN streaming pipeline, whose cache model is Conformer-specific.
"""

from __future__ import annotations

import logging
from typing import List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import BaseAsrModel, BaseEncoder
from ..heads.ctc import CTCHead
from .config import ZipformerEncoderConfig, ZipformerModelConfig
from .encoder import Zipformer2, _to_tuple
from .subsampling import Conv2dSubsampling

logger = logging.getLogger(__name__)


def make_pad_mask(lengths: Tensor, max_len: int) -> Tensor:
    """``(B,)`` lengths -> ``(B, max_len)`` bool, True where **padded**."""
    row = torch.arange(max_len, device=lengths.device)
    return row.unsqueeze(0) >= lengths.unsqueeze(1)


class ZipformerEncoder(BaseEncoder):
    """Zipformer acoustic encoder: Conv2dSubsampling (2x) + Zipformer2 (2x out) = 4x total.

    Operates batch-first ``(B, T, feat)`` at its API boundary (matching the engine
    contract), transposing to time-first internally to match icefall.
    """

    supports_packing = False
    supports_paged_streaming = False  # uses its own streaming cache, see streaming API

    def __init__(self, config: ZipformerEncoderConfig):
        super().__init__()
        self.config = config
        n = config.num_stacks
        encoder_dim = config.encoder_dim
        self.encoder_embed = Conv2dSubsampling(config.feature_dim, encoder_dim[0])
        self.encoder = Zipformer2(
            output_downsampling_factor=config.output_downsampling_factor,
            downsampling_factor=config.downsampling_factor,
            encoder_dim=encoder_dim,
            num_encoder_layers=config.num_encoder_layers,
            query_head_dim=config.query_head_dim,
            pos_head_dim=config.pos_head_dim,
            value_head_dim=config.value_head_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            cnn_module_kernel=config.cnn_module_kernel,
            pos_dim=config.pos_dim,
            causal=config.causal,
            chunk_size=config.chunk_size,
            left_context_frames=config.left_context_frames,
        )
        # Normalized per-stack tuples for cache-spec introspection.
        self._num_heads = _to_tuple(config.num_heads, n)
        self._value_head_dim = _to_tuple(config.value_head_dim, n)
        self._query_head_dim = _to_tuple(config.query_head_dim, n)
        self._num_layers = sum(_to_tuple(config.num_encoder_layers, n))

    # -- offline forward (BaseEncoder contract) ----------------------------
    def forward(self, xs: Tensor, xs_lens: Tensor) -> Tuple[Tensor, Tensor]:
        x, x_lens = self.encoder_embed(xs, xs_lens)  # (B, T', C0)
        src_key_padding_mask = make_pad_mask(x_lens, x.size(1))  # True=padded
        x = x.permute(1, 0, 2)  # (T', B, C0)
        out, out_lens = self.encoder(x, x_lens, src_key_padding_mask)  # (T'', B, Cmax)
        out = out.permute(1, 0, 2)  # (B, T'', Cmax)
        masks = (~make_pad_mask(out_lens, out.size(1))).unsqueeze(1)  # (B, 1, T'') True=valid
        return out, masks

    # -- chunk-wise streaming (Zipformer-specific) -------------------------
    def get_streaming_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> List[Tensor]:
        """Initial streaming state: ``[embed_cache] + encoder per-layer caches``."""
        embed_state = self.encoder_embed.get_init_states(batch_size, device, dtype)
        enc_states = self.encoder.get_init_states(batch_size, device, dtype)
        return [embed_state] + enc_states

    def streaming_forward(
        self, xs: Tensor, xs_lens: Tensor, states: List[Tensor]
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        embed_state, enc_states = states[0], states[1:]
        x, x_lens, new_embed = self.encoder_embed.streaming_forward(xs, xs_lens, embed_state)
        x = x.permute(1, 0, 2)  # (T', B, C0)
        batch_size, seq_len = x.size(1), x.size(0)
        left = self.encoder.left_context_frames[0]
        # No padding within a single chunk for a contiguous stream.
        src_key_padding_mask = torch.zeros(
            batch_size, left + seq_len, dtype=torch.bool, device=x.device
        )
        out, out_lens, new_enc = self.encoder.streaming_forward(
            x, x_lens, enc_states, src_key_padding_mask
        )
        out = out.permute(1, 0, 2)  # (B, T'', Cmax)
        return out, out_lens, [new_embed] + new_enc

    # -- introspection (feeds CacheSpec) -----------------------------------
    @property
    def num_encoder_layers(self) -> int:
        return self._num_layers

    @property
    def n_kv_head(self) -> int:
        return max(self._num_heads)

    @property
    def head_dim(self) -> int:
        return max(max(self._value_head_dim), max(self._query_head_dim))

    @property
    def output_size(self) -> int:
        return max(self.config.encoder_dim)


class ZipformerModel(BaseAsrModel):
    """Zipformer + CTC head (icefall ``egs/librispeech/ASR/zipformer``, ``--use-ctc 1``)."""

    def __init__(self, config: ZipformerModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ZipformerEncoder(config.encoder)
        # Registered as ``ctc`` (head is a property alias), matching the other models.
        self.ctc = CTCHead(config.vocab_size, self.encoder.output_size)

    @property
    def head(self) -> CTCHead:
        return self.ctc

    @classmethod
    def from_config(cls, config: ZipformerModelConfig, **aux) -> "ZipformerModel":
        return cls(config)

    def forward(self, input_features: Tensor, lengths: Tensor) -> Tensor:
        hidden, _ = self.encoder(input_features, lengths)
        return self.ctc(hidden)

    # -- chunk-wise streaming API ------------------------------------------
    def get_streaming_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tensor]:
        """Initial streaming state.  ``dtype`` defaults to the model's parameter dtype."""
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self.encoder.get_streaming_init_states(batch_size, device, dtype)

    def streaming_forward(
        self, features: Tensor, lengths: Tensor, states: List[Tensor]
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """One chunk forward → ``(ctc_log_probs (B, T, V), out_lengths, new_states)``."""
        hidden, out_lens, new_states = self.encoder.streaming_forward(features, lengths, states)
        return self.ctc(hidden), out_lens, new_states

    # -- weight loading -----------------------------------------------------
    def load_weights(
        self, state_dict: Mapping[str, Tensor], *, strict: bool = False
    ) -> None:
        """Map an icefall ``AsrModel`` state-dict into this model.

        icefall keys ``encoder_embed.*`` / ``encoder.*`` / ``ctc_output.1.*`` map
        to ``encoder.encoder_embed.*`` / ``encoder.encoder.*`` / ``ctc.ctc_lo.*``.
        Transducer / attention-decoder parameters (if present) are ignored.  The
        CTC weight/bias is zero-padded up to this model's (8-aligned) vocab when
        the checkpoint's vocab is smaller (the GEMM kernels require N % 8 == 0).
        """
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("encoder_embed."):
                remapped["encoder.encoder_embed." + k[len("encoder_embed.") :]] = v
            elif k.startswith("encoder."):
                remapped["encoder.encoder." + k[len("encoder.") :]] = v
            elif k.startswith("ctc_output.1."):
                remapped["ctc.ctc_lo." + k[len("ctc_output.1.") :]] = v
            # else: decoder / joiner / simple_*_proj / attention_decoder -> ignored

        if "ctc.ctc_lo.weight" in remapped:
            target_vocab = self.ctc.ctc_lo.weight.shape[0]
            w = remapped["ctc.ctc_lo.weight"]
            b = remapped["ctc.ctc_lo.bias"]
            pad = target_vocab - w.shape[0]
            if pad > 0:
                remapped["ctc.ctc_lo.weight"] = F.pad(w, (0, 0, 0, pad))
                remapped["ctc.ctc_lo.bias"] = F.pad(b, (0, pad))

        missing, unexpected = self.load_state_dict(remapped, strict=strict)
        if missing:
            logger.warning("Missing keys when loading Zipformer weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading Zipformer weights: %s", unexpected)
