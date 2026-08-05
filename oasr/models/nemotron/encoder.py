# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron FastConformer encoder.

A Conformer encoder in the macaron arrangement (FF/2 → attention → conv → FF/2 →
norm) over 8x causally-subsampled features, with two things that are *not* the
WeNet Conformer this repo already has and are the reason this is its own module
rather than a config of that one:

**Transformer-XL relative position, with the shift.**  ``pos_emb`` spans the
``2L - 1`` relative distances ``L-1 … -(L-1)`` and the ``(b) + (d)`` term is
recovered by the classic pad-view-slice shift (:func:`rel_shift`).  WeNet's
rel-pos encoding is indexed per *key position* and needs no shift, which is why
``oasr.layers.RelPositionMultiHeadedAttention`` has none — feeding it this
checkpoint's positional table would silently move the diagonal.

**A ``chunked_limited`` attention mask that applies offline too.**  The model is
trained for streaming: a query sees its own chunk of ``num_lookahead_tokens + 1``
frames plus ``(sliding_window - 1) // chunk`` earlier chunks, and *nothing else*,
whether or not the caller is streaming.  Dropping the mask offline would be
"full attention on a model trained without it" — different arithmetic, not a
better-conditioned version of the same one.

Layout: ``(B, T, C)`` throughout, including inside the convolution module.
Upstream transposes to ``(B, C, T)`` four times per layer for ``nn.Conv1d``;
OASR's depthwise and pointwise conv1d kernels are channel-last, so those
transposes are simply absent here.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from oasr.layers import (
    Attention,
    ColumnParallelLinear,
    DepthwiseConv1d,
    FeedForward,
    Glu,
    LayerNorm,
    PointwiseConv1d,
    RowParallelLinear,
)
from oasr.layers.norm import LayerNormActivation

from ..base import BaseEncoder
from .config import NemotronEncoderConfig
from .subsampling import NemotronSubsampling

__all__ = [
    "MASK_FLOOR",
    "NemotronEncoder",
    "NemotronEncoderLayer",
    "chunked_limited_mask",
    "rel_shift",
    "relative_position_embedding",
]

#: Additive bias given to a masked attention key.  A **finite** floor, where
#: upstream writes ``float("-inf")``, and the difference is not cosmetic.
#:
#: Masking is exact either way: this bias is added after the softmax scale, the
#: real logits here reach ~±120, and ``exp(-1e4 - 120)`` underflows to *exactly*
#: zero in fp32.  What the finite form avoids is a defect in the fused CuteDSL
#: attention kernel — with ``-inf`` in ``attn_bias`` it is accurate only while the
#: **finite** part of the bias stays small, and this model's Transformer-XL bias
#: does not.  Measured on an RTX 5090, fp16, ``B3 H8 T122 D128``, a 38%-dense
#: ``chunked_limited`` mask, error against an fp32 SDPA reference:
#:
#: ============  ==============  ==========  =========
#: bias range    floor           fused       SDPA fp16
#: ============  ==============  ==========  =========
#: ±10           ``-inf``        0.00086     0.00086
#: ±20           ``-inf``        0.00093     0.00081
#: **±40**       ``-inf``        **1.365**   0.00066
#: **±80**       ``-inf``        **1.494**   0.00090
#: ±40           ``-1e4``        0.00066     0.00066
#: ±80           ``-1e4``        0.00090     0.00090
#: ============  ==============  ==========  =========
#:
#: End to end that was worth 0.69 of absolute error on the encoder output (vs
#: 0.004-0.02 for HF's own fp16 run) and two truncated transcripts in the
#: LJSpeech-200 gate.  Conformer passes ``-inf`` too — ``mask_to_bias``'s
#: ``-1e10`` overflows to it in fp16 — and is unaffected only because its
#: rel-pos bias is an order of magnitude smaller.  The kernel-side fix is
#: tracked in ``.artifacts/known_issues.md``; until it lands, a large finite
#: floor is the form to pass.
#:
#: A second, smaller reason to prefer it: a *fully* masked query row softmaxes
#: over a uniform floor and comes back finite, where ``-inf`` gives SDPA a NaN
#: row that ``0 * NaN`` then spreads to real rows in the next layer.
MASK_FLOOR = -1.0e4


def rel_shift(scores: torch.Tensor) -> torch.Tensor:
    """Transformer-XL relative shift on ``(B, H, T_q, P)`` scores.

    Appendix B of https://arxiv.org/abs/1901.02860: prepend a column, reinterpret
    the ``(T_q, P + 1)`` block as ``(P + 1, T_q)``, drop its first row, and
    reinterpret back.  Row ``i``, column ``j`` ends up holding source column
    ``(L - 1) + j - i``, and since source column ``p`` carries relative distance
    ``(L - 1) - p``, key ``j`` is scored against the embedding of distance
    ``i - j`` — Transformer-XL's ``R_{i-j}``, positive into the past.  Pinned as
    an identity in ``tests/test_nemotron.py``.
    """
    b, h, t_q, p = scores.shape
    padded = F.pad(scores, (1, 0))
    shifted: torch.Tensor = padded.view(b, h, -1, t_q)[:, :, 1:].view(b, h, t_q, p)
    return shifted


def relative_position_embedding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    inv_freq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sinusoidal table over the ``2 * length - 1`` relative distances.

    Returns ``(1, 2 * length - 1, dim)`` with **sin and cos interleaved** (upstream
    ``torch.stack([sin, cos], -1).reshape(...)``), positions running from
    ``length - 1`` down to ``-(length - 1)``.  Built in fp32 and cast at the end:
    at ``length`` in the thousands the phase argument is large enough that fp16
    accumulation visibly rounds it.
    """
    if inv_freq is None:
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
    positions = torch.arange(length - 1, -length, -1, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq.float())  # (2L-1, dim/2)
    table = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(freqs.size(0), dim)
    return table.unsqueeze(0).to(dtype)


def chunked_limited_mask(
    query_len: int,
    key_len: int,
    left_context: int,
    right_context: int,
    device: torch.device,
) -> torch.Tensor:
    """``(1, 1, T_q, T_k)`` bool mask for NeMo's ``chunked_limited`` attention.

    Frames are grouped into chunks of ``right_context + 1``; a query may attend to
    its own chunk (hence up to ``right_context`` frames of *lookahead*) and to the
    ``left_context // chunk`` chunks before it.  ``left_context < 0`` means
    unlimited history.

    The chunk boundaries are absolute frame indices, not relative offsets, so the
    mask is not a band: two adjacent queries can see different amounts of history
    depending on where they sit inside their chunk.  That is the trained
    behaviour — it is what makes one forward pass equal to the concatenation of
    the streaming chunks.
    """
    chunk = right_context + 1
    left_chunks = left_context // chunk if left_context >= 0 else key_len
    q_chunk = torch.arange(query_len, device=device).div(chunk, rounding_mode="trunc")
    k_chunk = torch.arange(key_len, device=device).div(chunk, rounding_mode="trunc")
    diff = q_chunk.unsqueeze(1) - k_chunk.unsqueeze(0)
    return ((diff >= 0) & (diff <= left_chunks)).view(1, 1, query_len, key_len)


class NemotronConvolutionModule(nn.Module):
    """``pointwise -> GLU -> [mask] -> causal depthwise -> LN+SiLU -> pointwise``.

    Stays in ``(B, T, C)``.  The depthwise convolution is **causal** (left pad
    ``kernel - 1``, no right pad), so a padded frame can only influence later
    frames — which are padding too.  That is why zeroing the fully-masked rows
    before it is sufficient to keep a mixed-length batch's valid frames clean.
    """

    def __init__(self, config: NemotronEncoderConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        bias = config.convolution_bias
        self.lorder = config.conv_kernel_size - 1
        self.pointwise_conv1 = PointwiseConv1d(channels, 2 * channels, bias=bias)
        self.glu = Glu()
        self.depthwise_conv = DepthwiseConv1d(
            channels, config.conv_kernel_size, padding=0, bias=bias
        )
        # LayerNorm + SiLU as one fused kernel; the parameter names stay ``norm.*``
        # so the checkpoint loads 1:1 (upstream has a bare ``nn.LayerNorm`` here
        # followed by a separate activation).
        self.norm = LayerNormActivation(channels, eps=1e-5, activation=config.hidden_act)
        self.pointwise_conv2 = PointwiseConv1d(channels, channels, bias=bias)

    def forward(self, x: torch.Tensor, silent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x (B, T, C)``; ``silent (B, T, 1)`` bool marks frames to zero."""
        x = self.glu(self.pointwise_conv1(x))
        if silent is not None:
            x = x.masked_fill(silent, 0.0)
        x = F.pad(x, (0, 0, self.lorder, 0))
        x = self.depthwise_conv(x)
        x = self.norm(x)
        out: torch.Tensor = self.pointwise_conv2(x)
        return out


class NemotronEncoderLayer(nn.Module):
    """Macaron Conformer block with Transformer-XL relative-position attention."""

    def __init__(self, config: NemotronEncoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        bias = config.attention_bias
        self.feed_forward1 = FeedForward(
            hidden,
            config.intermediate_size,
            activation=config.hidden_act,
            bias=bias,
            names=("linear1", "linear2"),
        )
        self.self_attn = NemotronRelPositionAttention(config)
        self.conv = NemotronConvolutionModule(config)
        self.feed_forward2 = FeedForward(
            hidden,
            config.intermediate_size,
            activation=config.hidden_act,
            bias=bias,
            names=("linear1", "linear2"),
        )
        self.norm_feed_forward1 = LayerNorm(hidden)
        self.norm_self_att = LayerNorm(hidden)
        self.norm_conv = LayerNorm(hidden)
        self.norm_feed_forward2 = LayerNorm(hidden)
        self.norm_out = LayerNorm(hidden)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        silent: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, attn_mask, silent)
        x = x + self.conv(self.norm_conv(x), silent)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class NemotronRelPositionAttention(nn.Module):
    """Multi-head attention with a Transformer-XL relative-position bias.

    Projections keep upstream's names (``q_proj`` / ``k_proj`` / ``v_proj`` /
    ``o_proj`` / ``relative_k_proj``, plus the two global bias vectors) and stay
    unfused, so the checkpoint loads 1:1; the *compute* goes through the shared
    :class:`oasr.layers.Attention` core with ``matrix_bd`` as an additive bias,
    which is the kernel-eligible mask form.
    """

    def __init__(self, config: NemotronEncoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        bias = config.attention_bias
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.q_proj = ColumnParallelLinear(hidden, hidden, bias=bias)
        self.k_proj = ColumnParallelLinear(hidden, hidden, bias=bias)
        self.v_proj = ColumnParallelLinear(hidden, hidden, bias=bias)
        self.o_proj = RowParallelLinear(hidden, hidden, bias=bias)
        # W_{k,R}: projects the relative-position table into key space.
        self.relative_k_proj = ColumnParallelLinear(hidden, hidden, bias=False)
        # Global content / positional biases (Transformer-XL's u and v).
        self.bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.attn = Attention(self.num_heads, self.head_dim, softmax_scale=self.scaling)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        silent: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self.attn.split_heads(self.q_proj(x))
        k = self.attn.split_heads(self.k_proj(x))
        v = self.attn.split_heads(self.v_proj(x))
        key_len = k.size(2)

        # (b) + (d): the position-biased query against the projected relative table.
        rel_k = self.relative_k_proj(pos_emb)  # (1, 2L-1, C)
        rel_k = rel_k.view(rel_k.size(0), -1, self.num_heads, self.head_dim)
        matrix_bd = (q + self.bias_v.unsqueeze(1)) @ rel_k.permute(0, 2, 3, 1)
        matrix_bd = rel_shift(matrix_bd)[..., :key_len] * self.scaling

        # A large *finite* floor rather than upstream's ``-inf``: mathematically
        # the same mask, and the one the fused kernel computes correctly at this
        # bias magnitude.  See :data:`MASK_FLOOR` for the measurement.
        bias = matrix_bd.masked_fill(attn_mask.logical_not(), MASK_FLOOR)
        out = self.attn(q + self.bias_u.unsqueeze(1), k, v, attn_bias=bias)
        if silent is not None:
            # A query row with no unmasked key softmaxes over all ``-inf``: the
            # fused kernel clamps it to zero, SDPA's math backend returns NaN.
            # Force zero so the two backends agree *and* so one padding row cannot
            # poison real rows in the next layer (a masked key still contributes
            # ``0 * NaN``).
            out = out.masked_fill(silent.transpose(1, 2).unsqueeze(-1), 0.0)
        projected: torch.Tensor = self.o_proj(self.attn.merge_heads(out))
        return projected


class NemotronEncoder(BaseEncoder):
    """8x-subsampling FastConformer encoder (offline).

    ``streaming_kind == "none"``: the checkpoint *is* a streaming model, but its
    streaming state is a per-stage causal-conv cache on the **subsampling** stack
    plus a sliding-window K/V cache, and the engine's paged backend models a K/V
    cache plus a single per-layer CNN cache.  Declaring ``"none"`` makes the
    engine refuse a streaming request at construction rather than serve one with
    a silently reset front-end; see the note in ``.artifacts/kernel_coverage.md``.
    """

    def __init__(self, config: NemotronEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.subsampling = NemotronSubsampling(config)
        self.layers = nn.ModuleList(
            NemotronEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        #: Right attention context in encoder frames.  Settable — the checkpoint
        #: was trained on several values (``supported_num_lookahead_tokens``) and
        #: it trades accuracy against streaming latency.
        self.num_lookahead_tokens = config.default_num_lookahead_tokens
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, config.hidden_size, 2, dtype=torch.float32) / config.hidden_size)
        )
        # Non-persistent: recomputed from the config, so it never appears in a
        # state dict and needs no ``_computed_buffer_suffixes`` entry.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # -- BaseEncoder introspection ------------------------------------------
    @property
    def num_encoder_layers(self) -> int:
        return len(self.layers)

    @property
    def output_size(self) -> int:
        return self.config.hidden_size

    @property
    def conv_kernel_size(self) -> int:
        return self.config.conv_kernel_size

    @property
    def streaming_kind(self) -> str:
        return "none"

    @property
    def subsampling_rate(self) -> int:
        return self.config.subsampling_factor

    def attention_context(self) -> Tuple[int, int]:
        """``(left, right)`` attention context in encoder frames."""
        return self.config.sliding_window - 1, int(self.num_lookahead_tokens)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, T, n_mels)`` + lengths → ``(hidden (B, T', C), masks (B, 1, T'))``."""
        lengths = xs_lens.to(device=xs.device, dtype=torch.long)
        hidden, out_lengths = self.subsampling(xs, lengths)
        if self.input_scale != 1.0:
            hidden = hidden * self.input_scale
        assert out_lengths is not None

        t_out = hidden.size(1)
        if t_out > self.config.max_position_embeddings:
            raise ValueError(
                f"encoder sequence length {t_out} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings} "
                f"({t_out * self.subsampling_rate} feature frames); split the audio "
                "or raise the config's limit"
            )
        keep = torch.arange(t_out, device=hidden.device).unsqueeze(0) < out_lengths.unsqueeze(1)
        # ``(1, 1, T, T)`` trained window ∧ ``(B, 1, 1, T)`` key padding.
        window = chunked_limited_mask(t_out, t_out, *self.attention_context(), hidden.device)
        attn_mask = window & keep.view(-1, 1, 1, t_out)
        # A row with nothing left to attend to: only reachable in a batch whose
        # rows differ enough in length that a padded query's whole chunk window
        # falls past the end.  Zeroed at the attention and before the depthwise
        # convolution, exactly as upstream does.
        silent = attn_mask.logical_not().all(dim=-1).transpose(1, 2)  # (B, T, 1)
        if not bool(silent.any()):
            silent = None

        pos_emb = relative_position_embedding(
            t_out,
            self.config.hidden_size,
            hidden.device,
            hidden.dtype,
            inv_freq=cast(torch.Tensor, self.inv_freq),
        )
        for layer in self.layers:
            hidden = layer(hidden, pos_emb, attn_mask, silent)
        return hidden, keep.unsqueeze(1)
