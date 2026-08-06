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
from typing import TYPE_CHECKING, List, Mapping, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import nn

from oasr.cache.state import StreamStateSpec
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

if TYPE_CHECKING:
    from oasr.cache.paged_kv import PagedKVCache
    from oasr.cache.state import SlotTensor

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

    Streaming replaces the zero left pad with the previous chunk's post-GLU tail
    (:meth:`forward_chunk`).  Unlike the subsampling stack this needs no
    alignment precondition — the convolution has stride 1, so there is no grid to
    fall off.
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
        return self._tail(x)

    def forward_chunk(
        self, x: torch.Tensor, cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(out (B, T, C), new_cache (B, lorder, C))`` given the previous tail.

        ``cache`` is the previous chunk's **post-GLU** activation tail, which is
        what the offline path zero-pads — so a stream's first chunk (an all-zero
        cache) computes exactly what an offline pass would.

        The new tail is taken from the *concatenation*, not from ``x``: a chunk
        shorter than ``lorder`` (``conv_kernel_size - 1`` can exceed the encoder
        frames one step produces — 8 against 4 on the released config) must carry
        part of the old cache forward, and slicing the concatenation does that
        without a special case.
        """
        x = self.glu(self.pointwise_conv1(x))
        padded = torch.cat([cache.to(dtype=x.dtype), x], dim=1)
        return self._tail(padded), padded[:, -self.lorder :]

    def _tail(self, x: torch.Tensor) -> torch.Tensor:
        """Everything after the left context is in place: conv → LN+SiLU → pointwise."""
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
        out: torch.Tensor = self.norm_out(x)
        return out

    def forward_chunk(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        att_cache: "PagedKVCache",
        conv_cache: torch.Tensor,
        cache_t1: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming counterpart of :meth:`forward` — same arithmetic, cached context.

        ``silent`` has no analogue here and needs none: a streaming chunk is real
        audio across its whole width for every row, which is exactly the condition
        the offline path's masking exists to handle.
        """
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn.forward_chunk(
            self.norm_self_att(x), pos_emb, attn_mask, att_cache, cache_t1
        )
        conv_out, new_conv_cache = self.conv.forward_chunk(self.norm_conv(x), conv_cache)
        x = x + conv_out
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x), new_conv_cache


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

    def forward_chunk(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        cache: "PagedKVCache",
        cache_t1: int,
    ) -> torch.Tensor:
        """Streaming attention over ``cache_t1`` cached frames plus this chunk.

        The relative-position arithmetic is the offline one with a different
        length: :func:`rel_shift` over a table built for ``L = cache_t1 + T_q``
        gives query ``i`` against key ``j`` the distance ``cache_t1 + i - j``,
        which is the absolute distance exactly when the cached region is
        right-aligned against the chunk — i.e. when the whole cohort shares
        ``cache_t1``.  That is what ``CacheConfig.prefill_kv_window`` guarantees,
        and why the table can be shared: a per-row table would mean running
        ``relative_k_proj`` per row, which at ``B = 32`` costs more than the layer.
        """
        q = self.attn.split_heads(self.q_proj(x))
        k = self.attn.split_heads(self.k_proj(x))
        v = self.attn.split_heads(self.v_proj(x))
        t_q = q.size(2)
        key_len = cache_t1 + t_q

        # Write this chunk's K/V into the paged pool.  ``cache_t1`` rather than the
        # ``cache_seqlens`` tensor: a prefilled window puts every stream at the same
        # committed length, so the homogeneous fast path applies and the
        # heterogeneous gather/scatter is not paid per layer per chunk.
        cache.write_kv_chunk(k, v, offset=cache_t1)

        rel_k = self.relative_k_proj(pos_emb)  # (1, 2L-1, C)
        rel_k = rel_k.view(rel_k.size(0), -1, self.num_heads, self.head_dim)
        matrix_bd = (q + self.bias_v.unsqueeze(1)) @ rel_k.permute(0, 2, 3, 1)
        matrix_bd = rel_shift(matrix_bd)[..., :key_len] * self.scaling

        bias = matrix_bd.masked_fill(attn_mask.logical_not(), MASK_FLOOR)
        out = self.attn(
            q + self.bias_u.unsqueeze(1),
            cache.k_cache,
            cache.v_cache,
            attn_bias=bias,
            kv_lens=cache.cache_seqlens + t_q,
            block_table=cache.block_table,
        )
        projected: torch.Tensor = self.o_proj(self.attn.merge_heads(out))
        return projected


class NemotronEncoder(BaseEncoder):
    """8x-subsampling FastConformer encoder, offline **and** cache-aware streaming.

    Streaming carries four kinds of state across chunks, and each maps onto a
    declared axis rather than a special case:

    * the **subsampling** stack's per-stage causal-conv tails — three
      :class:`~oasr.cache.StreamStateSpec` entries (:attr:`streaming_state_specs`);
    * the convolution module's per-layer post-GLU tail — the engine's existing
      ``"conv"`` slot cache, whose ``kernel_size - 1`` frames are exactly what this
      encoder needs;
    * attention K/V — the engine's paged pool, with the window **prefilled** at
      admission because :attr:`fixed_attention_window` is a trained constant;
    * the frame grid of the log-mel frontend — declared on the extractor
      (:class:`~oasr.features.StreamingFraming`), not here.

    One precondition ties them together and :meth:`streaming_geometry` enforces it:
    the chunk must be a whole number of trained attention chunks *and* its feature
    window a multiple of the subsampling factor.  Both are alignment conditions on
    a strided grid, and violating either is silent — the transcript stays plausible
    while the arithmetic stops matching the offline pass.
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

    #: The engine's paged-KV + slot-state runtime serves this encoder.
    supports_paged_streaming: bool = True

    @property
    def streaming_kind(self) -> str:
        return "paged"

    @property
    def n_kv_head(self) -> int:
        return self.config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.config.head_dim

    @property
    def subsampling_rate(self) -> int:
        return self.config.subsampling_factor

    @property
    def fixed_attention_window(self) -> int:
        """Trained attention left-context in encoder frames.

        Part of the mask, not a cache-sizing preference — which is why the engine
        derives the retained window from it and pre-fills it (see
        :class:`~oasr.models.base.CacheSpec`).
        """
        return self.config.sliding_window - 1

    @property
    def streaming_state_specs(self) -> Tuple[StreamStateSpec, ...]:
        """Per-stage subsampling tails; the conv cache comes from ``conv_kernel_size``."""
        return tuple(
            self.subsampling.state_specs(
                self.config.num_mel_bins, self.config.subsampling_conv_channels
            )
        )

    def streaming_geometry(self, chunk_size: int) -> Tuple[int, int]:
        """``(decoding_window, stride)`` in feature frames — equal, and validated.

        The window equals the stride because the subsampling is *causal with a
        cache*: a chunk of ``chunk_size`` encoder frames needs exactly
        ``chunk_size * 8`` input frames and no lookahead, unlike a centred
        subsampling front-end which needs a receptive field beyond its chunk.

        Two alignment conditions are checked here rather than discovered later,
        because both fail silently:

        * ``chunk_size`` must be a whole number of **trained attention chunks**
          (``num_lookahead_tokens + 1``).  The ``chunked_limited`` mask groups
          *absolute* frame positions, so a query in the first half of a
          misaligned step would need keys from the second half — frames that do
          not exist yet.
        * the resulting feature window must be a multiple of the **subsampling
          factor**, so every stage's input length is a multiple of its stride and
          the cached ``kernel - 1`` frames land on the stride grid (see
          ``_CausalPad.stream_left``).
        """
        chunk = int(self.num_lookahead_tokens) + 1
        if chunk_size <= 0 or chunk_size % chunk:
            raise ValueError(
                f"streaming chunk_size must be a positive multiple of "
                f"{chunk} (num_lookahead_tokens + 1 = the trained attention chunk), "
                f"got {chunk_size}. The chunked_limited mask groups absolute frame "
                "positions, so a partial trained chunk would need keys from the "
                "future. Use a multiple: "
                f"{', '.join(str(chunk * m) for m in (1, 2, 4))}, ..."
            )
        window = chunk_size * self.subsampling_rate
        return window, window

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

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def forward_chunk_paged(
        self,
        xs: torch.Tensor,
        offset: Union[int, torch.Tensor],
        att_caches: List["PagedKVCache"],
        cnn_cache: "SlotTensor",
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
        cache_t1: int = -1,
        states: Optional[Mapping[str, "SlotTensor"]] = None,
    ) -> torch.Tensor:
        """One streaming chunk → ``(B, chunk_size, hidden)``.

        ``xs`` is ``(B, chunk_size * 8, n_mels)`` — a **full** window for every row,
        which :meth:`streaming_geometry` is what guarantees.  ``offset`` is the
        per-row count of encoder frames already produced (a device tensor, so the
        whole forward is CUDA-graph capturable); ``cache_t1`` is the *constant*
        number of cached K/V frames the prefilled window reports.
        """
        if states is None:
            raise ValueError(
                "Nemotron streaming needs its per-stage subsampling caches; the "
                "backend passes them as `states` when the encoder declares "
                "`streaming_state_specs` (a raise rather than an assert because "
                "the failure mode without them is a silently reset front-end)"
            )
        if cache_t1 < 0:
            raise ValueError(
                "Nemotron streaming needs an explicit cache_t1 (the prefilled "
                "window's constant cached-frame count)"
            )
        hidden = self.subsampling.forward_chunk(xs, states)
        if self.input_scale != 1.0:
            hidden = hidden * self.input_scale

        t_q = hidden.size(1)
        key_len = cache_t1 + t_q
        offsets = (
            offset
            if isinstance(offset, torch.Tensor)
            else torch.full((hidden.size(0),), int(offset), device=hidden.device)
        )
        attn_mask = self._streaming_mask(offsets, t_q, cache_t1)
        pos_emb = relative_position_embedding(
            key_len,
            self.config.hidden_size,
            hidden.device,
            hidden.dtype,
            inv_freq=cast(torch.Tensor, self.inv_freq),
        )

        conv_in = cnn_cache.gather()  # (L, B, lorder, C)
        new_conv: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            hidden, new_tail = cast(NemotronEncoderLayer, layer).forward_chunk(
                hidden, pos_emb, attn_mask, att_caches[i], conv_in[i], cache_t1
            )
            new_conv.append(new_tail)
        cnn_cache.scatter(torch.stack(new_conv, dim=0))
        return hidden

    def _streaming_mask(self, offsets: torch.Tensor, t_q: int, cache_t1: int) -> torch.Tensor:
        """``(B, 1, T_q, cache_t1 + T_q)`` bool mask for one streaming chunk.

        Two conditions, both expressed against **absolute** frame positions so the
        result is the same mask the offline pass builds over the whole utterance:

        * ``chunked_limited``: query and key chunk indices come from
          ``pos // (right_context + 1)``, and the difference must lie in
          ``[0, left_context // chunk]``;
        * ``k_abs >= 0``: the leading columns of a *young* stream's prefilled
          window are zero-filled placeholders for history it does not have yet.
          They are masked here rather than by ``cache_seqlens`` because the whole
          point of prefilling is that the reported length is uniform.

        Key column ``j`` sits at absolute ``offsets[b] - cache_t1 + j`` — the cached
        region is right-aligned against the chunk, which is what makes the shared
        relative-position table correct too.
        """
        device = offsets.device
        left_context, right_context = self.attention_context()
        chunk = right_context + 1
        left_chunks = left_context // chunk if left_context >= 0 else cache_t1 + t_q

        q_abs = offsets.view(-1, 1, 1) + torch.arange(t_q, device=device).view(1, -1, 1)
        k_abs = (
            offsets.view(-1, 1, 1)
            - cache_t1
            + torch.arange(cache_t1 + t_q, device=device).view(1, 1, -1)
        )
        diff = q_abs.div(chunk, rounding_mode="trunc") - k_abs.div(chunk, rounding_mode="trunc")
        # ``k_abs >= 0`` has to be in the conjunction, not merely implied by it:
        # ``trunc`` rounds a negative ``k_abs`` *toward zero*, so a prefill column at
        # absolute -1 would land in chunk 0 and satisfy the window test.
        keep = (k_abs >= 0) & (diff >= 0) & (diff <= left_chunks)
        return keep.unsqueeze(1)
