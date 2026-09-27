# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Zipformer CTC model: encoder_embed + Zipformer2 encoder + CTC head.

Offline (:meth:`ZipformerEncoder.forward`) and chunk-wise streaming
(:meth:`ZipformerModel.streaming_forward`) are both supported, faithfully
ported from icefall — streaming including its chunk geometry (windows of
``2 * chunk_size + 13`` frames at a stride of ``2 * chunk_size``) and its
processed-length mask over the left context.  The per-layer cache is declared
as fixed-extent slot state (:attr:`ZipformerEncoder.slot_state_specs`), so
the engine's slot runtime owns it; attention K/V is not paged, because each
stack keeps its own ``left_context_frames // downsample`` of a different width.
"""

from __future__ import annotations

import logging
import math
from typing import List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from oasr.cache.state import StreamStateSpec

from ..base import BaseAsrModel, BaseEncoder, LoadReport
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
        """Initial streaming state: ``[embed_cache] + encoder caches + [processed_lens]``.

        ``processed_lens`` (``(B,)`` int32) counts the embed frames a stream has
        been through.  It is what masks the part of the left-context cache that
        is still the initial zeros: icefall's streaming recipe does exactly this
        (``processed_mask`` in ``streaming_decode.py`` and in the exported
        streaming encoder), and without it the first ``left_context_frames`` of
        every stream attend a window of zero keys as though it were audio.
        """
        embed_state = self.encoder_embed.get_init_states(batch_size, device, dtype)
        enc_states = self.encoder.get_init_states(batch_size, device, dtype)
        processed = torch.zeros(batch_size, dtype=torch.int32, device=device)
        return [embed_state] + enc_states + [processed]

    def streaming_forward(
        self, xs: Tensor, xs_lens: Tensor, states: List[Tensor]
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """One chunk: ``xs (B, window, F)`` → ``(hidden, out_lens, new_states)``.

        ``window`` must be :attr:`streaming_window_frames` input frames — the
        embed then yields exactly ``chunk_size`` frames, the unit the encoder was
        trained to stream in.  Consecutive windows overlap by the embed's
        receptive field (:attr:`streaming_window_frames` minus
        :attr:`streaming_chunk_frames`); that overlap is the chunk's lookahead,
        not state.
        """
        embed_state, enc_states, processed = states[0], states[1:-1], states[-1]
        x, x_lens, new_embed = self.encoder_embed.streaming_forward(xs, xs_lens, embed_state)
        x = x.permute(1, 0, 2)  # (T', B, C0)
        batch_size, seq_len = x.size(1), x.size(0)
        left = self.encoder.left_context_frames[0]
        # The left-context cache holds ``left`` frames of which only the
        # ``processed`` most recent are real (icefall's ``processed_mask``, True =
        # masked); the chunk itself masks nothing past ``x_lens``.
        pos = torch.arange(left, device=x.device).expand(batch_size, left)
        processed_mask = (processed.unsqueeze(1) <= pos).flip(1)
        chunk_mask = make_pad_mask(x_lens, seq_len)
        src_key_padding_mask = torch.cat([processed_mask, chunk_mask], dim=1)
        out, out_lens, new_enc = self.encoder.streaming_forward(
            x, x_lens, enc_states, src_key_padding_mask
        )
        out = out.permute(1, 0, 2)  # (B, T'', Cmax)
        return out, out_lens, [new_embed] + new_enc + [processed + x_lens.to(processed.dtype)]

    # The embedding and convolution caches batch on dimension 0; attention and
    # value caches batch on dimension 1, repeated in six tensors per layer; the
    # trailing ``processed_lens`` is a plain ``(B,)``.
    _STATE_BATCH_DIM_CYCLE = (1, 1, 1, 1, 0, 0)

    def _state_batch_dim(self, i: int, n: int) -> int:
        if i == 0 or i == n - 1:
            return 0
        return self._STATE_BATCH_DIM_CYCLE[(i - 1) % 6]

    def stack_streaming_states(self, states_list: List[List[Tensor]]) -> List[Tensor]:
        """Stack per-stream state lists into one batched state list.

        Enables the stateful streaming backend to run one ``B = N`` chunk
        forward over N streams instead of N sequential ``B = 1`` forwards.
        """
        n = len(states_list[0])
        return [
            torch.cat([s[i] for s in states_list], dim=self._state_batch_dim(i, n))
            for i in range(n)
        ]

    def unstack_streaming_states(self, states: List[Tensor]) -> List[List[Tensor]]:
        """Split a batched state list back into per-stream state lists.

        Returns views into the batched tensors (no copy); each stream's next
        chunk re-stacks them, so the shared storage is transient.
        """
        outs: List[List[Tensor]] = []
        n = len(states)
        for i, t in enumerate(states):
            dim = self._state_batch_dim(i, n)
            rows = t.split(1, dim=dim)
            if not outs:
                outs = [[] for _ in range(len(rows))]
            for b, row in enumerate(rows):
                outs[b].append(row)
        return outs

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

    # -- streaming spec ----------------------------------------------------
    @property
    def _streaming_capable(self) -> bool:
        """Whether this *config* can be decoded chunk-by-chunk at all.

        icefall only trains the chunk-wise path when ``causal=True`` with a
        positive ``chunk_size``; a non-causal release (e.g. the
        ``zipformer-large-cr-ctc`` CTC models) has no streaming forward.
        """
        cs = self.config.chunk_size
        cs = cs[0] if isinstance(cs, (tuple, list)) else cs
        return bool(self.config.causal) and int(cs) > 0

    @property
    def streaming_kind(self) -> str:
        """Zipformer's streaming cache is fixed-extent per stream, so it runs on
        the engine's **slot** runtime: every icefall cache tensor (the per-layer
        left-context keys/values, conv tails, the embed's cached frames, the
        processed-length counter) is one :class:`~oasr.cache.StreamStateSpec`
        in a :class:`~oasr.cache.SlotStateCache`, at stable addresses, which is
        what lets the chunk forward be graph-captured.  The per-request list API
        (:meth:`get_streaming_init_states` / :meth:`streaming_forward`) stays —
        the ``"stateful"`` runtime drives it, and it is the parity oracle.

        Reports ``"none"`` for a non-causal config rather than claiming a
        capability the weights don't have. That distinction is load-bearing:
        the engine refuses any ``streaming_kind == "none"`` model in streaming
        service mode *at construction*, so a non-causal checkpoint now fails
        with a clear message up front instead of building an engine that
        raises out of :attr:`streaming_chunk_frames` on its first request.
        It also makes :attr:`BaseEncoder.cache_spec` ``None``, so no paged
        pool is allocated for weights that can never stream.
        """
        return "slot" if self._streaming_capable else "none"

    @property
    def slot_state_specs(self) -> Tuple[StreamStateSpec, ...]:
        """Every streaming-state tensor as a slot-cache declaration, in list order.

        The slot runtime's contract (:attr:`BaseEncoder.slot_state_specs`): the
        **whole** streaming cache, where a paged encoder's
        ``streaming_state_specs`` is only the extras beside paged K/V.

        Derived from :meth:`get_streaming_init_states` itself (on the ``meta``
        device, so nothing is allocated) rather than restated: the order, the
        shapes and each tensor's batch axis then cannot drift from the list API
        the ``"stateful"`` runtime and the parity tests drive.  All-zero is every
        state's initial value — including the processed-length counter — which
        is exactly what :class:`~oasr.cache.SlotStateCache` zeroes a slot to.
        """
        init = self.get_streaming_init_states(1, device=torch.device("meta"))
        n = len(init)
        kinds = ("key", "nonlin_attn", "val1", "val2", "conv1", "conv2")
        specs = []
        for i, t in enumerate(init):
            axis = self._state_batch_dim(i, n)
            if i == 0:
                name = "embed"
            elif i == n - 1:
                name = "processed_lens"
            else:
                name = f"layer{(i - 1) // 6}.{kinds[(i - 1) % 6]}"
            shape = tuple(int(d) for j, d in enumerate(t.shape) if j != axis)
            specs.append(
                StreamStateSpec(
                    name=name,
                    shape=shape,
                    slot_axis=axis,
                    dtype=None if t.dtype.is_floating_point else t.dtype,
                )
            )
        return tuple(specs)

    @property
    def subsampling_rate(self) -> int:
        """2x Conv2dSubsampling embed × ``output_downsampling_factor`` = total."""
        return 2 * self.config.output_downsampling_factor

    #: What icefall's streaming recipe pads a feature window with past the end of
    #: the audio (``LOG_EPS = log(1e-10)`` in ``streaming_decode.py``).  A
    #: short final window is padded with it to a full one, because the encoder
    #: only ever streams whole chunks (see :attr:`streaming_window_frames`).
    streaming_pad_value: float = math.log(1e-10)

    def _streaming_chunk(self) -> int:
        cs = self.config.chunk_size
        cs = cs[0] if isinstance(cs, (tuple, list)) else cs
        cs = int(cs)
        if cs <= 0:
            raise ValueError(
                "ZipformerEncoder is not configured for streaming "
                f"(chunk_size={cs}); build with causal=True and chunk_size>0."
            )
        return cs

    @property
    def streaming_chunk_frames(self) -> int:
        """Input fbank frames a stream advances per chunk — the window **stride**.

        ``chunk_size`` is in embed-output frames (Zipformer2 input) and the
        Conv2dSubsampling embed is 2x, so a chunk is ``chunk_size * 2`` input
        frames of new audio.  Requires a causal/streaming config
        (``chunk_size > 0``).
        """
        return self._streaming_chunk() * 2

    @property
    def streaming_window_frames(self) -> int:
        """Input frames per streaming window: one chunk plus the embed's lookahead.

        In streaming the embed contracts ``T`` input frames to
        ``(T - 7) // 2 - 3``: a 7-frame conv stack, then the ConvNeXt consuming
        its 3-frame right context.  So ``chunk_size`` output frames take
        ``2 * (chunk_size + 3) + 7`` input frames — 45 for a 16-frame chunk,
        icefall's ``chunk_size * 2 + pad_length`` with ``pad_length = 13`` — and
        consecutive windows overlap by 13 frames, advancing by
        :attr:`streaming_chunk_frames`.

        Exactly ``chunk_size`` embed frames per chunk is what the encoder was
        trained on, and what its downsampled stacks need: a chunk that is not a
        multiple of the deepest downsampling pads by *repeating* its last frame,
        and the repeated frames go into the left-context caches.  A window of
        only ``chunk_size * 2`` frames yields 9 embed frames for a 16-frame
        chunk, which is what this runtime used to feed.
        """
        right = int(self.encoder_embed.convnext.padding[0])
        return 2 * (self._streaming_chunk() + right) + 7


class ZipformerModel(BaseAsrModel):
    """Zipformer + CTC head (icefall ``egs/librispeech/ASR/zipformer``, ``--use-ctc 1``)."""

    @property
    def default_decode_type(self) -> str:
        return "ctc"

    @property
    def capabilities(self) -> frozenset:
        """Declared, not derived: the conformance test in
        ``tests/test_model_contract.py`` checks every registered architecture's
        advertised capabilities against ``oasr.models.interfaces.CAPABILITIES``,
        and can only do that without building the model when it is a constant."""
        return frozenset({"ctc"})

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
    def load_weights(self, state_dict: Mapping[str, Tensor], *, strict: bool = False) -> LoadReport:
        """Map an icefall ``AsrModel`` state-dict into this model.

        icefall keys ``encoder_embed.*`` / ``encoder.*`` / ``ctc_output.1.*`` map
        to ``encoder.encoder_embed.*`` / ``encoder.encoder.*`` / ``ctc.ctc_lo.*``.
        The CTC weight/bias is zero-padded up to this model's (8-aligned) vocab
        when the checkpoint's vocab is smaller (the GEMM kernels require
        N % 8 == 0).  Non-consumed checkpoint keys (transducer predictor/joiner,
        attention decoder, pruned-RNNT ``simple_*_proj``) land in
        ``LoadReport.dropped`` — the registry decides which of those are
        expected vs. a named capability loss.
        """
        remapped = {}
        dropped = []
        for k, v in state_dict.items():
            if k.startswith("encoder_embed."):
                remapped["encoder.encoder_embed." + k[len("encoder_embed.") :]] = v
            elif k.startswith("encoder."):
                remapped["encoder.encoder." + k[len("encoder.") :]] = v
            elif k.startswith("ctc_output.1."):
                remapped["ctc.ctc_lo." + k[len("ctc_output.1.") :]] = v
            else:
                dropped.append(k)

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
        return LoadReport.build(remapped, missing, unexpected, dropped)
