# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron ASR: FastConformer encoder + language prompt + RNN-T.

Engine integration: ``decode_type`` is ``"transducer"``, so the engine selects
:class:`~oasr.engine.decode.TransducerDecodeStrategy`, which consumes raw encoder
hidden states (``consumes="hidden"``) and drives ``model.decoder`` +
``model.joiner`` frame-synchronously — offline **and** cache-aware streaming
(``streaming_kind == "paged"``); see
:class:`~oasr.models.nemotron.encoder.NemotronEncoder` for the four kinds of state
a streaming chunk carries and the one alignment precondition that ties them
together.

Two things this model does that the icefall transducer does not:

**The prompt fusion is part of encoding.**  ``encode_offline`` *and*
``encode_chunk_paged`` return the encoder output **after** the language-prompt
projector, because that is what the joint's encoder projection consumes.  The
projector has no residual — its output replaces the hidden state — so it is not
something a caller can opt out of, and the streaming path cannot inherit the base
wrapper (which returns the raw encoder output).

**The predictor is recurrent.**  ``model.decoder`` is a 2-layer LSTM exposing the
:class:`~oasr.models.decoders.base.TransducerPredictor` protocol, and beam search
(which reorders a label-window state across the beam) does not apply; the decode
strategy refuses ``beam_size > 1`` at construction.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch

from oasr.cache.state import StreamStateSpec
from oasr.quantization.layout import GemmChain, QuantLayout, SmoothGroup
from oasr.utils.staging import to_device

from ..base import (
    BaseAsrModel,
    CacheSpec,
    LoadReport,
    align_out_features,
    pad_output_projection,
)
from .config import NemotronEncoderConfig, NemotronModelConfig
from .encoder import NemotronEncoder, NemotronEncoderLayer
from .predictor import NemotronPromptProjector, NemotronRnntJoint, NemotronRnntPredictor
from .subsampling import NemotronSubsampling

if TYPE_CHECKING:
    from oasr.cache.paged_kv import PagedKVCache
    from oasr.cache.state import SlotTensor

logger = logging.getLogger(__name__)

#: Upstream key prefix → this model's, for the two modules OASR relocates.  The
#: joint's projections live upstream as a top-level ``encoder_projector`` and as
#: the predictor's ``decoder_projector``; putting both on the joiner is what lets
#: the shared transducer strategy project the encoder once and the predictor
#: per emission.
_RENAMES = {
    "encoder_projector.": "joint.encoder_proj.",
    "decoder.decoder_projector.": "joint.decoder_proj.",
}

#: The per-stream prompt slot, as the streaming backend carries it: one int64 per
#: slot row, written from the request at admission and kept across a VAD turn
#: reset (``persistent``), because a turn boundary rewinds what the encoder has
#: heard, not which language the request asked for.  Read inside the chunk
#: forward, so a captured graph gathers each replay's rows rather than baking in
#: the ids it was captured with.
PROMPT_STATE = "prompt"
_PROMPT_STATE_SPEC = StreamStateSpec(
    PROMPT_STATE, (), slot_axis=0, dtype=torch.int64, persistent=True
)


class NemotronModel(BaseAsrModel):
    """Encoder + prompt projector + LSTM predictor (``decoder``) + ``joint``."""

    @property
    def default_decode_type(self) -> str:
        return "transducer"

    @property
    def capabilities(self) -> frozenset:
        """Declared rather than derived, so ``tests/test_model_contract.py`` can
        check it against ``oasr.models.interfaces.CAPABILITIES`` without building
        the model."""
        return frozenset({"transducer"})

    def __init__(
        self,
        encoder: NemotronEncoder,
        decoder: NemotronRnntPredictor,
        joint: NemotronRnntJoint,
        prompt_projector: Optional[NemotronPromptProjector] = None,
        *,
        blank_id: int,
        prompt_id: int,
        prompt_dictionary: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.prompt_projector = prompt_projector
        self._blank_id = int(blank_id)
        #: Language-prompt slot for a request that names no language.  Settable:
        #: the released checkpoint's default is the ``"auto"`` slot (its own
        #: language ID), and pinning a deployment to one language is a
        #: per-deployment choice.  A request's ``language`` overrides it for that
        #: request (:meth:`resolve_prompt_id`).
        self.prompt_id = int(prompt_id)
        #: Locale → slot, from the checkpoint's ``processor_config.json``.
        self.prompt_dictionary: Dict[str, int] = dict(prompt_dictionary or {})
        if prompt_projector is not None:
            slots = prompt_projector.num_prompts
            bad = {k: v for k, v in self.prompt_dictionary.items() if not 0 <= v < slots}
            if bad or not 0 <= self.prompt_id < slots:
                raise ValueError(
                    f"prompt slots must lie in [0, {slots}): default_prompt_id="
                    f"{self.prompt_id}, out-of-range dictionary entries {bad}"
                )
        self._joint_out = align_out_features(int(joint.vocab_size))

    @property
    def blank_id(self) -> int:
        return self._blank_id

    @property
    def joiner(self) -> NemotronRnntJoint:
        """The name the transducer decode strategy reaches for.

        A property rather than a second attribute: assigning the module twice
        would register it twice and duplicate every joint tensor in the state
        dict.  Upstream calls it ``joint``; keeping that as the registered name
        is what makes ``joint.head.*`` load 1:1.
        """
        return self.joint

    @property
    def head(self):  # type: ignore[override]
        """Transducers have no CTC head; ``None`` keeps generic probes safe."""
        return None

    @classmethod
    def from_config(  # type: ignore[override]  # narrower config, as every arch does
        cls, config: NemotronModelConfig, **aux: Any
    ) -> "NemotronModel":
        del aux
        encoder = NemotronEncoder(config.encoder)
        decoder = NemotronRnntPredictor(
            config.vocab_size,
            config.decoder_hidden_size,
            config.num_decoder_layers,
            blank_id=config.blank_token_id,
        )
        joint = NemotronRnntJoint(
            config.encoder.hidden_size,
            config.decoder_hidden_size,
            config.vocab_size,
            activation=config.hidden_act,
        )
        prompt_projector = (
            NemotronPromptProjector(
                config.encoder.hidden_size,
                config.num_prompts,
                config.prompt_intermediate_size,
            )
            if config.num_prompts > 0
            else None
        )
        return cls(
            encoder,
            decoder,
            joint,
            prompt_projector,
            blank_id=config.blank_token_id,
            prompt_id=config.default_prompt_id,
            prompt_dictionary=config.prompt_dictionary,
        )

    # -- the language prompt ------------------------------------------------

    def resolve_prompt_id(self, language: Optional[str]) -> int:
        """The prompt slot for a request's ``language`` (``None``: :attr:`prompt_id`).

        ``language`` is a bare ISO-639 subtag (``DecodingOptions`` refuses a
        region, and the serving layer reduces ``"vi-VN"`` to ``"vi"``), while the
        checkpoint's table is keyed by locale.  Resolved in the table's own terms,
        first match wins:

        1. the bare code, when the table lists it (``"en"`` -> 0, ``"pt"`` -> 13);
        2. the table's ``xx-XX`` form, its spelling of the language without a
           region (``"zh"`` -> ``"zh-ZH"`` -> 4, as ``"ja-JA"``, ``"ko-KO"``);
        3. the one slot every ``xx-*`` locale shares (``"vi"`` -> ``"vi-VN"`` -> 33).

        Anything else raises: a code the table does not know, or one whose locales
        disagree with no generic entry to settle it.  Choosing a region for the
        caller would decode in a language they did not name, confidently.
        """
        if language is None:
            return self.prompt_id
        if self.prompt_projector is None or not self.prompt_dictionary:
            raise ValueError(
                "this Nemotron checkpoint has no language-prompt dictionary "
                "(processor_config.json: prompt_dictionary), so a per-request "
                f"language={language!r} cannot be honoured; drop it"
            )
        table = self.prompt_dictionary
        for key in (language, f"{language}-{language.upper()}"):
            if key in table:
                return table[key]
        slots = {v for k, v in table.items() if k.split("-", 1)[0] == language}
        if len(slots) == 1:
            return slots.pop()
        if slots:
            locales = sorted(k for k in table if k.split("-", 1)[0] == language)
            raise ValueError(
                f"language={language!r} is ambiguous for this checkpoint: its locales "
                f"{locales} use different prompt slots"
            )
        known = sorted({k.split("-", 1)[0] for k in table if k.split("-", 1)[0].islower()})
        raise ValueError(f"language={language!r} is not one this checkpoint knows: {known}")

    @property
    def encoder_options(self) -> Tuple[str, ...]:
        """``("language",)`` when there is a projector and a table to resolve against."""
        if self.prompt_projector is None or not self.prompt_dictionary:
            return ()
        return ("language",)

    def validate_encoder_options(self, options: Any) -> None:
        self.resolve_prompt_id(getattr(options, "language", None))

    def encoder_conditioning(
        self, options: Sequence[Any], device: torch.device
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Per-row prompt ids, or ``None`` when no row names a language.

        Staged through pinned memory: a pageable ``torch.tensor(..., device=)``
        would make the host wait for the forward queued ahead of it.
        """
        if self.prompt_projector is None:
            return None
        languages = [getattr(o, "language", None) for o in options]
        if all(lang is None for lang in languages):
            return None
        ids = [self.resolve_prompt_id(lang) for lang in languages]
        return {"prompt_ids": to_device(ids, dtype=torch.int64, device=device)}

    def stream_conditioning(self, options: Any) -> Dict[str, float]:
        if self.prompt_projector is None:
            return {}
        return {PROMPT_STATE: self.resolve_prompt_id(getattr(options, "language", None))}

    @property
    def cache_spec(self) -> Optional[CacheSpec]:
        """The encoder's streaming cache plus one per-stream prompt slot."""
        spec = self.encoder.cache_spec
        if spec is None or self.prompt_projector is None:
            return spec
        return dataclasses.replace(spec, stream_states=(*spec.stream_states, _PROMPT_STATE_SPEC))

    # -- engine-facing forward -----------------------------------------------

    def encode_offline(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        prompt_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, T, n_mels)`` → ``(prompt-fused hidden (B, T', C), out_lengths)``.

        The prompt projector runs here, not in the joint, because its output *is*
        the encoder representation from the joint's point of view (upstream's
        ``get_audio_features``).  ``prompt_ids (B,)`` conditions each row on its
        own language (:meth:`encoder_conditioning`); ``None`` uses
        :attr:`prompt_id` for all of them.
        """
        hidden, masks = self.encoder(features, lengths)
        return self._fuse_prompt(hidden, prompt_ids), self._lengths_from_mask(masks)

    def encode_chunk_paged(
        self,
        input_features: torch.Tensor,
        offset: Union[int, torch.Tensor],
        att_caches: List["PagedKVCache"],
        cnn_cache: "SlotTensor",
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
        cache_t1: int = -1,
        states: Optional[Mapping[str, "SlotTensor"]] = None,
    ) -> torch.Tensor:
        """Streaming counterpart of :meth:`encode_offline` — prompt fused the same way.

        Overridden rather than inherited because the base wrapper returns the raw
        encoder output: the prompt projector is part of *this* model's encoder
        representation, and leaving it out of the streaming path would hand the
        joint a differently-conditioned hidden state than the offline path does.
        """
        prompt_ids = None
        if states is not None and PROMPT_STATE in states:
            # Each stream's own slot, gathered here so a captured graph reads the
            # rows it is replayed with.  The encoder is handed only what it
            # declared.
            prompt_ids = states[PROMPT_STATE].gather()
            states = {k: v for k, v in states.items() if k != PROMPT_STATE}
        hidden = self.encoder.forward_chunk_paged(
            input_features, offset, att_caches, cnn_cache, att_mask, cache_t1, states
        )
        return self._fuse_prompt(hidden, prompt_ids)

    def _fuse_prompt(
        self, hidden: torch.Tensor, prompt_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Condition every frame on its row's language prompt (upstream ``get_audio_features``).

        The prompt projector runs here, not in the joint, because its output *is*
        the encoder representation from the joint's point of view.
        """
        if self.prompt_projector is None:
            return hidden
        if prompt_ids is None:
            prompt_ids = torch.full(
                (hidden.size(0),), self.prompt_id, dtype=torch.long, device=hidden.device
            )
        fused: torch.Tensor = self.prompt_projector(hidden, prompt_ids)
        return fused

    # -- quantization -------------------------------------------------------

    def quantization_layout(self) -> QuantLayout:
        """The FastConformer encoder; predictor, joint and prompt projector stay fp.

        Read off :class:`~oasr.models.nemotron.encoder.NemotronEncoderLayer`
        (``forward`` and ``forward_chunk`` take the same path):

        * ``norm_feed_forward1`` / ``norm_feed_forward2`` feed only their FFN's
          ``linear1``; SiLU sits between ``linear1`` and ``linear2``, so
          ``linear2`` is smoothed at run time, and nothing but ``linear2`` reads
          ``linear1``'s activated output (a chain);
        * ``norm_self_att`` feeds ``q_proj`` / ``k_proj`` / ``v_proj`` and nothing
          else (the Transformer-XL bias projects the positional table, not this);
        * V reaches ``o_proj`` only through attention -- the bias terms touch the
          query side, the silent-row ``masked_fill`` zeroes whole rows -- and the
          paged K/V cache holds V in the smoothed domain, read back only by this
          attention (``v_to_o``, ``v_proj`` the whole producer);
        * ``norm_conv`` feeds only ``conv.pointwise_conv1``: unlike the WeNet
          Conformer, the mask, the causal pad and the streaming cache ``cat`` all
          come *after* the GLU, so this norm may emit int8; ``pointwise_conv2``
          follows LayerNorm + SiLU and is smoothed at run time;
        * ``norm_out``'s output is the next block's residual stream: no group.

        Kept full precision: the subsampling stack's ``linear`` (the first layer),
        ``relative_k_proj`` (its input is the sinusoidal table), the LSTM
        predictor (``decoder``), the ``joint`` and the language-prompt projector.
        Pinned batch-invariant when quantized: the subsampling projection
        (``M = B x T``) and ``relative_k_proj`` (``M = 2L - 1`` with ``L`` the
        padded length, or one attention block's keys once the offline pass runs
        in blocks, so it can depend on what shares the batch) -- the two fp
        GEMMs ahead of int8 layers.
        """
        groups: List[SmoothGroup] = []
        chains: List[GemmChain] = []
        encoder = cast(NemotronEncoder, self.encoder)
        for i, layer in enumerate(cast(Iterable[NemotronEncoderLayer], encoder.layers)):
            p = f"encoder.layers.{i}"
            attn = layer.self_attn
            for ff in ("feed_forward1", "feed_forward2"):
                groups.append(
                    SmoothGroup(f"{p}.norm_{ff}", (f"{p}.{ff}.linear1",), "norm_affine", True)
                )
                groups.append(SmoothGroup(None, (f"{p}.{ff}.linear2",), "runtime"))
                chains.append(GemmChain(f"{p}.{ff}.linear1", f"{p}.{ff}.linear2"))
            groups.append(
                SmoothGroup(
                    f"{p}.norm_self_att",
                    tuple(f"{p}.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj")),
                    "norm_affine",
                    fuse_output_quant=True,
                )
            )
            groups.append(
                SmoothGroup(
                    f"{p}.self_attn.v_proj",
                    (f"{p}.self_attn.o_proj",),
                    "v_to_o",
                    v_rows=(0, attn.v_proj.out_features),
                    heads=(attn.num_heads, attn.num_heads, attn.head_dim),
                )
            )
            groups.append(
                SmoothGroup(f"{p}.norm_conv", (f"{p}.conv.pointwise_conv1",), "norm_affine", True)
            )
            groups.append(SmoothGroup(None, (f"{p}.conv.pointwise_conv2",), "runtime"))
        return QuantLayout(
            groups=tuple(groups),
            # The subsampling's convolutions are selected by ``module_types``
            # (``conv2d``) only; its output projection stays full precision.
            keep_full_precision=(
                "encoder.subsampling.linear",
                "*.relative_k_proj",
                "decoder.*",
                "joint.*",
                "prompt_projector.*",
            ),
            chains=tuple(chains),
            batch_invariant=("encoder.subsampling.linear", "*.relative_k_proj"),
        )

    # -- weights --------------------------------------------------------------

    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> LoadReport:
        """Map an HF ``Nemotron3_5AsrForRNNT`` state dict into this model.

        Almost everything is 1:1 — the layer, convolution, LSTM and prompt keys
        keep their upstream names.  Three things are not:

        * the two joint projections move onto the joiner (:data:`_RENAMES`);
        * the subsampling projection's **input axis is permuted**, because
          upstream flattens the conv output ``(channel, freq)`` and this
          implementation stays in NHWC and flattens ``(freq, channel)``.  Doing
          it here rather than per forward costs one permute at load and nothing
          afterwards; doing it in a module hook would double-apply on a native
          reload, whose state dict is already in this key space;
        * the vocabulary head is widened if its width is not GEMM-aligned (a
          no-op for the released 13088).

        The Conv2d NCHW→NHWC and depthwise-conv1d layout conversions are handled
        by the layers' own ``_load_from_state_dict`` hooks, which are shape-guarded
        and therefore idempotent across a native round trip.
        """
        remapped: Dict[str, torch.Tensor] = {}
        dropped = []
        for key, value in state_dict.items():
            for old, new in _RENAMES.items():
                if key.startswith(old):
                    remapped[new + key[len(old) :]] = value
                    break
            else:
                if key.startswith(("encoder.", "decoder.", "joint.", "prompt_projector.")):
                    remapped[key] = value
                else:
                    dropped.append(key)

        self._permute_subsampling_projection(remapped)
        pad_output_projection(remapped, "joint.head.", self._joint_out)

        missing, unexpected = self.load_state_dict(remapped, strict=strict)
        if missing:
            logger.warning("Missing keys when loading nemotron weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading nemotron weights: %s", unexpected)
        return LoadReport.build(remapped, missing, unexpected, dropped)

    def _permute_subsampling_projection(self, sd: Dict[str, torch.Tensor]) -> None:
        """Reorder ``encoder.subsampling.linear.weight`` columns ``(c, f)`` → ``(f, c)``."""
        key = "encoder.subsampling.linear.weight"
        weight = sd.get(key)
        if weight is None:
            return
        subsampling = cast(NemotronSubsampling, self.encoder.subsampling)
        channels, freq = subsampling.flatten_order()
        if weight.shape[1] != channels * freq:
            raise ValueError(
                f"{key} has {weight.shape[1]} input features but the subsampling "
                f"stack flattens to {channels} channels x {freq} frequency bins "
                f"({channels * freq}); check num_mel_bins / subsampling_factor"
            )
        sd[key] = (
            weight.view(weight.shape[0], channels, freq)
            .transpose(1, 2)
            .reshape(weight.shape[0], freq * channels)
            .contiguous()
        )


__all__ = ["NemotronEncoderConfig", "NemotronModel", "NemotronModelConfig"]
