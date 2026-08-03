# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Architecture-agnostic base classes for OASR ASR models.

This module defines the contract the inference engine relies on, so new
encoder architectures (Transformer, Branchformer, …) and decode heads (CTC,
Transducer, AED) can be added by *subclassing* rather than by editing the
engine.  The layering mirrors vLLM / SGLang ``model_executor``: small reusable
layers (:mod:`oasr.layers`) compose into an encoder (:class:`BaseEncoder`) plus
a head (:class:`BaseHead`), wrapped by a model (:class:`BaseAsrModel`) that the
engine drives through a stable interface.

What the engine touches on **every** model:

* :meth:`BaseAsrModel.from_config` / :meth:`load_weights` — construction;
* :attr:`BaseAsrModel.capabilities` / :attr:`default_decode_type` — which decode
  families this checkpoint can serve, and which one to run by default;
* :attr:`BaseAsrModel.cache_spec` (``None`` for offline-only encoders) and
  ``encoder.streaming_kind`` / ``subsampling_rate`` / ``right_context`` — cache
  sizing and streaming geometry;
* :meth:`encode_offline` (raw hidden) and/or the fused
  :meth:`forward_offline` / :meth:`forward_offline_packed` /
  :meth:`forward_chunk_paged` — which of these is called depends on the active
  strategy's ``consumes``.

Beyond that, **each decode family reaches for its own surface** — ``model.decoder``,
``model.joiner``, ``model.predict``/``nar_decode``, specific ``model.config`` fields.
That per-family requirement is not prose here: it is the declarative table in
:mod:`oasr.models.interfaces` (``CAPABILITIES``), which
``build_decode_strategy`` validates every model against once, and which
``tests/test_model_contract.py`` checks against every registered architecture.  Read
that table for the authoritative list; keep it in sync when a family's needs change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import nn

if TYPE_CHECKING:
    from oasr.cache.paged_kv import PagedKVCache
    from oasr.cache.slot_cnn import SlotCnnCache
    from oasr.models.decoders import BaseDecoder

# Streaming-cache model an encoder uses, read by the engine to select a
# ``StreamingEncoderBackend``.  Values: "paged" (engine paged-KV + slot-CNN,
# Conformer-style), "stateful" (encoder owns per-layer recurrent state,
# Zipformer-style), "none" (offline-only).  Kept a plain ``str``.
StreamingKind = str

# Decode-path selector, resolved to a registered ``DecodeStrategy`` by
# ``oasr.engine.decode.build_decode_strategy``.  Kept a plain ``str`` so heads and
# decoders can declare it without importing an enum.  Wired families: "ctc"
# (splitting into ``ctc_cuda`` / ``ctc_wfst`` on ``EngineConfig.decoder_type``),
# "transducer", "ctc_aed_rescoring", "aed", "llm", "paraformer".
DecodeType = str


@dataclass(frozen=True)
class CacheSpec:
    """Architecture-agnostic descriptor the engine needs to size streaming caches.

    Replaces the engine reaching into Conformer-specific config fields.  An
    encoder with no convolutional left-context (e.g. a plain Transformer)
    reports ``conv_kernel_size == 1`` → zero CNN-cache frames.
    """

    num_layers: int
    n_kv_head: int
    head_dim: int
    hidden_dim: int
    conv_kernel_size: int = 1


@dataclass
class LoadReport:
    """Weight-load accounting returned by :meth:`BaseAsrModel.load_weights`.

    Kills silent weight drops: every checkpoint key is either *mapped* into the
    model or listed in *dropped*; *missing* holds model keys the checkpoint did
    not fill (beyond declared computed buffers).  The registry cross-references
    *dropped* against the converter's ``expected_unused_prefixes`` /
    ``capability_drop_hints`` and logs a warning naming any capability lost.
    """

    mapped: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.mapped)} tensors loaded"]
        if self.dropped:
            parts.append(f"{len(self.dropped)} checkpoint tensors dropped")
        if self.missing:
            parts.append(f"{len(self.missing)} model tensors not filled")
        return "LoadReport: " + ", ".join(parts)

    @classmethod
    def build(
        cls,
        loaded: Mapping[str, Any],
        missing: Sequence[str],
        unexpected: Sequence[str],
        dropped: Optional[Sequence[str]] = None,
        expected_missing: Sequence[str] = (),
    ) -> "LoadReport":
        """Assemble a report from a ``load_state_dict`` result.

        One builder because the six models disagreed on two things that matter.
        *Membership*: four tested ``k not in unexpected`` against a **list**, so
        the accounting was O(n²) in checkpoint size — noticeable on an 8 B
        model. *Completeness*: only two folded ``unexpected`` into ``dropped``,
        so for the other four the registry's capability-drop check never saw the
        keys the model actually refused, which is exactly the silent weight drop
        :class:`LoadReport` exists to prevent.

        ``expected_missing`` names model keys a checkpoint is not required to
        fill (computed buffers such as Conformer's ``pos_enc.pe``); they are
        filtered out rather than reported.
        """
        unexpected_set = set(unexpected)
        skip = set(expected_missing)
        return cls(
            mapped=[k for k in loaded if k not in unexpected_set],
            dropped=list(dropped or ()) + list(unexpected),
            missing=[k for k in missing if k not in skip],
        )


#: Bias given to padding rows of an aligned output projection.  Large enough
#: that ``exp`` underflows to 0 in fp16 (whose max is 65504, so this is
#: representable), small enough not to be an inf that could make a softmax
#: denominator NaN.
PAD_LOGIT = -1e4


def align_out_features(out_features: int, alignment: int = 8) -> int:
    """Round an output width up to what the GEMM kernels can address.

    CUTLASS 2.x alignment-8 iterators reject an unpadded vocabulary outright,
    so a projection of that width has no kernel at all.  Padding it is how the
    gap gets *closed* — the alternative, routing the call to torch, leaves the
    model permanently off the kernel path and hides the fact.
    """
    return ((out_features + alignment - 1) // alignment) * alignment


def init_pad_rows(projection: Any, raw_out_features: int) -> None:
    """Neutralize the padding rows of a freshly constructed aligned projection.

    :func:`pad_output_projection` establishes "a padding class can never win an
    argmax" when a *checkpoint* is loaded.  A module that has only been
    constructed has random values there, so the invariant would hold by load
    order rather than by construction — and a test (or any consumer that loads a
    state dict non-strictly) can observe a padding class winning.  Setting the
    rows here makes it true unconditionally; a subsequent load overwrites them
    with the same values.
    """
    with torch.no_grad():
        if projection.weight.shape[0] > raw_out_features:
            projection.weight[raw_out_features:].zero_()
        bias = getattr(projection, "bias", None)
        if bias is not None and bias.shape[0] > raw_out_features:
            bias[raw_out_features:].fill_(PAD_LOGIT)


def pad_output_projection(
    state_dict: MutableMapping[str, torch.Tensor],
    prefix: str,
    target_out: int,
) -> None:
    """Widen an output projection in a checkpoint to ``target_out`` rows, in place.

    Alignment happens **here**, when the checkpoint is loaded, rather than as a
    pad-and-slice inside the layer: the released weights keep their true width,
    the waist's :class:`~oasr.layers.linear.Linear` stays a plain projection,
    and the cost is paid once instead of per forward.

    Padding rows get zero weights and a :data:`PAD_LOGIT` bias, so a
    padding class emits a logit far below any real one and can never win an
    argmax or take meaningful mass in a softmax.  Zeroing the bias too — the
    obvious thing, and what this codebase did first — leaves the pad classes
    at logit ``0.0``, which beats every real class whenever they are all
    negative.  It has never been observed to bite, but it is free to rule out.

    ``prefix`` names the projection (``"decoder.output_layer."``); missing keys
    and already-wide keys are left alone, so this is safe to call
    unconditionally and safe to call on a native checkpoint that was already
    saved padded.
    """
    weight = state_dict.get(prefix + "weight")
    if weight is None or weight.shape[0] >= target_out:
        return
    pad = target_out - weight.shape[0]
    state_dict[prefix + "weight"] = torch.cat(
        [weight, weight.new_zeros(pad, weight.shape[1])], dim=0
    )
    bias = state_dict.get(prefix + "bias")
    if bias is not None:
        state_dict[prefix + "bias"] = torch.cat([bias, bias.new_full((pad,), PAD_LOGIT)], dim=0)


def coerce_config(cls: type, d: Mapping[str, Any]) -> Any:
    """Build dataclass ``cls`` from ``d``, coercing values by declared field type.

    The native checkpoint format writes configs with a generic ``asdict``
    (:mod:`oasr.checkpoints.native`) but every config hand-wrote the read side, in
    **four** different spellings of "filter to known fields": ``__dataclass_fields__``,
    ``hasattr(SomeConfig, k)`` (which admits properties and methods, and misses a field
    declared without a default — such a field is not a class attribute), a hardcoded
    ``known`` tuple of field names, and two of the six additionally hand-restoring
    tuples with ``tuple(v) if isinstance(v, list) else v``.

    None of them was losing data at the time this was written — checked, rather than
    assumed: no config had a defaultless field, ``ConformerEncoderConfig`` exposed no
    public non-field attribute, and the two configs with ``Tuple`` fields were the two
    that hand-restored them.  The problem is that each spelling fails on the *next*
    edit, differently and quietly: add a field and the hardcoded tuple drops it (it
    already omitted ``model_type`` / ``encoder_type``); add a ``Tuple`` field to a config
    without the ad-hoc restore and it comes back a ``list``, which compares unequal and
    breaks anything that indexes it as a tuple; add a property whose name collides with
    a checkpoint key and ``hasattr`` lets it through into the constructor.  One reader
    driven by the declared types removes the class of bug instead of the instances, and
    ``tests/test_config_round_trip.py`` now pins it for every registered architecture.

    Coercions, all derived from the annotation rather than from the value:

    * ``Tuple[X, ...]`` → ``tuple`` (JSON has no tuples, so every tuple field came
      back as a list — the reason two configs carried an ad-hoc
      ``tuple(v) if isinstance(v, list) else v``);
    * ``List[X]`` / ``Tuple[X, Y]`` → elements coerced recursively, which is how
      ``List[Tuple[int, int]]`` (Whisper's ``forced_decoder_ids``) round-trips;
    * a nested dataclass → recursed into;
    * ``Optional[X]`` → ``None`` passes through, anything else coerces to ``X``;
    * ``Any`` and primitives → left alone.

    Unknown keys are ignored (checkpoint configs legitimately carry extra keys).
    A field whose declared type cannot be resolved is passed through untouched
    rather than raising: a config is data, and refusing to load one because an
    annotation is exotic would be worse than under-coercing it.
    """
    import dataclasses
    import typing

    try:
        hints = typing.get_type_hints(cls)
    except Exception:  # pragma: no cover - unresolvable forward refs
        hints = {}

    overrides = getattr(cls, "_from_dict_overrides", {})
    kwargs: dict = {}
    for name in getattr(cls, "__dataclass_fields__", {}):
        if name in overrides:
            hook = overrides[name]
            if (out := hook(d)) is not _UNSET:
                kwargs[name] = out
            continue
        if name not in d:
            continue
        kwargs[name] = _coerce_value(hints.get(name, typing.Any), d[name], dataclasses, typing)
    return cls(**kwargs)


class _Unset:
    """Sentinel: an override hook declining to supply a value."""


_UNSET = _Unset()


def _coerce_value(hint: Any, value: Any, dataclasses, typing) -> Any:
    """Coerce one JSON value against one declared type hint."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)

    # Optional[X] / Union[...]: None passes through; otherwise try the first
    # non-None member.  Configs do not use genuinely ambiguous unions.
    if origin is typing.Union:
        if value is None:
            return None
        for arg in args:
            if arg is not type(None):
                return _coerce_value(arg, value, dataclasses, typing)
        return value

    if origin in (tuple, list) and isinstance(value, (list, tuple)):
        # Tuple[X, ...] is homogeneous; Tuple[X, Y] is positional.
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            elem_hints = [args[0]] * len(value)
        elif origin is tuple and args:
            elem_hints = list(args)
        elif args:
            elem_hints = [args[0]] * len(value)
        else:
            elem_hints = [typing.Any] * len(value)
        coerced = [_coerce_value(h, v, dataclasses, typing) for h, v in zip(elem_hints, value)]
        return tuple(coerced) if origin is tuple else coerced

    if dataclasses.is_dataclass(hint) and isinstance(value, Mapping):
        return coerce_config(hint, value)

    return value


@dataclass
class BaseModelConfig:
    """Common model-config fields shared by every architecture.

    Architecture-specific configs subclass this and add their own
    hyperparameters (e.g. :class:`~oasr.models.conformer.ConformerModelConfig`).
    ``model_type`` keys the model registry; ``vocab_size`` is read by the engine
    and the serving layer.

    :meth:`from_dict` is inherited and type-driven (see :func:`coerce_config`), so a
    subclass normally needs no reader at all.  Override
    :attr:`_from_dict_overrides` — ``{field_name: hook(full_dict) -> value}`` — only
    for a field the annotation cannot describe: a polymorphic one
    (``TransducerModelConfig.encoder``, whose class depends on a sibling
    ``encoder_type`` key) or one read from a legacy flat layout.
    """

    model_type: str = "base"
    vocab_size: Optional[int] = None

    #: ``{field: hook(source_dict) -> value | _UNSET}``.  See the class docstring.
    _from_dict_overrides: ClassVar[Mapping[str, Any]] = {}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BaseModelConfig":
        """Build from a native ``oasr_config.json`` (or any superset dict)."""
        return coerce_config(cls, d)


class BaseHead(nn.Module, ABC):
    """Output (decode-side) head: projects encoder hidden states for decoding.

    ``decode_type`` tells the engine which decode algorithm to run (CTC beam
    search, transducer beam search, …).
    """

    decode_type: DecodeType = "ctc"

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map encoder hidden ``(B, T, D)`` to the head's output tensor."""
        raise NotImplementedError


class BaseEncoder(nn.Module, ABC):
    """Acoustic encoder contract: offline, streaming-paged, and (optional) packed.

    The forward signatures match exactly what
    :class:`oasr.engine.model_runner.ModelRunner` and
    :class:`oasr.engine.graph_cache.EncoderGraphCache` call, so any conforming
    encoder plugs into the engine unchanged.  The introspection properties feed
    :class:`CacheSpec`.
    """

    #: Whether :meth:`forward_packed` is implemented (sequence packing).
    supports_packing: bool = False
    #: Whether :meth:`forward_chunk_paged` (paged-KV streaming) is implemented.
    #: Conformer-style encoders set this True; encoders with a different
    #: streaming-cache model (e.g. Zipformer) leave it False and expose their
    #: own streaming API instead.
    supports_paged_streaming: bool = False

    @abstractmethod
    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Offline forward → ``(hidden (B, T_out, D), masks (B, 1, T_out) bool)``."""
        raise NotImplementedError

    def forward_chunk_paged(
        self,
        xs: torch.Tensor,
        offset: Union[int, torch.Tensor],
        att_caches: List["PagedKVCache"],
        cnn_cache: "SlotCnnCache",
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
        cache_t1: int = -1,
    ) -> torch.Tensor:
        """Streaming chunk forward (paged KV + slot CNN cache) → ``(B, chunk, D)``.

        Default: unsupported.  Only encoders whose streaming cache maps onto the
        engine's paged-KV + slot-CNN model implement this (``supports_paged_streaming
        = True``).  Other encoders expose their own streaming API.
        """
        del xs, offset, att_caches, cnn_cache, att_mask, cache_t1
        raise NotImplementedError(f"{type(self).__name__} does not support paged-KV streaming")

    def forward_packed(
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-packing offline forward (optional).  Default: unsupported."""
        del xs, xs_lens
        raise NotImplementedError(f"{type(self).__name__} does not support sequence packing")

    # -- introspection used to build CacheSpec ------------------------------
    @property
    @abstractmethod
    def num_encoder_layers(self) -> int:
        """Number of encoder layers (== paged KV cache layers)."""
        raise NotImplementedError

    # ``n_kv_head`` / ``head_dim`` describe the *engine's paged-KV* layout, so only
    # ``streaming_kind="paged"`` encoders need them.  They were abstract, which
    # forced every offline-only encoder (Whisper, Paraformer SANM, the Qwen2-Audio
    # tower) to implement two properties purely to satisfy the ABC — ceremony that
    # reads as a requirement.  They are now paired with the existing
    # ``supports_paged_streaming`` flag: an encoder that sets it True must override
    # both, and one that does not need not.  Deliberately a raising default rather
    # than a separate mixin the paged encoders inherit — the default is reachable
    # from any encoder reference and says *why* the geometry is absent, which an
    # ``AttributeError`` from a missing mixin would not.
    @property
    def n_kv_head(self) -> int:
        """Number of KV attention heads per layer (paged streaming only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not declare a paged-KV layout "
            "(n_kv_head/head_dim); those are required only for "
            'streaming_kind="paged" encoders.'
        )

    @property
    def head_dim(self) -> int:
        """Per-head key/value dimension (paged streaming only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not declare a paged-KV layout "
            "(n_kv_head/head_dim); those are required only for "
            'streaming_kind="paged" encoders.'
        )

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Encoder hidden dimension."""
        raise NotImplementedError

    @property
    def conv_kernel_size(self) -> int:
        """Depthwise-conv kernel for streaming left-context; 1 == no CNN cache."""
        return 1

    # -- streaming spec (read by the engine to pick a StreamingEncoderBackend) --
    @property
    def streaming_kind(self) -> "StreamingKind":
        """Streaming-cache model this encoder uses.

        ``"paged"`` — the engine's paged-KV + slot-CNN cache (Conformer-style);
        the encoder implements :meth:`forward_chunk_paged`.
        ``"stateful"`` — the encoder owns per-layer recurrent state
        (Zipformer-style); it implements :meth:`get_streaming_init_states` /
        :meth:`streaming_forward` instead.
        ``"none"`` — offline only.

        Default derives from :attr:`supports_paged_streaming`; stateful encoders
        override this to return ``"stateful"``.
        """
        return "paged" if self.supports_paged_streaming else "none"

    @property
    def subsampling_rate(self) -> int:
        """Total temporal subsampling factor (input frames per encoder frame)."""
        return 1

    @property
    def right_context(self) -> int:
        """Extra future input frames the subsampling needs beyond one chunk."""
        return 0

    # -- stateful streaming (``streaming_kind == "stateful"`` encoders) --------
    def get_streaming_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> List[torch.Tensor]:
        """Initial per-request streaming state for a stateful encoder.

        Default: unsupported.  Only ``streaming_kind == "stateful"`` encoders
        implement this (paged encoders use :meth:`forward_chunk_paged`).
        """
        del batch_size, device, dtype
        raise NotImplementedError(f"{type(self).__name__} does not expose a stateful streaming API")

    def streaming_forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Stateful chunk forward → ``(hidden (B, chunk, D), out_lens, new_states)``.

        Default: unsupported (see :meth:`get_streaming_init_states`).
        """
        del xs, xs_lens, states
        raise NotImplementedError(f"{type(self).__name__} does not expose a stateful streaming API")

    @property
    def cache_spec(self) -> Optional[CacheSpec]:
        """Streaming cache descriptor derived from the live encoder dims.

        ``None`` for an offline-only encoder (``streaming_kind == "none"``): there
        is no streaming cache to size, and asking for one would mean reporting a
        paged-KV geometry the encoder does not have.  The engine skips building a
        ``CacheConfig`` in that case, so no paged pool is allocated.
        """
        if self.streaming_kind == "none":
            return None
        return CacheSpec(
            num_layers=self.num_encoder_layers,
            n_kv_head=self.n_kv_head,
            head_dim=self.head_dim,
            hidden_dim=self.output_size,
            conv_kernel_size=self.conv_kernel_size,
        )


class BaseAsrModel(nn.Module, ABC):
    """Encoder + head ASR model the engine drives.

    Subclasses set ``self.encoder`` (a :class:`BaseEncoder`) and expose
    ``self.head`` (a :class:`BaseHead`, possibly via a property aliasing a
    differently-named submodule for checkpoint compatibility), then implement
    :meth:`from_config` and :meth:`load_weights`.  The offline / packed /
    streaming entry points and the engine-facing :attr:`cache_spec` /
    :attr:`decode_type` are provided here so the runner stays
    architecture-agnostic.
    """

    encoder: BaseEncoder
    head: BaseHead
    # Autoregressive decoder (transducer / AED / LLM).  CTC models leave this
    # unset and use :attr:`head` instead; AR model subclasses register
    # ``self.decoder = <BaseDecoder>`` in ``__init__``.  Declared as a bare
    # annotation (no class-level value) so it never shadows the registered
    # submodule via ``nn.Module.__getattr__``; read it with
    # ``getattr(self, "decoder", None)``.
    decoder: Optional["BaseDecoder"]

    # -- construction & weights --------------------------------------------
    @classmethod
    @abstractmethod
    def from_config(cls, config: BaseModelConfig, **aux: Any) -> "BaseAsrModel":
        """Build a model (random weights) from its config + format aux buffers."""
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_id_or_path: str, **kwargs: Any) -> "BaseAsrModel":
        """Load a weight-loaded model from a local dir or HuggingFace Hub id.

        Thin convenience wrapper over :func:`oasr.models.from_pretrained` that
        **auto-detects** the architecture from the checkpoint (so the concrete
        subclass it is called on is advisory).  Returns the model only; use the
        module-level function if you also need the config object.
        """
        from oasr.models.loaders import from_pretrained as _from_pretrained

        model, _config = _from_pretrained(model_id_or_path, **kwargs)
        return model

    #: State-dict keys the model recomputes from config (e.g. positional-encoding
    #: tables).  The native checkpoint format skips them at save time and
    #: tolerates them missing at load time.  Suffix-matched against full keys.
    _computed_buffer_suffixes: Tuple[str, ...] = ()

    @abstractmethod
    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> Optional["LoadReport"]:
        """Map an external checkpoint state-dict into this model's parameters.

        Each architecture owns the name-mapping / fusion knowledge (vLLM-style);
        in-module reshaping (e.g. fused QKV, conv reorder) is handled by the
        layers' ``_load_from_state_dict`` hooks.  Returns a :class:`LoadReport`
        (``None`` allowed for legacy implementations) so no checkpoint tensor is
        ever dropped silently.
        """
        raise NotImplementedError

    # -- engine-facing metadata --------------------------------------------
    @property
    def cache_spec(self) -> Optional[CacheSpec]:
        """Streaming cache descriptor, or ``None`` for an offline-only encoder."""
        return self.encoder.cache_spec

    @property
    def streaming_kind(self) -> "StreamingKind":
        """Streaming-cache model (delegates to the encoder)."""
        return self.encoder.streaming_kind

    @property
    def default_decode_type(self) -> DecodeType:
        """Decode family the engine runs when the caller picks none.

        AR models declare it via :attr:`decoder`; CTC models via :attr:`head`.
        Hybrid models (e.g. CTC + AED-rescoring) override this to name their
        production default and expose the rest via :attr:`capabilities`.
        """
        decoder = getattr(self, "decoder", None)
        if decoder is not None:
            return decoder.decode_type
        return self.head.decode_type

    @property
    def capabilities(self) -> frozenset:
        """Decode families this checkpoint's weights support.

        ``EngineConfig.decode_method`` must name one of these (``None`` selects
        :attr:`default_decode_type`).  Single-objective models get the derived
        one-element set for free; hybrids (U2++ CTC+AED) override to advertise
        every branch they loaded.
        """
        return frozenset({self.default_decode_type})

    @property
    def decode_type(self) -> DecodeType:
        """Compatibility alias for :attr:`default_decode_type`."""
        return self.default_decode_type

    # -- engine-facing forward entry points --------------------------------
    @staticmethod
    def _lengths_from_mask(masks: torch.Tensor) -> torch.Tensor:
        """``(B, 1, T)`` bool mask → ``(B,)`` int32 valid output lengths."""
        return masks.squeeze(1).sum(dim=-1).to(torch.int32)

    # -- encoder-only (acoustic hidden states; AR decode strategies use these) --
    def encode_offline(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched offline encode → ``(hidden (B, T, D), out_lengths (B,))``.

        Returns the raw encoder hidden states (no head/decoder).  CTC decoding
        goes through :meth:`forward_offline` (encoder+head fused for the
        CUDA-graph fast path); autoregressive families consume this hidden.
        """
        hidden, masks = self.encoder(features, lengths)
        return hidden, self._lengths_from_mask(masks)

    def encode_offline_packed(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-packing offline encode → ``(hidden, out_lengths)``."""
        if not self.encoder.supports_packing:
            raise NotImplementedError(
                f"{type(self).__name__} encoder does not support sequence packing"
            )
        hidden, masks = self.encoder.forward_packed(features, lengths)
        return hidden, self._lengths_from_mask(masks)

    def encode_chunk_paged(
        self,
        input_features: torch.Tensor,
        offset: Union[int, torch.Tensor],
        att_caches: List["PagedKVCache"],
        cnn_cache: "SlotCnnCache",
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
        cache_t1: int = -1,
    ) -> torch.Tensor:
        """Streaming chunk encode → encoder hidden ``(B, chunk, D)`` (no head)."""
        return self.encoder.forward_chunk_paged(
            input_features, offset, att_caches, cnn_cache, att_mask, cache_t1
        )

    # -- encoder + head fused (CTC fast path; CUDA-graph captured) -------------
    def forward_offline(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched offline forward → ``(log_probs (B, T, V), out_lengths (B,))``."""
        hidden, out_lengths = self.encode_offline(features, lengths)
        return self.head(hidden), out_lengths

    def forward_offline_packed(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-packing offline forward → ``(log_probs, out_lengths)``."""
        hidden, out_lengths = self.encode_offline_packed(features, lengths)
        return self.head(hidden), out_lengths

    def forward_chunk_paged(
        self,
        input_features: torch.Tensor,
        offset: Union[int, torch.Tensor],
        att_caches: List["PagedKVCache"],
        cnn_cache: "SlotCnnCache",
        att_mask: torch.Tensor = torch.zeros((0, 0, 0)),
        cache_t1: int = -1,
    ) -> torch.Tensor:
        """Streaming chunk forward → head output ``(B, chunk, V)``."""
        hidden = self.encode_chunk_paged(
            input_features, offset, att_caches, cnn_cache, att_mask, cache_t1
        )
        return self.head(hidden)
