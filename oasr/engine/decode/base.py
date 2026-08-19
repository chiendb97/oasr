# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Decode-strategy contract + registry.

A :class:`DecodeStrategy` turns encoder output into detokenized text.  It is the
seam that decouples *decoding* from the engine: CTC (GPU prefix-beam / WFST),
transducer, AED, and LLM families each implement this interface and register
under a name, so adding a decode family never edits the engine core.

The engine drives a strategy through ``OutputProcessor`` (a thin facade):

* **offline** — :meth:`decode_offline` over a batched encoder output (one-shot);
* **streaming** — :meth:`create_session` per admitted request, :meth:`decode_streaming_batch`
  per tick over the ready streams, :meth:`finalize` on drain, :meth:`free_session`
  on finalize/abort.

``consumes`` declares what the runner should feed the strategy: ``"log_probs"``
(CTC — encoder+head fused, the CUDA-graph fast path), ``"hidden"`` (raw encoder
states for autoregressive families that own their head/decoder), or ``"both"``
(one encoder pass + head applied — an :class:`EncodeOutput` carrying hidden
*and* log-probs, needed for CTC+AED rescoring).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import torch

from ..request import DecodingOptions, Request, RequestOutput
from .alignment import (
    AlignmentFields,
    FrameClock,
    TokenAlignment,
    alignment_fields,
    emission_fields,
)

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from ..generation import StepBudget
    from .detokenize import Detokenizer


@dataclass
class EncodeOutput:
    """Encoder products for strategies consuming more than one tensor.

    The offline executor passes a plain hidden / log-probs tensor for
    ``consumes == "hidden"`` / ``"log_probs"`` (the unchanged fast paths) and
    an :class:`EncodeOutput` for ``consumes == "both"`` — one encoder forward,
    both views.  Lengths stay a separate argument (same for every view).
    """

    hidden: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None


class DecodeStrategy(ABC):
    """Pluggable decoding algorithm for one decode family.

    Subclasses set :attr:`decode_type` (and, for CTC, register under the
    ``decoder_type`` name e.g. ``"ctc_cuda"`` / ``"ctc_wfst"``).  Streaming
    session methods default to no-ops so stateless strategies need not override
    them.
    """

    #: Decode family this strategy serves ("ctc", "transducer", "aed", "llm").
    decode_type: ClassVar[str]
    #: Encoder output the engine feeds: "log_probs" (fused head), "hidden",
    #: or "both" (an :class:`EncodeOutput` with hidden + log-probs).
    consumes: ClassVar[str] = "log_probs"
    #: Label-synchronous AR strategies (AED / LLM) set this True and implement
    #: the incremental protocol below; the offline executor then runs bounded
    #: decoder steps per tick instead of the one-shot :meth:`decode_offline`.
    incremental: ClassVar[bool] = False
    #: Options dataclass this family owns, or ``None`` for a family with no
    #: knobs.  Resolved into :attr:`options` by the constructor — see
    #: :mod:`oasr.engine.decode.options`.  Declaring one here is what keeps a
    #: new family from having to add fields to ``EngineConfig``.
    options_cls: ClassVar[Optional[type]] = None

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        """Store the three things every strategy gets, validate the model,
        and resolve this family's options.

        The capability check lives **here** rather than only in
        :func:`build_decode_strategy` so that constructing a strategy directly —
        which is public, and what tests do — is guarded too.  A strategy whose
        model lacks the surface it will reach for should say so now, with the
        missing members named, instead of raising ``AttributeError`` from the
        middle of a decode.
        """
        from oasr.models.interfaces import require_capability

        from .options import build_options

        require_capability(model, self.decode_type, decode_method=self.decode_type)
        self._config = config
        self._detok = detok
        self._model = model
        #: This family's resolved options (``None`` when ``options_cls`` is).
        self.options = build_options(self.options_cls, config)
        #: Encoder-frame → seconds conversion for word timings, or ``None``
        #: when the geometry cannot be resolved (no model, no feature config).
        #: :meth:`validate_options` refuses ``word_timestamps`` in that case
        #: rather than emitting timings against a guessed frame rate.
        self._clock: Optional[FrameClock] = FrameClock.resolve(
            getattr(config, "feature_config", None), model
        )

    # -- per-request options ------------------------------------------------

    #: Options that change what is decoded. Unsupported entries are rejected;
    #: sampling-only controls may be ignored when they cannot change the result.
    selective_options: ClassVar[Tuple[str, ...]] = ()

    @property
    def word_timing_modes(self) -> Tuple[str, ...]:
        """Supported word-timing modes for this configured strategy instance.

        Returns a subset of ``{"offline", "streaming"}``; support may depend on
        search configuration and available alignment surfaces.
        """
        return ()

    #: Selective option → the value that means "the caller did not ask".  Drives
    #: :meth:`validate_options`; ``word_timestamps`` is a bool, so its unset
    #: value is ``False`` rather than ``None`` and a hardcoded ``is None`` test
    #: would silently accept it everywhere.
    _SELECTIVE_UNSET: ClassVar[Mapping[str, object]] = {
        "task": None,
        "language": None,
        "word_timestamps": False,
    }

    def validate_options(
        self, options: Optional["DecodingOptions"], *, streaming: bool = False
    ) -> None:
        """Reject per-request options this family cannot act on.

        Called once per request at admission
        (``ASREngine._admit_one_checked``), so a rejection is scoped to its own
        request rather than to the admit batch it was coalesced into.
        ``streaming`` is the mode the request will run in — the only selective
        option whose answer depends on it is ``word_timestamps``.

        Subclasses that *can* act on a selective option override this to add
        the checkpoint-level check — a Whisper checkpoint knows a fixed set of
        language tokens, and ``language="xx"`` has to fail here rather than
        silently fall back to the checkpoint's own.
        """
        if options is None:
            return
        for name, unset in self._SELECTIVE_UNSET.items():
            if getattr(options, name, unset) == unset:
                continue
            if name == "word_timestamps":
                self._require_word_timings(streaming)
            elif name not in self.selective_options:
                raise ValueError(
                    f"decode_method={self.decode_type!r} cannot honour the "
                    f"per-request {name!r} option (it has no {name} control); "
                    "remove it, or serve a checkpoint whose decode family does "
                    f"(supported here: {list(self.selective_options) or 'none'})"
                )

    def _require_word_timings(self, streaming: bool) -> None:
        """Raise unless this family can align in the mode the request will run in."""
        mode = "streaming" if streaming else "offline"
        if mode not in self.word_timing_modes:
            supported = ", ".join(self.word_timing_modes) or "neither mode"
            raise ValueError(
                f"decode_method={self.decode_type!r} cannot produce word "
                f"timestamps for a {mode} request (supported: {supported}); "
                "drop word_timestamps, or send the audio to an offline engine"
            )
        if self._clock is None:
            # Emitting timings against a guessed frame rate is worse than
            # refusing: every span would be plausible and uniformly wrong.
            raise ValueError(
                f"decode_method={self.decode_type!r} cannot resolve its "
                "encoder frame rate (feature config or encoder subsampling "
                "unavailable), so word timestamps would be scaled by an "
                "unknown constant"
            )

    # -- alignment ----------------------------------------------------------

    def attach_alignment(
        self,
        output: RequestOutput,
        alignments: Sequence[TokenAlignment],
        *,
        words: bool = True,
        offset: float = 0.0,
    ) -> None:
        """Fill ``timestamps`` / ``words`` / ``confidence`` from token alignments.

        The one call every family makes once it has produced its own
        :class:`~oasr.engine.decode.alignment.TokenAlignment` list, so the
        frames→seconds conversion, the token→word grouping and the confidence
        aggregation are written once rather than per family.  ``words=False``
        keeps the per-token timings and the confidence but skips the word pass,
        for a family (Paraformer) whose alignment is free and therefore always
        computed, on a request that did not ask for words.
        """
        self._write_alignment(
            output,
            alignment_fields(alignments, self._detok, self._clock, offset=offset, want_words=words),
        )

    def attach_emission_alignment(
        self,
        output: RequestOutput,
        tokens: Sequence[int],
        frames: Sequence[int],
        confidences: Optional[Sequence[float]] = None,
        *,
        offset: float = 0.0,
        frame_offset: int = 0,
    ) -> None:
        """:meth:`attach_alignment` for a family that recorded emission frames.

        Same result as building :class:`TokenAlignment` values and passing them
        in — and the point of having both is that this route never builds them:
        frames, posteriors and rendered pieces go into one call and words come
        back, so a 400-token micro-batch costs no per-token Python object at
        all.  CTC and the transducer both take it.
        """
        self._write_alignment(
            output,
            emission_fields(
                tokens,
                frames,
                confidences,
                self._detok,
                self._clock,
                offset=offset,
                frame_offset=frame_offset,
            ),
        )

    @staticmethod
    def _write_alignment(output: RequestOutput, fields: AlignmentFields) -> None:
        """Publish an alignment pass onto the output, leaving absent what is absent."""
        if fields.timestamps is None:
            return
        output.timestamps = fields.timestamps
        output.confidence = fields.confidence
        if fields.words is not None:
            output.words = fields.words

    # -- offline -----------------------------------------------------------
    @abstractmethod
    def decode_offline(
        self,
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
        requests: Optional[List[Request]] = None,
    ) -> List[RequestOutput]:
        """Decode a batched encoder output.

        Returns one :class:`RequestOutput` per batch row (``finished=True``,
        ``request_id=""`` — the executor fills the id), in batch order.

        ``requests`` is the micro-batch in the same row order, so a family can
        read the per-request :class:`~oasr.engine.request.DecodingOptions` that
        change *what it computes* rather than only what it returns —
        ``word_timestamps`` today.  It is optional because the prewarm path and
        the tests decode without requests, and because a family that reads
        nothing from it must not have to accept one; it mirrors the
        ``begin_offline(requests, ...)`` the incremental families already get.
        """
        raise NotImplementedError

    # -- incremental offline protocol (``incremental = True`` strategies) ---
    def begin_offline(
        self,
        requests: List[Request],
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
    ) -> None:
        """Prefill for a freshly-encoded micro-batch: stash the encoder
        output, initialize per-request hypotheses + decoder state.  The
        requests stay ``RUNNING`` across engine steps; their outputs are
        produced by :meth:`advance`.  Only ``incremental = True`` strategies
        implement this."""
        raise NotImplementedError(f"{type(self).__name__} is not an incremental strategy")

    def advance(self, budget: "StepBudget") -> List[RequestOutput]:
        """Run at most ``budget.max_steps`` *batched* decoder steps across all
        pending requests (continuous batching) and return the outputs produced
        this tick — partials (``finished=False``) and/or finals.  The executor
        finalizes requests whose output has ``finished=True``."""
        raise NotImplementedError(f"{type(self).__name__} is not an incremental strategy")

    def has_pending(self) -> bool:
        """Whether any request begun via :meth:`begin_offline` is unfinished."""
        return False

    def kv_bytes_per_row(self) -> Optional[int]:
        """Decoder-KV bytes one in-flight row can occupy, or ``None`` if unbounded.

        Admission uses this to budget in **bytes** rather than request count
        (C3): a batch of 30 s utterances preallocates far more decoder KV than
        the same number of 2 s ones, so a slot cap alone does not bound memory.
        ``None`` (the default, and every one-shot family) leaves the byte budget
        disabled — those families allocate no decoder KV at all.
        """
        return None

    # -- streaming session lifecycle --------------------------------------
    def create_session(self, request: Request) -> None:
        """Allocate per-request decode state on admission.  Default: no-op."""
        return None

    def free_session(self, request: Request) -> None:
        """Release per-request decode state on finalize/abort.  Default: no-op."""
        return None

    # -- streaming decode --------------------------------------------------
    @abstractmethod
    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        """Advance N ready streams one chunk; return any partial outputs."""
        raise NotImplementedError

    @abstractmethod
    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        """Advance a single stream one chunk; return a partial output."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self, request: Request) -> RequestOutput:
        """Finalize a stream and return its complete transcript."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[DecodeStrategy]] = {}


def register_decode_strategy(name: str):
    """Class decorator registering a :class:`DecodeStrategy` under ``name``.

    Names are the engine selectors: the CTC ``decoder_type`` values
    (``"ctc_cuda"`` / ``"ctc_wfst"``) and the AR ``decode_type`` values
    (``"transducer"`` / ``"aed"`` / ``"llm"``).
    """

    def _wrap(cls: Type[DecodeStrategy]) -> Type[DecodeStrategy]:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def _strategy_name(decode_type: str, config: "EngineConfig") -> str:
    """Resolve the registry key from the decode family + engine config.

    ``decode_type`` is either the model's default family or an explicit
    ``EngineConfig.decode_method`` capability name.  CTC splits into GPU vs
    WFST by ``config.decoder_type``; every other family keys directly.
    """
    if decode_type == "ctc":
        return config.decoder_type  # "ctc_cuda" | "ctc_wfst"
    return decode_type


def get_decode_strategy_class(decode_type: str, config: "EngineConfig") -> Type[DecodeStrategy]:
    """Resolve the strategy *class* for a model's ``decode_type``.

    Lets the engine read class-level strategy metadata (notably ``consumes``)
    **before** any component is constructed — the ``ModelRunner`` / streaming
    backends need it at build time, ahead of the ``OutputProcessor``.  Raises
    ``NotImplementedError`` with the available names when the family /
    ``decoder_type`` has no registered strategy (the extension point for new
    decode families).
    """
    name = _strategy_name(decode_type, config)
    cls = _REGISTRY.get(name)
    if cls is None:
        raise NotImplementedError(
            f"No decode strategy registered for decode_type={decode_type!r} "
            f"(resolved name {name!r}).  Registered: {sorted(_REGISTRY)}.  "
            "Add one by subclassing DecodeStrategy + @register_decode_strategy."
        )
    return cls


def build_decode_strategy(
    decode_type: str,
    config: "EngineConfig",
    detok: "Detokenizer",
    model: "BaseAsrModel" = None,
) -> DecodeStrategy:
    """Construct the decode strategy for a model's ``decode_type``.

    ``model`` is threaded through so autoregressive strategies can reach
    ``model.decoder`` / ``model.joiner`` (CTC strategies ignore it).

    The model's surface is validated here, once, against
    :data:`oasr.models.interfaces.CAPABILITIES` — so a checkpoint advertising a
    capability it cannot actually serve fails at engine construction with a
    message naming the missing members, instead of at first decode with an
    ``AttributeError`` (or not at all, for the families that used to check
    nothing).
    """
    from oasr.models.interfaces import require_capability

    require_capability(model, decode_type, decode_method=decode_type)
    return get_decode_strategy_class(decode_type, config)(config, detok, model)
