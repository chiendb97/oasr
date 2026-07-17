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
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Type

import torch

from ..request import Request, RequestOutput

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

    # -- offline -----------------------------------------------------------
    @abstractmethod
    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        """Decode a batched encoder output.

        Returns one :class:`RequestOutput` per batch row (``finished=True``,
        ``request_id=""`` — the executor fills the id), in batch order.
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
    """
    return get_decode_strategy_class(decode_type, config)(config, detok, model)
