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
(CTC — encoder+head fused, the CUDA-graph fast path) or ``"hidden"`` (raw encoder
states for autoregressive families that own their head/decoder).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Dict, List, Type

import torch

from ..request import Request, RequestOutput

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer


class DecodeStrategy(ABC):
    """Pluggable decoding algorithm for one decode family.

    Subclasses set :attr:`decode_type` (and, for CTC, register under the
    ``decoder_type`` name e.g. ``"ctc_cuda"`` / ``"ctc_wfst"``).  Streaming
    session methods default to no-ops so stateless strategies need not override
    them.
    """

    #: Decode family this strategy serves ("ctc", "transducer", "aed", "llm").
    decode_type: ClassVar[str]
    #: Encoder output the engine feeds: "log_probs" (fused head) or "hidden".
    consumes: ClassVar[str] = "log_probs"

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
    """Resolve the registry key from the model's decode family + engine config.

    CTC splits into GPU vs WFST by ``config.decoder_type``; AR families key
    directly on ``decode_type``.
    """
    if decode_type == "ctc":
        return config.decoder_type  # "ctc_cuda" | "ctc_wfst"
    return decode_type


def build_decode_strategy(
    decode_type: str,
    config: "EngineConfig",
    detok: "Detokenizer",
    model: "BaseAsrModel" = None,
) -> DecodeStrategy:
    """Construct the decode strategy for a model's ``decode_type``.

    ``model`` is threaded through so autoregressive strategies can reach
    ``model.decoder`` / ``model.joiner`` (CTC strategies ignore it).  Raises
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
    return cls(config, detok, model)
