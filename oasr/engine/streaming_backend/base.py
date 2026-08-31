# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Streaming-encoder backend contract + registry.

A :class:`StreamingEncoderBackend` runs the encoder chunk-by-chunk for streaming
inference and owns whatever per-request encoder cache that streaming model needs.
It is the seam that lets different encoder streaming models share one engine:

* ``"paged"`` (:class:`~oasr.engine.streaming_backend.paged.PagedStreamingBackend`)
  — Conformer-style paged-KV + slot-CNN cache, CUDA-graph captured.
* ``"stateful"`` (:class:`~oasr.engine.streaming_backend.stateful.StatefulStreamingBackend`)
  — Zipformer-style per-layer recurrent state.

The engine (via :class:`~oasr.engine.model_runner.ModelRunner`) selects a backend
from ``model.encoder.streaming_kind`` and drives it through ``allocate`` /
``forward_step`` / ``free`` — never touching the encoder-specific cache directly.
``forward_step`` returns whatever the decode strategy consumes (CTC log-probs for
the fused-head fast path; raw hidden states for autoregressive families).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, ClassVar, Dict, List, Optional, Sequence, Type, cast

import torch

from ..request import Request

if TYPE_CHECKING:
    from oasr.cache.types import CacheConfig
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig


class StreamingEncoderBackend(ABC):
    """Per-tick streaming encoder runtime for one encoder streaming kind.

    ``forward_step`` produces what the active decode strategy declared via its
    ``consumes`` class attribute: ``"log_probs"`` (encoder + head fused) or
    ``"hidden"`` (raw encoder states).  Backends receive the value at
    construction (:func:`build_streaming_backend`).
    """

    #: Encoder ``streaming_kind`` this backend serves.
    streaming_kind: ClassVar[str]

    #: Whether this runtime allocates from the shared paged KV pool
    #: (:class:`~oasr.cache.block_pool.BlockPool`).  Declared rather than inferred
    #: because it decides two engine-level questions before the backend exists:
    #: whether a :class:`~oasr.cache.types.CacheConfig` is built at all, and
    #: whether ``EngineConfig.max_num_blocks=None`` has anything to derive.
    #: A recurrent-state runtime allocates its caches per request and needs
    #: neither, so probing VRAM for it would be work — and a possible startup
    #: failure — over a pool nothing will build.
    allocates_paged_pool: ClassVar[bool] = False

    # -- per-request cache lifecycle ---------------------------------------
    @abstractmethod
    def allocate(self, request: Request) -> None:
        """Allocate the per-request encoder cache on admission."""
        raise NotImplementedError

    @abstractmethod
    def free(self, request: Request) -> None:
        """Release the per-request encoder cache on finalize/abort."""
        raise NotImplementedError

    # -- per-tick forward --------------------------------------------------
    @abstractmethod
    def forward_step(self, requests: List[Request]) -> Dict[str, torch.Tensor]:
        """Run at most one encoder chunk per ready request.

        Returns ``{request_id: enc_out}`` (``(1, T_chunk, V|D)``) for the
        requests that produced output this tick; advances each request's
        feature cursor.
        """
        raise NotImplementedError

    def reset(self, request: Request) -> None:
        """Return this stream's encoder cache and position to their initial state.

        The primitive AGENTS.md rule 13 asks for.  Both backends assume every
        chunk is contiguous in encoder-frame time — the paged one derives
        ``cache_t1`` from ``request.offset``, which it advances only on a forward
        — so advancing a stream past frames the encoder never saw is only sound
        when the cache and the position go back to zero together.  That is why
        both halves live in one call: a caller that reset the cache and forgot
        the offset would splice the next chunk onto the old turn's positions, and
        the transcript would stay plausible.

        The default is ``free`` + ``allocate``, which is correct for any backend
        whose ``allocate`` fully initialises a stream.  A backend that can rewind
        in place should override — the paged one does, to keep its slot (and with
        it the persistent rows a CUDA graph captured by address) rather than
        taking a new one from the pool.
        """
        self.free(request)
        self.allocate(request)
        request.offset = 0

    # -- streaming window geometry (engine windows the feature buffer) -----
    @property
    @abstractmethod
    def decoding_window(self) -> int:
        """Input feature frames consumed per encoder chunk."""
        raise NotImplementedError

    @property
    @abstractmethod
    def stride(self) -> int:
        """Feature-frame stride between consecutive chunk windows."""
        raise NotImplementedError

    @property
    def finalize_align_frames(self) -> int:
        """Alignment the closing silence pad must round the stream up to, or ``0``.

        ``0`` (the default) means this runtime can forward a **partial** final
        window, so the trailing audio is decoded whatever its length and one
        ``decoding_window`` of closing silence is enough to flush the decoder.

        A runtime whose geometry is exact cannot: a short tail would fall off its
        subsampling stride grid, so it is skipped — and then the closing silence
        the decoder actually gets is ``window - (frames % window)``, i.e. anywhere
        from **one frame** to a full window depending on the utterance length.  That
        is not a rounding detail: measured on Nemotron's LJSpeech-200 gate, the
        trailing subword goes missing when the remainder lands badly (9 deletions
        at a 128-frame window against 42 at 32, purely from how much silence
        survived the truncation).  Declaring the alignment lets
        :class:`~oasr.engine.input_processor.InputProcessor` round the stream up
        first, so every stream gets *at least* a full window of flush silence
        regardless of its length.
        """
        return 0

    @property
    def cache_bucket_ladder(self) -> Sequence[int]:
        """Every ``cache_t1`` rung this backend can key a graph on.  Default: none."""
        return ()

    # -- optional graph pre-warm -------------------------------------------
    def prewarm(
        self, batch_sizes: Sequence[int], cache_t1_buckets: Optional[Sequence[int]] = None
    ) -> None:
        """Pre-capture any per-shape CUDA graphs.  Default: no-op."""
        return None


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[StreamingEncoderBackend]] = {}


def register_streaming_backend(name: str):
    """Class decorator registering a backend under an encoder ``streaming_kind``."""

    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_streaming_backend_class(streaming_kind: str) -> Type[StreamingEncoderBackend]:
    """The registered backend **class** for ``streaming_kind``.

    The engine needs a couple of class-level declarations
    (:attr:`StreamingEncoderBackend.allocates_paged_pool`) *before* it can build
    the backend — sizing the paged pool has to happen before something allocates
    it.  Same lookup and same error as :func:`build_streaming_backend`.
    """
    cls = _REGISTRY.get(streaming_kind)
    if cls is None:
        raise NotImplementedError(
            f"No streaming backend registered for streaming_kind={streaming_kind!r}. "
            f"Registered: {sorted(_REGISTRY)}.  Add one by subclassing "
            "StreamingEncoderBackend + @register_streaming_backend."
        )
    return cls


def build_streaming_backend(
    streaming_kind: str,
    model: "BaseAsrModel",
    config: "EngineConfig",
    cache_config: "Optional[CacheConfig]",
    *,
    graph_pool=None,
    consumes: str = "log_probs",
) -> StreamingEncoderBackend:
    """Construct the streaming backend for an encoder's ``streaming_kind``.

    ``consumes`` is the active decode strategy's declared input
    (``"log_probs"`` — fused encoder+head, the CUDA-graph fast path — or
    ``"hidden"`` — raw encoder states for autoregressive families); the backend
    routes its per-chunk forward accordingly.  Raises ``NotImplementedError``
    (listing the registered kinds) when the encoder declares a kind with no
    backend — the extension point for new streaming runtimes.
    """
    # Called as a factory, not as the ABC: the base class deliberately declares
    # no ``__init__`` (each runtime takes what its cache model needs), so the
    # constructor signature is a convention of this call site.
    factory = cast(
        Callable[..., StreamingEncoderBackend], get_streaming_backend_class(streaming_kind)
    )
    return factory(model, config, cache_config, graph_pool=graph_pool, consumes=consumes)
