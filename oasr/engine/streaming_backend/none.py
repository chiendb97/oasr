# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""No-op streaming backend for offline-only encoders (``streaming_kind="none"``).

Lets a model whose encoder has no streaming cache model construct a
``ModelRunner`` (and thus run the offline path) without a real streaming backend.
Any streaming operation raises a clear error; the window geometry is reported as
``0`` (the offline path never reads it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List

import torch

from ..request import Request
from .base import StreamingEncoderBackend, register_streaming_backend

if TYPE_CHECKING:
    from oasr.cache.types import CacheConfig
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig


@register_streaming_backend("none")
class NoStreamingBackend(StreamingEncoderBackend):
    """Offline-only placeholder; streaming operations are unsupported."""

    streaming_kind: ClassVar[str] = "none"

    def __init__(
        self,
        model: "BaseAsrModel",
        config: "EngineConfig",
        cache_config: "CacheConfig",
        *,
        graph_pool=None,
        consumes: str = "log_probs",
    ) -> None:
        del model, config, cache_config, graph_pool, consumes

    @staticmethod
    def _unsupported():
        raise NotImplementedError(
            "This encoder does not support streaming (streaming_kind='none'); "
            "use service_mode='offline'."
        )

    def allocate(self, request: Request) -> None:
        self._unsupported()

    def free(self, request: Request) -> None:
        return None  # nothing allocated

    def forward_step(self, requests: List[Request]) -> Dict[str, torch.Tensor]:
        self._unsupported()

    @property
    def decoding_window(self) -> int:
        return 0

    @property
    def stride(self) -> int:
        return 0
