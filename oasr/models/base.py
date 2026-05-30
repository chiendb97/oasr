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

The engine touches a model only through:

* :attr:`BaseAsrModel.cache_spec` — to size the streaming KV / CNN caches,
* :attr:`BaseAsrModel.decode_type` — to pick the decode algorithm,
* :meth:`BaseAsrModel.forward_offline` / :meth:`forward_offline_packed` /
  :meth:`forward_chunk_paged` — the three forward entry points,
* :meth:`BaseAsrModel.from_config` / :meth:`load_weights` — construction.

Anything conforming to that interface plugs in without engine changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn

if TYPE_CHECKING:
    from oasr.cache.paged_kv import PagedKVCache
    from oasr.cache.slot_cnn import SlotCnnCache

# Decode-path selector. ``OutputProcessor`` dispatches on this. Kept as a plain
# ``str`` (values: "ctc", "transducer", "aed") so heads can declare it without
# importing an enum; only "ctc" is wired today.
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
class BaseModelConfig:
    """Common model-config fields shared by every architecture.

    Architecture-specific configs subclass this and add their own
    hyperparameters (e.g. :class:`~oasr.models.conformer.ConformerModelConfig`).
    ``model_type`` keys the model registry; ``vocab_size`` is read by the engine
    and the serving layer.
    """

    model_type: str = "base"
    vocab_size: Optional[int] = None


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
    def forward(
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        raise NotImplementedError(
            f"{type(self).__name__} does not support paged-KV streaming"
        )

    def forward_packed(
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-packing offline forward (optional).  Default: unsupported."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support sequence packing"
        )

    # -- introspection used to build CacheSpec ------------------------------
    @property
    @abstractmethod
    def num_encoder_layers(self) -> int:
        """Number of encoder layers (== paged KV cache layers)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_kv_head(self) -> int:
        """Number of KV attention heads per layer."""
        raise NotImplementedError

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Per-head key/value dimension."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Encoder hidden dimension."""
        raise NotImplementedError

    @property
    def conv_kernel_size(self) -> int:
        """Depthwise-conv kernel for streaming left-context; 1 == no CNN cache."""
        return 1

    @property
    def cache_spec(self) -> CacheSpec:
        """Streaming cache descriptor derived from the live encoder dims."""
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

    # -- construction & weights --------------------------------------------
    @classmethod
    @abstractmethod
    def from_config(cls, config: BaseModelConfig, **aux: Any) -> "BaseAsrModel":
        """Build a model (random weights) from its config + format aux buffers."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> None:
        """Map an external checkpoint state-dict into this model's parameters.

        Each architecture owns the name-mapping / fusion knowledge (vLLM-style);
        in-module reshaping (e.g. fused QKV, conv reorder) is handled by the
        layers' ``_load_from_state_dict`` hooks.
        """
        raise NotImplementedError

    # -- engine-facing metadata --------------------------------------------
    @property
    def cache_spec(self) -> CacheSpec:
        return self.encoder.cache_spec

    @property
    def decode_type(self) -> DecodeType:
        return self.head.decode_type

    # -- engine-facing forward entry points --------------------------------
    @staticmethod
    def _lengths_from_mask(masks: torch.Tensor) -> torch.Tensor:
        """``(B, 1, T)`` bool mask → ``(B,)`` int32 valid output lengths."""
        return masks.squeeze(1).sum(dim=-1).to(torch.int32)

    def forward_offline(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched offline forward → ``(log_probs (B, T, V), out_lengths (B,))``."""
        hidden, masks = self.encoder(features, lengths)
        return self.head(hidden), self._lengths_from_mask(masks)

    def forward_offline_packed(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-packing offline forward → ``(log_probs, out_lengths)``."""
        if not self.encoder.supports_packing:
            raise NotImplementedError(
                f"{type(self).__name__} encoder does not support sequence packing"
            )
        hidden, masks = self.encoder.forward_packed(features, lengths)
        return self.head(hidden), self._lengths_from_mask(masks)

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
        hidden = self.encoder.forward_chunk_paged(
            input_features, offset, att_caches, cnn_cache, att_mask, cache_t1
        )
        return self.head(hidden)
