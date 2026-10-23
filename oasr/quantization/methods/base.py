# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""How a quantized GEMM layer stores and runs its weights.

A :class:`QuantMethod` is installed on a :class:`~oasr.layers.QuantizableLayer`
(``layer.quant_method``) and takes over its forward — vLLM's
``LinearMethodBase`` pattern.  Life cycle, in order:

1. :meth:`create_weights` — replace the fp ``weight`` with storage-dtype slots
   (and register scales) so a quantized state dict loads into them strictly;
2. :meth:`quantize_weight` — or fill them from an fp32 weight (the online path
   and ``oasr quantize`` share this one implementation);
3. :meth:`process_weights_after_loading` — host-side layout work (none for int8);
4. :meth:`bind` — after the device move, choose the kernel lane once.  Never
   consulted per forward, never dependent on capture state (AGENTS.md rule 11);
5. :meth:`apply` per forward; :meth:`apply_reference` is the torch emulation of
   exactly the same numerics (CPU / fp32 / ``OASR_LAYERS_BACKEND=torch`` and the
   parity oracle).

A method is a plain object, not an ``nn.Module``: it must not show up in
``named_modules()`` or the state dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

if TYPE_CHECKING:  # pragma: no cover
    from oasr.layers._quantizable import QuantizableLayer

    from ..kernels.lanes import ScaledMMLane
    from ..spec import QuantScheme


class QuantMethod(ABC):
    """Per-layer strategy for one :class:`~oasr.quantization.spec.QuantScheme`."""

    def __init__(self, scheme: "QuantScheme") -> None:
        self.scheme = scheme
        self.lane: Optional["ScaledMMLane"] = None
        #: Why :attr:`lane` is ``None`` after :meth:`bind` (for the report).
        self.lane_note: str = "unbound"
        #: Who quantizes this layer's input when a producer does it (a fused norm,
        #: a chained GEMM); set by ``bind_quantization``, for the report.
        self.input_source: Optional[str] = None
        #: The dtype the layer's quantized input is stored in (``torch.int8``,
        #: ``torch.float8_e4m3fn``), or ``None`` for a method with no quantized
        #: input.  A producer that quantizes for this layer must emit exactly it.
        self.quant_dtype: Optional[torch.dtype] = None

    @classmethod
    @abstractmethod
    def supports(cls, scheme: "QuantScheme") -> bool:
        """Whether this method implements *scheme*."""

    @classmethod
    def serves(cls, layer: Any) -> bool:
        """Whether this method runs *layer*'s kind of layer (a GEMM, a convolution)."""
        return True

    @abstractmethod
    def create_weights(self, layer: "QuantizableLayer", *, runtime_smoothed: bool) -> None:
        """Replace the layer's fp weight with quantized storage slots."""

    @abstractmethod
    def quantize_weight(self, layer: "QuantizableLayer", weight_fp32: torch.Tensor) -> None:
        """Fill the slots from the fp32 ``quant_weight()``: ``[out_features, -1]``."""

    def process_weights_after_loading(self, layer: "QuantizableLayer") -> None:
        """Host-side layout work after the slots hold their values (idempotent).

        Nothing by default: int8 ``[N, K]`` is already the kernel's layout.  A format
        whose kernel wants a swizzle (NVFP4 scale factors) does it here.
        """
        return None

    @abstractmethod
    def bind(self, layer: "QuantizableLayer") -> None:
        """Choose the kernel lane for the layer's current device and shape."""

    @abstractmethod
    def preserved_tensors(self, layer: "QuantizableLayer") -> Tuple[str, ...]:
        """Names of slots a dtype cast must not touch (moved by device only)."""

    @abstractmethod
    def apply(self, layer: "QuantizableLayer", x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """The quantized forward; *kwargs* are the layer's own forward arguments
        (a convolution's ``padding`` override)."""

    @abstractmethod
    def apply_reference(
        self, layer: "QuantizableLayer", x: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Torch emulation of :meth:`apply`'s numerics."""

    # -- links between layers (armed by ``bind_quantization`` after :meth:`bind`) --
    def quantized_input(self, layer: "QuantizableLayer") -> Optional[str]:
        """The pre-quantized input this bound layer can consume, if any.

        ``"token"``: a :class:`~oasr.layers.QuantizedActivation` with one scale
        per row and no smoothing left to apply (what a fused norm emits);
        ``"tensor"``: one carrying this layer's own static scale.  ``None`` (the
        default): the layer quantizes its input itself.
        """
        return None

    def static_input(
        self, layer: "QuantizableLayer"
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``(multiplier, scale)`` a producer must quantize with to feed this layer.

        Only for a bound layer with a static input scale; ``None`` otherwise.
        """
        return None

    def emit_for(
        self, layer: "QuantizableLayer", consumer: str, mul: torch.Tensor, scale: torch.Tensor
    ) -> bool:
        """Quantize this layer's output for *consumer*; ``False`` if it cannot."""
        return False

    def describe(self, layer: "QuantizableLayer") -> Dict[str, Any]:
        """One row of the quantization report."""
        note = self.lane_note
        if self.input_source is not None:
            note += f"; input quantized by {self.input_source}"
        return {
            "scheme": self.scheme.name,
            "lane": self.lane.name if self.lane is not None else None,
            "note": note,
        }


_METHODS: List[Type[QuantMethod]] = []


def register_quant_method(cls: Type[QuantMethod]) -> Type[QuantMethod]:
    """Class decorator: make *cls* a candidate for :func:`method_for`."""
    _METHODS.append(cls)
    return cls


def method_for(scheme: "QuantScheme", layer: Any = None) -> QuantMethod:
    """A fresh method instance implementing *scheme* for *layer*'s kind; refuses otherwise.

    With no *layer*, the first method implementing *scheme* at all.
    """
    from . import w8a8  # noqa: F401  -- registers the built-in methods

    for cls in _METHODS:
        if cls.supports(scheme) and (layer is None or cls.serves(layer)):
            return cls(scheme)
    kind = getattr(layer, "quant_kind", None)
    if kind in ("conv1d", "conv2d"):
        raise NotImplementedError(
            f"a {kind} layer cannot be quantized with scheme {scheme.name or scheme}: a "
            "convolution needs int8 weights and a calibrated static input scale "
            "(int8_w8a8_static), which `oasr quantize --module-types ...` produces"
        )
    raise NotImplementedError(
        f"no quantization method implements scheme {scheme.name or scheme}"
        + (f" for a {kind} layer" if kind else "")
        + f"; registered methods: {[c.__name__ for c in _METHODS]}"
    )
