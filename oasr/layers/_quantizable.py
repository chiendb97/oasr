# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The quantization seam of the waist's dense-GEMM layers.

:class:`Linear`, :class:`LinearActivation` and :class:`PointwiseConv1d` are the
three layers whose forward is ``act(x @ Wᵀ + b)``.  They mix in
:class:`QuantizableGemm`, which gives every one of them the same two things:

* a ``quant_method`` slot.  ``None`` (the class default) leaves the layer
  exactly as it was; a :class:`oasr.quantization.methods.QuantMethod` installed
  by :func:`oasr.quantization.apply_quantization` takes over the forward
  (vLLM's ``LinearMethodBase`` pattern) — the module's type, path and
  parameter names stay what they were, so hooks, ``isinstance`` checks and
  state-dict keys keep working.
* a dtype-preserving ``_apply``.  ``Module.to(dtype)`` / ``.half()`` cast every
  floating tensor, float8 storage and fp32 scales included; a quantized layer
  moves those by device only.

Nothing here imports :mod:`oasr.quantization`: the layer only knows the method
interface (``apply`` and ``preserved_tensors``).

:class:`QuantizedActivation` is what travels between a producer that already
emits int8 (a LayerNorm with ``quantize_output``, a GEMM whose epilogue
quantizes for its consumer) and the quantized GEMM that consumes it, so the
consumer skips its own quantize launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from oasr.quantization.methods.base import QuantMethod


@dataclass(frozen=True)
class QuantizedActivation:
    """An activation already quantized for the GEMM that consumes it.

    ``x ≈ data * scale``: ``data`` is int8 ``[..., K]``; ``scale`` is fp32 with
    one entry per row (``data.shape[:-1]``, per-token) or a single entry
    (``[1]``, a static per-tensor scale).  ``dtype`` is the floating dtype the
    producer would otherwise have returned -- the consumer's output dtype.

    Only a producer whose output nothing but quantized consumers reads may
    return one; the model declares that (``SmoothGroup.fuse_output_quant``, a
    ``GemmChain``) and :func:`oasr.quantization.bind_quantization` arms it.  Code
    between the two may read ``shape`` / ``size()`` / ``dim()`` and pass it on,
    nothing else: a hook or an op that expects a tensor fails loudly.
    """

    data: torch.Tensor
    scale: torch.Tensor
    dtype: torch.dtype

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def is_cuda(self) -> bool:
        return self.data.is_cuda

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def dim(self) -> int:
        return self.data.dim()

    def size(self, dim: Optional[int] = None):  # type: ignore[no-untyped-def]
        return self.data.size() if dim is None else self.data.size(dim)

    @property
    def per_token(self) -> bool:
        """One scale per row (else one per-tensor scale; a single row is both)."""
        return self.scale.numel() == self.data.numel() // max(self.data.shape[-1], 1)


class QuantizableGemm:
    """Mixin for a waist layer computing ``act(x @ gemm_weight()ᵀ + bias)``."""

    #: The installed quantization method, or ``None`` for the full-precision
    #: path.  A class attribute, so an unquantized layer pays one attribute read.
    quant_method: Optional["QuantMethod"] = None

    #: ``False`` when something besides this layer's own forward reads its
    #: ``weight`` raw — a fused kernel of the parent (the gated MLP's dual-B
    #: GEMM, the transducer's fused decode).  Quantizing it would hand that
    #: reader int8 storage, so a spec selecting it is refused.
    quantizable: bool = True

    #: The activation fused into the GEMM epilogue, spelled as
    #: :class:`LinearActivation` spells it (``gelu`` = exact erf,
    #: ``gelu_tanh`` = tanh approximation), or ``None``.
    fused_activation: Optional[str] = None

    in_features: int
    out_features: int

    def gemm_weight(self) -> torch.Tensor:
        """The weight as the ``[out_features, in_features]`` matrix of the GEMM."""
        weight: torch.Tensor = self.weight  # type: ignore[attr-defined]
        return weight

    def _apply(self, fn, recurse: bool = True):  # type: ignore[no-untyped-def]
        method = self.quant_method
        if method is None:
            return super()._apply(fn, recurse)  # type: ignore[misc]
        module: nn.Module = self  # type: ignore[assignment]
        held: Dict[str, Tuple[str, Any]] = {}
        param_order = list(module._parameters)
        buffer_order = list(module._buffers)
        for name in method.preserved_tensors(self):
            if module._parameters.get(name) is not None:
                held[name] = ("param", module._parameters.pop(name))
            elif module._buffers.get(name) is not None:
                held[name] = ("buffer", module._buffers.pop(name))
        try:
            result = super()._apply(fn, recurse)  # type: ignore[misc]
        finally:
            for name, (kind, tensor) in held.items():
                # The target device, read off a probe the cast leaves alone.
                probe = torch.empty(0, dtype=torch.int32, device=tensor.device)
                device = fn(probe).device
                moved = tensor.data if tensor.device == device else tensor.data.to(device)
                if kind == "param":
                    module._parameters[name] = nn.Parameter(moved, requires_grad=False)
                else:
                    module._buffers[name] = moved
            _reorder(module._parameters, param_order)
            _reorder(module._buffers, buffer_order)
        return result


def _reorder(store: Dict[str, Any], order: list) -> None:
    """Restore *store*'s original key order (state-dict order is observable)."""
    items = [(k, store[k]) for k in order if k in store]
    items += [(k, v) for k, v in store.items() if k not in order]
    store.clear()
    store.update(items)


__all__ = ["QuantizableGemm"]
