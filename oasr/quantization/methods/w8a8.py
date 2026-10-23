# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""W8A8: 8-bit per-channel weights; per-token dynamic or static per-tensor activations.

One method, two formats -- the scheme's ``type`` picks them: ``int`` is int8
(``Q = 127``, an exact int32 GEMM accumulator), ``float`` is fp8 e4m3 (``Q = 448``,
an fp32 accumulator; sm89+).  Everything else -- storage, scales, the forward,
the fused producers -- is shared.

Storage on a quantized layer (state-dict names follow compressed-tensors):

* ``weight``       — int8 / float8_e4m3fn, the layer's own shape (``[N, K]``;
  ``[N, K, 1]`` for a pointwise conv), so a checkpoint key means the same thing
  quantized or not;
* ``weight_scale`` — fp32 ``[N]``: ``W ≈ weight * weight_scale[:, None]``;
* ``smooth_inv``   — fp32 ``[K]``, only on runtime-smoothed layers (SmoothQuant
  for a GEMM whose producer cannot absorb the scale);
* ``input_scale``  — fp32 ``[1]``, only with a static activation scale: the
  quantization step of the layer's (smoothed) input, calibrated by ``oasr quantize``.

Forward, by what arrives:

* a :class:`~oasr.layers.QuantizedActivation` -- a norm or a producer GEMM
  already quantized for this layer -- goes straight to the GEMM;
* otherwise a static layer quantizes with its bound multiplier
  (``quantize_static``) and a dynamic one per token (``quantize_per_token``,
  the smoother applied inside);

then the lane's scaled GEMM, which dequantizes, adds the bias and applies the
layer's fused activation in one epilogue.  When this layer feeds a static
consumer through a declared :class:`~oasr.quantization.layout.GemmChain`
(:attr:`W8A8Method.emit`, armed by ``bind_quantization``) the epilogue
quantizes for that consumer instead and the layer returns a
``QuantizedActivation``.  The output dtype is the input's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from oasr.functionals.quant import (
    QUANT_EPS,
    quant_qmax,
    quantize_per_token,
    quantize_static,
    reference_quantize_per_token,
    reference_quantize_static,
    round_to_quant,
    static_multiplier,
)
from oasr.functionals.scaled_mm import reference_scaled_mm, reference_scaled_mm_quant
from oasr.layers._backend import layers_backend, take_gap
from oasr.layers._quantizable import QuantizedActivation

from ..kernels.lanes import choose_lane, forced_backend, w8a8_format
from .base import QuantMethod, register_quant_method

if TYPE_CHECKING:  # pragma: no cover
    from oasr.layers._quantizable import QuantizableGemm

    from ..spec import QuantScheme

_SERVED = (torch.float16, torch.bfloat16)

Activation = Union[torch.Tensor, QuantizedActivation]


def _tensor(module: nn.Module, name: str) -> torch.Tensor:
    """A tensor slot of *module* (``getattr`` keeps ``nn.Module``'s union out of the types)."""
    value = getattr(module, name)
    assert isinstance(value, torch.Tensor), name
    return value


def _optional(module: nn.Module, name: str) -> Optional[torch.Tensor]:
    value = getattr(module, name, None)
    return value if isinstance(value, torch.Tensor) else None


def quantize_weight_per_channel(
    weight_fp32: torch.Tensor, eps: float = QUANT_EPS, dtype: torch.dtype = torch.int8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale)`` for a ``[N, K]`` fp32 weight: symmetric 8-bit per output row.

    *dtype* is ``torch.int8`` or ``torch.float8_e4m3fn``.  The eps floor matters:
    ``pad_output_projection`` pads vocabularies with all-zero rows, whose amax is 0.
    """
    w = weight_fp32.float()
    wmax = w.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    qmax = torch.full_like(wmax, quant_qmax(dtype))
    q = round_to_quant(w * (qmax / wmax), dtype)
    return q, (wmax / qmax).squeeze(1)


@dataclass(frozen=True)
class OutputQuant:
    """How a producer GEMM quantizes its output for its chained static consumer."""

    #: The consumer's per-channel multiplier (``static_multiplier``), fp32 ``[N]``.
    mul: torch.Tensor
    #: The consumer's ``input_scale`` (fp32 ``[1]``): the scale the 8-bit output carries.
    scale: torch.Tensor
    #: The consumer's module path (for the report).
    consumer: str


@register_quant_method
class W8A8Method(QuantMethod):
    """8-bit (int8 or fp8 e4m3) per-channel W x 8-bit A, per-token dynamic or static."""

    def __init__(self, scheme: "QuantScheme") -> None:
        super().__init__(scheme)
        #: ``"int8"`` or ``"fp8"``.
        self.fmt = w8a8_format(scheme)
        if self.fmt is None:
            raise ValueError(f"{type(self).__name__} cannot serve scheme {scheme.name or scheme}")
        #: The dtype the weight and the quantized activations are stored in.
        self.storage_dtype: torch.dtype = scheme.weights.storage_dtype
        self.quant_dtype = self.storage_dtype
        #: A static layer's quantize multiplier on its device, set by :meth:`bind`.
        self.static_mul: Optional[torch.Tensor] = None
        #: Set by ``bind_quantization`` when this layer's output feeds a static
        #: consumer through a declared GEMM chain: quantize for it in the epilogue.
        self.emit: Optional[OutputQuant] = None

    @classmethod
    def supports(cls, scheme: "QuantScheme") -> bool:
        return w8a8_format(scheme) is not None

    @property
    def static(self) -> bool:
        """The activation scale is a calibrated per-tensor constant."""
        a = self.scheme.input_activations
        return a is not None and a.granularity == "tensor" and not a.dynamic

    # -- storage -------------------------------------------------------------
    def create_weights(self, layer: "QuantizableGemm", *, runtime_smoothed: bool) -> None:
        module: nn.Module = layer  # type: ignore[assignment]
        old = _tensor(module, "weight")
        device = old.device
        module.weight = nn.Parameter(
            torch.zeros(old.shape, dtype=self.storage_dtype, device=device), requires_grad=False
        )
        module.register_buffer(
            "weight_scale", torch.ones(layer.out_features, dtype=torch.float32, device=device)
        )
        if runtime_smoothed:
            module.register_buffer(
                "smooth_inv",
                _take_installed(module, "smooth_inv", torch.ones(layer.in_features), device),
            )
        if self.static:
            # Installed by calibration on the fp layer; a native load overwrites it.
            module.register_buffer(
                "input_scale", _take_installed(module, "input_scale", torch.ones(1), device)
            )

    def quantize_weight(self, layer: "QuantizableGemm", weight_fp32: torch.Tensor) -> None:
        module: nn.Module = layer  # type: ignore[assignment]
        q, scale = quantize_weight_per_channel(weight_fp32, dtype=self.storage_dtype)
        weight = _tensor(module, "weight")
        weight.data.copy_(q.reshape(weight.shape))
        _tensor(module, "weight_scale").copy_(scale)

    def preserved_tensors(self, layer: "QuantizableGemm") -> Tuple[str, ...]:
        return ("weight", "weight_scale", "smooth_inv", "input_scale")

    # -- binding ---------------------------------------------------------------
    def bind(self, layer: "QuantizableGemm") -> None:
        module: nn.Module = layer  # type: ignore[assignment]
        self.emit = None  # links are re-armed by bind_quantization
        self.input_source = None
        self.static_mul = self._multiplier(layer) if self.static else None
        device = _tensor(module, "weight").device
        if device.type != "cuda":
            self.lane, self.lane_note = None, f"reference ({device.type})"
            return
        bias = _optional(module, "bias")
        out_dtype = bias.dtype if bias is not None else torch.float16
        if out_dtype not in _SERVED:
            self.lane, self.lane_note = None, f"reference ({out_dtype})"
            return
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor
        lane, note = choose_lane(self.scheme, layer.out_features, layer.in_features, out_dtype, sm)
        if lane is not None:
            self.lane, self.lane_note = lane, note
            return
        if forced_backend() is not None:
            raise RuntimeError(
                f"OASR_SCALED_MM_BACKEND={forced_backend()} cannot serve a "
                f"{layer.out_features}x{layer.in_features} {self.scheme.name} layer on sm{sm}: "
                f"{note}"
            )
        if self.fmt == "fp8" and sm < 89:
            # Not a missing kernel: the GPU has no fp8 tensor cores.
            raise RuntimeError(
                f"scheme {self.scheme.name} needs fp8 tensor cores (sm89+), and this GPU is "
                f"sm{sm}; quantize to int8 (int8_w8a8) instead"
            )
        if self.fmt == "int8" and sm == 75 and take_gap("scaled-mm-int8-sm75", note) is False:
            self.lane, self.lane_note = None, "reference (declared gap scaled-mm-int8-sm75)"
            return
        raise RuntimeError(
            f"no kernel serves a {layer.out_features}x{layer.in_features} "
            f"{self.scheme.name} layer on sm{sm} ({note}); a missing kernel must be declared "
            "in oasr.layers._backend.KERNEL_GAPS, not routed around"
        )

    def quantized_input(self, layer: "QuantizableGemm") -> Optional[str]:
        if self.lane is None:
            return None
        if self.static:
            return "tensor"
        module: nn.Module = layer  # type: ignore[assignment]
        return None if _optional(module, "smooth_inv") is not None else "token"

    def static_input(self, layer: "QuantizableGemm") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.lane is None or not self.static or self.static_mul is None:
            return None
        module: nn.Module = layer  # type: ignore[assignment]
        return self.static_mul, _tensor(module, "input_scale")

    def emit_for(
        self, layer: "QuantizableGemm", consumer: str, mul: torch.Tensor, scale: torch.Tensor
    ) -> bool:
        if self.lane is None or mul.numel() != layer.out_features:
            return False
        self.emit = OutputQuant(mul, scale, consumer)
        return True

    def _multiplier(self, layer: "QuantizableGemm") -> torch.Tensor:
        module: nn.Module = layer  # type: ignore[assignment]
        return static_multiplier(
            _tensor(module, "input_scale"), _optional(module, "smooth_inv"), layer.in_features
        )

    # -- forward ---------------------------------------------------------------
    def apply(self, layer: "QuantizableGemm", x: Activation) -> Activation:  # type: ignore[override]
        dtype = x.dtype
        if (
            self.lane is None
            or not x.is_cuda
            or dtype not in _SERVED
            or layers_backend() == "torch"
        ):
            return self.apply_reference(layer, x)
        module: nn.Module = layer  # type: ignore[assignment]
        if isinstance(x, QuantizedActivation):
            q, a_scale = self._check_format(x), x.scale
        elif self.static:
            if not x.is_contiguous():
                x = x.contiguous()
            mul = self.static_mul if self.static_mul is not None else self._multiplier(layer)
            q = quantize_static(x, mul, dtype=self.storage_dtype)
            a_scale = _tensor(module, "input_scale")
        else:
            if not x.is_contiguous():
                x = x.contiguous()
            q, a_scale = quantize_per_token(
                x, _optional(module, "smooth_inv"), dtype=self.storage_dtype
            )
        weight, w_scale = layer.gemm_weight(), _tensor(module, "weight_scale")
        bias, act = _optional(module, "bias"), layer.fused_activation
        if self.emit is not None:
            data = self.lane.run_quant(q, weight, a_scale, w_scale, bias, act, self.emit.mul, dtype)
            return QuantizedActivation(data, self.emit.scale, dtype)
        return self.lane.run(q, weight, a_scale, w_scale, bias, act, dtype)

    def apply_reference(  # type: ignore[override]
        self, layer: "QuantizableGemm", x: Activation
    ) -> Activation:
        module: nn.Module = layer  # type: ignore[assignment]
        dtype = x.dtype
        if isinstance(x, QuantizedActivation):
            q, a_scale = self._check_format(x), x.scale
        elif self.static:
            a_scale = _tensor(module, "input_scale")
            q = reference_quantize_static(
                x, self._multiplier(layer).to(x.device), dtype=self.storage_dtype
            )
        else:
            q, a_scale = reference_quantize_per_token(
                x, _optional(module, "smooth_inv"), dtype=self.storage_dtype
            )
        weight, w_scale = layer.gemm_weight(), _tensor(module, "weight_scale")
        bias, act = _optional(module, "bias"), layer.fused_activation
        if self.emit is not None:
            mul = self.emit.mul.to(q.device)
            data = reference_scaled_mm_quant(q, weight, a_scale, w_scale, bias, act, mul, dtype)
            return QuantizedActivation(data, self.emit.scale, dtype)
        return reference_scaled_mm(q, weight, a_scale, w_scale, bias, act, out_dtype=dtype)

    def _check_format(self, x: QuantizedActivation) -> torch.Tensor:
        """The pre-quantized data, refused if it is not this layer's format."""
        if x.data.dtype != self.storage_dtype:
            raise RuntimeError(
                f"a {self.scheme.name} layer was handed {x.data.dtype} activations; a producer "
                f"armed for it must emit {self.quant_dtype}"
            )
        return x.data

    def describe(self, layer: "QuantizableGemm") -> Dict[str, Any]:
        row = super().describe(layer)
        notes = [str(row["note"])]
        if self.static:
            notes.append("static input scale")
        if self.emit is not None:
            notes.append("quantizes its output for its consumer")
        row["note"] = "; ".join(notes)
        return row


def _take_installed(
    module: nn.Module, name: str, default: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """A buffer a calibration algorithm installed on the fp layer, else *default*.

    Removed from the module so the caller can register it again in its final
    position (state-dict order is observable).
    """
    existing = module._buffers.pop(name, None)
    if isinstance(existing, torch.Tensor):
        return existing.detach().float().to(device)
    return default.float().to(device)
