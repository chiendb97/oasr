# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Spec × model → quantized layers; validation and the quantization report.

The engine and the loaders call four functions, in this order, and know nothing
about any particular scheme:

* :func:`apply_quantization` — install a method on every layer the spec
  selects and create its storage (``quantize=False``: slots for a strict
  load; ``quantize=True``: fill them from the layer's fp32 weight);
* :func:`finalize_quantization` — host-side layout after the load;
* :func:`bind_quantization` — after the device move, pick each layer's lane,
  then arm the producer -> consumer links the model's layout declares (a norm
  that emits int8 for its GEMMs, a GEMM that quantizes for a chained one);
* :func:`validate_runtime` — the per-layer report the engine logs.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import torch
from torch import nn

from oasr.layers._quantizable import QuantizableGemm

from .layout import QuantLayout
from .methods.base import method_for
from .spec import QuantizationSpec

logger = logging.getLogger(__name__)

#: ``OASR_QUANT_FUSION=0`` (read once at import) leaves every link disarmed: each
#: quantized layer quantizes its own input -- the A/B and rollback switch for the
#: fused norms and GEMM chains.  Numerics are unchanged either way (every fused
#: kernel is bit-identical to the pair it replaces); only the launch count moves.
_FUSION_ENABLED = os.environ.get("OASR_QUANT_FUSION", "1").strip().lower() not in (
    "0",
    "off",
    "false",
    "no",
)


def quantizable_layers(model: nn.Module) -> Dict[str, QuantizableGemm]:
    """``{path: layer}`` for every dense-GEMM waist layer that may be quantized."""
    return {
        path: module
        for path, module in model.named_modules()
        if isinstance(module, QuantizableGemm) and getattr(module, "quantizable", True)
    }


def quantized_layers(model: nn.Module) -> Dict[str, QuantizableGemm]:
    """``{path: layer}`` for every layer with a quantization method installed."""
    return {
        path: module
        for path, module in model.named_modules()
        if isinstance(module, QuantizableGemm) and module.quant_method is not None
    }


def apply_quantization(model: nn.Module, spec: QuantizationSpec, *, quantize: bool) -> List[str]:
    """Install *spec*'s methods on *model*; returns the quantized paths.

    ``quantize=True`` quantizes each selected layer's current (fp) weight —
    the online path and ``oasr quantize``.  It must run *after* the fp weights
    are loaded: a converted checkpoint's legacy remaps run inside
    ``_load_from_state_dict`` and would otherwise be fed into int8 slots
    (``param.copy_`` casts silently).  ``quantize=False`` only creates the slots
    a native quantized checkpoint is then strictly loaded into.
    """
    spec.check_against(model)
    smoothed = set(spec.runtime_smoothed)
    done: List[str] = []
    for path, layer in quantizable_layers(model).items():
        scheme = spec.resolve(path)
        if scheme is None:
            continue
        if layer.quant_method is not None:
            raise RuntimeError(f"{path} is already quantized")
        method = method_for(scheme)
        weight_fp32 = layer.gemm_weight().detach().float().clone() if quantize else None
        layer.quant_method = method
        method.create_weights(layer, runtime_smoothed=path in smoothed)
        if weight_fp32 is not None:
            method.quantize_weight(layer, weight_fp32)
        done.append(path)
    if not done:
        raise ValueError("the quantization spec selects no layer of this model")
    _pin_batch_invariant(model)
    return done


def _layout_of(model: nn.Module) -> Optional[QuantLayout]:
    layout_fn = getattr(model, "quantization_layout", None)
    return layout_fn() if layout_fn is not None else None


def _pin_batch_invariant(model: nn.Module) -> None:
    """Pin the full-precision GEMMs the layout names to a batch-invariant kernel.

    An int8 layer is batch-invariant by construction; an fp GEMM upstream of it
    whose kernel choice depends on M (cuBLAS, split-K, Stream-K) is not, and the
    int8 rounding turns its ulp differences into different tokens.  Declare,
    don't ignore: a glob that matches nothing, or a module that has no pinned
    path, is refused.
    """
    layout = _layout_of(model)
    if layout is None:
        return
    modules = dict(model.named_modules())
    for glob in layout.batch_invariant:
        hits = [p for p in modules if fnmatch.fnmatchcase(p, glob)]
        if not hits:
            raise ValueError(f"batch_invariant glob {glob!r} matches no module of the model")
        for path in hits:
            module = modules[path]
            if not hasattr(type(module), "batch_invariant"):
                raise ValueError(
                    f"{path} ({type(module).__name__}) cannot be pinned batch-invariant: only "
                    "a layer with a pinned kernel path (Linear, LinearActivation) can"
                )
            if isinstance(module, QuantizableGemm) and module.quant_method is not None:
                continue  # an int8 layer is batch-invariant already
            setattr(module, "batch_invariant", True)  # noqa: B010 - a class-declared knob


def finalize_quantization(model: nn.Module) -> None:
    """Run every quantized layer's host-side ``process_weights_after_loading``."""
    for layer in quantized_layers(model).values():
        assert layer.quant_method is not None
        layer.quant_method.process_weights_after_loading(layer)


def bind_quantization(model: nn.Module) -> None:
    """Choose each quantized layer's kernel lane for where it now lives, then link.

    Call after the device move, and again after anything that changes lane
    availability (``ASREngine.reload_tuning``).  Links are re-derived from
    scratch every time, so a rebind never keeps one its layers can no longer serve.
    """
    layers = quantized_layers(model)
    for layer in layers.values():
        assert layer.quant_method is not None
        layer.quant_method.bind(layer)
    _link(model, layers)


def _link(model: nn.Module, layers: Dict[str, QuantizableGemm]) -> None:
    """Arm the producer -> consumer quantization the layout declares, where it can run.

    * A norm whose group says ``fuse_output_quant`` emits per-token 8-bit values
      when every consumer can take them (a bound lane, a per-token scale, no
      runtime smoother left to apply, one quantized dtype the norm can write):
      one launch instead of norm + quantize per GEMM.
    * A :class:`~oasr.quantization.layout.GemmChain` whose consumer has a static
      input scale makes the producer's epilogue quantize for it: the pair
      exchanges 8-bit values (both layers must use the same format) and the
      consumer's quantize launch disappears.

    Anything that cannot run fused stays unfused -- correct, just slower -- and
    the report says which inputs were quantized by whom.
    """
    modules = dict(model.named_modules())
    for module in modules.values():
        for knob in ("quantize_output", "quantize_dtype"):
            if knob in vars(module):
                delattr(module, knob)  # back to the class default (off, int8)
    layout = _layout_of(model)
    if layout is None or not _FUSION_ENABLED:
        return
    for group in layout.groups:
        if not group.fuse_output_quant or group.producer is None:
            continue
        norm = modules.get(group.producer)
        if norm is None or not hasattr(type(norm), "quantize_output"):
            continue
        consumers = [layers.get(c) for c in group.consumers]
        if not all(
            c is not None
            and c.quant_method is not None
            and c.quant_method.quantized_input(c) == "token"
            for c in consumers
        ):
            continue
        dtypes = {c.quant_method.quant_dtype for c in consumers}  # type: ignore[union-attr]
        emits = getattr(type(norm), "quantize_output_dtypes", (torch.int8,))
        if len(dtypes) == 1 and next(iter(dtypes)) in emits:
            setattr(norm, "quantize_output", True)  # noqa: B010 - a class-declared knob
            setattr(norm, "quantize_dtype", next(iter(dtypes)))  # noqa: B010
            for c in consumers:
                assert c is not None and c.quant_method is not None
                c.quant_method.input_source = f"its norm ({group.producer})"
    for chain in layout.chains:
        producer, consumer = layers.get(chain.producer), layers.get(chain.consumer)
        if producer is None or consumer is None:
            continue
        assert producer.quant_method is not None and consumer.quant_method is not None
        if consumer.quant_method.quantized_input(consumer) != "tensor":
            continue
        if producer.quant_method.quant_dtype != consumer.quant_method.quant_dtype:
            continue  # the epilogue writes the producer's own format
        static = consumer.quant_method.static_input(consumer)
        if static is None:
            continue
        mul, scale = static
        if producer.quant_method.emit_for(producer, chain.consumer, mul, scale):
            consumer.quant_method.input_source = f"its producer GEMM ({chain.producer})"


def check_quantized_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Refuse a state dict whose tensors would be cast into quantized slots.

    ``load_state_dict`` copies with ``param.copy_``, which converts dtypes
    without a word: an fp32 weight loaded into an int8 slot truncates to
    garbage.  Every quantized slot must arrive in its own dtype.
    """
    bad = []
    for path, layer in quantized_layers(model).items():
        assert layer.quant_method is not None
        for name in layer.quant_method.preserved_tensors(layer):
            slot = getattr(layer, name, None)
            key = f"{path}.{name}" if path else name
            if isinstance(slot, torch.Tensor) and key in state_dict:
                if state_dict[key].dtype != slot.dtype:
                    bad.append(f"{key}: checkpoint {state_dict[key].dtype}, slot {slot.dtype}")
    if bad:
        raise ValueError(
            "quantized checkpoint tensors do not have their slots' dtypes "
            f"(first few: {bad[:5]}); it was not written by this quantization scheme"
        )


@dataclass(frozen=True)
class QuantReportRow:
    path: str
    scheme: str
    lane: Optional[str]
    note: str
    shape: str


def validate_runtime(model: nn.Module) -> List[QuantReportRow]:
    """One row per quantized layer: scheme, chosen lane and why."""
    rows = []
    for path, layer in quantized_layers(model).items():
        assert layer.quant_method is not None
        desc = layer.quant_method.describe(layer)
        rows.append(
            QuantReportRow(
                path=path,
                scheme=str(desc["scheme"]),
                lane=desc["lane"],
                note=str(desc["note"]),
                shape=f"{layer.out_features}x{layer.in_features}",
            )
        )
    return rows


def format_quant_report(rows: List[QuantReportRow]) -> str:
    """A compact summary: layer counts per (scheme, lane, note)."""
    if not rows:
        return "quantization: none"
    counts: Dict[tuple, int] = {}
    for row in rows:
        key = (row.scheme, row.lane or "reference", row.note)
        counts[key] = counts.get(key, 0) + 1
    lines = [f"quantization: {len(rows)} layer(s)"]
    for (scheme, lane, note), n in sorted(counts.items()):
        lines.append(f"  {n:4d} x {scheme} via {lane} ({note})")
    return "\n".join(lines)


def describe_quantization(model: nn.Module) -> Optional[Dict[str, object]]:
    """A JSON-safe summary for model info endpoints, or ``None`` if unquantized."""
    rows = validate_runtime(model)
    if not rows:
        return None
    schemes = sorted({r.scheme for r in rows})
    lanes = sorted({r.lane or "reference" for r in rows})
    return {"schemes": schemes, "layers": len(rows), "lanes": lanes}
