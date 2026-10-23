# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Calibrate static per-tensor input scales for the consumers of GEMM chains.

A :class:`~oasr.quantization.layout.GemmChain` (the FFN's ``w_1`` -> ``w_2``)
can exchange 8-bit values -- the producer's epilogue quantizing for the
consumer, with no fp round trip and no quantize launch -- only if the
consumer's input scale is known before the producer runs.  A per-token scale is
not: it needs the whole row, which no single output tile of the producer sees.
This algorithm gives each chain consumer a static per-tensor scale instead:

    input_scale = clip(|x * smooth_inv|) / Q     over the calibration audio

(``Q`` = 127 for int8, 448 for fp8 e4m3) measured on the input the quantize
kernel will see -- smoothed by the consumer's runtime ``smooth_inv`` when
SmoothQuant installed one, so it runs after ``smoothquant`` in a recipe -- and
moves those layers to the static scheme of their format (``int8_w8a8_static`` /
``fp8_w8a8_static``).  Every other layer keeps its per-token scale.

A static scale is batch-invariant (it is a constant), unlike a per-tensor
*dynamic* one, which the spec refuses.  It is coarser than per-token, and *how*
coarse is the whole question.  Measured on the U2++ Conformer's ``w_2`` inputs
(mean SQNR over the 24 layers, 40 LJSpeech utterances): per-token 29.8 dB; a
static scale at the calibration maximum 17.3 dB -- the median frame spans 22 %
of that range, ~28 int8 levels -- which cost 0.44 WER in streaming; clipping
at the 99.999th percentile 24.9 dB.  So the clip is searched, not taken at the
maximum (``observer``):

* ``absmax`` -- the largest value seen (one pass);
* ``percentile`` -- the ``percentile``-th percentile of ``|x|`` (two passes:
  the maximum, then a histogram under it);
* ``mse`` -- the clip minimising the expected quantization error of that
  histogram: rounding below the clip (int8's uniform step, or fp8's step
  relative to each value), saturation above it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

from oasr.functionals.quant import QUANT_EPS

from ..calibration.observers import (
    AbsMaxPerTensor,
    HistogramPerTensor,
    Observer,
    clip_mse,
    clip_percentile,
)
from ..kernels.lanes import w8a8_format
from ..layout import QuantLayout
from ..spec import FP8_W8A8_STATIC, INT8_W8A8_STATIC, QuantGroup, QuantizationSpec
from .base import ForwardLoop, QuantAlgorithm, register_quant_algorithm

#: Each format's static scheme.
_STATIC_SCHEME = {"int8": INT8_W8A8_STATIC, "fp8": FP8_W8A8_STATIC}


@dataclass(frozen=True)
class StaticInputsOptions:
    """Static input-scale calibration knobs."""

    #: How the clip is chosen: ``absmax``, ``percentile`` or ``mse`` (module docstring).
    observer: str = "mse"
    #: The percentile of ``|x|`` the ``percentile`` observer clips at.
    percentile: float = 99.999
    #: Histogram resolution of the two-pass observers.
    bins: int = 4096


@register_quant_algorithm("static_inputs")
class StaticInputs(QuantAlgorithm):
    """Give every GEMM-chain consumer a calibrated static per-tensor input scale."""

    options_cls = StaticInputsOptions
    needs_calibration = True

    def apply(
        self,
        model: nn.Module,
        layout: QuantLayout,
        spec: QuantizationSpec,
        forward_loop: Optional[ForwardLoop],
    ) -> QuantizationSpec:
        opts: StaticInputsOptions = self.options or StaticInputsOptions()
        if opts.observer not in _OBSERVERS:
            raise ValueError(
                f"unknown static_inputs observer {opts.observer!r}; known: {sorted(_OBSERVERS)}"
            )
        if forward_loop is None:
            raise ValueError("static_inputs needs calibration audio (a forward_loop)")
        if not layout.chains:
            raise ValueError(
                f"{type(model).__name__} declares no GEMM chains (quantization_layout().chains); "
                "static_inputs has nothing to calibrate"
            )
        targets: List[str] = []
        formats = set()
        for chain in layout.chains:
            p_scheme, c_scheme = spec.resolve(chain.producer), spec.resolve(chain.consumer)
            if p_scheme is None or c_scheme is None:
                continue  # a chain with a full-precision end cannot run fused
            fmt = w8a8_format(c_scheme)
            if fmt is None or w8a8_format(p_scheme) != fmt:
                raise ValueError(
                    "static_inputs calibrates the 8-bit input scale a chained producer "
                    f"quantizes for; {chain.producer} -> {chain.consumer} is quantized with "
                    f"{p_scheme.name or p_scheme} -> {c_scheme.name or c_scheme}"
                )
            formats.add(fmt)
            targets.append(chain.consumer)
        if not targets:
            raise ValueError(
                "no GEMM chain has both ends quantized; static_inputs has nothing to do"
            )
        if len(formats) != 1:
            raise ValueError(f"static_inputs: the chains mix formats {sorted(formats)}")
        fmt = formats.pop()
        static_scheme = _STATIC_SCHEME[fmt]
        qmax = static_scheme.weights.qmax

        modules = dict(model.named_modules())
        amax_obs: Dict[str, Observer] = {p: AbsMaxPerTensor() for p in targets}
        _run(model, modules, amax_obs, forward_loop)
        amax = {p: float(amax_obs[p].result()) for p in targets}
        clip = dict(amax)
        if opts.observer != "absmax":
            hists: Dict[str, Observer] = {
                p: HistogramPerTensor(max(amax[p], QUANT_EPS), opts.bins) for p in targets
            }
            _run(model, modules, hists, forward_loop)
            for p in targets:
                counts = hists[p].result()
                if opts.observer == "percentile":
                    clip[p] = clip_percentile(counts, max(amax[p], QUANT_EPS), opts.percentile)
                else:
                    clip[p] = clip_mse(counts, max(amax[p], QUANT_EPS), qmax, fmt)

        for path in targets:
            c = torch.tensor([max(clip[path], QUANT_EPS)], dtype=torch.float32)
            # Tensor / tensor: an IEEE division, as everywhere in the pinned numerics.
            scale = c / torch.full_like(c, qmax)
            module = modules[path]
            if "input_scale" in module._buffers:
                raise RuntimeError(f"{path} already has an input_scale")
            module.register_buffer("input_scale", scale.cpu().clone())

        producer = dict(spec.producer)
        record = {"observer": opts.observer, "layers": len(targets)}
        if opts.observer == "percentile":
            record["percentile"] = opts.percentile
        if opts.observer != "absmax":
            # How far below the maximum the clip sits, worst and median layer.
            ratios = sorted(clip[p] / max(amax[p], QUANT_EPS) for p in targets)
            record["clip_over_max"] = [round(ratios[0], 4), round(ratios[len(ratios) // 2], 4)]
        producer["static_inputs"] = record
        # First match wins in ``QuantizationSpec.resolve``, so the static group goes
        # first; an exact path it now owns is also dropped from the group that had
        # it, so the written spec lists every layer once.  Globs stay as written.
        moved = set(targets)
        rest = []
        for group in spec.groups:
            kept = tuple(t for t in group.targets if t not in moved)
            if kept:
                rest.append(QuantGroup(group.scheme, kept))
        return dataclasses.replace(
            spec,
            groups=(QuantGroup(static_scheme, tuple(targets)),) + tuple(rest),
            producer=producer,
        )


#: The clip estimators ``observer`` names.
_OBSERVERS = ("absmax", "percentile", "mse")


def _run(
    model: nn.Module,
    modules: Dict[str, nn.Module],
    observers: Dict[str, Observer],
    forward_loop: ForwardLoop,
) -> None:
    """One calibration pass feeding each target's (smoothed) input to its observer."""
    handles = [
        modules[p].register_forward_pre_hook(_observe(obs, modules[p]))
        for p, obs in observers.items()
    ]
    try:
        forward_loop(model)
    finally:
        for handle in handles:
            handle.remove()


def _observe(observer: Observer, module: nn.Module):
    def hook(mod: nn.Module, args):  # noqa: ARG001 - torch hook signature
        x = args[0].detach().float()
        smooth = module._buffers.get("smooth_inv")
        if smooth is not None:
            x = x * smooth.float().to(x.device)
        observer.update(x)

    return hook


__all__ = ["StaticInputs", "StaticInputsOptions"]
