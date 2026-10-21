# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline post-training quantization: model → quantized model → native checkpoint.

:func:`quantize_model` is ModelOpt's ``mtq.quantize(model, config, forward_loop)``
shape: the recipe's algorithms transform the fp32 master in order, then every
selected layer quantizes its own weight through ``QuantMethod.quantize_weight``
— the same code the online path runs, so the two cannot disagree.
:func:`quantize_checkpoint` is ``oasr quantize``: load, calibrate, quantize,
write a format-version-2 native checkpoint plus a provenance report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import nn

from .algorithms.base import ForwardLoop, build_algorithm
from .apply import apply_quantization, finalize_quantization, quantized_layers
from .recipes import Recipe, get_recipe
from .spec import QuantizationSpec, get_scheme

logger = logging.getLogger(__name__)

REPORT_NAME = "quantization_report.json"


def quantize_model(
    model: nn.Module,
    recipe: Recipe,
    forward_loop: Optional[ForwardLoop] = None,
    *,
    algorithm_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ignore: Iterable[str] = (),
    producer: Optional[Mapping[str, Any]] = None,
) -> QuantizationSpec:
    """Quantize *model* in place with *recipe*; returns the spec it now carries.

    Args:
        model: An fp32 master model (weights loaded, not yet quantized).
        recipe: Scheme + algorithms.
        forward_loop: Runs calibration audio through *model*; required by any
            algorithm with ``needs_calibration``.
        algorithm_options: ``{algorithm name: {option: value}}``.
        ignore: Extra module-path globs to keep at full precision.
        producer: Extra provenance to record (calibration data, versions).
    """
    if getattr(model, "quantization_spec", None) is not None:
        raise ValueError("the model is already quantized")
    spec = QuantizationSpec.for_model(
        model, get_scheme(recipe.scheme), ignore=ignore, producer=producer
    )
    spec.check_against(model)
    layout_fn = getattr(model, "quantization_layout", None)
    from .layout import QuantLayout

    layout = layout_fn() if layout_fn is not None else QuantLayout()
    options = dict(algorithm_options or {})
    for name in recipe.algorithms:
        algorithm = build_algorithm(name, options.pop(name, None))
        if algorithm.needs_calibration and forward_loop is None:
            raise ValueError(f"recipe {recipe.name!r}: {name} needs calibration audio")
        spec = algorithm.apply(model, layout, spec, forward_loop)
    if options:
        raise ValueError(f"options given for algorithms not in the recipe: {sorted(options)}")
    spec = QuantizationSpec(
        groups=spec.groups,
        ignore=spec.ignore,
        runtime_smoothed=spec.runtime_smoothed,
        producer={**dict(spec.producer), "recipe": recipe.name},
    )
    apply_quantization(model, spec, quantize=True)
    finalize_quantization(model)
    setattr(model, "quantization_spec", spec)  # noqa: B010 - not an nn.Module slot
    return spec


def layer_report(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Per quantized layer: shape, weight-scale range, smoother range."""
    rows: Dict[str, Dict[str, Any]] = {}
    for path, layer in quantized_layers(model).items():
        scale = getattr(layer, "weight_scale").detach().float()  # noqa: B009
        row: Dict[str, Any] = {
            "shape": [layer.out_features, layer.in_features],
            "weight_scale": [float(scale.min()), float(scale.max())],
        }
        smooth = getattr(layer, "smooth_inv", None)
        if isinstance(smooth, torch.Tensor):
            inv = smooth.detach().float()
            row["smooth_inv"] = [float(inv.min()), float(inv.max())]
        rows[path] = row
    return rows


def quantize_checkpoint(
    src: str,
    dst: str,
    *,
    recipe: str,
    calib_manifest: Optional[str] = None,
    audio_root: Optional[str] = None,
    calib_utterances: int = 128,
    algorithm_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ignore: Iterable[str] = (),
    architecture: Optional[str] = None,
    checkpoint_name: str = "final.pt",
    device: str = "cuda",
) -> Path:
    """``oasr quantize``: write a quantized native checkpoint of *src* at *dst*."""
    from oasr.checkpoints.native import save_native
    from oasr.models.registry import load_checkpoint_bundle, materialize_model

    rec = get_recipe(recipe)
    arch, bundle = load_checkpoint_bundle(
        Path(src), checkpoint_name=checkpoint_name, architecture=architecture
    )
    if getattr(bundle, "quantization", None) is not None:
        raise ValueError(f"{src} is already a quantized checkpoint")
    # The fp32 master, on the host, through the shared load sequence (legacy
    # remaps and padding included); then onto the calibration device in fp32.
    model, config, _report = materialize_model(arch, bundle)
    model = model.to(device)

    forward_loop = None
    producer: Dict[str, Any] = {"source": Path(src).name, "torch": torch.__version__}
    needs_audio = any(build_algorithm(a).needs_calibration for a in rec.algorithms)
    if needs_audio:
        if calib_manifest is None:
            raise ValueError(f"recipe {recipe!r} needs --calib-manifest")
        from .calibration import CheckpointFrontend, asr_forward_loop, select_manifest

        calibration = select_manifest(
            Path(calib_manifest), Path(audio_root) if audio_root else None, calib_utterances
        )
        frontend = CheckpointFrontend(bundle.features, torch.device(device))
        forward_loop = asr_forward_loop(calibration, frontend)
        producer["calibration"] = calibration.provenance

    spec = quantize_model(
        model,
        rec,
        forward_loop,
        algorithm_options=algorithm_options,
        ignore=ignore,
        producer=producer,
    )
    model = model.to("cpu")
    out = save_native(
        Path(dst),
        architecture=arch,
        model=model,
        model_config=config,
        aux=bundle.aux,
        tokenizer=bundle.tokenizer,
        features=bundle.features,
        decoding=bundle.decoding,
        quantization=spec,
    )
    report = {"spec": spec.to_dict(), "layers": layer_report(model)}
    with open(Path(dst) / REPORT_NAME, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "Wrote %s-quantized checkpoint to %s (%d layers)",
        rec.name,
        dst,
        len(quantized_layers(model)),
    )
    return out


__all__ = ["REPORT_NAME", "layer_report", "quantize_checkpoint", "quantize_model"]
