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
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import nn

from .algorithms.base import ForwardLoop, build_algorithm
from .algorithms.static_inputs import uncalibrated_static_layers
from .apply import apply_quantization, finalize_quantization, quantized_layers
from .recipes import Recipe, get_recipe
from .spec import STATIC_INPUT_KINDS, QuantizationSpec, get_scheme, resolve_module_types

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
    module_types: Optional[Iterable[str]] = None,
) -> QuantizationSpec:
    """Quantize *model* in place with *recipe*; returns the spec it now carries.

    Args:
        model: An fp32 master model (weights loaded, not yet quantized).
        recipe: Scheme + algorithms.
        forward_loop: Runs calibration audio through *model*; required by any
            algorithm with ``needs_calibration``, and by a ``conv1d`` / ``conv2d``
            in *module_types* (its static input scale is calibrated).
        algorithm_options: ``{algorithm name: {option: value}}``.
        ignore: Extra module-path globs to keep at full precision.
        producer: Extra provenance to record (calibration data, versions).
        module_types: Layer kinds to quantize (``QuantizationConfig.module_types``);
            ``None`` is the dense GEMMs.
    """
    if getattr(model, "quantization_spec", None) is not None:
        raise ValueError("the model is already quantized")
    spec = QuantizationSpec.for_model(
        model,
        get_scheme(recipe.scheme),
        ignore=ignore,
        producer=producer,
        module_types=module_types,
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
    # A statically quantized layer the recipe did not calibrate (a convolution,
    # whose input scale cannot be per-token) is calibrated here.
    conv_options = options.pop("static_conv_inputs", None)
    if uncalibrated_static_layers(model, spec):
        algorithm = build_algorithm("static_conv_inputs", conv_options)
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
    calibration_dtype: Optional[str] = None,
    module_types: Optional[Iterable[str]] = None,
) -> Path:
    """``oasr quantize``: write a quantized native checkpoint of *src* at *dst*.

    ``module_types`` selects the layer kinds to quantize (``QuantizationConfig``'s
    names; ``None`` is the dense GEMMs).  A ``conv1d`` / ``conv2d`` kind takes a
    calibrated static input scale, so it needs ``calib_manifest`` with any recipe.

    ``calibration_dtype`` (``"bfloat16"`` / ``"float16"``) keeps the fp32 master
    on the host and runs the calibration audio through a copy of it in that dtype
    on *device* (:func:`mirrored_forward_loop`): the statistics are taken in the
    reduced precision, every scale, fold and weight quantization stays fp32 on the
    host.  It is what lets a model whose fp32 master does not fit the device -- an
    8B speech LLM is 33.6 GB -- be calibrated at all.  ``None`` / ``"float32"``
    calibrates the fp32 master itself on *device*.
    """
    from oasr.checkpoints.native import save_native
    from oasr.models.registry import load_checkpoint_bundle, materialize_model

    rec = get_recipe(recipe)
    arch, bundle = load_checkpoint_bundle(
        Path(src), checkpoint_name=checkpoint_name, architecture=architecture
    )
    if getattr(bundle, "quantization", None) is not None:
        raise ValueError(f"{src} is already a quantized checkpoint")
    calib_dtype = _calibration_dtype(calibration_dtype)
    # The fp32 master, on the host, through the shared load sequence (legacy
    # remaps and padding included); then onto the calibration device in fp32 --
    # or left on the host, with a reduced-precision twin on the device.
    model, config, _report = materialize_model(arch, bundle)
    needs_audio = any(build_algorithm(a).needs_calibration for a in rec.algorithms) or any(
        k in STATIC_INPUT_KINDS for k in resolve_module_types(module_types)
    )
    twin: Optional[nn.Module] = None
    if calib_dtype == torch.float32 or not needs_audio:
        model = model.to(device)
    else:
        twin, _, _ = materialize_model(arch, bundle, device=device, dtype=calib_dtype)

    forward_loop = None
    producer: Dict[str, Any] = {"source": Path(src).name, "torch": torch.__version__}
    if needs_audio:
        if calib_manifest is None:
            raise ValueError(
                f"recipe {recipe!r} with module types {list(resolve_module_types(module_types))} "
                "needs --calib-manifest"
            )
        from .calibration import CheckpointFrontend, asr_forward_loop, select_manifest

        calibration = select_manifest(
            Path(calib_manifest), Path(audio_root) if audio_root else None, calib_utterances
        )
        frontend = CheckpointFrontend(bundle.features, torch.device(device))
        forward_loop = asr_forward_loop(calibration, frontend, _calibration_tokenizer(bundle))
        if twin is not None:
            forward_loop = mirrored_forward_loop(forward_loop, twin)
        producer["calibration"] = {
            **calibration.provenance,
            "dtype": str(calib_dtype).replace("torch.", ""),
        }

    spec = quantize_model(
        model,
        rec,
        forward_loop,
        algorithm_options=algorithm_options,
        ignore=ignore,
        producer=producer,
        module_types=module_types,
    )
    del twin, forward_loop
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


def _calibration_dtype(name: Optional[str]) -> torch.dtype:
    if name is None:
        return torch.float32
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtypes.get(str(name).replace("torch.", ""))
    if dtype is None:
        raise ValueError(f"calibration dtype must be one of {sorted(dtypes)}, got {name!r}")
    return dtype


def _calibration_tokenizer(bundle: Any) -> Any:
    """The checkpoint's tokenizer, for a decoder whose calibration prompt is text."""
    spec = getattr(bundle, "tokenizer", None)
    if spec is None:
        return None
    from oasr.tokenizers import build_tokenizer

    try:
        return build_tokenizer(spec)
    except Exception as exc:  # noqa: BLE001 - only a text-prompted decoder needs it
        logger.warning("calibration runs without the checkpoint's tokenizer: %s", exc)
        return None


def mirrored_forward_loop(forward_loop: ForwardLoop, twin: nn.Module) -> ForwardLoop:
    """Run *forward_loop* on *twin* whenever an algorithm asks to run it on the master.

    An algorithm hooks the master's modules and calls ``forward_loop(master)``;
    this copies the master's current parameters and buffers into the twin (cast to
    its dtype and device, so every fold applied so far is in the activations the
    hooks see), mirrors the master's forward hooks onto the twin's modules at the
    same paths, and runs the loop there.  The hooks are the algorithm's own
    closures, so their statistics land where the algorithm reads them.
    """

    def loop(master: nn.Module) -> None:
        _copy_state(master, twin)
        handles = _mirror_hooks(master, twin)
        try:
            forward_loop(twin)
        finally:
            for handle in handles:
                handle.remove()

    return loop


@torch.no_grad()
def _copy_state(master: nn.Module, twin: nn.Module) -> None:
    targets: Dict[str, torch.Tensor] = dict(twin.named_parameters())
    targets.update(dict(twin.named_buffers()))
    for name, tensor in [*master.named_parameters(), *master.named_buffers()]:
        target = targets.get(name)
        if target is None or not tensor.is_floating_point() or target.shape != tensor.shape:
            continue  # an algorithm's own buffers (smooth_inv): read off the master
        target.copy_(tensor)


def _mirror_hooks(master: nn.Module, twin: nn.Module) -> List[Any]:
    twin_modules = dict(twin.named_modules())
    handles: List[Any] = []
    for path, module in master.named_modules():
        pre, post = module._forward_pre_hooks, module._forward_hooks
        if not pre and not post:
            continue
        target = twin_modules.get(path)
        if target is None:
            raise RuntimeError(f"calibration twin has no module {path!r} to mirror hooks onto")
        for key, hook in pre.items():
            kwargs = key in module._forward_pre_hooks_with_kwargs
            handles.append(target.register_forward_pre_hook(hook, with_kwargs=kwargs))
        for key, hook in post.items():
            kwargs = key in module._forward_hooks_with_kwargs
            handles.append(target.register_forward_hook(hook, with_kwargs=kwargs))
    return handles


__all__ = [
    "REPORT_NAME",
    "layer_report",
    "mirrored_forward_loop",
    "quantize_checkpoint",
    "quantize_model",
]
