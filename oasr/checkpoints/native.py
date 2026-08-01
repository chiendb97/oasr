# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Native OASR on-disk checkpoint format (read/write).

Layout of a native checkpoint directory::

    oasr_config.json      # format_version, architecture, model/feature/tokenizer/decoding config
    model.safetensors     # OASR-native state dict (no name remapping needed)
    tokenizer/...         # tokenizer assets copied verbatim (units.txt, bpe.model, ...)

The weights are the *converted* ``model.state_dict()`` (written by
``oasr convert`` after ``load_weights`` ran), so loading is a strict
``load_state_dict`` — deterministic, mmap-able via safetensors, and free of any
WeNet / icefall / transformers dependency on the serving host.  Computed
buffers a model rebuilds from config (declared via
``Model._computed_buffer_suffixes``) are excluded from the file and expected to
be missing on load.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from oasr.features import FeatureSpec
from oasr.tokenizers import TokenizerSpec

from .bundle import ConvertedCheckpoint, DecodingDefaults

logger = logging.getLogger(__name__)

NATIVE_CONFIG_NAME = "oasr_config.json"
NATIVE_WEIGHTS_NAME = "model.safetensors"
NATIVE_TOKENIZER_DIR = "tokenizer"
FORMAT_VERSION = 1

# Aux-module placeholder builders, keyed by the aux kind recorded in
# oasr_config.json.  A placeholder only needs the right buffer shapes — the
# strict state-dict load overwrites the values.  New aux kinds register here
# alongside their model package.
AuxBuilder = Callable[[Dict[str, Any]], nn.Module]
_AUX_BUILDERS: Dict[str, AuxBuilder] = {}


def register_aux_builder(kind: str, builder: AuxBuilder) -> None:
    _AUX_BUILDERS[kind] = builder


def _build_global_cmvn(desc: Dict[str, Any]) -> nn.Module:
    from oasr.layers.norm import GlobalCMVN

    dim = int(desc["buffers"]["mean"][0])
    return GlobalCMVN(torch.zeros(dim), torch.ones(dim))


register_aux_builder("global_cmvn", _build_global_cmvn)


def is_native_checkpoint(ckpt_dir: Path) -> bool:
    return (Path(ckpt_dir) / NATIVE_CONFIG_NAME).exists()


def read_native_config(ckpt_dir: Path) -> Dict[str, Any]:
    path = Path(ckpt_dir) / NATIVE_CONFIG_NAME
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    version = cfg.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported native format_version {version!r} "
            f"(this OASR reads version {FORMAT_VERSION})"
        )
    return cfg


def _computed_suffixes(model: nn.Module) -> tuple:
    return tuple(getattr(model, "_computed_buffer_suffixes", ()))


def _describe_aux(aux: Dict[str, Any]) -> Dict[str, Any]:
    """Aux dict → JSON-safe descriptors (``None`` values are recorded as absent)."""
    desc: Dict[str, Any] = {}
    for name, value in aux.items():
        if value is None:
            desc[name] = None
        elif isinstance(value, nn.Module):
            desc[name] = {
                "kind": name,
                "buffers": {k: list(v.shape) for k, v in value.state_dict().items()},
            }
        else:
            raise TypeError(
                f"aux[{name!r}] is {type(value).__name__}; the native format can "
                "serialize None or nn.Module aux values only"
            )
    return desc


def _build_aux(desc: Dict[str, Any]) -> Dict[str, Any]:
    aux: Dict[str, Any] = {}
    for name, d in desc.items():
        if d is None:
            aux[name] = None
            continue
        try:
            builder = _AUX_BUILDERS[d["kind"]]
        except KeyError:
            raise KeyError(
                f"No aux builder registered for kind {d['kind']!r} "
                f"(registered: {sorted(_AUX_BUILDERS)})"
            ) from None
        aux[name] = builder(d)
    return aux


def save_native(
    dst_dir: Path,
    *,
    architecture: str,
    model: nn.Module,
    model_config: Any,
    aux: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[TokenizerSpec] = None,
    features: Optional[FeatureSpec] = None,
    decoding: Optional[DecodingDefaults] = None,
) -> Path:
    """Write a weight-loaded *model* (+ metadata) as a native checkpoint dir."""
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "safetensors is required to write native checkpoints; install it "
            "with `pip install oasr[hub]` or `pip install safetensors`"
        ) from exc

    from dataclasses import asdict

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    suffixes = _computed_suffixes(model)
    state = {
        k: v.detach().to("cpu").clone().contiguous()
        for k, v in model.state_dict().items()
        if not any(k.endswith(s) for s in suffixes)
    }
    save_file(state, str(dst_dir / NATIVE_WEIGHTS_NAME))

    # Copy tokenizer assets in and rewrite the spec's paths relative to dst.
    tokenizer_dict = None
    if tokenizer is not None:
        tok_dir = dst_dir / NATIVE_TOKENIZER_DIR
        tok_dir.mkdir(exist_ok=True)
        rel_files = {}
        for key, src in tokenizer.files.items():
            dst_file = tok_dir / Path(src).name
            if Path(src).resolve() != dst_file.resolve():
                shutil.copyfile(src, dst_file)
            rel_files[key] = f"{NATIVE_TOKENIZER_DIR}/{dst_file.name}"
        tokenizer_dict = {
            "kind": tokenizer.kind,
            "files": rel_files,
            "options": dict(tokenizer.options),
        }

    config = {
        "format_version": FORMAT_VERSION,
        "architecture": architecture,
        "model_config": asdict(model_config),
        "aux": _describe_aux(aux or {}),
        "tokenizer": tokenizer_dict,
        "features": features.to_dict() if features is not None else None,
        "decoding": (decoding or DecodingDefaults()).to_dict(),
    }
    with open(dst_dir / NATIVE_CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Wrote native checkpoint to %s (%d tensors)", dst_dir, len(state))
    return dst_dir


def load_native(ckpt_dir: Path, map_location: Any = "cpu") -> ConvertedCheckpoint:
    """Read a native checkpoint dir into a :class:`ConvertedCheckpoint` bundle."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "safetensors is required to read native checkpoints; install it "
            "with `pip install oasr[hub]` or `pip install safetensors`"
        ) from exc

    from oasr.models.registry import get_model_entry

    ckpt_dir = Path(ckpt_dir)
    cfg = read_native_config(ckpt_dir)
    architecture = cfg["architecture"]
    entry = get_model_entry(architecture)
    model_config = entry.config_cls.from_dict(cfg["model_config"])

    device = map_location if isinstance(map_location, str) else str(map_location)
    state_dict = load_file(str(ckpt_dir / NATIVE_WEIGHTS_NAME), device=device)

    tokenizer = None
    if cfg.get("tokenizer") is not None:
        spec = TokenizerSpec.from_dict(cfg["tokenizer"])
        spec.files = {k: str(ckpt_dir / rel) for k, rel in spec.files.items()}
        tokenizer = spec

    features = FeatureSpec.from_dict(cfg["features"]) if cfg.get("features") else None
    decoding = DecodingDefaults.from_dict(cfg.get("decoding") or {})

    return ConvertedCheckpoint(
        architecture=architecture,
        model_config=model_config,
        aux=_build_aux(cfg.get("aux") or {}),
        state_dict=state_dict,
        tokenizer=tokenizer,
        features=features,
        decoding=decoding,
        source_format="native",
    )


def load_native_weights(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Strict native state-dict load; only declared computed buffers may be missing.

    safetensors yields a plain tensor dict with no ``_metadata``, which would
    make version-gated ``_load_from_state_dict`` legacy remaps (e.g. the
    Conformer v1→v2 embed-linear permutation) re-fire on already-converted
    weights.  A native state dict is by definition in the *current* code's
    layout, so stamp the model's own per-module versions onto it.
    """
    from collections import OrderedDict

    sd: "OrderedDict[str, torch.Tensor]" = OrderedDict(state_dict)
    sd._metadata = model.state_dict()._metadata  # type: ignore[attr-defined]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    suffixes = _computed_suffixes(model)
    real_missing = [k for k in missing if not any(k.endswith(s) for s in suffixes)]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Native checkpoint does not match the model: "
            f"missing={real_missing} unexpected={list(unexpected)}"
        )
