# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""What quantized numbers look like, and which layers carry them.

:class:`QuantArgs` describes one tensor's format, :class:`QuantScheme` pairs a
weight format with an activation format, and :class:`QuantizationSpec` — the
``quantization`` section of a native checkpoint's ``oasr_config.json`` — maps
module paths to schemes.  The on-disk shape follows compressed-tensors'
``config_groups`` / ``targets`` / ``ignore`` (and its tensor names
``weight_scale`` / ``input_scale``), so an importer for those checkpoints is a
key mapping rather than a new format.

Pure data: nothing here knows about kernels or algorithms.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

if TYPE_CHECKING:  # pragma: no cover
    from torch import nn

#: Version of the ``quantization`` section's own schema.
SPEC_FORMAT = 1

_TYPES = ("int", "float")
_GRANULARITIES = ("tensor", "channel", "token", "group", "block")


@dataclass(frozen=True)
class QuantArgs:
    """The format of one quantized tensor."""

    num_bits: int
    type: str
    granularity: str
    symmetric: bool = True
    dynamic: bool = False
    group_size: Optional[int] = None
    scale_dtype: str = "float32"
    global_scale: bool = False

    def __post_init__(self) -> None:
        if self.type not in _TYPES:
            raise ValueError(f"QuantArgs.type must be one of {_TYPES}, got {self.type!r}")
        if self.granularity not in _GRANULARITIES:
            raise ValueError(
                f"QuantArgs.granularity must be one of {_GRANULARITIES}, got "
                f"{self.granularity!r}"
            )
        if self.granularity in ("group", "block") and not self.group_size:
            raise ValueError(f"{self.granularity} granularity needs a group_size")

    @property
    def qmax(self) -> float:
        """The largest representable magnitude (127, 448 for e4m3, 6 for e2m1)."""
        if self.type == "int":
            return float(2 ** (self.num_bits - 1) - 1)
        return {8: 448.0, 4: 6.0}[self.num_bits]

    @property
    def storage_dtype(self) -> torch.dtype:
        """The dtype the quantized values are stored in."""
        if self.type == "int" and self.num_bits == 8:
            return torch.int8
        if self.type == "float" and self.num_bits == 8:
            return torch.float8_e4m3fn
        if self.num_bits == 4:
            return torch.uint8  # two values per byte
        raise ValueError(f"no storage dtype for {self}")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "num_bits": self.num_bits,
            "type": self.type,
            "granularity": self.granularity,
            "symmetric": self.symmetric,
            "dynamic": self.dynamic,
        }
        if self.group_size is not None:
            d["group_size"] = self.group_size
        if self.scale_dtype != "float32":
            d["scale_dtype"] = self.scale_dtype
        if self.global_scale:
            d["global_scale"] = True
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuantArgs":
        return cls(
            num_bits=int(d["num_bits"]),
            type=str(d["type"]),
            granularity=str(d["granularity"]),
            symmetric=bool(d.get("symmetric", True)),
            dynamic=bool(d.get("dynamic", False)),
            group_size=None if d.get("group_size") is None else int(d["group_size"]),
            scale_dtype=str(d.get("scale_dtype", "float32")),
            global_scale=bool(d.get("global_scale", False)),
        )


@dataclass(frozen=True)
class QuantScheme:
    """A weight format and an activation format (``None`` = weight-only)."""

    weights: QuantArgs
    input_activations: Optional[QuantArgs]
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"scheme": self.name, "weights": self.weights.to_dict()}
        if self.input_activations is not None:
            d["input_activations"] = self.input_activations.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuantScheme":
        acts = d.get("input_activations")
        return cls(
            weights=QuantArgs.from_dict(d["weights"]),
            input_activations=None if acts is None else QuantArgs.from_dict(acts),
            name=str(d.get("scheme", "")),
        )


INT8_W8A8 = QuantScheme(
    QuantArgs(8, "int", "channel"), QuantArgs(8, "int", "token", dynamic=True), "int8_w8a8"
)
INT8_W8A8_STATIC = QuantScheme(
    QuantArgs(8, "int", "channel"), QuantArgs(8, "int", "tensor"), "int8_w8a8_static"
)
FP8_W8A8 = QuantScheme(
    QuantArgs(8, "float", "channel"), QuantArgs(8, "float", "token", dynamic=True), "fp8_w8a8"
)
FP8_W8A8_STATIC = QuantScheme(
    QuantArgs(8, "float", "channel"), QuantArgs(8, "float", "tensor"), "fp8_w8a8_static"
)
_NVFP4_ARGS = QuantArgs(
    4, "float", "group", group_size=16, scale_dtype="float8_e4m3fn", global_scale=True
)
NVFP4_W4A4 = QuantScheme(
    _NVFP4_ARGS,
    QuantArgs(
        4,
        "float",
        "group",
        dynamic=True,
        group_size=16,
        scale_dtype="float8_e4m3fn",
        global_scale=True,
    ),
    "nvfp4",
)
NVFP4_W4A16 = QuantScheme(_NVFP4_ARGS, None, "nvfp4_w4a16")

#: Named schemes.  A preset existing is not a promise that a method serves it:
#: :func:`oasr.quantization.methods.method_for` decides that, and refuses by name.
PRESETS: Dict[str, QuantScheme] = {
    s.name: s
    for s in (INT8_W8A8, INT8_W8A8_STATIC, FP8_W8A8, FP8_W8A8_STATIC, NVFP4_W4A4, NVFP4_W4A16)
}


def get_scheme(name: str) -> QuantScheme:
    """The preset called *name*."""
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown quantization scheme {name!r}; presets: {sorted(PRESETS)}"
        ) from None


def check_scheme_is_batch_invariant(scheme: QuantScheme) -> None:
    """Refuse a per-tensor *dynamic* activation scale.

    Offline transcripts must not depend on what shares a batch (``docs/ci.md``:
    bit-identical across ``max_batch_size``), and CUDA graphs zero-pad rows and
    frames.  A scale computed over the whole activation tensor would let padding
    and co-batched utterances change every valid row's quantization.  Per-token
    (row-local) dynamic and static per-tensor scales are both invariant.
    """
    acts = scheme.input_activations
    if acts is not None and acts.dynamic and acts.granularity == "tensor":
        raise ValueError(
            f"scheme {scheme.name or scheme}: a per-tensor dynamic activation scale makes a "
            "row's quantization depend on the rest of the batch (padding included); use "
            "granularity='token' (dynamic) or a static per-tensor scale"
        )


@dataclass(frozen=True)
class QuantGroup:
    """One scheme and the module-path globs it applies to."""

    scheme: QuantScheme
    targets: Tuple[str, ...]


@dataclass(frozen=True)
class QuantizationSpec:
    """Which layers are quantized, how, and by what process.

    Attributes:
        groups: ``(scheme, targets)`` pairs; a layer takes the first group a
            target glob of which matches its module path.
        ignore: Globs kept full precision; they win over every target.
        runtime_smoothed: Paths of layers carrying a runtime SmoothQuant
            ``smooth_inv`` vector (their producer could not absorb it).
        producer: Provenance — algorithm, options, calibration data hash.
    """

    groups: Tuple[QuantGroup, ...]
    ignore: Tuple[str, ...] = ()
    runtime_smoothed: Tuple[str, ...] = ()
    producer: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for group in self.groups:
            check_scheme_is_batch_invariant(group.scheme)

    # -- resolution ----------------------------------------------------------
    def is_ignored(self, path: str) -> bool:
        return any(fnmatch.fnmatchcase(path, glob) for glob in self.ignore)

    def resolve(self, path: str) -> Optional[QuantScheme]:
        """The scheme for the layer at *path*, or ``None`` (full precision)."""
        if self.is_ignored(path):
            return None
        for group in self.groups:
            if any(fnmatch.fnmatchcase(path, glob) for glob in group.targets):
                return group.scheme
        return None

    @property
    def schemes(self) -> Tuple[QuantScheme, ...]:
        return tuple(g.scheme for g in self.groups)

    def check_against(self, model: "nn.Module") -> None:
        """Declare, don't ignore: refuse a spec that does not fit *model*.

        * a target glob that matches no quantizable layer (a typo, or a spec
          written for another architecture);
        * a target that selects a layer whose weight something else reads raw
          (``quantizable = False``: the transducer decode kernels) — quantizing
          it would hand that reader 8-bit storage;
        * a runtime-smoothed path that is not a quantized layer.
        """
        from oasr.layers._quantizable import QuantizableGemm

        gemms = {p: m for p, m in model.named_modules() if isinstance(m, QuantizableGemm)}
        for group in self.groups:
            for glob in group.targets:
                hits = [p for p in gemms if fnmatch.fnmatchcase(p, glob)]
                if not hits:
                    raise ValueError(
                        f"quantization target {glob!r} matches no quantizable layer of "
                        f"{type(model).__name__}"
                    )
                for path in hits:
                    if self.is_ignored(path):
                        continue
                    if not getattr(gemms[path], "quantizable", True):
                        raise ValueError(
                            f"quantization target {glob!r} selects {path}, whose weight is "
                            "read raw by a fused kernel (quantizable=False); add it to "
                            "`ignore`"
                        )
        for path in self.runtime_smoothed:
            if path not in gemms or self.resolve(path) is None:
                raise ValueError(f"runtime_smoothed names {path!r}, which is not quantized")

    # -- construction ----------------------------------------------------------
    @classmethod
    def for_model(
        cls,
        model: "nn.Module",
        scheme: QuantScheme,
        *,
        ignore: Iterable[str] = (),
        producer: Optional[Mapping[str, Any]] = None,
    ) -> "QuantizationSpec":
        """A spec quantizing every quantizable layer of *model* with *scheme*.

        Kept at full precision: what the model's ``quantization_layout()``
        declares, layers whose weight is read raw (``quantizable = False``) and
        any extra *ignore* globs.  Targets are written out as exact paths, so the
        checkpoint records precisely what was quantized.
        """
        from oasr.layers._quantizable import QuantizableGemm

        keep = tuple(_model_keep_full_precision(model)) + tuple(ignore)
        targets = tuple(
            p
            for p, m in model.named_modules()
            if isinstance(m, QuantizableGemm)
            and getattr(m, "quantizable", True)
            and not any(fnmatch.fnmatchcase(p, g) for g in keep)
        )
        if not targets:
            raise ValueError(f"{type(model).__name__} has no quantizable layer left to quantize")
        return cls(
            groups=(QuantGroup(scheme, targets),),
            ignore=tuple(ignore),
            producer=dict(producer or {}),
        )

    # -- serialization ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "format": SPEC_FORMAT,
            "groups": [{**g.scheme.to_dict(), "targets": list(g.targets)} for g in self.groups],
            "ignore": list(self.ignore),
        }
        if self.runtime_smoothed:
            d["runtime_smoothed"] = list(self.runtime_smoothed)
        if self.producer:
            d["producer"] = dict(self.producer)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuantizationSpec":
        version = d.get("format")
        if version != SPEC_FORMAT:
            raise ValueError(
                f"unsupported quantization section format {version!r} (this OASR reads "
                f"{SPEC_FORMAT})"
            )
        groups = tuple(
            QuantGroup(QuantScheme.from_dict(g), tuple(str(t) for t in g["targets"]))
            for g in d["groups"]
        )
        return cls(
            groups=groups,
            ignore=tuple(str(g) for g in d.get("ignore", ())),
            runtime_smoothed=tuple(str(p) for p in d.get("runtime_smoothed", ())),
            producer=dict(d.get("producer") or {}),
        )


def _model_keep_full_precision(model: "nn.Module") -> Tuple[str, ...]:
    layout_fn = getattr(model, "quantization_layout", None)
    if layout_fn is None:
        return ()
    return tuple(layout_fn().keep_full_precision)
