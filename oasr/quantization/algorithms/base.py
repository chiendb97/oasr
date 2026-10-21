# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Quantization algorithms: how the numbers are chosen.

An algorithm transforms the fp32 master model before quantization — folding
smoothing scales, adjusting weights (GPTQ), searching clipping ranges — given
the model's declared :class:`~oasr.quantization.layout.QuantLayout`, the
:class:`~oasr.quantization.spec.QuantizationSpec` being built and a
``forward_loop(model)`` over calibration audio.  It never quantizes: that is
``apply_quantization(quantize=True)``, run once after every algorithm, through
the same ``QuantMethod.quantize_weight`` the online path uses.

Registered by name with an ``options_cls`` (a frozen dataclass), like the
decode strategies.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Type

from torch import nn

from ..layout import QuantLayout
from ..spec import QuantizationSpec

ForwardLoop = Callable[[nn.Module], None]


class QuantAlgorithm(ABC):
    """One step of a quantization recipe."""

    name: ClassVar[str]
    options_cls: ClassVar[Optional[type]] = None
    #: Whether the algorithm needs calibration audio (a ``forward_loop``).
    needs_calibration: ClassVar[bool] = False

    def __init__(self, options: Any = None) -> None:
        self.options = options

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        layout: QuantLayout,
        spec: QuantizationSpec,
        forward_loop: Optional[ForwardLoop],
    ) -> QuantizationSpec:
        """Transform *model* in place; return the (possibly amended) spec."""


_ALGORITHMS: Dict[str, Type[QuantAlgorithm]] = {}


def register_quant_algorithm(name: str):
    """Class decorator registering a :class:`QuantAlgorithm` under *name*."""

    def wrap(cls: Type[QuantAlgorithm]) -> Type[QuantAlgorithm]:
        cls.name = name
        _ALGORITHMS[name] = cls
        return cls

    return wrap


def build_algorithm(name: str, options: Optional[Mapping[str, Any]] = None) -> QuantAlgorithm:
    """An algorithm instance, its options typed from *options* (strings allowed)."""
    from . import rtn, smoothquant, static_inputs  # noqa: F401  -- register the built-ins

    cls = _ALGORITHMS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown quantization algorithm {name!r}; registered: {sorted(_ALGORITHMS)}"
        )
    return cls(_build_options(cls, dict(options or {})))


def _build_options(cls: Type[QuantAlgorithm], values: Dict[str, Any]) -> Any:
    if cls.options_cls is None:
        if values:
            raise ValueError(f"algorithm {cls.name!r} takes no options, got {sorted(values)}")
        return None
    fields = {f.name: f for f in dataclasses.fields(cls.options_cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown {cls.name} option(s) {sorted(unknown)}; known: {sorted(fields)}")
    typed = {}
    for key, value in values.items():
        default = fields[key].default
        if isinstance(value, str) and default is not dataclasses.MISSING:
            if isinstance(default, bool):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, (int, float)) and default is not None:
                value = type(default)(value)
            elif default is None:
                value = float(value)
        typed[key] = value
    return cls.options_cls(**typed)


def list_algorithms() -> Dict[str, Type[QuantAlgorithm]]:
    from . import rtn, smoothquant  # noqa: F401

    return dict(_ALGORITHMS)
