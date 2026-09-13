# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Which module implements which benchmark family.

Name -> dotted path, imported lazily.  Laziness is load-bearing: resolving the
whole table to list the families would import ``torch``, every kernel module and
``oasr.engine`` just to answer ``--help``.

``KERNEL_FAMILIES`` map onto ``oasr/functionals/`` one for one, which is the
rule for where a new benchmark goes: it belongs to the module it exercises.
``WORKLOAD_FAMILIES`` are whole pipelines and emit the workload schema.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    import types

from benchmarks.core.schema import CATEGORY_KERNEL, CATEGORY_WORKLOAD

KERNEL_FAMILIES: Dict[str, str] = {
    "gemm": "benchmarks.kernels.gemm",
    "norm": "benchmarks.kernels.norm",
    "conv": "benchmarks.kernels.conv",
    "activation": "benchmarks.kernels.activation",
    "attention": "benchmarks.kernels.attention",
    "softmax": "benchmarks.kernels.softmax",
    "topk": "benchmarks.kernels.topk",
    "fft": "benchmarks.kernels.fft",
    "pooling": "benchmarks.kernels.pooling",
    "recurrent": "benchmarks.kernels.recurrent",
    "mlp": "benchmarks.kernels.mlp",
    "composite": "benchmarks.kernels.composite",
    "feature": "benchmarks.features.frontend",
}

WORKLOAD_FAMILIES: Dict[str, str] = {
    "ctc_decoder": "benchmarks.decoders.ctc",
    "wfst_decoder": "benchmarks.decoders.wfst",
    "engine": "benchmarks.engine.engine",
    "service": "benchmarks.service.service",
    "accuracy": "benchmarks.accuracy.accuracy",
}

FAMILIES: Dict[str, str] = {**KERNEL_FAMILIES, **WORKLOAD_FAMILIES}

#: Families renamed by the consolidation, so an old command still resolves.
ALIASES: Dict[str, str] = {
    "gemm_log_softmax": "gemm",  # became a gemm subroutine -- same epilogue story
    "ctc": "ctc_decoder",
    "wfst": "wfst_decoder",
}


def category_of(name: str) -> str:
    return CATEGORY_KERNEL if resolve(name) in KERNEL_FAMILIES else CATEGORY_WORKLOAD


def resolve(name: str) -> str:
    """Canonical family name, following :data:`ALIASES`."""
    return ALIASES.get(name, name)


def get_family(name: str) -> "types.ModuleType":
    canonical = resolve(name)
    if canonical not in FAMILIES:
        raise ValueError(f"Unknown family '{name}'. Available: {list_families()}")
    return importlib.import_module(FAMILIES[canonical])


def list_families() -> List[str]:
    return sorted(FAMILIES)


def list_subroutines() -> Dict[str, List[str]]:
    """``{family: [subroutines]}`` -- imports every family, so call it sparingly."""
    return {name: list(get_family(name).SUBROUTINES) for name in list_families()}
