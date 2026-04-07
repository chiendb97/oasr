"""OASR benchmark routine registry.

Each routine module exposes:
  - SUBROUTINES: list[str]      — available sub-kernel names
  - parse_args(parser)           — add routine-specific CLI args
  - run_test(args, output)       — run a single benchmark test
  - get_default_configs()        — default configs per subroutine
  - run_standalone()             — backwards-compat standalone entry point
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

ROUTINE_REGISTRY: dict[str, str] = {
    "gemm": "benchmarks.routines.gemm",
    "norm": "benchmarks.routines.norm",
    "conv": "benchmarks.routines.conv",
    "activation": "benchmarks.routines.activation",
    "composite": "benchmarks.routines.composite",
    "ctc_decoder": "benchmarks.routines.ctc_decoder",
}


def get_routine(name: str) -> types.ModuleType:
    """Import and return the routine module for *name*."""
    if name not in ROUTINE_REGISTRY:
        raise ValueError(
            f"Unknown routine '{name}'. Available: {list(ROUTINE_REGISTRY.keys())}"
        )
    return importlib.import_module(ROUTINE_REGISTRY[name])


def list_routines() -> list[str]:
    """Return sorted list of available routine names."""
    return sorted(ROUTINE_REGISTRY.keys())


def list_all_subroutines() -> dict[str, list[str]]:
    """Return {routine: [subroutines]} for all registered routines."""
    result = {}
    for name in sorted(ROUTINE_REGISTRY.keys()):
        mod = get_routine(name)
        result[name] = list(mod.SUBROUTINES)
    return result
