# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-style runtime autotuning for OASR kernels.

Quick start::

    import oasr

    # Profile and cache the fastest tile config for each operation
    with oasr.autotune(cache="oasr_tune.json"):
        output = oasr.gemm(A, B)

    # Reuse cached configs without profiling
    with oasr.autotune(False, cache="oasr_tune.json"):
        output = oasr.gemm(A, B)

    # Or use the global toggle (no context manager)
    oasr.enable_autotune(cache="oasr_tune.json")
    output = oasr.gemm(A, B)
    oasr.disable_autotune()
"""

# Single-file autotuner following FlashInfer's pattern.
# All types, registry, cache, profiling, and orchestration live in autotuner.py.
from .autotuner import (
    AutoTuner,
    AutoTunerStatistics,
    BackendEntry,
    BackendRegistry,
    OpKey,
    ProfileKey,
    Tactic,
    TuneResult,
    _global_registry,
    autotune,
    clear_cache,
    disable_autotune,
    enable_autotune,
    get_selected_config,
    get_tuner,
    is_tuning_enabled,
    load_configs,
    save_configs,
)

__all__ = [
    # Context manager and global toggles
    "autotune",
    "enable_autotune",
    "disable_autotune",
    "is_tuning_enabled",
    # AutoTuner singleton
    "AutoTuner",
    "AutoTunerStatistics",
    "get_tuner",
    # Config persistence
    "load_configs",
    "save_configs",
    "clear_cache",
    "get_selected_config",
    # Types
    "OpKey",
    "ProfileKey",
    "Tactic",
    "TuneResult",
    # Registry
    "BackendEntry",
    "BackendRegistry",
    "_global_registry",
]
