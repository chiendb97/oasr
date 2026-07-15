# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Torch/cuBLAS backend registrations for GEMM autotuning.

Registers a single ``Tactic("torch")`` candidate per op (``gemm`` /
``gemm_activation`` / ``bmm``) so the autotuner can discover the shapes where
cuBLAS beats CUTLASS — typically thin streaming shapes where the fixed CUTLASS
tile wastes rows.  The runners live in :mod:`oasr.gemm_torch` and are shared with
the shape-aware production selector in :mod:`oasr.gemm`, so a tuned ``"torch"``
tactic dispatches identically in both the autotuning and production paths.
"""

from oasr.gemm_torch import (
    torch_bmm,
    torch_gemm,
    torch_gemm_activation,
    torch_gemm_log_softmax,
)
from oasr.tune.autotuner import BackendEntry, OpKey, Tactic, _global_registry

_global_registry.register(
    OpKey("gemm", "gemm"),
    BackendEntry(
        tactic=Tactic("torch"),
        is_available=lambda: True,
        get_runner=lambda: torch_gemm,
        is_fallback=False,
    ),
)

_global_registry.register(
    OpKey("gemm", "gemm_activation"),
    BackendEntry(
        tactic=Tactic("torch"),
        is_available=lambda: True,
        get_runner=lambda: torch_gemm_activation,
        is_fallback=False,
    ),
)

_global_registry.register(
    OpKey("gemm", "bmm"),
    BackendEntry(
        tactic=Tactic("torch"),
        is_available=lambda: True,
        get_runner=lambda: torch_bmm,
        is_fallback=False,
    ),
)

_global_registry.register(
    OpKey("gemm", "gemm_log_softmax"),
    BackendEntry(
        tactic=Tactic("torch"),
        is_available=lambda: True,
        get_runner=lambda: torch_gemm_log_softmax,
        is_fallback=False,
    ),
)
