# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GEMM backend registrations for autotuning (FlashInfer-style).

All tile variants are pre-compiled into a single shared library by the JIT
layer (``oasr.jit.gemm``).  The autotuner selects which pre-compiled variant
to call — no JIT compilation is triggered during tuning.
"""

import functools
import logging
from typing import Union

from oasr.tune.autotuner import BackendEntry, _global_registry, OpKey, Tactic
from oasr.jit.gemm import (
    CutlassGemmConfig,
    CutlassGemmConfigSm90,
    TileShapeConfigs,
    TileShapeConfigsSm90,
    ClusterShapeConfigsSm90,
    get_unique_compile_configs,
    gemm_func_name,
    gemm_activation_func_name,
    bmm_func_name,
    group_gemm_func_name,
)
from oasr.jit.core import _get_target_sm

logger = logging.getLogger("oasr.tune")

_sm = _get_target_sm()
_all_gemm_configs = get_unique_compile_configs(
    _sm, TileShapeConfigs, TileShapeConfigsSm90, ClusterShapeConfigsSm90
)

# Default configs (mid-range tile picked heuristically)
_GEMM_DEFAULT_SM_LT90 = CutlassGemmConfig(
    BM=128, BN=128, BK=64, WM=64, WN=64, WK=64, kStages=3, kSmVersion=_sm
)
_GEMM_DEFAULT_SM90 = CutlassGemmConfigSm90(
    BM=128, BN=128, BK=128, CM=1, CN=1, CK=1, kSMs=1, kStages=3, kSmVersion=_sm
)
_GEMM_DEFAULT = _GEMM_DEFAULT_SM_LT90 if _sm < 90 else _GEMM_DEFAULT_SM90


# ---------------------------------------------------------------------------
# Pre-compiled module loaders (one module per family, contains ALL variants)
# ---------------------------------------------------------------------------

@functools.cache
def _get_gemm_module():
    from oasr.jit.gemm import gen_gemm_module
    return gen_gemm_module().build_and_load()


@functools.cache
def _get_bmm_module():
    from oasr.jit.gemm import gen_bmm_module
    return gen_bmm_module().build_and_load()


@functools.cache
def _get_group_gemm_module():
    from oasr.jit.gemm import gen_group_gemm_module
    return gen_group_gemm_module().build_and_load()


# ---------------------------------------------------------------------------
# GEMM registration
# ---------------------------------------------------------------------------

def _make_gemm_runner(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]):
    """Create a GEMM runner that calls a specific variant from the module."""
    fn_name = gemm_func_name(cfg)

    def runner():
        mod = _get_gemm_module()
        fn = getattr(mod, fn_name)

        def call(out, A, B, C, split_k_slices=1):
            fn(out, A, B, C, split_k_slices)

        return call

    return runner


def _make_gemm_activation_runner(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]):
    fn_name = gemm_activation_func_name(cfg)

    def runner():
        mod = _get_gemm_module()
        fn = getattr(mod, fn_name)

        def call(out, A, B, C, activation_type, split_k_slices=1):
            fn(out, A, B, C, activation_type, split_k_slices)

        return call

    return runner


for _cfg in _all_gemm_configs.values():
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.compile_name == _GEMM_DEFAULT.compile_name)

    _global_registry.register(
        OpKey("gemm", "gemm"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_gemm_runner(_cfg),
            is_fallback=_is_default,
        ),
    )

    _global_registry.register(
        OpKey("gemm", "gemm_activation"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_gemm_activation_runner(_cfg),
            is_fallback=_is_default,
        ),
    )


# ---------------------------------------------------------------------------
# BMM registration
# ---------------------------------------------------------------------------

def _make_bmm_runner(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]):
    fn_name = bmm_func_name(cfg)

    def runner():
        mod = _get_bmm_module()
        return getattr(mod, fn_name)

    return runner


for _cfg in _all_gemm_configs.values():
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.compile_name == _GEMM_DEFAULT.compile_name)

    _global_registry.register(
        OpKey("gemm", "bmm"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_bmm_runner(_cfg),
            is_fallback=_is_default,
        ),
    )


# ---------------------------------------------------------------------------
# Group GEMM registration
# ---------------------------------------------------------------------------

def _make_group_gemm_runner(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]):
    fn_name = group_gemm_func_name(cfg)

    def runner():
        mod = _get_group_gemm_module()
        return getattr(mod, fn_name)

    return runner


for _cfg in _all_gemm_configs.values():
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.compile_name == _GEMM_DEFAULT.compile_name)

    _global_registry.register(
        OpKey("gemm", "group_gemm"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_group_gemm_runner(_cfg),
            is_fallback=_is_default,
        ),
    )
