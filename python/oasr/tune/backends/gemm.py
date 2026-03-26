# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GEMM backend registrations for autotuning (FlashInfer-style).

All tile variants are pre-compiled into a single shared library by the JIT
layer (``oasr.jit.gemm``).  The autotuner selects which pre-compiled variant
to call — no JIT compilation is triggered during tuning.
"""

import functools
import logging

from oasr.tune.autotuner import BackendEntry, _global_registry, OpKey, Tactic
from oasr.jit.gemm import (
    TileConfig,
    GEMM_DEFAULT,
    GEMM_TILE_CONFIGS,
    BMM_TILE_CONFIGS,
    GROUP_GEMM_TILE_CONFIGS,
    gemm_func_name,
    gemm_activation_func_name,
    bmm_func_name,
    group_gemm_func_name,
)

logger = logging.getLogger("oasr.tune")


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

def _make_gemm_runner(cfg: TileConfig):
    """Create a GEMM runner that calls a specific variant from the module."""
    split_k = cfg.split_k
    fn_name = gemm_func_name(cfg)

    def runner():
        mod = _get_gemm_module()
        fn = getattr(mod, fn_name)

        def call(out, A, B, C, split_k_slices=split_k):
            fn(out, A, B, C, split_k_slices)

        return call

    return runner


def _make_gemm_activation_runner(cfg: TileConfig):
    split_k = cfg.split_k
    fn_name = gemm_activation_func_name(cfg)

    def runner():
        mod = _get_gemm_module()
        fn = getattr(mod, fn_name)

        def call(out, A, B, C, activation_type, split_k_slices=split_k):
            fn(out, A, B, C, activation_type, split_k_slices)

        return call

    return runner


for _cfg in GEMM_TILE_CONFIGS:
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.tile_m == GEMM_DEFAULT.tile_m
                   and _cfg.tile_n == GEMM_DEFAULT.tile_n
                   and _cfg.tile_k == GEMM_DEFAULT.tile_k
                   and _cfg.stages == GEMM_DEFAULT.stages
                   and _cfg.split_k == 1)

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

def _make_bmm_runner(cfg: TileConfig):
    fn_name = bmm_func_name(cfg)

    def runner():
        mod = _get_bmm_module()
        return getattr(mod, fn_name)

    return runner


for _cfg in BMM_TILE_CONFIGS:
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.tile_m == GEMM_DEFAULT.tile_m
                   and _cfg.tile_n == GEMM_DEFAULT.tile_n
                   and _cfg.tile_k == GEMM_DEFAULT.tile_k
                   and _cfg.stages == GEMM_DEFAULT.stages)

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

def _make_group_gemm_runner(cfg: TileConfig):
    fn_name = group_gemm_func_name(cfg)

    def runner():
        mod = _get_group_gemm_module()
        return getattr(mod, fn_name)

    return runner


for _cfg in GROUP_GEMM_TILE_CONFIGS:
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.tile_m == GEMM_DEFAULT.tile_m
                   and _cfg.tile_n == GEMM_DEFAULT.tile_n
                   and _cfg.tile_k == GEMM_DEFAULT.tile_k
                   and _cfg.stages == GEMM_DEFAULT.stages)

    _global_registry.register(
        OpKey("gemm", "group_gemm"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_group_gemm_runner(_cfg),
            is_fallback=_is_default,
        ),
    )
