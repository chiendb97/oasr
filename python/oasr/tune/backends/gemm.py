# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GEMM backend registrations with tile-variant autotuning.

Registers the default CUTLASS GEMM configuration as the fallback, plus
additional tile variants that the autotuner can profile and select from.
Each tile variant is JIT-compiled into a separate shared library.

For split-K, the same compiled module is reused with different runtime
``split_k_slices`` values — no recompilation needed.
"""

import functools
import logging

from oasr.tune._registry import BackendEntry, _global_registry
from oasr.tune._types import OpKey, Tactic
from oasr.tune.kernel_configs import (
    GEMM_DEFAULT,
    GEMM_TILE_CONFIGS,
    BMM_TILE_CONFIGS,
    GROUP_GEMM_TILE_CONFIGS,
    TileConfig,
    get_unique_compile_configs,
)

logger = logging.getLogger("oasr.tune")


# ---------------------------------------------------------------------------
# Module cache — each (family, tile_config) maps to a lazily-compiled module
# ---------------------------------------------------------------------------

_gemm_modules = {}
_bmm_modules = {}
_group_gemm_modules = {}


def _get_gemm_variant_module(cfg: TileConfig):
    """Get or compile a GEMM module for a specific tile config."""
    key = cfg.name
    if key not in _gemm_modules:
        from oasr.jit.gemm import gen_gemm_module_variant

        spec = gen_gemm_module_variant(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages,
        )
        _gemm_modules[key] = spec.build_and_load()
    return _gemm_modules[key]


def _get_bmm_variant_module(cfg: TileConfig):
    key = cfg.name
    if key not in _bmm_modules:
        from oasr.jit.gemm import gen_bmm_module_variant

        spec = gen_bmm_module_variant(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages,
        )
        _bmm_modules[key] = spec.build_and_load()
    return _bmm_modules[key]


def _get_group_gemm_variant_module(cfg: TileConfig):
    key = cfg.name
    if key not in _group_gemm_modules:
        from oasr.jit.gemm import gen_group_gemm_module_variant

        spec = gen_group_gemm_module_variant(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages,
        )
        _group_gemm_modules[key] = spec.build_and_load()
    return _group_gemm_modules[key]


# ---------------------------------------------------------------------------
# GEMM registration
# ---------------------------------------------------------------------------

def _make_gemm_runner(cfg: TileConfig):
    """Create a GEMM runner that calls the variant module with split_k."""
    split_k = cfg.split_k
    # The compile config is the same cfg but with split_k=1
    compile_cfg = TileConfig(
        tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
        warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
        stages=cfg.stages, split_k=1,
    )

    def runner():
        mod = _get_gemm_variant_module(compile_cfg)

        def call(out, A, B, C, split_k_slices=split_k):
            mod.gemm(out, A, B, C, split_k_slices)

        return call

    return runner


def _make_gemm_activation_runner(cfg: TileConfig):
    split_k = cfg.split_k
    compile_cfg = TileConfig(
        tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
        warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
        stages=cfg.stages, split_k=1,
    )

    def runner():
        mod = _get_gemm_variant_module(compile_cfg)

        def call(out, A, B, C, activation_type, split_k_slices=split_k):
            mod.gemm_activation(out, A, B, C, activation_type, split_k_slices)

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
    def runner():
        mod = _get_bmm_variant_module(cfg)
        return mod.bmm

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
    def runner():
        mod = _get_group_gemm_variant_module(cfg)
        return mod.group_gemm

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
