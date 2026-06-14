# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GEMM backend registrations for autotuning (FlashInfer-style).

All tile variants are pre-compiled into a single shared library by the JIT
layer (``oasr.jit.gemm``).  The autotuner selects which pre-compiled variant
to call — no JIT compilation is triggered during tuning.

For SM<90, each ``CutlassGemmConfig`` carries a ``split_k`` field.  Variants
that differ only in ``split_k`` share the same compiled binary; the runner
passes the per-config ``split_k`` value to the kernel at call time.
"""

import functools
import logging
from typing import Union

from oasr.tune.autotuner import BackendEntry, _global_registry, OpKey, Tactic
from oasr.jit.gemm import (
    CutlassGemmConfig,
    CutlassGemmConfigSm90,
    GEMM_DEFAULT,
    get_all_autotune_configs,
    get_unique_compile_configs,
    gemm_func_name,
    gemm_activation_func_name,
    bmm_func_name,
    group_gemm_func_name,
)
from oasr.jit.core import _get_target_sm

logger = logging.getLogger("oasr.tune")

_sm = _get_target_sm()

# Full set — includes split_k variants for SM<90; used for tactic registration.
_all_autotune_configs = get_all_autotune_configs(_sm)

# Compile set — deduplicated by compile_name; used to determine the default.
_all_compile_configs = get_unique_compile_configs(_sm)

# Use the canonical default from oasr.jit.gemm so the fallback tactic matches the
# non-tuning production path for every arch (notably SM120, which uses the
# CUTLASS 2.x / SM<90 path — re-deriving the default here previously mis-picked
# the SM90 config and left no candidate flagged as the fallback).
_GEMM_DEFAULT = GEMM_DEFAULT


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
    """Create a GEMM runner that calls a specific variant from the module.

    For SM<90 configs the runner passes ``cfg.split_k`` as the default
    ``split_k_slices`` value so the autotuner exercises each split-K factor.
    """
    fn_name = gemm_func_name(cfg)
    split_k = getattr(cfg, "split_k", 1)

    def runner():
        mod = _get_gemm_module()
        fn = getattr(mod, fn_name)

        def call(out, A, B, C, split_k_slices=split_k):
            fn(out, A, B, C, split_k_slices)

        return call

    return runner


def _make_gemm_activation_runner(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]):
    fn_name = gemm_activation_func_name(cfg)
    split_k = getattr(cfg, "split_k", 1)

    def runner():
        mod = _get_gemm_module()
        fn = getattr(mod, fn_name)

        def call(out, A, B, C, activation_type, split_k_slices=split_k):
            fn(out, A, B, C, activation_type, split_k_slices)

        return call

    return runner


for _cfg in _all_autotune_configs.values():
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.compile_name == _GEMM_DEFAULT.compile_name
                   and getattr(_cfg, "split_k", 1) == 1)

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


for _cfg in _all_autotune_configs.values():
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.compile_name == _GEMM_DEFAULT.compile_name
                   and getattr(_cfg, "split_k", 1) == 1)

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


for _cfg in _all_autotune_configs.values():
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.compile_name == _GEMM_DEFAULT.compile_name
                   and getattr(_cfg, "split_k", 1) == 1)

    _global_registry.register(
        OpKey("gemm", "group_gemm"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_group_gemm_runner(_cfg),
            is_fallback=_is_default,
        ),
    )
