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

from oasr.jit.core import _get_target_sm
from oasr.jit.gemm import (
    GEMM_DEFAULT,
    CutlassGemmConfig,
    CutlassGemmConfigSm90,
    bmm_func_name,
    gemm_activation_func_name,
    gemm_func_name,
    get_all_autotune_configs,
    get_unique_compile_configs,
    group_gemm_func_name,
)
from oasr.tune.autotuner import BackendEntry, OpKey, Tactic, _global_registry

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
    _is_default = (
        _cfg.compile_name == _GEMM_DEFAULT.compile_name and getattr(_cfg, "split_k", 1) == 1
    )

    _global_registry.register(
        OpKey("gemm", "gemm"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_gemm_runner(_cfg),
            is_fallback=_is_default,
        ),
    )

    # Serial split-K applies the epilogue per K-partition, which would nest
    # the activation around partial sums — the kernel rejects it, so those
    # tactics would only pollute profiling with guaranteed failures.  The
    # parallel split-K variants apply the activation once post-reduction and
    # remain valid candidates.
    if getattr(_cfg, "split_k", 1) > 1 and not getattr(_cfg, "parallel_split_k", False):
        continue

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
    # Stream-K / parallel split-K are GEMM-only template paths; bmm has
    # neither, and its launcher takes no runtime split-K argument.
    if getattr(_cfg, "stream_k", False) or getattr(_cfg, "parallel_split_k", False):
        continue
    if getattr(_cfg, "split_k", 1) != 1:
        continue
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (
        _cfg.compile_name == _GEMM_DEFAULT.compile_name and getattr(_cfg, "split_k", 1) == 1
    )

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
    # Stream-K / parallel split-K are GEMM-only template paths; group_gemm has
    # neither, and its launcher takes no runtime split-K argument.
    if getattr(_cfg, "stream_k", False) or getattr(_cfg, "parallel_split_k", False):
        continue
    if getattr(_cfg, "split_k", 1) != 1:
        continue
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (
        _cfg.compile_name == _GEMM_DEFAULT.compile_name and getattr(_cfg, "split_k", 1) == 1
    )

    _global_registry.register(
        OpKey("gemm", "group_gemm"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_group_gemm_runner(_cfg),
            is_fallback=_is_default,
        ),
    )


# ---------------------------------------------------------------------------
# Fused GEMM + log_softmax registration (the CTC head)
# ---------------------------------------------------------------------------
#
# Three backend flavours compete per shape:
#   * ``Tactic("cutlass", config=...)`` — the corresponding GEMM variant
#     composed with the OASR online log_softmax kernel in-place on the output
#     (identical numerics to the fused launcher, shape-selectable tile);
#   * ``Tactic("cutlass_fused")``       — the legacy single-call launcher
#     (fixed 16x128 tile); registered as the fallback so untuned behaviour
#     matches the historical path;
#   * ``Tactic("torch")``               — registered in ``torch_gemm.py``.


@functools.cache
def _get_gemm_log_softmax_module():
    from oasr.jit.gemm import gen_gemm_log_softmax_module

    return gen_gemm_log_softmax_module().build_and_load()


@functools.cache
def _get_softmax_module():
    from oasr.jit.softmax import gen_softmax_module

    return gen_softmax_module().build_and_load()


def _make_gemm_log_softmax_runner(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]):
    fn_name = gemm_func_name(cfg)
    split_k = getattr(cfg, "split_k", 1)

    def runner():
        gemm_fn = getattr(_get_gemm_module(), fn_name)
        log_softmax_fn = _get_softmax_module().log_softmax

        def call(out, A, B, C, split_k_slices=split_k):
            gemm_fn(out, A, B, C, split_k_slices)
            log_softmax_fn(out, out)

        return call

    return runner


def _fused_gemm_log_softmax_runner():
    fn = _get_gemm_log_softmax_module().gemm_log_softmax

    def call(out, A, B, C, split_k_slices=1):
        fn(out, A, B, C, split_k_slices)

    return call


for _cfg in _all_autotune_configs.values():
    _global_registry.register(
        OpKey("gemm", "gemm_log_softmax"),
        BackendEntry(
            tactic=Tactic("cutlass", config=_cfg.to_tactic_config()),
            is_available=lambda: True,
            get_runner=_make_gemm_log_softmax_runner(_cfg),
            is_fallback=False,
        ),
    )

_global_registry.register(
    OpKey("gemm", "gemm_log_softmax"),
    BackendEntry(
        tactic=Tactic("cutlass_fused"),
        is_available=lambda: True,
        get_runner=_fused_gemm_log_softmax_runner,
        is_fallback=True,
    ),
)
