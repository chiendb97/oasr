# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conv2D backend registrations for autotuning (FlashInfer-style).

All CUTLASS tile variants are pre-compiled into a single shared library by
the JIT layer (``oasr.jit.conv``).  The autotuner selects which pre-compiled
variant to call — no JIT compilation is triggered during tuning.

cuDNN backends are also registered as additional candidates.
"""

import functools
import logging
from typing import Union

from oasr.tune.autotuner import BackendEntry, _global_registry, OpKey, Tactic
from oasr.jit.conv import (
    CutlassConv2dConfig,
    CutlassConv2dConfigSm90,
    CONV2D_DEFAULT,
    CONV2D_DEFAULT_SM90,
    CONV2D_TILE_CONFIGS,
    CONV2D_TILE_CONFIGS_SM90,
    CONV2D_CLUSTER_CONFIGS_SM90,
    conv2d_func_name,
    conv2d_activation_func_name,
    _get_unique_conv2d_compile_configs,
)
from oasr.jit.core import _get_target_sm

logger = logging.getLogger("oasr.tune")


# ---------------------------------------------------------------------------
# cuDNN module loaders
# ---------------------------------------------------------------------------

@functools.cache
def _get_cudnn_conv2d_module():
    from oasr.jit.conv import gen_cudnn_conv2d_module

    return gen_cudnn_conv2d_module().build_and_load()


def _has_cudnn() -> bool:
    """Check if cuDNN is available at runtime."""
    try:
        _get_cudnn_conv2d_module()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pre-compiled CUTLASS module (contains ALL variants)
# ---------------------------------------------------------------------------

@functools.cache
def _get_conv2d_module():
    from oasr.jit.conv import gen_conv2d_module
    return gen_conv2d_module().build_and_load()


# ---------------------------------------------------------------------------
# CUTLASS tile-variant runner factories
# ---------------------------------------------------------------------------

def _make_conv2d_runner(cfg: Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]):
    """Create a CUTLASS Conv2D runner for a specific variant."""
    fn_name = conv2d_func_name(cfg)

    def runner():
        mod = _get_conv2d_module()
        return getattr(mod, fn_name)

    return runner


def _make_conv2d_activation_runner(cfg: Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]):
    """Create a CUTLASS Conv2D+activation runner for a specific variant."""
    fn_name = conv2d_activation_func_name(cfg)

    def runner():
        mod = _get_conv2d_module()
        return getattr(mod, fn_name)

    return runner


# ---------------------------------------------------------------------------
# Registration: CUTLASS tile variants for conv2d and conv2d_activation
# ---------------------------------------------------------------------------

_sm = _get_target_sm()
_all_configs = _get_unique_conv2d_compile_configs(_sm)
_default_cfg = CONV2D_DEFAULT_SM90 if _sm >= 90 else CONV2D_DEFAULT

for _cfg in _all_configs.values():
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.compile_name == _default_cfg.compile_name)

    _global_registry.register(
        OpKey("conv", "conv2d"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_conv2d_runner(_cfg),
            is_fallback=_is_default,
        ),
    )

    _global_registry.register(
        OpKey("conv", "conv2d_activation"),
        BackendEntry(
            tactic=_tactic,
            is_available=lambda: True,
            get_runner=_make_conv2d_activation_runner(_cfg),
            is_fallback=_is_default,
        ),
    )


# ---------------------------------------------------------------------------
# Registration: cuDNN backends
# ---------------------------------------------------------------------------

def _cudnn_conv2d_runner():
    mod = _get_cudnn_conv2d_module()
    return mod.cudnn_conv2d


def _cudnn_conv2d_activation_runner():
    mod = _get_cudnn_conv2d_module()
    return mod.cudnn_conv2d_activation


_global_registry.register(
    OpKey("conv", "conv2d"),
    BackendEntry(
        tactic=Tactic("cudnn"),
        is_available=_has_cudnn,
        get_runner=_cudnn_conv2d_runner,
        is_fallback=False,
    ),
)

_global_registry.register(
    OpKey("conv", "conv2d_activation"),
    BackendEntry(
        tactic=Tactic("cudnn"),
        is_available=_has_cudnn,
        get_runner=_cudnn_conv2d_activation_runner,
        is_fallback=False,
    ),
)
