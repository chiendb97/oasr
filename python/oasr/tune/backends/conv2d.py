# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conv2D backend registrations with tile-variant autotuning.

Registers CUTLASS Conv2D tile variants from ``CONV2D_TILE_CONFIGS`` plus
cuDNN backends.  Each CUTLASS tile variant is JIT-compiled into a separate
shared library; the autotuner profiles and selects the best one at runtime.
"""

import functools
import logging

from oasr.tune._registry import BackendEntry, _global_registry
from oasr.tune._types import OpKey, Tactic
from oasr.tune.kernel_configs import (
    CONV2D_DEFAULT,
    CONV2D_TILE_CONFIGS,
    TileConfig,
)

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
# CUTLASS tile-variant module cache
# ---------------------------------------------------------------------------

_conv2d_modules = {}


def _get_conv2d_variant_module(cfg: TileConfig):
    """Get or compile a Conv2D module for a specific tile config."""
    key = cfg.name
    if key not in _conv2d_modules:
        from oasr.jit.conv import gen_conv2d_module_variant

        spec = gen_conv2d_module_variant(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages,
        )
        _conv2d_modules[key] = spec.build_and_load()
    return _conv2d_modules[key]


# ---------------------------------------------------------------------------
# CUTLASS tile-variant runner factories
# ---------------------------------------------------------------------------

def _make_conv2d_runner(cfg: TileConfig):
    """Create a CUTLASS Conv2D runner for a specific tile config."""
    def runner():
        mod = _get_conv2d_variant_module(cfg)
        return mod.conv2d

    return runner


def _make_conv2d_activation_runner(cfg: TileConfig):
    """Create a CUTLASS Conv2D+activation runner for a specific tile config."""
    def runner():
        mod = _get_conv2d_variant_module(cfg)
        return mod.conv2d_activation

    return runner


# ---------------------------------------------------------------------------
# Registration: CUTLASS tile variants for conv2d and conv2d_activation
# ---------------------------------------------------------------------------

for _cfg in CONV2D_TILE_CONFIGS:
    _tactic = Tactic("cutlass", config=_cfg.to_tactic_config())
    _is_default = (_cfg.tile_m == CONV2D_DEFAULT.tile_m
                   and _cfg.tile_n == CONV2D_DEFAULT.tile_n
                   and _cfg.tile_k == CONV2D_DEFAULT.tile_k
                   and _cfg.stages == CONV2D_DEFAULT.stages)

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
