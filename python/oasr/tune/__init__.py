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

import contextlib
import logging
import warnings
from pathlib import Path
from typing import Optional

from ._cache import TuneCache
from ._profiler import Profiler
from ._registry import _global_registry
from ._tuner import AutoTuner
from ._types import OpKey, ProfileKey, Tactic, TuneResult

__all__ = [
    "autotune",
    "enable_autotune",
    "disable_autotune",
    "is_tuning_enabled",
    "get_tuner",
    "load_configs",
    "save_configs",
    "clear_cache",
    "get_selected_config",
    "OpKey",
    "ProfileKey",
    "Tactic",
    "TuneResult",
]

# -------------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------------

_tuner: Optional[AutoTuner] = None
_enabled: bool = False
_active_cache_path: Optional[Path] = None

logger = logging.getLogger("oasr.tune")


def _ensure_backends_registered() -> None:
    """Import backend modules to trigger registration."""
    from .backends import conv2d as _conv2d_backends  # noqa: F401
    from .backends import gemm as _gemm_backends  # noqa: F401


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def is_tuning_enabled() -> bool:
    """Return ``True`` if autotuning dispatch is active.

    This is a fast module-level bool check used by functional API functions
    to decide whether to route through the autotuner.
    """
    return _enabled


def get_tuner() -> AutoTuner:
    """Return the global ``AutoTuner`` instance, creating it lazily."""
    global _tuner
    if _tuner is None:
        _ensure_backends_registered()
        _tuner = AutoTuner(
            registry=_global_registry,
            cache=TuneCache(),
            profiler=Profiler(),
            tune_mode=False,
        )
    return _tuner


@contextlib.contextmanager
def autotune(
    enabled: bool = True,
    *,
    cache: Optional[str] = None,
    warmup: int = 25,
    rep: int = 100,
    log_level: str = "INFO",
    **kwargs,
):
    """Context manager to enable and configure autotuning.

    Args:
        enabled: If ``True``, profile missing configs on first encounter.
                 If ``False``, only use cached configs or fallback.
        cache: Path to a JSON cache file.  If the file exists, its entries
               are loaded.  On context exit, newly profiled entries are saved
               back (merged with the existing file).
        warmup: Number of warmup iterations for profiling.
        rep: Number of measurement iterations for profiling.
        log_level: Logging verbosity for ``oasr.tune`` logger.

    Example::

        import oasr

        with oasr.autotune(cache="cache.json"):
            oasr.gemm(A, B)
    """
    # Backward compat: accept tune_mode= as deprecated alias for enabled=
    if "tune_mode" in kwargs:
        warnings.warn(
            "autotune(tune_mode=...) is deprecated, use autotune(enabled=...) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        enabled = kwargs.pop("tune_mode")
    if kwargs:
        raise TypeError(f"autotune() got unexpected keyword arguments: {list(kwargs)}")

    global _enabled

    tuner = get_tuner()
    prev_tune_mode = tuner.tune_mode
    prev_enabled = _enabled
    prev_log_level = logger.level

    # Configure
    tuner.tune_mode = enabled
    tuner._profiler.warmup = warmup
    tuner._profiler.rep = rep
    _enabled = True
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Load cache if provided
    cache_path = Path(cache) if cache else None
    if cache_path is not None:
        tuner.cache.load(cache_path)

    try:
        yield tuner
    finally:
        # Save cache if provided
        if cache_path is not None:
            try:
                tuner.cache.save(cache_path)
            except OSError as exc:
                logger.warning("Failed to save tune cache: %s", exc)

        # Restore
        tuner.tune_mode = prev_tune_mode
        _enabled = prev_enabled
        logger.setLevel(prev_log_level)


def enable_autotune(
    cache: Optional[str] = None,
    *,
    warmup: int = 25,
    rep: int = 100,
    log_level: str = "INFO",
) -> None:
    """Enable autotuning globally.

    Unlike :func:`autotune`, this is not a context manager — autotuning
    stays active until :func:`disable_autotune` is called.  Good for
    scripts that want to enable once at startup.

    Args:
        cache: Optional JSON cache file.  Loaded immediately; saved when
               :func:`disable_autotune` is called with ``save_cache=True``.
        warmup: Number of warmup iterations for profiling.
        rep: Number of measurement iterations for profiling.
        log_level: Logging verbosity for ``oasr.tune`` logger.

    Example::

        import oasr

        oasr.enable_autotune(cache="tune.json")
        output = oasr.gemm(A, B)   # auto-profiles on first call
        oasr.disable_autotune()    # saves cache and disables
    """
    global _enabled, _active_cache_path

    tuner = get_tuner()
    tuner.tune_mode = True
    tuner._profiler.warmup = warmup
    tuner._profiler.rep = rep
    _enabled = True
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    _active_cache_path = Path(cache) if cache else None
    if _active_cache_path is not None:
        tuner.cache.load(_active_cache_path)


def disable_autotune(save_cache: bool = True) -> None:
    """Disable autotuning globally.

    Args:
        save_cache: If ``True`` and a cache path was provided to
                    :func:`enable_autotune`, save profiled configs to disk.
    """
    global _enabled, _active_cache_path

    if save_cache and _active_cache_path is not None:
        try:
            get_tuner().cache.save(_active_cache_path)
        except OSError as exc:
            logger.warning("Failed to save tune cache: %s", exc)

    _enabled = False
    _active_cache_path = None


def load_configs(path: str) -> None:
    """Load tuned configs from a JSON file into the global cache."""
    get_tuner().cache.load(Path(path))


def save_configs(path: str) -> None:
    """Save all cached configs to a JSON file."""
    get_tuner().cache.save(Path(path))


def clear_cache() -> None:
    """Clear all in-memory cached tuning results."""
    get_tuner().cache.clear()


def get_selected_config(
    op_key: OpKey,
    shape_sig: tuple,
    dtype: str,
    device_sm: int,
) -> Optional[Tactic]:
    """Look up the cached tactic for a specific profile, if any."""
    key = ProfileKey(op_key=op_key, shape_sig=shape_sig, dtype=dtype, device_sm=device_sm)
    return get_tuner().cache.lookup(key)
