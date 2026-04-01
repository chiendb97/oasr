# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Validation decorators for compute capability and backend requirements.

Provides ``@supported_compute_capability`` and ``@backend_requirement``
decorators used by OASR functional API functions.
"""

import functools
from typing import Callable, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _get_device_cc(device: Optional[torch.device] = None) -> int:
    """Return the compute capability as an integer (e.g. 90 for SM9.0)."""
    if device is None:
        device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def is_sm90a_supported(device: Optional[torch.device] = None) -> bool:
    """Check if the device supports SM90a (Hopper) or later.

    Parameters
    ----------
    device : torch.device, optional
        CUDA device to check. Defaults to current device.
    """
    if not torch.cuda.is_available():
        return False
    return _get_device_cc(device) >= 90


# ---------------------------------------------------------------------------
# @supported_compute_capability
# ---------------------------------------------------------------------------


def supported_compute_capability(cc_list: List[int]):
    """Decorator that marks a function with its supported CUDA compute capabilities.

    The decorated function gains a ``supported_ccs`` attribute containing
    the list of supported compute capabilities.

    Parameters
    ----------
    cc_list : list of int
        Supported compute capabilities (e.g. ``[80, 86, 89, 90, 100]``).

    Example
    -------
    ::

        @supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120])
        def _check_my_kernel(input, output):
            if input.shape[-1] > 256:
                raise ValueError("Head dim must be <= 256")
            return True
    """

    def decorator(func):
        func.supported_ccs = cc_list

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.supported_ccs = cc_list
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# @backend_requirement
# ---------------------------------------------------------------------------


def backend_requirement(
    backend_checks: Dict[str, Callable],
    common_check: Optional[Callable] = None,
    heuristic_func: Optional[Callable] = None,
):
    """Decorator enforcing compute-capability and backend requirements at runtime.

    Supports three patterns:

    1. **No backend choices** (``backend_checks={}``): only ``common_check``
       is run. The decorated function has no ``backend`` parameter.

    2. **Multiple backends** (``backend_checks={"cutlass": ..., "cudnn": ...}``):
       the check function matching the ``backend`` kwarg is run.

    3. **Auto backend selection**: when ``backend="auto"`` is passed and
       ``heuristic_func`` is provided, suitable backends are ranked and
       stored in ``func.suitable_auto_backends``.

    Parameters
    ----------
    backend_checks : dict[str, callable]
        Mapping from backend name to its check function.  Each check
        function must accept the same args as the decorated function and
        return ``True`` on success or raise ``ValueError``.
    common_check : callable, optional
        Validation function run for **all** backends (before the
        backend-specific check).
    heuristic_func : callable, optional
        ``heuristic_func(suitable_backends, *args, **kwargs)`` returns a
        list of backends sorted by preference.  Required when
        ``backend="auto"`` is used.

    Added attributes / methods on the decorated function
    ----------------------------------------------------
    ``is_backend_supported(name, cc=None)``
        Check if *name* is a registered backend (optionally for a given CC).
    ``is_compute_capability_supported(cc)``
        True if any backend supports this CC.
    ``has_backend(name)``
        True if *name* exists in ``backend_checks``.
    ``has_backend_choices()``
        True if there are multiple backend choices.
    ``suitable_auto_backends``
        List set by the heuristic when ``backend="auto"``.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, skip_check: bool = False, **kwargs):
            if skip_check:
                return func(*args, **kwargs)

            backend = kwargs.get("backend")

            # Run common check
            if common_check is not None:
                common_check(*args, **kwargs)

            if backend_checks:
                if backend == "auto" and heuristic_func is not None:
                    # Determine suitable backends
                    cc = _get_device_cc()
                    suitable = []
                    for name, check_fn in backend_checks.items():
                        ccs = getattr(check_fn, "supported_ccs", None)
                        if ccs is not None and cc not in ccs:
                            continue
                        try:
                            check_fn(*args, **kwargs)
                            suitable.append(name)
                        except (ValueError, RuntimeError):
                            continue
                    ranked = heuristic_func(suitable, *args, **kwargs)
                    if not ranked:
                        raise RuntimeError(
                            f"No suitable backend found for the given inputs "
                            f"on SM{cc}."
                        )
                    wrapper.suitable_auto_backends = ranked
                elif backend is not None and backend in backend_checks:
                    check_fn = backend_checks[backend]
                    # CC check
                    cc = _get_device_cc()
                    ccs = getattr(check_fn, "supported_ccs", None)
                    if ccs is not None and cc not in ccs:
                        raise RuntimeError(
                            f"Backend '{backend}' does not support compute "
                            f"capability SM{cc}. Supported: {ccs}"
                        )
                    check_fn(*args, **kwargs)
                elif backend is not None and backend not in backend_checks:
                    raise ValueError(
                        f"Unknown backend '{backend}'. "
                        f"Available: {list(backend_checks.keys())}"
                    )

            return func(*args, **kwargs)

        # --- Helper methods ---

        def is_backend_supported(name: str, cc: Optional[int] = None) -> bool:
            if name not in backend_checks:
                return False
            if cc is not None:
                ccs = getattr(backend_checks[name], "supported_ccs", None)
                if ccs is not None:
                    return cc in ccs
            return True

        def is_compute_capability_supported(cc: int) -> bool:
            if not backend_checks:
                # No backend choices — check common_check
                if common_check is not None:
                    ccs = getattr(common_check, "supported_ccs", None)
                    if ccs is not None:
                        return cc in ccs
                return True
            for check_fn in backend_checks.values():
                ccs = getattr(check_fn, "supported_ccs", None)
                if ccs is None or cc in ccs:
                    return True
            return False

        def has_backend(name: str) -> bool:
            return name in backend_checks

        def has_backend_choices() -> bool:
            return len(backend_checks) > 1

        wrapper.is_backend_supported = is_backend_supported
        wrapper.is_compute_capability_supported = is_compute_capability_supported
        wrapper.has_backend = has_backend
        wrapper.has_backend_choices = has_backend_choices
        wrapper.suitable_auto_backends = []

        return wrapper

    return decorator
