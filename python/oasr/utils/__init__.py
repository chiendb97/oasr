"""
OASR Utilities Module

Helper functions for profiling, C extension backend, and dtype mapping.

The backend utilities (`get_dtype`, `get_activation_type`, etc.) are imported
lazily to avoid circular imports between `oasr`, `oasr.layers`, and
`oasr.utils.backend`.
"""

from .timer import Timer


def get_dtype(dtype):
    from .backend import get_dtype as _get_dtype

    return _get_dtype(dtype)


def get_activation_type(activation_type):
    from .backend import get_activation_type as _get_activation_type

    return _get_activation_type(activation_type)


def get_activation(activation_type):
    from .backend import get_activation as _get_activation

    return _get_activation(activation_type)


def get_norm_type(norm_type):
    from .backend import get_norm_type as _get_norm_type

    return _get_norm_type(norm_type)


def get_norm(norm_type):
    from .backend import get_norm as _get_norm

    return _get_norm(norm_type)


__all__ = [
    "Timer",
    "get_dtype",
    "get_activation",
    "get_norm_type",
    "get_activation_type",
    "get_norm",
]
