# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""API logging decorator for OASR public functions."""

import functools
import logging

logger = logging.getLogger("oasr")


def oasr_api(func):
    """Decorator that marks a function as a public OASR API entry point.

    Adds debug-level logging of calls (function name, args shape/dtype)
    and wraps exceptions with context.

    Usage::

        @oasr_api
        def my_kernel(input, output, factor):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            arg_summaries = []
            for a in args:
                if hasattr(a, "shape") and hasattr(a, "dtype"):
                    arg_summaries.append(f"Tensor({a.shape}, {a.dtype})")
                else:
                    arg_summaries.append(repr(a))
            logger.debug(
                "oasr.%s(%s)", func.__name__, ", ".join(arg_summaries)
            )
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.debug("oasr.%s raised an exception", func.__name__, exc_info=True)
            raise

    return wrapper
