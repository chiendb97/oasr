"""Shared utilities for OASR benchmark scripts.

This module delegates to benchmarks.routines.bench_utils for the actual
implementation, preserving the original API for backwards compatibility.
"""

from benchmarks.routines.bench_utils import (  # noqa: F401
    make_bench_parser,
    profile_kernel,
    run_main,
    run_profile,
)
