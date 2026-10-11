# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark kinds.  A profile's ``kind:`` names one of these."""

from __future__ import annotations

from typing import Dict, Type

from benchmarks.asrbench.runners.base import Benchmark, RunContext
from benchmarks.asrbench.runners.load import Load
from benchmarks.asrbench.runners.offline import Batch, ColdStart, LongForm, Single, Throughput
from benchmarks.asrbench.runners.streaming import Streaming

KINDS: Dict[str, Type[Benchmark]] = {
    cls.kind: cls for cls in (ColdStart, Single, Batch, Throughput, Load, LongForm, Streaming)
}

__all__ = ["KINDS", "Benchmark", "RunContext", "get_kind"]


def get_kind(kind: str) -> Type[Benchmark]:
    try:
        return KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown benchmark kind {kind!r}; known: {sorted(KINDS)}") from None
