# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Runners -- the backends a tuning region chooses among.  See :mod:`.base`."""

from oasr.tune.runners.base import (
    Runner,
    RunnerChoice,
    RunnerTraits,
    get_runner,
    kind_of,
    register_backend_entries,
    register_runner,
    runner_for_backend,
    runner_for_kind,
    runner_name_of,
    runner_of,
    runner_versions,
    runners,
    unregister_runner,
    unservable,
)

__all__ = [
    "Runner",
    "RunnerChoice",
    "RunnerTraits",
    "get_runner",
    "kind_of",
    "register_backend_entries",
    "register_runner",
    "runner_for_backend",
    "runner_for_kind",
    "runner_name_of",
    "runner_of",
    "runner_versions",
    "runners",
    "unregister_runner",
    "unservable",
]
