# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Pluggable batching + partition policies for the ASR scheduler.

Importing this package registers the built-in policies (fcfs / bucket / sjf and
count / frames / packing) so the builders resolve them.  Add a new policy by
subclassing :class:`BatchingPolicy` / :class:`PartitionPolicy` and decorating it
with :func:`register_batching_policy` / :func:`register_partition_policy`.
"""

# Import for side effects: each module registers its policies on import.
from . import partition, policies  # noqa: E402,F401
from .base import (
    BatchingPolicy,
    PartitionPolicy,
    build_batching_policy,
    build_partition_policy,
    register_batching_policy,
    register_partition_policy,
    snap_to_preferred,
    sort_by_length,
)

__all__ = [
    "BatchingPolicy",
    "PartitionPolicy",
    "build_batching_policy",
    "build_partition_policy",
    "register_batching_policy",
    "register_partition_policy",
    "snap_to_preferred",
    "sort_by_length",
]
