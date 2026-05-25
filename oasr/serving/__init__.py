# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Serving layer for OASR.

Contains the Python-side engine worker that the Rust frontend talks to over
ZeroMQ.  Most users do not import from this package directly — launch the
``oasr-server`` console script (a shim around the Rust binary) or run
``python -m oasr.serving`` to spawn a standalone worker.
"""

from .engine_worker import EngineWorker, WorkerOptions

__all__ = ["EngineWorker", "WorkerOptions"]
