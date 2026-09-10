# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Builders for a real engine and for the stubs that stand in for one.

Ten modules built an :class:`~oasr.engine.ASREngine` from ``ckpt_dir`` with
their own defaults, and the defaults disagreed -- ``max_batch_size``,
``service_mode`` and ``decoder_type`` each differed between files testing the
same thing.  That is worse than duplication: two tests claiming to measure the
same engine were measuring two.

Imports of ``oasr.engine`` stay inside the functions.  Importing it at module
scope pulls the whole engine in at *collection* time, which is why several of
these helpers were written deferred in the first place.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["engine_config", "make_engine"]


def engine_config(ckpt_dir: str, **overrides: Any):
    """An ``EngineConfig`` on the settings the suite's real-checkpoint tests share.

    fp16 on CUDA, offline, small batch: the working point that keeps a
    real-checkpoint test to seconds. Override anything per test; the point of a
    shared base is that an override is *visible* as the thing under test.
    """
    from oasr.engine import EngineConfig

    kwargs: dict[str, Any] = {
        "ckpt_dir": ckpt_dir,
        "device": "cuda",
        "dtype": torch.float16,
        "service_mode": "offline",
        "max_batch_size": 8,
    }
    kwargs.update(overrides)
    return EngineConfig(**kwargs)


def make_engine(ckpt_dir: str, **overrides: Any):
    """:func:`engine_config` plus construction."""
    from oasr.engine import ASREngine

    return ASREngine(engine_config(ckpt_dir, **overrides))
