# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded autoregressive generation primitives (multi-paradigm keystone K2).

Label-synchronous decode families (AED / LLM) cannot loop to EOS inside one
engine ``step()`` — the GIL-owning serving dispatcher runs
``drain cmds → step() → extract`` synchronously per tick, so an unbounded loop
starves every other stream.  Instead, an ``incremental`` decode strategy runs
at most :class:`StepBudget` decoder steps per tick with continuous batching
across all pending requests (see ``DecodeStrategy.begin_offline`` /
``advance`` / ``has_pending`` in ``oasr/engine/decode/base.py``).
"""

from .budget import Hypothesis, StepBudget

__all__ = ["Hypothesis", "StepBudget"]
