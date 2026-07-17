# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-tick decoder-step budget + hypothesis bookkeeping for AR strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepBudget:
    """Decoder-step allowance for one engine tick.

    One *step* is one batched decoder forward over the currently-active
    request set (continuous batching), not one step per request — a tick's
    worst-case work is ``max_steps`` decoder forwards regardless of how many
    requests are pending.  Strategies call :meth:`take` before each batched
    step and stop when it returns ``False``; the engine sizes ``max_steps``
    from ``EngineConfig.decode_steps_per_tick``.
    """

    max_steps: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.used)

    def exhausted(self) -> bool:
        return self.used >= self.max_steps

    def take(self) -> bool:
        """Consume one batched decoder step; ``False`` when the tick is spent."""
        if self.exhausted():
            return False
        self.used += 1
        return True


@dataclass
class Hypothesis:
    """One request's in-flight AR hypothesis (greedy: a single growing row).

    Beam strategies keep one :class:`Hypothesis` per beam entry.  ``score``
    accumulates token log-probs; ``finished`` flips on EOS or on hitting the
    request's ``max_new_tokens``.
    """

    tokens: List[int] = field(default_factory=list)
    score: float = 0.0
    finished: bool = False
    finish_reason: Optional[str] = None  # "eos" | "length"
