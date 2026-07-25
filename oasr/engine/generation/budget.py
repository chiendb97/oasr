# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-tick decoder-step budget + hypothesis bookkeeping for AR strategies."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepBudget:
    """Decoder-step allowance for one engine tick — a step cap *and* a deadline.

    One *step* is one batched decoder forward over the currently-active
    request set (continuous batching), not one step per request.  Strategies
    call :meth:`take` before each batched step and stop when it returns
    ``False``.

    Two limits, because a step count alone does not bound tick *time*: the cost
    of one decoder step spans two orders of magnitude across models (~1.5 ms for
    whisper-tiny at ``B=8``, ~18 ms for Qwen2-Audio-7B at ``B=4``, both measured).
    A fixed ``max_steps=32`` therefore means a ~50 ms tick on one model and a
    ~580 ms tick on another — and the serving dispatcher holds the GIL for a whole
    tick, so that number is the floor on cancel latency, admission latency, and
    the interval between streaming partials.

    * ``max_steps`` (``EngineConfig.decode_steps_per_tick``) caps the work.
    * ``deadline_s`` (``EngineConfig.max_tick_ms``) caps the wall-clock time,
      which is what a client actually feels.

    The deadline stops *starting* new steps; a step already in flight is never
    preempted, so the real bound is ``deadline + one step``.  Whichever limit
    binds first wins, so light models still batch many steps per tick (efficient)
    while heavy models emit tokens at an interactive cadence.
    """

    max_steps: int
    #: ``time.monotonic()`` value past which no further step is started.
    #: ``None`` disables the time limit (step cap only — the historical behaviour).
    deadline_s: Optional[float] = None
    used: int = 0

    @classmethod
    def for_tick(cls, max_steps: int, max_tick_ms: float = 0.0) -> "StepBudget":
        """Build a budget for a tick starting now.  ``max_tick_ms <= 0`` = no deadline."""
        deadline = time.monotonic() + (max_tick_ms / 1000.0) if max_tick_ms > 0 else None
        return cls(max_steps=max_steps, deadline_s=deadline)

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.used)

    def out_of_time(self) -> bool:
        """Whether the tick's wall-clock allowance is spent."""
        return self.deadline_s is not None and time.monotonic() >= self.deadline_s

    def exhausted(self) -> bool:
        """Whether either limit is spent — no further step should start."""
        return self.used >= self.max_steps or self.out_of_time()

    def take(self) -> bool:
        """Consume one batched decoder step; ``False`` when the tick is spent.

        Always grants the first step of a tick: making progress matters more than
        holding a deadline that a single step cannot fit inside.
        """
        if self.used > 0 and self.exhausted():
            return False
        if self.used >= self.max_steps:
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
