# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture of one transducer predictor step.

The frame-synchronous greedy loop
(:meth:`oasr.engine.decode.transducer.TransducerStrategy._greedy_loop`) folds an
emitted label into the predictor and reprojects it:

    state    = decoder.advance(state, tok, emit)
    dec_proj = joiner.decoder_proj(decoder.predict(state))

For Nemotron's recurrent predictor that is an embedding lookup, two LSTM layers
(each an input projection plus a fused recurrent step), three ``torch.where``
masks and the joint projection -- **nine kernel launches for about 12-39 us of
GPU work.**  Measured at the layer, one step costs 89 us of host at batch 1 and
134 us at batch 128, and a real 16-utterance Nemotron decode spends 34% of its
greedy loop right here.  It is host-bound by a wide margin, and the launches are
the reason: an eager launch costs 4.7 us on this class of machine, which
``torch.relu_`` on eight elements pays too.

Replaying the same nine launches from a graph costs 0.045 us each.  So this is
not a kernel change and deliberately not a *dispatch* change -- the captured
graph runs exactly the launches the eager path runs, in the same order, and
produced bit-identical state and projections at every batch tested.  What it
removes is the per-step trip through the Python interpreter and the driver.

Three hazards, all of which this module owns:

* **The state the caller gets back is graph memory.**  It is valid only until the
  next replay.  The streaming path stores per-session state across ticks and
  slices it with ``unstack_states`` -- which for an LSTM predictor returns
  *views* -- so a session would silently read another tick's state.
  :meth:`PredictorStepGraphCache.detach` makes the owned copy, and the greedy
  loop calls it once before the state leaves.
* **Capture is expensive and can fail.**  Each attempt costs a warm-up forward
  and then runs eager anyway, so a shape that failed is never retried and an
  out-of-memory disables the whole cache: that is a fact about the process, not
  about the shape.  Same discipline as :class:`~oasr.engine.decoder_graph.
  DecoderStepGraphCache`.
* **Nested capture is illegal.**  If the caller is already capturing, this
  declines and the caller steps eagerly -- which records the identical launches
  into the caller's graph.  Declining changes *when* a launch is recorded, never
  which kernel runs, so it cannot make a captured decode diverge from an eager
  one the way a capture-dependent kernel choice would.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch

logger = logging.getLogger("oasr.engine.predictor_graph")

__all__ = ["PredictorStepGraphCache"]


@dataclass
class _Captured:
    """One captured predictor step plus the buffers it reads and writes."""

    graph: "torch.cuda.CUDAGraph"
    #: The static state the graph reads *and* writes back into, so a replay
    #: leaves the next replay's input in place and the steady state is copy-free.
    state: Tuple[torch.Tensor, ...]
    tok: torch.Tensor
    emit: torch.Tensor
    dec_proj: torch.Tensor


class PredictorStepGraphCache:
    """Lazily captured predictor steps, keyed by batch width.

    Parameters
    ----------
    predictor :
        The transducer predictor surface (``advance`` / ``predict``).
    joiner :
        Provides ``decoder_proj``; captured together with the step so the
        projection is not a tenth eager launch.
    max_captures :
        Ceiling on distinct batch widths.  Past it :meth:`step` returns ``None``
        and the caller runs eager rather than growing graph memory unbounded.
    """

    def __init__(
        self,
        predictor: Any,
        joiner: Any,
        *,
        max_captures: int = 8,
        pool: Any = None,
    ) -> None:
        self._predictor = predictor
        self._joiner = joiner
        self._max_captures = int(max_captures)
        self._pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self._captured: Dict[Tuple[int, ...], _Captured] = {}
        self._failed: Set[Tuple[int, ...]] = set()
        self._disabled = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_captured(self) -> int:
        return len(self._captured)

    @property
    def disabled(self) -> bool:
        return self._disabled

    @staticmethod
    def capturable(state: Any) -> bool:
        """Whether this predictor state is one a captured step can carry.

        A flat sequence of CUDA tensors is; anything else (a nested structure, a
        Python scalar the predictor keeps alongside) is not, and says so rather
        than being partially captured.  The state is opaque to the greedy loop by
        design, so this is the one place that inspects it.
        """
        if not isinstance(state, (tuple, list)) or not state:
            return False
        return all(isinstance(t, torch.Tensor) and t.is_cuda for t in state)

    @staticmethod
    def detach(state: Any) -> Any:
        """An owned copy of ``state``, safe to keep past the next replay."""
        if isinstance(state, (tuple, list)):
            return tuple(t.clone() for t in state)
        return state

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: Any,
        tok: torch.Tensor,
        emit: torch.Tensor,
    ) -> Optional[Tuple[Any, torch.Tensor]]:
        """One predictor step + projection through a captured graph.

        Returns ``(state, dec_proj)``, both **graph-owned** and valid only until
        the next replay -- call :meth:`detach` before keeping either.  Returns
        ``None`` when this shape is not capturable and the caller should step
        eagerly.
        """
        if self._disabled or not self.capturable(state):
            return None
        if torch.cuda.is_current_stream_capturing():
            # Nested capture is illegal; the caller's own graph records the same
            # launches.  See the module docstring.
            return None
        key = self._key(state)
        cap = self._captured.get(key)
        if cap is None:
            if key in self._failed or len(self._captured) >= self._max_captures:
                return None
            cap = self._capture(state, tok, emit, key)
            if cap is None:
                return None
        # Steady state: the graph wrote its own output back into the buffers it
        # reads, and the caller handed that same tuple straight back, so there is
        # nothing to copy.
        if not all(a is b for a, b in zip(state, cap.state)):
            for dst, src in zip(cap.state, state):
                dst.copy_(src)
        cap.tok.copy_(tok)
        cap.emit.copy_(emit)
        cap.graph.replay()
        return cap.state, cap.dec_proj

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @staticmethod
    def _key(state: Sequence[torch.Tensor]) -> Tuple[int, ...]:
        shape: list = [len(state)]
        for t in state:
            shape.append(len(t.shape))
            shape.extend(int(d) for d in t.shape)
        return tuple(shape)

    def _capture(
        self,
        state: Sequence[torch.Tensor],
        tok: torch.Tensor,
        emit: torch.Tensor,
        key: Tuple[int, ...],
    ) -> Optional[_Captured]:
        static = tuple(t.clone() for t in state)
        tok_buf = tok.clone()
        emit_buf = emit.clone()
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    warm = self._predictor.advance(static, tok_buf, emit_buf)
                    self._joiner.decoder_proj(self._predictor.predict(warm))
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                stepped = self._predictor.advance(static, tok_buf, emit_buf)
                # Write the new state back into the buffers the graph reads, so
                # the next replay continues from it with no copy.
                for dst, src in zip(static, stepped):
                    dst.copy_(src)
                proj = self._joiner.decoder_proj(self._predictor.predict(static))
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            # A property of the process, not of the shape.  Each further attempt
            # would burn a warm-up forward and still run eager.
            logger.warning("predictor step graph capture ran out of memory; disabling capture")
            self._disabled = True
            return None
        except Exception as exc:
            logger.warning("predictor step graph capture failed for %s: %s", key, exc)
            self._failed.add(key)
            return None
        cap = _Captured(graph=graph, state=static, tok=tok_buf, emit=emit_buf, dec_proj=proj)
        self._captured[key] = cap
        logger.info("captured predictor step graph for state shape key %s", key)
        return cap
