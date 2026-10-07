# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture of the transducer beam-search frame loop, ``unroll`` frames per replay.

Modified beam search takes exactly one step per encoder frame, and each step --
the predictor over every hypothesis, the joiner, ``log_softmax``, the ``top-k``
over ``k * V`` candidates, the parent gathers, the masks -- is about thirty
launches for a few microseconds of GPU work each.  Profiled offline at ``B = 64``,
``k = 4`` over 250 frames: **146 ms of wall time for 14.8 ms of GPU work**, the
GPU 10% busy, and the wall time the same at ``B = 128`` and ``B = 256``.  The host
issues, the GPU waits -- the greedy loop's problem (``greedy_graph.py``), in a
shape that is easier to capture: there is no per-frame host decision at all.

Every op of a step is fixed-shape for a given ``(B, k)``, so ``unroll`` steps
capture back to back, and a chunk of ``T`` frames is ``ceil(T / unroll)``
replays decided before the first one -- no termination read, unlike greedy.

Exactness
---------
The captured body is :func:`~oasr.engine.decode.transducer_beam.beam_search_step`
op for op at the same batch width, so it is bit-identical to the eager loop:

* **The key's batch axis is exact**, for the reason ``greedy_graph.py`` gives:
  padding rows would change ``M`` for the predictor and joiner GEMMs, and a
  shape-aware GEMM may pick a different tile -- a different reduction order.
* **The time axis is bucketed, and that is inert.**  The encoder projection is
  copied into a static ``(B, T_cap, J)`` buffer.  An active row always has
  ``t < length <= T``, a real frame; the frames a replay runs past ``T`` are
  inactive for every row, which leaves the beam untouched and records "every
  slot its own parent, emitting blank" -- a no-op when the back-pointers are
  walked.  ``T_cap`` is a power of two at least ``unroll``-aligned, so the
  ``ceil(T / unroll) * unroll`` frames a call runs always fit in it.
* **Nothing in it is data-dependent on the host.**  Back-pointers and labels
  land in static frame-major ``(T_cap, B, k)`` buffers; the chunk's first ``T``
  frames are copied out once and walked by
  :func:`~oasr.engine.decode.transducer_beam.walk_chunk`.

The strategy pads a chunk's batch to a power of two before it gets here
(``TransducerDecodeStrategy._beam_frames``), with rows that have no frames, so
the exact-width key sees a handful of widths rather than every cohort size.

A failed capture is remembered and not retried, and an out-of-memory disables
the cache: the ``GreedyLoopGraphCache`` discipline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import torch
import tvm_ffi

from .capture_recovery import capture_warmup_stream, recover_from_failed_capture
from .decode.transducer_beam import beam_search_step, step_constants
from .greedy_graph import _MIN_T_CAP, _t_capacity

logger = logging.getLogger(__name__)

__all__ = ["BeamLoopGraphCache"]


@dataclass
class _Captured:
    graph: "torch.cuda.CUDAGraph"
    enc: torch.Tensor  # (B, T_cap, J) encoder projection, copied in per call
    lengths: torch.Tensor  # (B,) int64
    t: torch.Tensor  # (1,) int64 frame index -- read and advanced by the graph
    context: torch.Tensor  # (B, k, ctx) int64 label windows, read and written
    scores: torch.Tensor  # (B, k) float32, read and written
    parents: torch.Tensor  # (T_cap, B, k) int64 back-pointers, written
    labels: torch.Tensor  # (T_cap, B, k) int64 labels taken, written
    # Constants the graph reads.  Held here, not as locals of ``_capture``: a
    # graph reads the *address* it was captured with (see greedy_graph.py).
    stay: torch.Tensor  # (B, k) arange(k)
    blank_label: torch.Tensor  # (B, k) blank
    offsets: torch.Tensor  # (unroll,) arange(unroll)


class BeamLoopGraphCache:
    """Lazily captured beam-search frame blocks, keyed by batch geometry and frame capacity.

    Parameters
    ----------
    model :
        The transducer :func:`beam_search_step` drives (``decoder``, ``joiner``,
        ``blank_id``).
    unroll :
        Frames per replay.  Must divide the smallest frame capacity, so a call's
        whole-replay frame count never runs past its buffers.
    fused :
        Passed to :func:`beam_search_step`: the frame's selection as one kernel.
    max_captures :
        Ceiling on live captures; past it :meth:`run` returns ``None`` and the
        caller runs eager.
    """

    def __init__(
        self,
        model: Any,
        *,
        unroll: int,
        fused: bool = True,
        max_captures: int = 48,
        pool: Optional[Tuple[int, int]] = None,
    ) -> None:
        if unroll < 1 or _MIN_T_CAP % unroll:
            raise ValueError(f"unroll={unroll} must divide the minimum frame capacity {_MIN_T_CAP}")
        self._model = model
        self._unroll = int(unroll)
        self._fused = bool(fused)
        self._max_captures = int(max_captures)
        self._pool: Optional[Tuple[int, int]] = (
            pool if pool is not None else torch.cuda.graph_pool_handle()
        )
        self._captured: Dict[Tuple[Any, ...], _Captured] = {}
        self._failed: Set[Tuple[Any, ...]] = set()
        self._disabled = False
        #: Accounting: chunks served by replays vs handed back to the eager loop.
        self.hits = 0
        self.fallbacks = 0
        self.captures = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_captured(self) -> int:
        return len(self._captured)

    @property
    def disabled(self) -> bool:
        return self._disabled

    def stats(self) -> Dict[str, int]:
        return {
            "captured": len(self._captured),
            "captures": self.captures,
            "hits": self.hits,
            "fallbacks": self.fallbacks,
        }

    def release(self) -> None:
        """Return the graph pool's VRAM.  Idempotent."""
        for cap in self._captured.values():
            try:
                cap.graph.reset()
            except Exception:  # pragma: no cover - teardown must not raise
                pass
        self._captured.clear()
        self._failed.clear()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        enc_proj: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
        scores: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Beam-search every frame of ``enc_proj`` from ``(context, scores)``.

        Returns ``(context, scores, parents, labels)`` -- the beam after the
        chunk and the chunk's ``(T, B, k)`` back-pointers and labels, all owned
        device copies -- or ``None`` when this call is not served and the caller
        should run the eager loop.
        """
        if self._disabled or torch.cuda.is_current_stream_capturing():
            self.fallbacks += 1
            return None
        B, T, J = (int(d) for d in enc_proj.shape)
        _, k, ctx = (int(d) for d in context.shape)
        t_cap = _t_capacity(T)
        key = (B, k, ctx, t_cap, J, enc_proj.dtype, context.dtype, scores.dtype)
        cap = self._captured.get(key)
        if cap is None:
            if key in self._failed or len(self._captured) >= self._max_captures:
                self.fallbacks += 1
                return None
            cap = self._capture(key, t_cap, enc_proj, context, scores)
            if cap is None:
                self.fallbacks += 1
                return None

        # Refill the inputs.  Frames past ``T`` keep whatever an earlier call
        # left; they are inactive for every row (see the module docstring).
        cap.enc[:, :T].copy_(enc_proj)
        cap.lengths.copy_(lengths)
        cap.context.copy_(context)
        cap.scores.copy_(scores)
        cap.t.zero_()
        for _ in range(-(-T // self._unroll)):
            cap.graph.replay()
        self.hits += 1
        return (
            cap.context.clone(),
            cap.scores.clone(),
            cap.parents[:T].clone(),
            cap.labels[:T].clone(),
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _body(self, c: _Captured) -> None:
        """``unroll`` frames of the eager beam loop, op for op."""
        t_last = c.enc.size(1) - 1
        context, scores = c.context, c.scores
        parents, labels = [], []
        for i in range(self._unroll):
            t = c.t + i
            enc_t = c.enc.index_select(1, t.clamp(max=t_last)).squeeze(1)
            context, scores, parent, label = beam_search_step(
                self._model,
                enc_t,
                context,
                scores,
                t < c.lengths,
                c.stay,
                c.blank_label,
                self._fused,
            )
            parents.append(parent)
            labels.append(label)
        # The block's frames, written at their own rows of the history.  The
        # clamp never binds (a call's replays fit in ``T_cap``); it only keeps a
        # misuse an out-of-range write rather than a corrupt one.
        rows = (c.t + c.offsets).clamp(max=t_last)
        c.parents.index_copy_(0, rows, torch.stack(parents, dim=0))
        c.labels.index_copy_(0, rows, torch.stack(labels, dim=0))
        c.context.copy_(context)
        c.scores.copy_(scores)
        c.t.add_(self._unroll)

    def _capture(
        self,
        key: Tuple[Any, ...],
        t_cap: int,
        enc_proj: torch.Tensor,
        context: torch.Tensor,
        scores: torch.Tensor,
    ) -> Optional[_Captured]:
        device = enc_proj.device
        B, T, J = (int(d) for d in enc_proj.shape)
        k = int(context.size(1))
        long = torch.long
        stay, blank_label = step_constants(B, k, int(self._model.blank_id), device)
        c = _Captured(
            graph=torch.cuda.CUDAGraph(),
            enc=torch.zeros(B, t_cap, J, dtype=enc_proj.dtype, device=device),
            lengths=torch.zeros(B, dtype=long, device=device),
            t=torch.zeros(1, dtype=long, device=device),
            context=context.clone(),
            scores=scores.clone(),
            parents=torch.zeros(t_cap, B, k, dtype=long, device=device),
            labels=torch.zeros(t_cap, B, k, dtype=long, device=device),
            stay=stay,
            blank_label=blank_label,
            offsets=torch.arange(self._unroll, dtype=long, device=device),
        )
        # Real frames and lengths for the warm-up, so its kernels see the same
        # shapes and plausible values the replays will.
        c.enc[:, :T].copy_(enc_proj)
        c.lengths.fill_(T)
        try:
            side = capture_warmup_stream(device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._body(c)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            ctx = torch.cuda.graph(c.graph, pool=self._pool)  # type: ignore[arg-type]
            with tvm_ffi.use_torch_stream(ctx):
                self._body(c)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            logger.warning("beam loop graph capture ran out of memory; disabling capture")
            self._disabled = True
            recover_from_failed_capture(device, self._pool)
            self._pool = None
            return None
        except Exception as exc:
            logger.warning(
                "beam loop graph capture failed for B=%d k=%d T_cap=%d: %s", B, k, t_cap, exc
            )
            self._failed.add(key)
            recover_from_failed_capture(device, self._pool)
            self._pool = torch.cuda.graph_pool_handle()
            return None
        # The warm-up advanced the carried buffers; that is harmless, because
        # :meth:`run` refills every input before each call's first replay.
        self._captured[key] = c
        self.captures += 1
        logger.info(
            "captured beam loop block: B=%d k=%d T_cap=%d unroll=%d (%d live)",
            B,
            k,
            t_cap,
            self._unroll,
            len(self._captured),
        )
        return c
