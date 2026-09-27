# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture of the transducer greedy loop, ``unroll`` iterations per replay.

:class:`~oasr.engine.predictor_graph.PredictorStepGraphCache` already folds the
predictor step into one replay.  What it leaves is everything around it -- the
encoder-frame gather, the joiner, ``argmax``, the blank/cap comparisons, the
row masks and the ``t`` / ``sym`` updates -- about twenty launches per loop
iteration for a few microseconds of GPU work each.  Profiled offline at
``B = 32``: **45 174 eager launches per 256 utterances** on the icefall
transducer, GPU **25% busy**, the loop 98% of decode wall.  The host issues, the
GPU waits.

Every one of those ops is fixed-shape for a given batch, so a whole iteration is
capturable, and so are ``unroll`` of them back to back.  One replay then runs
``unroll`` iterations for a single launch, and the host only comes back to ask
whether any row is still inside its utterance -- the same question the eager
loop asks, at the same stride (``_TERMINATION_CHECK_STRIDE``), which is what
makes the two run exactly the same number of iterations.

Exactness
---------
The captured body is the eager ``track=False`` iteration op for op, at the same
batch width, so it is bit-identical to it:

* **The key's batch axis is exact.**  Padding rows up to a bucket would change
  ``M`` for the joiner GEMMs, and a shape-aware GEMM may pick a different tile --
  a different reduction order -- at a different ``M``.  Streaming widths are
  bounded by ``max_batch_size`` anyway.
* **The time axis is bucketed, and that is inert.**  The encoder projection is
  copied into a static ``(B, T_cap, J)`` buffer; the per-iteration gather reads
  row ``t`` of it, and an *active* row always has ``t < length <= T``, i.e. a
  real frame.  An inactive row reads anything, and its result is masked out of
  every update exactly as in the eager loop.
* **Nothing in it is data-dependent on the host.**  Emitted tokens land in a
  static ``(unroll, B)`` buffer (``-1`` where a row did not emit), copied out
  after each replay and filtered once at the end -- the same snapshots the eager
  loop stacks.

What is not captured: the ``track=True`` loop (word timings / speech activity),
whose emission gate is a host decision per iteration, and beam search.  Both keep
the eager path.  A failed capture is remembered and not retried, and an
out-of-memory disables the cache: the ``DecoderStepGraphCache`` discipline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import tvm_ffi

from .capture_recovery import capture_warmup_stream, recover_from_failed_capture

logger = logging.getLogger(__name__)

__all__ = ["GreedyLoopGraphCache"]

#: Smallest static time capacity; below it every chunk would key its own capture.
_MIN_T_CAP = 64


def _t_capacity(frames: int) -> int:
    """Power-of-two frame capacity ``>= frames`` (floor :data:`_MIN_T_CAP`)."""
    cap = _MIN_T_CAP
    while cap < frames:
        cap *= 2
    return cap


@dataclass
class _Captured:
    graph: "torch.cuda.CUDAGraph"
    enc: torch.Tensor  # (B, T_cap, J) encoder projection, copied in per call
    lengths: torch.Tensor  # (B,) int64
    t: torch.Tensor  # (B,) int64 -- read and written by the graph
    sym: torch.Tensor  # (B,) int64
    state: Tuple[torch.Tensor, ...]  # predictor state, read and written
    dec_proj: torch.Tensor  # (B, J)
    out: torch.Tensor  # (unroll, B) emitted token per iteration, -1 = none
    live: torch.Tensor  # () bool: any row still inside its utterance
    bare: bool  # the predictor carries its state as one tensor, not a tuple
    # Constants the graph reads.  Held here, not as locals of ``_capture``: a
    # graph reads the *address* it was captured with, and a tensor freed after
    # capture hands that address to whatever the allocator serves next -- the
    # frame gather then indexes with garbage (a device-side out-of-bounds
    # assert, the first time this was written).
    rows: torch.Tensor  # (B,) arange
    no_emit: torch.Tensor  # (B,) -1
    zero_sym: torch.Tensor  # (B,) 0


class GreedyLoopGraphCache:
    """Lazily captured greedy-loop blocks, keyed by batch width and frame capacity.

    Parameters
    ----------
    predictor, joiner :
        The transducer surface the eager loop drives (``advance`` / ``predict`` /
        ``decoder_proj`` / ``joiner(enc, dec, project_input=False)``).
    blank, max_sym :
        The loop's constants -- baked into the graph, so a strategy with another
        value needs another cache.
    unroll :
        Iterations per replay.  Must equal the eager loop's termination stride,
        or the two would run different iteration counts (inert, but no longer
        the same launch sequence).
    max_captures :
        Ceiling on live captures; past it :meth:`run` returns ``None`` and the
        caller runs eager.
    """

    def __init__(
        self,
        predictor: Any,
        joiner: Any,
        *,
        blank: int,
        max_sym: int,
        unroll: int,
        max_captures: int = 48,
        pool: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._predictor = predictor
        self._joiner = joiner
        self._blank = int(blank)
        self._max_sym = int(max_sym)
        self._unroll = int(unroll)
        self._max_captures = int(max_captures)
        self._pool: Optional[Tuple[int, int]] = (
            pool if pool is not None else torch.cuda.graph_pool_handle()
        )
        self._captured: Dict[Tuple[Any, ...], _Captured] = {}
        self._failed: Set[Tuple[Any, ...]] = set()
        self._disabled = False
        #: Accounting: loops served by a replay vs handed back to the eager loop.
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
    # State shape helpers (the predictor state is opaque; see transducer.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _as_seq(state: Any) -> Optional[Tuple[Tuple[torch.Tensor, ...], bool]]:
        """``(buffers, was_bare)``, or ``None`` for a state a graph cannot carry."""
        if isinstance(state, torch.Tensor):
            return ((state,), True) if state.is_cuda else None
        if not isinstance(state, (tuple, list)) or not state:
            return None
        if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in state):
            return None
        return tuple(state), False

    @staticmethod
    def _as_state(seq: Sequence[torch.Tensor], bare: bool) -> Any:
        return seq[0] if bare else tuple(seq)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        enc_proj: torch.Tensor,
        lengths: torch.Tensor,
        state: Any,
        dec_proj: torch.Tensor,
        max_steps: int,
    ) -> Optional[Tuple[List[List[int]], Any, torch.Tensor]]:
        """Greedy-decode ``enc_proj`` from ``(state, dec_proj)``.

        Returns ``(hyps, state, dec_proj)`` -- the state and projection are owned
        copies, safe to keep across ticks -- or ``None`` when this call is not
        served and the caller should run the eager loop.
        """
        seq = self._as_seq(state)
        if self._disabled or seq is None or torch.cuda.is_current_stream_capturing():
            self.fallbacks += 1
            return None
        state_seq, bare = seq
        B, T, J = (int(d) for d in enc_proj.shape)
        t_cap = _t_capacity(T)
        key = (
            B,
            t_cap,
            J,
            enc_proj.dtype,
            bare,
            tuple((tuple(s.shape), s.dtype) for s in state_seq),
            tuple(dec_proj.shape),
            dec_proj.dtype,
        )
        cap = self._captured.get(key)
        if cap is None:
            if key in self._failed or len(self._captured) >= self._max_captures:
                self.fallbacks += 1
                return None
            cap = self._capture(key, B, t_cap, enc_proj, state_seq, bare, dec_proj)
            if cap is None:
                self.fallbacks += 1
                return None

        # Refill the inputs.  Frames past ``T`` keep whatever an earlier call
        # left; only inactive rows ever read them (see the module docstring).
        cap.enc[:, :T].copy_(enc_proj)
        cap.lengths.copy_(lengths)
        cap.t.zero_()
        cap.sym.zero_()
        for dst, src in zip(cap.state, state_seq):
            dst.copy_(src)
        cap.dec_proj.copy_(dec_proj)

        chunks: List[torch.Tensor] = []
        done = 0
        while done < max_steps:
            cap.graph.replay()
            chunks.append(cap.out.clone())
            done += self._unroll
            # The eager loop's termination check, at its stride: one host read
            # per ``unroll`` iterations.
            if not bool(cap.live):
                break
        self.hits += 1

        snap = torch.cat(chunks, dim=0).t().tolist()  # B x steps, one readback
        hyps = [[tk for tk in row if tk >= 0] for row in snap]
        out_state = self._as_state(tuple(s.clone() for s in cap.state), bare)
        return hyps, out_state, cap.dec_proj.clone()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _body(self, c: _Captured) -> None:
        """``unroll`` iterations of the eager ``track=False`` loop, op for op."""
        joiner, predictor = self._joiner, self._predictor
        rows, no_emit, zero_sym = c.rows, c.no_emit, c.zero_sym
        t_last = c.enc.size(1) - 1
        t, sym, dec_proj = c.t, c.sym, c.dec_proj
        state = self._as_state(c.state, c.bare)
        outs = []
        for _ in range(self._unroll):
            active = t < c.lengths
            enc_t = c.enc[rows, t.clamp(max=t_last)]
            logits = joiner(enc_t, dec_proj, project_input=False)
            tok = logits.argmax(dim=-1)
            is_blank = (tok == self._blank) | (sym >= self._max_sym)
            emit = active & ~is_blank
            advance = active & is_blank
            state = predictor.advance(state, tok, emit)
            dec_proj = joiner.decoder_proj(predictor.predict(state))
            outs.append(torch.where(emit, tok, no_emit))
            sym = sym + emit.long()
            t = t + advance.long()
            sym = torch.where(advance, zero_sym, sym)
        # Write the carried values back into the buffers the next replay reads.
        torch.stack(outs, dim=0, out=c.out)
        c.t.copy_(t)
        c.sym.copy_(sym)
        new_seq = (state,) if c.bare else tuple(state)
        for dst, src in zip(c.state, new_seq):
            dst.copy_(src)
        c.dec_proj.copy_(dec_proj)
        c.live.copy_((t < c.lengths).any())

    def _capture(
        self,
        key: Tuple[Any, ...],
        B: int,
        t_cap: int,
        enc_proj: torch.Tensor,
        state_seq: Tuple[torch.Tensor, ...],
        bare: bool,
        dec_proj: torch.Tensor,
    ) -> Optional[_Captured]:
        device = enc_proj.device
        J = int(enc_proj.size(2))
        long = torch.long
        c = _Captured(
            graph=torch.cuda.CUDAGraph(),
            enc=torch.zeros(B, t_cap, J, dtype=enc_proj.dtype, device=device),
            lengths=torch.zeros(B, dtype=long, device=device),
            t=torch.zeros(B, dtype=long, device=device),
            sym=torch.zeros(B, dtype=long, device=device),
            state=tuple(s.clone() for s in state_seq),
            dec_proj=dec_proj.clone(),
            out=torch.empty(self._unroll, B, dtype=long, device=device),
            live=torch.zeros((), dtype=torch.bool, device=device),
            bare=bare,
            rows=torch.arange(B, device=device),
            no_emit=torch.full((B,), -1, dtype=long, device=device),
            zero_sym=torch.zeros(B, dtype=long, device=device),
        )
        # Real frames and lengths for the warm-up, so its kernels see the same
        # shapes and plausible values the replays will.
        c.enc[:, : enc_proj.size(1)].copy_(enc_proj)
        c.lengths.fill_(int(enc_proj.size(1)))
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
            logger.warning("greedy loop graph capture ran out of memory; disabling capture")
            self._disabled = True
            recover_from_failed_capture(device, self._pool)
            self._pool = None
            return None
        except Exception as exc:
            logger.warning("greedy loop graph capture failed for B=%d T_cap=%d: %s", B, t_cap, exc)
            self._failed.add(key)
            recover_from_failed_capture(device, self._pool)
            self._pool = torch.cuda.graph_pool_handle()
            return None
        # The warm-up advanced the carried buffers; that is harmless, because
        # :meth:`run` refills every input before each call's first replay.
        self._captured[key] = c
        self.captures += 1
        logger.info(
            "captured greedy loop block: B=%d T_cap=%d unroll=%d (%d live)",
            B,
            t_cap,
            self._unroll,
            len(self._captured),
        )
        return c
