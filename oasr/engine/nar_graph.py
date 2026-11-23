# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture of a CIF predictor and non-autoregressive decoder (Paraformer).

The offline encoder has been captured since :mod:`offline_graph`; the decode
behind it was not, and for Paraformer that decode is a 16-layer NAR decoder plus
the CIF predictor -- about 340 launches per request with nothing between them
for the GPU to do.  Measured on ``paraformer-zh`` (bf16, RTX 5090, AISHELL):

=================  ====================  ==================================
Shape              eager decode (wall)   captured predictor + decoder (GPU)
=================  ====================  ==================================
B=1, T=128, U=14   7.37 ms               0.06 + 0.78 ms
B=64, T=128, U=17  8.99 ms               0.11 + 2.33 ms
B=64, T=192, U=25  9.45 ms               0.15 + 3.41 ms
=================  ====================  ==================================

Why two graphs
--------------
The decoder's width is the token count ``U``, which only the predictor knows.
One graph over both would have to fix ``U`` before seeing it: at ``T / 2`` or
``2 * U`` the decoder costs 40-100 % more at ``B=64`` (it scales with ``U``
there: 2.33 / 3.10 / 4.79 ms at ``U`` = 17 / 32 / 64).  So the predictor's
weights are one graph keyed ``(B, T)``, the host reads back the per-row token
counts -- the one read the eager path already made, now a ``(B,)`` vector
instead of its maximum -- and the CIF integration, decoder and greedy pick are a
second graph keyed ``(B, T, U)``, with ``U`` rounded up to
:data:`DEFAULT_TOKEN_GRANULARITY`.  Rounding costs about 1 % at ``B=64`` and
nothing measurable at ``B=1``, where the decode is launch-bound.

The decode graph reads the predictor graph's buffers in place -- the encoder
output it was fed, the lengths, the weights and the counts -- so it is stored
under that predictor graph and released with it.

Padding
-------
``T`` is not bucketed here: the encoder graph cache already rounds it, on both
of its paths, so the encoder output arrives at a bucket width.  A width past
``max_frames`` (where the encoder cache stops padding) runs eager.

``U`` is rounded on **both** paths, by the caller: rows past a row's own count
are zero and the decoder masks them, but the rounded width changes the decoder's
GEMM shapes, so an eager fallback at the unrounded width could pick other
kernels than the replay and decode a row differently.  The batch axis is padded
on the captured path only, as the encoder cache does: a padded row is a
zero-length-1 utterance, finite throughout.

Lifetime of the outputs
-----------------------
All captures share one pool, so every returned tensor is a view of a replay
buffer that the next replay of *any* graph here may overwrite.  The caller reads
the counts back before the decode replay and queues the decode's read-back
before anything else runs; that is the whole protocol.

Memory
------
The encoder output is copied into one flat staging buffer that every predictor
graph views, sized to the widest ``B * T`` seen so far: at ``B=64`` a per-shape
copy would hold 32 MiB per 512-frame bucket and GiBs across a long-form ladder.
A capture is kept only if the device keeps
:data:`~oasr.engine.memory.ACTIVATION_SAFETY` times the largest measured
warm-up free afterwards (the encoder cache's rule); a capture that runs out of
memory turns the cache off and releases it.
"""

from __future__ import annotations

import gc
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Optional, Sequence, Set, Tuple

import torch
import tvm_ffi

from oasr.utils.capture_guard import capture_guard

from .capture_recovery import recover_from_failed_capture
from .memory import ACTIVATION_SAFETY, MIN_ACTIVATION_RESERVE_BYTES, read_device_memory
from .offline_graph import BUDGET_RETRY_S

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_TOKEN_GRANULARITY", "NarDecodeGraphs", "WeightsStage"]

#: Token-axis rounding of the decode graph's key.  Finer keys more shapes for a
#: saving under 1 % at ``B=64``; see the module docstring.
DEFAULT_TOKEN_GRANULARITY = 8

#: ``(enc_out, enc_lens) -> (alphas, token_lens)``.
WeightsFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
#: ``(enc_out, enc_lens, alphas, token_lens, num_tokens) -> (best_lp, best_ids, fires)``.
DecodeFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


@dataclass
class _Decode:
    graph: "torch.cuda.CUDAGraph"
    outputs: Tuple[torch.Tensor, ...]


@dataclass
class WeightsStage:
    """One predictor graph and the buffers its decode graphs read.

    After :meth:`NarDecodeGraphs.weights` returns it, ``alphas`` and
    ``token_lens`` hold the batch's values in their first ``active`` rows.  They
    are replay buffers: valid until the next replay of any graph in the cache.
    """

    key: Tuple[int, int]
    graph: "torch.cuda.CUDAGraph"
    enc: torch.Tensor
    lens: torch.Tensor
    alphas: torch.Tensor
    token_lens: torch.Tensor
    active: int = 0
    decodes: "OrderedDict[int, _Decode]" = field(default_factory=OrderedDict)


class NarDecodeGraphs:
    """Two-stage capture cache for a CIF + NAR decode; see the module docstring.

    Parameters
    ----------
    device : torch.device
        The model's device; anything but CUDA disables the cache.
    weights_fn, decode_fn
        The two halves, captured as they are called.  ``decode_fn`` takes the
        token width as a plain ``int``.
    batch_buckets : sequence of int
        Batch widths to capture at (the encoder cache's).
    max_frames : int
        Widest encoder output served.
    max_captures : int
        Live graphs of both kinds together.
    evict_lru : bool
        At ``max_captures``, release the least recently replayed predictor graph
        (with its decode graphs) instead of running the new shape eager.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        weights_fn: WeightsFn,
        decode_fn: DecodeFn,
        batch_buckets: Sequence[int],
        max_frames: int = 4096,
        max_captures: int = 512,
        token_granularity: int = DEFAULT_TOKEN_GRANULARITY,
        evict_lru: bool = True,
        eager_reserve_factor: float = ACTIVATION_SAFETY,
    ) -> None:
        self._device = device
        self._weights_fn = weights_fn
        self._decode_fn = decode_fn
        self._buckets = sorted({int(b) for b in batch_buckets if int(b) >= 1})
        self._max_frames = int(max_frames)
        self._max_captures = max(1, int(max_captures))
        self._granularity = max(1, int(token_granularity))
        self._evict_lru = bool(evict_lru)
        self._eager_reserve_factor = float(eager_reserve_factor)
        self._enabled = device.type == "cuda" and bool(self._buckets)
        self._disabled = False
        self._pool: Optional[Tuple[int, int]] = None
        if self._enabled:
            self._pool = torch.cuda.graph_pool_handle()
        #: Predictor graphs, least recently replayed first.
        self._stages: "OrderedDict[Tuple[int, int], WeightsStage]" = OrderedDict()
        self._failed: Set[Hashable] = set()
        #: Keys refused for memory, with when; asked again after BUDGET_RETRY_S.
        self._refused: Dict[Hashable, float] = {}
        self._logged_budget = False
        self._logged_full = False
        self._staging: Optional[torch.Tensor] = None
        self._largest_bytes = 0

        self.hits = 0
        self.captures = 0
        self.capture_failures = 0
        self.evictions = 0
        self.fallback_oversized = 0
        self.fallback_saturated = 0
        self.fallback_failed = 0
        self.fallback_budget = 0
        self.fallback_disabled = 0
        self.pool_bytes = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._disabled

    @property
    def num_captured(self) -> int:
        return len(self._stages) + sum(len(s.decodes) for s in self._stages.values())

    def token_bucket(self, tokens: int) -> int:
        """``tokens`` rounded up to the token granularity (at least one bucket)."""
        g = self._granularity
        return -(-max(1, int(tokens)) // g) * g

    def stats(self) -> Dict[str, float]:
        return {
            "captured_shapes": float(self.num_captured),
            "captures": float(self.captures),
            "hits": float(self.hits),
            "fallback_oversized": float(self.fallback_oversized),
            "fallback_saturated": float(self.fallback_saturated),
            "fallback_failed": float(self.fallback_failed),
            "fallback_budget": float(self.fallback_budget),
            "fallback_disabled": float(self.fallback_disabled),
            "evictions": float(self.evictions),
            "capture_failures": float(self.capture_failures),
            "pool_gib": self.pool_bytes / float(1024**3),
        }

    def release(self) -> None:
        """Reset every capture and drop the staging buffer.  Idempotent.

        ``CUDAGraph.reset()`` is the only thing that returns a capture's share of
        the private pool (see :meth:`GraphedOfflineForward.release`).  The cache
        stays usable and re-captures lazily, which is what a tuning reload needs.
        """
        for key in list(self._stages):
            self._drop_stage(key)
        self._failed.clear()
        self._refused.clear()
        self._staging = None
        self.pool_bytes = 0

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    @torch.no_grad()
    def weights(self, enc_out: torch.Tensor, enc_lens: torch.Tensor) -> Optional[WeightsStage]:
        """Replay (or first capture) the predictor's weights for this batch.

        ``None`` when the shape is not served, and the caller runs ``weights_fn``
        eagerly; every such return is counted.
        """
        if not self.enabled:
            if self._enabled:
                self.fallback_disabled += 1
            return None
        rows, frames = int(enc_out.size(0)), int(enc_out.size(1))
        bucket = next((b for b in self._buckets if b >= rows), None)
        if bucket is None or frames < 1 or frames > self._max_frames:
            self.fallback_oversized += 1
            return None
        key = (bucket, frames)
        stage = self._stages.get(key)
        if stage is None:
            stage = self._capture_weights(key, enc_out, enc_lens)
            if stage is None:
                return None
            self._stages[key] = stage
        else:
            self._stages.move_to_end(key)
            self._refill(stage.enc, stage.lens, enc_out, enc_lens)
        stage.graph.replay()
        stage.active = rows
        self.hits += 1
        return stage

    @torch.no_grad()
    def decode(self, stage: WeightsStage, num_tokens: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """Replay (or first capture) the decode over ``stage`` at ``num_tokens``.

        Call it right after :meth:`weights` returned ``stage``, with a width
        :meth:`token_bucket` produced.  Returns ``(best_lp, best_ids, fires)`` for
        the ``stage.active`` rows, as replay-buffer views, or ``None`` when not
        served.
        """
        if not self.enabled or stage.key not in self._stages:
            if self._enabled:
                self.fallback_disabled += 1
            return None
        width = int(num_tokens)
        decode = stage.decodes.get(width)
        if decode is None:
            decode = self._capture_decode(stage, width)
            if decode is None:
                return None
            stage.decodes[width] = decode
        else:
            stage.decodes.move_to_end(width)
        decode.graph.replay()
        self.hits += 1
        rows = stage.active
        return tuple(t[:rows] for t in decode.outputs)

    @staticmethod
    def _refill(
        enc: torch.Tensor, lens: torch.Tensor, enc_out: torch.Tensor, enc_lens: torch.Tensor
    ) -> None:
        """Copy the batch into the static inputs; padded rows become length-1 silence.

        The staging buffer is shared with every other predictor graph, so the
        rows past the batch hold whatever the last, differently shaped batch
        left there: zeroed every time, not once.
        """
        rows = int(enc_out.size(0))
        enc[:rows].copy_(enc_out)
        lens[:rows].copy_(enc_lens)
        if rows < enc.size(0):
            enc[rows:].zero_()
            lens[rows:].fill_(1)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture_weights(
        self, key: Tuple[int, int], enc_out: torch.Tensor, enc_lens: torch.Tensor
    ) -> Optional[WeightsStage]:
        tag = ("weights",) + key
        if not self._admit(tag, protect=None):
            return None
        bucket, frames = key
        enc = self._staging_view(bucket, frames, enc_out)
        lens = torch.ones(bucket, dtype=enc_lens.dtype, device=self._device)
        self._refill(enc, lens, enc_out, enc_lens)
        recorded = self._record(tag, lambda: self._weights_fn(enc, lens))
        if recorded is None:
            return None
        graph, (alphas, token_lens) = recorded
        return WeightsStage(
            key=key, graph=graph, enc=enc, lens=lens, alphas=alphas, token_lens=token_lens
        )

    def _capture_decode(self, stage: WeightsStage, width: int) -> Optional[_Decode]:
        tag = ("decode",) + stage.key + (width,)
        if not self._admit(tag, protect=stage.key):
            return None
        recorded = self._record(
            tag,
            lambda: self._decode_fn(stage.enc, stage.lens, stage.alphas, stage.token_lens, width),
        )
        if recorded is None:
            return None
        graph, outputs = recorded
        return _Decode(graph=graph, outputs=tuple(outputs))

    def _admit(self, tag: Hashable, protect: Optional[Tuple[int, int]]) -> bool:
        """Whether ``tag`` may be captured now: not failed, not refused, room left."""
        if tag in self._failed:
            self.fallback_failed += 1
            return False
        refused_at = self._refused.get(tag)
        if refused_at is not None:
            if time.monotonic() - refused_at < BUDGET_RETRY_S:
                self.fallback_budget += 1
                return False
            del self._refused[tag]
        if self.num_captured >= self._max_captures and not self._evict(protect):
            self.fallback_saturated += 1
            if not self._logged_full:
                self._logged_full = True
                logger.info(
                    "NAR decode graph cache full (%d graphs); further shapes run eager",
                    self._max_captures,
                )
            return False
        return True

    def _evict(self, protect: Optional[Tuple[int, int]]) -> bool:
        """Free one slot: the least recently replayed predictor graph and its decodes.

        ``protect`` is the predictor graph a decode is being captured under; when
        it is the only one, its own least recent decode goes instead.
        """
        if not self._evict_lru:
            return False
        for key in self._stages:
            if key != protect:
                self._drop_stage(key)
                return True
        stage = self._stages.get(protect) if protect is not None else None
        if stage is not None and stage.decodes:
            _width, decode = stage.decodes.popitem(last=False)
            self._reset(decode.graph)
            self.evictions += 1
            return True
        return False

    def _drop_stage(self, key: Tuple[int, int]) -> None:
        stage = self._stages.pop(key)
        for decode in stage.decodes.values():
            self._reset(decode.graph)
        stage.decodes.clear()
        self._reset(stage.graph)
        self.evictions += 1

    @staticmethod
    def _reset(graph: "torch.cuda.CUDAGraph") -> None:
        try:
            graph.reset()
        except Exception:  # pragma: no cover - teardown must not raise
            logger.debug("NAR decode graph reset failed", exc_info=True)

    def _staging_view(self, rows: int, frames: int, like: torch.Tensor) -> torch.Tensor:
        """A ``(rows, frames, D)`` view of the shared encoder-output staging buffer.

        Grown to at least twice its size when too small; graphs captured against
        the old buffer keep it alive through their views, so it is freed when the
        last of them is evicted.
        """
        numel = rows * frames * int(like.size(2))
        buf = self._staging
        if buf is None or buf.dtype != like.dtype or buf.numel() < numel:
            size = numel if buf is None or buf.dtype != like.dtype else max(numel, 2 * buf.numel())
            buf = torch.empty(size, dtype=like.dtype, device=self._device)
            self._staging = buf
        return buf[:numel].view(rows, frames, int(like.size(2)))

    def _record(
        self, tag: Hashable, run: Callable[[], Tuple[torch.Tensor, ...]]
    ) -> Optional[Tuple["torch.cuda.CUDAGraph", Tuple[torch.Tensor, ...]]]:
        """Warm ``run`` up, check the memory bound, then capture it."""
        oom: Optional[str] = None
        try:
            need = self._warmup_bytes(run)
            if not self._fits(tag, need):
                return None
            reserved = int(torch.cuda.memory_reserved(self._device))
            graph, outputs = self._capture(run)
            grown = max(0, int(torch.cuda.memory_reserved(self._device)) - reserved)
        except torch.cuda.OutOfMemoryError as exc:
            # Text only: the traceback holds the failed graph and its tensors.
            oom = str(exc)
        except Exception as exc:  # pragma: no cover - capture is best-effort
            self._failed.add(tag)
            self.capture_failures += 1
            torch.cuda.synchronize(self._device)
            recover_from_failed_capture(self._device, self._pool)
            self._pool = torch.cuda.graph_pool_handle()
            logger.warning("NAR decode graph capture failed for %s (%s); it runs eager", tag, exc)
            if self.captures == 0:
                # Nothing has ever captured: the forward itself is not capturable
                # and every other shape would fail the same way.
                self._disabled = True
                logger.warning("NAR decode graphs are off for this engine")
            return None
        if oom is not None:
            self._disable_after_oom(tag, oom)
            return None
        self.captures += 1
        self.pool_bytes += grown
        logger.debug(
            "captured NAR decode graph %s (+%.1f MiB, %d live)",
            tag,
            grown / float(1024**2),
            self.num_captured + 1,
        )
        return graph, outputs

    def _capture(
        self, run: Callable[[], Tuple[torch.Tensor, ...]]
    ) -> Tuple["torch.cuda.CUDAGraph", Tuple[torch.Tensor, ...]]:
        """One capture into the shared pool, in its own frame so a failure's
        traceback is the only thing holding the graph."""
        graph = torch.cuda.CUDAGraph()
        # torch types the pool handle as an opaque ``_POOL_HANDLE``.
        ctx = torch.cuda.graph(graph, pool=self._pool)  # type: ignore[arg-type]
        with capture_guard(), tvm_ffi.use_torch_stream(ctx):
            outputs = run()
        torch.cuda.synchronize(self._device)
        return graph, tuple(outputs)

    def _warmup_bytes(self, run: Callable[[], Tuple[torch.Tensor, ...]]) -> int:
        """Run once eagerly from an emptied cache; the bytes it took from the driver.

        Warms library workspaces and JIT builds before the capture, and measures
        what the capture is about to add to the pool (see
        :meth:`GraphedOfflineForward._forward_bytes`).
        """
        torch.cuda.synchronize(self._device)
        torch.cuda.empty_cache()
        before = int(torch.cuda.memory_reserved(self._device))
        run()
        torch.cuda.synchronize(self._device)
        return max(0, int(torch.cuda.memory_reserved(self._device)) - before)

    def _fits(self, tag: Hashable, need: int) -> bool:
        """Keep a capture only if the eager path keeps room to run afterwards."""
        self._largest_bytes = max(self._largest_bytes, int(need))
        reserve = max(
            MIN_ACTIVATION_RESERVE_BYTES, int(self._eager_reserve_factor * self._largest_bytes)
        )
        free, _total = read_device_memory(self._device)
        if free - need >= reserve:
            return True
        self._refused[tag] = time.monotonic()
        self.fallback_budget += 1
        log = logger.debug if self._logged_budget else logger.info
        self._logged_budget = True
        gib = float(1024**3)
        log(
            "NAR decode graph %s would leave %.2f GiB free, below the %.2f GiB kept for "
            "eager decodes; it runs eager and is measured again in %.0f s",
            tag,
            (free - need) / gib,
            reserve / gib,
            BUDGET_RETRY_S,
        )
        return False

    def _disable_after_oom(self, tag: Hashable, reason: str) -> None:
        """Out of memory while capturing: turn the cache off and give its memory back."""
        self._disabled = True
        recover_from_failed_capture(self._device, self._pool)
        self._pool = None
        live = self.num_captured
        self.release()
        gc.collect()
        torch.cuda.empty_cache()
        logger.warning(
            "NAR decode graph capture ran out of memory at %s (%s); NAR decode graphs are "
            "off for this engine and the %d already captured are released",
            tag,
            reason,
            live,
        )
