# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA-Graph cache for the **offline** encoder forward.

The streaming encoder has been graph-captured since ``graph_cache.py``; the
offline one never was.  Profiling put a number on the gap: at ``B=1`` one
offline forward issues ~437 kernel launches and ~36 ``cudaMemcpyAsync`` for
**0.99 ms** of GPU work, and takes 9.69 ms of wall time.  Roughly 2.4 ms of that
is the driver's per-launch cost and ~7.3 ms is Python plus torch dispatch, so
the encoder is host-issue-bound: eager wall time stays flat at ~9-10 ms from
``B=1`` to ``B=32`` because the host, not the GPU, sets the pace.  Replacing the
launch storm with one ``cudaGraphLaunch`` removes both halves at once.

Why this and not a native runtime
---------------------------------
Rewriting the orchestration in C++/Rust removes the ~7.3 ms of Python but still
pays the ~2.4 ms of launch API, one call at a time -- a **4.0x** ceiling.
Capture removes both and measured **5.40x** (Conformer) / **7.35x** (Zipformer)
end-to-end at ``B=1``, with byte-identical transcripts.  That is why this file
exists instead of a rewrite.

The shape problem, and what it actually turned out to be
--------------------------------------------------------
A streaming graph is keyed on ``(B_active, cache_t1_bucket)`` and both axes are
naturally small -- the chunk width is fixed by config and the cache length grows
in known steps.  Offline has neither property, so the naive key ``(B, T)`` looks
unbounded on both axes.  Measured on 200 mixed-length utterances at
``max_batch_size=32``, it is not: **3 captured shapes, 2 distinct B and 2
distinct T.**  The scheduler length-sorts and bucket-fills before the forward
ever sees a batch, so a micro-batch's maximum length is highly repetitive, and
``max_captures=64`` is ~20x more headroom than the traffic asks for.

That reframes the three levers, two of them against expectation:

``preferred_batch_size`` (the B axis) -- **not the lever it looks like.**
    The scheduler does already snap every offline micro-batch to one of these
    widths (:class:`~oasr.engine.batching.partition.CountPartition`), so
    :func:`resolve_batch_buckets` prefers it and B-padding is then zero by
    construction.  But it does not *reduce* anything: the captured key set is
    identical with it and without it, because the B axis was already 2 wide.
    And a single preferred value is a trap unrelated to graphs --
    ``preferred_batch_size=[32]`` measured **3.4x slower end-to-end**, because
    the sub-32 tail batch is held until ``max_wait_time`` expires on every queue
    drain.  Lowering ``max_wait_time`` recovers it with the same key set.  Set
    this for admission shaping; do not reach for it to control capture.

Frame bucketing (the T axis) -- **the only real lever, and nearly free.**
    ``T`` is rounded up to ``frame_granularity`` (default 64, the FMHA
    ``N_BLOCK`` tile, matching :func:`~oasr.engine.graph_cache.round_up_bucket`).
    Coarser means fewer captures and more padded compute -- but sweeping the
    granularity over a 32x range at ``B=1`` moved throughput by less than
    run-to-run noise (7.21-7.35x) while the shape count fell 50 -> 4.  Padding is
    close to free exactly where capture pays, because the graphed path at small
    ``B`` is *still* not GPU-bound.  It stops being free at wide batches, which
    is also where capture stops paying at all.
    :attr:`GraphedOfflineForward.pad_overhead` reports what it cost rather than
    leaving it assumed.

Sequence packing -- **enlarges the space, and is capturable but not wired.**
    The intuition is that packing concatenates utterances into one gapless row,
    pinning ``B`` at 1 and bounding ``T`` by ``max_packed_frames``: one graph for
    the whole service.  Measured, that is backwards.  A packed key is
    ``(S, T_total, max_seg, bias_size)``, and ``T_total`` is a *sum* of segment
    lengths while ``bias_size`` is a sum of *squares*, so both scatter -- the same
    200 utterances that need **3** keys unpacked need **6** packed, with every
    packed row a unique length.

    It is, however, capturable: see
    :func:`~oasr.models.conformer.packing.build_packed_layout_device`, which
    rebuilds the layout with no host round-trip and no data-dependent shape and is
    verified field-identical to the host builder.  Fully captured on realistic
    batches it beats this cache's non-packed path by **13-15% at B=32** and
    **7-13% at B=64** -- exactly the widths where capture alone does least (1.33x
    and 0.977x) -- while *losing* ~10% at B=8.  Wiring it needs a packed-specific
    4-axis key here plus bucket padding of the packed row, and that padding is not
    numerically free: past a kernel-selection threshold it moves the real segments
    by ~1.9e-1 in bf16, so a bucketed packed forward is bit-exact to itself rather
    than to ``B=1``.  Until that is wired, packed batches run eager here and are
    counted under ``fallback_failed``.

So the shipped answer is: the space was small to begin with, frame granularity
is what bounds it, packing would make it larger rather than smaller, and
everything that misses falls back to eager and is counted -- never silently.

Capture constraints
-------------------
* Input buffers are allocated per shape and never reallocated -- the graph
  captures their addresses.  The caller's features are copied in.
* The region outside ``[:B_active, :T]`` is zeroed on every replay.  Stale
  activations there are not merely discarded output: a NaN parked in a padded
  row's ``V`` can reach a *valid* row through the attention epilogue.
* Outputs are **cloned** before returning.  Captures share one memory pool, so a
  returned buffer is invalidated by the next replay at the same key *and* by the
  next capture at any key -- and an AR family holds the encoder hidden as
  cross-attention memory for the whole decode, many ticks later.
* A capture that fails is remembered and never retried.  Retrying costs a
  warm-up forward every call and then runs eager anyway, which is strictly worse
  than never having tried (the ``DecoderStepGraphCache`` lesson).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import tvm_ffi

if TYPE_CHECKING:
    from .config import EngineConfig

logger = logging.getLogger(__name__)

#: Time-axis bucket granularity, matching the FMHA ``N_BLOCK`` tile that the
#: streaming cache already rounds its cache length to.
DEFAULT_FRAME_GRANULARITY = 64

#: Forward entry points this cache can capture, by the name callers key on.
FUSED = "fused"  # encoder + CTC head  -> (B, T, V) log-probs
ENCODE = "encode"  # encoder only        -> (B, T, D) hidden

ForwardFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def resolve_batch_buckets(config: "EngineConfig") -> List[int]:
    """Batch widths to capture at, in ascending order.

    Precedence, and the reason for it:

    1. ``offline_graph_batch_buckets`` -- an explicit deployment override.
    2. ``preferred_batch_size`` -- the widths the partitioner already emits, so
       capturing here costs **no** B-padding at all.  This is why a deployment
       that wants a small, predictable capture set should set
       ``preferred_batch_size`` rather than tuning this cache directly: it
       constrains the scheduler and the graph cache with one number, and keeps
       them in agreement by construction.
    3. Powers of two up to ``max_batch_size`` -- the fallback when admission is
       unconstrained, mirroring :class:`GraphedFeatureExtraction`.  Padding a
       partial batch up to the next power of two is nearly free at small ``B``
       (the encoder is host-bound there, so the extra rows ride along in launches
       that were already being paid for) and is why the ladder is acceptable at
       all.
    """
    explicit = getattr(config, "offline_graph_batch_buckets", None)
    if explicit:
        return sorted({int(b) for b in explicit if int(b) >= 1})
    preferred = getattr(config, "preferred_batch_size", None)
    if preferred:
        return sorted({int(b) for b in preferred if int(b) >= 1})
    cap = max(1, int(config.max_batch_size))
    buckets: List[int] = []
    b = 1
    while b < cap:
        buckets.append(b)
        b *= 2
    buckets.append(cap)
    return sorted(set(buckets))


@dataclass
class _Captured:
    """One captured graph plus the static buffers its addresses are baked to."""

    graph: "torch.cuda.CUDAGraph"
    features: torch.Tensor  # (B_bucket, T_bucket, F)
    lengths: torch.Tensor  # (B_bucket,)
    out: torch.Tensor  # (B_bucket, T_out, V|D)
    out_lengths: torch.Tensor  # (B_bucket,)


class GraphedOfflineForward:
    """Lazy, shape-bucketed CUDA-Graph cache for the offline encoder forward.

    One instance serves both offline entry points -- the fused encoder+head
    (``FUSED``, what CTC families consume) and the encoder-only forward
    (``ENCODE``, what transducer / AED / LLM families consume) -- keyed apart by
    name, because a checkpoint served under a rescoring family runs both.

    Parameters
    ----------
    device : torch.device
        Device the model lives on.  A non-CUDA device disables the cache.
    batch_buckets : sequence of int
        Widths to capture at; see :func:`resolve_batch_buckets`.
    frame_granularity : int
        Time-axis rounding.  Larger means fewer captures and more padded compute.
    max_frames : int
        Refuse to capture above this padded width.  Long-form outliers are both
        rare (so a capture rarely pays for itself) and the regime where the GPU
        is genuinely busy (so there is little host overhead left to remove).
    max_captures : int
        Cap on live captures, bounding graph-pool growth.  Once saturated, new
        shapes run eager and are counted.
    pool : tuple of int, optional
        Shared graph memory-pool handle.  Shared *within* this cache across shape
        buckets -- that is where the fragmentation win is -- but never with the
        streaming or feature caches, which cost silent output aliasing once.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        batch_buckets: Sequence[int],
        frame_granularity: int = DEFAULT_FRAME_GRANULARITY,
        max_frames: int = 4096,
        max_captures: int = 64,
        pool: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._device = device
        self._buckets = sorted({int(b) for b in batch_buckets if int(b) >= 1})
        self._granularity = max(1, int(frame_granularity))
        self._max_frames = int(max_frames)
        self._max_captures = int(max_captures)
        self._enabled = device.type == "cuda" and bool(self._buckets)
        # One pool for this cache, shared across its shape buckets -- that is
        # where the fragmentation win is -- and never with the streaming or
        # feature caches, which once put two families' buffers at one address.
        self._pool: Optional[Tuple[int, int]] = pool
        if self._enabled and self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        self._captured: Dict[Tuple[str, int, int], _Captured] = {}
        self._failed: Set[Tuple[str, int, int]] = set()
        self._disabled = False
        self._refused = False

        # Accounting.  A cache whose fallbacks are invisible is a cache that
        # silently stops working; every miss below has a named counter.
        self.hits = 0
        self.captures = 0
        self.fallback_oversized = 0  # B or T past the captured envelope
        self.fallback_saturated = 0  # capture budget spent
        self.fallback_failed = 0  # this shape is not capturable
        self._useful_frames = 0
        self._padded_frames = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._disabled

    @property
    def batch_buckets(self) -> List[int]:
        return list(self._buckets)

    @property
    def num_captured(self) -> int:
        return len(self._captured)

    @property
    def pad_overhead(self) -> float:
        """Padded frame-rows issued per useful frame-row, or ``0.0`` if unused.

        ``1.0`` means the buckets fit the traffic exactly.  ``1.2`` means 20% of
        the encoder's work went into padding introduced *by bucketing* -- the
        number that decides whether the frame granularity is set too coarse, and
        the reason the trade-off above is reported rather than assumed.
        """
        if self._useful_frames == 0:
            return 0.0
        return self._padded_frames / self._useful_frames

    def stats(self) -> Dict[str, float]:
        """Counters for the engine's debug log and the benchmark harness."""
        return {
            "captured_shapes": float(len(self._captured)),
            "captures": float(self.captures),
            "hits": float(self.hits),
            "fallback_oversized": float(self.fallback_oversized),
            "fallback_saturated": float(self.fallback_saturated),
            "fallback_failed": float(self.fallback_failed),
            "pad_overhead": self.pad_overhead,
        }

    # ------------------------------------------------------------------
    # Shape resolution
    # ------------------------------------------------------------------

    def pick_batch_bucket(self, batch: int) -> Optional[int]:
        """Smallest captured width ``>= batch``, or ``None`` when oversized."""
        if batch < 1:
            return None
        for b in self._buckets:
            if b >= batch:
                return b
        return None

    def frame_bucket(self, frames: int) -> int:
        g = self._granularity
        return -(-max(1, int(frames)) // g) * g

    @torch.no_grad()
    def pad_time(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-pad the time axis out to its bucket, for **both** execution paths.

        This is not an optimisation, it is the correctness half of bucketing.

        An encoder is not obliged to be invariant to how much padding trails its
        valid frames, and the two shipped ones are not: Zipformer's
        ``SimpleDownsample`` fills its last window by *replicating the final
        frame*, so the padded width reaches the valid outputs, and a Conformer
        forward moves too once the padding crosses an attention tile.  Measured
        on real weights in bf16, the leak is ~2.5e-1 -- three orders above
        rounding.

        The consequence, if only the captured path padded to a bucket: an
        utterance's transcript would depend on whether *its shape happened to be
        captured*, and a fallback to eager -- a saturated cache, an oversized
        batch -- would silently decode it differently.  That is the same family
        of defect as rules 10/11/13: plausible output, no error, no test failure.

        Applying the same padding on both paths makes them bit-identical, and is
        strictly more reproducible than the status quo it replaces: the eager
        width was ``max(lengths)`` over whatever micro-batch the scheduler
        happened to assemble, so it already moved with batch composition under
        load.  After this it is ``ceil(max_len / granularity) * granularity`` --
        a function of the utterance alone, up to one bucket.

        ``lengths`` is returned unchanged; the valid extent is unaffected.
        """
        if not self.enabled:
            return features, lengths
        t_active = int(features.size(1))
        t_bucket = self.frame_bucket(t_active)
        if t_bucket == t_active or t_bucket > self._max_frames:
            return features, lengths
        padded = features.new_zeros((features.size(0), t_bucket, features.size(2)))
        padded[:, :t_active].copy_(features)
        return padded, lengths

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        name: str,
        fn: ForwardFn,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Replay (or first-capture) ``fn`` at this batch's bucketed shape.

        Returns ``None`` when the shape is not served by the cache, in which case
        the caller runs ``fn`` eagerly -- every such return is counted under one
        of the ``fallback_*`` attributes, never silent.

        The returned tensors are **clones**, safe to hold across later replays
        and captures.
        """
        if not self.enabled:
            return None

        b_active, t_active = int(features.size(0)), int(features.size(1))
        b_bucket = self.pick_batch_bucket(b_active)
        t_bucket = self.frame_bucket(t_active)
        if b_bucket is None or t_bucket > self._max_frames:
            self.fallback_oversized += 1
            return None

        key = (name, b_bucket, t_bucket)
        state = self._captured.get(key)
        if state is None:
            if key in self._failed:
                self.fallback_failed += 1
                return None
            if len(self._captured) >= self._max_captures:
                self.fallback_saturated += 1
                if not self._refused:
                    self._refused = True
                    logger.info(
                        "offline forward graph cache full (%d shapes); further " "shapes run eager",
                        self._max_captures,
                    )
                return None
            state = self._capture(key, fn, features, lengths)
            if state is None:
                self.fallback_failed += 1
                return None
            self._captured[key] = state

        # Refill the static inputs.  Everything outside the live window is
        # zeroed: a padded row is not merely discarded output, because a stale
        # NaN in its ``V`` reaches valid rows through the attention epilogue.
        state.features[:b_active, :t_active].copy_(features)
        if t_active < t_bucket:
            state.features[:b_active, t_active:].zero_()
        if b_active < b_bucket:
            state.features[b_active:].zero_()
        state.lengths[:b_active].copy_(lengths)
        if b_active < b_bucket:
            # One frame, not zero: a length-0 row divides by its own frame count
            # in any mean-over-valid-frames op, and masks its every K position.
            state.lengths[b_active:].fill_(1)

        state.graph.replay()
        self.hits += 1
        self._useful_frames += b_active * t_active
        self._padded_frames += b_bucket * t_bucket
        return state.out[:b_active].clone(), state.out_lengths[:b_active].clone()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _capture(
        self,
        key: Tuple[str, int, int],
        fn: ForwardFn,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Optional[_Captured]:
        name, b_bucket, t_bucket = key
        feat_dim = int(features.size(2))
        feats_buf = torch.zeros(
            b_bucket, t_bucket, feat_dim, dtype=features.dtype, device=self._device
        )
        # Capture at the full bucket width so the warm-up hits the same kernel
        # tiles the replays will.  Real lengths arrive with the first replay.
        lens_buf = torch.full((b_bucket,), t_bucket, dtype=lengths.dtype, device=self._device)
        feats_buf[: features.size(0), : features.size(1)].copy_(features)

        try:
            fn(feats_buf, lens_buf)  # warm up workspaces outside the capture
            torch.cuda.synchronize(self._device)
            graph = torch.cuda.CUDAGraph()
            # ``tvm_ffi.use_torch_stream`` is what records a TVM-FFI kernel
            # launch into the graph rather than letting it escape to the
            # default stream.
            # torch types the pool handle as an opaque ``_POOL_HANDLE``; the
            # engine passes the ``(int, int)`` tuple it actually is.
            ctx = torch.cuda.graph(graph, pool=self._pool)  # type: ignore[arg-type]
            with tvm_ffi.use_torch_stream(ctx):
                out, out_lengths = fn(feats_buf, lens_buf)
            torch.cuda.synchronize(self._device)
        except torch.cuda.OutOfMemoryError as exc:
            # Capture OOM is a fact about the process, not about this shape.
            self._disabled = True
            torch.cuda.empty_cache()
            logger.warning(
                "offline forward graph capture ran out of memory at %s B=%d T=%d "
                "(%s); offline graphs are off for this engine and forwards run eager",
                name,
                b_bucket,
                t_bucket,
                exc,
            )
            return None
        except Exception as exc:  # pragma: no cover - capture is best-effort
            # Most often a host read inside the forward (``.item()`` /
            # ``.tolist()``) invalidating the capture stream: a data-dependent
            # output extent is not capturable at a fixed shape, and the honest
            # response is to run it eagerly and say so.
            self._failed.add(key)
            torch.cuda.synchronize(self._device)
            logger.warning(
                "offline forward graph capture failed for %s B=%d T=%d (%s); "
                "this shape runs eager",
                name,
                b_bucket,
                t_bucket,
                exc,
            )
            return None

        self.captures += 1
        logger.info(
            "captured offline forward: %s B=%d T=%d (%d shapes live)",
            name,
            b_bucket,
            t_bucket,
            len(self._captured) + 1,
        )
        return _Captured(
            graph=graph,
            features=feats_buf,
            lengths=lens_buf,
            out=out,
            out_lengths=out_lengths,
        )

    # ------------------------------------------------------------------
    # Pre-warm
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prewarm(
        self,
        name: str,
        fn: ForwardFn,
        feat_dim: int,
        dtype: torch.dtype,
        shapes: Sequence[Tuple[int, int]],
    ) -> int:
        """Capture ``(B, T)`` shapes ahead of traffic; returns the count taken.

        Inline capture costs a warm-up forward plus the capture itself, which
        lands on whichever request happens to arrive first -- in streaming that
        is measurable as a p99 tail well above p50.  Offline absorbs it into one
        slow batch instead, but a latency-sensitive deployment can pay it at
        construction.
        """
        if not self.enabled:
            return 0
        taken = 0
        for b, t in shapes:
            b_bucket = self.pick_batch_bucket(int(b))
            t_bucket = self.frame_bucket(int(t))
            if b_bucket is None or t_bucket > self._max_frames:
                continue
            key = (name, b_bucket, t_bucket)
            if key in self._captured or key in self._failed:
                continue
            if len(self._captured) >= self._max_captures:
                break
            probe = torch.zeros(b_bucket, t_bucket, feat_dim, dtype=dtype, device=self._device)
            lens = torch.full((b_bucket,), t_bucket, dtype=torch.int32, device=self._device)
            state = self._capture(key, fn, probe, lens)
            del probe, lens
            if state is None:
                continue
            self._captured[key] = state
            taken += 1
        return taken


__all__ = [
    "DEFAULT_FRAME_GRANULARITY",
    "ENCODE",
    "FUSED",
    "GraphedOfflineForward",
    "resolve_batch_buckets",
]
