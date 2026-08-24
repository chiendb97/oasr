#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark registered GEMM candidates for representative ASR shapes.

Shapes come from a captured workload or the analytic fallback, are bucketed by
work weight, and produce selection rules. Candidate timings hide dispatch cost,
so near ties require end-to-end validation and ``--min-speedup`` filtering.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# Runtime M uses the first SM120-aligned edge at or above it; the last edge is a
# catch-all. High edges keep large fixed-window batches in distinct tune buckets.
_M_LADDER = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

_ACTIVATION_SWISH = 2


# ─────────────────────────────────────────────────────────────────────────────
# Representative-shape extraction
# ─────────────────────────────────────────────────────────────────────────────


class RepShape:
    """One representative shape to benchmark."""

    __slots__ = ("op", "M", "N", "K", "dtype", "batch", "m_max", "weight")

    def __init__(self, op, M, N, K, dtype, batch, m_max, weight):
        self.op = op
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch = batch
        self.m_max = m_max  # ladder edge this bucket covers; None = catch-all
        self.weight = weight

    def __repr__(self):
        return (
            f"RepShape({self.op} M={self.M} N={self.N} K={self.K} "
            f"{self.dtype} batch={self.batch} m_max={self.m_max})"
        )


def _ladder_edge(m: int) -> int:
    for e in _M_LADDER:
        if m <= e:
            return e
    return _M_LADDER[-1]


def _weighted_median(counter: Counter) -> int:
    """Call-count-weighted median of observed M values."""
    items = sorted(counter.items())
    total = sum(c for _, c in items)
    acc = 0
    for m, c in items:
        acc += c
        if acc * 2 >= total:
            return int(m)
    return int(items[-1][0])


def buckets_from_stats(stats, coverage: float) -> List[RepShape]:
    """Bucket per-(op,N,K,dtype,batch) shape stats into representative shapes.

    *stats* is a list of ``_ShapeStat`` (from oasr.tune.capture) or dicts with the
    same fields.  Within each group, observed M values are snapped to the ladder;
    each bucket's representative M is the call-count-weighted median of the M's
    that fell into it.  Buckets are kept (per group) in descending FLOP order
    until cumulative coverage of the group's FLOPs reaches *coverage*.
    """
    reps: List[RepShape] = []
    for st in stats:
        op = st.op if hasattr(st, "op") else st["op"]
        N = st.N if hasattr(st, "N") else st["N"]
        K = st.K if hasattr(st, "K") else st["K"]
        dtype = st.dtype if hasattr(st, "dtype") else st["dtype"]
        batch = st.batch if hasattr(st, "batch") else st.get("batch", 1)
        m_counts = (
            st.m_counts
            if hasattr(st, "m_counts")
            else Counter({int(m): c for m, c in st["m_counts"].items()})
        )

        # Snap observed M into ladder buckets.
        per_edge_counts: Dict[int, Counter] = defaultdict(Counter)
        per_edge_flops: Dict[int, float] = defaultdict(float)
        per_edge_calls: Dict[int, int] = defaultdict(int)
        for m, c in m_counts.items():
            edge = _ladder_edge(int(m))
            per_edge_counts[edge][int(m)] += c
            per_edge_flops[edge] += 2.0 * int(m) * N * K * batch * c
            per_edge_calls[edge] += c

        # Keep buckets covering *coverage* of FLOPs (offline, large-M dominated)
        # UNION buckets covering *coverage* of call-count (streaming, small-M but
        # frequent — where the fixed-tile default is most wasteful).  Without the
        # call-count arm, FLOP weighting alone would prune exactly the high-value
        # small-M streaming shapes.
        def _cover(weights: Dict[int, float]) -> set:
            total = sum(weights.values()) or 1.0
            kept_, acc_ = set(), 0.0
            for edge in sorted(weights, key=weights.get, reverse=True):
                kept_.add(edge)
                acc_ += weights[edge]
                if acc_ / total >= coverage:
                    break
            return kept_

        kept = sorted(_cover(per_edge_flops) | _cover(dict(per_edge_calls)))

        for edge in sorted(kept):
            rep_m = _weighted_median(per_edge_counts[edge])
            reps.append(RepShape(op, rep_m, N, K, dtype, batch, edge, per_edge_flops[edge]))
    # Mark the largest kept bucket per (op,N,K,dtype,batch) as the catch-all.
    by_group: Dict[Tuple, List[RepShape]] = defaultdict(list)
    for r in reps:
        by_group[(r.op, r.N, r.K, r.dtype, r.batch)].append(r)
    for grp in by_group.values():
        grp.sort(key=lambda r: r.m_max)
        grp[-1].m_max = None  # catch-all for large/unseen M
    return reps


def shapes_from_capture(path: str, coverage: float) -> List[RepShape]:
    from oasr.tune.capture import GemmShapeRecorder

    stats = GemmShapeRecorder.load_json(path)
    return buckets_from_stats(stats, coverage)


def shapes_from_analytic(families, sizes, batches, durations, dtype, coverage):
    """Analytic fallback — reuse derive_problems, keep only OASR-path ops."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_asr_cutlass_configs import MODEL_REGISTRY, derive_problems

    from oasr.tune.capture import GemmShapeRecorder

    rec = GemmShapeRecorder()
    for fam in families:
        for size in sizes:
            spec = MODEL_REGISTRY.get((fam, size))
            if spec is None:
                continue
            for b in batches:
                for dur in durations:
                    gemms, _ = derive_problems(spec, b, dur)
                    for p in gemms:
                        op = _oasr_op_for(p.op_name)
                        if op is None:
                            continue  # not an OASR GEMM-path op (attn/ctc/etc.)
                        if p.batch != 1:
                            continue  # batched attn goes through fmha, skip
                        rec.record(op, p.M, p.N, p.K, dtype, 1)
    return buckets_from_stats(rec.aggregate(), coverage)


def _oasr_op_for(op_name: str) -> Optional[str]:
    """Map an analytic op_name to the OASR functional op, or None if off-path."""
    if op_name.endswith("_expand"):
        return "gemm_activation"  # FF/conv expand fuse swish
    if op_name.endswith("_contract"):
        return "gemm"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Real benchmarking over the registered candidate set
# ─────────────────────────────────────────────────────────────────────────────


#: Calls captured per graph for the loop measurement.  A graph launch costs about
#: 2.6 us, so at 64 calls it adds ~0.04 us to each -- about 1% of the fastest arm
#: seen at ASR sizes (3.6 us) and less for everything slower.
_GRAPH_ITERS = 64

#: Event-timed replays per arm.  The estimator is the *minimum*: on an idle GPU
#: noise is additive, so the minimum is the cleanest estimate of the kernel and
#: the median only adds whatever else the box was doing.
_GRAPH_REPS = 7

#: One graph memory pool for every capture in the sweep.
#:
#: A ``CUDAGraph`` with no pool gets its own, and the pool is not returned
#: promptly when the graph is dropped; a sweep captures twice per candidate,
#: ~30 candidates per shape, hundreds of shapes.  Sharing is safe here because
#: captures are strictly sequential: each graph is captured, replayed and dropped
#: before the next is captured.
#:
#: Fewer pools is strictly better, but this is not what made the first sweeps run
#: out of memory -- a control run that captured and dropped 40 graphs per config
#: held reserved memory flat at 22.0 MiB across every variation.  The 30 GiB was
#: the workspace cache; see ``_side_stream``.
_POOL = None


def _graph_pool():
    global _POOL
    if _POOL is None:
        import torch

        _POOL = torch.cuda.graph_pool_handle()
    return _POOL


#: One side stream for every warm-up in the sweep.
#:
#: Not one per capture.  OASR's split-K / Stream-K workspace cache
#: (``include/oasr/common/workspace_cache.h``) is keyed on ``(device, stream,
#: pool)`` and never frees, so a stream per capture spread the sweep's workspaces
#: over every key the cache could hold and kept all of them: 30 GiB of
#: non-PyTorch memory over a 121-shape sweep, dying on a 66 MiB allocation while
#: PyTorch itself held 1 GiB.
#:
#: Note what the key count is NOT: ``torch.cuda.Stream()`` hands out one of a
#: POOL of 32 handles per device and cycles, so "a stream per capture" never made
#: more than 32 keys.  What grew was the bytes behind each -- a parallel split-K
#: workspace is ``M*N*4*split``, so one 4096x5008 shape is 328 MiB per key.  The
#: cache now bounds those bytes, but the tuner still should not be spreading its
#: workspaces over 32 keys to begin with.
_SIDE_STREAM = None


def _side_stream():
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        import torch

        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


def _capture(fn, iters: int):
    """Capture ``iters`` back-to-back calls into one CUDA graph.

    The warm-up runs on a side stream first because a capture on the default
    stream inherits whatever the allocator did on it, and PyTorch requires the
    side-stream warm-up before ``torch.cuda.graph``.
    """
    import torch

    side = _side_stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=_graph_pool()):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def _replay_ms(graph, iters: int) -> float:
    import torch

    samples = []
    for _ in range(_GRAPH_REPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) / iters)
    return float(min(samples))


def _bench(fn, warmup: int, rep: int) -> Tuple[float, float]:
    """``(loop_ms, solo_ms)`` per call, both measured through CUDA graphs.

    Two numbers, because neither alone is trustworthy at these sizes.

    ``loop_ms`` -- ``_GRAPH_ITERS`` calls captured in one graph.  This is the
    number to rank on: it charges no host issue cost and keeps L2 in the state a
    real inner loop leaves it in.

    ``solo_ms`` -- one call per graph replay.  Back-to-back *independent* launches
    can overlap on the GPU, which flatters a low-occupancy configuration that
    would never overlap inside a real layer (a thin tile at small M is 20 CTAs on
    170 SMs).  A single-call replay cannot overlap with itself.  Every arm carries
    the same graph-launch constant here, which is additive: it compresses the
    ratio (2.6 us against a 3.6 us kernel is most of the number) so this must not
    be *ranked* on, but adding a constant to both sides cannot flip which is
    larger -- so the *sign* is trustworthy, and that is all the gate needs.  A
    configuration that beats the fallback on ``loop_ms`` and loses on ``solo_ms``
    won only by self-overlap, and :func:`emit_rules` refuses it.

    What this replaces: ``triton.do_bench``, an eager back-to-back loop that also
    wipes L2 between iterations.  Every arm at ASR sizes is faster than the
    ~9.6 us it costs to *issue* a GEMM call, so that loop read 10-20 us for all of
    them -- it reported 2.00x where the truth was 4.6x on the LSTM gate projection
    and picked a tile 22% slower than the best.
    """
    import gc

    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = _capture(fn, _GRAPH_ITERS)
    loop_ms = _replay_ms(graph, _GRAPH_ITERS)
    # Drop it before capturing the next: the shared pool is only reusable when at
    # most one graph is holding it.
    del graph
    gc.collect()
    graph = _capture(fn, 1)
    solo_ms = _replay_ms(graph, 1)
    del graph
    gc.collect()
    return loop_ms, solo_ms


def _alloc_args(shape: RepShape):
    import torch

    dt = getattr(torch, shape.dtype)
    dev = "cuda"
    M, N, K, batch = shape.M, shape.N, shape.K, shape.batch
    if shape.op == "bmm":
        A = torch.randn(batch, M, K, device=dev, dtype=dt)
        B = torch.randn(batch, N, K, device=dev, dtype=dt)
        out = torch.empty(batch, M, N, device=dev, dtype=dt)
        return out, A, B
    A = torch.randn(M, K, device=dev, dtype=dt)
    B = torch.randn(N, K, device=dev, dtype=dt)
    C = torch.randn(N, device=dev, dtype=dt)
    out = torch.empty(M, N, device=dev, dtype=dt)
    if shape.op == "gemm_activation":
        return out, A, B, C, _ACTIVATION_SWISH
    # gemm and gemm_log_softmax share the (out, A, B, C) runner signature
    return out, A, B, C


def _reference_output(shape: RepShape, args):
    """fp32 reference for the numerics guard (None → no check for this op)."""
    import torch
    import torch.nn.functional as F

    if shape.op == "bmm":
        out, A, B = args
        return torch.matmul(A.float(), B.float().transpose(-1, -2))
    out, A, B, C = args[:4]
    ref = torch.addmm(C.float(), A.float(), B.float().t())
    if shape.op == "gemm_activation":
        return F.silu(ref)
    if shape.op == "gemm_log_softmax":
        return F.log_softmax(ref, dim=-1)
    return ref


class TacticResult:
    """One tactic's timings.  ``median_ms`` is the loop measurement (the ranking
    number); ``solo_ms`` is the single-replay one (see :func:`_bench`)."""

    def __init__(self, tactic, median_ms, is_default, solo_ms=None):
        self.tactic = tactic
        self.median_ms = median_ms
        self.is_default = is_default
        self.solo_ms = median_ms if solo_ms is None else solo_ms


def _pick_winner(results: List["TacticResult"], tie_tol: float = 0.05) -> "TacticResult":
    """Pick the winner, breaking near-ties deterministically.

    At small M many tiles bottom out at the same latency floor, so the raw
    fastest flips arbitrarily between equivalent tiles.  Among all tactics
    within ``tie_tol`` of the measured best, prefer structures by launch cost
    (fewer kernels / no per-launch workspace work first) with CUTLASS ahead of
    torch on ties, then the smallest ``(block_m, block_n, kStages, split_k)``
    tile.  Preference order: plain CUTLASS (or the fused launcher), CUTLASS
    serial split-K (single launch), torch, parallel split-K (2 launches),
    Stream-K (memset + kernel).  A backend that is truly faster than the
    tolerance band is always kept.
    """
    best_ms = results[0].median_ms
    band = [r for r in results if r.median_ms <= best_ms * (1.0 + tie_tol)]

    def key(r):
        if r.tactic.backend == "torch":
            return (2, 0, 0, 0, 0)
        if r.tactic.backend == "cutlass_fused":
            return (0, 0, 0, 0, 0)
        c = dict(r.tactic.config)
        if c.get("stream_k", 0):
            rank = 4
        elif c.get("parallel_split_k", 0):
            rank = 3
        elif c.get("split_k", 1) > 1:
            rank = 1
        else:
            rank = 0
        return (
            rank,
            c.get("block_m", 1 << 30),
            c.get("block_n", 1 << 30),
            c.get("kStages", 0),
            c.get("split_k", 1),
        )

    return min(band, key=key)


def benchmark_shape(shape: RepShape, warmup: int, rep: int) -> Optional[List[TacticResult]]:
    import gc

    import torch

    from oasr.tune.autotuner import OpKey, _ensure_backends_registered, _global_registry

    _ensure_backends_registered()
    op_key = OpKey("gemm", shape.op)
    candidates = _global_registry.get_candidates(op_key)
    if not candidates:
        return None
    args = _alloc_args(shape)
    ref = _reference_output(shape, args)

    # Numerics guard: a tactic whose max abs error exceeds 4× the torch
    # backend's own low-precision error is disqualified (e.g. very deep serial
    # split-K accumulates partials in the output dtype).
    torch_err = None
    for entry in candidates:
        if entry.tactic.backend == "torch":
            try:
                entry.get_runner()(*args)
                torch.cuda.synchronize()
                torch_err = (args[0].float() - ref).abs().max().item()
            except Exception:
                pass
            break

    results: List[TacticResult] = []
    for entry in candidates:
        try:
            runner = entry.get_runner()
            runner(*args)
            torch.cuda.synchronize()
            if torch_err is not None:
                err = (args[0].float() - ref).abs().max().item()
                if err > max(4.0 * torch_err, 1e-3):
                    continue  # numerically disqualified
            # Bind `runner` at definition time: it is a loop variable, and the
            # late-binding form only happens to work because _bench calls back
            # synchronously within this iteration.
            ms, solo = _bench(lambda r=runner: r(*args), warmup, rep)
        except Exception:
            ms = solo = float("inf")
        results.append(TacticResult(entry.tactic, ms, entry.is_fallback, solo))
    results.sort(key=lambda r: r.median_ms)
    # Release this shape's operands and whatever the captures pooled before the
    # next shape allocates its own: a full sweep is hundreds of shapes.  Rebound
    # rather than ``del``'d, because the timing lambdas above close over ``args``
    # and unbinding the name makes that a static undefined-name error.
    args = ref = None  # type: ignore[assignment]
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rule emission
# ─────────────────────────────────────────────────────────────────────────────


def _cutlass_literal(tactic, sm: int) -> str:
    d = dict(tactic.config)
    extra = ", stream_k=True" if d.get("stream_k", 0) else ""
    if d.get("parallel_split_k", 0):
        extra += ", parallel_split_k=True"
    return (
        "CutlassGemmConfig("
        f"block_m={d['block_m']}, block_n={d['block_n']}, block_k={d['block_k']}, "
        f"warp_m={d['warp_m']}, warp_n={d['warp_n']}, warp_k={d['warp_k']}, "
        f"kStages={d['kStages']}, kSmVersion={sm}, split_k={d.get('split_k', 1)}{extra})"
    )


def _choice_literal(tactic, sm: int, is_default: bool) -> str:
    if tactic.backend == "torch":
        return '"torch"'
    if tactic.backend == "cutlass_fused":
        return '"fused"'
    if is_default:
        return "GEMM_DEFAULT"
    return _cutlass_literal(tactic, sm)


def emit_rules(
    per_shape: Dict[Tuple, List[TacticResult]],
    reps: List[RepShape],
    sm: int,
    min_speedup: float = 1.05,
) -> str:
    """Build the _GEMM_HEURISTIC_RULES_SM<sm> Python literal from sweep winners.

    A bucket whose winner is not at least *min_speedup* faster than the fallback
    keeps the fallback, so it collapses into a neighbour instead of becoming a
    rule.  Two reasons, both learned from the whisper-tiny run:

    * ``_pick_winner`` can return a tactic **slower than the fallback**.  Its
      tie-break prefers a smaller tile within 5% of the measured best, and the
      fallback is often *in* that band — one emitted bucket read ``0.96x vs
      default``, i.e. a rule that made things worse.
    * every arm is measured once here, so adjacent buckets came back alternating
      torch / cutlass / default at 1.02-1.08x.  Re-measuring those pairs with the
      arms interleaved showed them to be ties.  Encoding a tie costs a compiled
      variant and a boundary that can be wrong, and buys nothing.

    A third gate came from the protocol change (see :func:`_bench`): a win must
    hold on the *single-replay* timing as well as the loop timing.  Back-to-back
    independent launches can overlap on the GPU, which flatters a low-occupancy
    tile that would never overlap inside a real layer, and self-overlap is not a
    speedup a model can spend.
    """
    # Group reps by (op, N, K) and order by m_max (None last).
    by_key: Dict[Tuple[str, int, int], List[RepShape]] = defaultdict(list)
    for r in reps:
        by_key[(r.op, r.N, r.K)].append(r)

    lines = [f"_GEMM_HEURISTIC_RULES_SM{sm}: Dict[Tuple[str, int, int], list] = {{"]
    for (op, N, K), grp in sorted(by_key.items()):
        grp.sort(key=lambda r: (r.m_max is None, r.m_max or 0))
        # The literal a rule-less lookup falls back to (see select_default_config
        # / _dispatch_gemm_log_softmax): the fused launcher for the CTC head,
        # GEMM_DEFAULT for everything else.
        fallback_literal = '"fused"' if op == "gemm_log_softmax" else "GEMM_DEFAULT"
        rule_entries = []  # (m_max | None, choice_literal, comment)
        all_default = True
        for r in grp:
            res = per_shape.get((r.op, r.M, r.N, r.K, r.dtype, r.batch))
            if not res:
                continue
            winner = _pick_winner(res)
            default = next((t for t in res if t.is_default), None)
            speedup = (
                (default.median_ms / winner.median_ms) if default and winner.median_ms > 0 else 1.0
            )
            # The same ratio on the overlap-free timing.  A tile that wins the
            # loop and loses this one won by overlapping with itself.
            solo_speedup = (
                (default.solo_ms / winner.solo_ms)
                if default and winner.solo_ms > 0 and default.solo_ms < float("inf")
                else speedup
            )
            if speedup >= min_speedup and solo_speedup < 1.0:
                print(
                    f"[tune] suppressed ({op}, {N}, {K}) m_max={r.m_max}: "
                    f"{winner.tactic.backend} is {speedup:.2f}x back-to-back but "
                    f"{solo_speedup:.2f}x on a single replay — self-overlap only"
                )
                speedup = solo_speedup
            if speedup < min_speedup:
                # Not a measured win — keep the fallback and say so, rather than
                # emitting a rule that a paired re-measurement would not support.
                print(
                    f"[tune] suppressed ({op}, {N}, {K}) m_max={r.m_max}: "
                    f"{winner.tactic.backend} only {speedup:.2f}x vs default "
                    f"(< {min_speedup:.2f}x) — keeping the fallback"
                )
                rule_entries.append(
                    (
                        r.m_max,
                        fallback_literal,
                        f"M~{r.M}: fallback ({winner.tactic.backend} was only "
                        f"{speedup:.2f}x vs default)",
                    )
                )
                continue
            choice = _choice_literal(winner.tactic, sm, winner.is_default)
            if choice != fallback_literal:
                all_default = False
            rule_entries.append(
                (
                    r.m_max,
                    choice,
                    f"M~{r.M}: {winner.tactic.backend} {winner.median_ms:.4f}ms "
                    f"({speedup:.2f}x vs default)",
                )
            )
        if all_default or not rule_entries:
            continue  # every bucket == the fallback → omit the rule entirely
        # Collapse runs of identical choice (ascending m_max): a later, larger
        # threshold subsumes earlier same-choice buckets.
        merged = []
        for m_max, choice, comment in rule_entries:
            if merged and merged[-1][1] == choice:
                merged[-1] = (m_max, choice, comment)
            else:
                merged.append((m_max, choice, comment))
        lines.append(f'    ("{op}", {N}, {K}): [')
        for m_max, choice, comment in merged:
            m_max_repr = "None" if m_max is None else str(m_max)
            lines.append(f"        ({m_max_repr}, {choice}),  # {comment}")
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def print_report(per_shape, reps, sm) -> None:
    print("\n" + "=" * 108)
    print(
        f"{'op':16} {'N':>6} {'K':>6} {'M':>7} {'m_max':>7}  "
        f"{'winner':>26} {'win ms':>9} {'dflt ms':>9} {'speedup':>8} {'solo':>8}"
    )
    print("-" * 108)
    reps_sorted = sorted(reps, key=lambda r: (r.op, r.N, r.K, r.M))
    for r in reps_sorted:
        res = per_shape.get((r.op, r.M, r.N, r.K, r.dtype, r.batch))
        if not res:
            continue
        w = _pick_winner(res)
        d = next((t for t in res if t.is_default), None)
        sp = (d.median_ms / w.median_ms) if d and w.median_ms > 0 else float("nan")
        if w.tactic.backend == "torch":
            wname = "torch"
        elif w.tactic.backend == "cutlass_fused":
            wname = "fused"
        else:
            _c = dict(w.tactic.config)
            _sk = "sk" if _c.get("stream_k", 0) else ""
            _pk = "pk" if _c.get("parallel_split_k", 0) else ""
            wname = (
                f"cutlass {_c.get('block_m')}x{_c.get('block_n')}"
                f"s{_c.get('kStages')}k{_c.get('split_k')}{_sk}{_pk}"
            )
        mm = "inf" if r.m_max is None else str(r.m_max)
        solo = (d.solo_ms / w.solo_ms) if d and w.solo_ms > 0 else float("nan")
        print(
            f"{r.op:16} {r.N:>6} {r.K:>6} {r.M:>7} {mm:>7}  "
            f"{wname:>26} {w.median_ms:>9.4f} "
            f"{(d.median_ms if d else float('nan')):>9.4f} {sp:>7.2f}x {solo:>7.2f}x"
        )
    print("=" * 92 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["capture", "analytic"], default="capture")
    p.add_argument("--shapes", help="captured shapes JSON (OASR_CAPTURE_GEMM output)")
    p.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES value (index or GPU-UUID). Set before torch import.",
    )
    p.add_argument(
        "--coverage",
        type=float,
        default=0.97,
        help="keep top FLOP buckets until this fraction of group FLOPs is covered",
    )
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument(
        "--emit-rules", metavar="FILE", help="write the _GEMM_HEURISTIC_RULES Python literal here"
    )
    p.add_argument(
        "--min-speedup",
        type=float,
        default=1.05,
        help="emit a rule only when the winner beats the fallback by at least this "
        "factor; suppressed buckets keep the fallback and are logged (default 1.05)",
    )
    # analytic-mode knobs
    p.add_argument("--families", nargs="+", default=["conformer"])
    p.add_argument("--sizes", nargs="+", default=["base"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 64])
    p.add_argument("--durations", nargs="+", type=int, default=[4, 16, 64])
    p.add_argument("--dtype", default="bfloat16")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # must precede torch import

    from oasr.jit.core import _get_target_sm

    sm = _get_target_sm()

    if args.mode == "capture":
        if not args.shapes:
            print(
                "ERROR: --mode capture requires --shapes <captured json> "
                "(produce it via OASR_CAPTURE_GEMM=… python benchmarks/bench_engine.py …)",
                file=sys.stderr,
            )
            return 2
        reps = shapes_from_capture(args.shapes, args.coverage)
    else:
        reps = shapes_from_analytic(
            args.families, args.sizes, args.batches, args.durations, args.dtype, args.coverage
        )

    if not reps:
        print("No representative shapes found.", file=sys.stderr)
        return 1

    print(f"[tune] sm={sm}  representative shapes: {len(reps)}")
    for r in reps:
        print("  ", r)

    # Progress on stderr, unbuffered.  A full sweep is 165 shapes against ~40
    # candidates each and takes the better part of an hour, and every line this
    # script prints used to come from ``print_report`` at the very end: a run that
    # died at shape 140 looked exactly like a run that had hung at shape 2, and
    # diagnosing which cost an hour.
    per_shape: Dict[Tuple, List[TacticResult]] = {}
    t_start = time.perf_counter()
    for i, r in enumerate(reps, 1):
        t_shape = time.perf_counter()
        res = benchmark_shape(r, args.warmup, args.rep)
        if res:
            per_shape[(r.op, r.M, r.N, r.K, r.dtype, r.batch)] = res
        elapsed = time.perf_counter() - t_start
        eta = elapsed / i * (len(reps) - i)
        print(
            f"[tune] {i:3d}/{len(reps)}  {r.op} M={r.M} N={r.N} K={r.K}  "
            f"{len(res) if res else 0} arm(s) in {time.perf_counter() - t_shape:5.1f}s  "
            f"(elapsed {elapsed / 60:5.1f}m, eta {eta / 60:5.1f}m)",
            file=sys.stderr,
            flush=True,
        )

    print_report(per_shape, reps, sm)

    rules = emit_rules(per_shape, reps, sm, args.min_speedup)
    print(rules)
    if args.emit_rules:
        with open(args.emit_rules, "w") as f:
            f.write("# Auto-generated by scripts/tune_asr_gemm.py — paste into oasr/jit/gemm.py\n")
            f.write(rules + "\n")
        print(f"\n[tune] wrote rules to {args.emit_rules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
