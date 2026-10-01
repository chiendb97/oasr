#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark registered GEMM candidates for representative ASR shapes.

Shapes come from a captured workload or the analytic fallback, are bucketed by
work weight, and produce selection rules -- written into the tuning DB
(``--emit-db user`` for this machine, ``--emit-db system`` for the shipped file).
Measurement is ``oasr.tune.bench``'s protocol, shared with ``oasr.autotune()``.
Candidate timings hide dispatch cost, so near ties require end-to-end validation
and ``--min-speedup`` filtering.  ``oasr tune build`` supersedes this for shapes
derived from an engine configuration.
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
#
# The protocol itself lives in ``oasr.tune.bench`` (shared with ``oasr.autotune()``
# and ``oasr tune build``) and the GEMM driver in ``oasr.tune.gemm_tune``: graph-
# captured N-call loops, a single-replay self-overlap gate, operand-copy rotation
# for honest L2 state, interleaved rounds with successive halving, and a numerics
# gate on every candidate.  What used to be here -- the one-shared-graph-pool and
# one-side-stream rules, the 64-call loop, the min-of-7 estimator -- is there now.

#: Lower bounds the protocol guarantees (kept as the script's documented floor;
#: ``BenchPolicy`` holds the values in force).  A one-call capture cannot separate
#: a kernel from its launch, and three rounds is the least a median means anything.
_GRAPH_ITERS = 8
_GRAPH_REPS = 3


class TacticResult:
    """One tactic's timings.  ``median_ms`` is the loop measurement (the ranking
    number); ``solo_ms`` is the single-replay one (see ``oasr.tune.bench``)."""

    def __init__(
        self, tactic, median_ms, is_default, solo_ms=None, sigma_ms=0.0, issue_us=None, name=None
    ):
        self.tactic = tactic
        self.median_ms = median_ms
        self.is_default = is_default
        self.solo_ms = median_ms if solo_ms is None else solo_ms
        self.sigma_ms = sigma_ms
        self.issue_us = issue_us
        self.name = name


def _working_set_bytes(reps: List[RepShape]) -> int:
    """Per-forward weight bytes of the captured model: every distinct (op, N, K) once."""
    seen = {}
    for r in reps:
        width = 2 if r.dtype in ("float16", "bfloat16") else 4
        seen[(r.op, r.N, r.K)] = r.N * r.K * width * (r.batch if r.op == "bmm" else 1)
    return sum(seen.values())


def benchmark_shape(
    shape: RepShape, warmup: int = 0, rep: int = 0, working_set_bytes: Optional[int] = None
) -> Optional[List[TacticResult]]:
    """Every registered candidate for *shape*, measured and sorted fastest first.

    ``warmup`` / ``rep`` are accepted for command-line compatibility and ignored:
    the protocol decides its own round counts (successive halving, adaptive stop).
    """
    from oasr.tune import gemm_tune

    case = gemm_tune.GemmCase(shape.op, shape.M, shape.N, shape.K, shape.dtype, shape.batch)
    try:
        res = gemm_tune.benchmark_case(case, working_set_bytes=working_set_bytes)
    except Exception as exc:  # noqa: BLE001
        print(f"[tune] {shape}: benchmark failed: {exc}", file=sys.stderr)
        return None
    for name, why in sorted(res.rejected.items()):
        print(f"[tune]   rejected {name}: {why}", file=sys.stderr)
    out: List[TacticResult] = []
    for name, m in res.results.items():
        entry = res.entries.get(name)
        if entry is None:
            continue
        ms = m.median_ms if (m.status == "ok" and m.eliminated_at is None) else float("inf")
        if m.status == "ok" and m.eliminated_at is not None:
            ms = m.median_ms  # eliminated arms keep their stage-0/1 estimate for the report
        out.append(
            TacticResult(
                entry.tactic, ms, entry.is_fallback, m.solo_ms, m.sigma_ms, m.issue_us, name
            )
        )
    out.sort(key=lambda r: r.median_ms)
    return out


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


def plan_rules(
    per_shape: Dict[Tuple, List[TacticResult]],
    reps: List[RepShape],
    min_speedup: float = 1.05,
) -> Dict[Tuple[str, int, int], List[Tuple]]:
    """Per ``(op, N, K)``: ascending ``(m_max, winner | None, comment, results)``.

    ``winner`` is ``None`` where the bucket keeps the fallback.  A bucket whose
    winner is not at least *min_speedup* faster than the fallback keeps the
    fallback, so it collapses into a neighbour instead of becoming a rule.  Two
    reasons, both learned from the whisper-tiny run:

    * ``_pick_winner`` can return a tactic **slower than the fallback**.  Its
      tie-break prefers a smaller tile within 5% of the measured best, and the
      fallback is often *in* that band — one emitted bucket read ``0.96x vs
      default``, i.e. a rule that made things worse.
    * adjacent buckets came back alternating torch / cutlass / default at
      1.02-1.08x.  Re-measuring those pairs with the arms interleaved showed
      them to be ties.  Encoding a tie costs a compiled variant and a boundary
      that can be wrong, and buys nothing.

    A third gate: a win must hold on the *single-replay* timing as well as the
    loop timing.  Back-to-back independent launches can overlap on the GPU,
    which flatters a low-occupancy tile that would never overlap inside a real
    layer, and self-overlap is not a speedup a model can spend.
    """
    by_key: Dict[Tuple[str, int, int], List[RepShape]] = defaultdict(list)
    for r in reps:
        by_key[(r.op, r.N, r.K)].append(r)
    plan: Dict[Tuple[str, int, int], List[Tuple]] = {}
    for (op, N, K), grp in sorted(by_key.items()):
        grp.sort(key=lambda r: (r.m_max is None, r.m_max or 0))
        rules = []
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
                rules.append(
                    (
                        r.m_max,
                        None,
                        f"M~{r.M}: fallback ({winner.tactic.backend} was only "
                        f"{speedup:.2f}x vs default)",
                        res,
                    )
                )
                continue
            rules.append(
                (
                    r.m_max,
                    winner,
                    f"M~{r.M}: {winner.tactic.backend} {winner.median_ms:.4f}ms "
                    f"({speedup:.2f}x vs default)",
                    res,
                )
            )
        plan[(op, N, K)] = rules
    return plan


def emit_rules(
    per_shape: Dict[Tuple, List[TacticResult]],
    reps: List[RepShape],
    sm: int,
    min_speedup: float = 1.05,
) -> str:
    """The sweep's rules as a Python literal, for reading in a review.

    Production no longer reads a literal -- the rules live in the tuning DB
    (``--emit-db``) -- but a literal is the most compact thing to read, and the
    decisions are the same ones :func:`plan_rules` makes for the DB.
    """
    lines = [f"_GEMM_HEURISTIC_RULES_SM{sm}: Dict[Tuple[str, int, int], list] = {{"]
    for (op, N, K), rules in plan_rules(per_shape, reps, min_speedup).items():
        # The literal a rule-less lookup falls back to (see select_default_config
        # / _dispatch_gemm_log_softmax): the fused launcher for the CTC head,
        # GEMM_DEFAULT for everything else.
        fallback_literal = '"fused"' if op == "gemm_log_softmax" else "GEMM_DEFAULT"
        entries = []
        for m_max, winner, comment, _res in rules:
            choice = (
                fallback_literal
                if winner is None
                else _choice_literal(winner.tactic, sm, winner.is_default)
            )
            entries.append((m_max, choice, comment))
        if not entries or all(c == fallback_literal for _, c, _ in entries):
            continue  # every bucket == the fallback → omit the rule entirely
        merged = []
        for m_max, choice, comment in entries:
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


def emit_db(
    per_shape: Dict[Tuple, List[TacticResult]],
    reps: List[RepShape],
    sm: int,
    min_speedup: float = 1.05,
    base=None,
):
    """The sweep's rules as a tuning file (``oasr.tune.database``), merged into *base*.

    Each ``(op, N, K)`` becomes one entry; every bucket's measured arms are its
    evidence.  Entries in *base* for signatures this sweep did not measure are
    kept, so a sweep of one model's widths does not erase another's.
    """
    import oasr.jit.gemm as jg
    from oasr.tune import bench, database
    from oasr.tune.gemm_tune import choice_of

    tf = base if base is not None else database.new_file("gemm", sm, jg.GEMM_LANE_BY_SM[sm])
    tf.validators["soft"] = database.current_soft_validators("gemm")
    tf.provenance["bench_protocol"] = bench.PROTOCOL_VERSION
    tf.provenance["tool"] = "scripts/tune_asr_gemm.py"
    for (op, N, K), rules in plan_rules(per_shape, reps, min_speedup).items():
        fallback = "fused" if op == "gemm_log_softmax" else "default"
        regions: List[Tuple] = []
        evidence: Dict[str, dict] = {}
        notes: List[str] = []
        dtype_class = "half"
        for m_max, winner, comment, res in rules:
            if winner is None:
                cid = fallback
                params = {"kind": fallback}
            else:
                choice = choice_of(winner.tactic, sm)
                cid = jg.gemm_config_id(choice)
                params = jg.gemm_config_to_params(choice)
            tf.configs[cid] = params
            if regions and regions[-1][1] == cid:
                regions[-1] = (m_max, cid)
            else:
                regions.append((m_max, cid))
            notes.append(comment)
            rep_m = comment.split(":", 1)[0].replace("M~", "M=")
            evidence[rep_m] = {
                (t.name or t.tactic.backend): [round(t.median_ms, 6), round(t.sigma_ms, 6)]
                for t in res[:5] + [t for t in res if t.is_default][:1]
                if t.median_ms < float("inf")
            }
        if not regions or all(c == fallback for _, c in regions):
            tf.entries.pop(jg.gemm_sig_key(op, N, K, dtype_class), None)
            continue
        tf.entries[jg.gemm_sig_key(op, N, K, dtype_class)] = database.Entry(
            regions=regions, evidence=evidence, source="aot", notes=notes
        )
    return tf


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
    p.add_argument("--warmup", type=int, default=25, help="ignored (the protocol decides)")
    p.add_argument("--rep", type=int, default=100, help="ignored (the protocol decides)")
    p.add_argument(
        "--emit-rules",
        metavar="FILE",
        help="also write the rules as a Python literal (review only)",
    )
    p.add_argument(
        "--emit-db",
        metavar="PATH",
        help="write the rules into a tuning file: a path, or 'user' for this machine's user "
        "tier (read by production dispatch), or 'system' for the shipped file of this arch",
    )
    p.add_argument(
        "--working-set-mb",
        type=float,
        default=None,
        help="the served model's per-forward weight MiB, for the L2 policy (default: the sum "
        "of the captured shapes' weights)",
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
                "(produce it via OASR_CAPTURE_GEMM=… python benchmarks/run.py --family engine …)",
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
    working_set = (
        int(args.working_set_mb * 2**20)
        if args.working_set_mb is not None
        else _working_set_bytes(reps)
    )
    print(f"[tune] weight working set {working_set / 2**20:.1f} MiB", file=sys.stderr)
    t_start = time.perf_counter()
    for i, r in enumerate(reps, 1):
        t_shape = time.perf_counter()
        res = benchmark_shape(r, working_set_bytes=working_set)
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
            f.write(
                "# Generated by scripts/tune_asr_gemm.py -- for review; production reads the "
                "tuning DB (--emit-db)\n"
            )
            f.write(rules + "\n")
        print(f"\n[tune] wrote rules to {args.emit_rules}")
    if args.emit_db:
        from pathlib import Path

        from oasr.tune import database

        if args.emit_db == "user":
            path = database.user_path(sm, "gemm")
            if path is None:
                print("[tune] the user tier is disabled (OASR_TUNE_USER_DB=off)", file=sys.stderr)
                return 2
        elif args.emit_db == "system":
            path = database.system_path(sm, "gemm")
        else:
            path = Path(args.emit_db)
        base = database.load_file(path, "gemm", sm) if path.is_file() else None
        tf = emit_db(per_shape, reps, sm, args.min_speedup, base=base)
        database.save_file(tf, path)
        print(f"\n[tune] wrote {len(tf.entries)} entries to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
