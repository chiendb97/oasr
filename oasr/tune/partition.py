# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shape-space partitioning: regions of M that share one config.

Given every candidate's time at every measured M of one signature (the
cross-evaluation matrix -- the protocol measures every arm at every point, the
losers just with fewer rounds), this chooses

1. a **shared config set** by weighted greedy set cover: the fewest configs
   such that every point has one within ``(1 + eps)`` of its best
   (:func:`config_cover`) -- which bounds the regret at every measured point;
2. **contiguous regions** by dynamic programming over the sorted points, each
   region one config from the set, minimising ``lambda x regions + weighted
   regret`` (:func:`segment`) -- so lookup stays a bisect of ascending
   ``(m_hi, config)`` exactly as the rule tables always were;
3. a per-region **gate** against the untuned fallback: a region whose config
   does not beat the fallback by ``min_speedup`` on its weighted time keeps the
   fallback, and adjacent equal regions merge (:func:`regions_for_signature`).

Across signatures, :func:`apply_compile_budget` removes configs from the union
until at most ``budget`` remain, each time dropping the one whose removal adds
the least weighted regret -- the knob that trades compile time for speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Matrix",
    "matrix_from_points",
    "config_cover",
    "segment",
    "regions_from_decisions",
    "regions_for_signature",
    "regions_for_matrix",
    "apply_compile_budget",
]


@dataclass
class Matrix:
    """``T[i][config]`` in ms at ascending points ``ms[i]`` with weights ``w[i]``."""

    ms: List[int]
    w: List[float]
    T: List[Dict[str, float]]

    def best(self, i: int, allowed: Optional[Set[str]] = None) -> float:
        row = self.T[i]
        vals = [t for c, t in row.items() if (allowed is None or c in allowed) and math.isfinite(t)]
        return min(vals) if vals else math.inf


def matrix_from_points(results: Sequence, alpha_issue: bool = True) -> Matrix:
    """The cross-evaluation matrix from ``build.PointResult`` s.

    The objective per cell is ``GPU time + alpha x issue time``.  Issue time is
    measured for the arms that reached the last halving stage; the others take
    the median issue time of their backend class at that point (every CUTLASS
    variant shares one launcher, cuBLAS has its own).
    """
    ms, w, T = [], [], []
    for pr in sorted(results, key=lambda r: r.M):
        row: Dict[str, float] = {}
        measured = [m for m in pr.case.results.values() if m.issue_us is not None]
        cls_issue: Dict[str, float] = {}
        for cls in ("torch", "cutlass"):
            vals = sorted(m.issue_us for m in measured if _cls(m.name) == cls)
            if vals:
                cls_issue[cls] = vals[len(vals) // 2]
        for name, m in pr.case.results.items():
            if m.status != "ok" or not math.isfinite(m.median_ms):
                continue
            t = m.median_ms
            if alpha_issue and pr.alpha:
                issue = m.issue_us if m.issue_us is not None else cls_issue.get(_cls(name), 0.0)
                t += pr.alpha * issue / 1000.0
            row[name] = t
        ms.append(int(pr.M))
        w.append(float(getattr(pr, "weight", 1.0) or 1.0))
        T.append(row)
    return Matrix(ms, w, T)


def _cls(name: str) -> str:
    return "torch" if name == "torch" else "cutlass"


def config_cover(mat: Matrix, eps: float = 0.03, always: Sequence[str] = ()) -> List[str]:
    """Weighted greedy set cover: configs so every point has one within ``(1+eps)``.

    Greedy is within ``ln(n)`` of the optimum and ``n`` (points per signature)
    is small.  The bound it gives is the property that matters: with the set,
    regret is at most ``eps`` at every measured point.
    """
    n = len(mat.ms)
    best = [mat.best(i) for i in range(n)]
    covers: Dict[str, Set[int]] = {}
    for i in range(n):
        for c, t in mat.T[i].items():
            if math.isfinite(best[i]) and t <= (1.0 + eps) * best[i]:
                covers.setdefault(c, set()).add(i)
    chosen: List[str] = [c for c in always if c in covers or any(c in r for r in mat.T)]
    left = {i for i in range(n) if math.isfinite(best[i])}
    for c in chosen:
        left -= covers.get(c, set())
    while left:
        c = max(sorted(covers), key=lambda c: sum(mat.w[i] for i in covers[c] & left))
        gained = covers[c] & left
        if not gained:
            break
        chosen.append(c)
        left -= gained
    return chosen


def segment(mat: Matrix, configs: Sequence[str], lam: float) -> List[Tuple[int, int, str]]:
    """Contiguous segments ``(i0, i1_inclusive, config)`` minimising regions + regret.

    ``cost(segment, c) = lam + sum_i w_i * (T[i][c] / best_i - 1)``, with
    ``best_i`` over *configs*.  O(n^2 |configs|); n is tens.
    """
    n = len(mat.ms)
    if n == 0:
        return []
    allowed = set(configs)
    best = [mat.best(i, allowed) for i in range(n)]

    def regret(i: int, c: str) -> float:
        t = mat.T[i].get(c, math.inf)
        if not math.isfinite(best[i]):
            return 0.0
        return mat.w[i] * (t / best[i] - 1.0) if math.isfinite(t) else math.inf

    # prefix sums of regret per config
    pre: Dict[str, List[float]] = {}
    for c in configs:
        acc, row = 0.0, [0.0]
        for i in range(n):
            acc += regret(i, c)
            row.append(acc)
        pre[c] = row
    D = [0.0] + [math.inf] * n
    arg: List[Optional[Tuple[int, str]]] = [None] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            for c in configs:
                cost = D[i] + lam + (pre[c][j] - pre[c][i])
                if cost < D[j]:
                    D[j], arg[j] = cost, (i, c)
    out = []
    j = n
    while j > 0:
        i, c = arg[j]  # type: ignore[misc]
        out.append((i, j - 1, c))
        j = i
    return out[::-1]


def _merge(regions: List[Tuple[Optional[int], str]]) -> List[Tuple[Optional[int], str]]:
    merged: List[Tuple[Optional[int], str]] = []
    for hi, c in regions:
        if merged and merged[-1][1] == c:
            merged[-1] = (hi, c)
        else:
            merged.append((hi, c))
    return merged


def _absorb_fallback_islands(regions, mat: Matrix, fallback: str, eps: float):
    """``[c, fallback, c]`` -> ``[c]`` where ``c`` is within ``eps`` of the best inside.

    The fallback gate turns a region whose config only *ties* the fallback into
    the fallback, which between two regions of the same config leaves an island
    that costs two boundaries and buys nothing measurable.
    """
    changed = True
    while changed and len(regions) >= 3:
        changed = False
        for k in range(1, len(regions) - 1):
            (lo_hi, c0), (mid_hi, cm), (_, c1) = regions[k - 1], regions[k], regions[k + 1]
            if cm != fallback or c0 != c1 or c0 == fallback:
                continue
            ok = True
            for i, m in enumerate(mat.ms):
                if m <= (lo_hi or -1) or (mid_hi is not None and m > mid_hi):
                    continue
                b = mat.best(i)
                t = mat.T[i].get(c0, math.inf)
                if not (math.isfinite(t) and t <= (1.0 + eps) * b):
                    ok = False
                    break
            if ok:
                regions = _merge(regions[:k] + [(mid_hi, c0)] + regions[k + 1 :])
                changed = True
                break
    return regions


def regions_from_decisions(results: Sequence, fallback: str):
    """Each measured point's own decision closes a region; the last is the catch-all."""
    rs = sorted(results, key=lambda r: r.M)
    regions: List[Tuple[Optional[int], str]] = [(int(r.M), r.choice) for r in rs]
    if regions:
        regions[-1] = (None, regions[-1][1])
    return _merge(regions), ["partition: per-point decisions"]


def regions_for_signature(
    results: Sequence,
    fallback: str,
    eps: float = 0.03,
    *,
    min_speedup: float = 1.05,
    lam_frac: float = 0.005,
) -> Tuple[List[Tuple[Optional[int], str]], List[str]]:
    """Shared-config regions for one signature (set cover + DP + fallback gate).

    ``lam_frac`` prices one extra region at that fraction of the signature's
    total weight: a boundary has to buy back at least that much weighted regret
    to exist, which is what keeps ties from alternating region by region.
    """
    return regions_for_matrix(
        matrix_from_points(results), fallback, eps, min_speedup=min_speedup, lam_frac=lam_frac
    )


def regions_for_matrix(
    mat: Matrix,
    fallback: str,
    eps: float = 0.03,
    *,
    min_speedup: float = 1.05,
    lam_frac: float = 0.005,
) -> Tuple[List[Tuple[Optional[int], str]], List[str]]:
    """:func:`regions_for_signature` over an already-built matrix.

    What a checkpointed or logged matrix is re-partitioned with, without
    re-measuring -- how a cheaper measurement plan is scored against a full one.
    """
    if not mat.ms:
        return [], []
    cover = config_cover(mat, eps, always=(fallback,))
    lam = lam_frac * sum(mat.w)
    segs = segment(mat, cover, lam)
    regions: List[Tuple[Optional[int], str]] = []
    notes = [f"partition: cover {cover} (eps {eps:.0%}), {len(segs)} segment(s)"]
    for i0, i1, c in segs:
        if c != fallback:
            num = sum(mat.w[i] * mat.T[i].get(fallback, math.inf) for i in range(i0, i1 + 1))
            den = sum(mat.w[i] * mat.T[i].get(c, math.inf) for i in range(i0, i1 + 1))
            sp = num / den if den > 0 else 1.0
            if not (sp >= min_speedup):
                notes.append(f"M {mat.ms[i0]}..{mat.ms[i1]}: {c} only {sp:.2f}x -> {fallback}")
                c = fallback
        regions.append((mat.ms[i1], c))
    regions[-1] = (None, regions[-1][1])
    regions = _absorb_fallback_islands(_merge(regions), mat, fallback, eps)
    worst = 0.0
    for i in range(len(mat.ms)):
        from oasr.tune.database import region_lookup

        sel = region_lookup(regions, mat.ms[i])
        b = mat.best(i)
        t = mat.T[i].get(sel, math.inf) if sel else math.inf
        if math.isfinite(b) and math.isfinite(t):
            worst = max(worst, t / b - 1.0)
    notes.append(f"worst measured regret {worst:.1%}")
    return regions, notes


def apply_compile_budget(
    matrices: Dict[str, Matrix],
    chosen: Dict[str, List[Tuple[Optional[int], str]]],
    budget: int,
    fallbacks: Dict[str, str],
    keep: Sequence[str] = ("default", "fused", "torch"),
) -> Dict[str, List[Tuple[Optional[int], str]]]:
    """Shrink the union of configs to *budget*, dropping the cheapest-to-lose first.

    ``matrices``/``chosen`` are keyed by signature.  Removing a config replaces
    it, in every region that used it, by the best remaining config for that
    region's points; the config whose removal adds the least weighted regret
    goes first.  Sentinels (*keep*) cost no compilation and are never removed.
    """

    def used() -> Set[str]:
        return {c for regs in chosen.values() for _, c in regs if c not in keep}

    def region_points(sig: str, regions) -> List[List[int]]:
        mat = matrices[sig]
        buckets: List[List[int]] = [[] for _ in regions]
        for i, m in enumerate(mat.ms):
            idx = next(k for k, (hi, _c) in enumerate(regions) if hi is None or m <= hi)
            buckets[idx].append(i)
        return buckets

    def replace(sig: str, regions, drop: Set[str], allowed: Set[str]):
        mat = matrices[sig]
        out, cost = [], 0.0
        for (hi, c), pts in zip(regions, region_points(sig, regions)):
            if c in drop:
                cands = [x for x in allowed if x not in drop] + [fallbacks[sig]]

                def tot(x, pts=pts):
                    return sum(mat.w[i] * mat.T[i].get(x, math.inf) for i in pts)

                new = min(cands, key=tot)
                cost += tot(new) - tot(c)
                c = new
            out.append((hi, c))
        return _merge(out), cost

    while len(used()) > budget:
        allowed = used()
        best_cost, best_new = math.inf, None
        for c in sorted(allowed):
            total, new = 0.0, {}
            for sig, regs in chosen.items():
                r, cost = replace(sig, regs, {c}, allowed)
                total += cost
                new[sig] = r
            if total < best_cost:
                best_cost, best_new = total, new
        if best_new is None:
            break
        chosen = best_new
    return chosen
