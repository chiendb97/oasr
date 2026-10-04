# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr tune build``: a census in, a tuning file out.

For every signature of a :class:`~oasr.tune.census.ShapeSet` this measures the
census points -- plus ladder fill points between them, so an eager tail or an
unseen batch width lands inside a measured region -- with the shared protocol,
decides each point against the untuned default, and partitions the M axis into
regions (:mod:`oasr.tune.partition`).  The output is a
:class:`~oasr.tune.database.TuningFile` ready for the user tier or, after
review, the shipped one.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from oasr.tune import bench, database, gemm_tune, shapes
from oasr.tune.census import ShapePoint, ShapeSet

logger = logging.getLogger("oasr.tune")

__all__ = [
    "BuildOptions",
    "PointResult",
    "TUNED_OPS",
    "build_gemm",
    "measure_signature",
    "thin_census",
]

#: The ops a GEMM build measures.  ``bmm`` is census data but not tuned here: the
#: tuned bmm lane is keyed without its batch count, which a region cannot index
#: (see the partition notes), so its points are skipped.  ``oasr tune prebuild``
#: reads this to compile only the modules a build will load.
TUNED_OPS = ("gemm", "gemm_activation", "gemm_log_softmax")


@dataclass
class BuildOptions:
    #: Replace the incumbent default only by at least this factor (and the noise).
    min_speedup: float = 1.05
    #: Tie band within which the structural preference decides.
    tie: float = 0.05
    #: Add ladder fill points between the census's smallest and largest M.
    fill: bool = True
    #: Largest number of M points per signature (census points always kept,
    #: unless ``thin``).
    max_points_per_sig: int = 24
    #: Hold the census points to ``max_points_per_sig`` too (:func:`thin_census`):
    #: a streaming census keys every cohort width, so one signature can carry
    #: 60-140 Ms a few percent apart, whose winners are mostly the same config.
    #: Measuring a log-spaced subset is what makes a rented-GPU build affordable.
    thin: bool = False
    #: ``"half"`` writes entries that serve fp16 and bf16 alike (measured in the
    #: census dtype); ``"exact"`` keys them at the measured dtype only.
    dtype_class: str = "half"
    #: Append every arm's timing at every point here (JSONL) -- the measurement
    #: log the cost model calibrates from.  ``None``: the user tier's default.
    log_path: Optional[str] = None
    #: Region partition: ``"points"`` (each measured point closes a region) or
    #: ``"cover"`` (shared config set + DP segmentation, oasr.tune.partition).
    partition: str = "cover"
    #: Regret the shared config set may add at any measured point.
    cover_eps: float = 0.03
    #: Benchmark only the union of the analytic and the calibrated model's top
    #: ``top_k`` (plus the forced set: default, fused, cuBLAS).  0 measures every
    #: candidate.  Measured on 149 Conformer points: prior recall@16 0.87,
    #: calibrated (leave-one-signature-out) 0.99; the best of the union's arms
    #: was within 1% of the exhaustive best at every point.
    top_k: int = 16
    policy: bench.BenchPolicy = field(default_factory=bench.BenchPolicy)
    progress: Optional[Callable[[str], None]] = None
    #: Cap on distinct CUTLASS configs the file references (``None``: no cap);
    #: see ``partition.apply_compile_budget``.
    compile_budget: Optional[int] = None
    #: Size of the coverage basis written into the file: the configs that cover
    #: the most measured points within ``cover_eps``, for the cost-model tier.
    basis_size: int = 8
    #: Stop starting new signatures after this many seconds (``None``: no limit).
    #: Signatures are taken heaviest first, so a budget spends itself on the
    #: shapes that move end-to-end time (the Ansor/MetaSchedule allocation rule).
    budget_s: Optional[float] = None
    #: Record each finished signature here (JSONL) and skip, on the next run,
    #: every signature recorded with the same points, options and kernels -- so
    #: a build interrupted mid-way (a preempted container, a budget cut) resumes
    #: where it stopped instead of paying for its measurements twice.
    checkpoint_path: Optional[str] = None


@dataclass
class PointResult:
    M: int
    must: bool
    alpha: float
    case: gemm_tune.CaseResult
    #: The decision: a config id, or the fallback id.
    choice: str
    speedup: float
    note: str
    #: The census weight (expected time share); fill points get a small one.
    weight: float = 1.0


def _fill_points(ms: Sequence[int], cap: int) -> List[int]:
    """The census Ms plus bucket edges between them, at most *cap* points."""
    ms = sorted({int(m) for m in ms})
    if not ms:
        return []
    fill = [e for e in shapes.ladder(ms[-1], ms[0]) if e not in ms]
    room = max(0, cap - len(ms))
    if len(fill) > room:
        # Keep the fill points spread evenly in log space.
        step = len(fill) / max(room, 1)
        fill = [fill[int(i * step)] for i in range(room)]
    return sorted(set(ms) | set(fill))


def _bin_heaviest(pts: Sequence[ShapePoint], idx: Sequence[int], key, nbins: int) -> List[int]:
    """The heaviest of each of *nbins* equal bins of ``key(i)`` (a fraction in [0, 1])."""
    bins: Dict[int, List[int]] = {}
    for i in idx:
        bins.setdefault(min(nbins - 1, int(key(i) * nbins)), []).append(i)
    return [max(b, key=lambda i: (pts[i].weight, pts[i].must, -pts[i].M)) for b in bins.values()]


def thin_census(points: Sequence[ShapePoint], cap: int) -> List[ShapePoint]:
    """At most *cap* of one signature's census points: where the range is, and where the time is.

    The smallest and largest M always stay, so the measured range is the
    census's.  Half the remaining slots cut the log-M span into equal bins, the
    other half cut the cumulative *weight* into equal bins, and each bin keeps
    its heaviest point; leftover slots go to the heaviest of the rest.  Log bins
    alone spent most slots on dense, cheap, small-M cohort widths that all pick
    one config, and missed the heavy wave-quantised top where the winner changes
    every few hundred rows (Conformer N=K=256: 7.5% weighted regret against 4.5%
    for measuring all 82 points); weight bins alone starve the light points.
    Scored offline against a full sm120 build of the 11-model census, 16 points:
    0.92% weighted regret against 0.82% for all of them, at 37% of the points.

    A dropped point's weight, calls and ``must`` flag move to the next kept
    point at or above it -- the one whose region serves it, since regions round
    up -- so the partition still weighs the traffic the census saw.
    """
    import dataclasses

    pts = sorted(points, key=lambda p: p.M)
    if cap <= 0 or len(pts) <= cap:
        return list(pts)
    inner = list(range(1, len(pts) - 1))
    lo, hi = math.log(max(pts[0].M, 1)), math.log(max(pts[-1].M, 1))
    total = sum(pts[i].weight for i in inner) or 1.0
    below: Dict[int, float] = {}
    acc = 0.0
    for i in inner:  # the weight up to the middle of point i
        below[i] = (acc + pts[i].weight / 2) / total
        acc += pts[i].weight
    slots = cap - 2
    keep = {0, len(pts) - 1}
    for cand in (
        _bin_heaviest(
            pts,
            inner,
            lambda i: (math.log(max(pts[i].M, 1)) - lo) / (hi - lo) if hi > lo else 0.0,
            (slots + 1) // 2,
        ),
        _bin_heaviest(pts, inner, lambda i: below[i], max(slots // 2, 1)),
    ):
        for i in cand:
            if len(keep) < cap:
                keep.add(i)
    rest = sorted((i for i in range(len(pts)) if i not in keep), key=lambda i: -pts[i].weight)
    keep.update(rest[: max(0, cap - len(keep))])
    out: List[ShapePoint] = []
    calls = weight = 0.0
    must, eager = False, 0.0
    for i, p in enumerate(pts):
        calls, weight = calls + p.calls, weight + p.weight
        must, eager = must or p.must, max(eager, p.eager_fraction)
        if i in keep:
            out.append(
                dataclasses.replace(p, calls=calls, weight=weight, must=must, eager_fraction=eager)
            )
            calls = weight = 0.0
            must, eager = False, 0.0
    return out


def _alpha_for(m: int, points: Sequence[ShapePoint]) -> float:
    """The eager fraction at *m*: the census point's, else the nearest one's."""
    best = min(points, key=lambda p: abs(math.log(max(p.M, 1)) - math.log(max(m, 1))))
    return float(best.eager_fraction)


def _pruned(entries, op: str, M: int, N: int, K: int, models, k: int):
    """The registry entries worth measuring at one point: top-k of each model + forced."""
    if k <= 0 or not models:
        return entries
    import oasr.jit.gemm as jg

    by_name, params = {}, {}
    for e in entries:
        try:
            choice = gemm_tune.choice_of(e.tactic)
        except Exception:  # noqa: BLE001
            continue
        name = gemm_tune.choice_name(choice)
        by_name.setdefault(name, e)
        if not isinstance(choice, str):
            params[name] = jg.gemm_config_to_params(choice, sentinel_default=False)
    from oasr.tune.cost_model import features_of

    keep = {"default", "fused", "torch"}
    # What no model describes -- the CUTLASS 3.x configs have no cost structure
    # yet -- is measured, never pruned: a mixed space (sm90/sm100: 3.x + the
    # mma.sync lane) ranks only its 2.x half, and keeping the top-k of *that*
    # would silently drop every native kernel.
    keep.update(n for n, prm in params.items() if features_of(prm) is None)
    for model in models:
        keep.update(model.top_k(params, M, N, K, k))
    return [e for n, e in by_name.items() if n in keep]


def measure_signature(
    sig: Tuple[str, int, int, str, int],
    points: Sequence[ShapePoint],
    working_set_bytes: int,
    opts: BuildOptions,
    models: Sequence = (),
) -> List[PointResult]:
    """Measure one signature at its census points (+ fill) and decide each point."""
    op, N, K, dtype, batch = sig
    if opts.thin:
        points = thin_census(points, opts.max_points_per_sig)
    census_ms = {p.M: p for p in points}
    ms = _fill_points(census_ms, opts.max_points_per_sig) if opts.fill else sorted(census_ms)
    fallback = "fused" if op == "gemm_log_softmax" else "default"
    out: List[PointResult] = []
    entries = gemm_tune.candidates(op)
    census_w = [p.weight for p in points if p.weight > 0]
    fill_w = 0.25 * min(census_w) if census_w else 1.0
    for M in ms:
        case = gemm_tune.GemmCase(op, M, N, K, dtype, batch)
        res = gemm_tune.benchmark_case(
            case,
            _pruned(entries, op, M, N, K, models, opts.top_k),
            policy=opts.policy,
            working_set_bytes=working_set_bytes,
        )
        alpha = _alpha_for(M, points)
        winner = res.winner(alpha=alpha, tie=opts.tie)
        base = res.results.get(fallback)
        if winner is None:
            choice, sp, note = fallback, 1.0, "no candidate measured"
        elif base is None or base.status != "ok" or not math.isfinite(base.median_ms):
            choice, sp, note = winner.name, math.inf, "fallback not measurable"
        elif winner.name == fallback:
            choice, sp, note = fallback, 1.0, "fallback is fastest"
        elif bench.replaces(winner, base, alpha=alpha, min_speedup=opts.min_speedup):
            sp = bench.speedup(winner, base, alpha)
            choice, note = winner.name, f"{sp:.2f}x vs {fallback}"
        else:
            sp = bench.speedup(winner, base, alpha)
            choice, note = fallback, f"{winner.name} {sp:.2f}x rejected: " + bench.why_not_replaced(
                winner, base, alpha=alpha, min_speedup=opts.min_speedup
            )
        w = census_ms[M].weight if M in census_ms and census_ms[M].weight > 0 else fill_w
        out.append(
            PointResult(M, M in census_ms and census_ms[M].must, alpha, res, choice, sp, note, w)
        )
        if opts.progress:
            opts.progress(f"{op} N={N} K={K} M={M}: {choice} ({note})")
    if opts.top_k > 0 and models:
        _cross_evaluate(out, op, N, K, dtype, batch, entries, working_set_bytes, opts)
    return out


def _cross_evaluate(results, op, N, K, dtype, batch, entries, working_set_bytes, opts) -> None:
    """Measure every point's finalists at every other point (design section 3.4.1).

    Pruning measures a different subset at each point, and a config absent from
    a point is an infinite cell to the partitioner -- which can then share only
    the arms measured everywhere (the forced ones), and did: a pruned Conformer
    build without this pass routed every signature to cuBLAS.  The union of each
    point's top three is small, so filling its column costs a few arms a point.
    """
    finalists = set()
    for pr in results:
        ok = sorted(
            (m for m in pr.case.results.values() if m.status == "ok"), key=lambda m: m.median_ms
        )
        finalists.update(m.name for m in ok[:3])
    for pr in results:
        missing = finalists - {n for n, m in pr.case.results.items() if m.status == "ok"}
        if not missing:
            continue
        case = gemm_tune.GemmCase(op, pr.M, N, K, dtype, batch)
        extra = gemm_tune.benchmark_case(
            case,
            entries,
            policy=opts.policy,
            working_set_bytes=working_set_bytes,
            only=missing,
            forced=(),
        )
        for name, m in extra.results.items():
            if m.status == "ok":
                pr.case.results[name] = m
        pr.case.entries.update(extra.entries)
    if opts.progress:
        opts.progress(f"{op} N={N} K={K}: cross-evaluated {len(finalists)} finalists")


def _log_rows(pr: PointResult, sig, sm: int, sku, impl_hash: str) -> List[Dict[str, Any]]:
    import oasr.jit.gemm as jg

    op, N, K, dtype, batch = sig
    rows = []
    for name, m in pr.case.results.items():
        if m.status != "ok" or not math.isfinite(m.median_ms):
            continue
        choice = m.payload
        params = (
            {"kind": choice}
            if isinstance(choice, str)
            else jg.gemm_config_to_params(choice, sentinel_default=False)
        )
        rows.append(
            {
                "sm": sm,
                "sku": sku,
                "impl_hash": impl_hash,
                "protocol": bench.PROTOCOL_VERSION,
                "op": op,
                "M": pr.M,
                "N": N,
                "K": K,
                "dtype": dtype,
                "batch": batch,
                "config": name,
                "params": params,
                "median_ms": round(m.median_ms, 7),
                "sigma_ms": round(m.sigma_ms, 7),
                "n": m.n,
                "stage": m.eliminated_at,
                "issue_us": None if m.issue_us is None else round(m.issue_us, 3),
                "l2": pr.case.l2_state,
                "calls": m.calls_per_graph,
                "contended": pr.case.conditions.contended,
            }
        )
    return rows


def _evidence(pr: PointResult, fallback: str) -> Dict[str, Any]:
    rows = sorted(
        (m for m in pr.case.results.values() if m.status == "ok"), key=lambda m: m.median_ms
    )
    ev: Dict[str, Any] = {}
    for m in rows[:4] + [pr.case.results[n] for n in (fallback,) if n in pr.case.results]:
        ev[m.name] = [round(m.median_ms, 6), round(m.sigma_ms, 6), m.n]
    ev["_l2"] = pr.case.l2_state
    if pr.alpha:
        ev["_alpha"] = round(pr.alpha, 3)
    return ev


def _apply_budget_and_basis(tf, matrices, fallbacks, payloads, opts) -> None:
    """Enforce ``compile_budget`` over this build's entries; write the coverage basis."""
    import oasr.jit.gemm as jg
    from oasr.tune import partition

    if opts.compile_budget is not None and matrices:
        chosen = {k: tf.entries[k].regions for k in matrices if k in tf.entries}
        chosen = partition.apply_compile_budget(
            {k: matrices[k] for k in chosen},
            chosen,
            opts.compile_budget,
            {k: fallbacks[k] for k in chosen},
        )
        for k, regions in chosen.items():
            tf.entries[k].regions = regions
            tf.entries[k].notes.append(f"compile budget {opts.compile_budget}: {regions}")
            for _hi, cid in regions:
                if cid not in tf.configs and cid in payloads:
                    tf.configs[cid] = jg.gemm_config_to_params(payloads[cid])
    if opts.basis_size > 0 and matrices:
        merged = partition.Matrix([], [], [])
        for mat in matrices.values():
            merged.ms += mat.ms
            merged.w += [1.0] * len(mat.ms)  # every measured point counts once
            merged.T += mat.T
        cover = partition.config_cover(merged, opts.cover_eps)
        basis = [c for c in cover if c not in ("default", "fused", "torch")][: opts.basis_size]
        for cid in basis:
            if cid not in tf.configs and cid in payloads:
                tf.configs[cid] = jg.gemm_config_to_params(payloads[cid])
        tf.coverage_basis = [c for c in basis if c in tf.configs]


def _sig_id(sig) -> str:
    return "|".join(str(v) for v in sig)


def _points_digest(pts: Sequence[ShapePoint], opts: BuildOptions) -> str:
    """What a checkpointed signature's result depends on besides the kernels."""
    h = hashlib.sha256()
    rows = sorted((p.M, round(p.weight, 9), bool(p.must), round(p.eager_fraction, 6)) for p in pts)
    h.update(json.dumps(rows).encode())
    knobs = [opts.top_k, opts.max_points_per_sig, opts.fill, opts.min_speedup, opts.tie]
    knobs += [opts.partition, opts.cover_eps, opts.dtype_class, bench.PROTOCOL_VERSION]
    if opts.thin:  # appended only when set: an unthinned checkpoint keeps its digest
        knobs.append("thin")
    h.update(json.dumps(knobs).encode())
    return h.hexdigest()[:16]


def _load_checkpoint(path: Optional[str], sm: int, impl_hash: str) -> Dict[str, Dict[str, Any]]:
    """Finished-signature records for this arch and kernel build, by signature id.

    A torn last line (the process died mid-write) is skipped, not fatal.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("sm") == sm and rec.get("impl") == impl_hash:
                out[str(rec["sig"])] = rec
    return out


def _append_checkpoint(f, rec: Dict[str, Any]) -> None:
    f.write(json.dumps(rec) + "\n")
    f.flush()
    os.fsync(f.fileno())


def build_gemm(
    shapes_: ShapeSet,
    *,
    sm: Optional[int] = None,
    opts: Optional[BuildOptions] = None,
    base: Optional[database.TuningFile] = None,
) -> database.TuningFile:
    """Tune every GEMM-family signature of *shapes_*; return the tuning file."""
    import oasr.jit.gemm as jg
    from oasr.jit.core import _get_target_sm
    from oasr.tune import partition

    opts = opts or BuildOptions()
    sm = int(sm if sm is not None else _get_target_sm())
    tf = base if base is not None else database.new_file("gemm", sm, jg.GEMM_LANE_BY_SM[sm])
    tf.validators["soft"] = database.current_soft_validators("gemm")
    t0 = time.time()
    cond_contended = False
    sigs = shapes_.signatures()
    # Heaviest signature first, so a time budget spends itself where it matters.
    sigs = dict(sorted(sigs.items(), key=lambda kv: -sum(p.weight for p in kv[1])))
    log_path = opts.log_path
    if log_path is None:
        root = database.user_root()
        log_path = str(root / "measurements" / f"sm{sm}.jsonl") if root is not None else None
    log_f = None
    if log_path:
        from pathlib import Path

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a")
    sku = database.current_soft_validators("gemm").get("sku")
    impl_hash = jg.gemm_impl_hash()
    from oasr.tune import arch as arch_mod
    from oasr.tune.cost_model import GemmCostModel

    prof = arch_mod.current(measure=True)
    prior = GemmCostModel.for_arch(prof)
    calibrated = GemmCostModel.for_arch(prof)
    if log_path:
        from pathlib import Path as _P

        if _P(log_path).is_file():
            from oasr.tune.evaluate import load_log

            calibrated.calibrate(
                [
                    r
                    for r in load_log(log_path)
                    if int(r.get("sm", sm)) == sm and r.get("impl_hash") == impl_hash
                ]
            )
    models = [prior] + ([calibrated] if calibrated.coeffs else [])
    matrices: Dict[str, "partition.Matrix"] = {}
    fallbacks: Dict[str, str] = {}
    payloads_all: Dict[str, Any] = {}
    done = _load_checkpoint(opts.checkpoint_path, sm, impl_hash)
    ckpt_f = None
    if opts.checkpoint_path:
        from pathlib import Path

        Path(opts.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_f = open(opts.checkpoint_path, "a")
    resumed = 0

    def key_of(sig) -> str:
        op_, N_, K_, dtype_, _b = sig
        dt_ = (
            "half"
            if opts.dtype_class == "half"
            else {"float16": "fp16", "bfloat16": "bf16"}[dtype_]
        )
        return jg.gemm_sig_key(op_, N_, K_, dt_)

    # Everything allocated so far -- the census above all, hundreds of thousands of
    # objects -- lives for the whole build, and ``bench.measure`` collects garbage
    # after every point to release that point's graphs: without the freeze each
    # collection walked the census again, a fifth of a build's wall time (py-spy,
    # 2026-10-01).  Only what the loop allocates is collected; nothing measured changes.
    gc.freeze()
    try:
        for i, (sig, pts) in enumerate(sigs.items(), 1):
            if opts.budget_s is not None and time.time() - t0 > opts.budget_s:
                if opts.progress:
                    left = sum(
                        1
                        for s in list(sigs)[i - 1 :]
                        if s[0] in TUNED_OPS and s[3] in ("float16", "bfloat16")
                    )
                    opts.progress(
                        f"budget {opts.budget_s:.0f}s spent; {left} tuned signature(s) left "
                        "unmeasured (the census's bmm signatures are never tuned)"
                    )
                break
            op, N, K, dtype, batch = sig
            if op not in TUNED_OPS or dtype not in ("float16", "bfloat16"):
                continue
            fallback = "fused" if op == "gemm_log_softmax" else "default"
            sid, digest = _sig_id(sig), _points_digest(pts, opts)
            rec = done.get(sid)
            if rec is not None and rec.get("digest") == digest:
                k = str(rec["key"])
                mx = rec["matrix"]
                matrices[k] = partition.Matrix(
                    ms=[int(m) for m in mx["ms"]],
                    w=[float(v) for v in mx["w"]],
                    T=[{str(c): float(t) for c, t in row.items()} for row in mx["T"]],
                )
                fallbacks[k] = str(rec["fallback"])
                payloads_all.update(
                    {n: jg.gemm_config_from_params(p, sm) for n, p in rec["payloads"].items()}
                )
                tf.configs.update(rec["configs"])
                if rec["entry"] is None:
                    tf.entries.pop(k, None)
                else:
                    tf.entries[k] = database.Entry.from_json(rec["entry"])
                cond_contended |= bool(rec["contended"])
                resumed += 1
                continue
            results = measure_signature(sig, pts, shapes_.working_set_bytes, opts, models)
            cond_contended |= any(r.case.conditions.contended for r in results)
            if log_f is not None:
                import json as _json

                for pr in results:
                    for row in _log_rows(pr, sig, sm, sku, impl_hash):
                        log_f.write(_json.dumps(row) + "\n")
                log_f.flush()
            matrices[key_of(sig)] = partition.matrix_from_points(results)
            fallbacks[key_of(sig)] = fallback
            if opts.partition == "cover":
                regions, notes = partition.regions_for_signature(
                    results, fallback, opts.cover_eps, min_speedup=opts.min_speedup
                )
            else:
                regions, notes = partition.regions_from_decisions(results, fallback)
            payloads = {n: m.payload for pr in results for n, m in pr.case.results.items()}
            payloads_all.update(payloads)
            for _, cid in regions:
                tf.configs[cid] = (
                    {"kind": cid}
                    if cid in ("default", "fused", "torch")
                    else jg.gemm_config_to_params(payloads[cid])
                )
            dt = (
                "half"
                if opts.dtype_class == "half"
                else {"float16": "fp16", "bfloat16": "bf16"}[dtype]
            )
            key = jg.gemm_sig_key(op, N, K, dt)
            entry = None
            if not regions or all(c == fallback for _, c in regions):
                tf.entries.pop(key, None)
            else:
                entry = database.Entry(
                    regions=regions,
                    evidence={f"M={pr.M}": _evidence(pr, fallback) for pr in results},
                    source="aot",
                    notes=notes
                    + [
                        f"M={pr.M}{'*' if pr.must else ''}: {pr.choice} ({pr.note})"
                        for pr in results
                    ],
                )
                tf.entries[key] = entry
            if ckpt_f is not None:
                mx = matrices[key]
                _append_checkpoint(
                    ckpt_f,
                    {
                        "sig": sid,
                        "digest": digest,
                        "sm": sm,
                        "impl": impl_hash,
                        "key": key,
                        "fallback": fallback,
                        "matrix": {"ms": mx.ms, "w": mx.w, "T": mx.T},
                        "payloads": {n: jg.gemm_config_to_params(c) for n, c in payloads.items()},
                        "configs": {
                            cid: tf.configs[cid] for _, cid in regions if cid in tf.configs
                        },
                        "entry": None if entry is None else entry.to_json(),
                        "contended": any(r.case.conditions.contended for r in results),
                    },
                )
            if entry is None:
                continue
            if opts.progress:
                opts.progress(f"[{i}/{len(sigs)}] {key}: {regions}  ({time.time() - t0:.0f}s)")
    finally:
        gc.unfreeze()
    if log_f is not None:
        log_f.close()
    if ckpt_f is not None:
        ckpt_f.close()
    if resumed and opts.progress:
        opts.progress(f"resumed {resumed} signature(s) from {opts.checkpoint_path}")
    _apply_budget_and_basis(tf, matrices, fallbacks, payloads_all, opts)
    # The device model ships with the file: the resolver's fallback for widths no
    # entry covers ranks with it (the analytic structure; see cost_model), and
    # the calibration seeds the next build's pruning.
    final = GemmCostModel.for_arch(prof)
    if log_path:
        from oasr.tune.evaluate import load_log

        final.calibrate(
            [
                r
                for r in load_log(log_path)
                if int(r.get("sm", sm)) == sm and r.get("impl_hash") == impl_hash
            ]
        )
    tf.model = {"gemm": final.to_json(), "arch": prof.to_json()}
    # Drop configs no entry references any more (a re-tune replaced them).
    referenced = set(tf.referenced_configs())
    tf.configs = {k: v for k, v in tf.configs.items() if k in referenced}
    tf.provenance.update(
        {
            "tool": "oasr tune build",
            "bench_protocol": bench.PROTOCOL_VERSION,
            "census": shapes_.provenance,
            "contended": cond_contended,
            "elapsed_s": round(time.time() - t0, 1),
        }
    )
    return tf
