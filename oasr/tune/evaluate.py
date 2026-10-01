# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Selection regret against an exhaustive oracle, from a measurement log.

``oasr tune build`` logs every candidate's time at every point it measured, so
the log *is* an exhaustive sweep of the tuning space at those points (every
numerics-passing, compiled arm; the losers with fewer halving rounds).  This
scores any selector against it::

    regret(selector, point) = T(selector's choice) / T(best measured) - 1

reported as geomean, p95 and max per selector, and per signature.  Selectors:
``default`` (what an untuned arch runs), ``shipped`` (the system tier),
any tuning file passed in, and optionally the calibrated cost model's top-1.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["load_log", "evaluate_log", "format_report"]


def load_log(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _points(rows) -> Dict[Tuple, Dict[str, Dict[str, Any]]]:
    """``(op, N, K, dtype, M) -> {config: row}``, the latest measurement per config."""
    out: Dict[Tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        key = (r["op"], int(r["N"]), int(r["K"]), r["dtype"], int(r["M"]))
        out[key][str(r["config"])] = r
    return out


def _stats(regrets: Sequence[float]) -> Dict[str, float]:
    vals = [r for r in regrets if math.isfinite(r)]
    missing = len(regrets) - len(vals)
    if not vals:
        return {"n": 0, "missing": missing}
    s = sorted(vals)
    geo = math.exp(sum(math.log1p(v) for v in vals) / len(vals)) - 1.0
    return {
        "n": len(vals),
        "missing": missing,
        "geomean": geo,
        "p95": s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)],
        "max": s[-1],
    }


def evaluate_log(
    log: str,
    dbs: Optional[Dict[str, str]] = None,
    *,
    with_model: bool = False,
    sm: Optional[int] = None,
) -> Dict[str, Any]:

    import oasr.jit.gemm as jg
    from oasr.jit.core import _get_target_sm
    from oasr.tune import database

    rows = load_log(log)
    sm = int(sm if sm is not None else (rows[0]["sm"] if rows else _get_target_sm()))
    pts = _points(rows)

    selectors: Dict[str, Callable[[Tuple], Optional[str]]] = {}
    fallback_of = {"gemm_log_softmax": "fused"}

    selectors["default"] = lambda k: fallback_of.get(k[0], "default")

    shipped = jg._GEMM_HEURISTIC_RULES.get(sm)

    def _shipped(k):
        op, N, K, dtype, M = k
        rules = (shipped or {}).get((op, N, K))
        if not rules:
            return fallback_of.get(op, "default")
        for m_max, choice in rules:
            if m_max is None or M <= m_max:
                return jg.gemm_config_id(choice)
        return fallback_of.get(op, "default")

    selectors["shipped"] = _shipped

    for name, path in (dbs or {}).items():
        tf = database.TuningFile.from_json(json.load(open(path)))

        def _sel(k, tf=tf):
            op, N, K, dtype, M = k
            dt = {"float16": "fp16", "bfloat16": "bf16"}.get(dtype, "half")
            for cls in (dt, "half"):
                e = tf.entries.get(jg.gemm_sig_key(op, N, K, cls))
                if e is not None:
                    c = e.lookup(M)
                    if c is not None:
                        return c
            return fallback_of.get(op, "default")

        selectors[name] = _sel

    if with_model:
        from oasr.tune import arch as arch_mod
        from oasr.tune.cost_model import GemmCostModel

        model = GemmCostModel.for_arch(arch_mod.current(measure=True))
        model.calibrate(rows)

        def _model(k):
            configs = {
                c: r["params"] for c, r in pts[k].items() if r["params"].get("kind") == "cutlass"
            }
            ranked = model.rank(configs, k[4], k[1], k[2])
            return ranked[0][0] if ranked else None

        selectors["model"] = _model

    per_sel: Dict[str, List[float]] = defaultdict(list)
    per_sig: Dict[Tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for k, arms in pts.items():
        best = min(float(r["median_ms"]) for r in arms.values())
        for name, sel in selectors.items():
            c = sel(k)
            r = arms.get(c) if c is not None else None
            reg = (float(r["median_ms"]) / best - 1.0) if r is not None else math.inf
            per_sel[name].append(reg)
            per_sig[k[:4]][name].append(reg)
    return {
        "sm": sm,
        "points": len(pts),
        "selectors": {n: _stats(v) for n, v in per_sel.items()},
        "signatures": {
            "|".join(map(str, s)): {n: _stats(v) for n, v in d.items()}
            for s, d in sorted(per_sig.items())
        },
    }


def format_report(rep: Dict[str, Any]) -> str:
    lines = [
        f"sm{rep['sm']}: {rep['points']} measured points (oracle = best measured arm)",
        f"    {'selector':<12} {'n':>5} {'geomean':>9} {'p95':>8} {'max':>8} {'missing':>8}",
    ]
    for name, st in rep["selectors"].items():
        if not st.get("n"):
            lines.append(f"    {name:<12} {'-':>5}")
            continue
        lines.append(
            f"    {name:<12} {st['n']:>5} {st['geomean']:>8.1%} {st['p95']:>7.1%} "
            f"{st['max']:>7.1%} {st['missing']:>8}"
        )
    return "\n".join(lines)
