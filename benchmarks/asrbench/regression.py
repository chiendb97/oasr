# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``compare`` -- performance regression checks between a baseline and a candidate.

Results are matched by :attr:`BenchmarkResult.key` (benchmark name, sweep point,
workload id), so only measurements of the *same inputs under the same protocol*
are ever diffed.  Before any metric is compared, the two runs must be
comparable: same GPU model (unless ``--allow-hardware-change``), same backend
configuration (unless ``--allow-config-change`` -- which is how two frameworks
are compared deliberately), and per result the same streaming mode.

A metric regresses when it moves in its worse direction (from
:mod:`benchmarks.asrbench.metrics`) by more than **all** of:

* ``max_regression_pct`` of the baseline value (relative),
* ``min_abs`` (an absolute floor -- a 0.3 ms move of a 2 ms p50 is not news),
* ``noise_sigmas`` x the combined across-repetition standard deviation, when
  both runs recorded one (``<metric>.rep_std``).

Absolute-only rules (``max_regression_abs``) suit rates such as ``error_rate``.
The thresholds are inputs, not facts: the defaults are a starting point for a
quiet machine, and a CI gate should carry its own file.

Exit codes: 0 no regression, 1 regression, 2 not comparable / bad input.
"""

from __future__ import annotations

import fnmatch
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from benchmarks.asrbench.metrics import HIGHER, LOWER, direction, min_samples_for
from benchmarks.asrbench.results import STATUS_OK, load_run

EXIT_OK, EXIT_REGRESSION, EXIT_INCOMPARABLE = 0, 1, 2


@dataclass
class Rule:
    metric: str  # fnmatch pattern
    max_regression_pct: Optional[float] = None
    max_regression_abs: Optional[float] = None
    min_abs: float = 0.0
    #: Skip results whose sweep point matches all of these (e.g. ``{mode: open}``).
    unless: Optional[Dict[str, Any]] = None

    def applies(self, point: Dict[str, Any]) -> bool:
        if not self.unless:
            return True
        return not all(point.get(k) == v for k, v in self.unless.items())


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "noise_sigmas": 2.0,
    "rules": [
        {"metric": "latency_ms.p50", "max_regression_pct": 10, "min_abs": 1.0},
        {"metric": "latency_ms.p95", "max_regression_pct": 15, "min_abs": 2.0},
        {"metric": "latency_ms.p99", "max_regression_pct": 20, "min_abs": 5.0},
        {"metric": "batch_latency_ms.p50", "max_regression_pct": 10, "min_abs": 1.0},
        {"metric": "bucket.*.latency_ms.p50", "max_regression_pct": 15, "min_abs": 1.0},
        {"metric": "first_partial_ms.p50", "max_regression_pct": 15, "min_abs": 10.0},
        {"metric": "final_latency_ms.p50", "max_regression_pct": 15, "min_abs": 10.0},
        {"metric": "final_latency_ms.p95", "max_regression_pct": 20, "min_abs": 20.0},
        {"metric": "emission_delay_ms.p50", "max_regression_pct": 15, "min_abs": 10.0},
        # Open loop: below saturation, achieved throughput is the offered rate --
        # not a property of the server.  Latency and errors carry the signal there.
        {"metric": "rtfx", "max_regression_pct": 10, "unless": {"mode": "open"}},
        {"metric": "throughput.req_per_s", "max_regression_pct": 10, "unless": {"mode": "open"}},
        {"metric": "achieved_ratio", "max_regression_abs": 0.05},
        {"metric": "init_s", "max_regression_pct": 25, "min_abs": 0.5},
        {"metric": "first_request_ms", "max_regression_pct": 25, "min_abs": 20.0},
        {"metric": "rss_mb.max", "max_regression_pct": 10, "min_abs": 64.0},
        {"metric": "gpu_alloc_mb.max", "max_regression_pct": 10, "min_abs": 64.0},
        {"metric": "gpu_mem_mb.max", "max_regression_pct": 10, "min_abs": 128.0},
        {"metric": "error_rate", "max_regression_abs": 0.01},
        {"metric": "empty_rate", "max_regression_abs": 0.01},
        {"metric": "late_chunk_pct", "max_regression_abs": 2.0},
        {"metric": "wer_pct", "max_regression_abs": 0.3},
        {"metric": "changed_pct", "max_regression_abs": 1.0},
    ],
}


@dataclass
class Thresholds:
    rules: List[Rule]
    noise_sigmas: float = 2.0


def load_thresholds(path: Optional[Path]) -> Thresholds:
    doc = DEFAULT_THRESHOLDS if path is None else yaml.safe_load(Path(path).read_text())
    rules = [Rule(**r) for r in doc.get("rules", [])]
    return Thresholds(rules=rules, noise_sigmas=float(doc.get("noise_sigmas", 2.0)))


# ---------------------------------------------------------------------------
# Loading either a run or a validation
# ---------------------------------------------------------------------------


@dataclass
class Comparable:
    path: Path
    label: str
    kind: str  # "run" | "validation"
    meta: Dict[str, Any]
    #: key -> (summary, extra) where extra carries streaming mode etc.
    results: Dict[str, Tuple[Dict[str, float], Dict[str, Any]]]


def load_comparable(path: Path) -> Comparable:
    path = Path(path)
    vfile = path / "validation.json" if path.is_dir() else path
    if vfile.name == "validation.json" and vfile.is_file():
        doc = json.loads(vfile.read_text(encoding="utf-8"))
        summary = {
            "wer_pct": doc.get("wer_pct"),
            "empty_rate": doc.get("empty_rate"),
        }
        ref = doc.get("reference") or {}
        if ref.get("changed_pct") is not None:
            summary["changed_pct"] = ref["changed_pct"]
        key = f"validate@{doc.get('workload', {}).get('id', '')}"
        return Comparable(
            path=vfile.parent,
            label=doc.get("backend", {}).get("name", "validation"),
            kind="validation",
            meta={
                "environment": doc.get("environment", {}),
                "backend": {"name": doc["backend"]["name"], "options": doc["backend"]["options"]},
            },
            results={key: ({k: v for k, v in summary.items() if v is not None}, {})},
        )
    run = load_run(path)
    results = {}
    for r in run.results:
        if r.status != STATUS_OK:
            continue
        results[r.key] = (
            r.summary,
            {"streaming_mode": r.server.get("streaming_mode"), "point": r.point},
        )
    return Comparable(path=run.path, label=run.label, kind="run", meta=run.meta, results=results)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _gpu_names(meta: Dict[str, Any]) -> List[str]:
    return sorted(g.get("name", "") for g in meta.get("environment", {}).get("gpus", []))


#: Options that name a process, not a configuration: a restarted server has a new PID.
VOLATILE_OPTIONS = ("server_pid",)


def _backend_config(meta: Dict[str, Any]) -> Dict[str, Any]:
    b = meta.get("backend", {})
    opts = {k: v for k, v in (b.get("options") or {}).items() if k not in VOLATILE_OPTIONS}
    out: Dict[str, Any] = {"name": b.get("name"), "options": opts}
    if opts.get("base_url"):
        # An attached server's configuration is not in the options; what it
        # reports about itself (checkpoint, dtype, batch width, mode) is.
        served = {
            json.dumps(i["info"].get("model") or {}, sort_keys=True)
            for i in (b.get("instances") or {}).values()
            if isinstance(i, dict) and i.get("info")
        }
        if served:
            out["served"] = [json.loads(x) for x in sorted(served)]
    return out


def _flatten(x: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(x, list) and any(isinstance(v, dict) for v in x):
        out = {}
        for i, v in enumerate(x):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: x}


def _config_diff(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """Only the fields that differ: ``options.dtype: 'float16' vs 'bfloat16'``."""
    fa, fb = _flatten(a), _flatten(b)
    return "; ".join(
        f"{k}: {fa.get(k, '<unset>')!r} vs {fb.get(k, '<unset>')!r}"
        for k in sorted(set(fa) | set(fb))
        if fa.get(k) != fb.get(k)
    )


def compatibility(
    base: Comparable, cand: Comparable, *, allow_hardware_change: bool, allow_config_change: bool
) -> Tuple[List[str], List[str]]:
    """``(blocking problems, notes)``."""
    problems, notes = [], []
    if base.kind != cand.kind:
        problems.append(f"cannot compare a {base.kind} with a {cand.kind}")
    gb, gc = _gpu_names(base.meta), _gpu_names(cand.meta)
    if gb != gc:
        msg = f"hardware differs: {gb or ['none']} vs {gc or ['none']}"
        (notes if allow_hardware_change else problems).append(msg)
    cb, cc = _backend_config(base.meta), _backend_config(cand.meta)
    if cb != cc:
        msg = f"backend configuration differs: {_config_diff(cb, cc)}"
        (notes if allow_config_change else problems).append(msg)
    for side, c in (("baseline", base), ("candidate", cand)):
        sim = c.meta.get("simulated") or {}
        if sim.get("backend") or sim.get("inputs"):
            notes.append(f"{side} is simulated ({sim}) -- its numbers are not measurements")
    pb = base.meta.get("profile", {}).get("name")
    pc = cand.meta.get("profile", {}).get("name")
    if pb != pc:
        notes.append(
            f"profiles differ ({pb} vs {pc}); only results with identical keys are compared"
        )
    return problems, notes


def _rule_for(rules: List[Rule], key: str) -> Optional[Rule]:
    for r in rules:
        if fnmatch.fnmatchcase(key, r.metric):
            return r
    return None


def compare_summaries(
    base: Dict[str, float],
    cand: Dict[str, float],
    thresholds: Thresholds,
    point: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """One verdict per thresholded metric present in both summaries."""
    out = []
    for key in sorted(set(base) & set(cand)):
        rule = _rule_for(thresholds.rules, key)
        if rule is None or not rule.applies(point or {}):
            continue
        better = direction(key)
        if better not in (LOWER, HIGHER):
            continue
        b, c = base[key], cand[key]
        if b is None or c is None or any(isinstance(x, float) and math.isnan(x) for x in (b, c)):
            continue
        b, c = float(b), float(c)
        # A percentile its sample count cannot resolve is noise; do not judge it.
        head, _, stat = key.rpartition(".")
        if stat.startswith("p") and stat[1:].isdigit():
            need = min_samples_for(float(stat[1:]))
            nb, nc = base.get(f"{head}.n"), cand.get(f"{head}.n")
            if (nb is not None and nb < need) or (nc is not None and nc < need):
                out.append(
                    {
                        "metric": key,
                        "better": better,
                        "baseline": b,
                        "candidate": c,
                        "delta": c - b,
                        "delta_pct": (c - b) / abs(b) * 100.0 if b else None,
                        "limit": None,
                        "noise": None,
                        "verdict": "insufficient_samples",
                    }
                )
                continue
        delta = c - b
        worse = delta if better == LOWER else -delta
        rel = (delta / abs(b) * 100.0) if b else None
        limits = []
        if rule.max_regression_abs is not None:
            limits.append(rule.max_regression_abs)
        if rule.max_regression_pct is not None:
            limits.append(rule.max_regression_pct / 100.0 * abs(b))
        if not limits:
            continue
        limit = max(max(limits), rule.min_abs)
        sb, sc = base.get(f"{key}.rep_std"), cand.get(f"{key}.rep_std")
        noise = None
        if sb is not None and sc is not None:
            noise = thresholds.noise_sigmas * math.sqrt(float(sb) ** 2 + float(sc) ** 2)
            limit = max(limit, noise)
        verdict = "regression" if worse > limit else ("improvement" if -worse > limit else "same")
        out.append(
            {
                "metric": key,
                "better": better,
                "baseline": b,
                "candidate": c,
                "delta": delta,
                "delta_pct": rel,
                "limit": limit,
                "noise": noise,
                "verdict": verdict,
            }
        )
    return out


def compare(
    base: Comparable,
    cand: Comparable,
    thresholds: Thresholds,
    *,
    allow_hardware_change: bool = False,
    allow_config_change: bool = False,
) -> Tuple[Dict[str, Any], int]:
    problems, notes = compatibility(
        base,
        cand,
        allow_hardware_change=allow_hardware_change,
        allow_config_change=allow_config_change,
    )
    summary: Dict[str, Any] = {
        "schema": "oasr.asrbench.comparison",
        "baseline": {"path": str(base.path), "label": base.label},
        "candidate": {"path": str(cand.path), "label": cand.label},
        "problems": problems,
        "notes": notes,
        "matched": 0,
        "unmatched_baseline": sorted(set(base.results) - set(cand.results)),
        "unmatched_candidate": sorted(set(cand.results) - set(base.results)),
        "regressions": [],
        "improvements": [],
        "unchanged": [],
        "insufficient_samples": [],
        "skipped": [],
    }
    if problems:
        summary["status"] = "incomparable"
        return summary, EXIT_INCOMPARABLE
    for key in sorted(set(base.results) & set(cand.results)):
        (sb, eb), (sc, ec) = base.results[key], cand.results[key]
        if eb.get("streaming_mode") != ec.get("streaming_mode"):
            summary["skipped"].append(
                {
                    "result": key,
                    "reason": f"streaming modes differ ({eb.get('streaming_mode')} vs "
                    f"{ec.get('streaming_mode')})",
                }
            )
            continue
        summary["matched"] += 1
        for v in compare_summaries(sb, sc, thresholds, eb.get("point")):
            v["result"] = key
            bucket = {
                "regression": "regressions",
                "improvement": "improvements",
                "insufficient_samples": "insufficient_samples",
            }.get(v["verdict"], "unchanged")
            summary[bucket].append(v)
    if summary["matched"] == 0:
        summary["problems"].append(
            "no result is present in both runs (different profiles or inputs?)"
        )
        summary["status"] = "incomparable"
        return summary, EXIT_INCOMPARABLE
    summary["status"] = "regression" if summary["regressions"] else "pass"
    return summary, EXIT_REGRESSION if summary["regressions"] else EXIT_OK


def compare_paths(
    baseline: Path,
    candidate: Path,
    thresholds: Optional[Thresholds] = None,
    **kw: Any,
) -> Tuple[Dict[str, Any], int]:
    try:
        base, cand = load_comparable(baseline), load_comparable(candidate)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return {"status": "error", "problems": [str(exc)]}, EXIT_INCOMPARABLE
    return compare(base, cand, thresholds or load_thresholds(None), **kw)


def render_text(summary: Dict[str, Any], verbose: bool = False) -> str:
    lines = [
        f"[asrbench] compare {summary.get('baseline', {}).get('label')} -> "
        f"{summary.get('candidate', {}).get('label')}: {summary.get('status', '').upper()}"
    ]
    for p in summary.get("problems", []):
        lines.append(f"  NOT COMPARABLE: {p}")
    for n in summary.get("notes", []):
        lines.append(f"  note: {n}")
    if "matched" in summary:
        lines.append(
            f"  {summary['matched']} result(s) matched; "
            f"{len(summary['unmatched_baseline'])} only in baseline, "
            f"{len(summary['unmatched_candidate'])} only in candidate"
        )

    def fmt(v: Dict[str, Any]) -> str:
        pct = "" if v["delta_pct"] is None else f" ({v['delta_pct']:+.1f}%)"
        limit = "n/a" if v["limit"] is None else f"{v['limit']:.3g}"
        return (
            f"    {v['result']}  {v['metric']}: {v['baseline']:.4g} -> {v['candidate']:.4g}"
            f"{pct}  [{v['better']} is better; limit {limit}]"
        )

    for title, key in (("REGRESSIONS", "regressions"), ("improvements", "improvements")):
        if summary.get(key):
            lines.append(f"  {title}:")
            lines.extend(fmt(v) for v in summary[key])
    if verbose and summary.get("unchanged"):
        lines.append("  unchanged:")
        lines.extend(fmt(v) for v in summary["unchanged"])
    if summary.get("insufficient_samples"):
        lines.append(
            f"  {len(summary['insufficient_samples'])} percentile(s) not judged: too few "
            "samples to resolve them (add requests or repetitions)"
        )
    for s in summary.get("skipped", []):
        lines.append(f"  skipped {s['result']}: {s['reason']}")
    return "\n".join(lines)
