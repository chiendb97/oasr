# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``report`` -- tables (and optional charts) from stored runs.  Never runs inference.

One run renders as its own tables; several runs (framework versions, backends,
machines) render side by side with a delta against the first, matched on the
same result keys ``compare`` uses.  Output::

    <out>/report.md      the tables, run metadata, unsupported/failed results, warnings
    <out>/summary.csv    one row per (run, result), every summary metric as a column
    <out>/plots/*.png    with --plots and matplotlib installed

Rows are only ever placed side by side when their keys match, so a batch-8
result is never shown next to a batch-16 one, and two different workloads are
never pooled into one row.
"""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmarks.asrbench.metrics import HIGHER, LOWER, direction
from benchmarks.asrbench.results import STATUS_OK, BenchmarkResult, Run, load_run

# Columns per benchmark kind: (summary key, header).
TABLES: Dict[str, List[Tuple[str, str]]] = {
    "cold_start": [
        ("init_s", "init s"),
        ("first_request_ms", "first req ms"),
        ("warm_request_ms", "warm req ms"),
    ],
    "single": [
        ("requests.ok", "ok"),
        ("latency_ms.p50", "p50 ms"),
        ("latency_ms.p95", "p95 ms"),
        ("latency_ms.p99", "p99 ms"),
        ("rtf.mean", "RTF mean"),
        ("rtfx", "RTFx"),
        ("gpu_alloc_mb.max", "GPU alloc MiB"),
    ],
    "batch": [
        ("batch_latency_ms.p50", "batch p50 ms"),
        ("latency_ms.p50", "req p50 ms"),
        ("rtfx", "RTFx"),
        ("throughput.req_per_s", "req/s"),
        ("padding_ratio", "padding"),
        ("gpu_alloc_mb.max", "GPU alloc MiB"),
        ("error_rate", "errors"),
    ],
    "throughput": [
        ("requests.ok", "ok"),
        ("rtfx", "RTFx"),
        ("rtfx.rep_std", "RTFx σ"),
        ("throughput.req_per_s", "req/s"),
        ("latency_ms.p50", "completion p50 ms"),
        ("gpu_util_pct.mean", "GPU util %"),
    ],
    "load": [
        ("offered_rate", "offered req/s"),
        ("throughput.req_per_s", "achieved req/s"),
        ("latency_ms.p50", "p50 ms"),
        ("latency_ms.p95", "p95 ms"),
        ("latency_ms.p99", "p99 ms"),
        ("error_rate", "errors"),
        ("send_lag_ms.p99", "client lag p99 ms"),
        ("queue_wait_ms.p50", "queue p50 ms"),
    ],
    "streaming": [
        ("sessions.ok", "ok"),
        ("first_partial_ms.p50", "first partial p50 ms"),
        ("final_latency_ms.p50", "final p50 ms"),
        ("final_latency_ms.p95", "final p95 ms"),
        ("emission_delay_ms.p50", "emission p50 ms"),
        ("partials_per_session.mean", "partials/session"),
        ("late_chunk_pct", "late chunks %"),
        ("rtfx", "RTFx (unpaced)"),
    ],
    "longform": [
        ("requests.ok", "ok"),
        ("audio_s", "audio s"),
        ("latency_ms.max", "max latency ms"),
        ("rtfx", "RTFx"),
        ("rss_mb.max", "RSS MiB"),
        ("gpu_mem_mb.max", "GPU mem MiB"),
    ],
}

RESOURCE_COLUMNS = [
    ("cpu_pct.mean", "CPU %"),
    ("rss_mb.max", "RSS MiB"),
    ("gpu_util_pct.mean", "GPU util %"),
    ("gpu_mem_mb.max", "GPU mem MiB"),
    ("gpu_alloc_mb.max", "GPU alloc MiB"),
    ("gpu_power_w.mean", "power W"),
    ("energy_j_per_audio_h", "J / audio-h"),
]


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        if v.is_integer() and abs(v) < 1e6:
            return str(int(v))
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.3g}"
    return str(v)


def _delta(key: str, base: Optional[float], value: Optional[float]) -> str:
    if base in (None, 0) or value is None:
        return ""
    pct = (value - base) / abs(base) * 100.0
    better = direction(key)
    mark = ""
    if abs(pct) >= 1.0 and better in (LOWER, HIGHER):
        good = (pct < 0) == (better == LOWER)
        mark = " ✓" if good else " ✗"
    return f" ({pct:+.0f}%{mark})"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _point_label(r: BenchmarkResult) -> str:
    return ", ".join(f"{k}={v}" for k, v in r.point.items()) or "–"


def _run_rows(runs: Sequence[Run]) -> List[Sequence[str]]:
    rows = []
    for run in runs:
        env = run.meta.get("environment", {})
        inst = run.meta.get("backend", {}).get("instances", {})
        info = next(iter(inst.values()), {}).get("info", {}) if inst else {}
        model = info.get("model", {})
        gpus = ", ".join(g.get("name", "") for g in env.get("gpus", [])) or "none"
        rows.append(
            [
                run.label,
                run.run_id,
                run.meta.get("backend", {}).get("name", ""),
                str(model.get("id") or model.get("path") or ""),
                str(info.get("dtype", "")),
                gpus,
                (env.get("git", {}).get("short") or "")
                + ("+dirty" if env.get("git", {}).get("dirty") else ""),
                run.meta.get("profile", {}).get("name", ""),
                str(run.meta.get("started_utc", ""))[:19],
            ]
        )
    return rows


def render_markdown(runs: Sequence[Run]) -> str:
    lines: List[str] = ["# ASR inference benchmark report", ""]
    simulated = [r.label for r in runs if any((r.meta.get("simulated") or {}).values())]
    if simulated:
        lines += [
            f"> **Simulated:** {', '.join(simulated)} used the mock backend or synthetic "
            "audio. These numbers test the suite; they are not performance measurements.",
            "",
        ]
    lines += ["## Runs", ""]
    lines += _table(
        ["label", "run", "backend", "model", "dtype", "GPU", "commit", "profile", "started (UTC)"],
        _run_rows(runs),
    )
    lines += ["", "## Workloads", ""]
    seen: "OrderedDict[tuple, List[str]]" = OrderedDict()
    for run in runs:
        for name, w in run.meta.get("workloads", {}).items():
            key = (
                name,
                w.get("id", ""),
                w.get("dataset", ""),
                w.get("num_items"),
                w.get("total_audio_s"),
            )
            seen.setdefault(key, []).append(run.label)
    wrows = [
        [
            name,
            ds,
            str(n),
            _fmt(audio),
            wid,
            "all" if len(labels) == len(runs) else ", ".join(labels),
        ]
        for (name, wid, ds, n, audio), labels in seen.items()
    ]
    lines += _table(["workload", "dataset", "items", "audio s", "workload id", "runs"], wrows)

    by_key: List[Dict[str, BenchmarkResult]] = [{r.key: r for r in run.results} for run in runs]
    order: "OrderedDict[str, BenchmarkResult]" = OrderedDict()
    for run in runs:
        for r in run.results:
            order.setdefault(r.key, r)

    kinds = OrderedDict((r.kind, None) for r in order.values())
    for kind in kinds:
        cols = TABLES.get(kind, [("latency_ms.p50", "p50 ms"), ("rtfx", "RTFx")])
        keys = [k for k, r in order.items() if r.kind == kind]
        lines += ["", f"## {kind}", ""]
        headers = (
            ["benchmark", "point"] + ([] if len(runs) == 1 else ["run"]) + [h for _, h in cols]
        )
        rows = []
        for key in keys:
            proto = order[key]
            for i, run in enumerate(runs):
                r = by_key[i].get(key)
                if r is None or r.status != STATUS_OK:
                    continue
                b = by_key[0].get(key) if i else None
                cells = []
                for metric, _ in cols:
                    v = r.summary.get(metric)
                    cell = _fmt(v)
                    if b is not None and b.status == STATUS_OK:
                        cell += _delta(metric, b.summary.get(metric), v)
                    cells.append(cell)
                rows.append(
                    [proto.benchmark, _point_label(proto)]
                    + ([] if len(runs) == 1 else [run.label])
                    + cells
                )
        if rows:
            lines += _table(headers, rows)
        if kind == "single":
            lines += _bucket_table(runs, by_key, keys)

    lines += ["", "## Resources (measured window)", ""]
    rrows = []
    for run in runs:
        for r in run.results:
            if r.status == STATUS_OK and any(k in r.summary for k, _ in RESOURCE_COLUMNS):
                rrows.append(
                    [run.label, r.result_id] + [_fmt(r.summary.get(k)) for k, _ in RESOURCE_COLUMNS]
                )
    lines += (
        _table(["run", "result"] + [h for _, h in RESOURCE_COLUMNS], rrows)
        if rrows
        else ["(no resource samples)"]
    )

    other = [(run, r) for run in runs for r in run.results if r.status != STATUS_OK]
    if other:
        lines += ["", "## Not measured", ""]
        lines += _table(
            ["run", "result", "status", "reason"],
            [
                [run.label, r.result_id, r.status, (r.reason or "").replace("|", "/")]
                for run, r in other
            ],
        )
    warns = [(run, r, w) for run in runs for r in run.results for w in r.warnings]
    if warns:
        lines += ["", "## Warnings", ""]
        lines += [f"- **{run.label} / {r.result_id}**: {w}" for run, r, w in warns]
    lines += [
        "",
        "## Definitions",
        "",
        "Latency percentiles are nearest-rank over successful requests; `RTFx` is total audio "
        "over the benchmark's wall time (higher is faster); `RTF` is per-request latency over "
        "audio duration. Full definitions: `docs/asrbench.md` and `benchmarks/asrbench/metrics.py`.",
        "",
    ]
    return "\n".join(lines)


def _bucket_table(
    runs: Sequence[Run], by_key: List[Dict[str, BenchmarkResult]], keys: List[str]
) -> List[str]:
    lines: List[str] = []
    for key in keys:
        r0 = next((m[key] for m in by_key if key in m and m[key].status == STATUS_OK), None)
        if r0 is None:
            continue
        buckets = sorted(
            {
                k.split(".")[1]
                for k in r0.summary
                if k.startswith("bucket.") and k.endswith(".latency_ms.n")
            },
            key=_bucket_order,
        )
        if not buckets:
            continue
        lines += ["", f"Latency by duration bucket — {r0.benchmark}", ""]
        headers = (
            ["bucket"]
            + [f"{run.label} p50 ms" for run in runs]
            + [f"{run.label} p95 ms" for run in runs]
        )
        rows = []
        for b in buckets:
            p50 = [
                _fmt(m.get(key).summary.get(f"bucket.{b}.latency_ms.p50")) if m.get(key) else "–"
                for m in by_key
            ]
            p95 = [
                _fmt(m.get(key).summary.get(f"bucket.{b}.latency_ms.p95")) if m.get(key) else "–"
                for m in by_key
            ]
            rows.append([b] + p50 + p95)
        lines += _table(headers, rows)
    return lines


def _bucket_order(name: str) -> Tuple[int, str]:
    from benchmarks.asrbench.workloads import DEFAULT_BUCKETS

    names = [b[0] for b in DEFAULT_BUCKETS]
    if name in names:
        return (names.index(name), name)
    if name.startswith("fixed_"):
        try:
            return (100 + int(float(name[6:-1]) * 100), name)
        except ValueError:
            pass
    return (10_000, name)


def write_csv(runs: Sequence[Run], path: Path) -> None:
    rows = []
    metric_keys: "OrderedDict[str, None]" = OrderedDict()
    point_keys: "OrderedDict[str, None]" = OrderedDict()
    for run in runs:
        for r in run.results:
            for k in r.point:
                point_keys.setdefault(k, None)
            for k in r.summary:
                metric_keys.setdefault(k, None)
    for run in runs:
        for r in run.results:
            row = {
                "run": run.label,
                "run_id": run.run_id,
                "backend": run.meta.get("backend", {}).get("name", ""),
                "benchmark": r.benchmark,
                "kind": r.kind,
                "mode": r.mode,
                "workload": r.workload,
                "workload_id": r.workload_id,
                "status": r.status,
            }
            row.update({f"point.{k}": r.point.get(k) for k in point_keys})
            row.update({k: r.summary.get(k) for k in metric_keys})
            rows.append(row)
    fields = list(rows[0].keys()) if rows else ["run"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _series(
    runs: Sequence[Run], kind: str, x: str, y: str, group: Callable[[BenchmarkResult], str]
) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for run in runs:
        for r in run.results:
            if r.kind != kind or r.status != STATUS_OK:
                continue
            xv = r.point.get(x, r.summary.get(x))
            yv = r.summary.get(y)
            if xv is None or yv is None:
                continue
            label = f"{run.label} {group(r)}".strip() if len(runs) > 1 else group(r) or r.benchmark
            out.setdefault(label, []).append((float(xv), float(yv)))
    return {k: sorted(v) for k, v in out.items()}


def write_plots(runs: Sequence[Run], out_dir: Path) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[asrbench] matplotlib is not installed; skipping plots")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    specs = [
        (
            "batch",
            "batch_size",
            "rtfx",
            lambda r: r.point.get("composition", ""),
            "throughput_by_batch",
            "RTFx",
            True,
        ),
        (
            "batch",
            "batch_size",
            "batch_latency_ms.p50",
            lambda r: r.point.get("composition", ""),
            "latency_by_batch",
            "batch latency p50 (ms)",
            True,
        ),
        (
            "load",
            "concurrency",
            "throughput.req_per_s",
            lambda r: r.benchmark,
            "throughput_by_concurrency",
            "req/s",
            True,
        ),
        (
            "load",
            "concurrency",
            "latency_ms.p95",
            lambda r: r.benchmark,
            "latency_by_concurrency",
            "latency p95 (ms)",
            True,
        ),
        (
            "load",
            "throughput.req_per_s",
            "latency_ms.p99",
            lambda r: r.benchmark,
            "latency_vs_throughput",
            "latency p99 (ms)",
            False,
        ),
        (
            "streaming",
            "sessions",
            "final_latency_ms.p95",
            lambda r: f"chunk {r.point.get('chunk_ms')} ms",
            "streaming_final_latency",
            "final latency p95 (ms)",
            True,
        ),
        (
            "streaming",
            "sessions",
            "first_partial_ms.p50",
            lambda r: f"chunk {r.point.get('chunk_ms')} ms",
            "streaming_first_partial",
            "first partial p50 (ms)",
            True,
        ),
    ]
    for kind, x, y, group, name, ylabel, logx in specs:
        series = _series(runs, kind, x, y, group)
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for label, pts in series.items():
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
        if logx:
            from matplotlib.ticker import ScalarFormatter

            ax.set_xscale("log", base=2)
            xs = sorted({p[0] for pts in series.values() for p in pts})
            ax.set_xticks(xs)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.minorticks_off()
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    # Latency by duration bucket (single).
    rows = []
    for run in runs:
        for r in run.results:
            if r.kind == "single" and r.status == STATUS_OK:
                for k, v in r.summary.items():
                    if k.startswith("bucket.") and k.endswith(".latency_ms.p50"):
                        rows.append((f"{run.label} {r.benchmark}", k.split(".")[1], v))
    if rows:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        labels = sorted({b for _, b, _ in rows}, key=_bucket_order)
        groups = sorted({g for g, _, _ in rows})
        width = 0.8 / max(1, len(groups))
        for gi, g in enumerate(groups):
            vals = {b: v for gg, b, v in rows if gg == g}
            ax.bar(
                [i + gi * width for i in range(len(labels))],
                [vals.get(b, 0.0) for b in labels],
                width=width,
                label=g,
            )
        ax.set_xticks([i + width * (len(groups) - 1) / 2 for i in range(len(labels))])
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("latency p50 (ms)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / "latency_by_duration.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def write_report(paths: Sequence[Path], out: Optional[Path] = None, plots: bool = False) -> Path:
    runs = [load_run(p) for p in paths]
    out = out or runs[0].path / "report"
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(runs)
    if plots:
        images = write_plots(runs, out / "plots")
        if images:
            md += (
                "\n## Charts\n\n" + "\n".join(f"![{p.stem}](plots/{p.name})" for p in images) + "\n"
            )
    (out / "report.md").write_text(md, encoding="utf-8")
    write_csv(runs, out / "summary.csv")
    return out / "report.md"
