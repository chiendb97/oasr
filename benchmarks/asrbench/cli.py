# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``python -m benchmarks.asrbench <command>`` -- the suite's command line.

    datasets                       registered datasets, prepared state, licence, size
    prepare NAME [--download]      index or fetch a dataset (fetching is opt-in)
    profiles [NAME]                list profiles, or print one resolved
    backends                       registered backends, options, availability
    run --profile P --backend B    measure; writes a run directory; exit 1 if a point failed
    runs                           list stored runs, newest first
    validate --backend B           WER guardrail on a fixed labelled subset (untimed)
    compare BASE CAND              regression check; exit 1 on a regression, a failed
                                   point, or a result or metric the candidate lost
    report RUN [RUN ...]           Markdown (+CSV, +plots) from stored runs
    recompute RUN                  re-derive summaries from raw records
    save-baseline RUN --name N     keep a run as a named baseline

It is a separate command from ``benchmarks/run.py`` because it is not one
measurement but a workflow (prepare -> run -> compare -> report); the
``run.py`` families remain the per-component harnesses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


def _kv(pairs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw) if raw != "" else ""
    return out


def _csv_list(value: Optional[str]) -> List[str]:
    return [v for v in (value or "").split(",") if v]


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_datasets(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.datasets import DATASETS, PreparedDataset, default_data_root

    root = Path(args.data_dir) if args.data_dir else default_data_root()
    rows = []
    for name, cls in DATASETS.items():
        adapter = cls()
        prepared = []
        base = root / name
        if base.is_dir():
            for sub in sorted(p for p in base.iterdir() if (p / "dataset.json").is_file()):
                card = PreparedDataset.load(sub).card
                prepared.append(
                    {
                        "subset": card.subset,
                        "items": card.num_items,
                        "hours": round(card.total_duration_s / 3600, 2),
                        "revision": card.revision,
                    }
                )
        rows.append(
            {
                "name": name,
                "license": adapter.license,
                "download": adapter.download_size,
                "purpose": adapter.purpose,
                "subsets": list(adapter.subsets),
                "prepared": prepared,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"data root: {root}\n")
    for r in rows:
        print(f"{r['name']}  [{r['license']}]  download: {r['download']}")
        print(f"    purpose: {r['purpose']}")
        if r["subsets"]:
            print(f"    subsets: {', '.join(r['subsets'])}")
        for p in r["prepared"]:
            print(
                f"    prepared: {p['subset']}  {p['items']} items, {p['hours']} h  ({p['revision']})"
            )
        if not r["prepared"]:
            print("    prepared: no")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.datasets import DatasetUnavailable, default_data_root, get_adapter

    adapter = get_adapter(args.name)
    root = Path(args.data_dir) if args.data_dir else default_data_root()
    options = _kv(args.option)
    if args.shards:
        options["shards"] = [int(s) for s in _csv_list(args.shards)]
    if args.source:
        options["source"] = args.source
    if args.targets:
        options["targets"] = [float(t) for t in _csv_list(args.targets)]
    if args.per_target is not None:
        options["per_target"] = args.per_target
    try:
        ds = adapter.prepare(
            root,
            args.subset,
            source_dir=Path(args.source_dir) if args.source_dir else None,
            download=args.download,
            options=options,
        )
    except DatasetUnavailable as exc:
        print(f"[asrbench] {exc}", file=sys.stderr)
        return 2
    problems = ds.verify(deep=args.verify)
    card = ds.card
    print(
        f"[asrbench] prepared {card.ref}: {card.num_items} items, "
        f"{card.total_duration_s / 3600:.2f} h, durations {card.duration_stats} -> {ds.root}"
    )
    for p in problems:
        print(f"[asrbench] integrity: {p}", file=sys.stderr)
    return 1 if problems else 0


def cmd_profiles(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.profiles import Profile, builtin_profiles

    if not args.name:
        for name in builtin_profiles():
            p = Profile.load(name)
            print(f"{name:<10} {p.description}")
        return 0
    p = _resolved_profile(args)
    print(yaml.safe_dump(p.to_json(), sort_keys=False, width=100))
    return 0


def cmd_backends(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.backends import BACKENDS, backend_class

    for name in BACKENDS:
        cls = backend_class(name)
        ok, why = cls.available()
        print(f"{name:<13} {'available' if ok else 'UNAVAILABLE: ' + why}")
        print(f"    {cls.description}")
        if args.verbose:
            for opt, help_ in cls.OPTIONS.items():
                print(f"      -o {opt}=...   {help_}")
    return 0


def _resolved_profile(args: argparse.Namespace):
    from benchmarks.asrbench.profiles import Profile

    p = Profile.load(args.profile if hasattr(args, "profile") and args.profile else args.name)
    dr = None
    if getattr(args, "duration_range", None):
        lo, hi = args.duration_range
        dr = [float(lo), None if hi in ("inf", "none") else float(hi)]
    return p.resolved(
        primary=getattr(args, "dataset", None),
        only=_csv_list(getattr(args, "only", None)),
        skip=_csv_list(getattr(args, "skip", None)),
        sweeps={
            "batch_sizes": getattr(args, "batch_sizes", None),
            "concurrency": getattr(args, "concurrency", None),
            "rates": getattr(args, "rates", None),
            "chunk_ms": getattr(args, "chunk_ms", None),
            "sessions": getattr(args, "sessions", None),
            "repetitions": getattr(args, "repetitions", None),
            "warmup": getattr(args, "warmup", None),
            "timeout_s": getattr(args, "timeout_s", None),
        },
        workload_overrides={
            "num_items": getattr(args, "num_items", None),
            "duration_range": dr,
            "seed": getattr(args, "seed", None),
        },
        sets=getattr(args, "set", None) or [],
    )


def cmd_run(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.datasets import DatasetUnavailable
    from benchmarks.asrbench.runner import run

    profile = _resolved_profile(args)
    try:
        path = run(
            profile,
            args.backend,
            _kv(args.option),
            data_root=Path(args.data_dir) if args.data_dir else None,
            results_root=Path(args.out) if args.out else None,
            tag=args.tag or "",
            monitor_interval_s=0.0 if args.no_monitor else args.monitor_interval,
            store_text=not args.no_text,
        )
    except DatasetUnavailable as exc:
        print(f"[asrbench] {exc}", file=sys.stderr)
        return 2
    print(path)
    return _failed_points(path)


def _failed_points(path: Path) -> int:
    """1 when any point of the run at *path* failed, after saying which.

    A failed point is a crash the run recorded and carried on from; exiting 0
    would let ``run && compare`` treat it as a measurement.  An *unsupported*
    point is a declared capability, not a failure.  The run directory is still
    printed first, so a caller can inspect or compare it.
    """
    from benchmarks.asrbench.results import STATUS_FAILED, load_run

    failed = [r for r in load_run(path).results if r.status == STATUS_FAILED]
    for r in failed:
        reason = (r.reason or "").splitlines()
        print(
            f"[asrbench] FAILED {r.result_id}: {reason[0] if reason else 'no reason recorded'}",
            file=sys.stderr,
        )
    if failed:
        print(f"[asrbench] {len(failed)} point(s) failed", file=sys.stderr)
    return 1 if failed else 0


def cmd_validate(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.correctness import Thresholds, run_validation
    from benchmarks.asrbench.datasets import DatasetUnavailable

    thresholds = Thresholds(
        max_wer_pct=args.max_wer,
        max_empty_rate=args.max_empty_rate,
        max_changed_pct=args.max_changed,
        max_drift_wer_pct=args.max_drift_wer,
        max_wer_delta_pct=args.max_wer_delta,
    )
    try:
        report, code = run_validation(
            args.backend,
            _kv(args.option),
            dataset=args.dataset,
            manifest=args.manifest,
            audio_root=Path(args.audio_root) if args.audio_root else None,
            num_items=args.num_items,
            seed=args.seed,
            normalizer=args.normalizer,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            reference=_run_ref(args.reference) if args.reference else None,
            thresholds=thresholds,
            data_root=Path(args.data_dir) if args.data_dir else None,
            out_root=Path(args.out) if args.out else None,
        )
    except DatasetUnavailable as exc:
        print(f"[asrbench] {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    return code


def cmd_runs(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.results import default_results_root, list_runs

    root = default_results_root()
    runs = list_runs(root)[: args.limit or None]
    if args.json:
        print(json.dumps(runs, indent=2))
        return 0
    print(f"results root: {root}\n")
    if not runs:
        print("no runs")
        return 0
    print(
        f"{'run id':<10} {'started (UTC)':<20} {'profile':<10} {'backend':<14} {'tag':<16} ok/total"
    )
    for r in runs:
        state = f"{r['ok']}/{r['results']}" + ("" if r["finished"] else " (incomplete)")
        print(
            f"{r['run_id']:<10} {r['started_utc'][:19]:<20} {r['profile']:<10} "
            f"{r['backend']:<14} {r['tag'][:16]:<16} {state}"
        )
    return 0


def _run_ref(ref: str) -> Path:
    from benchmarks.asrbench.results import resolve_run

    try:
        return resolve_run(ref)
    except ValueError as exc:
        raise SystemExit(f"[asrbench] {exc}") from None


def cmd_compare(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.regression import compare_paths, load_thresholds, render_text

    thresholds = load_thresholds(Path(args.thresholds) if args.thresholds else None)
    summary, code = compare_paths(
        _run_ref(args.baseline),
        _run_ref(args.candidate),
        thresholds,
        allow_hardware_change=args.allow_hardware_change,
        allow_config_change=args.allow_config_change,
        allow_missing=args.allow_missing,
    )
    print(render_text(summary, verbose=args.verbose))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return code


def cmd_report(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.report import write_report

    out = write_report(
        [_run_ref(p) for p in args.runs],
        Path(args.out) if args.out else None,
        plots=args.plots,
    )
    print(out)
    return 0


def cmd_recompute(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.runner import recompute

    report = recompute(_run_ref(args.run))
    print(json.dumps(report, indent=2))
    return 1 if report["mismatches"] else 0


def cmd_save_baseline(args: argparse.Namespace) -> int:
    from benchmarks.asrbench.results import copy_run, default_results_root

    root = Path(args.out) if args.out else default_results_root()
    dest = copy_run(_run_ref(args.run), root / "baselines" / args.name)
    print(dest)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _add_profile_overrides(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("profile overrides (applied to every matching benchmark/workload)")
    g.add_argument("--dataset", help="primary dataset, e.g. ljspeech/all (replaces ${primary})")
    g.add_argument("--only", help="comma list of benchmark names or kinds to keep")
    g.add_argument("--skip", help="comma list of benchmark names or kinds to drop")
    g.add_argument("--batch-sizes", type=int, nargs="+")
    g.add_argument("--concurrency", type=int, nargs="+", help="closed-loop levels")
    g.add_argument("--rates", type=float, nargs="+", help="open-loop rates (req/s)")
    g.add_argument("--chunk-ms", type=int, nargs="+")
    g.add_argument("--sessions", type=int, nargs="+", help="concurrent streaming sessions")
    g.add_argument("--repetitions", type=int)
    g.add_argument("--warmup", type=int)
    g.add_argument("--timeout-s", type=float)
    g.add_argument("--num-items", type=int, help="items per workload")
    g.add_argument("--duration-range", nargs=2, metavar=("LO", "HI"), help="seconds; HI may be inf")
    g.add_argument("--seed", type=int)
    g.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="any field: defaults.timeout_s=30, workloads.utterances.num_items=50, "
        "batch.batch_sizes=[1,2] (benchmark by name)",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.asrbench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", help="prepared datasets ($ASRBENCH_DATA_DIR)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("datasets", help="list datasets")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_datasets)

    p = sub.add_parser("prepare", help="prepare a dataset")
    p.add_argument("name")
    p.add_argument("--subset")
    p.add_argument(
        "--source-dir",
        help="use a local copy instead of downloading (manifest: the .jsonl or its directory)",
    )
    p.add_argument("--download", action="store_true", help="allow fetching (explicit opt-in)")
    p.add_argument("--verify", action="store_true", help="also check every file's duration")
    p.add_argument("--shards", help="earnings21: comma list of shard indices (default 8)")
    p.add_argument("--source", help="longform: source dataset, e.g. librispeech/test-clean")
    p.add_argument("--targets", help="longform: comma list of target seconds")
    p.add_argument("--per-target", type=int, help="longform: recordings per target")
    p.add_argument("-o", "--option", action="append", default=[], metavar="K=V")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("profiles", help="list or show profiles")
    p.add_argument("name", nargs="?")
    _add_profile_overrides(p)
    p.set_defaults(fn=cmd_profiles)

    p = sub.add_parser("backends", help="list backends")
    p.add_argument("-v", "--verbose", action="store_true", help="show options")
    p.set_defaults(fn=cmd_backends)

    p = sub.add_parser("run", help="run a profile")
    p.add_argument("--profile", default="smoke", help="built-in name or YAML path")
    p.add_argument("--backend", required=True)
    p.add_argument(
        "-o", "--option", action="append", default=[], metavar="K=V", help="backend option"
    )
    p.add_argument(
        "--out",
        help="results root ($ASRBENCH_RESULTS_DIR, default ~/.local/share/oasr/asrbench/results)",
    )
    p.add_argument("--tag", help="label for this run in reports and comparisons")
    p.add_argument("--monitor-interval", type=float, help="resource sampling period, s")
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--no-text", action="store_true", help="do not store transcripts in records")
    _add_profile_overrides(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("validate", help="WER guardrail (untimed)")
    p.add_argument("--backend", required=True)
    p.add_argument("-o", "--option", action="append", default=[], metavar="K=V")
    p.add_argument("--dataset", default="librispeech/test-clean")
    p.add_argument("--manifest", help="a fixed {id, audio, text} JSONL instead of --dataset")
    p.add_argument(
        "--audio-root",
        default=os.environ.get("WAV_DIR") or os.environ.get("AUDIO_DIR"),
        help="root for relative manifest audio paths (default $WAV_DIR, then $AUDIO_DIR)",
    )
    p.add_argument("--num-items", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normalizer", default="english", choices=("english", "basic", "none"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--reference", help="known-good predictions.jsonl or validation dir")
    p.add_argument("--max-wer", type=float, help="absolute WER ceiling, %%")
    p.add_argument("--max-empty-rate", type=float, default=0.0)
    p.add_argument(
        "--max-changed", type=float, help="max %% of utterances differing from --reference"
    )
    p.add_argument(
        "--max-drift-wer", type=float, help="max WER %% of hyps against --reference hyps"
    )
    p.add_argument("--max-wer-delta", type=float, help="max WER increase over --reference, points")
    p.add_argument("--out", help="output root (default <results root>/validation)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("runs", help="list stored runs, newest first")
    p.add_argument("-n", "--limit", type=int, default=20, help="how many (0 = all)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_runs)

    p = sub.add_parser("compare", help="regression check between two runs")
    p.add_argument("baseline", help="run directory, run id, or baseline name")
    p.add_argument("candidate", help="run directory, run id, or baseline name")
    p.add_argument("--thresholds", help="YAML of per-metric rules (default: built-in)")
    p.add_argument("--json", help="write the machine-readable summary here")
    p.add_argument("--allow-hardware-change", action="store_true")
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="list, without failing on, baseline results and metrics the candidate lacks "
        "(a deliberate --only subset, or a validation without --reference)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="list unchanged metrics too")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("report", help="render stored runs")
    p.add_argument("runs", nargs="+", help="run directories, run ids, or baseline names")
    p.add_argument("--out", help="output directory (default: <first run>/report)")
    p.add_argument("--plots", action="store_true", help="also render PNG charts (matplotlib)")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("recompute", help="re-derive summaries from raw records")
    p.add_argument("run", help="run directory, run id, or baseline name")
    p.set_defaults(fn=cmd_recompute)

    p = sub.add_parser("save-baseline", help="copy a run to <results>/baselines/NAME")
    p.add_argument("run", help="run directory, run id, or baseline name")
    p.add_argument("--name", required=True)
    p.add_argument("--out", help="results root")
    p.set_defaults(fn=cmd_save_baseline)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_dir:
        os.environ["ASRBENCH_DATA_DIR"] = args.data_dir
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
