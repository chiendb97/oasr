# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr tune`` -- census, build, diff and status for the tuning database.

    oasr tune census --ckpt-dir CKPT --service-mode offline \\
        --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root $WAV_DIR \\
        --out census.json
    oasr tune build --census census.json --out user        # this machine's user tier
    oasr tune build --census a.json --census b.json --out system   # the shipped file
    oasr tune diff oasr/tune/db/sm120/gemm.json ~/.cache/oasr/tune/v2/sm120/gemm.json
    oasr tune status
    oasr tune export-misses --out misses.json               # after a workload ran

``build`` measures on the GPU it runs on and writes the file for that GPU's
architecture family; a shipped file for another family is built on that family
(``ci/modal_app.py``'s tune entry point runs one per GPU of the matrix).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _cmd_census(args) -> int:
    import os

    from oasr.engine import ASREngine, EngineConfig
    from oasr.tune import census as C

    kw = {
        "ckpt_dir": args.ckpt_dir,
        "service_mode": args.service_mode,
        "max_batch_size": args.max_batch_size,
    }
    if args.architecture:
        kw["architecture"] = args.architecture
    if args.chunk_size:
        kw["chunk_size"] = args.chunk_size
    if args.preferred_batch_size:
        kw["preferred_batch_size"] = args.preferred_batch_size
    cfg = EngineConfig(**kw)
    if args.manifest:
        traffic = C.TrafficModel.from_manifest(args.manifest, args.audio_root)
    else:
        traffic = C.TrafficModel()
    if args.streaming_widths:
        traffic.streaming_widths = {int(w): 1.0 for w in args.streaming_widths}
    engine = ASREngine(C.census_engine_config(cfg))
    ss = C.census(engine, cfg, traffic, coverage=args.coverage)
    if args.capture:
        extra = C.from_capture(args.capture)
        ss.points.extend(extra.points)
        ss.working_set_bytes = max(ss.working_set_bytes, extra.working_set_bytes)
    ss.provenance["ckpt_dir"] = os.path.abspath(args.ckpt_dir)
    if args.architecture:
        ss.provenance["architecture"] = args.architecture
    ss.save(args.out)
    sigs = ss.signatures()
    print(
        f"[census] {len(ss.points)} points over {len(sigs)} signatures, "
        f"{sum(p.must for p in ss.points)} must-tune; weights {ss.working_set_bytes / 2**20:.1f} "
        f"MiB -> {args.out}"
    )
    return 0


def _resolve_out(out: str, sm: int, family: str = "gemm") -> Path:
    from oasr.tune import database

    if out == "user":
        p = database.user_path(sm, family)
        if p is None:
            raise SystemExit("the user tier is disabled (OASR_TUNE_USER_DB=off)")
        return p
    if out == "system":
        return database.system_path(sm, family)
    return Path(out)


def _cmd_prebuild(args) -> int:
    import os

    from oasr.jit.core import _get_target_sm
    from oasr.tune import prebuild as P
    from oasr.tune.build import TUNED_OPS
    from oasr.tune.census import ShapeSet

    ops = set(TUNED_OPS)
    if args.census:
        ops = {p.op for path in args.census for p in ShapeSet.load(path).points}
    jobs = args.jobs or os.cpu_count() or 8
    print(f"[prebuild] sm{_get_target_sm()}: ops {sorted(ops & set(TUNED_OPS))}, -j{jobs}")
    t0 = time.time()
    results = P.prebuild(ops, jobs, progress=lambda s: print(f"[prebuild] {s}", flush=True))
    rss = P.peak_child_rss_mib()
    for r in results:
        print(f"[prebuild] {r.name:<18} {r.status:<7} {r.sources:>3} TUs  {r.seconds:>6.0f}s")
        if r.error:
            print(r.error[-1500:], file=sys.stderr)
    wall = time.time() - t0
    print(f"[prebuild] {wall:.0f}s wall; peak compiler RSS {rss or 0:.0f} MiB")
    if args.json:
        report = {
            "sm": _get_target_sm(),
            "jobs": jobs,
            "wall_s": round(wall, 1),
            "peak_rss_mib": rss,
            "modules": [r.to_json() for r in results],
        }
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 1 if any(r.status == "failed" for r in results) else 0


def _cmd_build(args) -> int:
    from oasr.jit.core import _get_target_sm
    from oasr.tune import database
    from oasr.tune.build import BuildOptions, build_gemm
    from oasr.tune.census import ShapeSet

    sm = _get_target_sm()
    merged: Optional[ShapeSet] = None
    sources: List[Dict[str, Any]] = []
    for path in args.census:
        ss = ShapeSet.load(path)
        sources.append({"path": os.path.basename(path), **ss.provenance})
        if merged is None:
            merged = ss
        else:
            merged.points.extend(ss.points)
            merged.working_set_bytes = max(merged.working_set_bytes, ss.working_set_bytes)
    assert merged is not None
    if len(sources) > 1:  # the file's provenance names every census, not the first
        merged.provenance = {"censuses": sources}
    # One point per (signature, M): merge duplicates across censuses.
    uniq: Dict[Tuple, Any] = {}
    for p in merged.points:
        key = (p.sig, p.M)
        if key in uniq:
            q = uniq[key]
            q.calls += p.calls
            q.weight += p.weight
            q.must = q.must or p.must
            q.eager_fraction = max(q.eager_fraction, p.eager_fraction)
        else:
            uniq[key] = p
    merged.points = list(uniq.values())
    out = _resolve_out(args.out, sm)
    base = database.load_file(out, "gemm", sm) if (out.is_file() and not args.fresh) else None
    opts = BuildOptions(
        min_speedup=args.min_speedup,
        partition=args.partition,
        cover_eps=args.eps,
        max_points_per_sig=args.max_points,
        thin=args.thin,
        fill=not args.no_fill,
        dtype_class=args.dtype_class,
        log_path=args.log,
        top_k=args.top_k,
        compile_budget=args.compile_budget,
        budget_s=args.budget_s,
        checkpoint_path=args.checkpoint,
        progress=(lambda s: print(f"[build] {s}", file=sys.stderr, flush=True)),
    )
    tf = build_gemm(merged, sm=sm, opts=opts, base=base)
    database.save_file(tf, out)
    print(f"[build] {len(tf.entries)} entries, {len(tf.configs)} configs -> {out}")
    return 0


def _load_any(path: str):
    from oasr.tune import database

    with open(path) as f:
        d = json.load(f)
    return database.TuningFile.from_json(d, path=Path(path))


def diff_files(old, new) -> List[str]:
    """Human-readable differences between two tuning files."""
    lines: List[str] = []
    keys = sorted(set(old.entries) | set(new.entries))
    for k in keys:
        a, b = old.entries.get(k), new.entries.get(k)
        if a is None:
            lines.append(f"+ {k}: {b.regions}")
        elif b is None:
            lines.append(f"- {k}: {a.regions}")
        elif a.regions != b.regions:
            lines.append(f"~ {k}:\n    was {a.regions}\n    now {b.regions}")
    ca, cb = set(old.referenced_configs()), set(new.referenced_configs())
    lines.append(
        f"configs referenced: {len(ca)} -> {len(cb)} " f"(+{sorted(cb - ca)}, -{sorted(ca - cb)})"
    )
    return lines


def _cmd_diff(args) -> int:
    for line in diff_files(_load_any(args.old), _load_any(args.new)):
        print(line)
    return 0


def _cmd_status(args) -> int:
    from oasr.jit.core import _get_target_sm
    from oasr.tune import database

    sm = _get_target_sm()
    for family in database.KERNEL_FAMILIES:
        t = database.tiers(family, sm)
        print(f"{family} (sm{sm}): snapshot {t.snapshot_id}")
        for tier, tf in t.ordered():
            stale = "; ".join(tf.stale) or "current"
            print(
                f"  {tier:<6} {tf.path}  {len(tf.entries)} entries, "
                f"{len(tf.referenced_configs())} configs  [{stale}]"
            )
        if not t.ordered():
            print("  (no tuning file: every shape uses the untuned default)")
    print(database.tuning_report())
    return 0


def _cmd_evaluate(args) -> int:
    from oasr.tune import evaluate

    dbs = {}
    for spec in args.db:
        name, _, path = spec.partition("=")
        dbs[name] = path
    report = evaluate.evaluate_log(args.log, dbs, with_model=args.model)
    print(evaluate.format_report(report))
    return 0


def _cmd_export_misses(args) -> int:
    from oasr.tune import telemetry

    data = telemetry.export_misses()
    Path(args.out).write_text(json.dumps(data, indent=1))
    print(f"[tune] {len(data.get('points', []))} missed points -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oasr tune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("census", help="derive the shapes to tune from a deployment")
    c.add_argument("--ckpt-dir", required=True)
    c.add_argument(
        "--architecture",
        help="force a registered architecture (explicit-only ones, e.g. transducer)",
    )
    c.add_argument("--service-mode", choices=("offline", "streaming"), default="offline")
    c.add_argument("--max-batch-size", type=int, default=32)
    c.add_argument("--preferred-batch-size", type=int, nargs="+")
    c.add_argument("--chunk-size", type=int)
    c.add_argument("--manifest", help="JSONL manifest of representative offline traffic")
    c.add_argument("--audio-root")
    c.add_argument(
        "--streaming-widths",
        type=int,
        nargs="+",
        help="observed active-stream counts (default: uniform 1..max)",
    )
    c.add_argument("--capture", help="also merge an OASR_CAPTURE_GEMM JSON (decode paths)")
    c.add_argument("--coverage", type=float, default=0.97)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_census)

    b = sub.add_parser("build", help="measure a census and write a tuning file")
    b.add_argument("--census", action="append", required=True)
    b.add_argument("--out", default="user", help="'user', 'system', or a path")
    b.add_argument("--fresh", action="store_true", help="ignore the existing file at --out")
    b.add_argument("--min-speedup", type=float, default=1.05)
    b.add_argument("--partition", choices=("cover", "points"), default="cover")
    b.add_argument("--eps", type=float, default=0.03)
    b.add_argument("--max-points", type=int, default=24)
    b.add_argument(
        "--thin",
        action="store_true",
        help="hold census points to --max-points too (log-spaced, heaviest kept; a dropped "
        "point's weight moves to the next kept one) -- for dense streaming censuses",
    )
    b.add_argument("--no-fill", action="store_true")
    b.add_argument("--dtype-class", choices=("half", "exact"), default="half")
    b.add_argument(
        "--compile-budget",
        type=int,
        default=None,
        help="cap the distinct CUTLASS configs this build's entries reference",
    )
    b.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="measure only each model's top-k (+ default/fused/cuBLAS); 0 = all",
    )
    b.add_argument(
        "--log",
        help="measurement log (JSONL) to append every arm's timing to "
        "(default: the user tier's measurements/)",
    )
    b.add_argument(
        "--budget-s",
        type=float,
        help="stop measuring after this many seconds and write what was measured "
        "(heaviest signatures go first, so a cut loses the least weight)",
    )
    b.add_argument(
        "--checkpoint",
        help="record finished signatures here (JSONL) and skip them on a rerun, so an "
        "interrupted build resumes instead of re-measuring",
    )
    b.set_defaults(func=_cmd_build)

    pb = sub.add_parser(
        "prebuild",
        help="compile the modules a build loads, without a GPU (target: OASR_CUDA_ARCH_LIST)",
    )
    pb.add_argument(
        "--census",
        action="append",
        help="compile only what these censuses' ops need (default: every tuned op)",
    )
    pb.add_argument("--jobs", type=int, default=None, help="parallel compiles (default: nproc)")
    pb.add_argument("--json", help="write the per-module report here")
    pb.set_defaults(func=_cmd_prebuild)

    v = sub.add_parser("evaluate", help="regret of selectors against a measurement log")
    v.add_argument("--log", required=True, help="measurement log from `oasr tune build --log`")
    v.add_argument(
        "--db",
        action="append",
        default=[],
        help="tuning file(s) to score, as NAME=PATH (the shipped tier is always scored)",
    )
    v.add_argument("--model", action="store_true", help="also score the calibrated cost model")
    v.set_defaults(func=_cmd_evaluate)

    d = sub.add_parser("diff", help="compare two tuning files")
    d.add_argument("old")
    d.add_argument("new")
    d.set_defaults(func=_cmd_diff)

    s = sub.add_parser("status", help="the tiers loaded for this GPU")
    s.set_defaults(func=_cmd_status)

    e = sub.add_parser("export-misses", help="dump this process's selection misses")
    e.add_argument("--out", required=True)
    e.set_defaults(func=_cmd_export_misses)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
