# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr tune`` -- census, build, diff and status for the tuning database.

    oasr tune census --ckpt-dir CKPT --service-mode offline \\
        --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root $WAV_DIR \\
        --out census.json
    oasr tune build --census census.json --out user        # this machine's user tier
    oasr tune build --census a.json --census b.json --out system   # the shipped file
    oasr tune census --ckpt-dir CKPT --quantization int8_w8a8 --out q.json  # a quantized model
    oasr tune build --census q.json --family scaled_mm --out user  # its int8 GEMMs only
    oasr tune census --ckpt-dir CKPT --quantization fp8_w8a8 --out f.json   # fp8 (sm89+)
    oasr tune census --ckpt-dir CKPT --family fmha --manifest M --audio-root R --out a.json
    oasr tune build --census a.json --family fmha --out user  # attention lanes and tiles
    oasr tune diff oasr/tune/db/sm120/gemm.json ~/.cache/oasr/tune/v2/sm120/gemm.json
    oasr tune status
    oasr tune export-misses --out misses.json               # after a workload ran
    oasr tune compare-inductor --ckpt-dir CKPT --out .artifacts/inductor  # a report, not a file

``build`` measures on the GPU it runs on and writes the file for that GPU's
architecture family; a shipped file for another family is built on that family
(``ci/modal_app.py``'s tune entry point runs one per GPU of the matrix).  A census
of a quantized model also holds 8-bit points: ``build`` then writes
``scaled_mm.json`` (int8) or ``scaled_mm_fp8.json`` (fp8) beside ``gemm.json``
(``--family`` restricts it to one).  An attention census (``census --family
fmha``) replays the manifest through the engine and is built into ``fmha.json``:
which attention lane, and which tile or split-KV setting, serves each shape.
``compare-inductor`` is a benchmark, not a tuner: it times ``torch.compile`` of
each repeated encoder block against the OASR path and writes a report of the
fusions Inductor finds that OASR's kernels miss (``oasr.tune.inductor``).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _cmd_census(args) -> int:
    from oasr.engine import ASREngine
    from oasr.tune import census as C

    if args.family == "fmha":
        return _cmd_census_fmha(args)
    if args.family in ("recurrent", "gated_mlp"):
        return _cmd_census_fused(args)
    if not args.ckpt_dir:
        raise SystemExit("census --family gemm needs --ckpt-dir")
    cfg = _deploy_config(args)
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
    ss.provenance["model_name"] = C.model_name(args.ckpt_dir)
    if args.architecture:
        ss.provenance["architecture"] = args.architecture
    if args.quantization:
        ss.provenance["quantization"] = args.quantization
    ss.save(args.out)
    sigs = ss.signatures()
    int8 = sum(1 for sig in sigs if C.kernel_family(sig[0]) == "scaled_mm")
    fp8 = sum(1 for sig in sigs if C.kernel_family(sig[0]) == "scaled_mm_fp8")
    quant = ", ".join(
        part
        for part in (f"{int8} int8 scaled_mm" if int8 else "", f"{fp8} fp8" if fp8 else "")
        if part
    )
    print(
        f"[census] {len(ss.points)} points over {len(sigs)} signatures"
        + (f" ({quant})" if quant else "")
        + f", {sum(p.must for p in ss.points)} must-tune; weights "
        f"{ss.working_set_bytes / 2**20:.1f} MiB -> {args.out}"
    )
    return 0


def _deploy_config(args, decode_options: Optional[Dict[str, Any]] = None):
    from oasr.engine import EngineConfig

    kw: Dict[str, Any] = {
        "ckpt_dir": args.ckpt_dir,
        "service_mode": args.service_mode,
        "max_batch_size": args.max_batch_size,
    }
    for name in ("architecture", "chunk_size", "preferred_batch_size", "quantization"):
        if getattr(args, name, None):
            kw[name] = getattr(args, name)
    if decode_options:
        kw["decode_options"] = dict(decode_options)
    return EngineConfig(**kw)


def _parse_dims(family: str, specs: Sequence[str]) -> List[Dict[str, Any]]:
    """``--dims``: ``lstm:H:I`` (recurrent) or ``silu:N:K[:bias]`` (gated_mlp)."""
    out: List[Dict[str, Any]] = []
    for spec in specs:
        parts = spec.split(":")
        if family == "recurrent":
            op, hidden, width = parts[0], int(parts[1]), int(parts[2])
            out.append({"op": op, "hidden": hidden, "input_size": width})
        else:
            bias = len(parts) > 3 and parts[3] in ("1", "bias", "true")
            out.append(
                {"activation": parts[0], "n": int(parts[1]), "k": int(parts[2]), "bias": bias}
            )
    return out


def _cmd_census_fused(args) -> int:
    """``census --family recurrent|gated_mlp``: the model's static dims x a dynamic ladder.

    Dims come from ``--dims`` and/or the model's own layers (``--ckpt-dir``,
    built on the CPU: nothing is served or put on a GPU).
    """
    from oasr.tune.build_fused import fused_census, model_dims

    dims = _parse_dims(args.family, args.dims or [])
    provenance: Dict[str, Any] = {}
    if args.ckpt_dir:
        from oasr.models.registry import build_model_from_checkpoint
        from oasr.tune.census import model_name

        model, _ = build_model_from_checkpoint(
            args.ckpt_dir, device="cpu", architecture=args.architecture
        )
        found = model_dims(model)[args.family]
        dims.extend(found)
        provenance["model_name"] = model_name(args.ckpt_dir)
        del model
    if not dims:
        raise SystemExit(f"no {args.family} layers found: pass --ckpt-dir with one, or --dims")
    uniq = list({json.dumps(d, sort_keys=True): d for d in dims}.values())
    ss = fused_census(
        args.family,
        uniq,
        dtype=args.dtype,
        ladder=args.ladder,
        seq_lens=args.seq_lens,
        provenance=provenance,
    )
    ss.save(args.out)
    print(
        f"[census] {args.family}: {len(uniq)} layer shape(s), {len(ss.points)} points -> {args.out}"
    )
    return 0


def _cmd_build_fused(args, family: str, paths: Sequence[str], sm: int) -> int:
    from oasr.tune import database
    from oasr.tune.build_fused import FusedShapeSet, build_gated_mlp, build_recurrent

    merged = FusedShapeSet(family, [], {})
    sources: List[Dict[str, Any]] = []
    for path in paths:
        ss = FusedShapeSet.load(path)
        sources.append({"path": os.path.basename(path), **ss.provenance})
        merged.points.extend(ss.points)
    merged.provenance = sources[0] if len(sources) == 1 else {"censuses": sources}
    out = _resolve_out(args.out, sm, family)
    base = database.load_file(out, family, sm) if (out.is_file() and not args.fresh) else None
    since = None if base is None else base.to_json()
    builder = build_recurrent if family == "recurrent" else build_gated_mlp
    tf = builder(merged, sm=sm, opts=_build_options(args), base=base)
    _write_built(args, tf, out, since)
    return 0


def _cmd_census_fmha(args) -> int:
    """``census --family fmha``: replay the manifest, count every ``fmha`` call shape."""
    from oasr.engine import ASREngine
    from oasr.tune import census as C
    from oasr.tune.attention_census import attention_census

    cfg = _deploy_config(args)
    traffic = C.TrafficModel()
    if args.manifest:
        traffic = C.TrafficModel.from_manifest(args.manifest, args.audio_root)
    if args.streaming_widths:
        traffic.streaming_widths = {int(w): 1.0 for w in args.streaming_widths}
    engine = ASREngine(C.census_engine_config(cfg))
    options = getattr(getattr(engine._output_processor, "strategy", None), "options", None)
    if getattr(options, "step_graphs", False):
        # An AR decode family captures its decoder step in a CUDA graph of its
        # own (an ``ArOptions`` knob, not ``use_cuda_graphs``), and a replay never
        # re-enters the recorder: every step after the capture would go uncounted.
        # Without step graphs ``kv_storage="auto"`` turns dense, so keep the
        # paged layout the served steps read.
        engine.shutdown()
        cfg = _deploy_config(args, {"step_graphs": False, "kv_storage": "paged"})
        engine = ASREngine(C.census_engine_config(cfg))
    waves = None
    if args.manifest:
        from oasr.testing.accuracy import load_audio, load_manifest

        root = Path(args.audio_root) if args.audio_root else None
        entries = load_manifest(Path(args.manifest), root, limit=args.limit)
        waves, _ = load_audio(entries, engine.sample_rate)
    ss = attention_census(engine, cfg, traffic, waves=waves)
    ss.provenance["model_name"] = C.model_name(args.ckpt_dir)
    if args.architecture:
        ss.provenance["architecture"] = args.architecture
    ss.save(args.out)
    print(
        f"[census] fmha: {len(ss.points)} call shapes over {len(ss.signatures())} signatures"
        + ("" if waves else " (no manifest: encoder only)")
        + f" -> {args.out}"
    )
    return 0


def _resolve_out(out: str, sm: int, family: str = "gemm") -> Path:
    """Where *family*'s file goes for ``--out``.

    ``user`` / ``system`` name the tier's ``<family>.json``.  An explicit path is
    the GEMM file; another family's goes beside it -- ``scaled_mm.json`` next to
    a ``gemm.json``, else ``<stem>.<family>.json``.
    """
    from oasr.tune import database

    if out == "user":
        p = database.user_path(sm, family)
        if p is None:
            raise SystemExit("the user tier is disabled (OASR_TUNE_USER_DB=off)")
        return p
    if out == "system":
        return database.system_path(sm, family)
    path = Path(out)
    if family == "gemm" or path.name == f"{family}.json":
        return path
    if path.name == "gemm.json":
        return path.with_name(f"{family}.json")
    return path.with_name(f"{path.stem}.{family}{path.suffix or '.json'}")


def _cmd_prebuild(args) -> int:
    import os

    from oasr.jit.core import _get_target_sm
    from oasr.tune import prebuild as P
    from oasr.tune.build import TUNED_OPS
    from oasr.tune.build_scaled_mm import SCALED_MM_TUNED_OPS
    from oasr.tune.census import ShapeSet

    tuned = set(TUNED_OPS) | set(SCALED_MM_TUNED_OPS)
    ops = set(TUNED_OPS)
    if args.census:
        ops = {p.op for path in args.census for p in ShapeSet.load(path).points}
    jobs = args.jobs or os.cpu_count() or 8
    print(f"[prebuild] sm{_get_target_sm()}: ops {sorted(ops & tuned)}, -j{jobs}")
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


def _build_options(args):
    from oasr.tune.build import BuildOptions

    return BuildOptions(
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


def _write_built(args, tf, out: Path, since) -> None:
    from oasr.tune import database

    if args.note:
        tf.provenance["notes"] = list(args.note)
    if args.fresh:
        database.save_file(tf, out)
    else:
        # Merged under the file's lock: a build takes minutes, and whatever
        # another process wrote to the file meanwhile is kept.
        tf = database.save_merged(tf, out, since=since)
    print(f"[build] {tf.family}: {len(tf.entries)} entries, {len(tf.configs)} configs -> {out}")


def _cmd_build_fmha(args, paths: Sequence[str], sm: int) -> int:
    from oasr.tune import database
    from oasr.tune.attention_census import AttnShapeSet
    from oasr.tune.build_fmha import build_fmha

    if not paths:
        print("[build] fmha needs an attention census (oasr tune census --family fmha)")
        return 1
    merged = AttnShapeSet(points=[], provenance={})
    sources: List[Dict[str, Any]] = []
    for path in paths:
        ss = AttnShapeSet.load(path)
        sources.append({"path": os.path.basename(path), **ss.provenance})
        merged.points.extend(ss.points)
    merged.provenance = sources[0] if len(sources) == 1 else {"censuses": sources}
    out = _resolve_out(args.out, sm, "fmha")
    base = database.load_file(out, "fmha", sm) if (out.is_file() and not args.fresh) else None
    since = None if base is None else base.to_json()
    opts = _build_options(args)
    if args.max_points == 24:  # the GEMM default; attention signatures need fewer
        opts.max_points_per_sig = 12
    tf = build_fmha(merged, sm=sm, opts=opts, base=base, allow_external=args.allow_external)
    _write_built(args, tf, out, since)
    return 0


def _cmd_build(args) -> int:
    from oasr.jit.core import _get_target_sm
    from oasr.tune import database
    from oasr.tune.attention_census import is_attention_census
    from oasr.tune.build import build_gemm
    from oasr.tune.build_fused import census_family
    from oasr.tune.build_scaled_mm import build_scaled_mm
    from oasr.tune.census import ShapeSet, kernel_family

    sm = _get_target_sm()
    attn_paths = [p for p in args.census if is_attention_census(p)]
    fused_paths: Dict[str, List[str]] = {}
    for p in args.census:
        fam = census_family(p)
        if fam in ("recurrent", "gated_mlp"):
            fused_paths.setdefault(fam, []).append(p)
    gemm_paths = [p for p in args.census if p not in attn_paths and census_family(p) == "gemm"]
    if args.family in ("recurrent", "gated_mlp"):
        paths = fused_paths.get(args.family, [])
        if not paths:
            print(
                f"[build] no {args.family} census given (oasr tune census --family {args.family})"
            )
            return 1
        return _cmd_build_fused(args, args.family, paths, sm)
    if args.family == "auto":
        for fam, paths in fused_paths.items():
            rc = _cmd_build_fused(args, fam, paths, sm)
            if rc:
                return rc
    if args.family == "fmha" or (args.family == "auto" and not gemm_paths):
        if args.family == "auto" and not attn_paths:
            return 0
        return _cmd_build_fmha(args, attn_paths, sm)
    if args.family == "auto" and attn_paths:
        rc = _cmd_build_fmha(args, attn_paths, sm)
        if rc:
            return rc
    merged: Optional[ShapeSet] = None
    sources: List[Dict[str, Any]] = []
    for path in gemm_paths:
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
    families = {kernel_family(p.op) for p in merged.points}
    # Typed: mypy 2.x infers this literal's values as `object` without the annotation.
    wanted: Dict[str, List[str]] = {
        "auto": [f for f in ("gemm", "scaled_mm", "scaled_mm_fp8") if f in families] or ["gemm"],
        "gemm": ["gemm"],
        "scaled_mm": ["scaled_mm"],
        "scaled_mm_fp8": ["scaled_mm_fp8"],
    }
    want = wanted[args.family]
    for family, what in (("scaled_mm", "int8"), ("scaled_mm_fp8", "fp8")):
        if family in want and family not in families:
            print(f"[build] the census has no {what} {family} points (census a quantized model)")
            return 1
    opts = _build_options(args)
    builders: Dict[str, Callable[..., Any]] = {
        "gemm": build_gemm,
        "scaled_mm": build_scaled_mm,
        "scaled_mm_fp8": functools.partial(build_scaled_mm, fmt="fp8"),
    }
    for family in want:
        out = _resolve_out(args.out, sm, family)
        base = database.load_file(out, family, sm) if (out.is_file() and not args.fresh) else None
        since = None if base is None else base.to_json()
        tf = builders[family](merged, sm=sm, opts=opts, base=base)
        _write_built(args, tf, out, since)
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


def _ints(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(",", " ").split()]


def _floats(text: str) -> List[float]:
    return [float(x) for x in str(text).replace(",", " ").split()]


def _cmd_compare_inductor(args) -> int:
    """``compare-inductor``: ``torch.compile`` of each repeated encoder block vs OASR.

    A benchmark only (``oasr.tune.inductor``): it writes a report, never a
    tuning file.  Each shape is measured in a child process (``--in-process``
    measures here): a compiled arm can leave a sticky CUDA error -- a Triton
    candidate faulting inside Inductor's autotuning -- and no later number in
    that process can be trusted, so the block it was on is recorded as failed and
    a fresh child carries on with the rest.
    """
    import datetime
    import re
    import shutil
    import tempfile

    import torch

    from oasr.models.loaders import load_pretrained
    from oasr.tune import bench, inductor as ind
    from oasr.tune.census import model_name

    ckpt = args.ckpt_dir or os.environ.get("CKPT_DIR")
    if not ckpt:
        raise SystemExit("compare-inductor: --ckpt-dir is required (or export CKPT_DIR)")
    if not torch.cuda.is_available():
        raise SystemExit("compare-inductor measures on a GPU; none is visible")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    backends = [b.strip().upper() for b in args.backends.split(",") if b.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    batches, seconds = _ints(args.batches), _floats(args.seconds)
    loaded = load_pretrained(ckpt, device="cpu", architecture=args.architecture)
    encoder = loaded.model.encoder
    policy = (
        bench.BenchPolicy(solo=False, stage_rounds=(2, 3, 5))
        if args.quick
        else bench.BenchPolicy(solo=False)
    )
    subproc = ind.autotune_subproc(args.autotune_subproc)
    paths = [p for p in args.paths.split(",") if p] if args.paths else None
    measure = functools.partial(
        ind.compare,
        encoder,
        loaded.feature_spec,
        select=args.blocks or None,
        paths=paths,
        arms=arms,
        dtype=dtype,
        backends=backends,
        policy=policy,
        max_copy_bytes=int(args.max_copy_mib) << 20,
        verbose=args.verbose,
        subproc=subproc,
    )

    if args.child_out:  # one shape, for an isolating parent: stream progress, no report
        with open(args.child_out, "a") as sink:

            def emit(rec: Dict[str, Any]) -> None:
                sink.write(json.dumps(rec) + "\n")
                sink.flush()

            measure(
                batches=batches,
                seconds=seconds,
                on_start=lambda path: emit({"start": path}),
                on_result=lambda r: emit({"result": r.to_json()}),
                on_skip=lambda path, why: emit({"skip": path, "reason": why}),
            )
        return 0

    t0 = time.time()
    children_log: Optional[Path] = None
    if args.in_process:
        results = measure(batches=batches, seconds=seconds)
    else:
        scratch = Path(tempfile.mkdtemp(prefix="oasr-inductor-"))
        children_log = scratch / "children.log"
        results = _compare_isolated(
            args,
            ckpt,
            ind,
            encoder,
            loaded.feature_spec,
            batches,
            seconds,
            arms,
            subproc,
            scratch,
            children_log,
        )
    from oasr.jit.core import _get_target_sm

    try:
        import triton

        triton_version = triton.__version__
    except ImportError:  # pragma: no cover -- Inductor itself needs Triton on CUDA
        triton_version = "missing"
    try:
        import cutlass

        cutedsl = str(getattr(cutlass, "__version__", "unknown"))
    except ImportError:
        cutedsl = "missing"
    name = model_name(ckpt)
    sm = _get_target_sm()
    meta = {
        "model": name,
        "architecture": loaded.architecture,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(),
        "sm": sm,
        "torch": torch.__version__,
        "triton": triton_version,
        "cutedsl": cutedsl,
        "backends": backends,
        "backend_status": ind.available_backends(backends),
        "autotune_in_subproc": subproc,
        "isolated": not args.in_process,
        "arms": arms,
        "batches": batches,
        "seconds": seconds,
        "quick": bool(args.quick),
        "bench_protocol": bench.PROTOCOL_VERSION,
        "wall_s": round(time.time() - t0, 1),
        "date": datetime.date.today().isoformat(),
    }
    stem = args.stem or re.sub(r"[^A-Za-z0-9._-]+", "_", f"{name}_{args.dtype}_sm{sm}")
    jpath, mpath = ind.save_report(results, meta, Path(args.out), stem, args.gap)
    if children_log is not None:
        if children_log.exists():
            shutil.copyfile(children_log, Path(args.out) / f"{stem}.children.log")
        shutil.rmtree(children_log.parent, ignore_errors=True)
    ahead = [r for r in results if r.verdict(args.gap) == "inductor-ahead"]
    died = [r for r in results if r.copies == 0]
    print(
        f"[inductor] {len(results)} block-shapes; Inductor ahead at {len(ahead)} "
        f"(>= {args.gap:.0%})"
        + (f"; {len(died)} lost to a dead process (see {stem}.children.log)" if died else "")
        + f" -> {mpath} ({jpath.name})"
    )
    return 0


def _last_error(log: Path) -> str:
    """The last line of *log* that names an error (a child's stderr), or ``""``."""
    try:
        lines = log.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    hits = [ln.strip() for ln in lines if "Error" in ln or "error:" in ln]
    return hits[-1][:300] if hits else ""


def _compare_isolated(
    args, ckpt, ind, encoder, feature_spec, batches, seconds, arms, subproc, scratch, log_path
) -> List[Any]:
    """Measure each shape in a child process; respawn past a block that killed one."""
    import subprocess

    blocks = ind.discover_blocks(encoder, args.blocks or None)
    if not blocks:
        raise SystemExit("compare-inductor: no repeated block found")
    kinds = {p: type(m).__name__ for p, m in blocks}
    results: List[Any] = []
    n = 0
    for b, s, frames in ind.shape_ladder(feature_spec, batches, seconds):
        remaining = [p for p, _ in blocks]
        while remaining:
            n += 1
            out = scratch / f"child{n}.jsonl"
            cmd = [
                sys.executable, "-m", "oasr.tune.cli", "compare-inductor",
                "--child-out", str(out), "--paths", ",".join(remaining),
                "--batches", str(b), "--seconds", repr(s), "--ckpt-dir", ckpt,
                "--dtype", args.dtype, "--arms", args.arms, "--backends", args.backends,
                "--max-copy-mib", str(args.max_copy_mib),
                "--autotune-subproc", "on" if subproc else "off",
            ]  # fmt: skip
            if args.architecture:
                cmd += ["--architecture", args.architecture]
            cmd += [
                flag for flag, on in (("--quick", args.quick), ("--verbose", args.verbose)) if on
            ]
            with open(log_path, "a") as err:
                err.write(f"=== child {n}: B={b} {s:g}s, {len(remaining)} block(s)\n")
                err.flush()
                rc = subprocess.run(cmd, stderr=err).returncode
            started, finished = None, set()
            for line in out.read_text().splitlines() if out.exists() else []:
                rec = json.loads(line)
                if "start" in rec:
                    started = rec["start"]
                elif "result" in rec:
                    r = ind.BlockResult.from_json(rec["result"])
                    results.append(r)
                    finished.add(r.block)
                elif "skip" in rec:
                    finished.add(rec["skip"])
            remaining = [p for p in remaining if p not in finished]
            if not remaining:
                break
            if rc != 0 and started in remaining:
                why = _last_error(log_path) or "no error message"
                reason = f"the measuring process died (exit {rc}): {why}"
                print(f"[inductor] {started} B={b} {s:g}s: {reason}")
                results.append(
                    ind.failed_block(started, kinds[started], b, s, frames, arms, reason)
                )
                remaining.remove(started)
                continue
            # A child that exits without reporting its blocks made no progress:
            # record them rather than loop.
            reason = f"the measuring process exited ({rc}) without measuring it"
            for p in remaining:
                results.append(ind.failed_block(p, kinds[p], b, s, frames, arms, reason))
            remaining = []
    return results


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
    c.add_argument(
        "--ckpt-dir",
        help="the checkpoint (required for gemm / fmha; recurrent / gated_mlp read its layers)",
    )
    c.add_argument(
        "--architecture",
        help="force a registered architecture (explicit-only ones, e.g. transducer)",
    )
    c.add_argument("--service-mode", choices=("offline", "streaming"), default="offline")
    c.add_argument("--max-batch-size", type=int, default=32)
    c.add_argument("--preferred-batch-size", type=int, nargs="+")
    c.add_argument("--chunk-size", type=int)
    c.add_argument(
        "--quantization",
        help="quantize online first (a scheme preset, e.g. int8_w8a8): the census then also "
        "records the int8 scaled_mm GEMMs (a quantized checkpoint needs no flag)",
    )
    c.add_argument("--manifest", help="JSONL manifest of representative offline traffic")
    c.add_argument("--audio-root")
    c.add_argument(
        "--streaming-widths",
        type=int,
        nargs="+",
        help="observed active-stream counts (default: uniform 1..max)",
    )
    c.add_argument("--capture", help="also merge an OASR_CAPTURE_GEMM JSON (decode paths)")
    c.add_argument(
        "--family",
        choices=("gemm", "fmha", "recurrent", "gated_mlp"),
        default="gemm",
        help="gemm: the GEMM-family census (probes the engine's keys); fmha: the attention "
        "census (replays --manifest through the engine, decode steps included); recurrent / "
        "gated_mlp: the model's layer dims (or --dims) crossed with a batch / row ladder",
    )
    c.add_argument("--limit", type=int, help="fmha: replay at most this many utterances")
    c.add_argument(
        "--dims",
        nargs="+",
        help="recurrent: KIND:H:I (e.g. lstm:640:640); gated_mlp: ACT:N:K[:bias] "
        "(e.g. silu:11008:4096)",
    )
    c.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        help="recurrent: batches (default 1..512, powers of two); gated_mlp: row counts",
    )
    c.add_argument(
        "--seq-lens", type=int, nargs="+", help="recurrent: timesteps per class (default 1 32)"
    )
    c.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    c.add_argument("--coverage", type=float, default=0.97)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_census)

    b = sub.add_parser("build", help="measure a census and write a tuning file")
    b.add_argument("--census", action="append", required=True)
    b.add_argument(
        "--out",
        default="user",
        help="'user', 'system', or a path (the gemm file; scaled_mm[_fp8].json go beside it)",
    )
    b.add_argument(
        "--family",
        choices=("auto", "gemm", "scaled_mm", "scaled_mm_fp8", "fmha", "recurrent", "gated_mlp"),
        default="auto",
        help="which kernel family's file to build (auto: every family the censuses have points "
        "for -- scaled_mm when one has int8 points, scaled_mm_fp8 when it has fp8 ones, fmha "
        "for an attention census)",
    )
    b.add_argument(
        "--allow-external",
        action="store_true",
        help="fmha: let torch SDPA win a region (it is always measured, as a reference; a "
        "region it wins is a policy hit at every dispatch)",
    )
    b.add_argument(
        "--note",
        action="append",
        help="free-text provenance note to record in the written file(s) (repeatable)",
    )
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

    ci = sub.add_parser(
        "compare-inductor",
        help="time torch.compile of each repeated encoder block against OASR (report only)",
    )
    ci.add_argument("--ckpt-dir", help="the checkpoint (default: $CKPT_DIR)")
    ci.add_argument(
        "--architecture",
        help="force a registered architecture (explicit-only ones, e.g. transducer)",
    )
    ci.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    ci.add_argument(
        "--batches",
        default=",".join(str(b) for b in _inductor_defaults()[0]),
        help="batch sizes (comma-separated)",
    )
    ci.add_argument(
        "--seconds",
        default=",".join(f"{s:g}" for s in _inductor_defaults()[1]),
        help="utterance lengths in seconds (a fixed-window frontend always uses its window)",
    )
    ci.add_argument(
        "--blocks",
        nargs="*",
        help="only blocks whose class name, or part of whose module path, matches",
    )
    ci.add_argument(
        "--arms",
        default="oasr,torch,inductor,inductor-max",
        help="arms to time: oasr, torch, inductor, inductor-max",
    )
    ci.add_argument(
        "--backends",
        default="ATEN,TRITON,CUTLASS,CUTEDSL",
        help="Inductor GEMM backends for the inductor-max arm",
    )
    ci.add_argument("--gap", type=float, default=0.05, help="how far ahead counts as ahead")
    ci.add_argument(
        "--max-copy-mib",
        type=int,
        default=256,
        help="bound on the rotated block copies' bytes (L2 honesty vs a shared GPU)",
    )
    ci.add_argument("--quick", action="store_true", help="fewer timing rounds")
    ci.add_argument("--verbose", action="store_true", help="keep Inductor's autotune logging")
    ci.add_argument(
        "--autotune-subproc",
        choices=("auto", "on", "off"),
        default="auto",
        help="benchmark max-autotune candidates in Inductor's subprocess "
        "(auto: when the GPU has under 8 GiB free)",
    )
    ci.add_argument(
        "--in-process",
        action="store_true",
        help="measure in this process (default: one child process per shape)",
    )
    ci.add_argument("--out", default=".artifacts/inductor", help="report directory")
    ci.add_argument("--stem", help="report file name (default: model_dtype_smNN)")
    ci.add_argument("--child-out", help=argparse.SUPPRESS)
    ci.add_argument("--paths", help=argparse.SUPPRESS)
    ci.set_defaults(func=_cmd_compare_inductor)
    return p


def _inductor_defaults() -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    from oasr.tune.inductor import DEFAULT_BATCHES, DEFAULT_SECONDS

    return DEFAULT_BATCHES, DEFAULT_SECONDS


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
