# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr tune build`` for the 8-bit scaled GEMM: a census in, ``scaled_mm.json`` out.

Two families, one build: ``scaled_mm`` (int8) and ``scaled_mm_fp8`` (fp8 e4m3),
each with its own file (``scaled_mm.json`` / ``scaled_mm_fp8.json``) measured on
its own lane; :func:`build_scaled_mm` takes the format.  Each has a fixed, small
production set (``oasr.jit.scaled_mm``: int8 six data-parallel tiles and two
Stream-K tiles, fp8 eight data-parallel tiles; one module per format) and a
shape heuristic that is within a few percent of the best tile on the shapes it
was written against.  So, unlike the fp GEMM build, there is no
candidate space to prune and no cost model to ship: every eligible variant is
measured at every point, and the question each point answers is *does anything
beat the heuristic by more than the gate and the noise?*

Per signature ``(op, N, K)`` of the census -- ``op`` is the format's fp16/bf16-output
GEMM (``scaled_mm`` / ``scaled_mm_fp8``) or its 8-bit-output producer
(``scaled_mm_quant`` / ``scaled_mm_fp8_quant``, data-parallel only):

1. measure every eligible variant at the census Ms plus ladder fill points
   (``BuildOptions.fill`` / ``thin`` / ``max_points_per_sig``, as the GEMM build)
   with the shared protocol (:func:`oasr.tune.bench.measure`: graph-captured
   loops, rotated weight copies when the served model would not keep them in L2,
   interleaved rounds, successive halving with the heuristic's variant forced to
   the end, the single-replay gate) on realistic operands -- random 8-bit values,
   a scale per token and per channel, a bias, the swish epilogue, and for the
   quantizing op an output multiplier per channel;
2. gate numerics: the heuristic's variant against the torch reference (within an
   output ulp, plus fp8's fp32 accumulation error), every other variant
   **bitwise** against it -- each format's invariant (int8: an exact int32
   accumulator, int32 Stream-K partials; fp8: one summation order for every
   tile), so a variant that differs is broken, not slower;
3. add a ``"heuristic"`` pseudo-arm per point whose measurement is that of the
   variant the heuristic picks there, and partition the M axis against it
   (:func:`oasr.tune.partition.regions_for_signature`): a region whose best is the
   heuristic names the sentinel instead of a tile, a region another variant wins
   by ``min_speedup`` names that variant.

Every signature measured gets an entry -- an all-heuristic one is
``[[null, "heuristic"]]`` -- because "measured, and the heuristic was best" is
coverage (``tuning.strict``, ``ASREngine.tuning_report``), which an absent entry
cannot say.
"""

from __future__ import annotations

import dataclasses
import functools
import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from oasr.tune import bench, database, gemm_tune, partition
from oasr.tune.build import (
    BuildOptions,
    PointResult,
    _alpha_for,
    _append_checkpoint,
    _evidence,
    _fill_points,
    _load_checkpoint,
    _points_digest,
    _sig_id,
    decide_point,
    thin_census,
)
from oasr.tune.census import ShapePoint, ShapeSet

logger = logging.getLogger("oasr.tune")

__all__ = [
    "SCALED_MM_TUNED_OPS",
    "build_scaled_mm",
    "candidates",
    "measure_case",
    "measure_signature",
]

#: The census ops this build measures, both formats (``oasr.jit.scaled_mm.SCALED_MM_OPS``).
SCALED_MM_TUNED_OPS = ("scaled_mm", "scaled_mm_quant", "scaled_mm_fp8", "scaled_mm_fp8_quant")

#: The census dtype of each format's operands.
_DTYPE_STR = {"int8": "int8", "fp8": "float8_e4m3fn"}

#: The epilogue the arms run: its cost does not depend on which activation.
_ACTIVATION = "swish"

#: The fp output (and the quantizing op's rounding) dtype the arms run in.
_OUT_DTYPE = torch.float16

#: Rows checked against the torch reference: rows are computed independently,
#: so a prefix checks the variant, and an fp64 reference at M=16k is not free.
_REFERENCE_ROWS = 512


def candidates(op: str, sm: int) -> Dict[str, Any]:
    """``{config id: config}`` of the variants *op* may run on *sm*.

    Every production variant of *op*'s format for the fp-output op; the
    data-parallel ones only for a quantizing op (Stream-K has no quantizing
    launcher).
    """
    import oasr.jit.scaled_mm as jsm

    cfgs = jsm.get_production_configs(sm, jsm.op_format(op))
    if jsm.is_quant_op(op):
        cfgs = {k: c for k, c in cfgs.items() if not c.stream_k}
    return {jsm.config_id(c): c for c in cfgs.values()}


def _random_8bit(shape, fmt: str, g: torch.Generator) -> torch.Tensor:
    if fmt == "int8":
        return torch.randint(-127, 128, shape, generator=g, dtype=torch.int8, device="cuda")
    x = torch.randn(shape, generator=g, device="cuda") * 64.0
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _operands(op: str, M: int, N: int, K: int, copies: int):
    import oasr.jit.scaled_mm as jsm

    fmt = jsm.op_format(op)
    g = torch.Generator(device="cuda").manual_seed(0)
    a = _random_8bit((M, K), fmt, g)
    bs = [_random_8bit((N, K), fmt, g) for _ in range(copies)]
    a_scale = torch.rand(M, generator=g, device="cuda") * 0.01 + 1e-3
    b_scale = torch.rand(N, generator=g, device="cuda") * 0.01 + 1e-3
    bias = torch.randn(N, generator=g, device="cuda").to(_OUT_DTYPE)
    if jsm.is_quant_op(op):
        out_mul = torch.rand(N, generator=g, device="cuda") * 32.0 + 16.0
        out = torch.empty(M, N, dtype=a.dtype, device="cuda")
    else:
        out_mul = None
        out = torch.empty(M, N, dtype=_OUT_DTYPE, device="cuda")
    return a, bs, a_scale, b_scale, bias, out_mul, out


def _check_reference(op: str, out, a, b, a_scale, b_scale, bias, out_mul, name: str) -> float:
    """Max error of the heuristic's variant against the torch reference; raise if off."""
    import oasr.jit.scaled_mm as jsm
    from oasr.functionals.scaled_mm import reference_scaled_mm, reference_scaled_mm_quant

    fp8 = jsm.op_format(op) == "fp8"
    r = min(int(a.shape[0]), _REFERENCE_ROWS)
    # fp8 sums in fp32 (not exact): a little more than an output ulp of slack.
    rel = 2.0**-8 if fp8 else 2.0**-9
    if jsm.is_quant_op(op):
        ref = reference_scaled_mm_quant(
            a[:r], b, a_scale[:r], b_scale, bias, _ACTIVATION, out_mul, _OUT_DTYPE
        )
        if fp8:
            got_f, ref_f = out[:r].float(), ref.float()
            diff = (got_f - ref_f).abs()
            err = float(diff.max()) if r else 0.0
            ok = bool((diff <= ref_f.abs() * 2.0**-3 + 2.0**-9).all())  # one e4m3 step
        else:
            err = float((out[:r].int() - ref.int()).abs().max()) if r else 0.0
            ok = err <= 1.0  # one int8 step at a rounding tie
    else:
        ref = reference_scaled_mm(a[:r], b, a_scale[:r], b_scale, bias, _ACTIVATION, _OUT_DTYPE)
        got = out[:r].float()
        diff = (got - ref.float()).abs()
        err = float(diff.max()) if r else 0.0
        ok = bool(torch.isfinite(got).all()) and bool(
            (diff <= ref.float().abs() * rel + 1e-3).all()
        )
    if not ok:
        raise RuntimeError(
            f"scaled_mm variant {name} disagrees with the torch reference ({op}, "
            f"M={int(a.shape[0])} N={int(b.shape[0])} K={int(a.shape[1])}: max|err| {err:.3g}); "
            "a broken kernel, not a tuning question"
        )
    return err


def measure_case(
    op: str,
    M: int,
    N: int,
    K: int,
    *,
    sm: int,
    policy: bench.BenchPolicy = bench.BenchPolicy(),
    working_set_bytes: Optional[int] = None,
    issue_cost: bool = True,
) -> gemm_tune.CaseResult:
    """Measure every eligible variant of *op* at one shape, plus the heuristic pseudo-arm.

    The result's ``"heuristic"`` measurement is a copy of the measurement of the
    variant :func:`oasr.jit.scaled_mm.heuristic_config` picks here (forced through
    every halving stage, so it is measured to the end).
    """
    import oasr.jit.scaled_mm as jsm
    from oasr.functionals.scaled_mm import _ROUND_DTYPE_CODE, _get_scaled_mm_module, activation_id

    fmt = jsm.op_format(op)
    case = gemm_tune.GemmCase(op, M, N, K, _DTYPE_STR[fmt], 1)
    cfgs = candidates(op, sm)
    heur_id = jsm.config_id(jsm.heuristic_config(M, N, K, sm, fmt=fmt))
    copies = bench.l2_copies(
        case.weight_bytes, working_set_bytes=working_set_bytes, max_copies=policy.max_rotation_calls
    )
    a, bs, a_scale, b_scale, bias, out_mul, out = _operands(op, M, N, K, copies)
    act = activation_id(_ACTIVATION)
    mod = _get_scaled_mm_module(fmt)

    if jsm.is_quant_op(op):
        code = _ROUND_DTYPE_CODE[_OUT_DTYPE]

        def call_args(i: int) -> tuple:
            return (out, a, bs[i % copies], a_scale, b_scale, bias, out_mul, act, code)

        func_name = jsm.scaled_mm_quant_func_name
    else:

        def call_args(i: int) -> tuple:
            return (out, a, bs[i % copies], a_scale, b_scale, bias, act)

        func_name = jsm.scaled_mm_func_name

    errors: Dict[str, float] = {}
    rejected: Dict[str, str] = {}
    fns: Dict[str, Any] = {}
    anchor = None
    for name in [heur_id] + sorted(n for n in cfgs if n != heur_id):
        try:
            fn = getattr(mod, func_name(cfgs[name]))
            fn(*call_args(0))
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 -- recorded per arm
            bench._sync_or_sticky(f"first call of {name}")
            rejected[name] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        if anchor is None:
            errors[name] = _check_reference(
                op, out, a, bs[0], a_scale, b_scale, bias, out_mul, name
            )
            anchor = out.clone()
        elif torch.equal(out, anchor):
            errors[name] = 0.0
        else:
            errors[name] = float((out.float() - anchor.float()).abs().max())
            rejected[name] = (
                f"numerics: not bit-identical to {heur_id} (max|diff| {errors[name]:.3g})"
            )
            continue
        fns[name] = fn

    arms = []
    for name, fn in fns.items():

        def make_call(i: int, fn=fn):
            args = call_args(i)
            return lambda: fn(*args)

        arms.append(bench.Arm(name, make_call, forced=name == heur_id, payload=cfgs[name]))
    results, cond = bench.measure(arms, policy, copies=copies)
    if issue_cost:
        for name, m in results.items():
            if m.status == "ok" and m.eliminated_at is None:
                m.issue_us = bench.measure_issue_us(functools.partial(fns[name], *call_args(0)))
    twin = results.get(heur_id)
    if twin is not None:
        results[jsm.HEURISTIC_ID] = dataclasses.replace(
            twin, name=jsm.HEURISTIC_ID, payload=jsm.HEURISTIC_ID, samples=list(twin.samples)
        )
    a = bs = a_scale = b_scale = bias = out_mul = out = anchor = None  # noqa: F841
    torch.cuda.empty_cache()
    return gemm_tune.CaseResult(case, results, cond, errors, rejected, copies, {})


def measure_signature(
    sig: Tuple[str, int, int, str, int],
    points: Sequence[ShapePoint],
    working_set_bytes: int,
    opts: BuildOptions,
    sm: int,
) -> List[PointResult]:
    """Measure one ``scaled_mm`` signature at its census points (+ fill); decide each."""
    import oasr.jit.scaled_mm as jsm

    op, N, K, _dtype, _batch = sig
    if opts.thin:
        points = thin_census(points, opts.max_points_per_sig)
    census_ms = {p.M: p for p in points}
    ms = _fill_points(list(census_ms), opts.max_points_per_sig) if opts.fill else sorted(census_ms)
    census_w = [p.weight for p in points if p.weight > 0]
    fill_w = 0.25 * min(census_w) if census_w else 1.0
    out: List[PointResult] = []
    for M in ms:
        res = measure_case(
            op, M, N, K, sm=sm, policy=opts.policy, working_set_bytes=working_set_bytes
        )
        alpha = _alpha_for(M, points)
        choice, sp, note = decide_point(res, jsm.HEURISTIC_ID, alpha, opts)
        heur = jsm.heuristic_config(M, N, K, sm, fmt=jsm.op_format(op))
        note += f" [heuristic: {jsm.config_id(heur)}]"
        w = census_ms[M].weight if M in census_ms and census_ms[M].weight > 0 else fill_w
        out.append(
            PointResult(M, M in census_ms and census_ms[M].must, alpha, res, choice, sp, note, w)
        )
        if opts.progress:
            opts.progress(f"{op} N={N} K={K} M={M}: {choice} ({note})")
    return out


def _log_rows(pr: PointResult, sig, sm: int, sku, impl_hash: str) -> List[Dict[str, Any]]:
    import oasr.jit.scaled_mm as jsm

    op, N, K, dtype, _batch = sig
    rows = []
    for name, m in pr.case.results.items():
        if name == jsm.HEURISTIC_ID or m.status != "ok" or not math.isfinite(m.median_ms):
            continue
        rows.append(
            {
                "family": jsm.op_family(op),
                "sm": sm,
                "sku": sku,
                "impl_hash": impl_hash,
                "protocol": bench.PROTOCOL_VERSION,
                "op": op,
                "M": pr.M,
                "N": N,
                "K": K,
                "dtype": dtype,
                "config": name,
                "params": jsm.choice_to_params(m.payload),
                "heuristic": jsm.config_id(
                    jsm.heuristic_config(pr.M, N, K, sm, fmt=jsm.op_format(op))
                ),
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


def _log_path(opts: BuildOptions, sm: int, family: str = "scaled_mm") -> Optional[str]:
    """Where the arms' timings go: beside the GEMM log, never inside it.

    The GEMM log calibrates the GEMM cost model; 8-bit rows have no business in it.
    """
    if opts.log_path:
        p = Path(opts.log_path)
        return str(p.with_name(f"{p.stem}.{family}{p.suffix or '.jsonl'}"))
    root = database.user_root()
    return str(root / "measurements" / f"sm{sm}_{family}.jsonl") if root is not None else None


def build_scaled_mm(
    shapes: ShapeSet,
    *,
    sm: Optional[int] = None,
    opts: Optional[BuildOptions] = None,
    base: Optional[database.TuningFile] = None,
    fmt: str = "int8",
) -> database.TuningFile:
    """Tune every *fmt* scaled-GEMM signature of *shapes* (both of its ops).

    Returns the format's tuning file -- ``scaled_mm`` (int8) or ``scaled_mm_fp8``
    -- on top of *base* when given (its entries for signatures this census does
    not name are kept).  Respects
    ``budget_s`` (heaviest signature first), ``progress``, ``min_speedup``,
    ``tie``, ``cover_eps``, ``partition``, ``fill``/``thin``/``max_points_per_sig``,
    ``policy``, ``log_path`` and ``checkpoint_path``; the GEMM-only knobs
    (``top_k``, ``compile_budget``, ``basis_size``, ``dtype_class``) do not apply.
    """
    import oasr.jit.scaled_mm as jsm

    opts = opts or BuildOptions()
    sm = int(sm if sm is not None else jsm._process_sm())
    family = jsm.TUNING_FAMILY[fmt]
    if not jsm.lane_available(sm, fmt):
        raise ValueError(f"no {fmt} scaled_mm lane is built for sm{sm}")
    tf = base if base is not None else database.new_file(family, sm, jsm.default_lane(sm, fmt))
    tf.validators["soft"] = database.current_soft_validators(family)
    t0 = time.time()
    contended = False
    sigs = {
        sig: pts
        for sig, pts in shapes.signatures().items()
        if sig[0] in jsm.FORMAT_OPS[fmt]
        and sig[1] % jsm.N_ALIGNMENT == 0
        and sig[2] % jsm.K_ALIGNMENT == 0
    }
    sigs = dict(sorted(sigs.items(), key=lambda kv: -sum(p.weight for p in kv[1])))
    impl_hash = jsm.scaled_mm_impl_hash()
    sku = database.current_soft_validators(family).get("sku")
    log_path = _log_path(opts, sm, family)
    log_f = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a")
    done = _load_checkpoint(opts.checkpoint_path, sm, impl_hash)
    ckpt_f = None
    if opts.checkpoint_path:
        Path(opts.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_f = open(opts.checkpoint_path, "a")
    resumed = 0
    gc.freeze()  # the census lives for the build; see build_gemm
    try:
        for i, (sig, pts) in enumerate(sigs.items(), 1):
            if opts.budget_s is not None and time.time() - t0 > opts.budget_s:
                if opts.progress:
                    opts.progress(
                        f"budget {opts.budget_s:.0f}s spent; {len(sigs) - i + 1} {family} "
                        "signature(s) left unmeasured"
                    )
                break
            op, N, K, _dtype, _batch = sig
            key = jsm.scaled_mm_sig_key(op, N, K)
            sid, digest = _sig_id(sig), _points_digest(pts, opts)
            rec = done.get(sid)
            if rec is not None and rec.get("digest") == digest and rec.get("key") == key:
                tf.configs.update(rec["configs"])
                tf.entries[key] = database.Entry.from_json(rec["entry"])
                contended |= bool(rec["contended"])
                resumed += 1
                continue
            results = measure_signature(sig, pts, shapes.working_set_bytes, opts, sm)
            sig_contended = any(r.case.conditions.contended for r in results)
            contended |= sig_contended
            if log_f is not None:
                for pr in results:
                    for row in _log_rows(pr, sig, sm, sku, impl_hash):
                        log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if opts.partition == "cover":
                regions, notes = partition.regions_for_signature(
                    results, jsm.HEURISTIC_ID, opts.cover_eps, min_speedup=opts.min_speedup
                )
            else:
                regions, notes = partition.regions_from_decisions(results, jsm.HEURISTIC_ID)
            if not regions:
                regions = [(None, jsm.HEURISTIC_ID)]
            if all(c == jsm.HEURISTIC_ID for _, c in regions):
                notes.append(
                    "every region keeps the shape heuristic's pick: no variant beats it by "
                    f"{opts.min_speedup}x over a region"
                )
            payloads = {n: m.payload for pr in results for n, m in pr.case.results.items()}
            payloads[jsm.HEURISTIC_ID] = jsm.HEURISTIC_ID
            configs = {cid: jsm.choice_to_params(payloads[cid]) for _, cid in regions}
            tf.configs.update(configs)
            entry = database.Entry(
                regions=regions,
                evidence={f"M={pr.M}": _evidence(pr, jsm.HEURISTIC_ID) for pr in results},
                source="aot",
                notes=notes
                + [f"M={pr.M}{'*' if pr.must else ''}: {pr.choice} ({pr.note})" for pr in results],
            )
            tf.entries[key] = entry
            if ckpt_f is not None:
                _append_checkpoint(
                    ckpt_f,
                    {
                        "family": family,
                        "sig": sid,
                        "digest": digest,
                        "sm": sm,
                        "impl": impl_hash,
                        "key": key,
                        "configs": configs,
                        "entry": entry.to_json(),
                        "contended": sig_contended,
                    },
                )
            if opts.progress:
                opts.progress(f"[{i}/{len(sigs)}] {key}: {regions}  ({time.time() - t0:.0f}s)")
    finally:
        gc.unfreeze()
        if log_f is not None:
            log_f.close()
        if ckpt_f is not None:
            ckpt_f.close()
    if resumed and opts.progress:
        opts.progress(f"resumed {resumed} {family} signature(s) from {opts.checkpoint_path}")
    referenced = set(tf.referenced_configs())
    tf.configs = {k: v for k, v in tf.configs.items() if k in referenced}
    tf.provenance.update(
        {
            "tool": "oasr tune build",
            "bench_protocol": bench.PROTOCOL_VERSION,
            "census": shapes.provenance,
            "contended": contended,
            "elapsed_s": round(time.time() - t0, 1),
            "fallback": "heuristic: oasr.jit.scaled_mm.heuristic_config ("
            + (
                "_INT8_LADDER: the largest tile filling enough of the GPU for its K)"
                if fmt == "int8"
                else "largest data-parallel tile launching >= half a wave of CTAs)"
            ),
            "operands": f"random {_DTYPE_STR[fmt]}, per-token/per-channel fp32 scales, "
            f"{_OUT_DTYPE} bias, {_ACTIVATION} epilogue",
        }
    )
    return tf
