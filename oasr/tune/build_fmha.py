# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr tune build --family fmha``: an attention census in, ``fmha.json`` out.

The joint search the attention family needs: per census signature
(:func:`oasr.jit.attention_select.sig_key`) and key length, every lane and
configuration that can serve the shape --

* the CuTeDSL kernel at each tile of
  :data:`~oasr.jit.attention_select.CUTE_TILES`,
* the C++ kernel at each split-KV setting of
  :data:`~oasr.jit.attention_select.CXX_SPLITS`,
* torch SDPA -- measured always, *selectable* only with ``allow_external``
  (``--allow-external``): an external runner winning a region is a decision
  that has to be asked for, and it is then a policy hit at every dispatch --

plus an ``"auto"`` pseudo-arm whose measurement is that of the arm today's
routing takes there (forced through every halving stage, so it is measured
to the end).  Each arm runs through :func:`oasr.functionals.attention.fmha`
itself, wrapper included, under the shared protocol
(:func:`oasr.tune.bench.measure`: graph-captured loops, rotated operand copies,
interleaved rounds, successive halving, the single-replay gate).

Numerics gate: every arm against an fp32 SDPA reference on the first rows of
the batch, within four times torch's own half-precision SDPA error (the GEMM
build's rule against cuBLAS).  The key lengths are random per row and a third
of the offline bias's key columns are ``-inf`` padding, so the masked paths are
exercised, not just the dense ones.

Regions run over the key length; the partition is the GEMM build's
(:func:`oasr.tune.partition.regions_for_signature`: set cover, segmentation,
the fallback gate) against ``"auto"``.  Every signature measured gets an entry
-- an all-auto one is ``[[null, "auto"]]`` -- because "measured, and today's
routing was best" is coverage.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from oasr.jit import attention_select as S
from oasr.tune import bench, database, partition
from oasr.tune.attention_census import AttnPoint, AttnShapeSet
from oasr.tune.build import (
    BuildOptions,
    PointResult,
    _append_checkpoint,
    _evidence,
    _load_checkpoint,
    decide_point,
)

logger = logging.getLogger("oasr.tune")

__all__ = ["AttnCaseResult", "build_fmha", "candidates", "measure_case", "plan_points"]

#: Batch rows checked against the fp32 reference: rows are independent.
_REFERENCE_ROWS = 4

#: Cap on the bytes of rotated operand copies one point allocates.
_MAX_ROTATION_BYTES = 192 << 20


@dataclass
class AttnCaseResult:
    """One measured attention point (duck-types ``gemm_tune.CaseResult``)."""

    shape: S.AttnShape
    results: Dict[str, bench.Measurement]
    conditions: bench.Conditions
    errors: Dict[str, float] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)
    l2_copies: int = 1
    #: Measured but not selectable (SDPA without ``allow_external``).
    reference: Dict[str, bench.Measurement] = field(default_factory=dict)

    @property
    def l2_state(self) -> str:
        return "warm" if self.l2_copies <= 1 else "rotated"

    def winner(self, alpha: float = 0.0, tie: float = 0.05) -> Optional[bench.Measurement]:
        return bench.pick_winner(self.results, alpha=alpha, tie=tie, prefer=_prefer)


def _prefer(m: bench.Measurement) -> Tuple:
    """Within a tie: today's routing, then an OASR lane, then the smaller configuration."""
    if m.name == S.AUTO_ID:
        return (0, 0)
    c = m.payload
    if isinstance(c, S.AttnChoice):
        lane = {"cute": 1, "cxx": 1, "sdpa": 2}[c.lane]
        size = c.m_block * c.n_block if c.lane == "cute" else c.splits
        return (lane, size)
    return (3, 0)


def candidates(shape: S.AttnShape) -> Dict[str, S.AttnChoice]:
    """``{config id: choice}`` of every lane configuration that can serve *shape* here."""
    out: Dict[str, S.AttnChoice] = {}
    for m, n, t in S.CUTE_TILES:
        c = S.AttnChoice("cute", m_block=m, n_block=n, threads=t)
        out[S.config_id(c)] = c
    for s in S.CXX_SPLITS:
        c = S.AttnChoice("cxx", splits=s)
        out[S.config_id(c)] = c
    out["sdpa"] = S.AttnChoice("sdpa")
    return {cid: c for cid, c in out.items() if S.unusable_reason(c, shape) is None}


def _routed_id(shape: S.AttnShape, cands: Dict[str, S.AttnChoice]) -> Optional[str]:
    """The candidate today's routing runs for *shape* (what ``"auto"`` stands for)."""
    lane = S._routed_lane(shape)
    cid = {"cute": "cute_m64n64t128", "cxx": "cxx", "sdpa": "sdpa"}.get(lane)
    return cid if cid in cands else None


# =============================================================================
# Operands
# =============================================================================


def _operands(pt: AttnPoint, shape: S.AttnShape, g: torch.Generator) -> Dict[str, Any]:
    """One set of realistic operands for *shape* (dense or paged), plus what built it."""
    dt = getattr(torch, shape.dtype)
    B, H, Hkv, Tq, Tk, D = (
        shape.batch,
        shape.num_heads,
        shape.num_kv_heads,
        shape.t_q,
        shape.t_k,
        shape.head_dim,
    )
    dev = "cuda"
    q = torch.randn(B, H, Tq, D, device=dev, dtype=dt, generator=g)
    lens = None
    if pt.lens:
        lo = max(1, (3 * Tk) // 5)
        lens = torch.randint(lo, Tk + 1, (B,), device=dev, generator=g, dtype=torch.int32)
        lens[0] = Tk
    starts = None
    if shape.seqstart and lens is not None:
        starts = torch.randint(0, max(1, Tk // 8), (B,), device=dev, generator=g, dtype=torch.int32)
        starts[0] = 0
    bias = None
    if shape.bias:
        # The bias spans the logical key axis, whose parity the signature keys:
        # a paged table is whole tiles, so an unaligned bias there is one short.
        tb = Tk - 1 if shape.paged and not shape.bias_aligned else Tk
        bias = (torch.randn(B, H, Tq, tb, device=dev, generator=g) * 2).to(dt)
        if lens is None:
            # Key padding folded into the bias, as WeNet does offline.
            keep = torch.randint(max(1, (2 * tb) // 3), tb + 1, (B,), device=dev, generator=g)
            keep[0] = tb
            pad = torch.arange(tb, device=dev)[None, :] >= keep[:, None]
            bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))
    block_table = None
    if shape.paged:
        bs = shape.block_size
        per_seq = -(-Tk // bs)
        pool_blocks = B * per_seq + 1
        k = torch.randn(pool_blocks, bs, Hkv, D, device=dev, dtype=dt, generator=g)
        v = torch.randn(pool_blocks, bs, Hkv, D, device=dev, dtype=dt, generator=g)
        perm = torch.randperm(pool_blocks - 1, device=dev, generator=g)[: B * per_seq] + 1
        block_table = perm.view(B, per_seq).to(torch.int32).contiguous()
    else:
        k = torch.randn(B, Hkv, Tk, D, device=dev, dtype=dt, generator=g)
        v = torch.randn(B, Hkv, Tk, D, device=dev, dtype=dt, generator=g)
    return {
        "q": q,
        "k": k,
        "v": v,
        "bias": bias,
        "lens": lens,
        "starts": starts,
        "block_table": block_table,
        "causal": shape.mask == "causal",
        "window": (pt.window_left, pt.window_right) if shape.mask == "window" else (-1, -1),
        "scale": 1.0 / math.sqrt(D),
    }


def _bytes(ops: Dict[str, Any]) -> int:
    return sum(
        int(t.numel() * t.element_size())
        for t in (ops["q"], ops["k"], ops["v"], ops["bias"])
        if t is not None
    )


def _call(ops: Dict[str, Any], out: torch.Tensor, backend: Any):
    from oasr.functionals.attention import fmha

    wl, wr = ops["window"]

    def run():
        return fmha(
            ops["q"],
            ops["k"],
            ops["v"],
            softmax_scale=ops["scale"],
            attn_bias=ops["bias"],
            cache_seqlens=ops["lens"],
            cache_seqstarts=ops["starts"],
            block_table=ops["block_table"],
            causal=ops["causal"],
            window_left=wl,
            window_right=wr,
            out=out,
            validate=False,
            backend=backend,
        )

    return run


def _reference(ops: Dict[str, Any], rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(fp32 reference, torch's own half-precision SDPA)`` on the first *rows* batch rows."""
    from oasr.functionals.attention import _sdpa_reference, gather_paged_kv

    q, k, v = ops["q"][:rows], ops["k"], ops["v"]
    if ops["block_table"] is not None:
        k, v = gather_paged_kv(k, v, ops["block_table"][:rows])
        tk = ops["bias"].size(-1) if ops["bias"] is not None else k.size(2)
        k, v = k[:, :, :tk], v[:, :, :tk]
    else:
        k, v = k[:rows], v[:rows]
    bias = None if ops["bias"] is None else ops["bias"][:rows]
    lens = None if ops["lens"] is None else ops["lens"][:rows]
    starts = None if ops["starts"] is None else ops["starts"][:rows]
    wl, wr = ops["window"]
    causal, scale = ops["causal"], ops["scale"]
    bias32 = None if bias is None else bias.float()
    ref = _sdpa_reference(
        q.float(), k.float(), v.float(), scale, bias32, lens, causal, starts, wl, wr
    )
    lib = _sdpa_reference(
        q, k.contiguous(), v.contiguous(), scale, bias, lens, causal, starts, wl, wr
    )
    return ref, lib


# =============================================================================
# Measurement
# =============================================================================


def measure_case(
    pt: AttnPoint,
    *,
    policy: bench.BenchPolicy = bench.BenchPolicy(),
    allow_external: bool = False,
) -> AttnCaseResult:
    """Measure every candidate at one point, plus the ``"auto"`` pseudo-arm."""
    shape = pt.shape
    cands = candidates(shape)
    routed = _routed_id(shape, cands)
    g = torch.Generator(device="cuda").manual_seed(0)
    base = _operands(pt, shape, g)
    copies = bench.l2_copies(
        _bytes(base),
        max_copies=policy.max_rotation_calls,
        max_total_bytes=_MAX_ROTATION_BYTES,
    )
    sets = [base] + [
        {k: (t.clone() if isinstance(t, torch.Tensor) else t) for k, t in base.items()}
        for _ in range(copies - 1)
    ]
    out = torch.empty_like(base["q"])
    rows = min(shape.batch, _REFERENCE_ROWS)
    ref, lib = _reference(base, rows)
    lib_err = bench.numerics_error(lib, ref)
    dtype = base["q"].dtype

    errors: Dict[str, float] = {}
    rejected: Dict[str, str] = {}
    usable: Dict[str, S.AttnChoice] = {}
    order = ([routed] if routed else []) + sorted(c for c in cands if c != routed)
    for cid in order:
        choice = cands[cid]
        try:
            _call(base, out, choice)()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 -- recorded per arm
            bench._sync_or_sticky(f"first call of {cid}")
            rejected[cid] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        err = bench.numerics_error(out[:rows], ref)
        errors[cid] = err
        if not bench.numerics_ok(err, lib_err, dtype):
            rejected[cid] = f"numerics: max|err| {err:.3g} vs torch SDPA {lib_err:.3g}"
            continue
        usable[cid] = choice

    arms = [
        bench.Arm(
            cid,
            lambda i, c=choice, sets=sets, out=out: _call(sets[i % copies], out, c),
            forced=cid == routed,
            payload=choice,
        )
        for cid, choice in usable.items()
    ]
    results, cond = bench.measure(arms, policy, copies=copies)
    reference: Dict[str, bench.Measurement] = {}
    if not allow_external and "sdpa" in results:
        reference["sdpa"] = results.pop("sdpa")
    if routed is not None:
        twin = (results if routed in results else reference).get(routed)
        if twin is not None:
            results[S.AUTO_ID] = dataclasses.replace(
                twin, name=S.AUTO_ID, payload=S.AUTO_ID, samples=list(twin.samples)
            )
    del sets, base, out, ref, lib
    torch.cuda.empty_cache()
    return AttnCaseResult(shape, results, cond, errors, rejected, copies, reference)


# =============================================================================
# Points per signature
# =============================================================================


def _round_tk(t_k: int, shape: S.AttnShape) -> int:
    """A fill key length the shape can have.

    A paged one is whole 64-key tiles; a dense one with a bias keeps the bias's
    parity, which the signature keys (``AttnShape.bias_aligned``).
    """
    if shape.paged:
        return max(64, -(-t_k // 64) * 64)
    t_k = max(1, t_k)
    if shape.bias and (t_k % 2 == 0) != shape.bias_aligned:
        t_k += 1
    return t_k


def plan_points(points: Sequence[AttnPoint], cap: int, fill: bool = True) -> List[AttnPoint]:
    """The key lengths one signature is measured at, one point each.

    Census points sharing a key length (batches inside one bucket, query lengths
    inside one class) are measured once, at the heaviest's batch and query
    length, with their weight summed.  More than *cap* lengths are thinned to
    log-spaced ones, a dropped point's weight moving to the next kept one up
    (regions round up).  Then fill points between the lengths, up to *cap*.
    """
    by_tk: Dict[int, List[AttnPoint]] = {}
    for p in points:
        by_tk.setdefault(int(p.t_k), []).append(p)
    merged: List[AttnPoint] = []
    for tk in sorted(by_tk):
        group = by_tk[tk]
        rep = max(group, key=lambda p: (p.weight, p.batch, p.t_q))
        m = dataclasses.replace(rep, keys=list(rep.keys))
        m.calls = sum(p.calls for p in group)
        m.weight = sum(p.weight for p in group)
        merged.append(m)
    if len(merged) > cap:
        lo, hi = math.log(merged[0].t_k), math.log(merged[-1].t_k)
        want = {merged[0].t_k, merged[-1].t_k}
        for i in range(1, cap - 1):
            target = math.exp(lo + (hi - lo) * i / (cap - 1))
            want.add(min(merged, key=lambda p: abs(p.t_k - target)).t_k)
        kept: List[AttnPoint] = []
        carry_w = carry_c = 0.0
        for p in merged:
            carry_w += p.weight
            carry_c += p.calls
            if p.t_k in want:
                p.weight, p.calls = carry_w, carry_c
                carry_w = carry_c = 0.0
                kept.append(p)
        merged = kept
    if not fill or len(merged) >= cap or not merged:
        return merged
    from oasr.tune.shapes import ladder

    self_attn = all(p.t_q == p.t_k for p in merged)
    have = {p.t_k for p in merged}
    rep = max(merged, key=lambda p: p.weight)
    floor_w = 0.25 * min(p.weight for p in merged if p.weight > 0) if merged else 1.0
    extra = []
    for e in ladder(merged[-1].t_k, merged[0].t_k):
        tk = _round_tk(e, rep.shape)
        if tk in have or tk > merged[-1].t_k:
            continue
        have.add(tk)
        fp = dataclasses.replace(rep, t_k=tk, calls=0.0, weight=floor_w, keys=["fill"])
        if self_attn:
            fp.t_q = tk
        extra.append(fp)
    room = cap - len(merged)
    if len(extra) > room:
        step = len(extra) / max(room, 1)
        extra = [extra[int(i * step)] for i in range(room)]
    return sorted(merged + extra, key=lambda p: p.t_k)


def _log_rows(pr: PointResult, key: str, sm: int, sku, impl: str) -> List[Dict[str, Any]]:
    res: AttnCaseResult = pr.case  # type: ignore[assignment]
    rows = []
    for name, m in list(res.results.items()) + list(res.reference.items()):
        if name == S.AUTO_ID or m.status != "ok" or not math.isfinite(m.median_ms):
            continue
        rows.append(
            {
                "family": S.FAMILY,
                "sm": sm,
                "sku": sku,
                "impl_hash": impl,
                "protocol": bench.PROTOCOL_VERSION,
                "sig": key,
                **dataclasses.asdict(res.shape),
                "config": name,
                "selectable": name in res.results,
                "median_ms": round(m.median_ms, 7),
                "sigma_ms": round(m.sigma_ms, 7),
                "solo_ms": None if m.solo_ms is None else round(m.solo_ms, 7),
                "n": m.n,
                "stage": m.eliminated_at,
                "l2": res.l2_state,
                "calls": m.calls_per_graph,
                "contended": res.conditions.contended,
            }
        )
    return rows


def _digest(pts: Sequence[AttnPoint], opts: BuildOptions, allow_external: bool) -> str:
    import hashlib

    h = hashlib.sha256()
    rows = sorted((p.t_k, p.t_q, p.batch, round(p.weight, 9)) for p in pts)
    h.update(json.dumps(rows).encode())
    knobs = [opts.max_points_per_sig, opts.fill, opts.min_speedup, opts.tie, opts.cover_eps]
    knobs += [opts.partition, bench.PROTOCOL_VERSION, allow_external, S.CUTE_TILES, S.CXX_SPLITS]
    h.update(json.dumps(knobs).encode())
    return h.hexdigest()[:16]


def _log_path(opts: BuildOptions, sm: int) -> Optional[str]:
    if opts.log_path:
        p = Path(opts.log_path)
        return str(p.with_name(f"{p.stem}.fmha{p.suffix or '.jsonl'}"))
    root = database.user_root()
    return str(root / "measurements" / f"sm{sm}_fmha.jsonl") if root is not None else None


def build_fmha(
    shapes: AttnShapeSet,
    *,
    sm: Optional[int] = None,
    opts: Optional[BuildOptions] = None,
    base: Optional[database.TuningFile] = None,
    allow_external: bool = False,
) -> database.TuningFile:
    """Tune every signature of an attention census; the ``fmha`` tuning file.

    On top of *base* when given (its entries for signatures this census does not
    name are kept).  Respects ``budget_s`` (heaviest signature first),
    ``progress``, ``min_speedup``, ``tie``, ``cover_eps``, ``partition``,
    ``fill``, ``max_points_per_sig``, ``policy``, ``log_path`` and
    ``checkpoint_path``.
    """
    opts = opts or BuildOptions()
    sm = int(sm if sm is not None else S._sm())
    tf = base if base is not None else database.new_file(S.FAMILY, sm, "attention")
    database.stamp_current(tf)
    t0 = time.time()
    impl = S.impl_hash()
    sku = database.current_soft_validators(S.FAMILY).get("sku")
    sigs = dict(sorted(shapes.signatures().items(), key=lambda kv: -sum(p.weight for p in kv[1])))
    log_path = _log_path(opts, sm)
    log_f = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a")
    done = _load_checkpoint(opts.checkpoint_path, sm, impl)
    ckpt_f = None
    if opts.checkpoint_path:
        Path(opts.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_f = open(opts.checkpoint_path, "a")
    contended = False
    resumed = 0
    headroom: Dict[str, float] = {"auto": 0.0, "tuned": 0.0, "with_sdpa": 0.0}
    gc.freeze()
    try:
        for i, (key, pts) in enumerate(sigs.items(), 1):
            if opts.budget_s is not None and time.time() - t0 > opts.budget_s:
                if opts.progress:
                    opts.progress(f"budget spent; {len(sigs) - i + 1} fmha signature(s) left")
                break
            plan = plan_points(pts, opts.max_points_per_sig, opts.fill)
            digest = _digest(plan, opts, allow_external)
            rec = done.get(key)
            if rec is not None and rec.get("digest") == digest:
                tf.configs.update(rec["configs"])
                tf.entries[key] = database.Entry.from_json(rec["entry"])
                contended |= bool(rec["contended"])
                resumed += 1
                continue
            results: List[PointResult] = []
            for p in plan:
                res = measure_case(p, policy=opts.policy, allow_external=allow_external)
                choice, sp, note = decide_point(res, S.AUTO_ID, 0.0, opts)  # type: ignore[arg-type]
                if res.reference:
                    r = res.reference["sdpa"]
                    note += f" [sdpa {r.median_ms * 1e3:.1f} us, reference only]"
                results.append(
                    PointResult(p.t_k, False, 0.0, res, choice, sp, note, max(p.weight, 1e-12))  # type: ignore[arg-type]
                )
                _headroom(headroom, res, p.calls)
                if opts.progress:
                    opts.progress(f"{key} T_k={p.t_k} B={p.batch} T_q={p.t_q}: {choice} ({note})")
            sig_contended = any(r.case.conditions.contended for r in results)
            contended |= sig_contended
            if log_f is not None:
                for pr in results:
                    for row in _log_rows(pr, key, sm, sku, impl):
                        log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if opts.partition == "cover":
                regions, notes = partition.regions_for_signature(
                    results, S.AUTO_ID, opts.cover_eps, min_speedup=opts.min_speedup
                )
            else:
                regions, notes = partition.regions_from_decisions(results, S.AUTO_ID)
            if not regions:
                regions = [(None, S.AUTO_ID)]
            payloads: Dict[str, Any] = {S.AUTO_ID: S.AUTO_ID}
            for pr in results:
                for n, m in pr.case.results.items():
                    payloads.setdefault(n, m.payload)
            configs = {cid: S.choice_to_params(payloads[cid]) for _, cid in regions}
            tf.configs.update(configs)
            entry = database.Entry(
                regions=regions,
                evidence={f"T_k={pr.M}": _evidence(pr, S.AUTO_ID) for pr in results},
                source="aot",
                notes=notes
                + [
                    f"T_k={pr.M} B={pr.case.shape.batch} T_q={pr.case.shape.t_q}: "  # type: ignore[attr-defined]
                    f"{pr.choice} ({pr.note})"
                    for pr in results
                ],
            )
            tf.entries[key] = entry
            if ckpt_f is not None:
                _append_checkpoint(
                    ckpt_f,
                    {
                        "family": S.FAMILY,
                        "sig": key,
                        "digest": digest,
                        "sm": sm,
                        "impl": impl,
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
        opts.progress(f"resumed {resumed} fmha signature(s) from {opts.checkpoint_path}")
    referenced = set(tf.referenced_configs())
    tf.configs = {k: v for k, v in tf.configs.items() if k in referenced}
    tf.provenance.update(
        {
            "tool": "oasr tune build",
            "bench_protocol": bench.PROTOCOL_VERSION,
            "census": shapes.provenance,
            "contended": contended,
            "elapsed_s": round(time.time() - t0, 1),
            "fallback": "auto: OASR_ATTN_BACKEND=auto routing (CuTeDSL 64x64x128 where built)",
            "candidates": {
                "cute_tiles": [list(t) for t in S.CUTE_TILES],
                "cxx_splits": list(S.CXX_SPLITS),
                "sdpa": "selectable" if allow_external else "measured as a reference only",
            },
            "weighted_ms": {k: round(v, 6) for k, v in headroom.items()},
        }
    )
    return tf


def _headroom(acc: Dict[str, float], res: AttnCaseResult, calls: float) -> None:
    """Accumulate census-weighted time (calls x measured median, in ms per unit of traffic).

    Today's routing, the best selectable arm, and the best with SDPA too.  Fill
    points carry no calls and add nothing.
    """
    auto = res.results.get(S.AUTO_ID)
    if auto is None or not math.isfinite(auto.median_ms) or calls <= 0:
        return
    ok = [m.median_ms for m in res.results.values() if m.status == "ok"]
    best = min(ok, default=auto.median_ms)
    ext = min([best] + [m.median_ms for m in res.reference.values() if m.status == "ok"])
    acc["auto"] += calls * auto.median_ms
    acc["tuned"] += calls * best
    acc["with_sdpa"] += calls * ext
