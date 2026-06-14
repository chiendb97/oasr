#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Real-GPU GEMM tuner for ASR workloads — companion to analyze_asr_cutlass_configs.py.

Where ``analyze_asr_cutlass_configs.py`` derives shapes *analytically* and asks
nvMatmulHeuristics for *predicted* configs (neither representative of, nor
executable by, the engine), this tool closes the loop with the real kernels:

  1. **Shapes** — load the EXACT shapes captured from a real engine run
     (``--mode capture --shapes shapes.json``, produced by setting
     ``OASR_CAPTURE_GEMM`` while running ``benchmarks/bench_engine.py``), or
     derive them analytically and filter to the ops that truly hit the OASR
     GEMM path (``--mode analytic``).
  2. **Bucket** — group by ``(op, N, K, dtype)`` and snap the M dimension to a
     tile-aligned ladder; keep the high-FLOP buckets covering ``--coverage``.
  3. **Benchmark** — time EVERY registered candidate (all CUTLASS tile/stage/
     split-k variants + the torch/cuBLAS backend) on GPU 1 for each
     representative shape, and pick the real winner.
  4. **Emit** — a human-readable report (winner + speedup vs ``GEMM_DEFAULT``)
     and a ready-to-paste ``_GEMM_HEURISTIC_RULES_SM120`` Python literal for the
     production selector in ``oasr/jit/gemm.py``.

Two-step capture workflow (recommended)::

    export CUDA_VISIBLE_DEVICES=GPU-...          # the healthy GPU (UUID)
    OASR_CAPTURE_GEMM=/tmp/asr_shapes.json \
        python benchmarks/bench_engine.py --subroutines offline streaming \
        --cuda-graphs off --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR"
    python scripts/tune_asr_gemm.py --mode capture --shapes /tmp/asr_shapes.json \
        --emit-rules /tmp/gemm_rules.py
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# Tile-M ladder (aligned to SM120 block_m ∈ {16,32,64,128,256}); a runtime M is
# routed to the first edge ≥ M.  The last bucket becomes the catch-all (m_max=None).
_M_LADDER = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

_ACTIVATION_SWISH = 2


# ─────────────────────────────────────────────────────────────────────────────
# Representative-shape extraction
# ─────────────────────────────────────────────────────────────────────────────


class RepShape:
    """One representative shape to benchmark."""

    __slots__ = ("op", "M", "N", "K", "dtype", "batch", "m_max", "weight")

    def __init__(self, op, M, N, K, dtype, batch, m_max, weight):
        self.op = op
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch = batch
        self.m_max = m_max  # ladder edge this bucket covers; None = catch-all
        self.weight = weight

    def __repr__(self):
        return (f"RepShape({self.op} M={self.M} N={self.N} K={self.K} "
                f"{self.dtype} batch={self.batch} m_max={self.m_max})")


def _ladder_edge(m: int) -> int:
    for e in _M_LADDER:
        if m <= e:
            return e
    return _M_LADDER[-1]


def _weighted_median(counter: Counter) -> int:
    """Call-count-weighted median of observed M values."""
    items = sorted(counter.items())
    total = sum(c for _, c in items)
    acc = 0
    for m, c in items:
        acc += c
        if acc * 2 >= total:
            return int(m)
    return int(items[-1][0])


def buckets_from_stats(stats, coverage: float) -> List[RepShape]:
    """Bucket per-(op,N,K,dtype,batch) shape stats into representative shapes.

    *stats* is a list of ``_ShapeStat`` (from oasr.tune.capture) or dicts with the
    same fields.  Within each group, observed M values are snapped to the ladder;
    each bucket's representative M is the call-count-weighted median of the M's
    that fell into it.  Buckets are kept (per group) in descending FLOP order
    until cumulative coverage of the group's FLOPs reaches *coverage*.
    """
    reps: List[RepShape] = []
    for st in stats:
        op = st.op if hasattr(st, "op") else st["op"]
        N = st.N if hasattr(st, "N") else st["N"]
        K = st.K if hasattr(st, "K") else st["K"]
        dtype = st.dtype if hasattr(st, "dtype") else st["dtype"]
        batch = st.batch if hasattr(st, "batch") else st.get("batch", 1)
        m_counts = st.m_counts if hasattr(st, "m_counts") else Counter(
            {int(m): c for m, c in st["m_counts"].items()}
        )

        # Snap observed M into ladder buckets.
        per_edge_counts: Dict[int, Counter] = defaultdict(Counter)
        per_edge_flops: Dict[int, float] = defaultdict(float)
        per_edge_calls: Dict[int, int] = defaultdict(int)
        for m, c in m_counts.items():
            edge = _ladder_edge(int(m))
            per_edge_counts[edge][int(m)] += c
            per_edge_flops[edge] += 2.0 * int(m) * N * K * batch * c
            per_edge_calls[edge] += c

        # Keep buckets covering *coverage* of FLOPs (offline, large-M dominated)
        # UNION buckets covering *coverage* of call-count (streaming, small-M but
        # frequent — where the fixed-tile default is most wasteful).  Without the
        # call-count arm, FLOP weighting alone would prune exactly the high-value
        # small-M streaming shapes.
        def _cover(weights: Dict[int, float]) -> set:
            total = sum(weights.values()) or 1.0
            kept_, acc_ = set(), 0.0
            for edge in sorted(weights, key=weights.get, reverse=True):
                kept_.add(edge)
                acc_ += weights[edge]
                if acc_ / total >= coverage:
                    break
            return kept_

        kept = sorted(_cover(per_edge_flops) | _cover(dict(per_edge_calls)))

        for edge in sorted(kept):
            rep_m = _weighted_median(per_edge_counts[edge])
            reps.append(RepShape(op, rep_m, N, K, dtype, batch, edge,
                                 per_edge_flops[edge]))
    # Mark the largest kept bucket per (op,N,K,dtype,batch) as the catch-all.
    by_group: Dict[Tuple, List[RepShape]] = defaultdict(list)
    for r in reps:
        by_group[(r.op, r.N, r.K, r.dtype, r.batch)].append(r)
    for grp in by_group.values():
        grp.sort(key=lambda r: r.m_max)
        grp[-1].m_max = None  # catch-all for large/unseen M
    return reps


def shapes_from_capture(path: str, coverage: float) -> List[RepShape]:
    from oasr.tune.capture import GemmShapeRecorder

    stats = GemmShapeRecorder.load_json(path)
    return buckets_from_stats(stats, coverage)


def shapes_from_analytic(families, sizes, batches, durations, dtype, coverage):
    """Analytic fallback — reuse derive_problems, keep only OASR-path ops."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_asr_cutlass_configs import MODEL_REGISTRY, derive_problems
    from oasr.tune.capture import GemmShapeRecorder

    rec = GemmShapeRecorder()
    for fam in families:
        for size in sizes:
            spec = MODEL_REGISTRY.get((fam, size))
            if spec is None:
                continue
            for b in batches:
                for dur in durations:
                    gemms, _ = derive_problems(spec, b, dur)
                    for p in gemms:
                        op = _oasr_op_for(p.op_name)
                        if op is None:
                            continue  # not an OASR GEMM-path op (attn/ctc/etc.)
                        if p.batch != 1:
                            continue  # batched attn goes through fmha, skip
                        rec.record(op, p.M, p.N, p.K, dtype, 1)
    return buckets_from_stats(rec.aggregate(), coverage)


def _oasr_op_for(op_name: str) -> Optional[str]:
    """Map an analytic op_name to the OASR functional op, or None if off-path."""
    if op_name.endswith("_expand"):
        return "gemm_activation"   # FF/conv expand fuse swish
    if op_name.endswith("_contract"):
        return "gemm"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Real benchmarking over the registered candidate set
# ─────────────────────────────────────────────────────────────────────────────


def _bench(fn, warmup: int, rep: int) -> float:
    """Median ms via triton.do_bench, CUDA-event fallback."""
    try:
        from triton.testing import do_bench

        return float(do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))
    except Exception:
        import torch

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(rep):
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        return float(statistics.median(times))


def _alloc_args(shape: RepShape):
    import torch

    dt = getattr(torch, shape.dtype)
    dev = "cuda"
    M, N, K, batch = shape.M, shape.N, shape.K, shape.batch
    if shape.op == "bmm":
        A = torch.randn(batch, M, K, device=dev, dtype=dt)
        B = torch.randn(batch, N, K, device=dev, dtype=dt)
        out = torch.empty(batch, M, N, device=dev, dtype=dt)
        return out, A, B
    A = torch.randn(M, K, device=dev, dtype=dt)
    B = torch.randn(N, K, device=dev, dtype=dt)
    C = torch.randn(N, device=dev, dtype=dt)
    out = torch.empty(M, N, device=dev, dtype=dt)
    if shape.op == "gemm_activation":
        return out, A, B, C, _ACTIVATION_SWISH
    return out, A, B, C


class TacticResult:
    def __init__(self, tactic, median_ms, is_default):
        self.tactic = tactic
        self.median_ms = median_ms
        self.is_default = is_default


def _pick_winner(results: List["TacticResult"], tie_tol: float = 0.05) -> "TacticResult":
    """Pick the winner, breaking near-ties deterministically.

    At small M many tiles bottom out at the same latency floor, so the raw
    fastest flips arbitrarily between equivalent tiles.  Among all tactics within
    ``tie_tol`` of the measured best, prefer (in order): torch, then the smallest
    ``(block_m, block_n, kStages, split_k)`` CUTLASS tile.  This keeps adjacent
    M-buckets consistent without changing genuine winners (a backend that is
    truly faster than the tolerance band is always kept).
    """
    best_ms = results[0].median_ms
    band = [r for r in results if r.median_ms <= best_ms * (1.0 + tie_tol)]

    def key(r):
        if r.tactic.backend == "torch":
            return (0, 0, 0, 0, 0)
        c = dict(r.tactic.config)
        return (1, c.get("block_m", 1 << 30), c.get("block_n", 1 << 30),
                c.get("kStages", 0), c.get("split_k", 1))

    return min(band, key=key)


def benchmark_shape(shape: RepShape, warmup: int, rep: int) -> Optional[List[TacticResult]]:
    from oasr.tune.autotuner import _ensure_backends_registered, _global_registry, OpKey

    _ensure_backends_registered()
    op_key = OpKey("gemm", shape.op)
    candidates = _global_registry.get_candidates(op_key)
    if not candidates:
        return None
    args = _alloc_args(shape)
    results: List[TacticResult] = []
    for entry in candidates:
        try:
            runner = entry.get_runner()
            ms = _bench(lambda: runner(*args), warmup, rep)
        except Exception:
            ms = float("inf")
        results.append(TacticResult(entry.tactic, ms, entry.is_fallback))
    results.sort(key=lambda r: r.median_ms)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rule emission
# ─────────────────────────────────────────────────────────────────────────────


def _cutlass_literal(tactic, sm: int) -> str:
    d = dict(tactic.config)
    return (
        "CutlassGemmConfig("
        f"block_m={d['block_m']}, block_n={d['block_n']}, block_k={d['block_k']}, "
        f"warp_m={d['warp_m']}, warp_n={d['warp_n']}, warp_k={d['warp_k']}, "
        f"kStages={d['kStages']}, kSmVersion={sm}, split_k={d.get('split_k', 1)})"
    )


def _choice_literal(tactic, sm: int, is_default: bool) -> str:
    if tactic.backend == "torch":
        return '"torch"'
    if is_default:
        return "GEMM_DEFAULT"
    return _cutlass_literal(tactic, sm)


def emit_rules(per_shape: Dict[Tuple, List[TacticResult]], reps: List[RepShape],
               sm: int) -> str:
    """Build the _GEMM_HEURISTIC_RULES_SM<sm> Python literal from sweep winners."""
    # Group reps by (op, N, K) and order by m_max (None last).
    by_key: Dict[Tuple[str, int, int], List[RepShape]] = defaultdict(list)
    for r in reps:
        by_key[(r.op, r.N, r.K)].append(r)

    lines = [f"_GEMM_HEURISTIC_RULES_SM{sm}: Dict[Tuple[str, int, int], list] = {{"]
    for (op, N, K), grp in sorted(by_key.items()):
        grp.sort(key=lambda r: (r.m_max is None, r.m_max or 0))
        rule_entries = []  # (m_max | None, choice_literal, comment)
        all_default = True
        for r in grp:
            res = per_shape.get((r.op, r.M, r.N, r.K, r.dtype, r.batch))
            if not res:
                continue
            winner = _pick_winner(res)
            default = next((t for t in res if t.is_default), None)
            speedup = (default.median_ms / winner.median_ms) if default and winner.median_ms > 0 else 1.0
            choice = _choice_literal(winner.tactic, sm, winner.is_default)
            if choice != "GEMM_DEFAULT":
                all_default = False
            rule_entries.append(
                (r.m_max, choice,
                 f"M~{r.M}: {winner.tactic.backend} {winner.median_ms:.4f}ms "
                 f"({speedup:.2f}x vs default)"))
        if all_default or not rule_entries:
            continue  # every bucket == default → omit; falls back to GEMM_DEFAULT
        # Collapse runs of identical choice (ascending m_max): a later, larger
        # threshold subsumes earlier same-choice buckets.
        merged = []
        for m_max, choice, comment in rule_entries:
            if merged and merged[-1][1] == choice:
                merged[-1] = (m_max, choice, comment)
            else:
                merged.append((m_max, choice, comment))
        lines.append(f'    ("{op}", {N}, {K}): [')
        for m_max, choice, comment in merged:
            m_max_repr = "None" if m_max is None else str(m_max)
            lines.append(f"        ({m_max_repr}, {choice}),  # {comment}")
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def print_report(per_shape, reps, sm) -> None:
    print("\n" + "=" * 92)
    print(f"{'op':16} {'N':>6} {'K':>6} {'M':>7} {'m_max':>7}  "
          f"{'winner':>26} {'win ms':>9} {'dflt ms':>9} {'speedup':>8}")
    print("-" * 92)
    reps_sorted = sorted(reps, key=lambda r: (r.op, r.N, r.K, r.M))
    for r in reps_sorted:
        res = per_shape.get((r.op, r.M, r.N, r.K, r.dtype, r.batch))
        if not res:
            continue
        w = _pick_winner(res)
        d = next((t for t in res if t.is_default), None)
        sp = (d.median_ms / w.median_ms) if d and w.median_ms > 0 else float("nan")
        wname = w.tactic.backend if w.tactic.backend == "torch" else \
            f"cutlass {dict(w.tactic.config).get('block_m')}x" \
            f"{dict(w.tactic.config).get('block_n')}s{dict(w.tactic.config).get('kStages')}" \
            f"k{dict(w.tactic.config).get('split_k')}"
        mm = "inf" if r.m_max is None else str(r.m_max)
        print(f"{r.op:16} {r.N:>6} {r.K:>6} {r.M:>7} {mm:>7}  "
              f"{wname:>26} {w.median_ms:>9.4f} "
              f"{(d.median_ms if d else float('nan')):>9.4f} {sp:>7.2f}x")
    print("=" * 92 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["capture", "analytic"], default="capture")
    p.add_argument("--shapes", help="captured shapes JSON (OASR_CAPTURE_GEMM output)")
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES value (index or GPU-UUID). Set before torch import.")
    p.add_argument("--coverage", type=float, default=0.97,
                   help="keep top FLOP buckets until this fraction of group FLOPs is covered")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--emit-rules", metavar="FILE",
                   help="write the _GEMM_HEURISTIC_RULES Python literal here")
    # analytic-mode knobs
    p.add_argument("--families", nargs="+", default=["conformer"])
    p.add_argument("--sizes", nargs="+", default=["base"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 64])
    p.add_argument("--durations", nargs="+", type=int, default=[4, 16, 64])
    p.add_argument("--dtype", default="bfloat16")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # must precede torch import

    from oasr.jit.core import _get_target_sm

    sm = _get_target_sm()

    if args.mode == "capture":
        if not args.shapes:
            print("ERROR: --mode capture requires --shapes <captured json> "
                  "(produce it via OASR_CAPTURE_GEMM=… python benchmarks/bench_engine.py …)",
                  file=sys.stderr)
            return 2
        reps = shapes_from_capture(args.shapes, args.coverage)
    else:
        reps = shapes_from_analytic(args.families, args.sizes, args.batches,
                                    args.durations, args.dtype, args.coverage)

    if not reps:
        print("No representative shapes found.", file=sys.stderr)
        return 1

    print(f"[tune] sm={sm}  representative shapes: {len(reps)}")
    for r in reps:
        print("  ", r)

    per_shape: Dict[Tuple, List[TacticResult]] = {}
    for r in reps:
        res = benchmark_shape(r, args.warmup, args.rep)
        if res:
            per_shape[(r.op, r.M, r.N, r.K, r.dtype, r.batch)] = res

    print_report(per_shape, reps, sm)

    rules = emit_rules(per_shape, reps, sm)
    print(rules)
    if args.emit_rules:
        with open(args.emit_rules, "w") as f:
            f.write("# Auto-generated by scripts/tune_asr_gemm.py — paste into oasr/jit/gemm.py\n")
            f.write(rules + "\n")
        print(f"\n[tune] wrote rules to {args.emit_rules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
