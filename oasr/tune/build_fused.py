# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Census and build for the path families: ``recurrent`` and ``gated_mlp``.

Both families choose a *path* per shape -- the recurrent layer's scalar kernel,
tensor-core GEMM path (and its tactic) or fused step (lane and tile); the gated
MLP's fused kernel (lane and tile) or two GEMMs -- where the shipped code applies
cut-offs timed on one RTX 5090 (:mod:`oasr.jit.measured`).

**Census.**  The static half of a signature is a property of the model, not of
the traffic: an LSTM's hidden and input widths, a gated MLP's ``N`` / ``K`` /
activation.  :func:`fused_census` reads them off the model's own
:class:`oasr.layers.LSTM` / :class:`~oasr.layers.RNN` /
:class:`~oasr.layers.GatedMLP` modules (built on the CPU: no GPU, no serving run),
or takes them as given, and crosses them with a ladder of the dynamic size --
batch for a recurrent layer, rows for a gated MLP -- which is what the engine
varies.  A decode step's batch is the number of live streams or hypotheses, so a
ladder is the honest census for it.

**Build.**  Per signature and dynamic size, every path that can serve the point,
run through the families' own functionals, under the shared protocol
(:func:`oasr.tune.bench.measure`):

* a recurrent point is timed as a **dependent chain** -- call ``i`` reads the
  state call ``i - 1`` wrote -- because that is what a recurrence imposes, and
  the only protocol that compares a fused step, a GEMM pair and a scalar kernel
  fairly (``oasr.jit.recurrent_cute`` records the harness error it replaced);
* a gated-MLP point rotates copies of its weights when they would not stay in
  L2, like the GEMM build.

Numerics: every arm against an fp32 formula on the first rows, within four times
the error of the arm today's rules take (the ``"auto"`` pseudo-arm, measured to
the end).  The partition is the GEMM build's, against ``auto``.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from oasr.tune import bench, database, partition
from oasr.tune.build import (
    BuildOptions,
    PointResult,
    _append_checkpoint,
    _evidence,
    _load_checkpoint,
    decide_point,
)

logger = logging.getLogger("oasr.tune")

__all__ = [
    "FAMILIES",
    "FusedShapeSet",
    "fused_census",
    "census_family",
    "build_recurrent",
    "build_gated_mlp",
    "recurrent_candidates",
    "gated_mlp_candidates",
    "DEFAULT_BATCHES",
    "DEFAULT_ROWS",
]

FAMILIES = ("recurrent", "gated_mlp")

#: The recurrent batch ladder: the grid the hand-timed tables were swept on.
DEFAULT_BATCHES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
#: Timesteps per recurrent signature class: a decode step, and a sequence.
DEFAULT_SEQ_LENS: Tuple[int, ...] = (1, 32)
#: The gated-MLP row ladder: decode batches through prefill.
DEFAULT_ROWS: Tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048)

#: Rows checked against the fp32 reference.
_REFERENCE_ROWS = 64

_MAX_ROTATION_BYTES = 512 << 20


# =============================================================================
# The census
# =============================================================================


@dataclass
class FusedShapeSet:
    """A census of one path family: points are the family's shape fields + a weight."""

    family: str
    points: List[Dict[str, Any]]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "family": self.family,
            "provenance": self.provenance,
            "points": self.points,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "FusedShapeSet":
        if d.get("family") not in FAMILIES:
            raise ValueError(f"not a recurrent / gated_mlp census (family {d.get('family')!r})")
        return cls(str(d["family"]), list(d["points"]), dict(d.get("provenance", {})))

    def save(self, path: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "FusedShapeSet":
        with open(path) as f:
            return cls.from_json(json.load(f))

    def signatures(self) -> Dict[str, List[Dict[str, Any]]]:
        """Points by tuning-DB signature, each list ascending in its dynamic size."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for p in self.points:
            out.setdefault(_sig(self.family, p), []).append(p)
        dyn = _DYNAMIC[self.family]
        for pts in out.values():
            pts.sort(key=lambda p: int(p[dyn]))
        return out


#: The dynamic field of each family's points.
_DYNAMIC = {"recurrent": "batch", "gated_mlp": "rows"}


def census_family(path: str) -> str:
    """The kernel family a census file is for: ``gemm`` unless it says otherwise."""
    with open(path) as f:
        return str(json.load(f).get("family") or "gemm")


def _rec_shape(p: Dict[str, Any]):
    from oasr.jit.recurrent_select import RecShape

    return RecShape(
        str(p["op"]),
        str(p["dtype"]),
        int(p["hidden"]),
        int(p["input_size"]),
        int(p["seq_len"]),
        int(p["batch"]),
    )


def _mlp_shape(p: Dict[str, Any]):
    from oasr.jit.mlp_select import MlpShape

    return MlpShape(
        str(p["dtype"]),
        str(p["activation"]),
        bool(p["bias"]),
        int(p["n"]),
        int(p["k"]),
        int(p["rows"]),
    )


def _sig(family: str, p: Dict[str, Any]) -> str:
    if family == "recurrent":
        from oasr.jit.recurrent_select import sig_key

        return sig_key(_rec_shape(p))
    from oasr.jit.mlp_select import sig_key as mlp_sig

    return mlp_sig(_mlp_shape(p))


def model_dims(model) -> Dict[str, List[Dict[str, Any]]]:
    """The static recurrent and gated-MLP dims of *model*'s layers, deduplicated."""
    from oasr.layers import LSTM, RNN, GatedMLP

    rec: Dict[tuple, Dict[str, Any]] = {}
    mlp: Dict[tuple, Dict[str, Any]] = {}
    for _, mod in model.named_modules():
        if isinstance(mod, (LSTM, RNN)):
            op = "lstm" if isinstance(mod, LSTM) else f"rnn_{mod.nonlinearity}"
            widths = {int(mod.input_size)}
            if int(mod.num_layers) > 1:
                widths.add(int(mod.hidden_size))
            for width in sorted(widths):
                rec[(op, int(mod.hidden_size), width)] = {
                    "op": op,
                    "hidden": int(mod.hidden_size),
                    "input_size": width,
                }
        elif isinstance(mod, GatedMLP):
            w = mod._proj(0).weight
            key = (mod.activation, mod._proj(0).bias is not None, int(w.shape[0]), int(w.shape[1]))
            mlp[key] = {
                "activation": key[0],
                "bias": key[1],
                "n": key[2],
                "k": key[3],
            }
    return {"recurrent": list(rec.values()), "gated_mlp": list(mlp.values())}


def fused_census(
    family: str,
    dims: Iterable[Dict[str, Any]],
    *,
    dtype: str = "float16",
    ladder: Optional[Sequence[int]] = None,
    seq_lens: Optional[Sequence[int]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> FusedShapeSet:
    """Cross the static *dims* of a family with its dynamic ladder."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, not {family!r}")
    points: List[Dict[str, Any]] = []
    if family == "recurrent":
        for d in dims:
            for t in seq_lens or DEFAULT_SEQ_LENS:
                for b in ladder or DEFAULT_BATCHES:
                    points.append({**d, "dtype": dtype, "seq_len": int(t), "batch": int(b)})
    else:
        for d in dims:
            for r in ladder or DEFAULT_ROWS:
                points.append({**d, "dtype": dtype, "rows": int(r)})
    for p in points:
        p.setdefault("weight", 1.0)
    prov = dict(provenance or {})
    prov.update({"family": family, "dtype": dtype, "ladder": list(ladder or [])})
    return FusedShapeSet(family, points, prov)


# =============================================================================
# Shared measurement plumbing
# =============================================================================


@dataclass
class FusedCase:
    """One measured point (duck-types ``gemm_tune.CaseResult``)."""

    shape: Any
    results: Dict[str, bench.Measurement]
    conditions: bench.Conditions
    errors: Dict[str, float] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)
    l2_copies: int = 1

    @property
    def l2_state(self) -> str:
        return "warm" if self.l2_copies <= 1 else "rotated"

    def winner(self, alpha: float = 0.0, tie: float = 0.05) -> Optional[bench.Measurement]:
        return bench.pick_winner(self.results, alpha=alpha, tie=tie, prefer=_prefer)


#: Within a tie: today's rules, then the plainest path.
_PATH_RANK = {"auto": 0, "scalar": 1, "gemm": 2, "cxx": 3, "cute": 4}


def _prefer(m: bench.Measurement) -> Tuple:
    return (_PATH_RANK.get(m.name.split("_")[0], 9), m.name)


def _measure_arms(
    calls: Dict[str, Callable[[int], Callable[[], Any]]],
    routed: Optional[str],
    check: Callable[[str], float],
    ref_err_of: Callable[[Dict[str, float]], Optional[float]],
    dtype: torch.dtype,
    policy: bench.BenchPolicy,
    copies: int,
    shape: Any,
) -> FusedCase:
    """First-call every arm, gate numerics, measure, add the ``auto`` twin."""
    errors: Dict[str, float] = {}
    rejected: Dict[str, str] = {}
    order = ([routed] if routed in calls else []) + sorted(c for c in calls if c != routed)
    ran: List[str] = []
    for name in order:
        try:
            errors[name] = check(name)
            ran.append(name)
        except Exception as exc:  # noqa: BLE001 -- recorded per arm
            bench._sync_or_sticky(f"first call of {name}")
            rejected[name] = f"{type(exc).__name__}: {exc}"[:300]
    ref_err = ref_err_of(errors)
    arms = []
    for name in ran:
        if not bench.numerics_ok(errors[name], ref_err, dtype):
            rejected[name] = f"numerics: max|err| {errors[name]:.3g} vs today's {ref_err}"
            continue
        arms.append(bench.Arm(name, calls[name], forced=name == routed))
    results, cond = bench.measure(arms, policy, copies=copies)
    if routed is not None and routed in results:
        twin = results[routed]
        results["auto"] = dataclasses.replace(
            twin, name="auto", payload="auto", samples=list(twin.samples)
        )
    torch.cuda.empty_cache()
    return FusedCase(shape, results, cond, errors, rejected, copies)


# =============================================================================
# Recurrent
# =============================================================================


def recurrent_candidates(shape) -> Dict[str, Any]:
    """``{config id: RecChoice}`` of every path that can serve *shape* here."""
    from oasr.jit import recurrent_select as R, recurrent_step as rs

    out: Dict[str, Any] = {"scalar": R.RecChoice("scalar")}
    for t, s in R.GEMM_TACTICS:
        c = R.RecChoice("gemm", tactic=t, split=s)
        out[R.config_id(c)] = c
    for lane in ("cxx", "cute"):
        for i in range(len(rs.TILES)):
            c = R.RecChoice(lane, tile=i)
            out[R.config_id(c)] = c
    return {cid: c for cid, c in out.items() if R.unusable_reason(c, shape) is None}


def _recurrent_routed_id(shape) -> str:
    """The config id today's hand-timed rules run for *shape* (what ``auto`` is)."""
    from oasr.jit import recurrent_cute as rc, recurrent_step as rs
    from oasr.layers.recurrent import _table_prefers_tensor_core

    gates = 4 if shape.op == "lstm" else 1
    tc = (
        torch.cuda.get_device_capability()[0] >= 8
        and shape.input_size % 8 == 0
        and shape.hidden % 8 == 0
        and _table_prefers_tensor_core(shape.step, shape.batch, shape.hidden, gates)
    )
    if not tc:
        return "scalar"
    if shape.step and shape.op == "lstm" and rc.should_use(4, shape.hidden, shape.batch):
        for lane in rc._lane_order():
            if lane == "cxx" and rc.cxx_config_supported(
                dtype_str=shape.dtype,
                gate_count=4,
                activation="lstm",
                hidden=shape.hidden,
                batch=shape.batch,
            ):
                return f"cxx_t{rs.routed_tile(kind='lstm', hidden=shape.hidden, batch=shape.batch)}"
            if lane == "cute" and rc.cute_config_supported(
                gate_count=4, activation="lstm", hidden=shape.hidden, batch=shape.batch
            ):
                spec = rc.select_tile(shape.hidden, shape.batch)
                for i, t in enumerate(rs.TILES):
                    if (t.block_m, t.block_n, t.block_k, t.stages, t.threads, t.warps_n) == spec:
                        return f"cute_t{i}"
    return "gemm"


def _recurrent_reference(op, x, h0, c0, w_ih, w_hh, b_ih, b_hh, rows):
    """fp32 ``(final_h, final_c)`` of one layer call on the first *rows* batch rows."""
    x, h = x[:, :rows].float(), h0[:rows].float()
    c = None if c0 is None else c0[:rows].float()
    w_ih, w_hh = w_ih.float(), w_hh.float()
    b = b_ih.float() + b_hh.float()
    for t in range(x.shape[0]):
        g = x[t] @ w_ih.t() + h @ w_hh.t() + b
        if op == "lstm":
            i, f, gg, o = g.chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
            h = torch.sigmoid(o) * torch.tanh(c)
        else:
            h = torch.tanh(g) if op == "rnn_tanh" else torch.relu(g)
    return h, c


def _recurrent_case(p: Dict[str, Any], policy: bench.BenchPolicy) -> FusedCase:
    import oasr
    from oasr.functionals.recurrent import _pack_lstm_parameters
    from oasr.jit import recurrent_cute as rc

    shape = _rec_shape(p)
    op, T, B, H, width = shape.op, shape.seq_len, shape.batch, shape.hidden, shape.input_size
    lstm = op == "lstm"
    gates = 4 if lstm else 1
    dt = getattr(torch, shape.dtype)
    g = torch.Generator(device="cuda").manual_seed(0)

    def rnd(*s, scale=1.0):
        return (torch.randn(*s, device="cuda", generator=g) * scale).to(dt)

    x = rnd(T, B, width, scale=0.5)
    w_ih, w_hh = rnd(gates * H, width, scale=width**-0.5), rnd(gates * H, H, scale=H**-0.5)
    b_ih, b_hh = rnd(gates * H, scale=0.1), rnd(gates * H, scale=0.1)
    # Ping-pong state: call i reads set i % 2 and writes the other -- a chain.
    hs = [rnd(B, H, scale=0.5), torch.empty(B, H, device="cuda", dtype=dt)]
    cs = [rnd(B, H, scale=0.5), torch.empty(B, H, device="cuda", dtype=dt)] if lstm else None
    h0 = hs[0].clone()
    c0 = cs[0].clone() if cs is not None else None
    outs = [torch.empty(T, B, H, device="cuda", dtype=dt) for _ in range(2)]
    # Packed once per point, as ``oasr.layers.LSTM`` caches it; the one-time
    # "repacked inside a layer call" warning is about callers that do it per step.
    import oasr.functionals.recurrent as _fr

    _fr._PACK_WARNED = True
    packed = _pack_lstm_parameters(w_ih, w_hh, b_ih, b_hh) if lstm else None
    nonlin = op.split("_")[-1]
    cands = recurrent_candidates(shape)
    routed = _recurrent_routed_id(shape)

    def make(cid: str) -> Callable[[int], Callable[[], Any]]:
        c = cands[cid]

        def factory(i: int) -> Callable[[], Any]:
            src, dst = i % 2, (i + 1) % 2
            if c.path == "scalar":
                if lstm:
                    return lambda: oasr.lstm_layer(
                        x, hs[src], cs[src], w_ih, w_hh, b_ih, b_hh,
                        out=outs[dst], final_h=hs[dst], final_c=cs[dst],
                    )  # fmt: skip
                return lambda: oasr.rnn_layer(
                    x, hs[src], w_ih, w_hh, b_ih, b_hh, nonlinearity=nonlin,
                    out=outs[dst], final_h=hs[dst],
                )  # fmt: skip
            if c.path == "gemm":
                from oasr.functionals.recurrent import _default_recurrent_tactic

                tactic = (c.tactic, c.split) if c.tactic >= 0 else (_default_recurrent_tactic(B), 1)
                if lstm:
                    return lambda: oasr.lstm_gemm_layer(
                        x, hs[src], cs[src], w_ih, w_hh, b_ih, b_hh,
                        final_h=hs[dst], final_c=cs[dst], _packed_parameters=packed,
                        _tactic=tactic, _outputs_are_fresh=True,
                    )  # fmt: skip
                return lambda: oasr.rnn_gemm_layer(
                    x, hs[src], w_ih, w_hh, b_ih, b_hh, nonlinearity=nonlin,
                    final_h=hs[dst], _tactic=tactic,
                )  # fmt: skip
            route = rc.step_for_tile(
                lane=c.path, dtype_str=shape.dtype, gate_count=4, activation="lstm", tile=c.tile
            )
            if route is None:
                raise RuntimeError(f"{cid} declined to build")
            return lambda: oasr.lstm_gemm_layer(
                x, hs[src], cs[src], w_ih, w_hh, b_ih, b_hh,
                final_h=hs[dst], final_c=cs[dst], _packed_parameters=packed,
                _outputs_are_fresh=True, _step_route=route,
            )  # fmt: skip

        return factory

    calls = {cid: make(cid) for cid in cands}
    rows = min(B, _REFERENCE_ROWS)
    ref_h, ref_c = _recurrent_reference(op, x, h0, c0, w_ih, w_hh, b_ih, b_hh, rows)

    def check(cid: str) -> float:
        hs[0].copy_(h0)
        if cs is not None and c0 is not None:
            cs[0].copy_(c0)
        calls[cid](0)()
        torch.cuda.synchronize()
        err = bench.numerics_error(hs[1][:rows], ref_h)
        if cs is not None and ref_c is not None:
            err = max(err, bench.numerics_error(cs[1][:rows], ref_c))
        return err

    case = _measure_arms(
        calls,
        routed,
        check,
        lambda errs: errs.get(routed) if routed in errs else None,
        dt,
        policy,
        1,
        shape,
    )
    for name, m in case.results.items():
        if name in cands:
            m.payload = cands[name]
    return case


def recurrent_census_dims(model) -> List[Dict[str, Any]]:
    return model_dims(model)["recurrent"]


# =============================================================================
# Gated MLP
# =============================================================================


def gated_mlp_candidates(shape) -> Dict[str, Any]:
    """``{config id: MlpChoice}`` of every path that can serve *shape* here."""
    from oasr.jit import mlp_select as M

    out: Dict[str, Any] = {"gemm": M.MlpChoice("gemm")}
    for lane in ("cxx", "cute"):
        for i in range(M.num_tiles()):
            c = M.MlpChoice(lane, tile=i)
            out[M.config_id(c)] = c
    return {cid: c for cid, c in out.items() if M.unusable_reason(c, shape) is None}


def _gated_mlp_routed_id(shape) -> str:
    """The config id today's rule runs for *shape*: one m-tile fused, else two GEMMs."""
    from oasr.jit import mlp as jm, mlp_select as M

    if shape.rows > jm._BAND_MAX_ROWS:
        return "gemm"
    for lane in jm._lane_order():
        if lane == "cxx" and jm.cxx_config_supported(
            dtype_str=shape.dtype,
            activation=shape.activation,
            rows=shape.rows,
            n=shape.n,
            k=shape.k,
        ):
            from oasr.jit import gated_mlp as gm

            return f"cxx_t{gm.routed_tile(rows=shape.rows, n=shape.n)}"
        if lane == "cute" and jm.cute_config_supported(rows=shape.rows, n=shape.n, k=shape.k):
            spec = jm.select_gated_mlp_tile(shape.rows, shape.n)
            for i in range(M.num_tiles()):
                if M.tile_spec(i) == tuple(spec or ()):
                    return f"cute_t{i}"
    return "gemm"


def _gated_mlp_case(p: Dict[str, Any], policy: bench.BenchPolicy) -> FusedCase:
    import oasr
    from oasr.jit import mlp as jm
    from oasr.layers.linear import _FUSED_ACTIVATION_ID
    from oasr.layers.mlp import _FUSABLE, _UNFUSED_ACTIVATION

    shape = _mlp_shape(p)
    rows, n, k, act = shape.rows, shape.n, shape.k, shape.activation
    dt = getattr(torch, shape.dtype)
    g = torch.Generator(device="cuda").manual_seed(0)
    weight_bytes = 2 * n * k * torch.tensor([], dtype=dt).element_size()
    copies = bench.l2_copies(
        weight_bytes, max_copies=policy.max_rotation_calls, max_total_bytes=_MAX_ROTATION_BYTES
    )
    x = (torch.randn(rows, k, device="cuda", generator=g) * 0.5).to(dt)
    ws = [
        (
            (torch.randn(n, k, device="cuda", generator=g) * k**-0.5).to(dt),
            (torch.randn(n, k, device="cuda", generator=g) * k**-0.5).to(dt),
        )
        for _ in range(copies)
    ]
    bias = None
    if shape.bias:
        bias = (
            (torch.randn(n, device="cuda", generator=g) * 0.1).to(dt),
            (torch.randn(n, device="cuda", generator=g) * 0.1).to(dt),
        )
    bg, bu = bias if bias is not None else (None, None)
    out = torch.empty(rows, n, device="cuda", dtype=dt)
    cands = gated_mlp_candidates(shape)
    routed = _gated_mlp_routed_id(shape)
    last: Dict[str, torch.Tensor] = {}

    def make(cid: str) -> Callable[[int], Callable[[], Any]]:
        c = cands[cid]
        route = None
        if c.path != "gemm":
            route = jm.route_for_tile(
                lane=c.path, dtype_str=shape.dtype, activation=act, has_bias=shape.bias, tile=c.tile
            )
            if route is None:
                raise RuntimeError(f"{cid} declined to build")

        def factory(i: int) -> Callable[[], Any]:
            wg, wu = ws[i % copies]
            if c.path == "gemm":
                if act in _FUSABLE:
                    # What ``LinearActivation`` passes: the epilogue's integer id.
                    act_id = oasr.get_activation_type_id(_FUSED_ACTIVATION_ID[act])

                    def run_gemm():
                        last[cid] = oasr.gemm_activation(x, wg, bg, act_id) * oasr.gemm(x, wu, bu)

                else:

                    def run_gemm():
                        last[cid] = _UNFUSED_ACTIVATION[act](oasr.gemm(x, wg, bg)) * oasr.gemm(
                            x, wu, bu
                        )

                return run_gemm
            return lambda: oasr.gated_mlp(
                x, wg, wu, bg, bu, activation=act, out=out, _lane_route=route
            )

        return factory

    calls: Dict[str, Callable[[int], Callable[[], Any]]] = {}
    for cid in cands:
        try:
            calls[cid] = make(cid)
        except Exception as exc:  # noqa: BLE001 -- a lane that cannot build is not an arm
            logger.info("[build] %s: %s", cid, exc)
    r = min(rows, _REFERENCE_ROWS)
    wg0, wu0 = ws[0]
    gate = x[:r].float() @ wg0.float().t()
    up = x[:r].float() @ wu0.float().t()
    if bias is not None:
        gate, up = gate + bias[0].float(), up + bias[1].float()
    ref = _UNFUSED_ACTIVATION[act](gate) * up

    def check(cid: str) -> float:
        calls[cid](0)()
        torch.cuda.synchronize()
        got = last[cid] if cid in last else out
        return bench.numerics_error(got[:r], ref)

    case = _measure_arms(
        calls,
        routed,
        check,
        lambda errs: errs.get(routed) if routed in errs else errs.get("gemm"),
        dt,
        policy,
        copies,
        shape,
    )
    for name, m in case.results.items():
        if name in cands:
            m.payload = cands[name]
    return case


# =============================================================================
# The build
# =============================================================================


def _log_path(opts: BuildOptions, sm: int, family: str) -> Optional[str]:
    if opts.log_path:
        pth = Path(opts.log_path)
        return str(pth.with_name(f"{pth.stem}.{family}{pth.suffix or '.jsonl'}"))
    root = database.user_root()
    return str(root / "measurements" / f"sm{sm}_{family}.jsonl") if root is not None else None


def _digest(pts: Sequence[Dict[str, Any]], opts: BuildOptions) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(sorted(json.dumps(p, sort_keys=True) for p in pts)).encode())
    knobs = [opts.min_speedup, opts.tie, opts.cover_eps, opts.partition, bench.PROTOCOL_VERSION]
    h.update(json.dumps(knobs).encode())
    return h.hexdigest()[:16]


def _build(
    family: str,
    shapes: FusedShapeSet,
    measure: Callable[[Dict[str, Any], bench.BenchPolicy], FusedCase],
    to_params: Callable[[Any], Dict[str, Any]],
    impl: str,
    *,
    sm: int,
    opts: BuildOptions,
    base: Optional[database.TuningFile],
    lane: str,
) -> database.TuningFile:
    if shapes.family != family:
        raise ValueError(f"a {shapes.family} census cannot build a {family} file")
    dyn = _DYNAMIC[family]
    tf = base if base is not None else database.new_file(family, sm, lane)
    database.stamp_current(tf)
    t0 = time.time()
    sku = database.current_soft_validators(family).get("sku")
    sigs = shapes.signatures()
    log_path = _log_path(opts, sm, family)
    log_f = open(log_path, "a") if log_path else None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(opts.checkpoint_path, sm, impl)
    ckpt_f = None
    if opts.checkpoint_path:
        Path(opts.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_f = open(opts.checkpoint_path, "a")
    contended = False
    gc.freeze()
    try:
        for i, (key, pts) in enumerate(sigs.items(), 1):
            if opts.budget_s is not None and time.time() - t0 > opts.budget_s:
                if opts.progress:
                    opts.progress(f"budget spent; {len(sigs) - i + 1} {family} signature(s) left")
                break
            digest = _digest(pts, opts)
            rec = done.get(key)
            if rec is not None and rec.get("digest") == digest:
                tf.configs.update(rec["configs"])
                tf.entries[key] = database.Entry.from_json(rec["entry"])
                contended |= bool(rec["contended"])
                continue
            results: List[PointResult] = []
            for p in pts:
                case = measure(p, opts.policy)
                choice, sp, note = decide_point(case, "auto", 0.0, opts)  # type: ignore[arg-type]
                results.append(
                    PointResult(int(p[dyn]), False, 0.0, case, choice, sp, note, float(p.get("weight", 1.0)))  # type: ignore[arg-type]
                )
                if opts.progress:
                    opts.progress(f"{key} {dyn}={p[dyn]}: {choice} ({note})")
            sig_contended = any(r.case.conditions.contended for r in results)
            contended |= sig_contended
            if log_f is not None:
                for pr in results:
                    for name, m in pr.case.results.items():
                        if name == "auto" or m.status != "ok" or not math.isfinite(m.median_ms):
                            continue
                        row = {
                            "family": family,
                            "sm": sm,
                            "sku": sku,
                            "impl_hash": impl,
                            "protocol": bench.PROTOCOL_VERSION,
                            "sig": key,
                            dyn: pr.M,
                            "config": name,
                            "median_ms": round(m.median_ms, 7),
                            "sigma_ms": round(m.sigma_ms, 7),
                            "solo_ms": None if m.solo_ms is None else round(m.solo_ms, 7),
                            "n": m.n,
                            "stage": m.eliminated_at,
                            "contended": pr.case.conditions.contended,
                        }
                        log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if opts.partition == "cover":
                regions, notes = partition.regions_for_signature(
                    results, "auto", opts.cover_eps, min_speedup=opts.min_speedup
                )
            else:
                regions, notes = partition.regions_from_decisions(results, "auto")
            if not regions:
                regions = [(None, "auto")]
            payloads: Dict[str, Any] = {"auto": "auto"}
            for pr in results:
                for n_, m in pr.case.results.items():
                    payloads.setdefault(n_, m.payload)
            configs = {cid: to_params(payloads[cid]) for _, cid in regions}
            tf.configs.update(configs)
            entry = database.Entry(
                regions=regions,
                evidence={f"{dyn}={pr.M}": _evidence(pr, "auto") for pr in results},
                source="aot",
                notes=notes + [f"{dyn}={pr.M}: {pr.choice} ({pr.note})" for pr in results],
            )
            tf.entries[key] = entry
            if ckpt_f is not None:
                _append_checkpoint(
                    ckpt_f,
                    {
                        "family": family,
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
    referenced = set(tf.referenced_configs())
    tf.configs = {k: v for k, v in tf.configs.items() if k in referenced}
    tf.provenance.update(
        {
            "tool": "oasr tune build",
            "bench_protocol": bench.PROTOCOL_VERSION,
            "census": shapes.provenance,
            "contended": contended,
            "elapsed_s": round(time.time() - t0, 1),
        }
    )
    return tf


def build_recurrent(
    shapes: FusedShapeSet,
    *,
    sm: Optional[int] = None,
    opts: Optional[BuildOptions] = None,
    base: Optional[database.TuningFile] = None,
) -> database.TuningFile:
    """The ``recurrent`` tuning file for a recurrent census."""
    from oasr.jit import recurrent_select as R
    from oasr.jit.selection import _target_sm

    opts = opts or BuildOptions()
    sm_ = int(sm if sm is not None else _target_sm())
    tf = _build(
        "recurrent",
        shapes,
        _recurrent_case,
        R.choice_to_params,
        R.impl_hash(),
        sm=sm_,
        opts=opts,
        base=base,
        lane="recurrent",
    )
    tf.provenance["fallback"] = (
        "auto: layers.recurrent._TENSOR_CORE_MIN_BATCH_*, recurrent_cute._LSTM_BANDS, "
        "the C++ lane first, kRecurrentStepRoutes, _default_recurrent_tactic"
    )
    tf.provenance["protocol_note"] = "dependent chain: call i reads the state call i-1 wrote"
    return tf


def build_gated_mlp(
    shapes: FusedShapeSet,
    *,
    sm: Optional[int] = None,
    opts: Optional[BuildOptions] = None,
    base: Optional[database.TuningFile] = None,
) -> database.TuningFile:
    """The ``gated_mlp`` tuning file for a gated-MLP census."""
    from oasr.jit import mlp_select as M
    from oasr.jit.selection import _target_sm

    opts = opts or BuildOptions()
    sm_ = int(sm if sm is not None else _target_sm())
    tf = _build(
        "gated_mlp",
        shapes,
        _gated_mlp_case,
        M.choice_to_params,
        M.impl_hash(),
        sm=sm_,
        opts=opts,
        base=base,
        lane="gated_mlp",
    )
    tf.provenance["fallback"] = (
        "auto: jit.mlp one m-tile band (rows <= 64), the C++ lane first, "
        "wave-arithmetic tile from _CANDIDATES; else two GEMMs"
    )
    return tf
