# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the int8 scaled GEMM (``oasr.scaled_mm``).

``D = act(a_scale[m] * b_scale[n] * (A_int8 @ B_int8ᵀ)[m, n] + bias[n])``: the
W8A8 GEMM of a quantized layer, with dequantization, bias and activation in one
CUTLASS 2.x epilogue visitor tree (``include/oasr/scaled_mm/``).

A family of its own rather than more ``gemm`` variants: ``gemm`` is fp16/bf16
throughout (instruction shape ``16x8x16``, float accumulator, half-only
dispatch, ``dt=half`` rule tables), and int8 variants there would grow every
user's cold build.  The config-space helpers are shared, at 1-byte operands.

One lane today — the Sm80 ``mma.sync`` int8 kernel — serving every SM >= 80:
natively on sm80/86/89/120 and as the mixed lane on sm90/sm100, the way the fp16
GEMM already runs there.  Turing (sm75) is a declared kernel gap
(``scaled-mm-int8-sm75`` in :data:`oasr.layers._backend.KERNEL_GAPS`).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from . import env
from .core import JitSpec, _get_target_sm, gen_jit_spec
from .gemm import _SM_MAX_SMEM_BYTES, TileShape, _epilogue_covers_warp, _smem_bytes

logger = logging.getLogger(__name__)

#: SM families the Sm80 int8 lane runs on, and whether that lane is the
#: family's native tensor-core path (``False`` = runs, but a 3.x wgmma/tcgen05
#: int8 kernel would be faster; reported, not a gap).
SCALED_MM_LANE_BY_SM: Dict[int, bool] = {
    80: True,
    86: True,
    89: True,
    120: True,
    90: False,
    100: False,
}

#: The epilogue's thread map partitions fp32 compute fragments, four per
#: access (``CutlassScaledMmKernel::kEpilogueElementsPerAccess``), so the
#: warp-coverage rule is evaluated at 32-bit elements, not the 16-bit output.
_EPILOGUE_ELEMENT_BITS = 32

#: Operand alignment: 16-byte int8 loads on K, 16-byte output stores on N.
K_ALIGNMENT = 16
N_ALIGNMENT = 8


@dataclass(frozen=True)
class ScaledMmConfig:
    """One CUTLASS 2.x int8 tile configuration."""

    block_m: int
    block_n: int
    block_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    stages: int
    #: Stream-K decomposition: K iterations spread over every SM, int32 partials
    #: reduced through a workspace -- an exact integer sum, so the same bits as
    #: the data-parallel kernel.  Compile-time (a different kernel); the fp
    #: output only (a quantizing producer has a short K).
    stream_k: bool = False

    @property
    def compile_name(self) -> str:
        return (
            f"b{self.block_m}x{self.block_n}x{self.block_k}_"
            f"w{self.warp_m}x{self.warp_n}x{self.warp_k}_s{self.stages}"
            + ("_sk" if self.stream_k else "")
        )

    @property
    def tile(self) -> TileShape:
        return TileShape(
            self.block_m, self.block_n, self.block_k, self.warp_m, self.warp_n, self.warp_k
        )


#: The data-parallel production set, ordered by M bucket.  Small-M tiles favour a
#: deep K tile (few rows, so the mainloop is the cost); large-M tiles the 128x128
#: square.  Each one also exports the int8-output (quantizing) launcher.
_PRODUCTION: Tuple[ScaledMmConfig, ...] = (
    ScaledMmConfig(16, 64, 128, 16, 64, 64, 5),
    ScaledMmConfig(32, 64, 128, 32, 64, 64, 5),
    ScaledMmConfig(64, 64, 128, 64, 64, 64, 5),
    ScaledMmConfig(64, 128, 128, 64, 64, 64, 3),
    ScaledMmConfig(128, 128, 64, 64, 64, 64, 5),
    ScaledMmConfig(128, 256, 64, 64, 64, 64, 3),
)

#: The Stream-K variants.  Exact (the partials are int32), so a Stream-K choice
#: never changes a bit -- but it is a *measured* choice only: the shape heuristic
#: (:func:`select_config`) never picks one, a tuning entry may.  Measured on an
#: RTX 5090 against the best data-parallel tile (graph-captured, 2026-10-20):
#: 0.35-0.69x on every Conformer shape (K <= 2048 is 16 int8 K-tiles, too short
#: to amortise the barrier memset and the partial-tile fixup), 0.5-0.9x at
#: K = 4864-8192, parity at K = 16384 and 1.09-1.16x only at a 7B-size
#: down-projection (N=896, K=18944, M >= 256).  These two tiles were the best
#: Stream-K arm at small and at large M respectively; 32x64x128 and 64x128x128
#: never were.
_STREAMK_PRODUCTION: Tuple[ScaledMmConfig, ...] = (
    ScaledMmConfig(64, 64, 128, 64, 64, 64, 5, stream_k=True),
    ScaledMmConfig(128, 128, 64, 64, 64, 64, 5, stream_k=True),
)


def _buildable(cfg: ScaledMmConfig, sm: int) -> bool:
    limit = _SM_MAX_SMEM_BYTES.get(sm, 100 * 1024)
    if _smem_bytes(cfg.block_m, cfg.block_n, cfg.block_k, cfg.stages, dtype_bytes=1) > limit:
        return False
    return _epilogue_covers_warp(cfg.tile, element_bits=_EPILOGUE_ELEMENT_BITS)


def lane_available(sm: int) -> bool:
    """Whether the int8 lane serves *sm* at all."""
    return sm in SCALED_MM_LANE_BY_SM


def is_native_lane(sm: int) -> bool:
    """Whether the int8 lane is *sm*'s native tensor-core path."""
    return SCALED_MM_LANE_BY_SM.get(sm, False)


@functools.lru_cache(maxsize=None)
def _production_configs(sm: int) -> Tuple[ScaledMmConfig, ...]:
    return tuple(cfg for cfg in _PRODUCTION + _STREAMK_PRODUCTION if _buildable(cfg, sm))


def get_production_configs(sm: int | None = None) -> Dict[str, ScaledMmConfig]:
    """``{compile_name: config}`` for every production variant buildable on *sm*."""
    sm = _get_target_sm() if sm is None else sm
    return {cfg.compile_name: cfg for cfg in _production_configs(sm)}


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=4096)
def _select(m: int, n: int, sm: int, num_sms: int) -> ScaledMmConfig:
    by_area = sorted(
        (c for c in get_production_configs(sm).values() if not c.stream_k),
        key=lambda c: c.block_m * c.block_n,
        reverse=True,
    )
    for cfg in by_area:
        ctas = -(-m // cfg.block_m) * -(-n // cfg.block_n)
        if 2 * ctas >= num_sms:
            return cfg
    return by_area[-1]


@functools.lru_cache(maxsize=1)
def _process_sm() -> int:
    """The SM family this process compiles the int8 module for (resolved once).

    One family per process, as ``gen_scaled_mm_module`` renders one: re-asking
    the device on every call cost ~5.7 us, most of a selection.
    """
    return _get_target_sm()


@functools.lru_cache(maxsize=1)
def _process_num_sms() -> int:
    """The SM count the heuristic sizes its wave against when the caller names none."""
    import torch

    return _num_sms(torch.cuda.current_device())


def heuristic_config(
    m: int, n: int, sm: int | None = None, num_sms: int | None = None
) -> ScaledMmConfig:
    """The shape heuristic's pick, ignoring the tuning DB (what a build measures against)."""
    sm = _process_sm() if sm is None else int(sm)
    num_sms = _process_num_sms() if num_sms is None else int(num_sms)
    return _select(int(m), int(n), sm, num_sms)


def select_config(
    m: int, n: int, k: int, sm: int | None = None, num_sms: int | None = None
) -> ScaledMmConfig:
    """The production variant for a ``[m, k] @ [n, k]ᵀ`` problem.

    A measured tuning entry first (``scaled_mm|op=scaled_mm|N=..|K=..``, user tier
    then system tier, M rounded *up* to a region; see :func:`covering_tier`), else
    the shape heuristic: the largest data-parallel tile that still launches at
    least half a wave of CTAs, else the smallest tile -- never Stream-K (see
    ``_STREAMK_PRODUCTION``), which only a measured entry can name.  Measured on
    the Conformer shapes on an RTX 5090 (170 SMs; N, K in {(768, 256), (256,
    256), (2048, 256), (256, 2048), (512, 256)}, M from 16 to 16384) the
    heuristic picks the fastest variant, or one within 3 % of it, at every
    point; the per-M bucket rule it replaced was up to 3x off at mid M.

    A function of shape, device and the tuning-DB snapshot alone — never of
    capture state (AGENTS.md rule 11) — so an eager call and a graph replay of a
    shape run one kernel.  Memoised per ``(epoch, shape)``: the per-call cost is
    one dict lookup.
    """
    key = (_epoch(), "scaled_mm", m, n, k, sm, num_sms)
    cfg = _MEMO.get(key)
    if cfg is None:
        cfg = _memoise(key, "scaled_mm", m, n, k, sm, num_sms)
    return cfg


def select_quant_config(
    m: int, n: int, k: int, sm: int | None = None, num_sms: int | None = None
) -> ScaledMmConfig:
    """The variant for a quantizing (int8-output) launch: data-parallel only.

    The producer of a GEMM chain (the FFN's ``w_1``) has a short K and a wide N,
    which is the data-parallel kernel's home; Stream-K is not built for it, so a
    tuning entry that names a Stream-K tile for ``op=scaled_mm_quant`` is warned
    about once and skipped.  Otherwise as :func:`select_config`, under its own
    signature key (the quantizing epilogue is measured separately).
    """
    key = (_epoch(), "scaled_mm_quant", m, n, k, sm, num_sms)
    cfg = _MEMO.get(key)
    if cfg is None:
        cfg = _memoise(key, "scaled_mm_quant", m, n, k, sm, num_sms)
    return cfg


def scaled_mm_func_name(cfg: ScaledMmConfig) -> str:
    """The TVM-FFI export name of a variant's fp16/bf16-output launcher."""
    return f"scaled_mm_{cfg.compile_name}"


def scaled_mm_quant_func_name(cfg: ScaledMmConfig) -> str:
    """The TVM-FFI export name of a data-parallel variant's int8-output launcher."""
    if cfg.stream_k:
        raise ValueError(f"{cfg.compile_name}: Stream-K variants have no quantizing launcher")
    return f"scaled_mm_q8_{cfg.compile_name}"


#: The MMA lane a tuning file for this family is measured on.
SCALED_MM_LANE = "sm80_int8_mma"


def config_id(cfg: ScaledMmConfig) -> str:
    """The id a tuning file stores a config under (its compile name)."""
    return cfg.compile_name


def config_to_params(cfg: ScaledMmConfig) -> Dict[str, object]:
    """A config as the explicit parameter dict a tuning file stores."""
    return {
        "kind": "cutlass",
        "lane": SCALED_MM_LANE,
        "tile": [cfg.block_m, cfg.block_n, cfg.block_k],
        "warp": [cfg.warp_m, cfg.warp_n, cfg.warp_k],
        "stages": cfg.stages,
        "stream_k": bool(cfg.stream_k),
    }


def config_from_params(params: Dict[str, object]) -> ScaledMmConfig:
    """Inverse of :func:`config_to_params`."""
    if params.get("kind") != "cutlass" or params.get("lane") != SCALED_MM_LANE:
        raise ValueError(f"not a {SCALED_MM_LANE} scaled_mm config: {params}")
    tile = [int(v) for v in params["tile"]]  # type: ignore[attr-defined]
    warp = [int(v) for v in params["warp"]]  # type: ignore[attr-defined]
    return ScaledMmConfig(
        block_m=tile[0],
        block_n=tile[1],
        block_k=tile[2],
        warp_m=warp[0],
        warp_n=warp[1],
        warp_k=warp[2],
        stages=int(params["stages"]),  # type: ignore[call-overload]
        stream_k=bool(params.get("stream_k", False)),
    )


# =============================================================================
# The tuning-DB tier (oasr.tune.database): measured selections, then the heuristic
# =============================================================================
#
# ``oasr/tune/db/sm<family>/scaled_mm.json`` (system) and the user tier's file of
# the same name hold measured selections keyed ``scaled_mm|op=<op>|N=<n>|K=<k>``,
# with M as the regions' dynamic axis, rounded *up* exactly as the GEMM family's
# are.  Every variant computes the same bits (an exact int32 accumulator; Stream-K
# reduces int32 partials), so an entry can only change speed, never a value.
#
# A region names a production config, or the ``"heuristic"`` sentinel: "measured,
# and the shape heuristic's own pick was the best within the noise here".  The
# build writes the sentinel rather than freezing the tile the heuristic happened
# to pick on the measuring card, so the entry follows the heuristic (and its SM
# count) wherever it is the right answer, and a signature where it always was is
# still *covered* -- what ``tuning.strict`` and ``ASREngine.tuning_report`` ask.

#: The ops a ``scaled_mm`` tuning file keys: the fp-output GEMM and its
#: int8-output (quantizing) twin -- different epilogues, measured separately.
SCALED_MM_OPS = ("scaled_mm", "scaled_mm_quant")

#: The config id (and ``{"kind": ...}``) of the "use the shape heuristic" sentinel.
HEURISTIC_ID = "heuristic"


def scaled_mm_sig_key(op: str, n: int, k: int) -> str:
    """The tuning-DB key of a ``scaled_mm`` static signature."""
    from oasr.tune.database import sig_key

    return sig_key("scaled_mm", op=op, N=int(n), K=int(k))


def choice_to_params(choice) -> Dict[str, object]:
    """A choice (a config or :data:`HEURISTIC_ID`) as a tuning file's parameter dict."""
    if choice == HEURISTIC_ID:
        return {"kind": HEURISTIC_ID}
    return config_to_params(choice)


def choice_from_params(params: Dict[str, object]):
    """Inverse of :func:`choice_to_params`: a :class:`ScaledMmConfig` or :data:`HEURISTIC_ID`."""
    if params.get("kind") == HEURISTIC_ID:
        return HEURISTIC_ID
    return config_from_params(params)


def scaled_mm_impl_hash() -> str:
    """Identity of the int8 GEMM implementation a tuning result measured.

    The template every variant is rendered from plus the family's headers.  A
    soft validator (``oasr.tune.database``): a change marks a tuning file stale
    -- re-measure -- without discarding it.
    """
    from oasr.tune.database import hash_paths

    paths = [env.OASR_TEMPLATE_DIR / "scaled_mm_cutlass_template.cu.jinja"]
    paths += sorted((env.OASR_INCLUDE_DIR / "oasr" / "scaled_mm").glob("*.h"))
    return hash_paths(paths)


def _scaled_mm_soft_validators() -> Dict[str, object]:
    return {"impl_hash": scaled_mm_impl_hash()}


#: ``(region m_hi | None, config id, decoded choice | None)``; ``None`` = undecodable.
_Rule = Tuple[Optional[int], str, object]

#: ``(epoch, sm) -> [(tier, {(op, N, K): [rule, ...]}), ...]``, highest tier first.
_TIER_VIEWS: Dict[Tuple[int, int], List[Tuple[str, Dict[Tuple[str, int, int], List[_Rule]]]]] = {}

#: ``(epoch, op, m, n, k, sm arg, num_sms arg) -> config``: the per-call path.
_MEMO: Dict[tuple, ScaledMmConfig] = {}

#: A long-form or varlen workload has many distinct M; rebuilding one entry is
#: microseconds, so dropping the whole table on overflow beats tracking recency.
_MEMO_MAX = 8192

#: ``(tier, op, N, K, config id, sm)`` already warned about (unusable entry).
_UNUSABLE_WARNED: set = set()


def _decode(params: Optional[Dict[str, object]]):
    if params is None:
        return None
    try:
        return choice_from_params(params)
    except (KeyError, TypeError, ValueError):
        return None


def _table_from_file(tf) -> Dict[Tuple[str, int, int], List[_Rule]]:
    from oasr.tune.database import parse_sig_key

    table: Dict[Tuple[str, int, int], List[_Rule]] = {}
    decoded: Dict[str, object] = {}
    for key, entry in tf.entries.items():
        family, fields = parse_sig_key(key)
        op = fields.get("op")
        if family != "scaled_mm" or op not in SCALED_MM_OPS or not entry.regions:
            continue
        try:
            n, k = int(fields["N"]), int(fields["K"])
        except (KeyError, ValueError):
            continue
        rules: List[_Rule] = []
        for m_hi, cid in entry.regions:
            if cid not in decoded:
                decoded[cid] = _decode(tf.configs.get(cid))
            rules.append((m_hi, cid, decoded[cid]))
        table[(str(op), n, k)] = rules
    return table


def _tier_views(sm: int) -> List[Tuple[str, Dict[Tuple[str, int, int], List[_Rule]]]]:
    """The decoded user and system tables for arch family *sm*, this epoch."""
    from oasr.tune import database

    key = (database.epoch(), int(sm))
    views = _TIER_VIEWS.get(key)
    if views is not None:
        return views
    views = [(tier, _table_from_file(tf)) for tier, tf in database.tiers("scaled_mm", sm).ordered()]
    _TIER_VIEWS.clear()  # only the current epoch is ever consulted
    _TIER_VIEWS[key] = views
    return views


def _region(rules: List[_Rule], m: int) -> Optional[_Rule]:
    for rule in rules:
        if rule[0] is None or m <= rule[0]:
            return rule
    return None


def _unusable_reason(op: str, choice, sm: int) -> Optional[str]:
    """Why *choice* cannot be dispatched for *op* on *sm* (``None``: it can)."""
    if choice == HEURISTIC_ID:
        return None
    if not isinstance(choice, ScaledMmConfig):
        return "cannot be decoded as a scaled_mm config"
    if get_production_configs(sm).get(choice.compile_name) != choice:
        return f"is not in sm{sm}'s production set"
    if op == "scaled_mm_quant" and choice.stream_k:
        return "is Stream-K, which the quantizing launcher does not build"
    return None


def _note_unusable(tier: str, op: str, n: int, k: int, cid: str, reason: str, sm: int) -> None:
    """An entry named a config this process cannot serve: say so once, then fall through.

    Serving it would raise ``AttributeError: Module has no function`` at
    dispatch; skipping it silently would hide a tuning file that no longer
    matches the build.
    """
    wkey = (tier, op, n, k, cid, sm)
    if wkey in _UNUSABLE_WARNED:
        return
    _UNUSABLE_WARNED.add(wkey)
    logger.warning(
        "scaled_mm %s-tier entry for %s N=%d K=%d names %s, which %s; falling through",
        tier,
        op,
        n,
        k,
        cid,
        reason,
    )


def _lookup(op: str, m: int, n: int, k: int, sm: int):
    """``(tier, choice)`` of the first tier whose entry serves this shape, or ``None``."""
    for tier, table in _tier_views(sm):
        rules = table.get((op, n, k))
        if not rules:
            continue
        rule = _region(rules, m)
        if rule is None:  # above every bounded region: never measured there
            continue
        _, cid, choice = rule
        reason = _unusable_reason(op, choice, sm)
        if reason is None:
            return tier, choice
        _note_unusable(tier, op, n, k, cid, reason, sm)
    return None


def _resolve(op: str, m: int, n: int, k: int, sm: int, num_sms: int) -> ScaledMmConfig:
    from oasr.tune.database import record_tier

    hit = _lookup(op, m, n, k, sm)
    if hit is not None:
        tier, choice = hit
        record_tier("scaled_mm", op, tier)
        if isinstance(choice, ScaledMmConfig):
            return choice
        return _select(m, n, sm, num_sms)  # the heuristic sentinel
    record_tier("scaled_mm", op, "heuristic")
    return _select(m, n, sm, num_sms)


def _memoise(key: tuple, op: str, m, n, k, sm, num_sms) -> ScaledMmConfig:
    sm_ = _process_sm() if sm is None else int(sm)
    num_sms_ = _process_num_sms() if num_sms is None else int(num_sms)
    cfg = _resolve(op, int(m), int(n), int(k), sm_, num_sms_)
    if len(_MEMO) >= _MEMO_MAX:
        _MEMO.clear()
    _MEMO[key] = cfg  # keyed by the epoch read *before* resolving: a racing reload strands it
    return cfg


def covering_tier(op: str, M: int, N: int, K: int, sm: int | None = None) -> Optional[str]:
    """The tuning tier (``"user"`` / ``"system"``) that serves this shape, or ``None``.

    Covered means an entry exists for ``(op, N, K)``, one of its regions contains
    M, and the config it names can be dispatched here (a production config, or
    the heuristic sentinel) -- exactly what :func:`select_config` /
    :func:`select_quant_config` would serve from measured data.
    """
    if op not in SCALED_MM_OPS:
        return None
    sm = _process_sm() if sm is None else int(sm)
    hit = _lookup(op, int(M), int(N), int(K), sm)
    return None if hit is None else hit[0]


def signature_tier(op: str, N: int, K: int, sm: int | None = None) -> Optional[str]:
    """The highest tier with a servable entry for ``(op, N, K)`` at *some* M, or ``None``.

    What a per-layer report asks: a layer has a signature but no single M.
    """
    if op not in SCALED_MM_OPS:
        return None
    sm = _process_sm() if sm is None else int(sm)
    for tier, table in _tier_views(sm):
        rules = table.get((op, int(N), int(K)))
        if rules and any(_unusable_reason(op, choice, sm) is None for _, _, choice in rules):
            return tier
    return None


def _on_tuning_reload() -> None:
    _TIER_VIEWS.clear()
    _MEMO.clear()


def _register_with_tuning_db() -> Callable[[], int]:
    from oasr.tune import database

    database.register_soft_validator("scaled_mm", _scaled_mm_soft_validators)
    database.register_reload_hook(_on_tuning_reload)
    return database.epoch


#: ``oasr.tune.database.epoch``, bound once: the per-call path runs no import statement.
_epoch: Callable[[], int] = _register_with_tuning_db()


def _render_variants(configs: Dict[str, ScaledMmConfig]) -> List:
    from .cubin_loader import write_if_different
    from .templates import render_template

    sm = _get_target_sm()
    paths = []
    for name, cfg in configs.items():
        variant = f"scaled_mm_sm{sm}_{name}"
        rendered = render_template(
            "scaled_mm_cutlass_template.cu.jinja",
            op_name=variant,
            func_name=scaled_mm_func_name(cfg),
            tile_m=cfg.block_m,
            tile_n=cfg.block_n,
            tile_k=cfg.block_k,
            warp_m=cfg.warp_m,
            warp_n=cfg.warp_n,
            warp_k=cfg.warp_k,
            stages=cfg.stages,
            stream_k=cfg.stream_k,
            quant_func_name=None if cfg.stream_k else scaled_mm_quant_func_name(cfg),
            sm_version=sm,
        )
        path = env.OASR_GEN_SRC_DIR / "scaled_mm" / f"{variant}.cu"
        write_if_different(path, rendered)
        paths.append(path)
    return paths


def gen_scaled_mm_module() -> JitSpec:
    """JIT spec for every production int8 scaled-GEMM variant, one TU each."""
    return gen_jit_spec("scaled_mm", _render_variants(get_production_configs()))
