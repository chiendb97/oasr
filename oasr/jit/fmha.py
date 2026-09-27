# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The C++ CUTLASS/CuTe lane for fused multi-head attention.

Unlike :mod:`oasr.jit.attention` -- which wraps ``cutlass.cute.compile()`` for
the CuTeDSL kernel -- this is the repo's ordinary C++ JIT pipeline: Jinja
renders one translation unit per feature variant, ninja builds them into one
``.so`` per *cell*, and the result is cached on disk in ``~/.cache/oasr/jit``.

``oasr.jit.attention`` stays the arbiter that answers "which backend?"; this
module is one of the answers, and never imports it back.

Cells and variants
------------------
A **cell** is ``(target_sm, dtype, padded_head_dim)`` -- the axes that change
the shared-memory layouts, the swizzle and the MMA k-count, i.e. the ones a
single ``.so`` cannot span.  Inside a cell the **variants** are

    3 (none / causal / local) x 2 (bias) x 2 (dense / paged) = 12

and all of them go into that one module, because nvcc parallelises across
translation units but not within one (the reasoning is written out at
``oasr/jit/gemm.py:1000``).  Measured on this box: one variant compiles in
31-34 s, and 12 of them are a single ninja wave on 64 cores.  So a cell costs
about one variant's wall time, once, and then covers every shape that cell will
ever be asked for.

Everything else is deliberately **runtime**, not a compile-time axis:
``num_heads`` / ``num_kv_heads``, the page size, the window bounds, the
per-stream ``cache_seqlens`` and ``cache_seqstarts``, whether the input is
packed, and whether the bias is vectorisable.  Each of those would have
multiplied the variant count for a compare or two of saving.

The tile is not an axis at all: ``fmhaResolveTile`` is ``constexpr`` in
``include/oasr/attention/cutlass_fmha_configs.h``, so the kernel derives it and
:func:`resolve_config` below only *mirrors* it for routing.  The two are pinned
equal by ``tests/kernels/test_fmha_cpp.py``, against the C++ answer exported as
``fmha_resolved_config`` -- a mirror nobody checks is a mirror that drifts.
"""

from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import arch_facts, env
from .core import JitSpec, _get_target_sm, gen_jit_spec
from .cubin_loader import write_if_different
from .templates import render_template

__all__ = [
    "FmhaTile",
    "resolve_config",
    "config_supported",
    "gen_fmha_module",
    "gen_fmha_modules",
    "get_fmha_fn",
    "num_splits",
    "split_range",
    "fmha_coverage_report",
    "variant_symbol",
]

#: Architectures this lane compiles for.  sm_90 / sm_100 would *run* the
#: Ampere-class collective -- mma.sync and cp.async are both available there --
#: but routing to it would be a performance claim with no measurement behind
#: it, and the right answer on those parts is a wgmma/TMA mainloop of their own.
#: The omission is counted in :func:`fmha_coverage_report`, not hidden.
SUPPORTED_SM: Tuple[int, ...] = (80, 86, 89, 120)

#: Opt-in shared memory per architecture, mirrored once for every C++ CuTe
#: family in :mod:`oasr.jit.arch_facts` (``oasr::smemCapacityForSm``).
_SMEM_CAPACITY = arch_facts.SMEM_CAPACITY
_MAX_STAGES = 3
_N_BLOCK_LADDER = (64, 32, 16)
_BLOCK_M = 64
_NUM_WARPS = 4

_DTYPES = {"float16": "cutlass::half_t", "bfloat16": "cutlass::bfloat16_t"}
_MASK_MODES = ("none", "causal", "local")

#: Dense and paged KV.  `page_size` is a *runtime* value inside each variant --
#: the loader walks a partial page -- so this axis is two entries, not one per
#: page size a caller might choose.
_PAGED_MODES: Tuple[bool, ...] = (False, True)

#: Split-KV (flash-decoding) is rendered for ``mask="none"`` only -- see the
#: header comment in ``csrc/templates/fmha_split_template.cu.jinja``.  Four
#: extra translation units per cell, which keeps a cell inside one ninja wave.
_SPLIT_MASK_MODES = ("none",)

#: Mirrors ``oasr::attention::kFmhaMaxSplits`` / ``kFmhaMinBlocksPerSplit``.
MAX_SPLITS = 16
MIN_BLOCKS_PER_SPLIT = 2
#: Mirrors ``kFmhaMinBlocksForSplit`` / ``kFmhaSplitMaxUtilPct`` /
#: ``kFmhaSplitDepthPerCta`` -- all *measured*, with the tables that produced
#: them in the C++ header.  They were re-derived after the combine pass was
#: rewritten: the old floor of 32 was measuring a combine that ran at 0.02
#: waves, not an intrinsic cost of splitting.
MIN_BLOCKS_FOR_SPLIT = 6
SPLIT_MAX_UTIL_PCT = 60
SPLIT_DEPTH_PER_CTA = 30

_ARCH_UNDERSERVED: "Counter[int]" = Counter()
_REFUSED: "Counter[str]" = Counter()


@dataclass(frozen=True)
class FmhaTile:
    """What ``fmhaResolveTile`` chose, or ``valid=False`` if nothing fits."""

    valid: bool
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    q_in_regs: bool
    smem_bytes: int


def padded_head_dim(head_dim: int) -> int:
    """Head dim rounded to the m16n8k16 k-stride, as the layouts allocate it."""
    return (head_dim + 31) // 32 * 32


#: Mirrors ``oasr::attention::fmhaSmemBudget`` (``oasr::smemBudgetForSm``).
smem_budget = arch_facts.smem_budget


def _smem_bytes(block_m: int, block_n: int, d_q: int, d_v: int, stages: int, elem: int) -> int:
    q = block_m * d_q * elem
    k = block_n * d_q * stages * elem
    v = block_n * d_v * stages * elem
    o = block_m * d_v * elem
    mainloop = q + k + v
    pad = max(0, o - v - k)
    return max(mainloop + pad, o)


@functools.lru_cache(maxsize=None)
def resolve_config(sm: int, elem_bits: int, head_dim: int) -> FmhaTile:
    """Mirror of ``oasr::attention::fmhaResolveTile``.

    Answers for *any* architecture from values alone -- no device, no compile.
    Pinned field-for-field against the C++ original by
    ``tests/kernels/test_fmha_cpp.py::TestTileSelectionAgrees``.
    """
    budget = smem_budget(sm)
    elem = elem_bits // 8
    d = padded_head_dim(head_dim)
    if budget <= 0 or head_dim <= 0 or elem <= 0:
        return FmhaTile(False, 0, 0, 0, 0, False, 0)
    for block_n in _N_BLOCK_LADDER:
        for stages in range(_MAX_STAGES, 0, -1):
            need = _smem_bytes(_BLOCK_M, block_n, d, d, stages, elem)
            if need <= budget:
                return FmhaTile(True, _BLOCK_M, block_n, _NUM_WARPS, stages, False, need)
    return FmhaTile(False, 0, 0, 0, 0, False, 0)


def config_supported(
    *,
    sm: int,
    dtype_str: str,
    head_dim: int,
    paged: bool = False,
    causal: bool = False,
    local: bool = False,
) -> bool:
    """Can this lane serve the shape?  Asked before anything is built.

    Two questions, and either "no" is a "no": does a tile fit the architecture's
    shared memory, and is the variant one this lane actually renders?  Answering
    only the first would route a caller to a symbol that does not exist.
    """
    if dtype_str not in _DTYPES:
        return False
    if sm not in SUPPORTED_SM:
        if sm in _SMEM_CAPACITY:
            _ARCH_UNDERSERVED[sm] += 1
        return False
    if head_dim % 8 != 0:
        _REFUSED["head_dim not a multiple of the 128-bit load width"] += 1
        return False
    if causal and local:
        return False
    if paged not in _PAGED_MODES:
        _REFUSED["paged KV (phase 2)"] += 1
        return False
    tile = resolve_config(sm, 16, head_dim)
    if not tile.valid:
        _REFUSED[f"head_dim {head_dim} overflows sm_{sm}'s shared memory"] += 1
        return False
    return True


_ceildiv = arch_facts.ceil_div  # mirrors ``oasr::tileCeilDiv``


def _split_eligible(s: int, n_blocks: int) -> bool:
    """Mirror of ``oasr::attention::fmhaSplitEligible``.

    With 16 K tiles a 5-way split hands out ``ceil(16/5) = 4`` tiles just as a
    4-way one does, so the fifth CTA set removes no work from the critical path
    and is pure combine cost.
    """
    return s <= 1 or _ceildiv(n_blocks, s) != _ceildiv(n_blocks, s - 1)


def num_splits(cta_count: int, n_blocks: int, num_sms: int) -> int:
    """Mirror of ``oasr::attention::fmhaNumSplits``.

    A pure function of ``(shape, SM count)`` and **nothing else** -- in
    particular not of CUDA-graph capture state (``AGENTS.md`` rule 11).
    Splitting changes the order the fp32 partials are summed, so an answer that
    differed under capture would make a replayed graph produce different
    numbers than eager, and a one-ulp difference in attention has changed a
    decoded token in this repo before.

    Pinned against the C++ original by ``tests/kernels/test_fmha_cpp.py``.
    """
    if cta_count <= 0 or n_blocks <= 0 or num_sms <= 0:
        return 1
    if n_blocks < MIN_BLOCKS_FOR_SPLIT:
        return 1
    # Utilisation of the unsplit grid, ``waves / ceil(waves)``.  Not raw CTA
    # count: on 170 SMs, 128 CTAs is one 75 %-full wave and splitting loses,
    # while 192 CTAs is a full wave plus a 13 %-full one and splitting wins.
    if 100 * cta_count > SPLIT_MAX_UTIL_PCT * num_sms * _ceildiv(cta_count, num_sms):
        return 1
    if n_blocks * num_sms < SPLIT_DEPTH_PER_CTA * cta_count:
        return 1
    hi = min(MAX_SPLITS, num_sms, n_blocks // MIN_BLOCKS_PER_SPLIT)
    if hi < 2:
        return 1

    # Integer arithmetic, matching the C++ original exactly: the efficiency
    # test is a ratio, and evaluating it in float here and in double there can
    # land the two on opposite sides of a tie.
    def _ab(s: int) -> Tuple[int, int]:
        a = cta_count * s
        return a, num_sms * _ceildiv(a, num_sms)

    am, bm = _ab(1)
    for s in range(2, hi + 1):
        if not _split_eligible(s, n_blocks):
            continue
        a, b = _ab(s)
        if a * bm > am * b:
            am, bm = a, b
    for s in range(1, hi + 1):
        if not _split_eligible(s, n_blocks):
            continue
        a, b = _ab(s)
        if 100 * a * bm >= 85 * am * b:
            return s
    return 1


def split_range(n_block_min: int, n_block_max: int, split_idx: int, splits: int):
    """Mirror of ``oasr::attention::fmhaSplitRange``."""
    total = n_block_max - n_block_min
    if total <= 0 or splits <= 1:
        return (n_block_min, n_block_max)
    per = -(-total // splits)
    lo = min(n_block_min + split_idx * per, n_block_max)
    return (lo, min(lo + per, n_block_max))


def _mask_mode(causal: bool, local: bool) -> str:
    if local:
        return "local"
    return "causal" if causal else "none"


def variant_symbol(
    sm: int,
    dtype_str: str,
    head_dim: int,
    mask: str,
    has_bias: bool,
    paged: bool,
    varlen: bool = False,
    split: bool = False,
) -> str:
    """The exported name of one variant.

    Produced by exactly one function so the renderer and the loader cannot
    disagree about it.  ``varlen`` is a *suffix*, not a variant axis: packed and
    dense inputs run the same instantiation and differ only in the arguments
    the launcher builds, so the cell cost is unchanged.
    """
    d = padded_head_dim(head_dim)
    sym = (
        f"fmha_sm{sm}_{dtype_str}_d{d}_{mask}"
        f"_{'bias' if has_bias else 'nobias'}_{'paged' if paged else 'dense'}"
    )
    if split:
        return f"{sym}_split"
    return f"{sym}_varlen" if varlen else sym


def cell_name(sm: int, dtype_str: str, head_dim: int) -> str:
    return f"fmha_sm{sm}_{dtype_str}_d{padded_head_dim(head_dim)}"


def _variants() -> List[Tuple[str, bool, bool]]:
    return [
        (mask, has_bias, paged)
        for mask in _MASK_MODES
        for has_bias in (False, True)
        for paged in _PAGED_MODES
    ]


def _split_variants() -> List[Tuple[str, bool, bool]]:
    return [
        (mask, has_bias, paged)
        for mask in _SPLIT_MASK_MODES
        for has_bias in (False, True)
        for paged in _PAGED_MODES
    ]


@functools.lru_cache(maxsize=None)
def gen_fmha_module(sm: int, dtype_str: str, head_dim: int) -> JitSpec:
    """Render and describe one cell's translation units."""
    if dtype_str not in _DTYPES:
        raise ValueError(f"fmha: unsupported dtype {dtype_str!r}")
    d = padded_head_dim(head_dim)
    tile = resolve_config(sm, 16, head_dim)
    if not tile.valid:
        raise RuntimeError(
            f"fmha: no (K tile, ring depth) fits sm_{sm}'s {smem_budget(sm)} B of shared "
            f"memory at head_dim {head_dim} (padded {d})"
        )
    name = cell_name(sm, dtype_str, head_dim)
    out_dir = env.OASR_GEN_SRC_DIR / "fmha"
    sources: List = []
    for mask, has_bias, paged in _variants():
        func = variant_symbol(sm, dtype_str, head_dim, mask, has_bias, paged)
        rendered = render_template(
            "fmha_template.cu.jinja",
            func_name=func,
            sm_version=sm,
            dtype=dtype_str,
            cutlass_dtype=_DTYPES[dtype_str],
            head_dim=d,
            is_causal=(mask == "causal"),
            is_local=(mask == "local"),
            has_bias=has_bias,
            paged_kv=paged,
        )
        path = out_dir / f"{func}.cu"
        write_if_different(path, rendered)
        sources.append(path)
    for mask, has_bias, paged in _split_variants():
        func = variant_symbol(sm, dtype_str, head_dim, mask, has_bias, paged, split=True)
        rendered = render_template(
            "fmha_split_template.cu.jinja",
            func_name=func,
            sm_version=sm,
            dtype=dtype_str,
            cutlass_dtype=_DTYPES[dtype_str],
            head_dim=d,
            has_bias=has_bias,
            paged_kv=paged,
        )
        path = out_dir / f"{func}.cu"
        write_if_different(path, rendered)
        sources.append(path)
    sources.append(env.OASR_CSRC_DIR / "fmha_jit_binding.cu")
    return gen_jit_spec(name, sources)


def gen_fmha_modules(cells: Optional[List[Tuple[str, int]]] = None) -> List[JitSpec]:
    """Specs for the AOT set: the ``(dtype, head_dim)`` cells shipped models reach.

    head_dim 64 covers Conformer, Zipformer's attention consumers and Whisper;
    128 covers Paraformer's SANM and the Qwen2-Audio decoder.
    """
    if cells is None:
        cells = [(dt, d) for dt in ("float16", "bfloat16") for d in (64, 128)]
    sm = _get_target_sm()
    if sm not in SUPPORTED_SM:
        return []
    return [gen_fmha_module(sm, dt, d) for dt, d in cells]


@functools.lru_cache(maxsize=None)
def _cell_module(sm: int, dtype_str: str, head_dim: int):
    return gen_fmha_module(sm, dtype_str, head_dim).build_and_load()


@functools.lru_cache(maxsize=None)
def _cell_available(sm: int, dtype_str: str, head_dim: int) -> bool:
    """Did this cell build?

    Separate from :func:`_cell_module` because ``functools.lru_cache`` does not
    memoise exceptions -- a failing build behind a cached getter would re-invoke
    ninja on *every* call.
    """
    try:
        _cell_module(sm, dtype_str, head_dim)
        return True
    except Exception:  # noqa: BLE001 -- the caller decides whether to fall back
        _REFUSED[f"cell {cell_name(sm, dtype_str, head_dim)} failed to build"] += 1
        return False


def get_fmha_fn(
    *,
    dtype_str: str,
    head_dim: int,
    causal: bool = False,
    local: bool = False,
    has_bias: bool = False,
    paged: bool = False,
    varlen: bool = False,
    split: bool = False,
    sm: Optional[int] = None,
):
    """The bound launcher for one variant, building its cell on first use."""
    if varlen and paged:
        raise ValueError("fmha: packed inputs and a paged KV pool are mutually exclusive")
    if split and (varlen or causal or local):
        raise ValueError(
            "fmha: split-KV is rendered for unmasked dense/paged attention only "
            "(a top-left causal or windowed decode step has no long K range to split)"
        )
    sm = _get_target_sm() if sm is None else sm
    mask = _mask_mode(causal, local)
    module = _cell_module(sm, dtype_str, head_dim)
    return getattr(
        module, variant_symbol(sm, dtype_str, head_dim, mask, has_bias, paged, varlen, split)
    )


def fmha_coverage_report() -> List[str]:
    """What this lane declined, and why.

    Neither a missing kernel (``KERNEL_GAPS``) nor a measured performance choice
    (``take_policy``), so it needs its own channel -- the same shape
    ``oasr.jit.gemm.rule_miss_report`` and ``oasr.jit.measured.extrapolations``
    already use.  An architecture that silently gets SDPA is worth a line.
    """
    lines: List[str] = []
    if _ARCH_UNDERSERVED:
        lines.append("fused attention has no C++ kernel for these architectures:")
        for sm, hits in sorted(_ARCH_UNDERSERVED.items()):
            lines.append(
                f"  sm_{sm}: {hits} request(s) -> SDPA. The Ampere-class collective would "
                f"run there; it is not routed to because nothing has measured it against "
                f"SDPA on that part."
            )
    if _REFUSED:
        lines.append("fused attention configs refused by the C++ lane:")
        for why, hits in sorted(_REFUSED.items()):
            lines.append(f"  {hits}x {why}")
    return lines


def reset_coverage() -> None:
    _ARCH_UNDERSERVED.clear()
    _REFUSED.clear()


def clear_caches() -> None:
    """Drop every memoised answer.  Called by ``set_backend_mode``."""
    resolve_config.cache_clear()
    gen_fmha_module.cache_clear()
    _cell_module.cache_clear()
    _cell_available.cache_clear()
