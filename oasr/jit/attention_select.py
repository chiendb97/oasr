# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Measured selection between the attention lanes: the ``fmha`` tuning family.

``OASR_ATTN_BACKEND=auto`` picks one attention lane per process -- the CuTeDSL
kernel wherever it is built, the C++ one only when the CuTeDSL kernel cannot
serve a call -- and runs it with one tile.  Neither is right everywhere.  On an
RTX 5090 the C++ lane's split-KV wins a small-batch streaming chunk and a
cross-attention decode step by 2x, the CuTeDSL kernel wins the offline encoder,
and *inside* the CuTeDSL lane a 64x32 tile is 1.7x the default 64x64 on the
offline encoder while a 32x32 one is 1.9x on a streaming chunk.  So the choice
is per shape, and it is a joint one over ``(lane, tile)``.

This module is the ``fmha`` family of the tuning database
(:mod:`oasr.tune.database`): a region of a tuning file names an
:class:`AttnChoice` -- a lane plus its configuration -- or :data:`AUTO_ID`,
"measured; today's routing is the best here".  ``oasr tune build --family
fmha`` writes the file from an attention census, and
:func:`oasr.functionals.attention.fmha` consults it through
:func:`tuned_choice`.

What is consulted, and what is not
----------------------------------
* Only when nobody named a lane: ``OASR_ATTN_BACKEND=auto`` and no
  ``backend=`` argument.  Naming one means requiring it.
* Only for fp16 / bf16, dense or paged :func:`~oasr.functionals.attention.fmha`
  calls.  ``fmha_varlen`` and the fp8 KV cache keep their own routing.
* With no ``fmha`` tuning file for this architecture the lookup is one global
  read and changes nothing.

The signature is everything that decides which kernel can run and how it
behaves -- dtype class, head dim, heads, KV heads, mask kind, bias and whether
its rows are 4-byte aligned, dense or paged (and the page size), left padding,
the query-length class, the batch rounded up to a power of two -- and the
regions run over the key length ``T_k``.  The choice is a pure function of the call's shape and the tuning
snapshot, never of CUDA-graph capture state (AGENTS.md rule 11).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("oasr.jit.attention")

__all__ = [
    "FAMILY",
    "OP",
    "AUTO_ID",
    "LANES",
    "AttnChoice",
    "AttnShape",
    "CUTE_TILES",
    "CXX_SPLITS",
    "config_id",
    "choice_to_params",
    "choice_from_params",
    "q_class",
    "batch_bucket",
    "sig_key",
    "impl_hash",
    "unusable_reason",
    "resolve",
    "tuned_choice",
    "covering_tier",
    "runner_name_for_lane",
]

#: The tuning-DB kernel family, and its one operation.
FAMILY = "fmha"
OP = "fmha"

#: The config id (and ``{"kind": ...}``) of "measured; today's routing is best here".
AUTO_ID = "auto"

#: The lanes a choice may name.  ``sdpa`` is torch's, an external runner.
LANES = ("cute", "cxx", "sdpa")

#: The CuTeDSL tiles ``(m_block, n_block, num_threads)`` the build measures.
#: The kernel's own default is the first.  ``can_implement`` requires
#: ``(m_block * 2) % num_threads == 0``, so a 32-row tile runs two warps and a
#: 128-row one up to eight; ``select_tile`` only ever narrows ``n_block``.
CUTE_TILES: Tuple[Tuple[int, int, int], ...] = (
    (64, 64, 128),
    (64, 32, 128),
    (64, 16, 128),
    (32, 64, 64),
    (32, 32, 64),
    (32, 16, 64),
    (128, 32, 256),
)

#: The C++ lane's split-KV settings the build measures: 0 is the lane's own
#: rule (``oasr.jit.fmha.num_splits``), 1 never splits, ``s`` splits ``s`` ways.
CXX_SPLITS: Tuple[int, ...] = (0, 1, 2, 4, 8)

#: The query-length classes a signature keys: a decode step, the row counts
#: one 16- / 32- / 64-row M tile covers, and everything longer.
_Q_EDGES = (1, 16, 32, 64)


@dataclass(frozen=True)
class AttnChoice:
    """One lane and its configuration: what a tuning-file region names."""

    lane: str
    #: CuTeDSL tile.
    m_block: int = 64
    n_block: int = 64
    threads: int = 128
    #: C++ split-KV: 0 = the lane's own rule, 1 = never, ``s`` = ``s`` ways.
    splits: int = 0

    def __post_init__(self) -> None:
        if self.lane not in LANES:
            raise ValueError(f"unknown attention lane {self.lane!r}; expected one of {LANES}")
        if self.splits < 0:
            raise ValueError(f"splits must be >= 0, got {self.splits}")


@dataclass(frozen=True)
class AttnShape:
    """One ``fmha`` call, as far as selection is concerned."""

    dtype: str  # "float16" | "bfloat16"
    head_dim: int
    num_heads: int
    num_kv_heads: int
    mask: str  # "none" | "causal" | "window"
    bias: bool
    #: Paged KV page size; 0 for a dense K/V.
    block_size: int
    #: Per-row left padding (``cache_seqstarts``).
    seqstart: bool
    batch: int
    t_q: int
    t_k: int
    #: The bias's key axis is even, so its row stride is 4-byte aligned.  Both
    #: lanes branch on it -- the CuTeDSL kernel compiles a vectorised bias load,
    #: the C++ one copies an unaligned bias -- and measured, it flips the winner
    #: between neighbouring key lengths.  ``False`` without a bias.
    bias_aligned: bool = False

    @property
    def paged(self) -> bool:
        return self.block_size > 0


# =============================================================================
# The codec
# =============================================================================


def config_id(choice: Any) -> str:
    """The stable id a choice is stored under: ``cute_m64n32t128``, ``cxx_s4``, ``sdpa``."""
    if choice == AUTO_ID:
        return AUTO_ID
    if not isinstance(choice, AttnChoice):
        raise TypeError(f"not an attention choice: {choice!r}")
    if choice.lane == "cute":
        return f"cute_m{choice.m_block}n{choice.n_block}t{choice.threads}"
    if choice.lane == "cxx":
        return "cxx" if choice.splits == 0 else f"cxx_s{choice.splits}"
    return "sdpa"


def choice_to_params(choice: Any) -> Dict[str, Any]:
    """A choice as a tuning file's parameter dict, ``kind`` = the lane (or ``"auto"``)."""
    if choice == AUTO_ID:
        return {"kind": AUTO_ID}
    if not isinstance(choice, AttnChoice):
        raise TypeError(f"not an attention choice: {choice!r}")
    if choice.lane == "cute":
        return {
            "kind": "cute",
            "m_block": choice.m_block,
            "n_block": choice.n_block,
            "threads": choice.threads,
        }
    if choice.lane == "cxx":
        return {"kind": "cxx", "splits": choice.splits}
    return {"kind": "sdpa"}


def choice_from_params(params: Dict[str, Any]) -> Any:
    """Inverse of :func:`choice_to_params`; raises ``ValueError`` on a kind it does not know."""
    kind = params.get("kind")
    if kind == AUTO_ID:
        return AUTO_ID
    if kind == "cute":
        return AttnChoice(
            "cute",
            m_block=int(params["m_block"]),
            n_block=int(params["n_block"]),
            threads=int(params["threads"]),
        )
    if kind == "cxx":
        return AttnChoice("cxx", splits=int(params.get("splits", 0)))
    if kind == "sdpa":
        return AttnChoice("sdpa")
    raise ValueError(f"unknown fmha config kind {kind!r}")


# =============================================================================
# Signatures
# =============================================================================


def q_class(t_q: int) -> str:
    """The query-length class a signature keys (see :data:`_Q_EDGES`)."""
    for edge in _Q_EDGES:
        if t_q <= edge:
            return str(edge)
    return "L"


def batch_bucket(batch: int) -> int:
    """The batch rounded up to a power of two: the CTA count is what moves a crossover."""
    b = max(1, int(batch))
    return 1 << (b - 1).bit_length()


def sig_key(shape: AttnShape) -> str:
    """The tuning-DB key of *shape*'s static signature (the regions run over ``t_k``)."""
    from oasr.tune.database import sig_key as _key

    return _key(
        FAMILY,
        op=OP,
        dt="half",
        D=int(shape.head_dim),
        H=int(shape.num_heads),
        Hkv=int(shape.num_kv_heads),
        mask=shape.mask,
        bias=int(bool(shape.bias)),
        ba=int(bool(shape.bias and shape.bias_aligned)),
        kv="dense" if shape.block_size <= 0 else f"paged{int(shape.block_size)}",
        ss=int(bool(shape.seqstart)),
        q=q_class(shape.t_q),
        B=batch_bucket(shape.batch),
    )


# =============================================================================
# Implementation identity (soft validators)
# =============================================================================


def impl_hash() -> str:
    """Identity of the attention kernels a tuning result measured.

    Both lanes' sources: the C++ lane's templates and headers, and the CuTeDSL
    kernel's Python.  A soft validator: a change marks the file stale.
    """
    from pathlib import Path

    from oasr.jit import env
    from oasr.tune.database import hash_paths

    paths = sorted(env.OASR_TEMPLATE_DIR.glob("fmha*_template.cu.jinja"))
    paths += sorted((env.OASR_INCLUDE_DIR / "oasr" / "attention").rglob("*.h*"))
    kernels = Path(__file__).resolve().parent.parent / "kernels" / "cute" / "attention"
    paths += sorted(kernels.glob("*.py"))
    return hash_paths(paths)


def _soft_validators() -> Dict[str, Any]:
    return {"impl_hash": impl_hash()}


# =============================================================================
# Servability
# =============================================================================


def runner_name_for_lane(lane: str) -> str:
    """The registered runner (``oasr.tune.runners``) a lane's calls are counted under."""
    return {"cute": "oasr_cute", "cxx": "oasr_cxx"}.get(lane, "sdpa")


def _cxx_n_blocks(shape: AttnShape) -> int:
    from oasr.jit.fmha import resolve_config

    tile = resolve_config(_sm(), 16, shape.head_dim)
    return -(-shape.t_k // tile.block_n) if tile.valid else 0


def unusable_reason(choice: Any, shape: AttnShape) -> Optional[str]:
    """Why *choice* cannot serve *shape* in this process, or ``None`` when it can."""
    from oasr.tune.runners import unservable

    from .attention import fmha_config_supported, fmha_cxx_supports, select_backend

    if choice == AUTO_ID:
        return None
    if not isinstance(choice, AttnChoice):
        return "cannot be decoded as an attention choice"
    why = unservable(FAMILY, OP, choice.lane, shape.t_q, shape.t_k, shape.head_dim, shape.dtype)
    if why is not None:
        return why
    causal = shape.mask == "causal"
    local = shape.mask == "window"
    if choice.lane == "cute":
        if select_backend() != "cute":
            return "the CuTeDSL lane is not available in this process"
        if local:
            return "the CuTeDSL kernel has no per-row sliding window"
        if shape.paged and choice.m_block > 64:
            # The paged path pads the bias and the block table to 64-row /
            # 64-column tiles (`_pad_paged_inputs`); a taller tile would read
            # past the padding.
            return f"a {choice.m_block}-row tile exceeds the paged path's 64-row padding"
        if not fmha_config_supported(
            head_dim=shape.head_dim,
            dtype_str=shape.dtype,
            has_bias=shape.bias,
            paged=shape.paged,
            block_size=shape.block_size,
            m_block=choice.m_block,
            n_block=choice.n_block,
            num_threads=choice.threads,
            bias_aligned=shape.bias and shape.bias_aligned,
            causal=causal,
        ):
            return (
                f"the CuTeDSL kernel cannot implement a {choice.m_block}x{choice.n_block} "
                f"tile on {choice.threads} threads here"
            )
        return None
    if choice.lane == "cxx":
        if not fmha_cxx_supports(
            head_dim=shape.head_dim,
            dtype_str=shape.dtype,
            paged=shape.paged,
            causal=causal,
            local=local,
        ):
            return "the C++ lane cannot serve this shape"
        if choice.splits > 1 and (causal or local):
            return "split-KV is rendered for unmasked attention only"
        from oasr.jit.fmha import MAX_SPLITS

        if choice.splits > MAX_SPLITS:
            return f"{choice.splits} splits exceeds the lane's {MAX_SPLITS}"
        return None
    return None  # sdpa serves every shape


# =============================================================================
# Tiers, the memo, and the per-call path
# =============================================================================

#: ``(region t_k hi | None, config id, decoded choice | None)``; ``None`` = undecodable.
_Rule = Tuple[Optional[int], str, Any]

#: ``(epoch, sm) -> [(tier, {sig key: [rule, ...]}), ...]``, highest tier first.
_TIER_VIEWS: Dict[Tuple[int, int], List[Tuple[str, Dict[str, List[_Rule]]]]] = {}

#: The per-call memo: ``(epoch, shape fields..., bias_aligned) -> choice | None``.
_MEMO: Dict[tuple, Optional[AttnChoice]] = {}
_MEMO_MAX = 8192

#: The epoch at which this process was found to have no ``fmha`` tuning file.
_INACTIVE_EPOCH = -1

#: ``(tier, sig key, config id)`` already warned about.
_UNUSABLE_WARNED: set = set()

_MISS = object()


@functools.lru_cache(maxsize=1)
def _sm() -> int:
    from oasr.jit.core import _get_target_sm

    return int(_get_target_sm())


def _decode(params: Optional[Dict[str, Any]]) -> Any:
    if params is None:
        return None
    try:
        return choice_from_params(params)
    except (KeyError, TypeError, ValueError):
        return None


def _table_from_file(tf) -> Dict[str, List[_Rule]]:
    from oasr.tune.database import parse_sig_key

    table: Dict[str, List[_Rule]] = {}
    decoded: Dict[str, Any] = {}
    for key, entry in tf.entries.items():
        family, fields = parse_sig_key(key)
        if family != FAMILY or fields.get("op") != OP or not entry.regions:
            continue
        rules: List[_Rule] = []
        for t_hi, cid in entry.regions:
            if cid not in decoded:
                decoded[cid] = _decode(tf.configs.get(cid))
            rules.append((t_hi, cid, decoded[cid]))
        table[key] = rules
    return table


def _tier_views(sm: int) -> List[Tuple[str, Dict[str, List[_Rule]]]]:
    from oasr.tune import database

    epoch = database.epoch()
    key = (epoch, int(sm))
    views = _TIER_VIEWS.get(key)
    if views is not None:
        return views
    views = [(tier, _table_from_file(tf)) for tier, tf in database.tiers(FAMILY, sm).ordered()]
    for stale in [k for k in _TIER_VIEWS if k[0] != epoch]:
        del _TIER_VIEWS[stale]
    _TIER_VIEWS[key] = views
    return views


def _region(rules: List[_Rule], t_k: int) -> Optional[_Rule]:
    for rule in rules:
        if rule[0] is None or t_k <= rule[0]:
            return rule
    return None


def _note_unusable(tier: str, key: str, cid: str, reason: str) -> None:
    wkey = (tier, key, cid)
    if wkey in _UNUSABLE_WARNED:
        return
    _UNUSABLE_WARNED.add(wkey)
    logger.warning(
        "fmha %s-tier entry %s names %s, which %s; falling through", tier, key, cid, reason
    )


def resolve(
    shape: AttnShape, sm: Optional[int] = None
) -> Tuple[Optional[str], Optional[AttnChoice]]:
    """``(tier, choice)`` for *shape*: the first tier whose region covers it with a servable choice.

    ``choice`` is ``None`` for "today's routing" -- a region naming
    :data:`AUTO_ID`, or no covering entry (``tier`` is then ``None``).
    """
    sm_ = _sm() if sm is None else int(sm)
    key = sig_key(shape)
    for tier, table in _tier_views(sm_):
        rules = table.get(key)
        if not rules:
            continue
        rule = _region(rules, shape.t_k)
        if rule is None:  # above every bounded region: never measured there
            continue
        _, cid, choice = rule
        if choice == AUTO_ID:
            return tier, None
        reason = unusable_reason(choice, shape)
        if reason is None:
            return tier, choice
        _note_unusable(tier, key, cid, reason)
    return None, None


def covering_tier(shape: AttnShape) -> Optional[str]:
    """The tuning tier that serves *shape* (``"user"`` / ``"system"``), or ``None``."""
    return resolve(shape)[0]


def _active(epoch: int) -> bool:
    """Whether any ``fmha`` tuning file is loaded for this architecture this epoch."""
    global _INACTIVE_EPOCH
    if _INACTIVE_EPOCH == epoch:
        return False
    try:
        views = _tier_views(_sm())
    except Exception:  # noqa: BLE001 -- no device, no target: nothing to consult
        views = []
    if not views:
        _INACTIVE_EPOCH = epoch
        return False
    return True


def _routed_lane(shape: AttnShape) -> str:
    """The lane today's routing takes for *shape* (what an untuned call runs)."""
    from .attention import fmha_backend_for, select_backend

    if shape.mask == "window":
        return fmha_backend_for(
            head_dim=shape.head_dim,
            dtype_str=shape.dtype,
            paged=shape.paged,
            causal=False,
            local=True,
        )
    return select_backend()


def _memoise(key: tuple, shape: AttnShape) -> Optional[AttnChoice]:
    from oasr.tune.database import record_runner, record_tier

    tier, choice = resolve(shape)
    record_tier(FAMILY, OP, tier or AUTO_ID)
    lane = choice.lane if choice is not None else _routed_lane(shape)
    record_runner(FAMILY, OP, runner_name_for_lane(lane))
    if len(_MEMO) >= _MEMO_MAX:
        _MEMO.clear()
    _MEMO[key] = choice
    return choice


def tuned_choice(
    dtype: str,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    mask: str,
    bias: bool,
    block_size: int,
    seqstart: bool,
    batch: int,
    t_q: int,
    t_k: int,
    bias_aligned: bool = False,
) -> Optional[AttnChoice]:
    """The measured choice for one :func:`~oasr.functionals.attention.fmha` call, or ``None``.

    ``None`` means "route as today".  The caller has already established that
    nobody named a lane and the dtype is one the fused lanes serve.
    """
    epoch = _epoch()
    if epoch == _INACTIVE_EPOCH:
        return None
    key = (
        epoch,
        dtype,
        head_dim,
        num_heads,
        num_kv_heads,
        mask,
        bias,
        block_size,
        seqstart,
        batch,
        t_q,
        t_k,
        bias_aligned,
    )
    hit = _MEMO.get(key, _MISS)
    if hit is not _MISS:
        return hit  # type: ignore[return-value]
    if not _active(epoch):
        return None
    shape = AttnShape(
        dtype,
        head_dim,
        num_heads,
        num_kv_heads,
        mask,
        bias,
        block_size,
        seqstart,
        batch,
        t_q,
        t_k,
        bias_aligned and bias,
    )
    return _memoise(key, shape)


def _on_tuning_reload() -> None:
    global _INACTIVE_EPOCH
    _TIER_VIEWS.clear()
    _MEMO.clear()
    _INACTIVE_EPOCH = -1


def _register_with_tuning_db() -> Callable[[], int]:
    from oasr.tune import database

    database.register_soft_validator(FAMILY, _soft_validators)
    database.register_reload_hook(_on_tuning_reload)
    return database.epoch


#: ``oasr.tune.database.epoch``, bound once: the per-call path runs no import statement.
_epoch: Callable[[], int] = _register_with_tuning_db()
