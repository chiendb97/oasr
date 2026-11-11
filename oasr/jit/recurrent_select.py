# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Measured selection of the recurrent path: the ``recurrent`` tuning family.

Three hand-timed cut-offs decide how an LSTM / RNN layer runs, all timed on one
RTX 5090 (:mod:`oasr.jit.measured` counts them as extrapolations anywhere else):

* ``oasr.layers.recurrent._TENSOR_CORE_MIN_BATCH_*`` -- the scalar cohort kernel
  (``oasr.lstm_layer``) or the tensor-core path (``oasr.lstm_gemm_layer``);
* ``oasr.jit.recurrent_cute._LSTM_BANDS`` -- at a single timestep, the fused
  recurrent step or the decomposed GEMM-plus-finalizer path;
* ``_TILES`` / ``kRecurrentStepRoutes`` and ``_AUTO_ORDER`` -- which fused lane,
  and which tile; ``_default_recurrent_tactic`` -- which tactic the GEMM path runs.

They are one decision, a path per ``(op, hidden, input width, single step or
sequence)`` and batch, and this module makes it a family of the tuning database.
A region names a :class:`RecChoice` or :data:`AUTO_ID`, "measured; today's
rules are best here", and ``oasr tune build --family recurrent`` measures every
path at each point.

Who consults it: :class:`oasr.layers.LSTM` / :class:`oasr.layers.RNN` for the
scalar-or-tensor-core half, and :func:`oasr.lstm_gemm_layer` /
:func:`oasr.rnn_gemm_layer` for the rest (fused step, its lane and tile, the GEMM
tactic).  A shape a file covers never reaches the hand-timed tables, so it is not
an extrapolation; one it does not cover runs exactly as before.  Not consulted
under ``OASR_RECURRENT_CUTE=0|1`` or a pinned ``OASR_RECURRENT_BACKEND`` (an
operator who names a path requires it), nor when a caller passes ``_tactic``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from . import recurrent_cute as _rc
from .selection import AUTO_ID, MeasuredTable

__all__ = [
    "FAMILY",
    "OP",
    "AUTO_ID",
    "PATHS",
    "OPS",
    "RecChoice",
    "RecShape",
    "GEMM_TACTICS",
    "config_id",
    "choice_to_params",
    "choice_from_params",
    "sig_key",
    "impl_hash",
    "unusable_reason",
    "resolve",
    "tuned_choice",
    "runner_name_for_path",
]

FAMILY = "recurrent"
#: The family's one table operation; the recurrent *kind* is a signature field.
OP = "layer"

#: The recurrent kinds, as ``oasr.jit.recurrent_step`` spells them.
OPS = ("lstm", "rnn_tanh", "rnn_relu")

#: ``scalar``: ``oasr.lstm_layer`` / ``oasr.rnn_layer`` (cohort or GEMV kernel).
#: ``gemm``: the tensor-core layer, decomposed (CUTLASS GEMM + fused epilogue).
#: ``cxx`` / ``cute``: the fused recurrent step, a single timestep, LSTM only.
PATHS = ("scalar", "gemm", "cxx", "cute")

#: The decomposed path's tactics, ``(tactic id, split)``, as
#: ``oasr.tune.backends.recurrent`` registers them for sm80-class targets.
#: ``(-1, 1)`` is :func:`oasr.functionals.recurrent._default_recurrent_tactic`.
GEMM_TACTICS: Tuple[Tuple[int, int], ...] = (
    (-1, 1),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 2),
    (4, 4),
    (5, 2),
    (5, 4),
)

#: Tactic 5 (serial split-K) is built for the LSTM launcher only.
_LSTM_ONLY_TACTICS = (5,)


@dataclass(frozen=True)
class RecChoice:
    """One recurrent path and its configuration."""

    path: str
    #: ``gemm``: the tactic id (``-1`` = today's batch rule) and its split.
    tactic: int = -1
    split: int = 1
    #: ``cxx`` / ``cute``: index into ``oasr.jit.recurrent_step.TILES`` (both
    #: lanes share the eight tiles).
    tile: int = -1

    def __post_init__(self) -> None:
        if self.path not in PATHS:
            raise ValueError(f"unknown recurrent path {self.path!r}; expected one of {PATHS}")


@dataclass(frozen=True)
class RecShape:
    """One recurrent layer call, as far as selection is concerned."""

    op: str
    dtype: str
    hidden: int
    input_size: int
    seq_len: int
    batch: int

    @property
    def step(self) -> bool:
        return self.seq_len == 1


# =============================================================================
# The codec
# =============================================================================


def config_id(choice: Any) -> str:
    """``scalar``, ``gemm``, ``gemm_t2``, ``gemm_t4_s2``, ``cxx_t3``, ``cute_t5`` or ``auto``."""
    if choice == AUTO_ID:
        return AUTO_ID
    if not isinstance(choice, RecChoice):
        raise TypeError(f"not a recurrent choice: {choice!r}")
    if choice.path == "scalar":
        return "scalar"
    if choice.path == "gemm":
        if choice.tactic < 0:
            return "gemm"
        return f"gemm_t{choice.tactic}" + (f"_s{choice.split}" if choice.split > 1 else "")
    return f"{choice.path}_t{choice.tile}"


def choice_to_params(choice: Any) -> Dict[str, Any]:
    if choice == AUTO_ID:
        return {"kind": AUTO_ID}
    if not isinstance(choice, RecChoice):
        raise TypeError(f"not a recurrent choice: {choice!r}")
    if choice.path == "scalar":
        return {"kind": "scalar"}
    if choice.path == "gemm":
        return {"kind": "gemm", "tactic": choice.tactic, "split": choice.split}
    return {"kind": choice.path, "tile": choice.tile}


def choice_from_params(params: Dict[str, Any]) -> Any:
    kind = params.get("kind")
    if kind == AUTO_ID:
        return AUTO_ID
    if kind == "scalar":
        return RecChoice("scalar")
    if kind == "gemm":
        return RecChoice("gemm", tactic=int(params["tactic"]), split=int(params.get("split", 1)))
    if kind in ("cxx", "cute"):
        return RecChoice(kind, tile=int(params["tile"]))
    raise ValueError(f"unknown recurrent config kind {kind!r}")


def sig_key(shape: RecShape) -> str:
    """``recurrent|op=layer|kind=lstm|dt=half|H=640|I=640|T=step``; regions run over the batch."""
    from oasr.tune.database import sig_key as _key

    return _key(
        FAMILY,
        op=OP,
        kind=shape.op,
        dt="half",
        H=int(shape.hidden),
        I=int(shape.input_size),
        T="step" if shape.step else "seq",
    )


def impl_hash() -> str:
    """Identity of every path's kernels (soft validator): a change stales the file."""
    from pathlib import Path

    from oasr.jit import env
    from oasr.tune.database import hash_paths

    paths = sorted((env.OASR_INCLUDE_DIR / "oasr" / "recurrent").rglob("*.h*"))
    paths += sorted(env.OASR_TEMPLATE_DIR.glob("recurrent*.jinja"))
    paths += sorted(env.OASR_CSRC_DIR.glob("recurrent*.cu"))
    kernels = Path(__file__).resolve().parent.parent / "kernels" / "cute" / "recurrent"
    paths += sorted(kernels.glob("*.py"))
    return hash_paths(paths)


def _soft_validators() -> Dict[str, Any]:
    return {"impl_hash": impl_hash()}


# =============================================================================
# Servability
# =============================================================================


def runner_name_for_path(path: str) -> str:
    return {
        "scalar": "oasr_scalar",
        "gemm": "oasr_gemm",
        "cxx": "oasr_cxx_step",
        "cute": "oasr_cute_step",
    }[path]


def _tensor_core_ok(shape: RecShape) -> Optional[str]:
    import torch

    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return "the tensor-core path is not selected below sm80"
    if shape.input_size % 8 or shape.hidden % 8:
        return "the tensor-core path needs input and hidden widths that are multiples of 8"
    return None


def unusable_reason(choice: Any, shape: RecShape) -> Optional[str]:
    """Why *choice* cannot serve *shape* in this process, or ``None`` when it can."""
    from oasr.tune.runners import unservable

    if choice == AUTO_ID:
        return None
    if not isinstance(choice, RecChoice):
        return "cannot be decoded as a recurrent choice"
    why = unservable(
        FAMILY, OP, choice.path, shape.batch, shape.hidden, shape.input_size, shape.dtype
    )
    if why is not None:
        return why
    if choice.path == "scalar":
        return None
    why = _tensor_core_ok(shape)
    if why is not None:
        return why
    if choice.path == "gemm":
        if (choice.tactic, choice.split) not in GEMM_TACTICS:
            return f"tactic {choice.tactic} split {choice.split} is not built for this target"
        if choice.tactic in _LSTM_ONLY_TACTICS and shape.op != "lstm":
            return f"tactic {choice.tactic} is built for the LSTM launcher only"
        return None
    # The fused step: one LSTM timestep.
    if shape.op != "lstm":
        return "the fused recurrent step is routed for the LSTM only"
    if not shape.step:
        return "the fused recurrent step serves a single timestep"
    from . import recurrent_step as rs

    rc = _rc

    if not (0 <= choice.tile < len(rs.TILES)):
        return f"no tile {choice.tile}"
    if choice.path == "cxx":
        sm = rc._cxx_probe()
        if sm is None:
            return "the C++ recurrent step lane is not available here"
        if not rs.shape_supported(batch=shape.batch, hidden=shape.hidden, k=shape.hidden):
            return "the C++ recurrent step cannot serve this width"
        if not rs.tile_valid(rs.TILES[choice.tile], sm, 2, 4):
            return f"tile {choice.tile} does not fit sm_{sm}"
        return None
    if rc._probe() is None:
        return "the CuTeDSL recurrent step is not available here"
    tile = rs.TILES[choice.tile]
    if shape.hidden % tile.block_k:
        return f"the CuTeDSL step needs hidden to be a multiple of {tile.block_k}"
    return None


# =============================================================================
# The per-call path
# =============================================================================


def _decode(params: Dict[str, Any]) -> Any:
    return choice_from_params(params)


_TABLE = MeasuredTable(FAMILY, OP, _decode, unusable_reason)

#: ``(epoch, op, dtype, hidden, input_size, step, batch) -> choice | None``.
_MEMO: Dict[tuple, Optional[RecChoice]] = {}
_MEMO_MAX = 8192
_MISS = object()


def resolve(shape: RecShape) -> Tuple[Optional[str], Optional[RecChoice]]:
    """``(tier, choice)`` for *shape*; ``choice`` ``None`` means today's rules."""
    tier, choice = _TABLE.resolve(sig_key(shape), shape.batch, shape)
    return tier, choice


def tuned_choice(
    op: str, dtype: str, hidden: int, input_size: int, seq_len: int, batch: int
) -> Optional[RecChoice]:
    """The measured path for one recurrent layer call, or ``None`` (today's rules).

    Memoised per ``(epoch, shape)``; with no ``recurrent`` file for this arch the
    lookup returns at once.  The first resolution of a shape is counted by tier
    and by runner.
    """
    epoch = _TABLE.epoch()
    if epoch == _TABLE.inactive_epoch:
        return None
    # An operator who names a path (``OASR_RECURRENT_CUTE=0|1``, a pinned
    # ``OASR_RECURRENT_BACKEND``) requires it: the table is not consulted.
    if _rc._MODE != "auto" or _rc._BACKEND != "auto":
        return None
    key = (epoch, op, dtype, hidden, input_size, seq_len == 1, batch)
    hit = _MEMO.get(key, _MISS)
    if hit is not _MISS:
        return hit  # type: ignore[return-value]
    if not _TABLE.active(epoch):
        return None
    from oasr.tune.database import record_runner, record_tier

    shape = RecShape(op, dtype, hidden, input_size, seq_len, batch)
    tier, choice = resolve(shape)
    record_tier(FAMILY, op, tier or AUTO_ID)
    if choice is not None:
        record_runner(FAMILY, op, runner_name_for_path(choice.path))
    if len(_MEMO) >= _MEMO_MAX:
        _MEMO.clear()
    _MEMO[key] = choice
    return choice


def _on_reload() -> None:
    _MEMO.clear()


def _register() -> None:
    from oasr.tune import database

    database.register_soft_validator(FAMILY, _soft_validators)
    database.register_reload_hook(_on_reload)


_register()
