# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Measured selection of the gated MLP path: the ``gated_mlp`` tuning family.

:func:`oasr.jit.mlp.routed_gated_mlp` decides by rule: fuse inside one m-tile
(``rows <= 64``), on the C++ lane first, at the tile wave arithmetic picks from a
candidate list chosen on one RTX 5090 (:mod:`oasr.jit.measured` counts that list
as an extrapolation anywhere else).  This module makes the decision a family of
the tuning database: per ``(activation, bias, N, K)``, regions over the row count
name a :class:`MlpChoice` -- the fused kernel on a lane with a named tile, or the
two-GEMM path (``oasr.gemm_activation`` + ``oasr.gemm`` + a multiply, each GEMM
resolved through the GEMM database) -- or :data:`AUTO_ID`, "measured; today's
rule is best here".  ``oasr tune build --family gated_mlp`` measures them.

Consulted by :func:`oasr.jit.mlp.routed_gated_mlp`, so by
:func:`oasr.gated_mlp`, :func:`oasr.gated_mlp_available` and
:class:`oasr.layers.GatedMLP` alike.  Not consulted under
``OASR_GATED_MLP_CUTE=0|1`` or a pinned ``OASR_GATED_MLP_BACKEND``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from . import mlp as _mlp
from .selection import AUTO_ID, MeasuredTable

__all__ = [
    "FAMILY",
    "OP",
    "AUTO_ID",
    "PATHS",
    "MlpChoice",
    "MlpShape",
    "config_id",
    "choice_to_params",
    "choice_from_params",
    "sig_key",
    "impl_hash",
    "unusable_reason",
    "resolve",
    "tuned_choice",
    "runner_name_for_path",
    "tile_spec",
    "num_tiles",
]

FAMILY = "gated_mlp"
OP = "gated_mlp"

#: ``gemm``: two GEMMs and a multiply.  ``cxx`` / ``cute``: the fused kernel.
PATHS = ("gemm", "cxx", "cute")


def num_tiles() -> int:
    """Both lanes rank the same six tiles (``kGatedMlpTiles`` / ``_CANDIDATES``)."""
    return sum(len(tiles) for _, tiles in _mlp._CANDIDATES)


def tile_spec(index: int) -> Tuple[int, int, int, int, int, int]:
    """``(m, n, k, stages, threads, warps_n)`` of shared tile *index*."""
    flat = [t for _, tiles in _mlp._CANDIDATES for t in tiles]
    return tuple(flat[index])  # type: ignore[return-value]


@dataclass(frozen=True)
class MlpChoice:
    path: str
    #: ``cxx`` / ``cute``: index of the tile both lanes list in the same order.
    tile: int = -1

    def __post_init__(self) -> None:
        if self.path not in PATHS:
            raise ValueError(f"unknown gated-MLP path {self.path!r}; expected one of {PATHS}")


@dataclass(frozen=True)
class MlpShape:
    dtype: str
    activation: str
    bias: bool
    n: int
    k: int
    rows: int


def config_id(choice: Any) -> str:
    """``gemm``, ``cxx_t2``, ``cute_t5`` or ``auto``."""
    if choice == AUTO_ID:
        return AUTO_ID
    if not isinstance(choice, MlpChoice):
        raise TypeError(f"not a gated-MLP choice: {choice!r}")
    return "gemm" if choice.path == "gemm" else f"{choice.path}_t{choice.tile}"


def choice_to_params(choice: Any) -> Dict[str, Any]:
    if choice == AUTO_ID:
        return {"kind": AUTO_ID}
    if not isinstance(choice, MlpChoice):
        raise TypeError(f"not a gated-MLP choice: {choice!r}")
    if choice.path == "gemm":
        return {"kind": "gemm"}
    return {"kind": choice.path, "tile": choice.tile}


def choice_from_params(params: Dict[str, Any]) -> Any:
    kind = params.get("kind")
    if kind == AUTO_ID:
        return AUTO_ID
    if kind == "gemm":
        return MlpChoice("gemm")
    if kind in ("cxx", "cute"):
        return MlpChoice(kind, tile=int(params["tile"]))
    raise ValueError(f"unknown gated-MLP config kind {kind!r}")


def sig_key(shape: MlpShape) -> str:
    """``gated_mlp|op=gated_mlp|dt=half|act=silu|bias=0|N=11008|K=4096``; regions over rows."""
    from oasr.tune.database import sig_key as _key

    return _key(
        FAMILY,
        op=OP,
        dt="half",
        act=shape.activation,
        bias=int(bool(shape.bias)),
        N=int(shape.n),
        K=int(shape.k),
    )


def impl_hash() -> str:
    """Identity of the fused kernels (soft validator); the GEMM path has its own file."""
    from pathlib import Path

    from oasr.jit import env
    from oasr.tune.database import hash_paths

    paths = sorted((env.OASR_INCLUDE_DIR / "oasr" / "mlp").rglob("*.h*"))
    paths += sorted(env.OASR_TEMPLATE_DIR.glob("gated_mlp*.jinja"))
    kernels = Path(__file__).resolve().parent.parent / "kernels" / "cute" / "mlp"
    paths += sorted(kernels.glob("*.py"))
    return hash_paths(paths)


def _soft_validators() -> Dict[str, Any]:
    return {"impl_hash": impl_hash()}


def runner_name_for_path(path: str) -> str:
    return {"gemm": "oasr_gemm", "cxx": "oasr_cxx", "cute": "oasr_cute"}[path]


def unusable_reason(choice: Any, shape: MlpShape) -> Optional[str]:
    """Why *choice* cannot serve *shape* in this process, or ``None`` when it can."""
    from oasr.tune.runners import unservable

    if choice == AUTO_ID:
        return None
    if not isinstance(choice, MlpChoice):
        return "cannot be decoded as a gated-MLP choice"
    why = unservable(FAMILY, OP, choice.path, shape.rows, shape.n, shape.k, shape.dtype)
    if why is not None:
        return why
    if choice.path == "gemm":
        return None
    if not (0 <= choice.tile < num_tiles()):
        return f"no tile {choice.tile}"
    if choice.path == "cxx":
        sm = _mlp._cxx_probe()
        if sm is None:
            return "the C++ gated-MLP lane is not available here"
        from oasr.jit import gated_mlp as _cxx

        if not _cxx.config_supported(
            sm=sm,
            dtype_str=shape.dtype,
            activation=shape.activation,
            rows=shape.rows,
            n=shape.n,
            k=shape.k,
        ):
            return "the C++ gated MLP cannot serve this shape"
        if not _cxx.tile_valid(_cxx.TILES[choice.tile], sm):
            return f"tile {choice.tile} does not fit sm_{sm}"
        return None
    if _mlp._capability_probe() is None:
        return "the CuTeDSL gated MLP is not available here"
    spec = tile_spec(choice.tile)
    if not _mlp.gated_mlp_shape_supported(rows=shape.rows, n=shape.n, k=shape.k, k_block=spec[2]):
        return f"K must be a whole number of {spec[2]}-wide tiles for the CuTeDSL kernel"
    return None


def _decode(params: Dict[str, Any]) -> Any:
    return choice_from_params(params)


_TABLE = MeasuredTable(FAMILY, OP, _decode, unusable_reason)

#: ``(epoch, dtype, activation, bias, n, k, rows) -> choice | None``.
_MEMO: Dict[tuple, Optional[MlpChoice]] = {}
_MEMO_MAX = 8192
_MISS = object()


def resolve(shape: MlpShape) -> Tuple[Optional[str], Optional[MlpChoice]]:
    """``(tier, choice)``; ``choice`` ``None`` means today's rule."""
    tier, choice = _TABLE.resolve(sig_key(shape), shape.rows, shape)
    return tier, choice


def tuned_choice(
    dtype: str, activation: str, bias: bool, n: int, k: int, rows: int
) -> Optional[MlpChoice]:
    """The measured path for one gated-MLP call, or ``None`` (today's rule).

    Memoised per ``(epoch, shape)``; with no ``gated_mlp`` file for this arch the
    lookup returns at once.  Not consulted when ``OASR_GATED_MLP_CUTE`` or
    ``OASR_GATED_MLP_BACKEND`` names a path.
    """
    epoch = _TABLE.epoch()
    if epoch == _TABLE.inactive_epoch:
        return None
    if _mlp._GATED_MLP_MODE != "auto" or _mlp._GATED_MLP_BACKEND != "auto":
        return None
    key = (epoch, dtype, activation, bias, n, k, rows)
    hit = _MEMO.get(key, _MISS)
    if hit is not _MISS:
        return hit  # type: ignore[return-value]
    if not _TABLE.active(epoch):
        return None
    from oasr.tune.database import record_runner, record_tier

    shape = MlpShape(dtype, activation, bias, n, k, rows)
    tier, choice = resolve(shape)
    record_tier(FAMILY, OP, tier or AUTO_ID)
    if choice is not None:
        record_runner(FAMILY, OP, runner_name_for_path(choice.path))
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
