# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the 8-bit scaled GEMM (``oasr.scaled_mm``).

``D = act(a_scale[m] * b_scale[n] * (A @ Bᵀ)[m, n] + bias[n])`` over 8-bit A and
B: the W8A8 GEMM of a quantized layer, with dequantization, bias and activation
in one CUTLASS 2.x epilogue visitor tree (``include/oasr/scaled_mm/``).

A family of its own rather than more ``gemm`` variants: ``gemm`` is fp16/bf16
throughout (instruction shape ``16x8x16``, float accumulator, half-only
dispatch, ``dt=half`` rule tables), and 8-bit variants there would grow every
user's cold build.  The config-space helpers are shared, at 1-byte operands.

Two operand formats, one JIT module per format and lane (so an int8 user never
compiles the fp8 variants, nor the reverse).  A *lane* is a kernel generation
with its own config space, module and tuning entries:

* The CUTLASS 2.x ``mma.sync`` lanes (:data:`LANE_ID`), one template:

  * ``int8`` -- the Sm80 int8 kernel with an exact int32 accumulator, serving
    every SM >= 80: natively on sm80/86/89/120, and on sm90/sm100 as the A/B
    arm of the native lanes below (``OASR_SCALED_MM_BACKEND=mma_sync``).
    Turing (sm75) is a declared kernel gap (``scaled-mm-int8-sm75`` in
    :data:`oasr.layers._backend.KERNEL_GAPS`).
  * ``fp8`` -- the Sm89 e4m3 kernel (Ada's instruction, which sm_120 executes
    too) with an fp32 accumulator and fast accumulation, natively on sm89/sm120.
    Below sm89 there are no fp8 tensor cores: not a gap, a scheme the GPU cannot
    run.  Its accumulator is not exact, so batch invariance rests on the config
    space instead: no tile slices K across warps and none splits K, so every
    tile sums each output in the same order and computes the same bits
    (``cutlass_scaled_mm_configs.h``).

* The native CUTLASS 3.x lanes (:data:`NATIVE_LANE_ID`), the default on Hopper
  and Blackwell data-center parts: TMA, a warp-specialized persistent kernel,
  and the architecture's own MMA -- wgmma with ping-pong or cooperative
  warpgroups on sm90, tcgen05 accumulating in (double-buffered) tensor memory on
  sm100 -- so the epilogue of one tile overlaps the mainloop of the next
  (``cutlass_scaled_mm_configs_sm90.h``).  The same epilogue arithmetic and the
  same launcher signatures as the 2.x lane; the same per-lane invariant (int8:
  exact; fp8: one summation order for every tile).  int8 on sm100 is
  narrower than its family: CUTLASS enables tcgen05 ``kind::i8`` for the
  arch-specific ``sm_100a`` target only, not ``sm_100f``, so that lane is built
  for ``sm_100a`` and serves compute capability 10.0 (B200) alone; B300 keeps
  the 2.x lane.

**Unmeasured:** the native production sets and their shape heuristic below are
starting points chosen without Hopper or Blackwell hardware.  They are
compile-checked, not tuned; a measured tuning entry (``oasr tune build``)
overrides the heuristic per shape.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from . import env
from .core import JitSpec, _get_target_sm, gen_jit_spec
from .gemm import _SM_MAX_SMEM_BYTES, TileShape, _epilogue_covers_warp, _smem_bytes

logger = logging.getLogger(__name__)

#: The operand formats, each its own JIT module, config space and tuning family.
FORMATS = ("int8", "fp8")

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

#: The same for the Sm89 fp8 lane (``mma.sync`` e4m3 needs sm89+).
SCALED_MM_FP8_LANE_BY_SM: Dict[int, bool] = {
    89: True,
    120: True,
    90: False,
    100: False,
}

_LANE_BY_SM = {"int8": SCALED_MM_LANE_BY_SM, "fp8": SCALED_MM_FP8_LANE_BY_SM}

#: The native CUTLASS 3.x lane of each ``(SM family, format)``.
NATIVE_LANE_ID: Dict[Tuple[int, str], str] = {
    (90, "int8"): "sm90_int8_wgmma",
    (90, "fp8"): "sm90_fp8_wgmma",
    (100, "int8"): "sm100_int8_umma",
    (100, "fp8"): "sm100_fp8_umma",
}

_NATIVE_LANE_KEY = {lane: key for key, lane in NATIVE_LANE_ID.items()}

#: The compute capabilities a native lane runs on, where narrower than its SM
#: family.  tcgen05 ``kind::i8`` is enabled by CUTLASS for ``sm_100a`` /
#: ``sm_101a`` / ``sm_110a`` only -- not for the ``sm_100f`` family target the
#: rest of the sm100 family builds with (``oasr.jit.core._GENCODE_TARGET``) --
#: so an int8 kernel built for the family has no int8 MMA at all, and B300
#: (10.3), which the family also covers, has none to build for.  Checked on the
#: generated PTX: ``kind::i8`` under ``sm_100a``, nothing under ``sm_100f``.
_NATIVE_CAPABILITIES: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]] = {
    (100, "int8"): ((10, 0),),
}

#: The gencode target of a native module, where it is not its family's default.
_NATIVE_GENCODE: Dict[Tuple[int, str], str] = {
    (100, "int8"): "100a",
}

#: N alignment of the quantizing (8-bit-output) launch: the native lanes store D
#: through TMA, whose rows are 16-byte aligned -- 16 one-byte elements.
NATIVE_QUANT_N_ALIGNMENT = 16

#: The epilogue's thread map partitions fp32 compute fragments, four per
#: access (``CutlassScaledMmKernel::kEpilogueElementsPerAccess``), so the
#: warp-coverage rule is evaluated at 32-bit elements, not the 16-bit output.
_EPILOGUE_ELEMENT_BITS = 32

#: Operand alignment: 16-byte int8 loads on K, 16-byte output stores on N.
K_ALIGNMENT = 16
N_ALIGNMENT = 8


@dataclass(frozen=True)
class ScaledMmConfig:
    """One CUTLASS 2.x 8-bit tile configuration."""

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
    #: output only (a quantizing producer has a short K).  int8 only.
    stream_k: bool = False
    #: fp8 e4m3 operands (the Sm89 lane) instead of int8.
    fp8: bool = False

    @property
    def fmt(self) -> str:
        """The operand format, ``"int8"`` or ``"fp8"``."""
        return "fp8" if self.fp8 else "int8"

    @property
    def compile_name(self) -> str:
        return (
            f"b{self.block_m}x{self.block_n}x{self.block_k}_"
            f"w{self.warp_m}x{self.warp_n}x{self.warp_k}_s{self.stages}"
            + ("_sk" if self.stream_k else "")
            + ("_e4m3" if self.fp8 else "")
        )

    @property
    def tile(self) -> TileShape:
        return TileShape(
            self.block_m, self.block_n, self.block_k, self.warp_m, self.warp_n, self.warp_k
        )

    @property
    def lane(self) -> str:
        """The lane this config belongs to (its format's ``mma.sync`` lane)."""
        return LANE_ID[self.fmt]

    #: SMs one output tile occupies (the shape heuristic's wave arithmetic).
    sms_per_tile = 1


#: Native schedules and their C++ (``oasr::scaled_mm::NativeSchedule``) names.
_NATIVE_SCHEDULES = {
    90: {"pingpong": "kPingpong", "cooperative": "kCooperative"},
    100: {"1sm": "k1Sm", "2sm": "k2Sm"},
}

_SCHEDULE_TAG = {"pingpong": "pp", "cooperative": "coop", "1sm": "1sm", "2sm": "2sm"}


@dataclass(frozen=True)
class NativeScaledMmConfig:
    """One native (CUTLASS 3.x) 8-bit tile configuration on sm90 or sm100.

    ``block_m`` is the MMA tile's M -- for the 2-SM sm100 schedule the CTA pair's
    combined extent.  ``schedule``: ``pingpong`` / ``cooperative`` (sm90),
    ``1sm`` / ``2sm`` (sm100).  Data-parallel only: no Stream-K, no split-K.
    """

    sm: int
    block_m: int
    block_n: int
    cluster_m: int = 1
    cluster_n: int = 1
    schedule: str = "pingpong"
    block_k: int = 128
    #: fp8 e4m3 operands instead of int8.
    fp8: bool = False

    #: No Stream-K variant exists (the 2.x config's field, for one interface).
    stream_k = False

    @property
    def fmt(self) -> str:
        """The operand format, ``"int8"`` or ``"fp8"``."""
        return "fp8" if self.fp8 else "int8"

    @property
    def lane(self) -> str:
        return NATIVE_LANE_ID[(self.sm, self.fmt)]

    @property
    def compile_name(self) -> str:
        return (
            f"sm{self.sm}_b{self.block_m}x{self.block_n}x{self.block_k}_"
            f"c{self.cluster_m}x{self.cluster_n}_{_SCHEDULE_TAG[self.schedule]}"
            + ("_e4m3" if self.fp8 else "")
        )

    @property
    def cpp_schedule(self) -> str:
        """The ``NativeSchedule`` enumerator the rendered translation unit names."""
        return _NATIVE_SCHEDULES[self.sm][self.schedule]

    @property
    def sms_per_tile(self) -> int:
        return 2 if self.schedule == "2sm" else 1


#: Either generation's config.
AnyScaledMmConfig = Union[ScaledMmConfig, NativeScaledMmConfig]


#: The int8 data-parallel tiles.  A Conformer-width GEMM has K = 256: two to four
#: K tiles, so a CTA's pipeline prologue and its dequant/activation epilogue are
#: most of its life, and only a second CTA resident on the same SM overlaps them
#: with tensor-core work.  The shallow-K tiles are therefore sized for two or more
#: CTAs per SM (4 warps, <= 48 KiB of shared memory); the deep-K and the sparse
#: launch -- fewer CTAs than SMs, where a second CTA is never there -- take a
#: deeper pipeline instead.
_T16 = ScaledMmConfig(16, 64, 128, 16, 64, 64, 5)
_T64 = ScaledMmConfig(64, 64, 64, 32, 32, 64, 4)
_T64_DEEP_K = ScaledMmConfig(64, 64, 128, 32, 32, 128, 4)
_T64X128 = ScaledMmConfig(64, 128, 64, 32, 64, 64, 4)
_T128 = ScaledMmConfig(128, 128, 64, 64, 64, 64, 3)
_T128_SPARSE = ScaledMmConfig(128, 128, 64, 64, 64, 64, 4)
_T128X256 = ScaledMmConfig(128, 256, 64, 64, 64, 64, 3)

#: The data-parallel production set.  Each one also exports the int8-output
#: (quantizing) launcher.  Chosen from a sweep of 25 tiles over 23 shapes of five
#: architectures (U2++, Nemotron, whisper-tiny, whisper-large, Qwen2-7B; M = 1 to
#: 16384; the tuner's protocol, ``oasr.tune.build_scaled_mm.measure_case``) on an
#: RTX 5090, 2026-10-23: the best of these seven is within 0.6 % (geomean) of the
#: best of all 25.  The set it replaced was sized for one CTA per SM (3-5 stages):
#: on the Conformer widths above M = 256 its best tile was 1.13x slower than this
#: set's (geomean; up to 1.39x).
_PRODUCTION: Tuple[ScaledMmConfig, ...] = (
    _T16,
    _T64,
    _T64_DEEP_K,
    _T64X128,
    _T128,
    _T128_SPARSE,
    _T128X256,
)


@dataclass(frozen=True)
class _Rung:
    """One step of the int8 shape heuristic (:data:`_INT8_LADDER`)."""

    tile: ScaledMmConfig
    #: The rung serves when its tile launches at least this many CTAs per SM.
    min_ctas_per_sm: float
    #: The K range the rung serves, ``[min_k, max_k)``.
    min_k: int = 0
    max_k: Optional[int] = None
    #: Served instead when the tile's CTAs do not cover every SM once: with one
    #: CTA per SM, a deeper pipeline beats the second CTA that is never there.
    sparse: Optional[ScaledMmConfig] = None


#: The int8 shape heuristic, largest tile first: the first rung whose tile is less
#: than twice as tall as M (a taller one computes mostly padding rows) and launches
#: at least ``min_ctas_per_sm x num_sms`` CTAs serves, else :data:`_T16`.  The
#: thresholds are fitted to the sweep above; every leave-one-architecture-out fold
#: refits the same values.  Regret against the best of all 25 tiles: 0.9 % geomean,
#: worst 1.23x (the previous half-wave rule over the previous set: 5.4 %, worst
#: 1.47x; on the Conformer widths 0.5 % vs 7.4 %).
_INT8_LADDER: Tuple[_Rung, ...] = (
    _Rung(_T128X256, 0.5, min_k=2048),
    _Rung(_T128, 0.5, sparse=_T128_SPARSE),
    _Rung(_T64X128, 0.5),
    _Rung(_T64_DEEP_K, 0.2, min_k=1024),
    _Rung(_T64, 0.25, max_k=1024),
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


#: The fp8 production set (fast accumulation, ``block_k == warp_k``: no K is
#: sliced across warps, so every tile sums in the same order -- see the module
#: docstring).  Chosen from a sweep of 16 such tiles over 18 shapes (Conformer,
#: Whisper-large, Nemotron, Qwen2-7B; M from 1 to 16384) on an RTX 5090,
#: graph-captured, 2026-10-21: these eight are within 0.9 % of the best of all
#: sixteen at every point, and the shape heuristic below picks within 5 % of the
#: best of these eight on geometric mean (worst 1.56x at one mid-M point; a
#: tuning entry corrects such a shape).  Unlike int8, the 128-wide tiles rarely
#: win: at fp8's half-rate fp32 accumulation (GeForce) the 64x64 tile keeps
#: more CTAs per SM and is often fastest even at M = 16384.  No Stream-K.
_FP8_PRODUCTION: Tuple[ScaledMmConfig, ...] = (
    ScaledMmConfig(16, 64, 128, 16, 16, 128, 4, fp8=True),
    ScaledMmConfig(32, 64, 128, 16, 32, 128, 4, fp8=True),
    ScaledMmConfig(64, 64, 64, 32, 32, 64, 6, fp8=True),
    ScaledMmConfig(64, 64, 128, 32, 32, 128, 4, fp8=True),
    ScaledMmConfig(64, 128, 64, 32, 64, 64, 5, fp8=True),
    ScaledMmConfig(128, 64, 64, 64, 32, 64, 5, fp8=True),
    ScaledMmConfig(128, 128, 64, 64, 64, 64, 4, fp8=True),
    ScaledMmConfig(128, 256, 64, 64, 64, 64, 3, fp8=True),
)


def _buildable(cfg: ScaledMmConfig, sm: int) -> bool:
    limit = _SM_MAX_SMEM_BYTES.get(sm, 100 * 1024)
    if _smem_bytes(cfg.block_m, cfg.block_n, cfg.block_k, cfg.stages, dtype_bytes=1) > limit:
        return False
    if cfg.fp8 and (cfg.block_k != cfg.warp_k or cfg.stream_k):
        return False  # the summation order would depend on the tile
    return _epilogue_covers_warp(cfg.tile, element_bits=_EPILOGUE_ELEMENT_BITS)


#: The native production tiles, ``(block_m, block_n, cluster_m, cluster_n,
#: schedule)``, shared by both formats (``block_k`` is 128 throughout: one
#: 128-byte swizzle row of 8-bit operands, and on sm90 fp8 a divisor of the
#: 128-element promotion interval, so every tile sums in the same order).
#:
#: sm90: ping-pong at 64 rows, where most ASR GEMMs live (a decode step, a
#: streaming chunk, a short utterance) and the epilogue is a large share of a
#: shallow-K tile's life -- the other warpgroup's mainloop hides it; ping-pong at
#: 128 rows with a 2x1 cluster (TMA multicasts B to both CTAs) for mid M; and two
#: cooperative tiles for large M and N.  sm100: 1-SM tcgen05 tiles up to 128
#: rows, and 2-SM (CTA-pair) tiles of 256 rows for large M, each SM then loading
#: half of B.  **Not measured on either architecture** (see the module docstring).
_NATIVE_TILES: Dict[int, Tuple[Tuple[int, int, int, int, str], ...]] = {
    90: (
        (64, 64, 1, 1, "pingpong"),
        (64, 128, 1, 1, "pingpong"),
        (64, 256, 1, 1, "pingpong"),
        (128, 128, 2, 1, "pingpong"),
        (128, 256, 2, 1, "cooperative"),
        (256, 128, 2, 1, "cooperative"),
    ),
    100: (
        (64, 64, 1, 1, "1sm"),
        (64, 128, 1, 1, "1sm"),
        (128, 128, 1, 1, "1sm"),
        (128, 256, 1, 1, "1sm"),
        (256, 128, 2, 1, "2sm"),
        (256, 256, 2, 1, "2sm"),
    ),
}


def _native_buildable(cfg: NativeScaledMmConfig) -> bool:
    """Whether CUTLASS builds *cfg* -- ``CutlassScaledMmConfigNative``'s asserts, mirrored.

    One unbuildable variant fails its whole JIT module, so nothing that would
    trip an assert is ever rendered.
    """
    if cfg.schedule not in _NATIVE_SCHEDULES.get(cfg.sm, {}):
        return False
    if cfg.block_k != 128 or cfg.block_n % 16 or cfg.block_n > 256:
        return False
    if cfg.schedule == "pingpong":
        return cfg.block_m in (64, 128)
    if cfg.schedule == "cooperative":
        return cfg.block_m % 128 == 0
    if cfg.schedule == "1sm":
        return cfg.block_m in (64, 128)
    # 2sm: a CTA pair over an even cluster M.  The 16-bit 2-SM TMA epilogue also
    # needs N <= 128 or N % 64 == 0 (sm100_builder.inl); this space's 2-SM tiles
    # (N = 128, 256) meet it, and the check keeps a new one honest.
    return (
        cfg.block_m in (128, 256)
        and cfg.cluster_m % 2 == 0
        and (cfg.block_n <= 128 or cfg.block_n % 64 == 0)
    )


def _check_fmt(fmt: str) -> str:
    if fmt not in FORMATS:
        raise ValueError(f"scaled_mm operand format must be one of {FORMATS}, got {fmt!r}")
    return fmt


def lane_available(sm: int, fmt: str = "int8") -> bool:
    """Whether the *fmt* ``mma.sync`` lane serves *sm* at all."""
    return sm in _LANE_BY_SM[_check_fmt(fmt)]


def is_native_lane(sm: int, fmt: str = "int8") -> bool:
    """Whether the *fmt* ``mma.sync`` lane is *sm*'s native tensor-core path."""
    return _LANE_BY_SM[_check_fmt(fmt)].get(sm, False)


def _device_capability() -> Tuple[int, int]:
    """This process's compute capability (``OASR_CUDA_ARCH_LIST`` wins, as for the build)."""
    from .core import _get_cuda_arch

    major, minor = _get_cuda_arch()
    return int(major), int(minor)


def native_lane_available(sm: int, fmt: str = "int8") -> bool:
    """Whether *sm*'s native *fmt* lane exists and runs on this device.

    A family without one (everything but 90 and 100) says no, and so does any
    family this device is not (nothing native runs here, so the default stays
    ``mma.sync``); so does int8 on an sm100-family device other than compute
    capability 10.0 (:data:`_NATIVE_CAPABILITIES`).
    """
    from .core import sm_family

    key = (int(sm), _check_fmt(fmt))
    if key not in NATIVE_LANE_ID:
        return False
    cap = _device_capability()
    if sm_family(*cap) != key[0]:
        return False
    caps = _NATIVE_CAPABILITIES.get(key)
    return caps is None or cap in caps


def default_lane(sm: int | None = None, fmt: str = "int8") -> str:
    """The lane an unqualified call runs on *sm*: the native one where it runs, else ``mma.sync``."""
    sm = _get_target_sm() if sm is None else int(sm)
    if native_lane_available(sm, fmt):
        return NATIVE_LANE_ID[(sm, fmt)]
    return LANE_ID[_check_fmt(fmt)]


def is_native_lane_id(lane: str) -> bool:
    """Whether *lane* names a native (CUTLASS 3.x) lane."""
    return lane in _NATIVE_LANE_KEY


def lane_format(lane: str) -> str:
    """The operand format a lane id serves."""
    if lane in _NATIVE_LANE_KEY:
        return _NATIVE_LANE_KEY[lane][1]
    for fmt, mma_lane in LANE_ID.items():
        if lane == mma_lane:
            return fmt
    raise ValueError(f"unknown scaled_mm lane {lane!r}")


def _resolve_lane(sm: int, fmt: str, lane: Optional[str]) -> str:
    """*lane*, or *sm*'s default, after checking it serves *fmt* on *sm*'s family.

    Naming a lane does not consult the device (a config space is a function of
    the family alone -- what rendering and the tests need); only the *default*
    asks whether the native lane runs here.
    """
    if lane is None:
        return default_lane(sm, fmt)
    if lane_format(lane) != fmt:
        raise ValueError(f"lane {lane!r} serves {lane_format(lane)}, not {fmt}")
    if lane in _NATIVE_LANE_KEY and _NATIVE_LANE_KEY[lane][0] != sm:
        raise ValueError(f"lane {lane!r} is sm{_NATIVE_LANE_KEY[lane][0]}'s, not sm{sm}'s")
    return lane


def quant_n_alignment(lane: str) -> int:
    """N alignment the fused quantizing launch of *lane* needs (the functional unfuses below it)."""
    return NATIVE_QUANT_N_ALIGNMENT if is_native_lane_id(lane) else N_ALIGNMENT


@functools.lru_cache(maxsize=None)
def _production_configs(sm: int, fmt: str, lane: str) -> Tuple[AnyScaledMmConfig, ...]:
    if is_native_lane_id(lane):
        native = tuple(
            NativeScaledMmConfig(sm, bm, bn, cm, cn, schedule, fp8=fmt == "fp8")
            for bm, bn, cm, cn, schedule in _NATIVE_TILES[sm]
        )
        return tuple(cfg for cfg in native if _native_buildable(cfg))
    space = _FP8_PRODUCTION if fmt == "fp8" else _PRODUCTION + _STREAMK_PRODUCTION
    return tuple(cfg for cfg in space if _buildable(cfg, sm))


def get_production_configs(
    sm: int | None = None, fmt: str = "int8", lane: Optional[str] = None
) -> Dict[str, AnyScaledMmConfig]:
    """``{compile_name: config}`` for every *fmt* production variant of *lane* on *sm*.

    *lane* defaults to :func:`default_lane` -- the native lane on a Hopper or
    Blackwell data-center device, the ``mma.sync`` lane elsewhere.
    """
    sm = _get_target_sm() if sm is None else int(sm)
    fmt = _check_fmt(fmt)
    lane = _resolve_lane(sm, fmt, lane)
    return {cfg.compile_name: cfg for cfg in _production_configs(sm, fmt, lane)}


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _ctas(m: int, n: int, cfg: AnyScaledMmConfig) -> int:
    return -(-m // cfg.block_m) * -(-n // cfg.block_n)


def _select_int8(
    m: int, n: int, k: int, num_sms: int, built: Dict[str, AnyScaledMmConfig]
) -> ScaledMmConfig:
    for rung in _INT8_LADDER:
        if k < rung.min_k or (rung.max_k is not None and k >= rung.max_k):
            continue
        if 2 * m <= rung.tile.block_m:
            continue
        ctas = _ctas(m, n, rung.tile)
        if ctas < rung.min_ctas_per_sm * num_sms:
            continue
        pick = rung.sparse if rung.sparse is not None and ctas < num_sms else rung.tile
        if pick.compile_name in built:
            return pick
    return _T16


def _select_native(
    m: int, n: int, num_sms: int, built: Dict[str, AnyScaledMmConfig]
) -> AnyScaledMmConfig:
    """The native lanes' shape heuristic -- a starting point, **not fitted** to measurements.

    The largest tile (by area, then height) that is not mostly padding rows -- a
    tile above 64 rows at least twice as tall as M is skipped -- and that still
    launches half a wave (the fp8 ``mma.sync`` rule; a 2-SM tile occupies two
    SMs); else the smallest.  A measured tuning entry overrides it per shape; the
    int8 ``mma.sync`` ladder above is the model of what a fitted rule looks like.
    """
    by_size = sorted(built.values(), key=lambda c: (c.block_m * c.block_n, c.block_m), reverse=True)
    for cfg in by_size:
        if cfg.block_m > 64 and 2 * m <= cfg.block_m:
            continue
        if 2 * _ctas(m, n, cfg) * cfg.sms_per_tile >= num_sms:
            return cfg
    return by_size[-1]


@functools.lru_cache(maxsize=4096)
def _select(
    m: int, n: int, k: int, sm: int, num_sms: int, fmt: str = "int8", lane: Optional[str] = None
) -> AnyScaledMmConfig:
    lane = _resolve_lane(sm, fmt, lane)
    built = get_production_configs(sm, fmt, lane)
    if is_native_lane_id(lane):
        return _select_native(m, n, num_sms, built)
    if fmt == "int8":
        return _select_int8(m, n, k, num_sms, built)
    # fp8: the largest data-parallel tile that still launches half a wave of CTAs.
    by_area = sorted(
        (c for c in built.values() if not c.stream_k),
        key=lambda c: c.block_m * c.block_n,
        reverse=True,
    )
    for cfg in by_area:
        if 2 * _ctas(m, n, cfg) >= num_sms:
            return cfg
    return by_area[-1]


@functools.lru_cache(maxsize=1)
def _process_sm() -> int:
    """The SM family this process compiles the 8-bit modules for (resolved once).

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
    m: int,
    n: int,
    k: int,
    sm: int | None = None,
    num_sms: int | None = None,
    fmt: str = "int8",
    lane: Optional[str] = None,
) -> AnyScaledMmConfig:
    """The shape heuristic's pick, ignoring the tuning DB (what a build measures against)."""
    sm = _process_sm() if sm is None else int(sm)
    num_sms = _process_num_sms() if num_sms is None else int(num_sms)
    fmt = _check_fmt(fmt)
    return _select(int(m), int(n), int(k), sm, num_sms, fmt, _resolve_lane(sm, fmt, lane))


def select_config(
    m: int,
    n: int,
    k: int,
    sm: int | None = None,
    num_sms: int | None = None,
    fmt: str = "int8",
    lane: Optional[str] = None,
) -> AnyScaledMmConfig:
    """The *fmt* production variant of *lane* for a ``[m, k] @ [n, k]ᵀ`` problem.

    A measured tuning entry first (``scaled_mm|op=scaled_mm|N=..|K=..``, user tier
    then system tier, M rounded *up* to a region; see :func:`covering_tier`), else
    the shape heuristic -- never Stream-K (see ``_STREAMK_PRODUCTION``), which only
    a measured entry can name.  For int8 that is :data:`_INT8_LADDER` (the largest
    tile that fills enough of the GPU for its K, within 1 % of the best of 25
    tiles on five architectures' shapes); for fp8 the largest data-parallel tile
    that still launches at least half a wave of CTAs, else the smallest; for a
    native lane :func:`_select_native`.  *lane* defaults to :func:`default_lane`;
    an entry naming another lane's config is warned about once and skipped.

    A function of shape, device and the tuning-DB snapshot alone — never of
    capture state (AGENTS.md rule 11) — so an eager call and a graph replay of a
    shape run one kernel.  Memoised per ``(epoch, shape)``: the per-call cost is
    one dict lookup.
    """
    op = FORMAT_OPS[fmt][0]
    key = (_epoch(), op, m, n, k, sm, num_sms, lane)
    cfg = _MEMO.get(key)
    if cfg is None:
        cfg = _memoise(key, op, m, n, k, sm, num_sms, lane)
    return cfg


def select_quant_config(
    m: int,
    n: int,
    k: int,
    sm: int | None = None,
    num_sms: int | None = None,
    fmt: str = "int8",
    lane: Optional[str] = None,
) -> AnyScaledMmConfig:
    """The variant for a quantizing (8-bit-output) launch: data-parallel only.

    The producer of a GEMM chain (the FFN's ``w_1``) has a short K and a wide N,
    which is the data-parallel kernel's home; Stream-K is not built for it, so a
    tuning entry that names a Stream-K tile for ``op=scaled_mm_quant`` is warned
    about once and skipped.  Otherwise as :func:`select_config`, under its own
    signature key (the quantizing epilogue is measured separately).
    """
    op = FORMAT_OPS[fmt][1]
    key = (_epoch(), op, m, n, k, sm, num_sms, lane)
    cfg = _MEMO.get(key)
    if cfg is None:
        cfg = _memoise(key, op, m, n, k, sm, num_sms, lane)
    return cfg


def scaled_mm_func_name(cfg: AnyScaledMmConfig) -> str:
    """The TVM-FFI export name of a variant's fp16/bf16-output launcher."""
    return f"scaled_mm_{cfg.compile_name}"


def scaled_mm_quant_func_name(cfg: AnyScaledMmConfig) -> str:
    """The TVM-FFI export name of a data-parallel variant's 8-bit-output launcher."""
    if cfg.stream_k:
        raise ValueError(f"{cfg.compile_name}: Stream-K variants have no quantizing launcher")
    return f"scaled_mm_q8_{cfg.compile_name}"


#: The CUTLASS 2.x ``mma.sync`` lane of each format.  A tuning file records the
#: lane it was measured on (:func:`default_lane` when built); its entries name
#: configs of that lane, which another lane skips.
LANE_ID = {"int8": "sm80_int8_mma", "fp8": "sm89_fp8_mma"}

#: The int8 ``mma.sync`` lane's id.
SCALED_MM_LANE = LANE_ID["int8"]

#: The tuning-DB family a format's measured selections live in: one file per
#: format, each measured on its own lane.
TUNING_FAMILY = {"int8": "scaled_mm", "fp8": "scaled_mm_fp8"}

#: Per format, the census / tuning ops: the fp-output GEMM and its 8-bit-output
#: (quantizing) twin -- different epilogues, measured separately.
FORMAT_OPS = {
    "int8": ("scaled_mm", "scaled_mm_quant"),
    "fp8": ("scaled_mm_fp8", "scaled_mm_fp8_quant"),
}

_OP_FORMAT = {op: fmt for fmt, ops in FORMAT_OPS.items() for op in ops}


def op_format(op: str) -> str:
    """The operand format of a census / tuning op (``"int8"`` or ``"fp8"``)."""
    try:
        return _OP_FORMAT[op]
    except KeyError:
        raise ValueError(f"not a scaled_mm op: {op!r}") from None


def op_family(op: str) -> str:
    """The tuning-DB family a census / tuning op's entries live in."""
    return TUNING_FAMILY[op_format(op)]


def is_quant_op(op: str) -> bool:
    """Whether *op* is a quantizing (8-bit-output) launch."""
    return FORMAT_OPS[op_format(op)][1] == op


def config_id(cfg: AnyScaledMmConfig) -> str:
    """The id a tuning file stores a config under (its compile name)."""
    return cfg.compile_name


def config_to_params(cfg: AnyScaledMmConfig) -> Dict[str, object]:
    """A config as the explicit parameter dict a tuning file stores."""
    if isinstance(cfg, NativeScaledMmConfig):
        return {
            "kind": "cutlass",
            "lane": cfg.lane,
            "tile": [cfg.block_m, cfg.block_n, cfg.block_k],
            "cluster": [cfg.cluster_m, cfg.cluster_n],
            "schedule": cfg.schedule,
        }
    return {
        "kind": "cutlass",
        "lane": LANE_ID[cfg.fmt],
        "tile": [cfg.block_m, cfg.block_n, cfg.block_k],
        "warp": [cfg.warp_m, cfg.warp_n, cfg.warp_k],
        "stages": cfg.stages,
        "stream_k": bool(cfg.stream_k),
    }


def config_from_params(params: Dict[str, object]) -> AnyScaledMmConfig:
    """Inverse of :func:`config_to_params`."""
    lanes = {lane: fmt for fmt, lane in LANE_ID.items()}
    lane = params.get("lane")
    if params.get("kind") == "cutlass" and lane in _NATIVE_LANE_KEY:
        sm, fmt = _NATIVE_LANE_KEY[str(lane)]
        tile = [int(v) for v in params["tile"]]  # type: ignore[attr-defined]
        cluster = [int(v) for v in params["cluster"]]  # type: ignore[attr-defined]
        return NativeScaledMmConfig(
            sm=sm,
            block_m=tile[0],
            block_n=tile[1],
            block_k=tile[2],
            cluster_m=cluster[0],
            cluster_n=cluster[1],
            schedule=str(params["schedule"]),
            fp8=fmt == "fp8",
        )
    if params.get("kind") != "cutlass" or lane not in lanes:
        known = sorted(lanes) + sorted(_NATIVE_LANE_KEY)
        raise ValueError(f"not a scaled_mm config ({known}): {params}")
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
        fp8=lanes[str(params["lane"])] == "fp8",
    )


# =============================================================================
# The tuning-DB tier (oasr.tune.database): measured selections, then the heuristic
# =============================================================================
#
# ``oasr/tune/db/sm<family>/scaled_mm.json`` (system) and the user tier's file of
# the same name hold measured int8 selections keyed ``scaled_mm|op=<op>|N=<n>|K=<k>``,
# with M as the regions' dynamic axis, rounded *up* exactly as the GEMM family's
# are; ``scaled_mm_fp8.json`` holds the fp8 ones (``scaled_mm_fp8|op=scaled_mm_fp8...``).
# Every variant of a format computes the same bits (int8: an exact int32
# accumulator, Stream-K reducing int32 partials; fp8: one summation order for
# every tile), so an entry can only change speed, never a value.
#
# A region names a production config, or the ``"heuristic"`` sentinel: "measured,
# and the shape heuristic's own pick was the best within the noise here".  The
# build writes the sentinel rather than freezing the tile the heuristic happened
# to pick on the measuring card, so the entry follows the heuristic (and its SM
# count) wherever it is the right answer, and a signature where it always was is
# still *covered* -- what ``tuning.strict`` and ``ASREngine.tuning_report`` ask.

#: Every op a scaled_mm tuning file of either format keys.
SCALED_MM_OPS = FORMAT_OPS["int8"] + FORMAT_OPS["fp8"]

#: The config id (and ``{"kind": ...}``) of the "use the shape heuristic" sentinel.
HEURISTIC_ID = "heuristic"


def scaled_mm_sig_key(op: str, n: int, k: int) -> str:
    """The tuning-DB key of a scaled_mm static signature (in *op*'s format's family)."""
    from oasr.tune.database import sig_key

    return sig_key(op_family(op), op=op, N=int(n), K=int(k))


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
    """Identity of the 8-bit GEMM implementation a tuning result measured.

    The template every variant is rendered from plus the family's headers.  A
    soft validator (``oasr.tune.database``): a change marks a tuning file stale
    -- re-measure -- without discarding it.
    """
    from oasr.tune.database import hash_paths

    paths = [
        env.OASR_TEMPLATE_DIR / "scaled_mm_cutlass_template.cu.jinja",
        env.OASR_TEMPLATE_DIR / "scaled_mm_cutlass_template_sm90.cu.jinja",
    ]
    paths += sorted((env.OASR_INCLUDE_DIR / "oasr" / "scaled_mm").glob("*.h"))
    return hash_paths(paths)


def _scaled_mm_soft_validators() -> Dict[str, object]:
    return {"impl_hash": scaled_mm_impl_hash()}


#: ``(region m_hi | None, config id, decoded choice | None)``; ``None`` = undecodable.
_Rule = Tuple[Optional[int], str, object]

#: ``(epoch, sm, family) -> [(tier, {(op, N, K): [rule, ...]}), ...]``, highest tier first.
_TIER_VIEWS: Dict[
    Tuple[int, int, str], List[Tuple[str, Dict[Tuple[str, int, int], List[_Rule]]]]
] = {}

#: ``(epoch, op, m, n, k, sm arg, num_sms arg, lane arg) -> config``: the per-call path.
_MEMO: Dict[tuple, AnyScaledMmConfig] = {}

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


def _table_from_file(tf, family: str) -> Dict[Tuple[str, int, int], List[_Rule]]:
    from oasr.tune.database import parse_sig_key

    table: Dict[Tuple[str, int, int], List[_Rule]] = {}
    decoded: Dict[str, object] = {}
    for key, entry in tf.entries.items():
        key_family, fields = parse_sig_key(key)
        op = fields.get("op")
        if (
            key_family != family
            or op not in SCALED_MM_OPS
            or op_family(op) != family
            or not entry.regions
        ):
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


def _tier_views(
    sm: int, family: str = "scaled_mm"
) -> List[Tuple[str, Dict[Tuple[str, int, int], List[_Rule]]]]:
    """The decoded user and system tables of *family* for arch family *sm*, this epoch."""
    from oasr.tune import database

    epoch = database.epoch()
    key = (epoch, int(sm), family)
    views = _TIER_VIEWS.get(key)
    if views is not None:
        return views
    views = [
        (tier, _table_from_file(tf, family)) for tier, tf in database.tiers(family, sm).ordered()
    ]
    for stale in [k for k in _TIER_VIEWS if k[0] != epoch]:
        del _TIER_VIEWS[stale]  # only the current epoch is ever consulted
    _TIER_VIEWS[key] = views
    return views


def _region(rules: List[_Rule], m: int) -> Optional[_Rule]:
    for rule in rules:
        if rule[0] is None or m <= rule[0]:
            return rule
    return None


def _unusable_reason(op: str, choice, sm: int, lane: Optional[str] = None) -> Optional[str]:
    """Why *choice* cannot be dispatched for *op* on *sm*'s *lane* (``None``: it can)."""
    if choice == HEURISTIC_ID:
        return None
    if not isinstance(choice, (ScaledMmConfig, NativeScaledMmConfig)):
        return "cannot be decoded as a scaled_mm config"
    fmt = op_format(op)
    if choice.fmt != fmt:
        return f"is a {choice.fmt} config, not {fmt}"
    lane = _resolve_lane(sm, fmt, lane)
    if choice.lane != lane:
        return f"belongs to lane {choice.lane}, not {lane}"
    if get_production_configs(sm, fmt, lane).get(choice.compile_name) != choice:
        return f"is not in sm{sm}'s {lane} production set"
    if is_quant_op(op) and choice.stream_k:
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


def _lookup(op: str, m: int, n: int, k: int, sm: int, lane: Optional[str] = None):
    """``(tier, choice)`` of the first tier whose entry serves this shape on *lane*, or ``None``."""
    for tier, table in _tier_views(sm, op_family(op)):
        rules = table.get((op, n, k))
        if not rules:
            continue
        rule = _region(rules, m)
        if rule is None:  # above every bounded region: never measured there
            continue
        _, cid, choice = rule
        reason = _unusable_reason(op, choice, sm, lane)
        if reason is None:
            return tier, choice
        _note_unusable(tier, op, n, k, cid, reason, sm)
    return None


def _resolve(
    op: str, m: int, n: int, k: int, sm: int, num_sms: int, lane: str
) -> AnyScaledMmConfig:
    from oasr.tune.database import record_tier

    fmt, family = op_format(op), op_family(op)
    hit = _lookup(op, m, n, k, sm, lane)
    if hit is not None:
        tier, choice = hit
        record_tier(family, op, tier)
        if isinstance(choice, (ScaledMmConfig, NativeScaledMmConfig)):
            return choice
        return _select(m, n, k, sm, num_sms, fmt, lane)  # the heuristic sentinel
    record_tier(family, op, "heuristic")
    return _select(m, n, k, sm, num_sms, fmt, lane)


def _memoise(key: tuple, op: str, m, n, k, sm, num_sms, lane) -> AnyScaledMmConfig:
    sm_ = _process_sm() if sm is None else int(sm)
    num_sms_ = _process_num_sms() if num_sms is None else int(num_sms)
    lane_ = _resolve_lane(sm_, op_format(op), lane)
    cfg = _resolve(op, int(m), int(n), int(k), sm_, num_sms_, lane_)
    if len(_MEMO) >= _MEMO_MAX:
        _MEMO.clear()
    _MEMO[key] = cfg  # keyed by the epoch read *before* resolving: a racing reload strands it
    return cfg


def covering_tier(
    op: str, M: int, N: int, K: int, sm: int | None = None, lane: Optional[str] = None
) -> Optional[str]:
    """The tuning tier (``"user"`` / ``"system"``) that serves this shape, or ``None``.

    Covered means an entry exists for ``(op, N, K)``, one of its regions contains
    M, and the config it names can be dispatched here (a production config, or
    the heuristic sentinel) -- exactly what :func:`select_config` /
    :func:`select_quant_config` would serve from measured data.
    """
    if op not in SCALED_MM_OPS:
        return None
    sm = _process_sm() if sm is None else int(sm)
    hit = _lookup(op, int(M), int(N), int(K), sm, lane)
    return None if hit is None else hit[0]


def signature_tier(
    op: str, N: int, K: int, sm: int | None = None, lane: Optional[str] = None
) -> Optional[str]:
    """The highest tier with a servable entry for ``(op, N, K)`` at *some* M, or ``None``.

    What a per-layer report asks: a layer has a signature but no single M.
    """
    if op not in SCALED_MM_OPS:
        return None
    sm = _process_sm() if sm is None else int(sm)
    for tier, table in _tier_views(sm, op_family(op)):
        rules = table.get((op, int(N), int(K)))
        if rules and any(_unusable_reason(op, choice, sm, lane) is None for _, _, choice in rules):
            return tier
    return None


def _on_tuning_reload() -> None:
    _TIER_VIEWS.clear()
    _MEMO.clear()


def _register_with_tuning_db() -> Callable[[], int]:
    from oasr.tune import database

    for family in TUNING_FAMILY.values():
        database.register_soft_validator(family, _scaled_mm_soft_validators)
    database.register_reload_hook(_on_tuning_reload)
    return database.epoch


#: ``oasr.tune.database.epoch``, bound once: the per-call path runs no import statement.
_epoch: Callable[[], int] = _register_with_tuning_db()


def _render_variants(configs: Dict[str, AnyScaledMmConfig]) -> List:
    from .cubin_loader import write_if_different
    from .templates import render_template

    sm = _get_target_sm()
    paths = []
    for name, cfg in configs.items():
        if isinstance(cfg, NativeScaledMmConfig):
            paths.append(_render_native_variant(cfg))
            continue
        variant = f"scaled_mm_sm{sm}_{name}"
        rendered = render_template(
            "scaled_mm_cutlass_template.cu.jinja",
            op_name=variant,
            config_type="CutlassScaledMmFp8Config" if cfg.fp8 else "CutlassScaledMmConfig",
            fp8=cfg.fp8,
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


def _render_native_variant(cfg: NativeScaledMmConfig):
    from .cubin_loader import write_if_different
    from .templates import render_template

    variant = f"scaled_mm_{cfg.compile_name}"
    rendered = render_template(
        "scaled_mm_cutlass_template_sm90.cu.jinja",
        op_name=variant,
        fp8=cfg.fp8,
        func_name=scaled_mm_func_name(cfg),
        quant_func_name=scaled_mm_quant_func_name(cfg),
        tile_m=cfg.block_m,
        tile_n=cfg.block_n,
        tile_k=cfg.block_k,
        cluster_m=cfg.cluster_m,
        cluster_n=cfg.cluster_n,
        schedule=cfg.cpp_schedule,
        sm_version=cfg.sm,
    )
    path = env.OASR_GEN_SRC_DIR / "scaled_mm_native" / f"{variant}.cu"
    write_if_different(path, rendered)
    return path


#: The JIT module of each format's ``mma.sync`` lane.
MODULE_NAME = {"int8": "scaled_mm", "fp8": "scaled_mm_fp8"}

#: The JIT module of each native lane.
NATIVE_MODULE_NAME = {
    "sm90_int8_wgmma": "scaled_mm_sm90",
    "sm90_fp8_wgmma": "scaled_mm_fp8_sm90",
    "sm100_int8_umma": "scaled_mm_sm100",
    "sm100_fp8_umma": "scaled_mm_fp8_sm100",
}


def module_name(fmt: str = "int8", lane: Optional[str] = None) -> str:
    """The JIT module that holds *lane*'s *fmt* variants (default: this device's lane)."""
    lane = _resolve_lane(_get_target_sm(), _check_fmt(fmt), lane)
    return NATIVE_MODULE_NAME[lane] if is_native_lane_id(lane) else MODULE_NAME[fmt]


def gen_scaled_mm_module(fmt: str = "int8", lane: Optional[str] = None) -> JitSpec:
    """JIT spec for every production *fmt* scaled-GEMM variant of *lane*, one TU each.

    *lane* defaults to :func:`default_lane`.  A native module is its own JIT
    module, so its variants and the ``mma.sync`` ones never share a library
    (CUTLASS's one-time statics are GNU-unique across ``.so`` files).  int8 on
    sm100 is built for ``sm_100a`` rather than the family target
    (:data:`_NATIVE_GENCODE`).
    """
    sm = _get_target_sm()
    fmt = _check_fmt(fmt)
    lane = _resolve_lane(sm, fmt, lane)
    sources = _render_variants(get_production_configs(sm, fmt, lane))
    gencode = _NATIVE_GENCODE.get(_NATIVE_LANE_KEY[lane]) if is_native_lane_id(lane) else None
    return gen_jit_spec(module_name(fmt, lane), sources, gencode_target=gencode)


def gen_scaled_mm_fp8_modules() -> List[JitSpec]:
    """The fp8 module, or nothing on an SM with no fp8 lane (for the AOT list)."""
    return [gen_scaled_mm_module("fp8")] if lane_available(_get_target_sm(), "fp8") else []
