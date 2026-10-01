# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
#
# FlashInfer-style autotuner for OASR kernels.
#
# Single-file implementation following FlashInfer's autotuner.py pattern.
# Consolidates types, registry, cache, profiling, and orchestration.
#
# Reference: flashinfer/autotuner.py

import contextlib
import functools
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger("oasr.tune")

_METADATA_KEY = "_metadata"


# =============================================================================
# JSON serialization helpers
# =============================================================================


def _tactic_to_json(tactic):
    """Convert a tactic value to a JSON-compatible format.

    Any iterable (tuples, lists, TVM FFI Array objects, etc.) is recursively
    converted to plain Python lists so that ``json.dump`` can serialize them.
    Scalars (int, float, bool, None) are returned as-is.
    """
    if isinstance(tactic, (tuple, list)):
        return [_tactic_to_json(v) for v in tactic]
    if hasattr(tactic, "__iter__") and not isinstance(tactic, (str, bytes, dict)):
        return [_tactic_to_json(v) for v in tactic]
    if isinstance(tactic, bool):
        return tactic
    if isinstance(tactic, int):
        return int(tactic)
    return tactic


def _json_to_tactic(val):
    """Convert a JSON-deserialized tactic value back to its original format.

    Lists are recursively converted to tuples so that compound tactics
    are restored to their expected tuple form.
    """
    if isinstance(val, list):
        return tuple(_json_to_tactic(v) for v in val)
    return val


# =============================================================================
# Environment metadata
# =============================================================================


def _rebucket_v1_entries(entries: Dict[str, Any]) -> Dict[str, Any]:
    """Re-key a version-1 cache (exact shapes) onto the bucketed keys lookups use.

    Where several exact shapes land in one bucket, the largest measured shape
    wins: a result is only ever applied at or below the size it was measured at.
    """
    from .shapes import bucket_shape_sig

    out: Dict[str, Any] = {}
    best_size: Dict[str, Tuple[int, ...]] = {}
    for key_str, value in entries.items():
        try:
            key = ProfileKey.from_str(key_str)
        except (ValueError, TypeError):
            out[key_str] = value
            continue
        bucketed = ProfileKey(
            op_key=key.op_key,
            shape_sig=bucket_shape_sig(key.op_key.family, key.op_key.op, key.shape_sig),
            dtype=key.dtype,
            device_sm=key.device_sm,
        ).to_str()
        if bucketed not in best_size or key.shape_sig > best_size[bucketed]:
            best_size[bucketed] = key.shape_sig
            out[bucketed] = value
    return out


def _collect_metadata() -> Dict[str, str]:
    """Collect environment metadata that can affect tactic-to-kernel mappings."""
    meta: Dict[str, str] = {}
    try:
        import oasr

        meta["oasr_version"] = oasr.__version__
    except Exception:
        meta["oasr_version"] = "unknown"
    meta["cuda_version"] = getattr(torch.version, "cuda", None) or "unknown"
    try:
        meta["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception:
        meta["cudnn_version"] = "unknown"
    try:
        meta["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        meta["gpu"] = "unknown"
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        meta["sm"] = str(props.major * 10 + props.minor)
    except Exception:
        pass
    return meta


def _same_call(call: Callable) -> Callable[[int], Callable]:
    """An :class:`~oasr.tune.bench.Arm` ``make_call`` that ignores the copy index.

    The registry's runners take their operands as arguments, so every graph node
    replays the same buffers -- the autotuner times the caller's own tensors.
    """
    return lambda _i: call


# =============================================================================
# Core data types
# =============================================================================


@dataclass(frozen=True)
class OpKey:
    """Identifies an operation type.

    Attributes:
        family: Kernel family, e.g. ``"gemm"``, ``"conv"``, ``"norm"``.
        op: Operation name within the family, e.g. ``"gemm"``, ``"conv2d"``.
        flags: Extra frozen flags that affect kernel choice,
               e.g. ``("activation_type=2",)``.
    """

    family: str
    op: str
    flags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileKey:
    """Cache lookup key combining operation, shape, dtype, and device arch.

    Attributes:
        op_key: The operation identifier.
        shape_sig: Canonicalized shape signature (operation-specific).
                   For GEMM: ``(M, N, K)``.
                   For Conv2D: ``(N, H, W, IC, K, R, S, stride_h, ...)``.
        dtype: Data type string, e.g. ``"float16"``, ``"bfloat16"``.
        device_sm: GPU SM version, e.g. ``80``, ``86``, ``90``.
    """

    op_key: OpKey
    shape_sig: Tuple[int, ...]
    dtype: str
    device_sm: int

    def to_str(self) -> str:
        """Stable, deterministic string for use as a JSON/dict key."""
        flags_part = ",".join(self.op_key.flags) if self.op_key.flags else ""
        shape_part = ",".join(str(s) for s in self.shape_sig)
        parts = [
            self.op_key.family,
            self.op_key.op,
            flags_part,
            f"({shape_part})",
            self.dtype,
            f"sm{self.device_sm}",
        ]
        return "|".join(parts)

    @classmethod
    def from_str(cls, s: str) -> "ProfileKey":
        """Parse a ``ProfileKey`` from its ``to_str()`` representation."""
        parts = s.split("|")
        family, op, flags_str, shape_str, dtype, sm_str = parts
        flags = tuple(flags_str.split(",")) if flags_str else ()
        shape_inner = shape_str.strip("()")
        shape_sig = tuple(int(x) for x in shape_inner.split(",")) if shape_inner else ()
        device_sm = int(sm_str.replace("sm", ""))
        return cls(
            op_key=OpKey(family=family, op=op, flags=flags),
            shape_sig=shape_sig,
            dtype=dtype,
            device_sm=device_sm,
        )


@dataclass(frozen=True)
class Tactic:
    """A specific backend + configuration combination.

    Attributes:
        backend: Backend name, e.g. ``"cutlass"``, ``"cudnn"``.
        config: Frozen key-value pairs for backend-specific knobs,
                e.g. ``(("tile_m", 128), ("tile_n", 128))``.
    """

    backend: str
    config: Tuple[Tuple[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "backend": self.backend,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tactic":
        """Deserialize from a dict."""
        config = tuple(sorted(d.get("config", {}).items()))
        return cls(backend=d["backend"], config=config)


@dataclass
class TuneResult:
    """Profiling result for one tactic.

    Attributes:
        tactic: The tactic that was profiled.
        median_ms: Median execution time in milliseconds.
        min_ms: Minimum execution time in milliseconds.
        max_ms: Maximum execution time in milliseconds.
        status: One of ``"ok"``, ``"error"``, ``"unsupported"``.
        error_msg: Error message if ``status != "ok"``.
    """

    tactic: Tactic
    median_ms: float
    min_ms: float
    max_ms: float
    status: str
    error_msg: str = ""


# =============================================================================
# Backend registry
# =============================================================================


@dataclass
class BackendEntry:
    """A registered backend for an operation.

    Attributes:
        tactic: The tactic this entry represents.
        is_available: Callable returning ``True`` if the backend can run
                      on the current system (e.g. cuDNN is installed).
        get_runner: Callable returning the kernel function. The returned
                    callable must accept the same arguments as the TVM-FFI
                    module call (e.g. ``runner(out, input, filter, ...)``).
        is_fallback: If ``True``, this entry is used when no tuned config
                     exists and tuning mode is off.
    """

    tactic: Tactic
    is_available: Callable[[], bool]
    get_runner: Callable[[], Callable]
    is_fallback: bool = False


class BackendRegistry:
    """Registry mapping operations to candidate backend entries."""

    def __init__(self) -> None:
        self._entries: Dict[OpKey, List[BackendEntry]] = {}
        self._lock = threading.Lock()

    def register(self, op_key: OpKey, entry: BackendEntry) -> None:
        """Register a backend entry for an operation."""
        with self._lock:
            if op_key not in self._entries:
                self._entries[op_key] = []
            self._entries[op_key].append(entry)

    def get_candidates(self, op_key: OpKey) -> List[BackendEntry]:
        """Return all *available* backend entries for an operation."""
        entries = self._entries.get(op_key, [])
        return [e for e in entries if e.is_available()]

    def get_fallback(self, op_key: OpKey) -> Optional[BackendEntry]:
        """Return the fallback entry for an operation, or ``None``."""
        for entry in self._entries.get(op_key, []):
            if entry.is_fallback and entry.is_available():
                return entry
        candidates = self.get_candidates(op_key)
        return candidates[0] if candidates else None

    def get_entry_for_tactic(self, op_key: OpKey, tactic: Tactic) -> Optional[BackendEntry]:
        """Find the entry matching a specific tactic, or ``None``."""
        for entry in self._entries.get(op_key, []):
            if entry.tactic == tactic and entry.is_available():
                return entry
        return None


# Module-level global registry, populated by backend modules on import.
_global_registry = BackendRegistry()


# =============================================================================
# Statistics (FlashInfer-style)
# =============================================================================


@dataclass
class AutoTunerStatistics:
    """Statistics collected by the AutoTuner.

    Attributes:
        cache_misses: Number of cache misses requiring fallback or profiling.
        tuned_op_total_configs: Total configurations tried per operation.
        tuned_op_successful_configs: Successful configurations per operation.
        failed_profiling_count: Number of failed profiling attempts per operation.
    """

    cache_misses: int = 0
    tuned_op_total_configs: Dict[str, int] = field(default_factory=dict)
    tuned_op_successful_configs: Dict[str, int] = field(default_factory=dict)
    failed_profiling_count: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        stats_str = f"Cache misses: {self.cache_misses}\n"
        if self.tuned_op_total_configs:
            stats_str += "Tuned operations:\n"
            for op in sorted(self.tuned_op_total_configs.keys()):
                total = self.tuned_op_total_configs[op]
                successful = self.tuned_op_successful_configs.get(op, 0)
                failed = self.failed_profiling_count.get(op, 0)
                success_rate = (successful / total * 100) if total > 0 else 0
                stats_str += f"  {op}:\n"
                stats_str += f"    - Total configs tried: {total}\n"
                stats_str += f"    - Successful configs: {successful}\n"
                stats_str += f"    - Failed profiling count: {failed}\n"
                stats_str += f"    - Success rate: {success_rate:.1f}%\n"
        return stats_str


# =============================================================================
# Helper functions
# =============================================================================


def _dtype_str(dtype: torch.dtype) -> str:
    """Convert a ``torch.dtype`` to a short string."""
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))


def _device_sm(device: torch.device) -> int:
    """Return the SM version for *device*."""
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


def _bench_cuda_events(
    fn: Callable, args: tuple, warmup: int, rep: int
) -> Tuple[float, float, float]:
    """Benchmark *fn* using CUDA events. Returns (median, min, max) in ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    timings: List[float] = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    timings.sort()
    median = timings[len(timings) // 2]
    return median, timings[0], timings[-1]


def _publish_to_tuning_db(profile_key: "ProfileKey", best: "TuneResult", results) -> None:
    """Record a GEMM-family winner in the user tier (``oasr.tune.database``).

    That is what makes a tuning run pay off outside ``autotune()``: the
    production selector reads the user tier first.  Keyed at the exact dtype it
    was measured in, at the bucket of M it was measured for.
    """
    op_key = profile_key.op_key
    if op_key.family != "gemm" or op_key.op not in (
        "gemm",
        "gemm_activation",
        "gemm_log_softmax",
        "bmm",
    ):
        return
    if profile_key.dtype not in ("float16", "bfloat16"):
        return
    try:
        import oasr.jit.gemm as jg
        from oasr.tune import database
        from oasr.tune.gemm_tune import choice_of

        sm = profile_key.device_sm
        from oasr.jit.core import _SM_FAMILY

        family_sm = _SM_FAMILY.get(sm, sm)
        if op_key.op == "bmm":
            _batch, m, n, k = profile_key.shape_sig
        else:
            m, n, k = profile_key.shape_sig
        choice = choice_of(best.tactic, family_sm)
        cid = jg.gemm_config_id(choice)
        dt = "fp16" if profile_key.dtype == "float16" else "bf16"
        timings = {
            (
                r.tactic.backend
                if r.tactic.backend != "cutlass"
                else jg.gemm_config_id(choice_of(r.tactic, family_sm))
            ): [round(r.median_ms, 6)]
            for r in results[:5]
            if r.status == "ok"
        }
        database.record_point(
            "gemm",
            family_sm,
            jg.GEMM_LANE_BY_SM.get(family_sm, "sm80_mma"),
            jg.gemm_sig_key(op_key.op, n, k, dt),
            m,
            cid,
            jg.gemm_config_to_params(choice),
            timings,
        )
    except Exception as exc:  # noqa: BLE001 -- publishing must never break a tuning run
        logger.debug(
            "[Autotuner]: could not publish %s to the tuning DB: %s", profile_key.to_str(), exc
        )


# =============================================================================
# AutoTuner — singleton, FlashInfer-style
# =============================================================================


class AutoTuner:
    """AutoTuner for optimizing OASR kernel operations.

    Handles automatic performance tuning by profiling different implementations
    and caching the best performing configurations. Follows the FlashInfer
    AutoTuner pattern with a singleton instance.

    Args:
        warmup: Number of warmup iterations before profiling (default: 25).
        repeat: Number of profiling iterations for averaging (default: 100).
    """

    _instance: Optional["AutoTuner"] = None
    _class_lock = threading.Lock()

    def __init__(self, warmup: int = 25, repeat: int = 100) -> None:
        self.warmup = warmup
        self.repeat = repeat
        self.is_tuning_mode = False
        self._active_tuning_contexts = 0

        # Reentrant lock protecting all mutable state on this instance.
        self._lock = threading.RLock()

        # In-memory profiling cache: profile_key_str → Tactic
        self.profiling_cache: Dict[str, Tactic] = {}
        # Timing data for profiled entries
        self._timings: Dict[str, float] = {}
        # User-loaded configs from JSON files
        self._file_configs: Dict[str, Tactic] = {}
        # Bundled default configs
        self._bundled_configs: Dict[str, Tactic] = {}
        # Track which file config keys have been logged
        self._logged_file_hits: Set[str] = set()
        # Dirty flag for save-on-exit
        self._dirty = False
        self._dirty_seq = 0

        # Statistics tracking
        self.stats = AutoTunerStatistics()

        # Write GEMM-family winners into the tuning DB's user tier, where the
        # production (non-autotuned) dispatch reads them too.
        self.publish = True

        # Backend registry
        self._registry = _global_registry

    @classmethod
    def get(cls) -> "AutoTuner":
        """Return the singleton AutoTuner instance, creating it lazily."""
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    _ensure_backends_registered()
                    cls._instance = AutoTuner()
        return cls._instance

    # -----------------------------------------------------------------
    # Cache lookup (FlashInfer search_cache pattern)
    # -----------------------------------------------------------------

    def search_cache(self, profile_key: ProfileKey) -> Optional[Tactic]:
        """Search for cached profiling results matching the profile key.

        Searches the following sources in priority order:
            1. In-memory profiling_cache (from live autotuning)
            2. User-loaded configs (via load_configs or autotune(cache=...))
            3. Bundled package configs
            4. None (fallback)
        """
        key_str = profile_key.to_str()
        with self._lock:
            # 1. In-memory cache (from live tuning)
            tactic = self.profiling_cache.get(key_str)
            if tactic is not None:
                return tactic

            # 2. User-loaded configs (from load_configs or autotune(cache=...))
            tactic = self._file_configs.get(key_str)
            if tactic is not None:
                if key_str not in self._logged_file_hits:
                    self._logged_file_hits.add(key_str)
                    logger.info(
                        "[Autotuner]: Config cache hit for %s (source=config file)",
                        key_str,
                    )
                return tactic

            # 3. Bundled package configs
            tactic = self._bundled_configs.get(key_str)
            if tactic is not None:
                return tactic

        # 4. No cached result
        return None

    # -----------------------------------------------------------------
    # Main dispatch (OASR-specific, wrapping FlashInfer choose_one)
    # -----------------------------------------------------------------

    def dispatch(
        self,
        op_key: OpKey,
        shape_sig: tuple,
        dtype: torch.dtype,
        device: torch.device,
        runner_args: tuple,
    ) -> None:
        """Select the best tactic and execute it.

        Steps:
        1. Build ``ProfileKey`` from inputs.
        2. Look up in cache — if hit, execute cached tactic.
        3. If miss and ``is_tuning_mode=True``: profile all candidates,
           cache the best, and execute it.
        4. If miss and ``is_tuning_mode=False``: use fallback tactic.

        Args:
            op_key: Operation identifier.
            shape_sig: Shape signature tuple (operation-specific).
            dtype: Input data type.
            device: CUDA device.
            runner_args: Positional arguments to pass to the runner.
        """
        from .shapes import bucket_shape_sig

        # Dynamic dimensions (M for a GEMM, batch and length for a conv) key by
        # bucket, not by value: an exact key re-profiled every candidate for
        # every new M, on the call path.  The bucket rounds up, so a result is
        # only applied at or below the size it was measured at.
        profile_key = ProfileKey(
            op_key=op_key,
            shape_sig=bucket_shape_sig(op_key.family, op_key.op, shape_sig),
            dtype=_dtype_str(dtype),
            device_sm=_device_sm(device),
        )

        # 1. Cache lookup
        tactic = self.search_cache(profile_key)
        if tactic is not None:
            logger.debug("Cache hit for %s -> %s", profile_key.to_str(), tactic.backend)
            self._execute(op_key, tactic, runner_args)
            return

        # 2. Cache miss
        self.stats.cache_misses += 1

        if self.is_tuning_mode:
            tactic = self._profile_and_cache(op_key, profile_key, runner_args)
        else:
            tactic = self._fallback_tactic(op_key, profile_key)

        if tactic is not None:
            self._execute(op_key, tactic, runner_args)
        else:
            raise RuntimeError(
                f"No available backend for {op_key}. "
                "Ensure at least one backend is registered and available."
            )

    # -----------------------------------------------------------------
    # Profiling (merged from _profiler.py)
    # -----------------------------------------------------------------

    def _profile_single_kernel(self, tactic: Tactic, runner: Callable, args: tuple) -> TuneResult:
        """Eager CUDA-event timing of one tactic -- the fallback protocol.

        Used only when an op cannot be graph-captured at all (every candidate
        failed capture): graph-captured loops are the protocol
        (:mod:`oasr.tune.bench`), because eager timing cannot resolve a kernel
        faster than its own issue cost.
        """
        try:
            median, min_ms, max_ms = _bench_cuda_events(runner, args, self.warmup, self.repeat)
            return TuneResult(
                tactic=tactic, median_ms=median, min_ms=min_ms, max_ms=max_ms, status="ok"
            )
        except Exception as exc:
            logger.debug("Tactic %s failed: %s", tactic, exc)
            return TuneResult(
                tactic=tactic,
                median_ms=float("inf"),
                min_ms=float("inf"),
                max_ms=float("inf"),
                status="error",
                error_msg=str(exc),
            )

    def _profile_candidates(
        self,
        op_key: OpKey,
        candidates: List[BackendEntry],
        args: tuple,
    ) -> List[TuneResult]:
        """Profile all candidates and return results sorted by median_ms.

        Measured with :func:`oasr.tune.bench.measure` -- graph-captured loops,
        interleaved rounds, successive halving -- on the live arguments.  An op
        none of whose candidates can be captured is timed eagerly instead.
        """
        from . import bench

        op_str = f"{op_key.family}.{op_key.op}"
        results: List[TuneResult] = []
        arms = []
        runners: Dict[str, Tuple[BackendEntry, Callable]] = {}
        for i, entry in enumerate(candidates):
            try:
                runner = entry.get_runner()
            except Exception as exc:
                logger.debug("Failed to get runner for %s: %s", entry.tactic, exc)
                self.stats.failed_profiling_count[op_str] = (
                    self.stats.failed_profiling_count.get(op_str, 0) + 1
                )
                results.append(
                    TuneResult(
                        tactic=entry.tactic,
                        median_ms=float("inf"),
                        min_ms=float("inf"),
                        max_ms=float("inf"),
                        status="error",
                        error_msg=str(exc),
                    )
                )
                continue
            name = f"c{i}"
            runners[name] = (entry, runner)
            arms.append(
                bench.Arm(
                    name,
                    _same_call(functools.partial(runner, *args)),
                    forced=entry.is_fallback,
                    payload=entry,
                )
            )
        measured, _cond = bench.measure(arms)
        if arms and all(m.status != "ok" for m in measured.values()):
            logger.info("[Autotuner]: %s is not graph-capturable; timing eagerly", op_str)
            for entry, runner in runners.values():
                results.append(self._profile_single_kernel(entry.tactic, runner, args))
        else:
            for name, m in measured.items():
                entry = runners[name][0]
                ok = m.status == "ok"
                results.append(
                    TuneResult(
                        tactic=entry.tactic,
                        median_ms=m.median_ms if ok else float("inf"),
                        min_ms=min(m.samples) if (ok and m.samples) else float("inf"),
                        max_ms=max(m.samples) if (ok and m.samples) else float("inf"),
                        status="ok" if ok else "error",
                        error_msg=m.error,
                    )
                )
        for r in results:
            if r.status == "ok":
                logger.info("  %s: %.4f ms", r.tactic.backend, r.median_ms)
            else:
                self.stats.failed_profiling_count[op_str] = (
                    self.stats.failed_profiling_count.get(op_str, 0) + 1
                )
        results.sort(key=lambda r: r.median_ms)
        return results

    @staticmethod
    def _select_best(results: List[TuneResult]) -> Optional[TuneResult]:
        """Return the fastest result with ``status='ok'``, or ``None``."""
        for r in results:
            if r.status == "ok":
                return r
        return None

    # -----------------------------------------------------------------
    # Internal dispatch helpers
    # -----------------------------------------------------------------

    def _profile_and_cache(
        self,
        op_key: OpKey,
        profile_key: ProfileKey,
        runner_args: tuple,
    ) -> Optional[Tactic]:
        """Profile all candidates and cache the best result.

        Uses double-check locking to handle concurrent profiling requests.
        """
        with self._lock:
            # Double-check: another thread may have profiled while we waited.
            cached = self.search_cache(profile_key)
            if cached is not None:
                return cached

            candidates = self._registry.get_candidates(op_key)
            if not candidates:
                logger.warning("No candidates for %s", profile_key.to_str())
                return self._fallback_tactic(op_key, profile_key)

            op_str = f"{op_key.family}.{op_key.op}"
            self.stats.tuned_op_total_configs[op_str] = len(candidates)

            logger.info(
                "[Autotuner]: Profiling %d candidate(s) for %s ...",
                len(candidates),
                profile_key.to_str(),
            )
            results = self._profile_candidates(op_key, candidates, runner_args)
            best = self._select_best(results)

            if best is not None:
                key_str = profile_key.to_str()
                self.profiling_cache[key_str] = best.tactic
                self._timings[key_str] = best.median_ms
                self._dirty = True
                self._dirty_seq += 1
                self.stats.tuned_op_successful_configs[op_str] = (
                    self.stats.tuned_op_successful_configs.get(op_str, 0) + 1
                )
                logger.info(
                    "[Autotuner]: Selected %s (%.4f ms) for %s",
                    best.tactic.backend,
                    best.median_ms,
                    profile_key.to_str(),
                )
                if self.publish:
                    _publish_to_tuning_db(profile_key, best, results)
                return best.tactic

            logger.warning(
                "[Autotuner]: All candidates failed for %s, using fallback",
                profile_key.to_str(),
            )
            return self._fallback_tactic(op_key, profile_key)

    def _fallback_tactic(self, op_key: OpKey, profile_key: ProfileKey) -> Optional[Tactic]:
        """Return the fallback tactic for *op_key*."""
        entry = self._registry.get_fallback(op_key)
        if entry is not None:
            logger.debug(
                "Using fallback %s for %s",
                entry.tactic.backend,
                profile_key.to_str(),
            )
            return entry.tactic
        return None

    def _execute(
        self,
        op_key: OpKey,
        tactic: Tactic,
        runner_args: tuple,
    ) -> None:
        """Execute *tactic* for *op_key* with the given arguments."""
        entry = self._registry.get_entry_for_tactic(op_key, tactic)
        if entry is None:
            # Tactic was cached but backend is no longer available. Fallback.
            entry = self._registry.get_fallback(op_key)
            if entry is None:
                raise RuntimeError(
                    f"Cached tactic {tactic} for {op_key} is unavailable " "and no fallback exists."
                )
            logger.warning(
                "Cached tactic %s unavailable, falling back to %s",
                tactic.backend,
                entry.tactic.backend,
            )
        runner = entry.get_runner()
        runner(*runner_args)

    # -----------------------------------------------------------------
    # Config persistence (FlashInfer save_configs / load_configs)
    # -----------------------------------------------------------------

    def save_configs(self, path: str) -> None:
        """Save the current profiling cache to a JSON file.

        Serializes all cached (tactic) results so they can be loaded later
        via ``load_configs()`` or ``autotune(cache=...)``, avoiding the need
        to re-run autotuning.

        When configs were previously loaded via ``load_configs()``, those
        entries are included in the output as well (with in-memory profiling
        results taking priority for overlapping keys).

        Args:
            path: File path to write the JSON config to.
        """
        abs_path = os.path.abspath(path)

        with self._lock:
            seq_at_snapshot = self._dirty_seq
            entries: Dict[str, Dict[str, Any]] = {}

            # Include previously loaded file configs as a base
            for key_str, tactic in self._file_configs.items():
                entries[key_str] = tactic.to_dict()

            num_previous = len(entries)

            # Overlay in-memory profiling results (take priority)
            for key_str, tactic in self.profiling_cache.items():
                entry = tactic.to_dict()
                if key_str in self._timings:
                    entry["median_ms"] = round(self._timings[key_str], 6)
                entry["profiled_at"] = datetime.now(timezone.utc).isoformat()
                entries[key_str] = entry

        current_meta = _collect_metadata()

        # Re-read from disk and merge to reduce lost updates from concurrent saves
        original_metadata = None
        try:
            with open(abs_path, "r") as f:
                disk_data = json.load(f)
            original_metadata = disk_data.pop(_METADATA_KEY, None)
            disk_entries = disk_data.get("entries", disk_data)
            # If old format (flat dict), treat entire file as entries
            if "entries" not in disk_data and _METADATA_KEY not in disk_data:
                disk_entries = disk_data
            for k, v in disk_entries.items():
                if k not in entries and k != _METADATA_KEY:
                    entries[k] = v
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        num_new = len(entries) - num_previous

        # Atomic write
        dir_name = os.path.dirname(abs_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", prefix=".autotuner_")
        try:
            data = {
                _METADATA_KEY: original_metadata or current_meta,
                "version": 2,
                "entries": dict(sorted(entries.items())),
            }
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, abs_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        with self._lock:
            if self._dirty_seq == seq_at_snapshot:
                self._dirty = False

        logger.info(
            "[Autotuner]: Saved %d configs to %s (%d new, %d from previous config)",
            len(entries),
            path,
            num_new,
            num_previous,
        )

    def load_configs(self, path: str) -> bool:
        """Load autotuner configs from a JSON file.

        Populates the internal config lookup table so that ``search_cache()``
        can return pre-tuned results without re-running autotuning.

        If the file contains metadata that does not match the current
        environment, the cache is **warned about** but still loaded (OASR
        uses a softer policy than FlashInfer here).

        Args:
            path: File path to the JSON config file.

        Returns:
            True if configs were loaded successfully, False if skipped.
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            return False

        try:
            with open(abs_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Autotuner]: Failed to load config %s: %s", path, exc)
            return False

        saved_meta = data.pop(_METADATA_KEY, data.get("env"))
        if saved_meta is not None:
            current_meta = _collect_metadata()
            mismatches = {
                k: (saved_meta.get(k), current_meta.get(k))
                for k in current_meta
                if saved_meta.get(k) not in (current_meta.get(k), "*", None)
            }
            if "sm" in mismatches:
                # Hard: a tactic tuned on another architecture is not a choice
                # here at all -- it may name a kernel this build does not have.
                logger.warning(
                    "[Autotuner]: Ignoring cache file %s: tuned on sm%s, running on sm%s.",
                    path,
                    mismatches["sm"][0],
                    mismatches["sm"][1],
                )
                return False
            if mismatches:
                details = ", ".join(
                    f"{k}: saved={old} vs current={new}" for k, (old, new) in mismatches.items()
                )
                logger.warning(
                    "[Autotuner]: Cache file %s has environment differences (%s). "
                    "Cached configs may not be optimal.",
                    path,
                    details,
                )

        entries = data.get("entries", {})
        # Handle legacy format (no "entries" wrapper)
        if not entries and "version" not in data:
            entries = {k: v for k, v in data.items() if k not in (_METADATA_KEY, "env")}

        if int(data.get("version", 1)) < 2:
            entries = _rebucket_v1_entries(entries)

        with self._lock:
            self._file_configs.clear()
            self._logged_file_hits.clear()
            for key_str, entry_data in entries.items():
                if isinstance(entry_data, dict):
                    self._file_configs[key_str] = Tactic.from_dict(entry_data)
                elif isinstance(entry_data, list) and len(entry_data) >= 2:
                    # FlashInfer format: [runner_name, tactic_value]
                    self._file_configs[key_str] = Tactic(
                        backend=str(entry_data[0]),
                        config=(
                            _json_to_tactic(entry_data[1])
                            if isinstance(entry_data[1], (list, dict))
                            else ()
                        ),
                    )

        logger.info("[Autotuner]: Loaded %d configs from %s", len(entries), path)
        return True

    def load_bundled(self, path: str) -> None:
        """Load bundled default configs (shipped with the package)."""
        if not os.path.isfile(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Autotuner]: Failed to load bundled configs %s: %s", path, exc)
            return
        entries = data.get("entries", {})
        with self._lock:
            for key_str, entry_data in entries.items():
                if key_str not in self._bundled_configs:
                    self._bundled_configs[key_str] = Tactic.from_dict(entry_data)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the profiling cache and user-loaded file configs."""
        with self._lock:
            self.profiling_cache.clear()
            self._timings.clear()
            self._file_configs.clear()
            self._logged_file_hits.clear()
            self._dirty = False
            self._dirty_seq = 0

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = AutoTunerStatistics()

    @property
    def registry(self) -> BackendRegistry:
        """Access the backend registry."""
        return self._registry


# =============================================================================
# Context manager (FlashInfer autotune pattern)
# =============================================================================


@contextlib.contextmanager
def autotune(
    tune_mode: bool = True,
    *,
    cache: Optional[str] = None,
    warmup: int = 25,
    rep: int = 100,
    log_level: str = "INFO",
    publish: bool = True,
):
    """Context manager for autotuning with optional file-based caching.

    Args:
        tune_mode: If True, profile uncovered shapes during execution.
            If False, only use cached/loaded configs (no profiling).
        cache: Optional path to a JSON config file.
            On entry, configs are loaded from this file (if it exists).
            On exit, configs are saved back to this file (only when
            ``tune_mode=True`` and new results were profiled).
        warmup: Warm-up iterations for the eager fallback timing (ops that
            cannot be graph-captured); the graph protocol sizes itself.
        rep: Measurement iterations for the eager fallback timing.
        log_level: Logging verbosity for ``oasr.tune`` logger.
        publish: Also write GEMM-family winners into the tuning DB's user tier
            (``~/.cache/oasr/tune``), where the production dispatch reads them
            after the context exits -- the point of tuning a deployment once.

    Examples::

        # Tune and persist results to a cache file
        with autotune(True, cache="my_configs.json"):
            model(inputs)

        # Load cached configs for inference (no profiling, no save)
        with autotune(False, cache="my_configs.json"):
            model(inputs)
    """
    tuner = AutoTuner.get()

    prev_log_level = logger.level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    cache_valid = True
    prev_publish = tuner.publish
    tuner.publish = publish
    if cache is not None:
        with tuner._lock:
            tuner._file_configs.clear()
            tuner._logged_file_hits.clear()
        if os.path.isfile(cache):
            cache_valid = tuner.load_configs(cache)

    # Reference-counted tuning mode: stays True as long as at least one
    # autotune(True) context is active, even with overlapping contexts.
    with tuner._lock:
        if tune_mode:
            tuner._active_tuning_contexts += 1
        old_mode = tuner.is_tuning_mode
        tuner.is_tuning_mode = tuner._active_tuning_contexts > 0
        tuner.warmup = warmup
        tuner.repeat = rep
        autotune_enabled = tune_mode and not old_mode
    if autotune_enabled:
        logger.info("[Autotuner]: Autotuning process starts ...")

    try:
        yield tuner
    finally:
        with tuner._lock:
            if tune_mode:
                tuner._active_tuning_contexts -= 1
            tuner.is_tuning_mode = tuner._active_tuning_contexts > 0
        if autotune_enabled:
            logger.info("[Autotuner]: Autotuning process ends")

        if cache is not None and cache_valid and tune_mode and tuner._dirty:
            try:
                tuner.save_configs(cache)
            except OSError as exc:
                logger.warning("[Autotuner]: Failed to save cache: %s", exc)
        if tune_mode and not tuner.is_tuning_mode:
            _flush_tuning_db()
        tuner.publish = prev_publish

        logger.setLevel(prev_log_level)


# =============================================================================
# Module-level convenience API
# =============================================================================

_enabled: bool = False
_active_cache_path: Optional[str] = None


def _ensure_backends_registered() -> None:
    """Import backend modules to trigger registration."""
    from .backends import (  # noqa: F401
        conv2d as _conv2d_backends,
        gemm as _gemm_backends,
        recurrent as _recurrent_backends,
        torch_gemm as _torch_gemm_backends,
    )


def is_tuning_enabled() -> bool:
    """Return ``True`` if autotuning dispatch is active.

    Returns True when either:
    - ``enable_autotune()`` has been called (``_enabled`` flag), or
    - An ``autotune()`` context manager is active (``tuner.is_tuning_mode``).
    """
    if _enabled:
        return True
    # Check the singleton directly without creating it (avoid side effects
    # on hot-path calls).
    instance = AutoTuner._instance
    return instance is not None and instance.is_tuning_mode


def get_tuner() -> AutoTuner:
    """Return the global ``AutoTuner`` instance, creating it lazily."""
    return AutoTuner.get()


def enable_autotune(
    cache: Optional[str] = None,
    *,
    warmup: int = 25,
    rep: int = 100,
    log_level: str = "INFO",
) -> None:
    """Enable autotuning globally (non-context-manager style).

    Args:
        cache: Optional JSON cache file. Loaded immediately; saved when
               ``disable_autotune()`` is called with ``save_cache=True``.
        warmup: Number of warmup iterations for profiling.
        rep: Number of measurement iterations for profiling.
        log_level: Logging verbosity for ``oasr.tune`` logger.
    """
    global _enabled, _active_cache_path

    tuner = AutoTuner.get()
    tuner.is_tuning_mode = True
    tuner.warmup = warmup
    tuner.repeat = rep
    _enabled = True
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    _active_cache_path = cache
    if _active_cache_path is not None:
        tuner.load_configs(_active_cache_path)


def _flush_tuning_db() -> None:
    try:
        from . import database

        for path in database.flush_user():
            logger.info("[Autotuner]: wrote tuned GEMM rules to %s", path)
    except OSError as exc:
        logger.warning("[Autotuner]: Failed to write the tuning DB user tier: %s", exc)


def disable_autotune(save_cache: bool = True) -> None:
    """Disable autotuning globally.

    Args:
        save_cache: If ``True`` and a cache path was provided to
                    ``enable_autotune()``, save profiled configs to disk.
                    Pending tuning-DB results are written either way.
    """
    global _enabled, _active_cache_path

    if save_cache and _active_cache_path is not None:
        try:
            AutoTuner.get().save_configs(_active_cache_path)
        except OSError as exc:
            logger.warning("[Autotuner]: Failed to save cache: %s", exc)
    _flush_tuning_db()

    _enabled = False
    _active_cache_path = None


def load_configs(path: str) -> None:
    """Load tuned configs from a JSON file into the global cache."""
    AutoTuner.get().load_configs(path)


def save_configs(path: str) -> None:
    """Save all cached configs to a JSON file."""
    AutoTuner.get().save_configs(path)


def clear_cache() -> None:
    """Clear all in-memory cached tuning results."""
    AutoTuner.get().clear_cache()


def get_selected_config(
    op_key: OpKey,
    shape_sig: tuple,
    dtype: str,
    device_sm: int,
) -> Optional[Tactic]:
    """Look up the cached tactic for a specific profile, if any."""
    key = ProfileKey(op_key=op_key, shape_sig=shape_sig, dtype=dtype, device_sm=device_sm)
    return AutoTuner.get().search_cache(key)
