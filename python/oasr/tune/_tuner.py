# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""AutoTuner orchestrator: lookup → profile → select → cache → execute."""

import logging
import threading
from typing import Optional

import torch

from ._cache import TuneCache
from ._profiler import Profiler
from ._registry import BackendRegistry
from ._types import OpKey, ProfileKey, Tactic

logger = logging.getLogger("oasr.tune")


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


class AutoTuner:
    """Central autotuning orchestrator.

    Called from functional API functions to select and execute the best
    ``(backend, tactic)`` for a given operation and input signature.
    """

    def __init__(
        self,
        registry: BackendRegistry,
        cache: TuneCache,
        profiler: Profiler,
        tune_mode: bool = False,
    ) -> None:
        self._registry = registry
        self._cache = cache
        self._profiler = profiler
        self._tune_mode = tune_mode
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tune_mode(self) -> bool:
        return self._tune_mode

    @tune_mode.setter
    def tune_mode(self, value: bool) -> None:
        self._tune_mode = value

    @property
    def cache(self) -> TuneCache:
        return self._cache

    @property
    def registry(self) -> BackendRegistry:
        return self._registry

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

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
        3. If miss and ``tune_mode=True``: profile all candidates, cache
           the best, and execute it.
        4. If miss and ``tune_mode=False``: use fallback tactic.

        Args:
            op_key: Operation identifier.
            shape_sig: Shape signature tuple (operation-specific).
            dtype: Input data type.
            device: CUDA device.
            runner_args: Positional arguments to pass to the runner
                         (e.g. ``(out, input, filter, bias, ...)``).
        """
        profile_key = ProfileKey(
            op_key=op_key,
            shape_sig=shape_sig,
            dtype=_dtype_str(dtype),
            device_sm=_device_sm(device),
        )

        # 1. Cache lookup
        tactic = self._cache.lookup(profile_key)
        if tactic is not None:
            logger.debug("Cache hit for %s → %s", profile_key.to_str(), tactic.backend)
            self._execute(op_key, tactic, runner_args)
            return

        # 2. Cache miss
        if self._tune_mode:
            tactic = self._profile_and_cache(op_key, profile_key, runner_args)
        else:
            tactic = self._fallback(op_key, profile_key)

        if tactic is not None:
            self._execute(op_key, tactic, runner_args)
        else:
            raise RuntimeError(
                f"No available backend for {op_key}. "
                "Ensure at least one backend is registered and available."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_and_cache(
        self,
        op_key: OpKey,
        profile_key: ProfileKey,
        runner_args: tuple,
    ) -> Optional[Tactic]:
        """Profile all candidates and cache the best result."""
        with self._lock:
            # Double-check: another thread may have profiled while we waited.
            cached = self._cache.lookup(profile_key)
            if cached is not None:
                return cached

            candidates = self._registry.get_candidates(op_key)
            if not candidates:
                logger.warning("No candidates for %s", profile_key.to_str())
                return self._fallback(op_key, profile_key)

            logger.info(
                "Profiling %d candidate(s) for %s ...",
                len(candidates),
                profile_key.to_str(),
            )
            results = self._profiler.profile_candidates(candidates, runner_args)
            best = Profiler.select_best(results)

            if best is not None:
                self._cache.store(profile_key, best)
                logger.info(
                    "Selected %s (%.4f ms) for %s",
                    best.tactic.backend,
                    best.median_ms,
                    profile_key.to_str(),
                )
                return best.tactic

            logger.warning(
                "All candidates failed for %s, using fallback",
                profile_key.to_str(),
            )
            return self._fallback(op_key, profile_key)

    def _fallback(self, op_key: OpKey, profile_key: ProfileKey) -> Optional[Tactic]:
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
                    f"Cached tactic {tactic} for {op_key} is unavailable "
                    "and no fallback exists."
                )
            logger.warning(
                "Cached tactic %s unavailable, falling back to %s",
                tactic.backend,
                entry.tactic.backend,
            )
        runner = entry.get_runner()
        runner(*runner_args)
