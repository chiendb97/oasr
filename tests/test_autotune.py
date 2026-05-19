# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the autotuning public API (``oasr.tune``).

The internal cache/profiler/kernel-config layout was consolidated into
``oasr.tune.autotuner`` in commit 55a327e, so these tests only exercise
what is re-exported from ``oasr.tune.__init__``.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from oasr.tune import (
    AutoTuner,
    AutoTunerStatistics,
    BackendEntry,
    BackendRegistry,
    OpKey,
    ProfileKey,
    Tactic,
    TuneResult,
    _global_registry,
    autotune,
    get_tuner,
    is_tuning_enabled,
)


# =========================================================================
# Tactic
# =========================================================================


class TestTactic:
    def test_serialization_roundtrip(self):
        t = Tactic("cutlass", (("tile_m", 128), ("tile_n", 64)))
        d = t.to_dict()
        t2 = Tactic.from_dict(d)
        assert t2.backend == "cutlass"
        assert dict(t2.config) == {"tile_m": 128, "tile_n": 64}

    def test_from_dict_empty_config(self):
        t = Tactic.from_dict({"backend": "cudnn"})
        assert t.backend == "cudnn"
        assert t.config == ()

    def test_frozen_hashable(self):
        t1 = Tactic("cutlass")
        t2 = Tactic("cutlass")
        assert hash(t1) == hash(t2)
        assert t1 == t2
        s = {t1, t2}
        assert len(s) == 1

    def test_different_tactics_not_equal(self):
        assert Tactic("cutlass") != Tactic("cudnn")


# =========================================================================
# ProfileKey
# =========================================================================


class TestProfileKey:
    def test_to_str_deterministic(self):
        key = ProfileKey(
            op_key=OpKey("conv", "conv2d"),
            shape_sig=(1, 64, 64, 3, 32, 3, 3, 1, 1, 1, 1, 1, 1),
            dtype="float16",
            device_sm=80,
        )
        s1 = key.to_str()
        s2 = key.to_str()
        assert s1 == s2
        assert "conv" in s1
        assert "sm80" in s1

    def test_roundtrip(self):
        key = ProfileKey(
            op_key=OpKey("gemm", "gemm", ("activation_type=2",)),
            shape_sig=(256, 512, 128),
            dtype="bfloat16",
            device_sm=90,
        )
        s = key.to_str()
        key2 = ProfileKey.from_str(s)
        assert key2.op_key == key.op_key
        assert key2.shape_sig == key.shape_sig
        assert key2.dtype == key.dtype
        assert key2.device_sm == key.device_sm

    def test_different_shapes_different_keys(self):
        base = dict(op_key=OpKey("gemm", "gemm"), dtype="float16", device_sm=80)
        k1 = ProfileKey(shape_sig=(128, 256, 64), **base)
        k2 = ProfileKey(shape_sig=(256, 256, 64), **base)
        assert k1.to_str() != k2.to_str()


# =========================================================================
# BackendRegistry
# =========================================================================


class TestBackendRegistry:
    def _make_entry(self, backend="cutlass", available=True, fallback=False):
        runner = MagicMock()
        return BackendEntry(
            tactic=Tactic(backend),
            is_available=lambda: available,
            get_runner=lambda: runner,
            is_fallback=fallback,
        )

    def test_register_and_get_candidates(self):
        reg = BackendRegistry()
        key = OpKey("gemm", "gemm")
        reg.register(key, self._make_entry("cutlass"))
        reg.register(key, self._make_entry("cublas"))
        assert len(reg.get_candidates(key)) == 2

    def test_unavailable_backend_filtered(self):
        reg = BackendRegistry()
        key = OpKey("conv", "conv2d")
        reg.register(key, self._make_entry("cutlass", available=True))
        reg.register(key, self._make_entry("cudnn", available=False))
        candidates = reg.get_candidates(key)
        assert len(candidates) == 1
        assert candidates[0].tactic.backend == "cutlass"

    def test_fallback_selection(self):
        reg = BackendRegistry()
        key = OpKey("conv", "conv2d")
        reg.register(key, self._make_entry("cudnn", fallback=False))
        reg.register(key, self._make_entry("cutlass", fallback=True))
        fb = reg.get_fallback(key)
        assert fb is not None
        assert fb.tactic.backend == "cutlass"

    def test_fallback_first_available_when_none_marked(self):
        reg = BackendRegistry()
        key = OpKey("conv", "conv2d")
        reg.register(key, self._make_entry("cutlass", fallback=False))
        fb = reg.get_fallback(key)
        assert fb is not None
        assert fb.tactic.backend == "cutlass"

    def test_unknown_op_returns_empty(self):
        reg = BackendRegistry()
        key = OpKey("unknown", "op")
        assert reg.get_candidates(key) == []
        assert reg.get_fallback(key) is None

    def test_get_entry_for_tactic(self):
        reg = BackendRegistry()
        key = OpKey("conv", "conv2d")
        reg.register(key, self._make_entry("cutlass"))
        found = reg.get_entry_for_tactic(key, Tactic("cutlass"))
        assert found is not None
        assert found.tactic.backend == "cutlass"
        assert reg.get_entry_for_tactic(key, Tactic("nonexistent")) is None


# =========================================================================
# AutoTuner — public dispatch / cache / statistics
# =========================================================================


class TestAutoTuner:
    """Exercise the public ``AutoTuner`` surface against a fresh instance.

    The singleton is bypassed so tests don't pollute the global registry.
    """

    def _make_tuner(self):
        tuner = AutoTuner(warmup=1, repeat=1)
        # Detach from the shared global registry so tests are isolated.
        tuner._registry = BackendRegistry()
        return tuner

    def test_search_cache_returns_stored_tactic(self):
        tuner = self._make_tuner()
        op_key = OpKey("gemm", "gemm")
        profile_key = ProfileKey(
            op_key=op_key, shape_sig=(128, 256, 64),
            dtype="float16", device_sm=80,
        )
        tactic = Tactic("cutlass")
        tuner.profiling_cache[profile_key.to_str()] = tactic
        assert tuner.search_cache(profile_key) == tactic

    def test_search_cache_miss_returns_none(self):
        tuner = self._make_tuner()
        profile_key = ProfileKey(
            op_key=OpKey("gemm", "gemm"), shape_sig=(1,),
            dtype="float16", device_sm=80,
        )
        assert tuner.search_cache(profile_key) is None

    def test_dispatch_cache_hit_runs_cached_tactic(self):
        tuner = self._make_tuner()
        op_key = OpKey("gemm", "gemm")
        runner_mock = MagicMock()
        tuner._registry.register(
            op_key,
            BackendEntry(
                tactic=Tactic("cutlass"),
                is_available=lambda: True,
                get_runner=lambda: runner_mock,
                is_fallback=True,
            ),
        )

        profile_key = ProfileKey(
            op_key=op_key, shape_sig=(128, 256, 64),
            dtype="float16", device_sm=80,
        )
        tuner.profiling_cache[profile_key.to_str()] = Tactic("cutlass")

        with patch("oasr.tune.autotuner._device_sm", return_value=80):
            tuner.dispatch(
                op_key=op_key,
                shape_sig=(128, 256, 64),
                dtype=torch.float16,
                device=torch.device("cuda:0"),
                runner_args=(MagicMock(), MagicMock(), MagicMock(), None),
            )
        runner_mock.assert_called_once()

    def test_dispatch_cache_miss_uses_fallback(self):
        tuner = self._make_tuner()
        tuner.is_tuning_mode = False
        op_key = OpKey("gemm", "gemm")

        runner_mock = MagicMock()
        tuner._registry.register(
            op_key,
            BackendEntry(
                tactic=Tactic("cutlass"),
                is_available=lambda: True,
                get_runner=lambda: runner_mock,
                is_fallback=True,
            ),
        )

        with patch("oasr.tune.autotuner._device_sm", return_value=80):
            tuner.dispatch(
                op_key=op_key,
                shape_sig=(128, 256, 64),
                dtype=torch.float16,
                device=torch.device("cuda:0"),
                runner_args=(MagicMock(),),
            )
        runner_mock.assert_called_once()
        assert tuner.stats.cache_misses == 1

    def test_dispatch_no_backend_raises(self):
        tuner = self._make_tuner()
        tuner.is_tuning_mode = False
        op_key = OpKey("unknown", "op")

        with patch("oasr.tune.autotuner._device_sm", return_value=80):
            with pytest.raises(RuntimeError, match="No available backend"):
                tuner.dispatch(
                    op_key=op_key,
                    shape_sig=(1,),
                    dtype=torch.float16,
                    device=torch.device("cuda:0"),
                    runner_args=(),
                )

    def test_select_best(self):
        results = [
            TuneResult(Tactic("slow"), 5.0, 5.0, 5.0, "ok"),
            TuneResult(Tactic("fast"), 1.0, 1.0, 1.0, "ok"),
            TuneResult(Tactic("err"), float("inf"), float("inf"), float("inf"), "error"),
        ]
        # Mirror the dispatch path: candidates are sorted by median_ms first.
        results.sort(key=lambda r: r.median_ms)
        best = AutoTuner._select_best(results)
        assert best is not None
        assert best.tactic.backend == "fast"

    def test_select_best_all_errors(self):
        results = [
            TuneResult(Tactic("a"), float("inf"), float("inf"), float("inf"), "error"),
            TuneResult(Tactic("b"), float("inf"), float("inf"), float("inf"), "error"),
        ]
        assert AutoTuner._select_best(results) is None

    def test_clear_cache(self):
        tuner = self._make_tuner()
        profile_key = ProfileKey(
            op_key=OpKey("gemm", "gemm"), shape_sig=(1,),
            dtype="float16", device_sm=80,
        )
        tuner.profiling_cache[profile_key.to_str()] = Tactic("cutlass")
        tuner.clear_cache()
        assert tuner.search_cache(profile_key) is None

    def test_reset_statistics(self):
        tuner = self._make_tuner()
        tuner.stats.cache_misses = 5
        tuner.reset_statistics()
        assert tuner.stats.cache_misses == 0
        assert isinstance(tuner.stats, AutoTunerStatistics)

    def test_concurrent_cache_writes_are_safe(self):
        tuner = self._make_tuner()
        errors = []

        def writer(i):
            try:
                key = ProfileKey(
                    op_key=OpKey("gemm", "gemm"),
                    shape_sig=(i, 256, 64),
                    dtype="float16",
                    device_sm=80,
                ).to_str()
                tuner.profiling_cache[key] = Tactic("cutlass")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(tuner.profiling_cache) == 50


# =========================================================================
# Config persistence (save_configs / load_configs)
# =========================================================================


class TestConfigPersistence:
    def _profile_key(self, shape=(128, 256, 64)):
        return ProfileKey(
            op_key=OpKey("gemm", "gemm"),
            shape_sig=shape,
            dtype="float16",
            device_sm=80,
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "cache.json"
        tuner = AutoTuner(warmup=1, repeat=1)
        tuner._registry = BackendRegistry()
        key = self._profile_key()
        tuner.profiling_cache[key.to_str()] = Tactic("cudnn")
        tuner.save_configs(str(path))

        # Fresh tuner should pick up the saved tactic via file configs.
        tuner2 = AutoTuner(warmup=1, repeat=1)
        tuner2._registry = BackendRegistry()
        assert tuner2.load_configs(str(path))
        assert tuner2.search_cache(key) == Tactic("cudnn")

    def test_save_then_save_preserves_disk_entries(self, tmp_path):
        path = tmp_path / "cache.json"
        tuner = AutoTuner(warmup=1, repeat=1)
        tuner._registry = BackendRegistry()

        key1 = self._profile_key((128, 256, 64))
        tuner.profiling_cache[key1.to_str()] = Tactic("cutlass")
        tuner.save_configs(str(path))

        tuner.clear_cache()
        tuner.load_configs(str(path))
        key2 = self._profile_key((256, 512, 128))
        tuner.profiling_cache[key2.to_str()] = Tactic("cudnn")
        tuner.save_configs(str(path))

        with open(path) as f:
            data = json.load(f)
        # Both the originally-saved entry and the new one survive.
        assert len(data["entries"]) == 2

    def test_load_corrupt_file_does_not_raise(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json{{{")
        tuner = AutoTuner(warmup=1, repeat=1)
        tuner._registry = BackendRegistry()
        # load_configs swallows the JSONDecodeError and returns False.
        assert tuner.load_configs(str(path)) is False

    def test_load_warns_on_env_mismatch(self, tmp_path, caplog):
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({
            "version": 1,
            "_metadata": {"sm": "70", "cuda_version": "11.0"},
            "entries": {},
        }))

        import logging
        tuner = AutoTuner(warmup=1, repeat=1)
        tuner._registry = BackendRegistry()
        with caplog.at_level(logging.WARNING, logger="oasr.tune"):
            with patch(
                "oasr.tune.autotuner._collect_metadata",
                return_value={"sm": "80", "cuda_version": "12.4", "oasr_version": "0.1.0"},
            ):
                tuner.load_configs(str(path))
        assert any("sm" in r.message for r in caplog.records)


# =========================================================================
# Autotune context manager
# =========================================================================


class TestAutotuneContextManager:
    def test_scoping(self):
        assert not is_tuning_enabled()
        with autotune(True):
            assert is_tuning_enabled()
        assert not is_tuning_enabled()

    def test_nested_scoping(self):
        # ``autotune(False)`` only loads cached configs — tuning mode is
        # ref-counted on ``autotune(True)`` entries.
        assert not is_tuning_enabled()
        with autotune(True):
            assert is_tuning_enabled()
            assert get_tuner().is_tuning_mode
            with autotune(True):
                assert is_tuning_enabled()
                assert get_tuner().is_tuning_mode
            # Inner context exits — outer ``autotune(True)`` still holds tuning on.
            assert is_tuning_enabled()
        assert not is_tuning_enabled()

    def test_autotune_false_does_not_enable_tuning(self):
        assert not is_tuning_enabled()
        with autotune(False):
            assert not is_tuning_enabled()
        assert not is_tuning_enabled()

    def test_cache_file_created(self, tmp_path):
        path = tmp_path / "test_cache.json"
        # Seed an in-memory profile so save_configs has something to write
        # (dirty flag is only set when tuning produces a result).
        tuner = get_tuner()
        key = ProfileKey(
            op_key=OpKey("gemm", "gemm"), shape_sig=(1, 1, 1),
            dtype="float16", device_sm=80,
        )
        with autotune(True, cache=str(path)):
            tuner.profiling_cache[key.to_str()] = Tactic("cutlass")
            tuner._dirty = True
            tuner._dirty_seq += 1
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert "_metadata" in data


# =========================================================================
# Backend registration (smoke tests via the global registry)
# =========================================================================


class TestBackendRegistration:
    """Verify that the bundled GEMM / Conv2D backends register on import."""

    def test_gemm_backends_register(self):
        from oasr.tune.backends import gemm as _  # noqa: F401
        assert _global_registry.get_candidates(OpKey("gemm", "gemm"))

    def test_gemm_activation_backends_register(self):
        from oasr.tune.backends import gemm as _  # noqa: F401
        assert _global_registry.get_candidates(OpKey("gemm", "gemm_activation"))

    def test_bmm_backends_register(self):
        from oasr.tune.backends import gemm as _  # noqa: F401
        assert _global_registry.get_candidates(OpKey("gemm", "bmm"))

    def test_conv2d_backends_register(self):
        from oasr.tune.backends import conv2d as _  # noqa: F401
        assert _global_registry.get_candidates(OpKey("conv", "conv2d"))

    def test_gemm_has_cutlass_fallback(self):
        from oasr.tune.backends import gemm as _  # noqa: F401
        fb = _global_registry.get_fallback(OpKey("gemm", "gemm"))
        assert fb is not None
        assert fb.tactic.backend == "cutlass"

    def test_conv2d_has_cutlass_fallback(self):
        from oasr.tune.backends import conv2d as _  # noqa: F401
        fb = _global_registry.get_fallback(OpKey("conv", "conv2d"))
        assert fb is not None
        assert fb.tactic.backend == "cutlass"

    def test_gemm_tactics_are_unique(self):
        from oasr.tune.backends import gemm as _  # noqa: F401
        candidates = _global_registry.get_candidates(OpKey("gemm", "gemm"))
        tactics = [c.tactic for c in candidates]
        assert len(tactics) == len(set(tactics))
