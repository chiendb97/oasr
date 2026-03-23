# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the autotuning module."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from oasr.tune._types import OpKey, ProfileKey, Tactic, TuneResult
from oasr.tune._cache import TuneCache
from oasr.tune._registry import BackendEntry, BackendRegistry
from oasr.tune._profiler import Profiler
from oasr.tune._tuner import AutoTuner


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
        t1 = Tactic("cutlass")
        t2 = Tactic("cudnn")
        assert t1 != t2


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
# TuneCache
# =========================================================================


class TestTuneCache:
    def _make_result(self, backend="cutlass", ms=1.0):
        return TuneResult(
            tactic=Tactic(backend),
            median_ms=ms,
            min_ms=ms,
            max_ms=ms,
            status="ok",
        )

    def _make_key(self, shape=(128, 256, 64)):
        return ProfileKey(
            op_key=OpKey("gemm", "gemm"),
            shape_sig=shape,
            dtype="float16",
            device_sm=80,
        )

    def test_store_and_lookup(self):
        cache = TuneCache()
        key = self._make_key()
        result = self._make_result()
        cache.store(key, result)
        tactic = cache.lookup(key)
        assert tactic is not None
        assert tactic.backend == "cutlass"

    def test_lookup_miss_returns_none(self):
        cache = TuneCache()
        key = self._make_key()
        assert cache.lookup(key) is None

    def test_error_result_not_stored(self):
        cache = TuneCache()
        key = self._make_key()
        result = TuneResult(
            tactic=Tactic("bad"),
            median_ms=float("inf"),
            min_ms=float("inf"),
            max_ms=float("inf"),
            status="error",
            error_msg="boom",
        )
        cache.store(key, result)
        assert cache.lookup(key) is None

    def test_json_persistence_roundtrip(self, tmp_path):
        cache = TuneCache()
        key = self._make_key()
        cache.store(key, self._make_result("cudnn", 0.5))
        path = tmp_path / "cache.json"
        cache.save(path)

        # Load in a new cache
        cache2 = TuneCache()
        cache2.load(path)
        tactic = cache2.lookup(key)
        assert tactic is not None
        assert tactic.backend == "cudnn"

    def test_merge_incremental(self, tmp_path):
        path = tmp_path / "cache.json"

        # First tuning session
        cache1 = TuneCache()
        key1 = self._make_key((128, 256, 64))
        cache1.store(key1, self._make_result("cutlass", 1.0))
        cache1.save(path)

        # Second tuning session
        cache2 = TuneCache()
        cache2.load(path)
        key2 = self._make_key((256, 512, 128))
        cache2.store(key2, self._make_result("cudnn", 0.5))
        cache2.save(path)

        # Verify both entries exist
        with open(path) as f:
            data = json.load(f)
        assert len(data["entries"]) == 2

    def test_env_metadata_mismatch_warns(self, tmp_path, caplog):
        path = tmp_path / "cache.json"
        data = {
            "version": 1,
            "env": {"sm": 70, "cuda_version": "11.0"},
            "entries": {},
        }
        with open(path, "w") as f:
            json.dump(data, f)

        import logging

        with caplog.at_level(logging.WARNING, logger="oasr.tune"):
            cache = TuneCache()
            with patch("oasr.tune._cache._get_env_metadata", return_value={"sm": 80, "cuda_version": "12.4", "oasr_version": "0.1.0"}):
                cache.load(path)
        assert any("sm" in record.message for record in caplog.records)

    def test_clear(self):
        cache = TuneCache()
        key = self._make_key()
        cache.store(key, self._make_result())
        cache.clear()
        assert cache.lookup(key) is None

    def test_thread_safety(self):
        cache = TuneCache()
        errors = []

        def writer(i):
            try:
                key = self._make_key((i, 256, 64))
                cache.store(key, self._make_result("cutlass", float(i)))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.size() == 50

    def test_corrupt_file_handled(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json{{{")
        cache = TuneCache()
        cache.load(path)  # should not raise


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
        candidates = reg.get_candidates(key)
        assert len(candidates) == 2

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
        entry = self._make_entry("cutlass")
        reg.register(key, entry)
        found = reg.get_entry_for_tactic(key, Tactic("cutlass"))
        assert found is not None
        assert found.tactic.backend == "cutlass"
        assert reg.get_entry_for_tactic(key, Tactic("nonexistent")) is None


# =========================================================================
# AutoTuner
# =========================================================================


class TestAutoTuner:
    def _make_tuner(self, tune_mode=False):
        reg = BackendRegistry()
        cache = TuneCache()
        profiler = Profiler(warmup=1, rep=1)
        return AutoTuner(reg, cache, profiler, tune_mode=tune_mode), reg, cache

    def test_cache_hit_skips_profiling(self):
        tuner, reg, cache = self._make_tuner()
        op_key = OpKey("gemm", "gemm")

        runner_mock = MagicMock()
        reg.register(
            op_key,
            BackendEntry(
                tactic=Tactic("cutlass"),
                is_available=lambda: True,
                get_runner=lambda: runner_mock,
                is_fallback=True,
            ),
        )

        # Pre-populate cache
        key = ProfileKey(op_key=op_key, shape_sig=(128, 256, 64), dtype="float16", device_sm=80)
        cache.store(key, TuneResult(
            tactic=Tactic("cutlass"),
            median_ms=1.0, min_ms=1.0, max_ms=1.0, status="ok",
        ))

        import torch

        with patch.object(tuner, "_profile_and_cache") as mock_profile:
            with patch("oasr.tune._tuner._device_sm", return_value=80):
                tuner.dispatch(
                    op_key=op_key,
                    shape_sig=(128, 256, 64),
                    dtype=torch.float16,
                    device=torch.device("cuda:0"),
                    runner_args=(MagicMock(), MagicMock(), MagicMock(), None),
                )
        mock_profile.assert_not_called()
        runner_mock.assert_called_once()

    def test_cache_miss_no_tune_uses_fallback(self):
        tuner, reg, cache = self._make_tuner(tune_mode=False)
        op_key = OpKey("gemm", "gemm")

        runner_mock = MagicMock()
        reg.register(
            op_key,
            BackendEntry(
                tactic=Tactic("cutlass"),
                is_available=lambda: True,
                get_runner=lambda: runner_mock,
                is_fallback=True,
            ),
        )

        import torch

        with patch("oasr.tune._tuner._device_sm", return_value=80):
            tuner.dispatch(
                op_key=op_key,
                shape_sig=(128, 256, 64),
                dtype=torch.float16,
                device=torch.device("cuda:0"),
                runner_args=(MagicMock(),),
            )
        runner_mock.assert_called_once()

    def test_no_backend_raises(self):
        tuner, reg, cache = self._make_tuner(tune_mode=False)
        op_key = OpKey("unknown", "op")

        import torch

        with patch("oasr.tune._tuner._device_sm", return_value=80):
            with pytest.raises(RuntimeError, match="No available backend"):
                tuner.dispatch(
                    op_key=op_key,
                    shape_sig=(1,),
                    dtype=torch.float16,
                    device=torch.device("cuda:0"),
                    runner_args=(),
                )


# =========================================================================
# Context manager
# =========================================================================


class TestAutotuneContextManager:
    def test_scoping(self):
        from oasr.tune import autotune, is_tuning_enabled

        assert not is_tuning_enabled()
        with autotune(False):
            assert is_tuning_enabled()
        assert not is_tuning_enabled()

    def test_nested_scoping(self):
        from oasr.tune import autotune, is_tuning_enabled, get_tuner

        assert not is_tuning_enabled()
        with autotune(False):
            assert is_tuning_enabled()
            assert not get_tuner().tune_mode
            with autotune(True):
                assert is_tuning_enabled()
                assert get_tuner().tune_mode
            # Outer context restores enabled=False
            assert not get_tuner().tune_mode
            assert is_tuning_enabled()
        assert not is_tuning_enabled()

    def test_cache_file_created(self, tmp_path):
        from oasr.tune import autotune

        path = tmp_path / "test_cache.json"
        with autotune(False, cache=str(path)):
            pass
        # Cache file is written on exit even if empty
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert "env" in data


# =========================================================================
# Profiler (unit-level, no CUDA needed)
# =========================================================================


class TestProfilerSelectBest:
    def test_select_best(self):
        results = [
            TuneResult(Tactic("slow"), 5.0, 5.0, 5.0, "ok"),
            TuneResult(Tactic("fast"), 1.0, 1.0, 1.0, "ok"),
            TuneResult(Tactic("err"), float("inf"), float("inf"), float("inf"), "error"),
        ]
        best = Profiler.select_best(results)
        assert best is not None
        # Results must be sorted by median_ms before calling select_best
        results.sort(key=lambda r: r.median_ms)
        best = Profiler.select_best(results)
        assert best.tactic.backend == "fast"

    def test_select_best_all_errors(self):
        results = [
            TuneResult(Tactic("a"), float("inf"), float("inf"), float("inf"), "error"),
            TuneResult(Tactic("b"), float("inf"), float("inf"), float("inf"), "error"),
        ]
        assert Profiler.select_best(results) is None


# =========================================================================
# TileConfig
# =========================================================================


class TestTileConfig:
    def test_name_without_split_k(self):
        from oasr.tune.kernel_configs import TileConfig

        cfg = TileConfig(tile_m=128, tile_n=128, tile_k=32,
                         warp_m=64, warp_n=64, warp_k=32, stages=3)
        assert cfg.name == "t128x128x32_w64x64x32_s3"

    def test_name_with_split_k(self):
        from oasr.tune.kernel_configs import TileConfig

        cfg = TileConfig(tile_m=128, tile_n=128, tile_k=32,
                         warp_m=64, warp_n=64, warp_k=32, stages=2, split_k=4)
        assert cfg.name == "t128x128x32_w64x64x32_s2_sk4"

    def test_num_warps(self):
        from oasr.tune.kernel_configs import TileConfig

        cfg = TileConfig(tile_m=128, tile_n=128, tile_k=32,
                         warp_m=64, warp_n=64, warp_k=32, stages=2)
        assert cfg.num_warps == 4  # (128/64) * (128/64) = 4

    def test_cuda_flags(self):
        from oasr.tune.kernel_configs import TileConfig

        cfg = TileConfig(tile_m=256, tile_n=128, tile_k=64,
                         warp_m=64, warp_n=64, warp_k=64, stages=3)
        flags = cfg.cuda_flags("OASR_GEMM")
        assert "-DOASR_GEMM_TILE_M=256" in flags
        assert "-DOASR_GEMM_STAGES=3" in flags
        assert len(flags) == 7

    def test_to_tactic_config(self):
        from oasr.tune.kernel_configs import TileConfig

        cfg = TileConfig(tile_m=128, tile_n=128, tile_k=32,
                         warp_m=64, warp_n=64, warp_k=32, stages=2, split_k=2)
        tc = cfg.to_tactic_config()
        d = dict(tc)
        assert d["tile_m"] == 128
        assert d["split_k"] == 2

    def test_to_tactic_config_no_split_k(self):
        from oasr.tune.kernel_configs import TileConfig

        cfg = TileConfig(tile_m=128, tile_n=128, tile_k=32,
                         warp_m=64, warp_n=64, warp_k=32, stages=2)
        tc = cfg.to_tactic_config()
        d = dict(tc)
        assert "split_k" not in d

    def test_frozen(self):
        from oasr.tune.kernel_configs import TileConfig

        cfg = TileConfig(tile_m=128, tile_n=128, tile_k=32,
                         warp_m=64, warp_n=64, warp_k=32, stages=2)
        with pytest.raises(AttributeError):
            cfg.tile_m = 256


# =========================================================================
# Kernel config lists
# =========================================================================


class TestKernelConfigs:
    def test_gemm_configs_include_default(self):
        from oasr.tune.kernel_configs import GEMM_TILE_CONFIGS, GEMM_DEFAULT

        assert GEMM_DEFAULT in GEMM_TILE_CONFIGS

    def test_bmm_configs_no_split_k(self):
        from oasr.tune.kernel_configs import BMM_TILE_CONFIGS

        for cfg in BMM_TILE_CONFIGS:
            assert cfg.split_k == 1

    def test_group_gemm_configs_no_split_k(self):
        from oasr.tune.kernel_configs import GROUP_GEMM_TILE_CONFIGS

        for cfg in GROUP_GEMM_TILE_CONFIGS:
            assert cfg.split_k == 1

    def test_conv2d_configs_include_default(self):
        from oasr.tune.kernel_configs import CONV2D_TILE_CONFIGS, CONV2D_DEFAULT

        assert CONV2D_DEFAULT in CONV2D_TILE_CONFIGS

    def test_get_unique_compile_configs_deduplicates_split_k(self):
        from oasr.tune.kernel_configs import TileConfig, get_unique_compile_configs

        configs = [
            TileConfig(128, 128, 32, 64, 64, 32, 2, split_k=1),
            TileConfig(128, 128, 32, 64, 64, 32, 2, split_k=2),
            TileConfig(128, 128, 32, 64, 64, 32, 2, split_k=4),
            TileConfig(256, 128, 32, 64, 64, 32, 3, split_k=1),
        ]
        unique = get_unique_compile_configs(configs)
        assert len(unique) == 2  # Two distinct compile configs

    def test_all_gemm_configs_valid_warp_counts(self):
        from oasr.tune.kernel_configs import GEMM_TILE_CONFIGS

        for cfg in GEMM_TILE_CONFIGS:
            assert cfg.tile_m % cfg.warp_m == 0
            assert cfg.tile_n % cfg.warp_n == 0
            assert cfg.num_warps in (1, 2, 4, 8, 16)


# =========================================================================
# Backend registration (tile variants)
# =========================================================================


class TestBackendTileVariants:
    def test_gemm_tile_variants_registered(self):
        from oasr.tune._registry import _global_registry
        from oasr.tune.kernel_configs import GEMM_TILE_CONFIGS

        # Force backend import
        from oasr.tune.backends import gemm as _  # noqa: F401

        candidates = _global_registry.get_candidates(OpKey("gemm", "gemm"))
        # Should have at least as many candidates as tile configs
        assert len(candidates) >= len(GEMM_TILE_CONFIGS)

    def test_gemm_activation_tile_variants_registered(self):
        from oasr.tune._registry import _global_registry
        from oasr.tune.kernel_configs import GEMM_TILE_CONFIGS

        from oasr.tune.backends import gemm as _  # noqa: F401

        candidates = _global_registry.get_candidates(OpKey("gemm", "gemm_activation"))
        assert len(candidates) >= len(GEMM_TILE_CONFIGS)

    def test_bmm_tile_variants_registered(self):
        from oasr.tune._registry import _global_registry
        from oasr.tune.kernel_configs import BMM_TILE_CONFIGS

        from oasr.tune.backends import gemm as _  # noqa: F401

        candidates = _global_registry.get_candidates(OpKey("gemm", "bmm"))
        assert len(candidates) >= len(BMM_TILE_CONFIGS)

    def test_conv2d_tile_variants_registered(self):
        from oasr.tune._registry import _global_registry
        from oasr.tune.kernel_configs import CONV2D_TILE_CONFIGS

        from oasr.tune.backends import conv2d as _  # noqa: F401

        candidates = _global_registry.get_candidates(OpKey("conv", "conv2d"))
        # CUTLASS tile variants + cuDNN
        assert len(candidates) >= len(CONV2D_TILE_CONFIGS)

    def test_conv2d_has_cudnn_backend(self):
        from oasr.tune._registry import _global_registry

        from oasr.tune.backends import conv2d as _  # noqa: F401

        candidates = _global_registry.get_candidates(OpKey("conv", "conv2d"))
        cudnn_entries = [c for c in candidates if c.tactic.backend == "cudnn"]
        # cuDNN may not be available, but entry should be registered
        all_entries = _global_registry._entries.get(OpKey("conv", "conv2d"), [])
        cudnn_all = [e for e in all_entries if e.tactic.backend == "cudnn"]
        assert len(cudnn_all) >= 1

    def test_gemm_fallback_is_default_tile(self):
        from oasr.tune._registry import _global_registry

        from oasr.tune.backends import gemm as _  # noqa: F401

        fallback = _global_registry.get_fallback(OpKey("gemm", "gemm"))
        assert fallback is not None
        assert fallback.tactic.backend == "cutlass"
        # Fallback should have the default tile config
        config_dict = dict(fallback.tactic.config)
        assert config_dict["tile_m"] == 128
        assert config_dict["tile_n"] == 128

    def test_conv2d_fallback_is_default_tile(self):
        from oasr.tune._registry import _global_registry

        from oasr.tune.backends import conv2d as _  # noqa: F401

        fallback = _global_registry.get_fallback(OpKey("conv", "conv2d"))
        assert fallback is not None
        assert fallback.tactic.backend == "cutlass"
        config_dict = dict(fallback.tactic.config)
        assert config_dict["tile_m"] == 128
        assert config_dict["tile_k"] == 64  # Conv2D default has tile_k=64

    def test_each_tile_variant_has_unique_tactic(self):
        from oasr.tune._registry import _global_registry

        from oasr.tune.backends import gemm as _  # noqa: F401

        candidates = _global_registry.get_candidates(OpKey("gemm", "gemm"))
        tactics = [c.tactic for c in candidates]
        # All tactics should be unique
        assert len(tactics) == len(set(tactics))
