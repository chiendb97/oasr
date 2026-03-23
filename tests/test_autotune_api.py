"""Tests for the simplified autotuning API surface."""

import warnings
import pytest


class TestAutotuneAPI:
    """Test the public autotuning API without requiring CUDA."""

    def test_autotune_context_enables(self):
        """autotune() context manager sets is_tuning_enabled() inside."""
        import oasr.tune as tune

        assert not tune.is_tuning_enabled()
        with tune.autotune():
            assert tune.is_tuning_enabled()
        assert not tune.is_tuning_enabled()

    def test_autotune_context_disabled_mode(self):
        """autotune(False) enables dispatch but with tune_mode=False."""
        import oasr.tune as tune

        with tune.autotune(False) as tuner:
            assert tune.is_tuning_enabled()
            assert not tuner.tune_mode
        assert not tune.is_tuning_enabled()

    def test_autotune_context_enabled_mode(self):
        """autotune(True) enables dispatch with tune_mode=True."""
        import oasr.tune as tune

        with tune.autotune(True) as tuner:
            assert tune.is_tuning_enabled()
            assert tuner.tune_mode

    def test_autotune_nested(self):
        """Nested autotune contexts restore state correctly."""
        import oasr.tune as tune

        assert not tune.is_tuning_enabled()
        with tune.autotune(True):
            assert tune.is_tuning_enabled()
            with tune.autotune(False) as inner:
                assert tune.is_tuning_enabled()
                assert not inner.tune_mode
            # Outer context restored
            assert tune.is_tuning_enabled()
        assert not tune.is_tuning_enabled()

    def test_enable_disable_toggle(self):
        """enable_autotune/disable_autotune toggle the global flag."""
        import oasr.tune as tune

        assert not tune.is_tuning_enabled()
        tune.enable_autotune()
        assert tune.is_tuning_enabled()
        tune.disable_autotune(save_cache=False)
        assert not tune.is_tuning_enabled()

    def test_backward_compat_tune_mode(self):
        """tune_mode= kwarg still works with deprecation warning."""
        import oasr.tune as tune

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with tune.autotune(tune_mode=True) as tuner:
                assert tune.is_tuning_enabled()
                assert tuner.tune_mode

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "tune_mode" in str(w[0].message)

    def test_autotune_unexpected_kwarg(self):
        """Unknown kwargs raise TypeError."""
        import oasr.tune as tune

        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            with tune.autotune(bad_param=42):
                pass

    def test_top_level_export(self):
        """autotune, enable_autotune, disable_autotune available via oasr.*"""
        import oasr

        assert callable(oasr.autotune)
        assert callable(oasr.enable_autotune)
        assert callable(oasr.disable_autotune)

    def test_cache_roundtrip(self, tmp_path):
        """Cache file is created on context exit and reloaded."""
        import oasr.tune as tune

        cache_file = str(tmp_path / "test_cache.json")

        # First context: should create the file (even if empty)
        with tune.autotune(cache=cache_file):
            pass

        assert (tmp_path / "test_cache.json").exists()

        # Second context: should load without error
        with tune.autotune(False, cache=cache_file):
            pass

    def test_enable_autotune_cache(self, tmp_path):
        """enable_autotune with cache saves on disable."""
        import oasr.tune as tune

        cache_file = str(tmp_path / "toggle_cache.json")

        tune.enable_autotune(cache=cache_file)
        assert tune.is_tuning_enabled()
        tune.disable_autotune(save_cache=True)
        assert not tune.is_tuning_enabled()

        assert (tmp_path / "toggle_cache.json").exists()
