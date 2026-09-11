# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``EngineConfig``: every field's validation, normalisation and default.

This was asserted in five files -- ``test_engine.py``, ``test_memory.py``,
``test_scheduler.py`` and twice in ``test_graphs.py`` -- because each feature
validated its own knobs where it happened to be tested. Two of those classes
were even both called ``TestConfigValidation``, and two of them opened with
the same ``test_default_is_none``. One config object, one file: adding a knob
now has an obvious place to be checked, and "is this rejected?" has one answer
to look up.

Pure Python -- no GPU, no checkpoint.
"""

from __future__ import annotations

import os

import pytest

from oasr.engine.config import EngineConfig
from oasr.engine.offline_graph import DEFAULT_FRAME_GRANULARITY


def _config(**kw) -> EngineConfig:
    """One builder. The three files this file came from had three, differing
    only in which keyword set they defaulted."""
    kw.setdefault("ckpt_dir", "/nonexistent")
    return EngineConfig(**kw)


# --------------------------------------------------------------------------
# Construction: ckpt autodetect, computed properties, the cache config
# --------------------------------------------------------------------------


class TestEngineConfig:
    def test_default_feature_config(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/tmp/fake")
        assert cfg.feature_config is not None
        assert cfg.feature_config.dither == 0.0

    def test_computed_properties(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/tmp/fake", chunk_size=16)
        assert cfg.subsampling_rate == 4
        assert cfg.right_context == 6
        # stride = 4 * 16 = 64
        assert cfg.stride == 64
        # decoding_window = (16 - 1) * 4 + 6 + 1 = 67
        assert cfg.decoding_window == 67

    def test_autodetect_sentencepiece(self, ckpt_dir: str):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir=ckpt_dir)
        assert cfg.sentencepiece_model is not None
        assert cfg.sentencepiece_model.endswith(".model")
        assert os.path.exists(cfg.sentencepiece_model)

    def test_autodetect_unit_table(self, ckpt_dir: str):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir=ckpt_dir)
        assert cfg.unit_table is not None
        assert os.path.exists(cfg.unit_table)

    def test_build_cache_config(self):
        from oasr.engine.config import EngineConfig
        from oasr.models.conformer.config import ConformerEncoderConfig, ConformerModelConfig

        cfg = EngineConfig(ckpt_dir="/tmp/fake", chunk_size=16, max_num_blocks=512)
        enc_cfg = ConformerEncoderConfig(
            output_size=256, num_blocks=12, attention_heads=4, cnn_module_kernel=15
        )
        model_cfg = ConformerModelConfig(encoder=enc_cfg, vocab_size=5002)
        cc = cfg.build_cache_config(model_cfg.cache_spec)
        assert cc.num_layers == 12
        assert cc.hidden_dim == 256
        assert cc.kernel_size == 15
        assert cc.n_kv_head == 4
        assert cc.chunk_size == 16
        assert cc.max_num_blocks == 512


# --------------------------------------------------------------------------
# The VRAM knobs: what ``None`` means and which combinations are refused
# --------------------------------------------------------------------------


class TestEngineConfigSurface:
    def test_none_means_derive_and_is_accepted(self):
        cfg = EngineConfig(max_num_blocks=None)
        assert cfg.max_num_blocks is None

    def test_zero_blocks_is_still_a_mistake(self):
        with pytest.raises(ValueError, match="max_num_blocks"):
            EngineConfig(max_num_blocks=0)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_utilization_bounds(self, bad):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            EngineConfig(gpu_memory_utilization=bad)

    def test_kv_budget_zero_is_off_negative_is_an_error(self):
        assert EngineConfig(decode_kv_budget_gib=0).decode_kv_budget_gib == 0
        with pytest.raises(ValueError, match="decode_kv_budget_gib"):
            EngineConfig(decode_kv_budget_gib=-1.0)

    def test_build_cache_config_refuses_an_unresolved_pool(self):
        """``None`` is a request for a derivation, not a value.  Reaching the
        cache config with it unresolved means nobody derived it."""
        from oasr.models.base import CacheSpec

        spec = CacheSpec(
            num_layers=4, n_kv_head=2, head_dim=32, hidden_dim=128, conv_kernel_size=15
        )
        cfg = EngineConfig(max_num_blocks=None)
        with pytest.raises(ValueError, match="derive from free VRAM"):
            cfg.build_cache_config(spec)

    def test_build_cache_config_passes_a_resolved_pool_through(self):
        from oasr.models.base import CacheSpec

        spec = CacheSpec(
            num_layers=4, n_kv_head=2, head_dim=32, hidden_dim=128, conv_kernel_size=15
        )
        cc = EngineConfig(max_num_blocks=777).build_cache_config(spec)
        assert cc.max_num_blocks == 777


# --------------------------------------------------------------------------
# The frame budget
# --------------------------------------------------------------------------


class TestFrameBudgetValidation:
    def test_default_is_none(self):
        assert _config().max_batch_frames is None

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="max_batch_frames"):
            _config(max_batch_frames=0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="max_batch_frames"):
            _config(max_batch_frames=-5)


# --------------------------------------------------------------------------
# The preferred-batch ladder: normalisation and rejection
# --------------------------------------------------------------------------


class TestPreferredBatchNormalisation:
    def test_default_is_none(self):
        cfg = _config()
        assert cfg.preferred_batch_size is None

    def test_dedupe_and_sort(self):
        cfg = _config(preferred_batch_size=[8, 4, 8, 2])
        assert cfg.preferred_batch_size == [2, 4, 8]

    def test_rejects_value_above_cap(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            _config(max_batch_size=8, preferred_batch_size=[4, 16])

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match=">= 1"):
            _config(preferred_batch_size=[0, 4])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            _config(preferred_batch_size=[])

    def test_defaults_feature_buckets(self):
        cfg = _config(preferred_batch_size=[4, 8])
        assert cfg.feature_graph_batch_buckets == [4, 8]

    def test_explicit_feature_buckets_win(self):
        cfg = EngineConfig(
            ckpt_dir="/tmp/fake",
            max_batch_size=16,
            preferred_batch_size=[4, 8],
            feature_graph_batch_buckets=[16],
        )
        assert cfg.feature_graph_batch_buckets == [16]


# --------------------------------------------------------------------------
# The offline graph cache
# --------------------------------------------------------------------------


class TestOfflineGraphValidation:
    @pytest.mark.parametrize(
        "kw",
        [
            {"offline_graph_frame_granularity": 0},
            {"offline_graph_max_frames": 8, "offline_graph_frame_granularity": 64},
            {"offline_graph_max_captures": 0},
            {"offline_graph_batch_buckets": [0]},
            {"offline_graph_batch_buckets": [1, 999]},
        ],
    )
    def test_rejects_incoherent_knobs(self, kw):
        from oasr.engine.config import EngineConfig

        with pytest.raises(ValueError):
            EngineConfig(ckpt_dir="/nonexistent", max_batch_size=32, **kw)

    def test_defaults_are_coherent(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/nonexistent")
        assert cfg.offline_graph_frame_granularity == DEFAULT_FRAME_GRANULARITY
        assert cfg.use_offline_cuda_graphs is True


# --------------------------------------------------------------------------
# The streaming graph ladder
# --------------------------------------------------------------------------


class TestLadderConfigValidation:
    @pytest.mark.parametrize(
        "kw",
        [
            {"streaming_graph_cache_growth": 0.5},
            {"streaming_graph_max_shapes": 0},
            {"streaming_graph_batch_ladder": [0]},
            {"streaming_graph_batch_ladder": [1, 999]},
        ],
    )
    def test_rejects_incoherent_knobs(self, kw):
        from oasr.engine.config import EngineConfig

        with pytest.raises(ValueError):
            EngineConfig(ckpt_dir="/nonexistent", max_batch_size=32, **kw)

    def test_defaults(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/nonexistent")
        assert cfg.streaming_graph_cache_growth > 1.0
        assert cfg.streaming_graph_batch_ladder is None  # None == every width
