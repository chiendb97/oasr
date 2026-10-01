"""Shape-space partitioning, dynamic-dimension buckets and census key enumeration.

All pure functions of synthetic inputs -- no kernel runs.
"""

from types import SimpleNamespace

import pytest

from oasr.tune import partition, shapes
from oasr.tune.bench import Measurement
from oasr.tune.census import TrafficModel, offline_keys, streaming_run_widths


def _point(M, times, weight=1.0, alpha=0.0, issue=None):
    results = {}
    for name, t in times.items():
        m = Measurement(name, median_ms=t, n=6)
        if issue and name in issue:
            m.issue_us = issue[name]
        results[name] = m
    return SimpleNamespace(M=M, alpha=alpha, weight=weight, case=SimpleNamespace(results=results))


class TestBuckets:
    def test_edges_ascend_and_round_up(self):
        edges = shapes.BUCKET_EDGES
        assert list(edges) == sorted(set(edges))
        assert shapes.bucket(1) == 8 and shapes.bucket(8) == 8 and shapes.bucket(9) == 16
        assert shapes.bucket(129) == 181 and shapes.bucket(1025) == 2048

    def test_never_rounds_down_past_the_ladder(self):
        big = (1 << 20) + 1
        assert shapes.bucket(big) >= big

    def test_dynamic_dims_are_bucketed_and_static_ones_kept(self):
        assert shapes.bucket_shape_sig("gemm", "gemm", (100, 256, 64)) == (104, 256, 64)
        assert shapes.bucket_shape_sig("gemm", "bmm", (3, 100, 64, 64)) == (8, 104, 64, 64)
        assert shapes.bucket_shape_sig("x", "unknown", (100, 5)) == (100, 5)


class TestConfigCover:
    def test_one_config_near_best_everywhere_covers_alone(self):
        mat = partition.matrix_from_points(
            [
                _point(64, {"a": 1.00, "b": 1.20, "c": 1.02}),
                _point(128, {"a": 1.30, "b": 1.00, "c": 1.01}),
            ]
        )
        assert partition.config_cover(mat, eps=0.03) == ["c"]

    def test_disjoint_winners_need_both(self):
        mat = partition.matrix_from_points(
            [
                _point(64, {"a": 1.0, "b": 2.0}),
                _point(128, {"a": 2.0, "b": 1.0}),
            ]
        )
        assert sorted(partition.config_cover(mat, eps=0.03)) == ["a", "b"]


class TestSegment:
    def _mat(self):
        return partition.matrix_from_points(
            [
                _point(16, {"a": 1.0, "b": 1.5}),
                _point(32, {"a": 1.0, "b": 1.5}),
                _point(64, {"a": 1.5, "b": 1.0}),
                _point(128, {"a": 1.5, "b": 1.0}),
            ]
        )

    def test_free_boundaries_follow_the_winners(self):
        segs = partition.segment(self._mat(), ["a", "b"], lam=0.0)
        assert segs == [(0, 1, "a"), (2, 3, "b")]

    def test_expensive_boundaries_collapse_to_one_region(self):
        segs = partition.segment(self._mat(), ["a", "b"], lam=100.0)
        assert len(segs) == 1


class TestRegionsForSignature:
    def test_a_marginal_win_keeps_the_fallback(self):
        pts = [_point(m, {"default": 1.00, "x": 0.98}) for m in (64, 128, 256)]
        regions, _ = partition.regions_for_signature(pts, "default", min_speedup=1.05)
        assert regions == [(None, "default")]

    def test_a_real_win_becomes_a_region_and_the_top_is_a_catch_all(self):
        pts = [
            _point(64, {"default": 2.0, "x": 1.0}),
            _point(128, {"default": 2.0, "x": 1.0}),
            _point(4096, {"default": 1.0, "x": 1.5}),
        ]
        regions, notes = partition.regions_for_signature(pts, "default", min_speedup=1.05)
        assert regions == [(128, "x"), (None, "default")]
        assert any("worst measured regret" in n for n in notes)

    def test_issue_cost_counts_for_eager_points(self):
        """At alpha=1 a GPU-faster cuBLAS arm loses to a CUTLASS arm that ties it on
        the GPU but issues 5 us cheaper -- the Whisper batch-1 lesson."""
        pts = [
            _point(
                1500,
                {"default": 0.030, "torch": 0.0180, "x": 0.0184},
                alpha=1.0,
                issue={"torch": 14.0, "x": 9.0, "default": 9.0},
            )
        ]
        regions, _ = partition.regions_for_signature(pts, "default", min_speedup=1.05)
        assert regions == [(None, "x")]


class TestCompileBudget:
    def test_the_cheapest_config_to_lose_goes_first(self):
        mats = {
            "s1": partition.matrix_from_points([_point(64, {"default": 3.0, "x": 1.0, "y": 1.1})]),
            "s2": partition.matrix_from_points([_point(64, {"default": 3.0, "x": 1.05, "y": 1.0})]),
        }
        chosen = {"s1": [(None, "x")], "s2": [(None, "y")]}
        out = partition.apply_compile_budget(mats, chosen, 1, {"s1": "default", "s2": "default"})
        used = {c for regs in out.values() for _, c in regs}
        assert used == {"x"}, "dropping y costs 0.05 on s2; dropping x costs 0.1 on s1"


def _cfg(**over):
    fc = SimpleNamespace(
        sample_rate=16000,
        frame_length_samples=400,
        frame_shift_samples=160,
        fixed_window_frames=None,
    )
    base = {
        "feature_config": fc,
        "use_cuda_graphs": True,
        "use_offline_cuda_graphs": True,
        "offline_graph_frame_granularity": 64,
        "offline_graph_max_frames": 4096,
        "preferred_batch_size": None,
        "max_batch_size": 8,
        "offline_graph_batch_buckets": None,
        "streaming_graph_batch_ladder": None,
        "streaming_graph_pad_batch": None,
    }
    base.update(over)
    return SimpleNamespace(**base)


class TestOfflineKeys:
    def test_length_sorted_batches_pad_to_graph_buckets(self):
        traffic = TrafficModel(durations_s=[1.0] * 8 + [9.0] * 3)
        keys = offline_keys(_cfg(), traffic)
        # 8 one-second utterances fill one batch; the three long ones pad to bucket 4.
        assert sum(keys.values()) == 2
        t_short = -(-(1 + (16000 - 400) // 160) // 64) * 64
        assert any(k.batch == 8 and k.frames == t_short and k.captured for k in keys)
        assert any(k.batch == 4 and k.captured for k in keys)
        assert all(k.frames % 64 == 0 for k in keys)

    def test_a_fixed_window_frontend_has_one_time_bucket(self):
        fc = SimpleNamespace(
            sample_rate=16000,
            frame_length_samples=400,
            frame_shift_samples=160,
            fixed_window_frames=3000,
        )
        keys = offline_keys(_cfg(feature_config=fc), TrafficModel(durations_s=[1.0, 20.0]))
        # Exact, never rounded to the frame granularity: Whisper/Qwen2-Audio
        # reject 3008 frames (AGENTS.md, fixed-window frontends).
        assert {k.frames for k in keys} == {3000}

    def test_graphs_off_means_nothing_is_captured(self):
        keys = offline_keys(_cfg(use_cuda_graphs=False), TrafficModel(durations_s=[3.0] * 5))
        assert not any(k.captured for k in keys)


class TestCensusEngineConfig:
    """The census engine must load the checkpoint's own frontend.

    ``dataclasses.replace`` re-runs ``__post_init__`` with the default frontend
    it filled in, which then reads as an explicit choice and overrides the
    checkpoint's ``FeatureSpec`` -- Whisper was probed with 448 fbank frames.
    """

    def test_an_implicit_frontend_stays_implicit(self):
        from oasr.engine.config import EngineConfig
        from oasr.tune.census import census_engine_config

        out = census_engine_config(EngineConfig(ckpt_dir="/nonexistent"))
        assert not out._feature_config_explicit
        assert not out.use_cuda_graphs and not out.use_offline_cuda_graphs

    def test_an_explicit_frontend_is_kept(self):
        from oasr.engine.config import EngineConfig
        from oasr.features import FeatureConfig
        from oasr.tune.census import census_engine_config

        cfg = EngineConfig(ckpt_dir="/x", feature_config=FeatureConfig(num_mel_bins=128))
        out = census_engine_config(cfg)
        assert out._feature_config_explicit and out.feature_config.num_mel_bins == 128


class TestStreamingWidths:
    @pytest.mark.parametrize(
        "over,expect",
        [
            ({}, [1, 2, 4, 8]),
            ({"streaming_graph_pad_batch": False}, [1, 2, 3, 4, 5, 6, 7, 8]),
            ({"streaming_graph_batch_ladder": [3, 6]}, [3, 6, 8]),
        ],
    )
    def test_run_widths(self, over, expect):
        assert streaming_run_widths(_cfg(**over), rungs=4) == expect


class TestPrebuild:
    """``oasr tune prebuild``: what it compiles, and how it splits the cores."""

    def test_the_job_budget_follows_the_tu_counts(self):
        from oasr.tune.prebuild import _split_jobs

        shares = _split_jobs([35, 10, 2, 2], 32)
        assert shares[0] > shares[1] > shares[2] >= 1
        assert all(1 <= s <= n for s, n in zip(shares, [35, 10, 2, 2]))
        assert sum(shares) <= 32 + len(shares)  # rounding, never a second budget

    def test_a_module_never_gets_more_jobs_than_it_has_units(self):
        from oasr.tune.prebuild import _split_jobs

        assert _split_jobs([2], 32) == [2]

    def test_only_what_the_build_loads(self):
        from oasr.tune.prebuild import modules_for

        names = [n for n, _ in modules_for({"gemm"})]
        assert names[0] == "gemm" and "softmax" not in names
        head = [n for n, _ in modules_for({"gemm_log_softmax"})]
        assert {"gemm_log_softmax", "softmax"} <= set(head)
        # bmm is census data the build does not tune -- nothing to compile for it.
        assert modules_for({"bmm"}) == []


class TestThinCensus:
    """``oasr tune build --thin``: a log-spaced subset that still carries the traffic."""

    @staticmethod
    def _pts(ms, weight=1.0, must=False):
        from oasr.tune.census import ShapePoint

        return [
            ShapePoint("gemm", 256, 256, "bfloat16", 1, m, calls=2.0, weight=weight, must=must)
            for m in ms
        ]

    def test_the_cap_holds_and_the_range_is_kept(self):
        from oasr.tune.build import thin_census

        out = thin_census(self._pts([128 * k for k in range(1, 65)]), 12)
        assert len(out) == 12
        assert (out[0].M, out[-1].M) == (128, 128 * 64)
        assert [p.M for p in out] == sorted({p.M for p in out})

    def test_a_dropped_point_is_served_by_the_next_kept_one(self):
        """Regions round up, so a dropped M's weight, calls and must flag go upward."""
        from oasr.tune.build import thin_census

        pts = self._pts([128 * k for k in range(1, 65)])
        pts[10].must = True
        out = thin_census(pts, 12)
        assert sum(p.weight for p in out) == pytest.approx(64.0)
        assert sum(p.calls for p in out) == pytest.approx(128.0)
        prev = 0
        for p in out:
            assert p.weight == pytest.approx(sum(1.0 for q in pts if prev < q.M <= p.M))
            prev = p.M
        holder = next(p for p in out if p.M >= pts[10].M)
        assert holder.must and sum(p.must for p in out) == 1

    def test_the_heaviest_point_of_a_bin_is_kept(self):
        from oasr.tune.build import thin_census

        pts = self._pts([128 * k for k in range(1, 65)])
        pts[40].weight = 50.0
        assert pts[40].M in {p.M for p in thin_census(pts, 12)}

    def test_the_kept_points_follow_the_weight(self):
        """Half the slots go where the time is: a heavy top end is sampled densely
        (Conformer N=K=256 -- log bins alone measured two of the 18 heavy widths
        whose winner kept changing, and cost 7.5% weighted regret)."""
        from oasr.tune.build import thin_census

        pts = self._pts([16 * k for k in range(1, 129)])
        for p in pts[96:]:
            p.weight = 20.0
        kept = [p.M for p in thin_census(pts, 16)]
        assert sum(m > pts[95].M for m in kept) >= 6
        assert sum(m <= pts[95].M for m in kept) >= 6  # ...and the range is still covered

    def test_a_census_within_the_cap_is_untouched(self):
        from oasr.tune.build import thin_census

        pts = self._pts([16, 64, 256])
        out = thin_census(pts, 12)
        assert [(p.M, p.weight, p.calls) for p in out] == [
            (16, 1.0, 2.0),
            (64, 1.0, 2.0),
            (256, 1.0, 2.0),
        ]

    def test_a_thinned_plan_is_never_resumed_from_an_unthinned_record(self):
        from oasr.tune.build import BuildOptions, _points_digest

        pts = self._pts([128, 256])
        full = _points_digest(pts, BuildOptions())
        assert _points_digest(pts, BuildOptions(thin=True)) != full
        # ...and adding the knob left every existing (unthinned) checkpoint valid.
        assert _points_digest(pts, BuildOptions(thin=False)) == full
