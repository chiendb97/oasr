"""The tuning database: file format, tiers, validators, and the selectors built on it.

Pure Python -- no kernel runs -- so it covers every arch family from any box.
"""

import json

import pytest
import torch

from oasr.jit.core import _TARGET_SMS
from oasr.tune import database as db


@pytest.fixture
def user_db(tmp_path, monkeypatch):
    """Point the user tier at a temp dir for one test, and restore the epoch after."""
    monkeypatch.setenv("OASR_TUNE_USER_DB", str(tmp_path))
    db.reload()
    yield tmp_path
    monkeypatch.setenv("OASR_TUNE_USER_DB", "off")
    db.reload()


def _gemm_user_file(tmp_path, entries, configs, sm=120):
    tf = db.new_file("gemm", sm, "sm80_mma")
    tf.configs.update(configs)
    tf.entries.update(entries)
    db.save_file(tf, tmp_path / f"sm{sm}" / "gemm.json")
    db.reload()
    return tf


class TestSignatureKeys:
    def test_roundtrip(self):
        key = db.sig_key("gemm", op="gemm", dt="half", N=512, K=256)
        assert key == "gemm|op=gemm|dt=half|N=512|K=256"
        assert db.parse_sig_key(key) == (
            "gemm",
            {"op": "gemm", "dt": "half", "N": "512", "K": "256"},
        )

    def test_reserved_characters_are_refused(self):
        with pytest.raises(ValueError):
            db.sig_key("gemm", op="a|b")


class TestRegions:
    """Lookup rounds M *up*: a config serves only sizes at or below where it was measured."""

    REGIONS = [(64, "a"), (1024, "b"), (None, "c")]

    @pytest.mark.parametrize(
        "m,expect", [(1, "a"), (64, "a"), (65, "b"), (1024, "b"), (1025, "c"), (10**7, "c")]
    )
    def test_round_up_with_catch_all(self, m, expect):
        assert db.region_lookup(self.REGIONS, m) == expect

    def test_no_catch_all_misses_above_the_last_bound(self):
        assert db.region_lookup([(64, "a")], 65) is None

    def test_entry_rejects_unordered_regions(self):
        with pytest.raises(ValueError):
            db.Entry.from_json({"regions": [[1024, "b"], [64, "a"]]})
        with pytest.raises(ValueError):
            db.Entry.from_json({"regions": [[None, "c"], [64, "a"]]})

    def test_validity_envelope(self):
        e = db.Entry(regions=[(64, "a")], valid_m={"a": [1, 64]})
        assert e.valid("a", 64) and not e.valid("a", 65) and e.valid("z", 10**6)


class TestFileFormat:
    def test_roundtrip(self, tmp_path):
        tf = db.new_file("gemm", 120, "sm80_mma")
        tf.configs["x"] = {"kind": "torch"}
        tf.entries["gemm|op=gemm|dt=half|N=8|K=8"] = db.Entry(
            regions=[(None, "x")], notes=["why"], evidence={"M=4": {"x": [0.1, 0.0, 3]}}
        )
        path = tmp_path / "sm120" / "gemm.json"
        db.save_file(tf, path)
        back = db.load_file(path, "gemm", 120)
        assert back is not None
        assert back.to_json() == tf.to_json()

    def test_an_undefined_config_reference_is_rejected(self, tmp_path):
        tf = db.new_file("gemm", 120, "sm80_mma")
        tf.entries["gemm|op=gemm|dt=half|N=8|K=8"] = db.Entry(regions=[(None, "nope")])
        path = tmp_path / "f.json"
        path.write_text(json.dumps(tf.to_json()))
        assert db.load_file(path, "gemm", 120) is None

    @pytest.mark.parametrize(
        "patch",
        [{"schema": 1}, {"family": "conv1d"}, {"arch": {"family": 80, "lane": "sm80_mma"}}],
    )
    def test_hard_validator_mismatch_is_unusable(self, tmp_path, patch):
        tf = db.new_file("gemm", 120, "sm80_mma")
        data = {**tf.to_json(), **patch}
        path = tmp_path / "f.json"
        path.write_text(json.dumps(data))
        assert db.load_file(path, "gemm", 120) is None

    def test_soft_validator_mismatch_is_stale_but_served(self, tmp_path):
        tf = db.new_file("gemm", 120, "sm80_mma")
        tf.validators["soft"]["impl_hash"] = "0000000000000000"
        path = tmp_path / "f.json"
        db.save_file(tf, path)
        back = db.load_file(path, "gemm", 120)
        assert back is not None
        assert any("impl_hash" in r for r in back.stale)


class TestUserTierLocation:
    def test_off_disables(self, monkeypatch):
        monkeypatch.setenv("OASR_TUNE_USER_DB", "off")
        assert db.user_root() is None and db.user_path(120, "gemm") is None

    def test_a_path_moves_it(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OASR_TUNE_USER_DB", str(tmp_path))
        assert db.user_path(120, "gemm") == tmp_path / "sm120" / "gemm.json"

    def test_default_is_schema_scoped(self, monkeypatch):
        monkeypatch.delenv("OASR_TUNE_USER_DB", raising=False)
        assert db.user_root().name == f"v{db.SCHEMA}"


class TestShippedFiles:
    """Every shipped file loads, validates, and names only configs its arch builds."""

    def _files(self):
        for path in sorted(db.system_dir().glob("sm*/*.json")):
            yield int(path.parent.name[2:]), path.stem, path

    def test_there_is_at_least_one(self):
        assert list(self._files())

    def test_each_file_loads_and_matches_its_location(self):
        for sm, family, path in self._files():
            assert sm in _TARGET_SMS, f"{path}: sm{sm} is not a compiled family"
            tf = db.load_file(path, family, sm)
            assert tf is not None, f"{path} failed its hard validators"

    def test_every_gemm_config_decodes_and_is_compiled(self):
        import oasr.jit.gemm as jg

        for sm, family, path in self._files():
            if family != "gemm":
                continue
            tf = db.load_file(path, "gemm", sm)
            compiled = jg.get_unique_compile_configs(sm)
            for cid in tf.referenced_configs():
                choice = jg.gemm_config_from_params(tf.configs[cid], sm)
                if isinstance(choice, str) or choice is jg.GEMM_DEFAULT:
                    continue
                assert choice.compile_name in compiled, f"{path}: {cid}"
                assert jg.gemm_config_id(choice) == cid, f"{path}: id of {cid} is not canonical"


class TestGemmTiers:
    """``select_default_config`` reads the user tier first, then the shipped one."""

    def test_user_entry_overrides_the_system_one(self, user_db):
        import oasr.jit.gemm as jg

        shipped = jg.select_default_config("gemm", 64, 256, 2048, torch.bfloat16, 120)
        assert shipped == "torch"
        key = jg.gemm_sig_key("gemm", 256, 2048)
        _gemm_user_file(
            user_db, {key: db.Entry(regions=[(None, "default")])}, {"default": {"kind": "default"}}
        )
        assert jg.select_default_config("gemm", 64, 256, 2048, torch.bfloat16, 120) is (
            jg.GEMM_DEFAULT
        )
        # A signature the user file does not mention still reads the shipped table.
        assert jg.select_default_config("gemm", 64, 256, 4864, torch.bfloat16, 120) == "torch"

    def test_a_user_file_makes_an_untuned_arch_tuned(self, user_db):
        import oasr.jit.gemm as jg

        sm = next((s for s in _TARGET_SMS if s not in jg._GEMM_HEURISTIC_RULES), None)
        if sm is None:
            pytest.skip("every family has a shipped table")
        jg.reset_rule_misses()
        key = jg.gemm_sig_key("gemm", 256, 2048)
        _gemm_user_file(
            user_db, {key: db.Entry(regions=[(None, "torch")])}, {"torch": {"kind": "torch"}}, sm=sm
        )
        assert jg.select_default_config("gemm", 64, 256, 2048, torch.bfloat16, sm) == "torch"
        assert not jg.heuristic_inactive()

    def test_a_dtype_specific_entry_wins_over_half(self, user_db):
        import oasr.jit.gemm as jg

        _gemm_user_file(
            user_db,
            {
                jg.gemm_sig_key("gemm", 256, 2048, "fp16"): db.Entry(regions=[(None, "torch")]),
                jg.gemm_sig_key("gemm", 256, 2048, "half"): db.Entry(regions=[(None, "default")]),
            },
            {"torch": {"kind": "torch"}, "default": {"kind": "default"}},
        )
        assert jg.select_default_config("gemm", 64, 256, 2048, torch.float16, 120) == "torch"
        assert jg.select_default_config("gemm", 64, 256, 2048, torch.bfloat16, 120) is (
            jg.GEMM_DEFAULT
        )

    def test_an_uncompiled_user_config_falls_through(self, user_db, caplog):
        """A user file tuned against another build must not reach dispatch: the
        lookup falls through to the next tier -- here the shipped table."""
        import oasr.jit.gemm as jg

        bogus = {
            "kind": "cutlass",
            "lane": "sm80_mma",
            "tile": [48, 48, 48],
            "warp": [16, 16, 48],
            "stages": 7,
            "split_k": 1,
        }
        _gemm_user_file(
            user_db,
            {jg.gemm_sig_key("gemm", 256, 2048): db.Entry(regions=[(None, "bogus")])},
            {"bogus": bogus},
        )
        with caplog.at_level("WARNING", logger="oasr.jit.gemm"):
            choice = jg.select_default_config("gemm", 64, 256, 2048, torch.bfloat16, 120)
        assert choice == "torch", "the shipped entry for this shape"
        assert "does not compile" in caplog.text

    def test_reload_clears_the_dispatch_plans(self, user_db):
        import oasr.functionals.gemm as fg

        fg._PLANS[("x",)] = ("stale",)
        db.reload()
        assert not fg._PLANS

    def test_tier_counts_name_the_tier(self, user_db):
        import oasr.jit.gemm as jg

        db.reset_tier_counts()
        jg.select_default_config("gemm", 64, 256, 2048, torch.bfloat16, 120)
        jg.select_default_config("gemm", 64, 4242, 777, torch.bfloat16, 120)
        counts = db.tier_counts()
        assert counts[("gemm", "gemm", "system")] == 1
        assert counts[("gemm", "gemm", "default")] == 1
        assert "system=1" in db.tuning_report()


class TestConv1dTiers:
    def test_user_point_overrides_the_shipped_one(self, user_db):
        import oasr.jit.conv as jc

        shape = (1, 3000, 80, 384, 3, 1, 1, 1)
        shipped = jc.select_default_conv1d_config(*shape, torch.float16, 120)
        assert shipped is not jc.CONV2D_DEFAULT
        tf = db.new_file("conv1d", 120, "sm80_mma")
        tf.configs["default"] = {"kind": "default"}
        key = db.sig_key(
            "conv1d", op="conv1d", dt="fp16", Cin=80, Cout=384, k=3, pad=1, stride=1, dil=1
        )
        tf.entries[key] = db.Entry(points={"1x3000": "default"})
        db.save_file(tf, user_db / "sm120" / "conv1d.json")
        db.reload()
        assert jc.select_default_conv1d_config(*shape, torch.float16, 120) is jc.CONV2D_DEFAULT
        assert jc.select_default_conv1d_config(*shape, torch.bfloat16, 120) is not (
            jc.CONV2D_DEFAULT
        )


@pytest.mark.cuda
class TestAutotunePublishes:
    """``oasr.autotune()`` writes GEMM winners where the production path reads them."""

    def test_a_tuned_shape_is_served_outside_the_context(self, user_db):
        import oasr
        import oasr.jit.gemm as jg
        from oasr.jit.core import _get_target_sm
        from oasr.tune import autotune, clear_cache

        sm = _get_target_sm()
        M, N, K = 48, 4096, 1024  # a width no shipped table covers
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(N, K, device="cuda", dtype=torch.float16)
        clear_cache()
        with autotune(True):
            oasr.gemm(A, B)
        path = db.user_path(sm, "gemm")
        assert path.is_file(), "the winner was not written to the user tier"
        tf = db.load_file(path, "gemm", sm)
        key = jg.gemm_sig_key("gemm", N, K, "fp16")
        entry = tf.entries[key]
        ((m_hi, cid),) = entry.regions
        assert m_hi >= M, "the region must cover the measured M (it rounds up)"
        choice = jg.select_default_config("gemm", M, N, K, torch.float16, sm)
        assert jg.gemm_config_id(choice) == cid
        # No catch-all: a larger M was never measured and is not served this winner.
        assert jg.select_default_config("gemm", 10 * m_hi, N, K, torch.float16, sm) is (
            jg.GEMM_DEFAULT
        )


def _model_section():
    return {
        "gemm": {
            "num_sms": 170,
            "smem_budget": 100352,
            "max_threads_per_sm": 1536,
            "dram_gbps": 1500.0,
            "tensor_tflops": 220.0,
            "launch_us_graph": 0.8,
            "coeffs": {},
            "fitted_rows": {},
        }
    }


class TestModelTier:
    """A shape no entry covers is ranked by the cost model the arch's file ships."""

    def _with_model(self, user_db):
        tf = db.new_file("gemm", 120, "sm80_mma")
        tf.model = _model_section()
        db.save_file(tf, user_db / "sm120" / "gemm.json")
        db.reload()

    def test_an_uncovered_aligned_shape_gets_a_compiled_model_pick(self, user_db):
        import oasr.jit.gemm as jg

        self._with_model(user_db)
        jg.reset_rule_misses()
        db.reset_tier_counts()
        choice = jg.select_default_config("gemm", 64, 4096, 1024, torch.float16, 120)
        assert choice is not jg.GEMM_DEFAULT and not isinstance(choice, str)
        assert choice.compile_name in jg.get_production_configs(120)
        assert db.tier_counts()[("gemm", "gemm", "model")] == 1
        assert ("gemm", 4096, 1024) in jg.rule_misses(), "a model pick is still a miss"

    def test_the_switch_turns_it_off(self, user_db, monkeypatch):
        import oasr.jit.gemm as jg

        self._with_model(user_db)
        monkeypatch.setattr(jg, "_MODEL_FALLBACK", "0")
        assert jg.select_default_config("gemm", 64, 4096, 1024, torch.float16, 120) is (
            jg.GEMM_DEFAULT
        )

    def test_unaligned_and_other_ops_keep_the_default(self, user_db):
        import oasr.jit.gemm as jg

        self._with_model(user_db)
        assert jg.select_default_config("gemm", 64, 4242, 777, torch.float16, 120) is (
            jg.GEMM_DEFAULT
        )
        assert jg.select_default_config("bmm", 64, 4096, 1024, torch.float16, 120) is (
            jg.GEMM_DEFAULT
        )

    def test_without_a_model_nothing_changes(self, user_db, monkeypatch, tmp_path):
        """No tier ships a model (the shipped tier hidden too): the default stays."""
        import oasr.jit.gemm as jg

        monkeypatch.setattr(db, "system_dir", lambda: tmp_path / "no_system")
        db.reload()
        try:
            assert jg.select_default_config("gemm", 64, 4096, 1024, torch.float16, 120) is (
                jg.GEMM_DEFAULT
            )
        finally:
            monkeypatch.undo()
            db.reload()

    def test_a_covered_shape_ignores_the_model(self, user_db):
        import oasr.jit.gemm as jg

        self._with_model(user_db)
        assert jg.select_default_config("gemm", 64, 256, 2048, torch.bfloat16, 120) == "torch"

    def test_the_pick_is_a_pure_function_of_the_shape(self, user_db):
        import oasr.jit.gemm as jg

        self._with_model(user_db)
        a = jg.select_default_config("gemm", 96, 4096, 1024, torch.float16, 120)
        b = jg.select_default_config("gemm", 96, 4096, 1024, torch.float16, 120)
        assert a == b


class TestTuningConfig:
    def test_coerces_a_mapping(self):
        from oasr.engine.config import TuningConfig

        assert TuningConfig.coerce({"mode": "prewarm", "budget_s": 5}).budget_s == 5
        assert TuningConfig.coerce(None).mode == "off"

    def test_rejects_an_unknown_mode(self):
        from oasr.engine.config import TuningConfig

        with pytest.raises(ValueError):
            TuningConfig(mode="sometimes")


@pytest.mark.cuda
class TestBuildEndToEnd:
    """census points in, a tuning file with regions and a model out."""

    def test_a_tiny_build(self, user_db, tmp_path):
        import oasr.jit.gemm as jg
        from oasr.tune.build import BuildOptions, build_gemm
        from oasr.tune.census import ShapePoint, ShapeSet

        pts = [ShapePoint("gemm", 256, 256, "float16", 1, m, 1.0, 1.0, True) for m in (16, 256)]
        tf = build_gemm(
            ShapeSet(pts, 256 * 256 * 2),
            sm=jg._get_target_sm(),
            opts=BuildOptions(top_k=4, fill=False, log_path=str(tmp_path / "m.jsonl")),
        )
        entry = tf.entries.get(jg.gemm_sig_key("gemm", 256, 256))
        assert entry is not None and entry.regions and entry.regions[-1][0] is None
        assert tf.model.get("gemm", {}).get("num_sms")
        assert (tmp_path / "m.jsonl").read_text().count("\n") > 4
        for _hi, cid in entry.regions:
            assert cid in tf.configs

    def test_a_resumed_build_measures_nothing_and_matches(self, user_db, tmp_path, monkeypatch):
        """A checkpoint makes a preempted build resume, not re-measure (and not drift)."""
        import oasr.jit.gemm as jg
        from oasr.tune import build as B
        from oasr.tune.census import ShapePoint, ShapeSet

        def shapes(weight=1.0):
            pts = [
                ShapePoint("gemm", 256, 256, "float16", 1, m, calls=1.0, weight=weight, must=True)
                for m in (16, 256)
            ]
            pts.append(
                ShapePoint("gemm", 512, 256, "float16", 1, 64, calls=1.0, weight=1.0, must=True)
            )
            return ShapeSet(pts, 256 * 256 * 2)

        def opts():
            return B.BuildOptions(
                top_k=4,
                fill=False,
                log_path=str(tmp_path / "m.jsonl"),
                checkpoint_path=str(tmp_path / "ck.jsonl"),
            )

        sm = jg._get_target_sm()
        first = B.build_gemm(shapes(), sm=sm, opts=opts())

        def measured_again(*_a, **_k):
            raise AssertionError("a checkpointed signature was measured again")

        monkeypatch.setattr(B, "measure_signature", measured_again)
        again = B.build_gemm(shapes(), sm=sm, opts=opts())
        as_json = lambda tf: {k: e.to_json() for k, e in tf.entries.items()}  # noqa: E731
        assert as_json(again) == as_json(first)
        assert again.configs == first.configs
        # Different census points are a different question: never served from the record.
        with pytest.raises(AssertionError, match="measured again"):
            B.build_gemm(shapes(weight=2.0), sm=sm, opts=opts())


class TestConv1dRegions:
    def test_a_region_over_m_serves_other_batch_sizes(self, user_db):
        import oasr.jit.conv as jc

        tf = db.new_file("conv1d", 120, "sm80_mma")
        tf.configs["t"] = {
            "kind": "cutlass",
            "lane": "sm80_mma",
            "tile": [64, 128, 64],
            "warp": [32, 64, 64],
            "stages": 3,
        }
        key = db.sig_key(
            "conv1d", op="conv1d", dt="fp16", Cin=80, Cout=384, k=3, pad=1, stride=1, dil=1
        )
        tf.entries[key] = db.Entry(regions=[(8192, "t")])
        db.save_file(tf, user_db / "sm120" / "conv1d.json")
        db.reload()
        # B=2, T=3000 -> M = 6000: inside the region; the shipped exact point is B=1 only.
        cfg = jc.select_default_conv1d_config(2, 3000, 80, 384, 3, 1, 1, 1, torch.float16, 120)
        assert (cfg.block_m, cfg.block_n) == (64, 128)
        # M = 3 * 3000 = 9000 lies above the region: the default.
        big = jc.select_default_conv1d_config(3, 3000, 80, 384, 3, 1, 1, 1, torch.float16, 120)
        assert big is jc.CONV2D_DEFAULT


class TestModelTierNumerics:
    def test_never_serial_split_k(self, user_db):
        """An unmeasured tier has no numerics gate, and serial split-K rounds its
        partials through the output dtype once per slice."""
        import oasr.jit.gemm as jg

        tf = db.new_file("gemm", 120, "sm80_mma")
        tf.model = _model_section()
        db.save_file(tf, user_db / "sm120" / "gemm.json")
        db.reload()
        for M, N, K in ((64, 128, 256), (256, 32, 128), (16, 4096, 4096), (8, 256, 8192)):
            c = jg.select_default_config("gemm", M, N, K, torch.bfloat16, 120)
            if isinstance(c, str) or c is jg.GEMM_DEFAULT:
                continue
            assert c.split_k == 1 or c.parallel_split_k, (M, N, K, c.name)
