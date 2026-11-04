"""The tuning database: file format, tiers, validators, and the selectors built on it.

Pure Python -- no kernel runs -- so it covers every arch family from any box.
"""

import json
import re

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

    def test_provenance_names_no_local_path(self):
        def strings(node):
            if isinstance(node, dict):
                for v in node.values():
                    yield from strings(v)
            elif isinstance(node, list):
                for v in node:
                    yield from strings(v)
            elif isinstance(node, str):
                yield node

        absolute = re.compile(r"(?:^|[\s:=\"'])(?:/|~/)[\w.-]+/")
        for _, _, path in self._files():
            provenance = json.loads(path.read_text())["provenance"]
            for s in strings(provenance):
                assert not absolute.search(s), f"{path}: local path in {s!r}"


class TestPortableProvenance:
    """A census records its checkpoint and manifest without the machine's layout."""

    def test_local_checkpoint_is_named_by_its_directory(self, tmp_path):
        from oasr.tune.census import model_name

        ckpt = tmp_path / "exp" / "u2pp_conformer"
        ckpt.mkdir(parents=True)
        assert model_name(str(ckpt)) == "u2pp_conformer"
        assert model_name(f"{ckpt}/") == "u2pp_conformer"

    def test_hub_snapshot_maps_back_to_its_repo_id(self, tmp_path):
        from oasr.tune.census import model_name

        snap = tmp_path / "hub" / "models--openai--whisper-tiny" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        assert model_name(str(snap)) == "openai/whisper-tiny"

    def test_hub_repo_id_is_kept(self):
        from oasr.tune.census import model_name

        assert model_name("openai/whisper-tiny") == "openai/whisper-tiny"

    def test_manifest_is_repo_relative_or_a_basename(self, tmp_path):
        from helpers import REPO_ROOT

        from oasr.tune.census import portable_path

        inside = REPO_ROOT / "benchmarks" / "manifests" / "ljspeech_200.jsonl"
        assert portable_path(str(inside)) == "benchmarks/manifests/ljspeech_200.jsonl"
        assert portable_path(str(tmp_path / "corpus" / "m.jsonl")) == "m.jsonl"


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

    def test_a_build_does_not_vouch_for_base_entries_it_did_not_measure(self, user_db, tmp_path):
        """The builder re-stamped its base with this machine's validators, so an
        entry measured under an older toolchain read as current after any build."""
        import oasr.jit.gemm as jg
        from oasr.tune.build import BuildOptions, build_gemm
        from oasr.tune.census import ShapePoint, ShapeSet

        sm = jg._get_target_sm()
        base = db.new_file("gemm", sm, jg.GEMM_LANE_BY_SM[sm])
        base.validators["soft"] = {"nvcc": "OLD"}
        base.configs["torch"] = {"kind": "torch"}
        untouched = jg.gemm_sig_key("gemm", 4096, 1024)
        base.entries[untouched] = db.Entry(regions=[(64, "torch")], source="jit")
        pts = [ShapePoint("gemm", 256, 256, "float16", 1, 16, 1.0, 1.0, True)]
        tf = build_gemm(
            ShapeSet(pts, 256 * 256 * 2),
            sm=sm,
            base=base,
            opts=BuildOptions(top_k=4, fill=False, log_path=str(tmp_path / "m.jsonl")),
        )
        assert any("measured under nvcc: saved 'OLD'" in r for r in tf.stale), tf.stale
        assert tf.entries[jg.gemm_sig_key("gemm", 256, 256)].soft is None

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


# ---------------------------------------------------------------------------
# User-tier writes merge under a lock, and keep what each entry was measured
# under (audit G19)
# ---------------------------------------------------------------------------


def _sig(n):
    return f"gemm|op=gemm|dt=fp16|N={n}|K=64"


def _file_with(path, entries, soft=None):
    tf = db.new_file("gemm", 120, "sm80_mma")
    if soft is not None:
        tf.validators["soft"] = dict(soft)
    tf.configs["torch"] = {"kind": "torch"}
    for n in entries:
        tf.entries[_sig(n)] = db.Entry(regions=[(64, "torch")], source="jit")
    db.save_file(tf, path)
    return tf


class TestUserTierWritesMerge:
    """``save_file`` was atomic, but the read-modify-write around it was not
    locked: a writer loaded the file, took minutes to measure, and wrote its
    copy back over whatever another process had written in between."""

    def test_a_flush_keeps_what_another_writer_wrote_meanwhile(self, user_db):
        path = db.user_path(120, "gemm")
        db.record_point("gemm", 120, "sm80_mma", _sig(512), 64, "torch", {"kind": "torch"})
        _file_with(path, [1024])  # another process's write, after this one's snapshot
        db.flush_user(reload_after=False)
        back = db.load_file(path, "gemm", 120)
        assert set(back.entries) == {_sig(512), _sig(1024)}

    def test_writers_flushing_together_all_land(self, user_db, tmp_path):
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        import oasr

        child = tmp_path / "child.py"
        child.write_text(
            "import sys, time\n"
            "from pathlib import Path\n"
            "from oasr.tune import database as db\n"
            "n, go = int(sys.argv[1]), Path(sys.argv[2])\n"
            "db.record_point('gemm', 120, 'sm80_mma', f'gemm|op=gemm|dt=fp16|N={n}|K=64', 64,"
            " 'torch', {'kind': 'torch'})\n"
            "Path(f'{go}.{n}').write_text('ready')\n"
            "while not go.exists():\n"
            "    time.sleep(0.002)\n"
            "db.flush_user(reload_after=False)\n"
        )
        # The tree under test, not the editable install a temp script would import.
        env = dict(
            os.environ,
            OASR_TUNE_USER_DB=str(user_db),
            PYTHONPATH=str(Path(oasr.__file__).resolve().parents[1]),
        )
        go = tmp_path / "go"
        ns = (256, 512, 768, 1024, 1280, 1536)
        procs = [
            subprocess.Popen([sys.executable, str(child), str(n), str(go)], env=env) for n in ns
        ]
        deadline = time.monotonic() + 120
        while not all(Path(f"{go}.{n}").exists() for n in ns):
            assert time.monotonic() < deadline, "the writers never got ready"
            time.sleep(0.02)
        go.write_text("go")
        assert all(p.wait(timeout=120) == 0 for p in procs)
        back = db.load_file(db.user_path(120, "gemm"), "gemm", 120)
        assert set(back.entries) == {_sig(n) for n in ns}

    def test_a_merged_build_keeps_entries_written_while_it_ran(self, user_db):
        path = db.user_path(120, "gemm")
        base = _file_with(path, [256, 512, 768, 1280])
        since = base.to_json()
        built = db.load_file(path, "gemm", 120)
        built.entries[_sig(256)] = db.Entry(regions=[(128, "torch")], source="aot")  # re-measured
        del built.entries[_sig(512)]  # nothing beat the default
        del built.entries[_sig(768)]
        # 1280: the build left it alone.  Meanwhile another process re-measured
        # 768 and 1280 and added 1024.
        other = db.load_file(path, "gemm", 120)
        other.entries[_sig(768)] = db.Entry(regions=[(32, "torch")], source="jit")
        other.entries[_sig(1280)] = db.Entry(regions=[(16, "torch")], source="jit")
        other.entries[_sig(1024)] = db.Entry(regions=[(64, "torch")], source="jit")
        db.save_file(other, path)
        db.save_merged(built, path, since=since)
        back = db.load_file(path, "gemm", 120)
        assert back.entries[_sig(256)].regions == [(128, "torch")]
        assert _sig(512) not in back.entries
        assert back.entries[_sig(768)].regions == [(32, "torch")], "a newer entry was dropped"
        assert back.entries[_sig(1280)].regions == [
            (16, "torch")
        ], "an untouched entry was reverted"
        assert _sig(1024) in back.entries


class TestSoftValidatorsDescribeTheEntries:
    """A write re-stamped the whole file with the writer's validators, so an
    entry measured under another kernel version read as current."""

    OLD, NEW = {"nvcc": "OLD"}, {"nvcc": "NEW"}

    @pytest.fixture
    def now_new(self, monkeypatch):
        monkeypatch.setattr(db, "current_soft_validators", lambda family: dict(self.NEW))

    def test_a_flush_does_not_vouch_for_entries_it_did_not_measure(self, user_db, now_new):
        path = db.user_path(120, "gemm")
        _file_with(path, [1024], soft=self.OLD)
        db.record_point("gemm", 120, "sm80_mma", _sig(512), 64, "torch", {"kind": "torch"})
        db.flush_user(reload_after=False)
        back = db.load_file(path, "gemm", 120)
        assert back.validators["soft"] == self.NEW
        assert back.entries[_sig(1024)].soft == self.OLD
        assert back.entries[_sig(512)].soft is None
        assert back.stale == ["1 entry measured under nvcc: saved 'OLD', now 'NEW'"]

    def test_an_old_reader_sees_the_new_entry_as_the_stale_one(self, user_db, monkeypatch):
        path = db.user_path(120, "gemm")
        _file_with(path, [1024], soft=self.OLD)
        monkeypatch.setattr(db, "current_soft_validators", lambda family: dict(self.NEW))
        db.record_point("gemm", 120, "sm80_mma", _sig(512), 64, "torch", {"kind": "torch"})
        db.flush_user(reload_after=False)
        monkeypatch.setattr(db, "current_soft_validators", lambda family: dict(self.OLD))
        back = db.load_file(path, "gemm", 120)
        # The file's NEW stamp covers the entry just measured, and only that one:
        # the old entry carries its own OLD, so to this reader it is current.
        assert back.entries[_sig(1024)].soft == self.OLD
        assert back.entries[_sig(512)].soft is None
        assert back.stale == ["nvcc: saved 'NEW', now 'OLD'"]

    def test_a_point_added_to_an_old_entry_leaves_it_stale(self, user_db, now_new):
        """Its other points are still the old measurements."""
        path = db.user_path(120, "gemm")
        _file_with(path, [1024], soft=self.OLD)
        db.record_point("gemm", 120, "sm80_mma", _sig(1024), 128, "torch", {"kind": "torch"})
        db.flush_user(reload_after=False)
        back = db.load_file(path, "gemm", 120)
        assert back.entries[_sig(1024)].soft == self.OLD

    def test_a_build_restamps_without_vouching(self, tmp_path, now_new):
        tf = _file_with(tmp_path / "f.json", [256, 512], soft=self.OLD)
        db.stamp_current(tf)
        tf.entries[_sig(512)] = db.Entry(regions=[(64, "torch")], source="aot")  # rebuilt
        assert tf.entries[_sig(256)].soft == self.OLD and tf.entries[_sig(512)].soft is None

    def test_an_entry_back_on_its_own_validators_drops_them(self, tmp_path, monkeypatch):
        tf = _file_with(tmp_path / "f.json", [256], soft=self.OLD)
        db.stamp_current(tf, self.NEW)
        db.stamp_current(tf, self.OLD)
        assert tf.entries[_sig(256)].soft is None and tf.validators["soft"] == self.OLD

    def test_a_file_without_per_entry_validators_loads_as_before(self, tmp_path):
        path = tmp_path / "f.json"
        tf = _file_with(path, [256], soft=self.OLD)
        assert "soft" not in json.loads(path.read_text())["entries"][_sig(256)]
        back = db.load_file(path, "gemm", 120)
        assert back.entries[_sig(256)].soft is None and back.to_json() == tf.to_json()
