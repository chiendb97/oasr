"""The int8 scaled GEMM's tuning-DB tier: ``oasr.jit.scaled_mm`` + ``oasr.tune``.

The resolver tests pass ``sm``/``num_sms`` explicitly and need no GPU; the
capture and build tests run the kernels.
"""

import json
import logging

import pytest
import torch

import oasr.jit.scaled_mm as jsm
from oasr.tune import database as db

SM = 120
NUM_SMS = 170
#: A width no shipped file covers, so an absent user entry means the heuristic.
N, K = 1024, 512

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="runs the int8 kernels")


@pytest.fixture
def user_db(tmp_path, monkeypatch):
    """Point the user tier at a temp dir for one test, and restore the epoch after."""
    monkeypatch.setenv("OASR_TUNE_USER_DB", str(tmp_path))
    db.reload()
    db.reset_tier_counts()
    yield tmp_path
    monkeypatch.setenv("OASR_TUNE_USER_DB", "off")
    db.reload()


def _user_file(root, entries, configs, sm=SM):
    tf = db.new_file("scaled_mm", sm, jsm.SCALED_MM_LANE)
    tf.configs.update(configs)
    tf.entries.update(entries)
    db.save_file(tf, root / f"sm{sm}" / "scaled_mm.json")
    db.reload()
    db.reset_tier_counts()
    return tf


def _params(cid, sm=SM):
    if cid == jsm.HEURISTIC_ID:
        return {"kind": jsm.HEURISTIC_ID}
    return jsm.config_to_params(jsm.get_production_configs(sm)[cid])


def _entry_file(root, op, n, k, regions, sm=SM):
    key = jsm.scaled_mm_sig_key(op, n, k)
    configs = {cid: _params(cid, sm) for _, cid in regions}
    return _user_file(root, {key: db.Entry(regions=regions)}, configs, sm)


BIG = "b128x128x64_w64x64x64_s5"
SMALL = "b16x64x128_w16x64x64_s5"
STREAM_K = "b64x64x128_w64x64x64_s5_sk"


def _select(m, n=N, k=K):
    return jsm.select_config(m, n, k, SM, NUM_SMS)


def _select_quant(m, n=N, k=K):
    return jsm.select_quant_config(m, n, k, SM, NUM_SMS)


class TestCodec:
    @pytest.mark.parametrize("sm", sorted(jsm.SCALED_MM_LANE_BY_SM))
    def test_every_production_config_round_trips(self, sm):
        for cid, cfg in jsm.get_production_configs(sm).items():
            params = json.loads(json.dumps(jsm.config_to_params(cfg)))
            assert jsm.config_from_params(params) == cfg
            assert jsm.config_id(cfg) == cid == cfg.compile_name
            assert jsm.choice_from_params(jsm.choice_to_params(cfg)) == cfg

    def test_the_heuristic_sentinel_round_trips(self):
        params = jsm.choice_to_params(jsm.HEURISTIC_ID)
        assert params == {"kind": "heuristic"}
        assert jsm.choice_from_params(params) == jsm.HEURISTIC_ID

    def test_a_config_of_another_lane_is_refused(self):
        params = dict(jsm.config_to_params(jsm.get_production_configs(SM)[BIG]), lane="sm80_mma")
        with pytest.raises(ValueError):
            jsm.config_from_params(params)

    def test_the_family_is_known_to_the_database(self):
        assert "scaled_mm" in db.KERNEL_FAMILIES
        assert db.current_soft_validators("scaled_mm").get("impl_hash") == jsm.scaled_mm_impl_hash()


class TestTiers:
    def test_an_entry_is_served_and_rounds_up(self, user_db):
        _entry_file(user_db, "scaled_mm", N, K, [(64, BIG), (1024, jsm.HEURISTIC_ID)])
        assert jsm.config_id(_select(64)) == BIG
        assert _select(65) == jsm.heuristic_config(65, N, SM, NUM_SMS)
        # No catch-all: above the measured range is the heuristic, and uncovered.
        assert _select(4096) == jsm.heuristic_config(4096, N, SM, NUM_SMS)
        assert jsm.covering_tier("scaled_mm", 64, N, K, SM) == "user"
        assert jsm.covering_tier("scaled_mm", 1024, N, K, SM) == "user"
        assert jsm.covering_tier("scaled_mm", 1025, N, K, SM) is None
        counts = db.tier_counts()
        assert counts[("scaled_mm", "scaled_mm", "user")] == 2
        assert counts[("scaled_mm", "scaled_mm", "heuristic")] == 1
        report = db.tuning_report()
        assert "scaled_mm.scaled_mm" in report and "user=2" in report and "heuristic=1" in report

    def test_without_an_entry_the_heuristic_serves(self, user_db):
        for m in (1, 17, 300, 5000):
            assert _select(m) == jsm.heuristic_config(m, N, SM, NUM_SMS)
            assert not _select(m).stream_k
        assert jsm.covering_tier("scaled_mm", 17, N, K, SM) is None
        assert db.tier_counts() == {("scaled_mm", "scaled_mm", "heuristic"): 4}

    def test_a_stream_k_entry_serves_the_fp_output_op(self, user_db):
        _entry_file(user_db, "scaled_mm", N, K, [(None, STREAM_K)])
        assert _select(512).stream_k

    def test_the_quantizing_op_never_serves_stream_k(self, user_db, caplog):
        _entry_file(user_db, "scaled_mm_quant", N, K, [(None, STREAM_K)])
        with caplog.at_level(logging.WARNING, logger="oasr.jit.scaled_mm"):
            cfg = _select_quant(512)
        assert not cfg.stream_k and cfg == jsm.heuristic_config(512, N, SM, NUM_SMS)
        assert "Stream-K" in caplog.text
        assert jsm.covering_tier("scaled_mm_quant", 512, N, K, SM) is None
        # The fp-output op has its own key: the quantizing entry does not leak into it.
        assert _select(512) == jsm.heuristic_config(512, N, SM, NUM_SMS)

    def test_an_uncompiled_config_falls_through_with_one_warning(self, user_db, caplog):
        bogus = {
            "kind": "cutlass",
            "lane": jsm.SCALED_MM_LANE,
            "tile": [48, 48, 48],
            "warp": [16, 16, 48],
            "stages": 7,
            "stream_k": False,
        }
        key = jsm.scaled_mm_sig_key("scaled_mm", N, K)
        _user_file(user_db, {key: db.Entry(regions=[(None, "bogus")])}, {"bogus": bogus})
        with caplog.at_level(logging.WARNING, logger="oasr.jit.scaled_mm"):
            for m in (8, 900, 3000):
                assert _select(m) == jsm.heuristic_config(m, N, SM, NUM_SMS)
        warned = [r for r in caplog.records if "production set" in r.getMessage()]
        assert len(warned) == 1, caplog.text
        assert jsm.covering_tier("scaled_mm", 8, N, K, SM) is None
        assert db.tier_counts()[("scaled_mm", "scaled_mm", "heuristic")] == 3

    def test_an_unreadable_config_falls_through(self, user_db):
        key = jsm.scaled_mm_sig_key("scaled_mm", N, K)
        _user_file(user_db, {key: db.Entry(regions=[(None, "x")])}, {"x": {"kind": "torch"}})
        assert _select(64) == jsm.heuristic_config(64, N, SM, NUM_SMS)

    def test_the_user_tier_wins_over_the_shipped_one(self, user_db):
        shipped = db.tiers("scaled_mm", SM).system
        if shipped is None:
            pytest.skip("no shipped sm120 scaled_mm file")
        key = next(iter(shipped.entries))
        _, fields = db.parse_sig_key(key)
        op, n, k = fields["op"], int(fields["N"]), int(fields["K"])
        assert jsm.covering_tier(op, 4096, n, k, SM) == "system"
        _entry_file(user_db, op, n, k, [(None, SMALL)])
        assert jsm.covering_tier(op, 4096, n, k, SM) == "user"
        pick = (_select if op == "scaled_mm" else _select_quant)(4096, n, k)
        assert jsm.config_id(pick) == SMALL

    def test_a_reload_drops_the_memo(self, user_db):
        before = _select(64)
        assert jsm._MEMO
        _entry_file(user_db, "scaled_mm", N, K, [(None, BIG)])  # reloads
        assert jsm.config_id(_select(64)) == BIG != jsm.config_id(before)
        db.reload()
        assert not jsm._MEMO

    def test_a_repeated_shape_is_a_dict_hit(self, user_db, monkeypatch):
        calls = []
        real = jsm._resolve

        def counting(*a):
            calls.append(a)
            return real(*a)

        monkeypatch.setattr(jsm, "_resolve", counting)
        for _ in range(3):
            _select(77)
            _select_quant(77)
        assert len(calls) == 2  # one per (op, shape)

    def test_signature_tier_ignores_m(self, user_db):
        _entry_file(user_db, "scaled_mm", N, K, [(64, BIG)])
        assert jsm.signature_tier("scaled_mm", N, K, SM) == "user"
        assert jsm.signature_tier("scaled_mm_quant", N, K, SM) is None
        assert jsm.signature_tier("gemm", N, K, SM) is None

    def test_the_report_can_leave_a_family_out(self, user_db):
        _select(64)
        assert "scaled_mm" in db.tuning_report()
        assert "scaled_mm" not in db.tuning_report(families=["gemm", "conv1d"])


class TestShippedFiles:
    """Every shipped ``scaled_mm[_fp8].json`` names only its arch's production configs."""

    def _files(self):
        for family, fmt in (("scaled_mm", "int8"), ("scaled_mm_fp8", "fp8")):
            for path in sorted(db.system_dir().glob(f"sm*/{family}.json")):
                yield int(path.parent.name[2:]), path, family, fmt

    def test_the_fp8_file_ships_beside_the_int8_one(self):
        families = {family for _, _, family, _ in self._files()}
        assert families == {"scaled_mm", "scaled_mm_fp8"}

    def test_each_config_decodes_and_is_built(self):
        for sm, path, family, fmt in self._files():
            tf = db.load_file(path, family, sm)
            assert tf is not None, f"{path} failed its hard validators"
            built = jsm.get_production_configs(sm, fmt)
            for cid in tf.referenced_configs():
                choice = jsm.choice_from_params(tf.configs[cid])
                if choice == jsm.HEURISTIC_ID:
                    assert cid == jsm.HEURISTIC_ID
                    continue
                assert built.get(cid) == choice, f"{path}: {cid}"
                assert jsm.config_id(choice) == cid, f"{path}: id of {cid} is not canonical"

    def test_every_entry_is_served_from_the_file(self):
        for sm, path, family, fmt in self._files():
            tf = db.load_file(path, family, sm)
            for key, entry in tf.entries.items():
                _, f = db.parse_sig_key(key)
                assert f["op"] in jsm.FORMAT_OPS[fmt]
                if jsm.is_quant_op(f["op"]):
                    for _, cid in entry.regions:
                        assert not tf.configs[cid].get("stream_k"), f"{path}: {key}"
                for m_hi, _ in entry.regions:
                    m = 1 << 20 if m_hi is None else m_hi
                    assert jsm.covering_tier(f["op"], m, int(f["N"]), int(f["K"]), sm) == "system"


@pytest.mark.cuda
@_cuda
class TestCapture:
    def test_both_ops_are_recorded_once_each(self):
        import oasr
        import oasr.functionals.scaled_mm as fsm
        from oasr.quantization.kernels.lanes import CutlassSm80Int8Lane
        from oasr.tune.capture import GemmShapeRecorder, capture_gemm_shapes

        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randint(-127, 128, (3, 40, 256), generator=g, dtype=torch.int8, device="cuda")
        b = torch.randint(-127, 128, (512, 256), generator=g, dtype=torch.int8, device="cuda")
        a_scale = torch.full((120,), 1e-3, device="cuda")
        b_scale = torch.full((512,), 1e-3, device="cuda")
        out_mul = torch.full((512,), 4.0, device="cuda")
        orig = (oasr.scaled_mm, fsm.scaled_mm, fsm.scaled_mm_quant)
        rec = GemmShapeRecorder()
        with capture_gemm_shapes(rec):
            oasr.scaled_mm(a, b, a_scale, b_scale, None, "swish")
            lane = CutlassSm80Int8Lane()
            lane.run(a, b, a_scale, b_scale, None, None, torch.float16)
            lane.run_quant(a, b, a_scale, b_scale, None, None, out_mul, torch.float16)
            fsm.scaled_mm(a=a, b=b, a_scale=a_scale, b_scale=b_scale)
        assert (oasr.scaled_mm, fsm.scaled_mm, fsm.scaled_mm_quant) == orig, "patches undone"
        stats = {st.op: st for st in rec.aggregate()}
        assert set(stats) == {"scaled_mm", "scaled_mm_quant"}
        for op, calls in (("scaled_mm", 3), ("scaled_mm_quant", 1)):
            st = stats[op]
            assert (st.N, st.K, st.dtype, st.batch) == (512, 256, "int8", 1)
            assert st.call_count == calls and dict(st.m_counts) == {120: calls}


class TestCensus:
    def test_an_int8_point_is_one_byte_wide(self):
        from oasr.tune import census as C

        # A memory-bound shape: half the bytes, (about) half the time.
        assert C._estimate_ms("scaled_mm", 8192, 256, 16, 1, 1) < C._estimate_ms(
            "gemm", 8192, 256, 16, 1, 2
        )
        assert C._weight_bytes("scaled_mm", 256, 512, "int8", 1) == 256 * 512
        assert C._weight_bytes("gemm", 256, 512, "bfloat16", 1) == 2 * 256 * 512
        assert C.kernel_family("scaled_mm_quant") == "scaled_mm"
        assert C.kernel_family("bmm") == "gemm"

    def test_int8_points_do_not_move_the_gemm_cut(self):
        from oasr.tune.census import ShapePoint, _keep

        def pts(op, dtype, weights):
            return {
                (op, 256, 256, dtype, 1, m): ShapePoint(op, 256, 256, dtype, 1, m, w, w)
                for m, w in weights.items()
            }

        fp = pts("gemm", "bfloat16", {16: 50.0, 32: 30.0, 64: 15.0, 128: 4.0, 256: 1.0})
        q = pts("scaled_mm", "int8", {16: 1000.0, 32: 0.01})
        alone = _keep(fp, 0.9)
        both = _keep({**fp, **q}, 0.9)
        assert {sk for sk in both if sk[0] == "gemm"} == alone
        assert ("scaled_mm", 256, 256, "int8", 1, 16) in both

    def test_a_recorded_int8_capture_becomes_census_points(self, tmp_path):
        from oasr.tune.capture import GemmShapeRecorder
        from oasr.tune.census import from_capture

        rec = GemmShapeRecorder()
        rec.record("scaled_mm", 64, 512, 256, "int8")
        rec.record("scaled_mm_quant", 64, 2048, 256, "int8")
        path = tmp_path / "cap.json"
        rec.to_json(str(path))
        ss = from_capture(str(path))
        assert {p.op for p in ss.points} == {"scaled_mm", "scaled_mm_quant"}
        assert ss.working_set_bytes == 512 * 256 + 2048 * 256


class TestTooling:
    def test_prebuild_compiles_the_int8_module_for_an_int8_census(self):
        from oasr.tune.prebuild import modules_for

        assert [n for n, _ in modules_for({"scaled_mm"})] == ["scaled_mm"]
        assert "scaled_mm" in [n for n, _ in modules_for({"gemm", "scaled_mm_quant"})]
        assert "scaled_mm" not in [n for n, _ in modules_for({"gemm"})]

    def test_the_int8_file_goes_beside_the_gemm_file(self, tmp_path):
        from oasr.tune.cli import _resolve_out

        assert _resolve_out(str(tmp_path / "gemm.json"), SM, "scaled_mm") == (
            tmp_path / "scaled_mm.json"
        )
        assert _resolve_out(str(tmp_path / "x.json"), SM, "scaled_mm") == (
            tmp_path / "x.scaled_mm.json"
        )
        assert _resolve_out(str(tmp_path / "x.json"), SM, "gemm") == tmp_path / "x.json"
        assert _resolve_out("system", SM, "scaled_mm") == db.system_path(SM, "scaled_mm")

    def test_build_refuses_an_int8_family_without_int8_points(self, tmp_path, capsys):
        from oasr.tune.census import ShapePoint, ShapeSet
        from oasr.tune.cli import main

        path = tmp_path / "fp.json"
        ShapeSet([ShapePoint("gemm", 256, 256, "bfloat16", 1, 64, 1.0, 1.0, True)], 0).save(
            str(path)
        )
        rc = main(["build", "--census", str(path), "--family", "scaled_mm", "--out", "x.json"])
        assert rc == 1 and "no int8" in capsys.readouterr().out


@pytest.mark.cuda
@_cuda
class TestBuild:
    """census points in, a loadable ``scaled_mm`` file whose regions resolve out."""

    def _shapes(self):
        from oasr.tune.census import ShapePoint, ShapeSet

        pts = [ShapePoint("scaled_mm", 256, 256, "int8", 1, m, 1.0, 1.0, True) for m in (16, 2048)]
        pts.append(ShapePoint("scaled_mm_quant", 512, 256, "int8", 1, 64, 1.0, 1.0, True))
        return ShapeSet(pts, 256 * 256 + 512 * 256)

    def _opts(self, tmp_path):
        from oasr.tune.build import BuildOptions

        return BuildOptions(
            fill=False,
            log_path=str(tmp_path / "m.jsonl"),
            checkpoint_path=str(tmp_path / "ck.jsonl"),
        )

    def test_a_tiny_build_loads_and_resolves(self, user_db, tmp_path, monkeypatch):
        from oasr.tune import build_scaled_mm as B

        sm = jsm._process_sm()
        tf = B.build_scaled_mm(self._shapes(), sm=sm, opts=self._opts(tmp_path))
        path = db.user_path(sm, "scaled_mm")
        db.save_file(tf, path)
        back = db.load_file(path, "scaled_mm", sm)
        assert back is not None and back.to_json() == tf.to_json()
        assert set(back.entries) == {
            jsm.scaled_mm_sig_key("scaled_mm", 256, 256),
            jsm.scaled_mm_sig_key("scaled_mm_quant", 512, 256),
        }
        for entry in back.entries.values():
            assert entry.regions and entry.regions[-1][0] is None
            assert all(cid in back.configs for _, cid in entry.regions)
            assert entry.evidence and entry.notes
        assert (tmp_path / "m.scaled_mm.jsonl").read_text().count("\n") >= 3
        db.reload()
        for op, n, m in (
            ("scaled_mm", 256, 16),
            ("scaled_mm", 256, 2048),
            ("scaled_mm_quant", 512, 64),
        ):
            assert jsm.covering_tier(op, m, n, 256) == "user"
            select = jsm.select_config if op == "scaled_mm" else jsm.select_quant_config
            cfg = select(m, n, 256)
            assert jsm.get_production_configs(sm)[jsm.config_id(cfg)] == cfg
            if op == "scaled_mm_quant":
                assert not cfg.stream_k

        # A checkpointed build resumes without measuring and writes the same entries.
        def measured_again(*_a, **_k):
            raise AssertionError("a checkpointed signature was measured again")

        monkeypatch.setattr(B, "measure_signature", measured_again)
        again = B.build_scaled_mm(self._shapes(), sm=sm, opts=self._opts(tmp_path))
        assert {k: e.to_json() for k, e in again.entries.items()} == {
            k: e.to_json() for k, e in tf.entries.items()
        }

    def test_numerics_gate_measures_every_variant_bit_identical(self):
        from oasr.tune.build_scaled_mm import candidates, measure_case

        sm = jsm._process_sm()
        res = measure_case("scaled_mm", 200, 256, 256, sm=sm)
        assert not res.rejected
        assert set(res.results) == set(candidates("scaled_mm", sm)) | {jsm.HEURISTIC_ID}
        heur = res.results[jsm.HEURISTIC_ID]
        twin = res.results[jsm.config_id(jsm.heuristic_config(200, 256, sm))]
        assert heur.median_ms == twin.median_ms and heur.eliminated_at is None
        q = measure_case("scaled_mm_quant", 200, 256, 256, sm=sm)
        assert not any(
            jsm.get_production_configs(sm)[n].stream_k for n in q.results if n != "heuristic"
        )


# -----------------------------------------------------------------------------
# The fp8 format: its own family (scaled_mm_fp8.json), ops and lane
# -----------------------------------------------------------------------------

FP8_BIG = "b128x128x64_w64x64x64_s4_e4m3"
FP8_SMALL = "b16x64x128_w16x16x128_s4_e4m3"


def _fp8_file(root, op, n, k, regions, sm=SM):
    configs = {
        cid: (
            {"kind": jsm.HEURISTIC_ID}
            if cid == jsm.HEURISTIC_ID
            else jsm.config_to_params(jsm.get_production_configs(sm, "fp8")[cid])
        )
        for _, cid in regions
    }
    tf = db.new_file("scaled_mm_fp8", sm, jsm.LANE_ID["fp8"])
    tf.configs.update(configs)
    tf.entries[jsm.scaled_mm_sig_key(op, n, k)] = db.Entry(regions=regions)
    db.save_file(tf, root / f"sm{sm}" / "scaled_mm_fp8.json")
    db.reload()
    db.reset_tier_counts()
    return tf


def _fp8_capable() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return jsm.lane_available(major * 10 + minor, "fp8")


class TestFp8Family:
    @pytest.mark.parametrize("sm", sorted(jsm.SCALED_MM_FP8_LANE_BY_SM))
    def test_every_fp8_config_round_trips(self, sm):
        cfgs = jsm.get_production_configs(sm, "fp8")
        assert cfgs
        for cid, cfg in cfgs.items():
            params = json.loads(json.dumps(jsm.config_to_params(cfg)))
            assert params["lane"] == jsm.LANE_ID["fp8"]
            assert jsm.config_from_params(params) == cfg and cfg.fp8 and cid.endswith("_e4m3")

    def test_no_fp8_lane_below_sm89(self):
        assert not jsm.lane_available(80, "fp8") and not jsm.lane_available(86, "fp8")
        assert jsm.lane_available(89, "fp8") and jsm.is_native_lane(120, "fp8")
        assert not jsm.is_native_lane(90, "fp8")

    def test_ops_keys_and_families(self):
        assert "scaled_mm_fp8" in db.KERNEL_FAMILIES
        assert jsm.op_format("scaled_mm_fp8_quant") == "fp8"
        assert jsm.scaled_mm_sig_key("scaled_mm_fp8", 256, 512).startswith("scaled_mm_fp8|")
        assert jsm.scaled_mm_sig_key("scaled_mm", 256, 512).startswith("scaled_mm|")

    def test_an_fp8_entry_is_served_to_the_fp8_op_only(self, user_db):
        _fp8_file(user_db, "scaled_mm_fp8", N, K, [(64, FP8_BIG), (None, jsm.HEURISTIC_ID)])
        cfg = jsm.select_config(64, N, K, SM, NUM_SMS, fmt="fp8")
        assert jsm.config_id(cfg) == FP8_BIG
        assert jsm.select_config(65, N, K, SM, NUM_SMS, fmt="fp8") == jsm.heuristic_config(
            65, N, SM, NUM_SMS, fmt="fp8"
        )
        assert jsm.covering_tier("scaled_mm_fp8", 64, N, K, SM) == "user"
        # The int8 op of the same shape has no entry.
        assert jsm.covering_tier("scaled_mm", 64, N, K, SM) is None
        assert not _select(64).fp8
        counts = db.tier_counts()  # M=65 is the entry's heuristic region: still the user tier
        assert counts[("scaled_mm_fp8", "scaled_mm_fp8", "user")] == 2

    def test_the_heuristic_picks_fp8_tiles(self):
        for m in (1, 300, 16384):
            cfg = jsm.heuristic_config(m, N, SM, NUM_SMS, fmt="fp8")
            assert cfg.fp8 and cfg.block_k == cfg.warp_k

    def test_an_int8_config_in_an_fp8_entry_falls_through(self, user_db, caplog):
        tf = db.new_file("scaled_mm_fp8", SM, jsm.LANE_ID["fp8"])
        tf.configs[BIG] = jsm.config_to_params(jsm.get_production_configs(SM)[BIG])
        tf.entries[jsm.scaled_mm_sig_key("scaled_mm_fp8", N, K)] = db.Entry(regions=[(None, BIG)])
        db.save_file(tf, user_db / f"sm{SM}" / "scaled_mm_fp8.json")
        db.reload()
        with caplog.at_level(logging.WARNING, logger="oasr.jit.scaled_mm"):
            cfg = jsm.select_config(512, N, K, SM, NUM_SMS, fmt="fp8")
        assert cfg.fp8 and "int8 config" in caplog.text

    def test_census_and_prebuild_know_the_fp8_ops(self):
        from oasr.tune import census as C
        from oasr.tune.prebuild import modules_for

        assert C.kernel_family("scaled_mm_fp8") == "scaled_mm_fp8"
        assert C.kernel_family("scaled_mm_fp8_quant") == "scaled_mm_fp8"
        assert C._weight_bytes("scaled_mm_fp8", 256, 512, "float8_e4m3fn", 1) == 256 * 512
        assert [n for n, _ in modules_for({"scaled_mm_fp8"})] == ["scaled_mm_fp8"]
        assert "scaled_mm_fp8" not in [n for n, _ in modules_for({"scaled_mm"})]


@pytest.mark.cuda
@pytest.mark.skipif(not _fp8_capable(), reason="fp8 tensor cores need sm89+")
class TestFp8CaptureAndBuild:
    def test_an_fp8_call_is_recorded_under_the_fp8_op(self):
        import oasr.functionals.scaled_mm as fsm
        from oasr.tune.capture import GemmShapeRecorder, capture_gemm_shapes

        g = torch.Generator(device="cuda").manual_seed(0)
        a = (torch.randn(40, 256, generator=g, device="cuda") * 8).to(torch.float8_e4m3fn)
        b = (torch.randn(512, 256, generator=g, device="cuda") * 8).to(torch.float8_e4m3fn)
        a_scale = torch.full((40,), 1e-3, device="cuda")
        b_scale = torch.full((512,), 1e-3, device="cuda")
        rec = GemmShapeRecorder()
        with capture_gemm_shapes(rec):
            fsm.scaled_mm(a, b, a_scale, b_scale, None, "swish")
            fsm.scaled_mm_quant(
                a, b, a_scale, b_scale, None, None, torch.ones(512, device="cuda"), torch.float16
            )
        stats = {st.op: st for st in rec.aggregate()}
        assert set(stats) == {"scaled_mm_fp8", "scaled_mm_fp8_quant"}
        assert stats["scaled_mm_fp8"].dtype == "float8_e4m3fn"

    def test_numerics_gate_measures_every_fp8_variant_bit_identical(self):
        from oasr.tune.build_scaled_mm import candidates, measure_case

        sm = jsm._process_sm()
        for op in ("scaled_mm_fp8", "scaled_mm_fp8_quant"):
            res = measure_case(op, 200, 256, 512, sm=sm, issue_cost=False)
            assert not res.rejected, res.rejected
            assert set(res.results) == set(candidates(op, sm)) | {jsm.HEURISTIC_ID}
