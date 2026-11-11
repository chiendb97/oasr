"""Measured path selection: the ``recurrent`` and ``gated_mlp`` tuning families.

``oasr.jit.recurrent_select`` / ``oasr.jit.mlp_select`` replace hand-timed
cut-offs -- the LSTM's scalar-or-tensor-core table, the fused step's band, lane
order and tile ladder, the gated MLP's one-m-tile band and candidate list -- with
regions a tuning file names.  The codec, signatures and census are pure Python;
dispatch, the extrapolation accounting and the build run small layers on CUDA.
"""

import logging

import pytest
import torch

import oasr
import oasr.functionals.recurrent as fr
from oasr.jit import (
    measured,
    mlp as jm,
    mlp_select as M,
    recurrent_cute as rc,
    recurrent_select as R,
)
from oasr.tune import database as db, runners as rn
from oasr.tune.build_fused import (
    FusedShapeSet,
    census_family,
    fused_census,
    model_dims,
    recurrent_candidates,
)

_CUDA = torch.cuda.is_available()


def _sm():
    from oasr.jit.core import _get_target_sm

    return _get_target_sm() if _CUDA else 0


_LANES_HERE = _CUDA and _sm() in (80, 86, 89, 120)
needs_lanes = pytest.mark.skipif(not _LANES_HERE, reason="needs the fused C++/CuTeDSL lanes")


# =============================================================================
# Codec, signatures, runners (no device)
# =============================================================================


class TestCodec:
    @pytest.mark.parametrize(
        "choice, cid",
        [
            (R.RecChoice("scalar"), "scalar"),
            (R.RecChoice("gemm"), "gemm"),
            (R.RecChoice("gemm", tactic=2), "gemm_t2"),
            (R.RecChoice("gemm", tactic=4, split=2), "gemm_t4_s2"),
            (R.RecChoice("cxx", tile=3), "cxx_t3"),
            (R.RecChoice("cute", tile=0), "cute_t0"),
            (R.AUTO_ID, "auto"),
        ],
    )
    def test_recurrent_round_trip(self, choice, cid):
        assert R.config_id(choice) == cid
        assert R.choice_from_params(R.choice_to_params(choice)) == choice

    @pytest.mark.parametrize(
        "choice, cid",
        [
            (M.MlpChoice("gemm"), "gemm"),
            (M.MlpChoice("cxx", tile=5), "cxx_t5"),
            (M.MlpChoice("cute", tile=1), "cute_t1"),
            (M.AUTO_ID, "auto"),
        ],
    )
    def test_gated_mlp_round_trip(self, choice, cid):
        assert M.config_id(choice) == cid
        assert M.choice_from_params(M.choice_to_params(choice)) == choice

    def test_unknown_kinds_raise(self):
        with pytest.raises(ValueError):
            R.choice_from_params({"kind": "cudnn_rnn"})
        with pytest.raises(ValueError):
            M.choice_from_params({"kind": "triton"})

    def test_signatures(self):
        rec = R.RecShape("lstm", "bfloat16", 640, 512, 1, 16)
        assert R.sig_key(rec) == "recurrent|op=layer|kind=lstm|dt=half|H=640|I=512|T=step"
        seq = R.RecShape("rnn_tanh", "float16", 640, 512, 7, 16)
        assert R.sig_key(seq).endswith("kind=rnn_tanh|dt=half|H=640|I=512|T=seq")
        mlp = M.MlpShape("bfloat16", "silu", False, 11008, 4096, 3)
        assert M.sig_key(mlp) == "gated_mlp|op=gated_mlp|dt=half|act=silu|bias=0|N=11008|K=4096"

    def test_both_lanes_share_the_tile_lists(self):
        """One index names one tile in both lanes, or a choice would mean two things."""
        from oasr.jit import gated_mlp, recurrent_step

        for i, t in enumerate(gated_mlp.TILES):
            assert M.tile_spec(i) == (
                t.block_m, t.block_n, t.block_k, t.stages, t.threads, t.warps_n
            )  # fmt: skip
        cute_tiles = {tile for _, _, tile in rc._TILES}
        cxx_tiles = {
            (t.block_m, t.block_n, t.block_k, t.stages, t.threads, t.warps_n)
            for t in recurrent_step.TILES
        }
        assert cute_tiles <= cxx_tiles


class TestRunners:
    def test_every_kind_has_an_owner(self):
        for family, kinds in (
            ("recurrent", ("scalar", "auto", "gemm", "cxx", "cute")),
            ("gated_mlp", ("gemm", "auto", "cxx", "cute")),
        ):
            for kind in kinds:
                r = rn.runner_for_kind(family, kind)
                assert r is not None, (family, kind)
                assert not r.traits.external, "every path here is an OASR kernel"

    def test_families_are_tuning_families(self):
        assert {"recurrent", "gated_mlp"} <= set(db.KERNEL_FAMILIES)


class TestCensus:
    def test_model_dims_reads_the_layers(self):
        model = torch.nn.Module()
        model.lstm = oasr.layers.LSTM(48, 64, num_layers=2)
        model.rnn = oasr.layers.RNN(32, 64, nonlinearity="relu")
        model.mlp = oasr.layers.GatedMLP(64, 128, activation="silu")
        dims = model_dims(model)
        assert sorted((d["op"], d["hidden"], d["input_size"]) for d in dims["recurrent"]) == [
            ("lstm", 64, 48),
            ("lstm", 64, 64),  # the second layer reads the first's hidden state
            ("rnn_relu", 64, 32),
        ]
        assert dims["gated_mlp"] == [{"activation": "silu", "bias": False, "n": 128, "k": 64}]

    def test_ladder_and_file(self, tmp_path):
        ss = fused_census(
            "recurrent",
            [{"op": "lstm", "hidden": 64, "input_size": 64}],
            ladder=[1, 8],
            seq_lens=[1, 4],
        )
        assert len(ss.points) == 4
        assert len(ss.signatures()) == 2  # step and sequence
        path = str(tmp_path / "r.json")
        ss.save(path)
        assert census_family(path) == "recurrent"
        assert FusedShapeSet.load(path).points == ss.points


# =============================================================================
# Dispatch (CUDA)
# =============================================================================


@pytest.fixture
def user_db(tmp_path, monkeypatch):
    """A user tier in a temp dir and an empty system tier, for one test."""
    system = tmp_path / "system"
    system.mkdir()
    monkeypatch.setattr(db, "system_dir", lambda: system)
    monkeypatch.setenv("OASR_TUNE_USER_DB", str(tmp_path))
    db.reload()
    db.reset_tier_counts()
    yield tmp_path
    monkeypatch.undo()
    monkeypatch.setenv("OASR_TUNE_USER_DB", "off")
    db.reload()


def _write(root, family, codec, key_of, entries):
    """A user-tier file of *family*: ``entries`` is ``{shape: [(hi, choice), ...]}``."""
    tf = db.new_file(family, _sm(), family)
    for shape, regions in entries.items():
        rs = []
        for hi, choice in regions:
            cid = codec.config_id(choice)
            tf.configs[cid] = codec.choice_to_params(choice)
            rs.append((hi, cid))
        tf.entries[key_of(shape)] = db.Entry(regions=rs)
    db.save_file(tf, root / f"sm{_sm()}" / f"{family}.json")
    db.reload()


def _lstm(hidden=64, width=64, batch=16, seed=0):
    torch.manual_seed(seed)
    layer = oasr.layers.LSTM(width, hidden).cuda().half()
    x = torch.randn(1, batch, width, device="cuda", dtype=torch.float16)
    return layer, x


def _reference(layer, x):
    ref = oasr.layers.LSTM(layer.input_size, layer.hidden_size).cuda()
    ref.load_state_dict({k: v.float() for k, v in layer.state_dict().items()})
    with torch.no_grad():
        out, _ = ref(x.float())
    return out


class _Calls:
    """Count which recurrent entry point the layer took."""

    def __init__(self, monkeypatch):
        self.calls = []
        for name in ("lstm_layer", "lstm_gemm_layer"):
            real = getattr(oasr, name)

            def spy(*a, _real=real, _name=name, **kw):
                self.calls.append((_name, kw.get("_tactic"), kw.get("_step_route") is not None))
                return _real(*a, **kw)

            monkeypatch.setattr(oasr, name, spy)
        real_step = rc.step_for_tile

        def step_spy(**kw):
            self.calls.append(("step", kw["lane"], kw["tile"]))
            return real_step(**kw)

        monkeypatch.setattr(rc, "step_for_tile", step_spy)


@needs_lanes
class TestRecurrentDispatch:
    def test_no_table_runs_todays_rules(self, monkeypatch, user_db):
        layer, x = _lstm(batch=16)  # H=64 B=16 T=1: the hand table says scalar
        spy = _Calls(monkeypatch)
        with torch.no_grad():
            layer(x)
        assert spy.calls == [("lstm_layer", None, False)]
        assert R._TABLE.inactive_epoch == db.epoch()

    def test_a_region_names_the_scalar_path(self, monkeypatch, user_db):
        layer, x = _lstm(batch=256)  # the hand table says tensor cores here
        shape = R.RecShape("lstm", "float16", 64, 64, 1, 256)
        _write(user_db, "recurrent", R, R.sig_key, {shape: [(None, R.RecChoice("scalar"))]})
        spy = _Calls(monkeypatch)
        with torch.no_grad():
            out, _ = layer(x)
        assert [c[0] for c in spy.calls] == ["lstm_layer"]
        assert (out.float() - _reference(layer, x)).abs().max().item() < 2e-2
        assert db.tier_counts()[("recurrent", "lstm", "user")] >= 1

    def test_a_region_names_a_fused_lane_and_tile(self, monkeypatch, user_db):
        layer, x = _lstm(batch=8)
        shape = R.RecShape("lstm", "float16", 64, 64, 1, 8)
        _write(user_db, "recurrent", R, R.sig_key, {shape: [(None, R.RecChoice("cxx", tile=2))]})
        spy = _Calls(monkeypatch)
        with torch.no_grad():
            out, _ = layer(x)
        assert ("step", "cxx", 2) in spy.calls
        assert spy.calls[0][0] == "lstm_gemm_layer"
        assert (out.float() - _reference(layer, x)).abs().max().item() < 2e-2

    def test_a_region_names_the_gemm_path_and_its_tactic(self, monkeypatch, user_db):
        layer, x = _lstm(batch=64)
        shape = R.RecShape("lstm", "float16", 64, 64, 1, 64)
        _write(user_db, "recurrent", R, R.sig_key, {shape: [(None, R.RecChoice("gemm", tactic=1))]})
        dispatched = []
        real = fr._dispatch_lstm_gemm

        def spy(*a, **kw):
            dispatched.append(a[-1] if len(a) == 7 else kw.get("tactic"))
            return real(*a, **kw)

        monkeypatch.setattr(fr, "_dispatch_lstm_gemm", spy)
        with torch.no_grad():
            out, _ = layer(x)
        assert dispatched == [(1, 1)]
        assert (out.float() - _reference(layer, x)).abs().max().item() < 2e-2

    def test_a_named_mode_ignores_the_table(self, monkeypatch, user_db):
        layer, x = _lstm(batch=256)
        shape = R.RecShape("lstm", "float16", 64, 64, 1, 256)
        _write(user_db, "recurrent", R, R.sig_key, {shape: [(None, R.RecChoice("scalar"))]})
        monkeypatch.setattr(rc, "_MODE", "off")
        spy = _Calls(monkeypatch)
        with torch.no_grad():
            layer(x)
        assert [c[0] for c in spy.calls] == ["lstm_gemm_layer"]

    def test_unservable_choice_falls_through(self, monkeypatch, user_db, caplog):
        """A CuTeDSL tile for a hidden width that is not whole K tiles: warned once."""
        layer, x = _lstm(hidden=72, width=72, batch=8)
        shape = R.RecShape("lstm", "float16", 72, 72, 1, 8)
        _write(user_db, "recurrent", R, R.sig_key, {shape: [(None, R.RecChoice("cute", tile=0))]})
        with caplog.at_level(logging.WARNING, logger="oasr.jit.selection"):
            with torch.no_grad():
                out, _ = layer(x)
        assert any("falling through" in r.message for r in caplog.records)
        assert (out.float() - _reference(layer, x)).abs().max().item() < 2e-2

    def test_a_covered_shape_is_not_an_extrapolation(self, monkeypatch, user_db):
        """Off the card the tables were timed on, an uncovered shape is counted and a
        covered one is not -- the hand tables are never consulted for it."""
        here = measured.current_machine()
        elsewhere = measured.Machine("Other GPU", here.sm, here.sms + 1)
        monkeypatch.setattr(measured, "current_machine", lambda: elsewhere)
        from oasr.layers import recurrent as lr

        lr._table_prefers_tensor_core.cache_clear()
        measured.reset_extrapolations()
        rc._ROUTE.clear()
        layer, x = _lstm(batch=24)
        shape = R.RecShape("lstm", "float16", 64, 64, 1, 24)
        _write(user_db, "recurrent", R, R.sig_key, {shape: [(None, R.RecChoice("gemm"))]})
        with torch.no_grad():
            layer(x)
        assert not measured.extrapolations()
        layer2, x2 = _lstm(hidden=96, width=96, batch=24)  # not in the file
        with torch.no_grad():
            layer2(x2)
        assert any("_TENSOR_CORE" in t for t in measured.extrapolations())
        measured.reset_extrapolations()
        lr._table_prefers_tensor_core.cache_clear()

    def test_candidates_cover_every_path(self):
        shape = R.RecShape("lstm", "float16", 640, 640, 1, 16)
        cands = recurrent_candidates(shape)
        assert {"scalar", "gemm", "gemm_t0", "cxx_t0", "cute_t0"} <= set(cands)
        seq = recurrent_candidates(R.RecShape("lstm", "float16", 640, 640, 8, 16))
        assert not any(c.startswith(("cxx", "cute")) for c in seq), "fused step is T=1 only"


@needs_lanes
class TestGatedMlpDispatch:
    def _x(self, rows, n=128, k=64, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(rows, k, device="cuda", dtype=torch.float16, generator=g)
        wg = torch.randn(n, k, device="cuda", dtype=torch.float16, generator=g) * k**-0.5
        wu = torch.randn(n, k, device="cuda", dtype=torch.float16, generator=g) * k**-0.5
        return x, wg, wu

    def test_a_region_declines_fusion_inside_the_band(self, user_db):
        x, wg, wu = self._x(8)
        assert oasr.gated_mlp_available(x, wg)
        shape = M.MlpShape("float16", "silu", False, 128, 64, 8)
        _write(user_db, "gated_mlp", M, M.sig_key, {shape: [(None, M.MlpChoice("gemm"))]})
        assert not oasr.gated_mlp_available(x, wg)

    def test_a_region_fuses_past_the_band_on_a_named_tile(self, user_db):
        x, wg, wu = self._x(256)
        assert not oasr.gated_mlp_available(x, wg), "today's rule: one m-tile only"
        shape = M.MlpShape("float16", "silu", False, 128, 64, 256)
        _write(user_db, "gated_mlp", M, M.sig_key, {shape: [(None, M.MlpChoice("cxx", tile=5))]})
        from oasr.functionals.mlp import gated_mlp_backend

        assert gated_mlp_backend(x, wg) == "cxx"
        out = oasr.gated_mlp(x, wg, wu)
        ref = torch.nn.functional.silu(x.float() @ wg.float().t()) * (x.float() @ wu.float().t())
        assert (out.float() - ref).abs().max().item() < 2e-2

    def test_a_named_backend_ignores_the_table(self, monkeypatch, user_db):
        x, wg, _ = self._x(8)
        shape = M.MlpShape("float16", "silu", False, 128, 64, 8)
        _write(user_db, "gated_mlp", M, M.sig_key, {shape: [(None, M.MlpChoice("gemm"))]})
        monkeypatch.setattr(jm, "_GATED_MLP_MODE", "always")
        jm._ROUTE.clear()
        assert oasr.gated_mlp_available(x, wg)
        jm._ROUTE.clear()

    def test_unknown_kind_does_not_poison_the_file(self, user_db):
        x, wg, _ = self._x(8)
        x2, wg2, _ = self._x(8, n=256)
        tf = db.new_file("gated_mlp", _sm(), "gated_mlp")
        tf.configs["mystery"] = {"kind": "triton"}
        tf.configs["gemm"] = {"kind": "gemm"}
        tf.entries[M.sig_key(M.MlpShape("float16", "silu", False, 128, 64, 8))] = db.Entry(
            regions=[(None, "mystery")]
        )
        tf.entries[M.sig_key(M.MlpShape("float16", "silu", False, 256, 64, 8))] = db.Entry(
            regions=[(None, "gemm")]
        )
        db.save_file(tf, user_db / f"sm{_sm()}" / "gated_mlp.json")
        db.reload()
        assert oasr.gated_mlp_available(x, wg), "undecodable: today's rule (fuse)"
        assert not oasr.gated_mlp_available(x2, wg2), "the other entry still serves"


@needs_lanes
class TestBuild:
    def test_recurrent_and_gated_mlp_builds_write_servable_files(self, user_db):
        from oasr.tune import bench
        from oasr.tune.build import BuildOptions
        from oasr.tune.build_fused import build_gated_mlp, build_recurrent

        opts = BuildOptions(
            log_path=str(user_db / "log.jsonl"),
            policy=bench.BenchPolicy(stage_rounds=(2, 2, 3), solo=False),
        )
        rec = fused_census(
            "recurrent",
            [{"op": "lstm", "hidden": 64, "input_size": 64}],
            ladder=[2, 128],
            seq_lens=[1],
        )
        tf = build_recurrent(rec, sm=_sm(), opts=opts)
        (key,) = tf.entries
        for _, cid in tf.entries[key].regions:
            R.choice_from_params(tf.configs[cid])
        mlp = fused_census(
            "gated_mlp", [{"activation": "silu", "bias": False, "n": 128, "k": 64}], ladder=[4, 128]
        )
        tf2 = build_gated_mlp(mlp, sm=_sm(), opts=opts)
        (key2,) = tf2.entries
        for _, cid in tf2.entries[key2].regions:
            M.choice_from_params(tf2.configs[cid])
        for t in (tf, tf2):
            db.save_file(t, user_db / f"sm{_sm()}" / f"{t.family}.json")
        db.reload()
        assert R.resolve(R.RecShape("lstm", "float16", 64, 64, 1, 2))[0] == "user"
        assert M.resolve(M.MlpShape("float16", "silu", False, 128, 64, 4))[0] == "user"


class TestShippedFiles:
    @pytest.mark.parametrize(
        "path",
        sorted(db.system_dir().glob("sm*/recurrent.json"))
        + sorted(db.system_dir().glob("sm*/gated_mlp.json")),
        ids=str,
    )
    def test_loads_and_decodes(self, path):
        family = path.stem
        codec = R if family == "recurrent" else M
        tf = db.load_file(path, family, int(path.parent.name[2:]))
        assert tf is not None and tf.entries
        for cid, params in tf.configs.items():
            assert codec.config_id(codec.choice_from_params(params)) == cid
