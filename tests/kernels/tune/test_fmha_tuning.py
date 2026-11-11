"""Measured attention selection: the ``fmha`` tuning family (``oasr.jit.attention_select``).

The codec, signatures, point planning and the runner registry are pure Python.
Dispatch, the census recorder and the build run small attention calls on CUDA,
fp16, against the lanes this GPU has (the CuTeDSL and C++ kernels on sm80 /
sm86 / sm89 / sm120).
"""

import logging
import math

import pytest
import torch

import oasr.functionals.attention as fa
import oasr.jit.attention as ja
from oasr.jit import attention_select as S
from oasr.tune import database as db, runners as rn
from oasr.tune.attention_census import (
    AttnPoint,
    AttnShapeSet,
    capture_attention_shapes,
    is_attention_census,
)
from oasr.tune.build_fmha import plan_points

_CUDA = torch.cuda.is_available()


def _sm():
    if not _CUDA:
        return 0
    from oasr.jit.core import _get_target_sm

    return _get_target_sm()


#: Both fused lanes are built for these.
_LANES_HERE = _CUDA and _sm() in (80, 86, 89, 120) and ja.select_backend() == "cute"
needs_lanes = pytest.mark.skipif(not _LANES_HERE, reason="needs the CuTeDSL and C++ lanes")


# =============================================================================
# Codec and signatures (no device)
# =============================================================================


class TestCodec:
    @pytest.mark.parametrize(
        "choice, cid",
        [
            (S.AttnChoice("cute", 32, 16, 64), "cute_m32n16t64"),
            (S.AttnChoice("cute"), "cute_m64n64t128"),
            (S.AttnChoice("cxx"), "cxx"),
            (S.AttnChoice("cxx", splits=4), "cxx_s4"),
            (S.AttnChoice("sdpa"), "sdpa"),
            (S.AUTO_ID, "auto"),
        ],
    )
    def test_round_trip(self, choice, cid):
        assert S.config_id(choice) == cid
        params = S.choice_to_params(choice)
        assert params["kind"] == (choice if choice == S.AUTO_ID else choice.lane)
        assert S.choice_from_params(params) == choice

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown fmha config kind"):
            S.choice_from_params({"kind": "flashinfer"})

    def test_bad_lane_refused(self):
        with pytest.raises(ValueError):
            S.AttnChoice("triton")

    @pytest.mark.parametrize(
        "t_q, cls", [(1, "1"), (2, "16"), (16, "16"), (17, "32"), (64, "64"), (65, "L")]
    )
    def test_q_class(self, t_q, cls):
        assert S.q_class(t_q) == cls

    @pytest.mark.parametrize("b, bucket", [(1, 1), (2, 2), (3, 4), (16, 16), (17, 32), (64, 64)])
    def test_batch_bucket(self, b, bucket):
        assert S.batch_bucket(b) == bucket

    def test_sig_key_names_every_axis(self):
        shape = S.AttnShape("bfloat16", 64, 6, 2, "causal", True, 16, True, 20, 1, 300, True)
        assert S.sig_key(shape) == (
            "fmha|op=fmha|dt=half|D=64|H=6|Hkv=2|mask=causal|bias=1|ba=1|kv=paged16|ss=1|q=1|B=32"
        )

    def test_bias_alignment_splits_the_signature(self):
        """Both lanes branch on the bias row's alignment, so it is keyed (and moot unbiased)."""
        odd = S.AttnShape("float16", 64, 4, 4, "none", True, 0, False, 16, 225, 225, False)
        even = S.AttnShape("float16", 64, 4, 4, "none", True, 0, False, 16, 226, 226, True)
        assert S.sig_key(odd) != S.sig_key(even)
        unbiased = S.AttnShape("float16", 64, 4, 4, "none", False, 0, False, 16, 1, 300, True)
        assert "|ba=0|" in S.sig_key(unbiased)


class TestRunners:
    def test_every_kind_has_an_owner(self):
        owners = {k: rn.runner_for_kind("fmha", k) for k in ("cute", "auto", "cxx", "sdpa")}
        assert {k: r.name for k, r in owners.items()} == {
            "cute": "oasr_cute",
            "auto": "oasr_cute",
            "cxx": "oasr_cxx",
            "sdpa": "sdpa",
        }

    def test_traits(self):
        assert rn.get_runner("fmha", "sdpa").traits.external
        assert not rn.get_runner("fmha", "oasr_cute").traits.external
        assert rn.get_runner("fmha", "oasr_cute").traits.compiled
        assert rn.get_runner("fmha", "oasr_cxx").traits.compiled

    def test_family_is_a_tuning_family(self):
        assert "fmha" in db.KERNEL_FAMILIES


class TestPlanPoints:
    def _pt(self, t_k, weight=1.0, batch=8, t_q=1, block_size=0, calls=1.0):
        return AttnPoint(
            "float16", 64, 6, 6, "none", False, block_size, False, batch, t_q, t_k,
            calls=calls, weight=weight,
        )  # fmt: skip

    def test_same_key_length_measured_once_at_the_heaviest(self):
        pts = [self._pt(128, 1.0, batch=5), self._pt(128, 3.0, batch=7), self._pt(256)]
        plan = plan_points(pts, cap=8, fill=False)
        assert [p.t_k for p in plan] == [128, 256]
        assert plan[0].batch == 7 and plan[0].weight == pytest.approx(4.0)

    def test_thinning_keeps_the_ends_and_folds_weight_upward(self):
        pts = [self._pt(t, 1.0) for t in range(16, 16 * 41, 16)]
        plan = plan_points(pts, cap=6, fill=False)
        assert plan[0].t_k == 16 and plan[-1].t_k == 16 * 40
        assert len(plan) <= 6
        assert sum(p.weight for p in plan) == pytest.approx(40.0)

    def test_paged_fill_points_are_whole_tiles(self):
        pts = [self._pt(64, block_size=16), self._pt(1536, block_size=16)]
        plan = plan_points(pts, cap=8)
        assert all(p.t_k % 64 == 0 for p in plan)
        assert len(plan) <= 8

    def test_self_attention_fill_keeps_t_q_equal_t_k(self):
        pts = [self._pt(64, t_q=64), self._pt(512, t_q=512)]
        plan = plan_points(pts, cap=6)
        assert all(p.t_q == p.t_k for p in plan)


class TestCensusFile:
    def test_round_trip_and_detection(self, tmp_path):
        ss = AttnShapeSet([TestPlanPoints()._pt(300)], {"source": "test"})
        path = str(tmp_path / "a.json")
        ss.save(path)
        assert is_attention_census(path)
        back = AttnShapeSet.load(path)
        assert back.points == ss.points
        assert list(back.signatures()) == [ss.points[0].sig]


# =============================================================================
# Dispatch (CUDA)
# =============================================================================


@pytest.fixture
def user_db(tmp_path, monkeypatch):
    """A user tier in a temp dir and an empty system tier, for one test.

    The system tier is hidden so a test sees only the table it writes, not the
    shipped ``fmha.json``; the epoch is restored after.
    """
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


def _write_table(root, entries, configs=None):
    """A user-tier ``fmha.json``: ``entries`` is ``{shape: [(t_k_hi, choice), ...]}``."""
    tf = db.new_file("fmha", _sm(), "attention")
    for shape, regions in entries.items():
        rs = []
        for hi, choice in regions:
            cid = S.config_id(choice)
            tf.configs[cid] = S.choice_to_params(choice)
            rs.append((hi, cid))
        tf.entries[S.sig_key(shape)] = db.Entry(regions=rs)
    tf.configs.update(configs or {})
    db.save_file(tf, root / f"sm{_sm()}" / "fmha.json")
    db.reload()


class _Spy:
    """Count which lane ``fmha`` called, and with what configuration."""

    def __init__(self, monkeypatch):
        self.calls = []
        cxx, cute = fa._call_cxx, fa._call_cute_dsl

        def spy_cxx(*a, **kw):
            self.calls.append(("cxx", kw.get("splits")))
            return cxx(*a, **kw)

        def spy_cute(*a, **kw):
            self.calls.append(("cute", kw.get("tile")))
            return cute(*a, **kw)

        monkeypatch.setattr(fa, "_call_cxx", spy_cxx)
        monkeypatch.setattr(fa, "_call_cute_dsl", spy_cute)


def _dense(B=16, H=4, T_q=16, T_k=300, D=64, bias=True, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = torch.float16
    q = torch.randn(B, H, T_q, D, device="cuda", dtype=dt, generator=g)
    k = torch.randn(B, H, T_k, D, device="cuda", dtype=dt, generator=g)
    v = torch.randn(B, H, T_k, D, device="cuda", dtype=dt, generator=g)
    lens = torch.randint(T_k // 2, T_k + 1, (B,), device="cuda", dtype=torch.int32, generator=g)
    b = None
    if bias:
        b = (torch.randn(B, H, T_q, T_k, device="cuda", generator=g) * 2).to(dt)
    shape = S.AttnShape("float16", D, H, H, "none", bias, 0, False, B, T_q, T_k, T_k % 2 == 0)
    return {"q": q, "k": k, "v": v, "cache_seqlens": lens, "attn_bias": b}, shape


def _run(ops, **kw):
    return fa.fmha(ops["q"], ops["k"], ops["v"], softmax_scale=1 / math.sqrt(8.0 * 8), **{
        **{k: v for k, v in ops.items() if k not in ("q", "k", "v")}, **kw
    })  # fmt: skip


def _close(a, b):
    return (a.float() - b.float()).abs().max().item() < 2e-2


@needs_lanes
class TestDispatch:
    def test_no_table_changes_nothing(self, monkeypatch, user_db):
        """Without an fmha file the lookup answers None and the default lane runs as before."""
        spy = _Spy(monkeypatch)
        ops, _ = _dense()
        _run(ops)
        assert spy.calls == [("cute", None)]
        assert S._INACTIVE_EPOCH == db.epoch()
        assert not any(f == "fmha" for f, _, _ in db.tier_counts())

    def test_region_selects_lane_and_configuration(self, monkeypatch, user_db):
        ops, shape = _dense(T_k=300)
        ops_short, shape_short = _dense(T_k=100, seed=1)
        assert S.sig_key(shape) == S.sig_key(shape_short)
        _write_table(
            user_db,
            {shape: [(128, S.AttnChoice("cxx", splits=4)), (None, S.AttnChoice("cute", 32, 32, 64))]},
        )  # fmt: skip
        spy = _Spy(monkeypatch)
        long_out, short_out = _run(ops), _run(ops_short)
        assert spy.calls == [("cute", (32, 32, 64)), ("cxx", 4)]
        assert _close(long_out, _run(ops, backend="sdpa"))
        assert _close(short_out, _run(ops_short, backend="sdpa"))
        assert db.tier_counts()[("fmha", "fmha", "user")] == 2
        counts = db.runner_counts()
        assert counts[("fmha", "fmha", "oasr_cute")] == 1
        assert counts[("fmha", "fmha", "oasr_cxx")] == 1

    def test_a_named_lane_ignores_the_table(self, monkeypatch, user_db):
        ops, shape = _dense()
        _write_table(user_db, {shape: [(None, S.AttnChoice("cxx", splits=2))]})
        spy = _Spy(monkeypatch)
        _run(ops, backend="cute")
        monkeypatch.setattr(ja, "_BACKEND_MODE", "cute")
        _run(ops)
        assert spy.calls == [("cute", None), ("cute", None)]

    def test_named_choice_runs_that_configuration(self, monkeypatch):
        ops, _ = _dense()
        spy = _Spy(monkeypatch)
        _run(ops, backend=S.AttnChoice("cute", 32, 16, 64))
        _run(ops, backend=S.AttnChoice("cxx", splits=1))
        assert spy.calls == [("cute", (32, 16, 64)), ("cxx", 1)]

    def test_unservable_choice_falls_through(self, monkeypatch, user_db, caplog):
        """A CuTeDSL tile named for a windowed shape: warned once, today's lane runs."""
        ops, _ = _dense(bias=False)
        shape = S.AttnShape("float16", 64, 4, 4, "window", False, 0, False, 16, 16, 300)
        _write_table(user_db, {shape: [(None, S.AttnChoice("cute", 32, 32, 64))]})
        spy = _Spy(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="oasr.jit.attention"):
            _run(ops, window_left=64, window_right=0)
            _run(ops, window_left=64, window_right=0)
        assert spy.calls == [("cxx", None), ("cxx", None)]
        assert sum("falling through" in r.message for r in caplog.records) == 1

    def test_unknown_kind_does_not_poison_the_file(self, monkeypatch, user_db):
        ops, shape = _dense(T_k=300)
        other, other_shape = _dense(T_k=300, B=4, seed=2)
        tf = db.new_file("fmha", _sm(), "attention")
        tf.configs["mystery"] = {"kind": "flashinfer"}
        tf.configs["cxx_s2"] = S.choice_to_params(S.AttnChoice("cxx", splits=2))
        tf.entries[S.sig_key(shape)] = db.Entry(regions=[(None, "mystery")])
        tf.entries[S.sig_key(other_shape)] = db.Entry(regions=[(None, "cxx_s2")])
        db.save_file(tf, user_db / f"sm{_sm()}" / "fmha.json")
        db.reload()
        spy = _Spy(monkeypatch)
        _run(ops)
        _run(other)
        assert spy.calls == [("cute", None), ("cxx", 2)]

    def test_sdpa_region_is_a_policy_hit(self, monkeypatch, user_db):
        from oasr.layers._backend import policy_hits, reset_backend_stats

        ops, shape = _dense()
        _write_table(user_db, {shape: [(None, S.AttnChoice("sdpa"))]})
        reset_backend_stats()
        out = _run(ops)
        assert policy_hits().get("tuned-sdpa:fmha.fmha") == 1
        assert _close(out, _run(ops, backend="cute"))
        assert "sdpa*=1" in db.tuning_report(["fmha"])

    def test_paged_decode_step(self, monkeypatch, user_db):
        """A paged single-query step on a tuned C++ split and a tuned CuTeDSL tile."""
        g = torch.Generator(device="cuda").manual_seed(3)
        B, H, D, bs, T = 8, 6, 64, 16, 640
        per = T // bs
        k = torch.randn(B * per + 1, bs, H, D, device="cuda", dtype=torch.float16, generator=g)
        v = torch.randn_like(k)
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16, generator=g)
        table = (torch.randperm(B * per, device="cuda", generator=g) + 1).view(B, per).int()
        lens = torch.randint(T // 2, T + 1, (B,), device="cuda", dtype=torch.int32, generator=g)
        shape = S.AttnShape("float16", D, H, H, "none", False, bs, False, B, 1, T)
        ops = {"q": q, "k": k, "v": v, "cache_seqlens": lens, "block_table": table}
        ref = _run(ops, backend="sdpa")
        spy = _Spy(monkeypatch)
        for choice, call in (
            (S.AttnChoice("cxx", splits=8), ("cxx", 8)),
            (S.AttnChoice("cute", 32, 32, 64), ("cute", (32, 32, 64))),
        ):
            _write_table(user_db, {shape: [(None, choice)]})
            spy.calls.clear()
            assert _close(_run(ops), ref)
            assert spy.calls == [call]

    def test_paged_refuses_a_tile_taller_than_the_padding(self):
        shape = S.AttnShape("float16", 64, 6, 6, "none", True, 16, False, 8, 16, 640, True)
        why = S.unusable_reason(S.AttnChoice("cute", 128, 32, 256), shape)
        assert why is not None and "64-row padding" in why


class TestShippedFile:
    """Every shipped ``fmha.json`` loads, and every config it names decodes."""

    @pytest.mark.parametrize("path", sorted(db.system_dir().glob("sm*/fmha.json")), ids=str)
    def test_loads_and_decodes(self, path):
        sm = int(path.parent.name[2:])
        tf = db.load_file(path, "fmha", sm)
        assert tf is not None and tf.entries
        for cid, params in tf.configs.items():
            choice = S.choice_from_params(params)
            assert S.config_id(choice) == cid
        for key in tf.entries:
            assert key.startswith("fmha|op=fmha|")


@needs_lanes
class TestSplitClamp:
    def test_short_range_is_cut_and_masked_never_splits(self):
        assert fa._clamp_splits(8, T_k=4096, D=64, causal=False, local=False) == 8
        assert fa._clamp_splits(8, T_k=128, D=64, causal=False, local=False) == 1
        assert fa._clamp_splits(16, T_k=640, D=64, causal=False, local=False) == 5
        assert fa._clamp_splits(4, T_k=4096, D=64, causal=True, local=False) == 1


@needs_lanes
class TestCensus:
    def test_records_dense_and_paged_calls_once(self):
        ops, shape = _dense(T_k=300)
        with capture_attention_shapes() as rec:
            _run(ops)
            _run(ops)
            _run(ops, backend="cute")  # a named lane is never selected: not recorded
        counts = rec.take()
        assert counts == {(shape, True, -1, -1): 2}


@needs_lanes
class TestBuild:
    def test_build_writes_a_file_the_dispatcher_serves(self, monkeypatch, user_db):
        from oasr.tune import bench
        from oasr.tune.build import BuildOptions
        from oasr.tune.build_fmha import build_fmha

        pts = [
            AttnPoint(
                "float16", 64, 4, 4, "none", True, 0, False, 16, 16, t, True, calls=1, weight=1
            )
            for t in (128, 512)
        ]
        opts = BuildOptions(
            max_points_per_sig=2,
            fill=False,
            log_path=str(user_db / "log.jsonl"),
            policy=bench.BenchPolicy(stage_rounds=(2, 2, 3), solo=False),
        )
        tf = build_fmha(AttnShapeSet(pts, {}), sm=_sm(), opts=opts)
        key = pts[0].sig
        assert list(tf.entries) == [key]
        regions = tf.entries[key].regions
        assert regions[-1][0] is None
        for _, cid in regions:
            assert cid in tf.configs
            assert cid != "sdpa"  # external, not allowed: a reference only
        assert tf.provenance["candidates"]["sdpa"] == "measured as a reference only"
        log = (user_db / "log.fmha.jsonl").read_text().splitlines()
        assert any('"config": "sdpa"' in row and '"selectable": false' in row for row in log)
        db.save_file(tf, user_db / f"sm{_sm()}" / "fmha.json")
        db.reload()
        ops, shape = _dense(B=16, T_k=512)
        assert S.covering_tier(shape) == "user"
        assert _close(_run(ops), _run(ops, backend="sdpa"))
