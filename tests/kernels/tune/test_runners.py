"""Runners: the backends a tuning region selects among (``oasr.tune.runners``).

The registry, the codec, resolution and the soft validators are pure Python and
run on any box; dispatch and the build's candidate path run one small GEMM on
CUDA.
"""

import json
from pathlib import Path

import pytest
import torch

import oasr.jit.gemm as jg
from oasr.tune import database as db, runners as rn

_KIND = "test_matmul"


class _MatmulRunner(rn.Runner):
    """A stand-in external library: ``torch.matmul`` behind the runner contract."""

    name = "test_matmul"
    family = "gemm"
    kinds = (_KIND,)
    ops = ("gemm", "bmm")
    backends = ("test_matmul",)
    traits = rn.RunnerTraits(capturable=True, external=True)

    def __init__(self, why=None, refuse_k=None, version="1"):
        self.why = why
        self.refuse_k = refuse_k
        self.version = version
        self.calls = 0

    def unavailable(self):
        return self.why

    def versions(self):
        return {"lib": self.version}

    def supports(self, op, M, N, K, dtype):
        if self.refuse_k is not None and K >= self.refuse_k:
            return f"K={K} is deeper than {self.refuse_k}"
        return None

    def config_id(self, params):
        return f"tm_v{params.get('variant', 0)}"

    def bind(self, op, choice):
        def run(out, A, B, C=None, split_k=1):
            self.calls += 1
            if C is None:
                torch.matmul(A, B.t(), out=out)
            else:
                torch.addmm(C, A, B.t(), out=out)

        return run


@pytest.fixture
def user_db(tmp_path, monkeypatch):
    """Point the user tier at a temp dir for one test, and restore the epoch after."""
    import oasr.functionals.gemm as fg

    monkeypatch.setenv("OASR_TUNE_USER_DB", str(tmp_path))
    db.reload()
    yield tmp_path
    monkeypatch.setenv("OASR_TUNE_USER_DB", "off")
    db.reload()
    fg._PLANS.clear()


@pytest.fixture
def fake():
    """Register a :class:`_MatmulRunner` for one test; unregister it after."""
    holder = {}

    def make(**kw):
        runner = _MatmulRunner(**kw)
        rn.register_runner(runner, replace=True)
        holder["r"] = runner
        db.reload()  # availability and versions are cached per epoch
        return runner

    yield make
    rn.unregister_runner("gemm", "test_matmul")
    db.reload()


def _user_file(tmp_path, sm, entries, configs):
    tf = db.new_file("gemm", sm, jg.GEMM_LANE_BY_SM.get(sm, "sm80_mma"))
    tf.configs.update(configs)
    tf.entries.update({jg.gemm_sig_key(*k): db.Entry(regions=v) for k, v in entries.items()})
    db.save_file(tf, tmp_path / f"sm{sm}" / "gemm.json")
    db.reload()
    return tf


_TM1 = {"kind": _KIND, "variant": 1}


class TestRegistry:
    def test_every_kind_a_shipped_file_stores_has_an_owner(self):
        """A shipped config whose kind no runner owns would be unservable everywhere."""
        root = Path(db.system_dir())
        files = sorted(root.glob("sm*/*.json"))
        assert files, "no shipped tuning files found"
        for path in files:
            d = json.loads(path.read_text())
            for cid, params in d["configs"].items():
                owner = rn.runner_for_kind(d["family"], params["kind"])
                assert owner is not None, f"{path}: config {cid} kind {params['kind']!r}"

    def test_the_builtins_and_their_traits(self):
        cublas = rn.get_runner("gemm", "cublas")
        assert cublas is rn.runner_for_kind("gemm", "torch")
        assert cublas is rn.runner_for_backend("gemm", "torch")
        assert cublas.traits.external and cublas.traits.capturable
        assert not cublas.traits.batch_invariant, "cuBLAS picks its kernel from M"
        cutlass = rn.runner_for_kind("gemm", "cutlass")
        assert cutlass is rn.runner_for_kind("gemm", "default")
        assert cutlass.traits.compiled and not cutlass.traits.external
        assert rn.runner_for_kind("scaled_mm", "heuristic").traits.numerics == "bitwise"

    def test_cublas_binds_every_op_it_serves(self):
        cublas = rn.get_runner("gemm", "cublas")
        for op in cublas.ops:
            assert callable(cublas.bind(op, "torch")), op

    def test_a_kind_has_one_owner(self, fake):
        class Thief(_MatmulRunner):
            name = "thief"
            kinds = ("torch",)
            backends = ()

        with pytest.raises(ValueError, match="owned by 'cublas'"):
            rn.register_runner(Thief())
        assert rn.runner_for_kind("gemm", "torch").name == "cublas"

    def test_register_replace_and_unregister(self, fake):
        fake()
        with pytest.raises(ValueError, match="already registered"):
            rn.register_runner(_MatmulRunner())
        second = rn.register_runner(_MatmulRunner(version="2"), replace=True)
        assert rn.runner_for_kind("gemm", _KIND) is second
        rn.unregister_runner("gemm", "test_matmul")
        assert rn.runner_for_kind("gemm", _KIND) is None
        assert rn.runner_for_backend("gemm", "test_matmul") is None

    def test_numerics_is_validated(self):
        with pytest.raises(ValueError, match="numerics"):
            rn.RunnerTraits(numerics="close enough")

    def test_versions_join_the_soft_validators(self, fake):
        fake(version="7")
        soft = db.current_soft_validators("gemm")
        assert soft["runners"]["test_matmul"] == {"lib": "7"}
        assert "torch" in soft["runners"]["cublas"]

    def test_an_unavailable_runner_has_no_versions(self, fake):
        fake(why="library not installed")
        assert "test_matmul" not in rn.runner_versions("gemm")


class TestCodec:
    def test_a_runner_choice_round_trips(self, fake):
        fake()
        choice = jg.gemm_config_from_params(dict(_TM1), 120, config_id="tm_v1")
        assert isinstance(choice, rn.RunnerChoice)
        assert jg.gemm_config_id(choice) == "tm_v1"
        assert jg.gemm_config_to_params(choice) == _TM1
        assert not jg.is_kernel_variant(choice) and not jg.is_batch_invariant(choice)
        # Without a stored id, the runner names it.
        assert jg.gemm_config_from_params(dict(_TM1), 120).config_id == "tm_v1"
        assert hash(choice) == hash(jg.gemm_config_from_params(dict(_TM1), 120, "tm_v1"))

    def test_an_unknown_kind_decodes_instead_of_raising(self):
        choice = jg.gemm_config_from_params({"kind": "nobody", "x": 1}, 120, config_id="n1")
        assert isinstance(choice, rn.RunnerChoice) and choice.kind == "nobody"
        assert rn.runner_of("gemm", choice) is None

    def test_a_malformed_kind_still_raises(self):
        with pytest.raises(ValueError, match="unknown GEMM config kind"):
            jg.gemm_config_from_params({"tile": [1, 2, 3]}, 120)

    def test_a_runner_tactic_maps_to_its_choice(self, fake):
        from oasr.tune import gemm_tune
        from oasr.tune.autotuner import Tactic

        fake()
        choice = gemm_tune.choice_of(Tactic("test_matmul", (("variant", 2),)), 120)
        assert isinstance(choice, rn.RunnerChoice) and choice.config_id == "tm_v2"
        assert gemm_tune.choice_name(choice) == "tm_v2"

    def test_an_unowned_tactic_is_refused(self):
        from oasr.tune import gemm_tune
        from oasr.tune.autotuner import Tactic

        with pytest.raises(ValueError, match="no GEMM runner owns"):
            gemm_tune.choice_of(Tactic("nobody"), 120)


class TestResolution:
    """Where a runner choice is served, and where it falls through (pure Python, sm120)."""

    def test_a_registered_runner_is_served(self, user_db, fake):
        fake()
        _user_file(user_db, 120, {("gemm", 4242, 777): [(None, "tm_v1")]}, {"tm_v1": _TM1})
        choice = jg.select_default_config("gemm", 64, 4242, 777, torch.bfloat16, 120)
        assert isinstance(choice, rn.RunnerChoice) and choice.config_id == "tm_v1"

    def test_an_unknown_kind_does_not_poison_the_file(self, user_db, caplog):
        """One config of a runner this process lacks used to make every shape default."""
        _user_file(
            user_db,
            120,
            {("gemm", 4242, 776): [(None, "mystery")], ("gemm", 4242, 777): [(None, "torch")]},
            {"mystery": {"kind": "nobody"}, "torch": {"kind": "torch"}},
        )
        with caplog.at_level("WARNING", logger="oasr.jit.gemm"):
            lacked = jg.select_default_config("gemm", 64, 4242, 776, torch.bfloat16, 120)
        assert lacked is jg.GEMM_DEFAULT
        assert "cannot serve" in caplog.text and "nobody" in caplog.text
        assert jg.select_default_config("gemm", 64, 4242, 777, torch.bfloat16, 120) == "torch"

    def test_an_unavailable_runner_falls_through(self, user_db, fake, caplog):
        fake(why="library not installed")
        _user_file(user_db, 120, {("gemm", 4242, 777): [(None, "tm_v1")]}, {"tm_v1": _TM1})
        with caplog.at_level("WARNING", logger="oasr.jit.gemm"):
            choice = jg.select_default_config("gemm", 64, 4242, 777, torch.bfloat16, 120)
        assert choice is jg.GEMM_DEFAULT
        assert "library not installed" in caplog.text

    def test_a_refused_problem_falls_through(self, user_db, fake):
        fake(refuse_k=1024)
        _user_file(
            user_db,
            120,
            {("gemm", 4242, 2048): [(None, "tm_v1")], ("gemm", 4242, 512): [(None, "tm_v1")]},
            {"tm_v1": _TM1},
        )
        assert jg.select_default_config("gemm", 64, 4242, 2048, torch.bfloat16, 120) is (
            jg.GEMM_DEFAULT
        )
        served = jg.select_default_config("gemm", 64, 4242, 512, torch.bfloat16, 120)
        assert isinstance(served, rn.RunnerChoice)

    def test_an_op_the_runner_does_not_serve_falls_through(self, user_db, fake):
        fake()
        _user_file(
            user_db, 120, {("gemm_log_softmax", 4242, 776): [(None, "tm_v1")]}, {"tm_v1": _TM1}
        )
        choice = jg.select_default_config("gemm_log_softmax", 64, 4242, 776, torch.bfloat16, 120)
        assert choice is jg.GEMM_DEFAULT

    def test_a_runner_choice_compiles_nothing(self, user_db, fake):
        """Only CUTLASS variants enter the production compile set."""
        fake()
        _user_file(user_db, 120, {("gemm", 4242, 777): [(None, "tm_v1")]}, {"tm_v1": _TM1})
        names = jg._tuned_compile_names(120)
        assert all("tm_v1" not in n for n in names)


class TestSoftValidators:
    """A runner's version change makes stale only the entries that select it."""

    def _file(self, use_torch: bool):
        tf = db.new_file("gemm", 120, "sm80_mma")
        tf.configs["torch"] = {"kind": "torch"}
        tf.configs["default"] = {"kind": "default"}
        tf.entries[jg.gemm_sig_key("gemm", 256, 2048)] = db.Entry(
            regions=[(None, "torch" if use_torch else "default")]
        )
        soft = tf.validators["soft"]
        assert "cublas" in soft.get("runners", {}), "this process stamps cuBLAS's version"
        soft["runners"] = {**soft["runners"], "cublas": {"torch": "0.0-elsewhere"}}
        return tf

    def test_an_entry_on_the_changed_runner_is_stale(self):
        stale = self._file(use_torch=True).stale
        assert any("runner cublas" in r for r in stale), stale

    def test_an_entry_on_another_runner_is_not(self):
        stale = self._file(use_torch=False).stale
        assert not any("runner cublas" in r for r in stale), stale

    def test_entries_carrying_their_own_stamp_are_counted_by_theirs(self):
        tf = self._file(use_torch=False)
        old = json.loads(json.dumps(tf.validators["soft"]))
        tf.validators["soft"] = db.current_soft_validators("gemm")
        entry = db.Entry(regions=[(None, "torch")], soft=old)
        tf.entries[jg.gemm_sig_key("gemm", 512, 2048)] = entry
        tf._stale = None
        assert any(r.startswith("1 entry measured under runner cublas") for r in tf.stale)

    def test_a_file_without_runner_versions_is_not_stale_for_it(self):
        tf = self._file(use_torch=True)
        del tf.validators["soft"]["runners"]
        assert not any("runner" in r for r in tf.stale)


@pytest.mark.cuda
class TestDispatch:
    """A runner choice reaches the bound callable; accounting names what ran."""

    @pytest.fixture(autouse=True)
    def _cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")

    def _operands(self, M=64, N=4240, K=776):
        g = torch.Generator(device="cuda").manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) / K**0.5
        C = torch.randn(N, device="cuda", dtype=torch.bfloat16, generator=g)
        return A, B, C

    def test_a_runner_choice_is_dispatched_and_counted(self, user_db, fake):
        import oasr
        from oasr.jit.core import _get_target_sm
        from oasr.layers._backend import policy_hits, reset_backend_stats

        runner = fake()
        A, B, C = self._operands()
        sm = _get_target_sm()
        _user_file(user_db, sm, {("gemm", 4240, 776): [(None, "tm_v1")]}, {"tm_v1": _TM1})
        db.reset_tier_counts()
        reset_backend_stats()
        out = oasr.gemm(A, B, C)
        torch.cuda.synchronize()
        assert runner.calls == 1
        ref = torch.addmm(C.float(), A.float(), B.float().t())
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
        assert db.runner_counts()[("gemm", "gemm", "test_matmul")] == 1
        assert policy_hits().get("tuned-test_matmul:gemm.gemm") == 1
        assert "test_matmul*=1" in db.tuning_report()
        oasr.gemm(A, B, C)  # the plan is memoised: bound once, counted once
        assert runner.calls == 2
        assert db.runner_counts()[("gemm", "gemm", "test_matmul")] == 1

    def test_a_pinned_call_does_not_claim_the_runner_it_declined(self, user_db):
        """The batch-invariant selector reads the tuned choice and may drop it."""
        import oasr
        from oasr.jit.core import _get_target_sm
        from oasr.layers._backend import policy_hits, reset_backend_stats

        sm = _get_target_sm()
        try:
            jg.batch_invariant_configs(sm)
        except ValueError:
            pytest.skip(f"no batch-invariant class on sm{sm}")
        if not jg.batch_invariant_configs(sm):
            pytest.skip(f"no batch-invariant class on sm{sm}")
        A, B, C = self._operands()
        _user_file(
            user_db, sm, {("gemm", 4240, 776): [(None, "torch")]}, {"torch": {"kind": "torch"}}
        )
        db.reset_tier_counts()
        reset_backend_stats()
        oasr.gemm(A, B, C, batch_invariant=True)
        assert ("gemm", "gemm", "cublas") not in db.runner_counts()
        assert not any(k.startswith("tuned-cublas") for k in policy_hits())
        oasr.gemm(A, B, C)
        assert db.runner_counts()[("gemm", "gemm", "cublas")] == 1
        assert policy_hits().get("tuned-cublas:gemm.gemm") == 1


@pytest.mark.cuda
class TestBuildCandidates:
    """A runner's backend entries are measured by the build under the one protocol."""

    def test_a_runner_is_measured_and_named_by_its_config_id(self, fake):
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        from oasr.tune import bench, gemm_tune
        from oasr.tune.autotuner import (
            BackendEntry,
            OpKey,
            Tactic,
            _ensure_backends_registered,
            _global_registry,
        )

        runner = fake()
        bound = runner.bind("gemm", None)
        entry = BackendEntry(
            tactic=Tactic("test_matmul", (("variant", 1),)),
            is_available=lambda: True,
            get_runner=lambda: bound,
        )
        runner.backend_entries = lambda: [(OpKey("gemm", "gemm"), entry)]
        try:
            _ensure_backends_registered()
            assert entry in gemm_tune.candidates("gemm")
            policy = bench.BenchPolicy(stage_rounds=(2,), solo=False, separator=False)
            res = gemm_tune.benchmark_case(
                gemm_tune.GemmCase("gemm", 64, 256, 128),
                policy=policy,
                only=["tm_v1", "torch"],
                issue_cost=False,
            )
            assert res.results["tm_v1"].status == "ok", res.rejected
            assert isinstance(res.results["tm_v1"].payload, rn.RunnerChoice)
            assert jg.gemm_config_to_params(res.results["tm_v1"].payload) == _TM1
        finally:
            entries = _global_registry._entries.get(OpKey("gemm", "gemm"), [])
            if entry in entries:
                entries.remove(entry)
