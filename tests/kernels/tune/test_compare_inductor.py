"""``oasr tune compare-inductor``: torch.compile of each repeated block vs OASR.

``oasr.tune.inductor`` is a benchmark, never a runner.  Discovery, argument
capture, the verdict and the report are pure Python on a tiny CPU Conformer;
the measurement compiles and times a tiny block on CUDA (the max-autotune arm,
which also checks Inductor's autotune record, is ``slow``).
"""

import dataclasses
import json
import math
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from torch import nn

from oasr.tune import bench, cli, inductor as ind
from oasr.tune.inductor import ArmResult, BlockResult

_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not _CUDA, reason="needs a CUDA GPU")


def _tiny_conformer(blocks: int = 2):
    from oasr.models.conformer.config import ConformerEncoderConfig, ConformerModelConfig
    from oasr.models.conformer.model import ConformerModel

    cfg = ConformerModelConfig(
        vocab_size=32,
        encoder=ConformerEncoderConfig(
            input_size=80,
            output_size=64,
            attention_heads=4,
            linear_units=128,
            num_blocks=blocks,
            cnn_module_norm="layer_norm",
        ),
    )
    torch.manual_seed(0)
    return ConformerModel.from_config(cfg).eval().encoder


class _Spec:
    """A FeatureSpec stand-in: 80 bins at 10 ms, optional LFR and fixed window."""

    def __init__(self, lfr_m=1, lfr_n=1, window=None):
        self.feature_dim = 80
        self.frame_shift_ms = 10.0
        self.lfr_m, self.lfr_n = lfr_m, lfr_n
        self._window = window

    def to_feature_config(self):
        return type("Cfg", (), {"fixed_window_frames": self._window})()


# =============================================================================
# Discovery and capture (CPU)
# =============================================================================


class TestDiscovery:
    def test_conformer_blocks_once_each_in_module_order(self):
        enc = _tiny_conformer()
        found = [(p, type(m).__name__) for p, m in ind.discover_blocks(enc)]
        assert found == [
            ("encoders.0", "ConformerEncoderLayer"),
            ("encoders.0.feed_forward", "PositionwiseFeedForward"),
            ("encoders.0.self_attn", "RelPositionMultiHeadedAttention"),
            ("encoders.0.conv_module", "ConvolutionModule"),
        ]

    def test_select_by_class_or_path(self):
        enc = _tiny_conformer()
        by_class = [p for p, _ in ind.discover_blocks(enc, ["ConvolutionModule"])]
        by_path = [p for p, _ in ind.discover_blocks(enc, ["self_attn"])]
        assert by_class == ["encoders.0.conv_module"]
        assert by_path == ["encoders.0.self_attn"]

    def test_one_block_per_weight_shape(self):
        class Block(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.a = nn.Linear(d, d)
                self.b = nn.Linear(d, d)

            def forward(self, x):
                return self.b(self.a(x))

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.stack = nn.ModuleList([Block(8), Block(8), Block(16), Block(16)])

        found = [p for p, _ in ind.discover_blocks(Enc())]
        # Each width once; nn.Linear repeats too but is a leaf (one kernel).
        assert found == ["stack.0", "stack.2"]

    def test_a_stack_of_layers_is_not_a_block(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8)
                self.b = nn.Linear(8, 8)

        class Stack(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Layer(), Layer()])

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.stacks = nn.ModuleList([Stack(), Stack()])

        # The stacks repeat too, but timing one only times its layers again.
        assert [p for p, _ in ind.discover_blocks(Enc())] == ["stacks.0.layers.0"]

    def test_shape_ladder_keeps_one_fixed_window(self):
        assert ind.shape_ladder(_Spec(), [1, 8], [4.0, 16.0]) == [
            (1, 4.0, 400),
            (1, 16.0, 1600),
            (8, 4.0, 400),
            (8, 16.0, 1600),
        ]
        assert ind.shape_ladder(_Spec(window=3000), [1, 8], [4.0, 16.0]) == [
            (1, 4.0, 3000),
            (8, 4.0, 3000),
        ]

    def test_encoder_inputs_follow_the_frontend(self):
        xs, lens, frames = ind.encoder_inputs(_Spec(), 2, 4.0)
        assert frames == 400 and xs.shape == (2, 400, 80) and lens.tolist() == [400, 400]
        xs, _, frames = ind.encoder_inputs(_Spec(lfr_m=7, lfr_n=6), 1, 6.0)
        assert frames == 100 and xs.shape == (1, 100, 560)
        # A fixed-window frontend always feeds its window, whatever the length.
        _, _, frames = ind.encoder_inputs(_Spec(window=3000), 1, 4.0)
        assert frames == 3000


class TestCapture:
    def test_records_every_block_and_stops_early(self):
        enc = _tiny_conformer()
        blocks = ind.discover_blocks(enc)
        ran = []
        enc.encoders[1].register_forward_hook(lambda *a: ran.append(1))
        xs, lens, _ = ind.encoder_inputs(_Spec(), 2, 1.0)
        calls = ind.capture_calls(enc, blocks, xs, lens)
        assert set(calls) == {p for p, _ in blocks}
        assert not ran, "the capture forward ran past the last block it needed"
        args, _ = calls["encoders.0.feed_forward"]
        assert args[0].shape[0] == 2 and args[0].shape[-1] == 64

    def test_structured_arguments_keep_their_type(self):
        # Nemotron passes its attention blocks as a NamedTuple read by attribute.
        class Blocks(NamedTuple):
            num: int
            mask: torch.Tensor

        @dataclasses.dataclass
        class State:
            x: torch.Tensor
            k: int = 3

        v = (Blocks(2, torch.ones(2)), State(torch.zeros(1)), [torch.ones(1)], {"a": 1})
        c = ind._clone(v)
        assert isinstance(c[0], Blocks) and c[0].num == 2 and c[0].mask is not v[0].mask
        assert isinstance(c[1], State) and c[1].k == 3
        h = ind._to_device(v, torch.device("cpu"), torch.float16)
        assert isinstance(h[0], Blocks) and h[0].mask.dtype == torch.float16
        assert h[1].x.dtype == torch.float16 and h[3] == {"a": 1}

    def test_recorded_arguments_replay_the_block(self):
        enc = _tiny_conformer()
        blocks = ind.discover_blocks(enc)
        xs, lens, _ = ind.encoder_inputs(_Spec(), 2, 1.0)
        calls = ind.capture_calls(enc, blocks, xs, lens)
        layer = dict(blocks)["encoders.0"]
        args, kwargs = calls["encoders.0"]
        with torch.no_grad():
            out = layer(*args, **kwargs)
        assert out[0].shape == args[0].shape and torch.isfinite(out[0]).all()


# =============================================================================
# Verdict and report (no device)
# =============================================================================


def _result(oasr_us, ind_us, *, sigma=0.1, ind_ok=True, block="enc.0.ffn", batch=8):
    arms = {
        "oasr": ArmResult(
            "oasr",
            us=oasr_us,
            sigma_us=sigma,
            err=0.01,
            numerics_ok=True,
            kernels=[
                {"name": "k1", "label": "oasr::gemm", "op": "", "calls": 2, "us": 9.0},
                {
                    "name": "k2",
                    "label": "at::ew",
                    "op": "aten::masked_fill_",
                    "calls": 2,
                    "us": 3.0,
                },
            ],
            launches=4,
        ),
        "torch": ArmResult("torch", us=oasr_us * 1.5, sigma_us=sigma, err=0.01, numerics_ok=True),
        "inductor-max": ArmResult(
            "inductor-max",
            us=ind_us,
            sigma_us=sigma,
            err=0.01 if ind_ok else 1.0,
            numerics_ok=ind_ok,
            kernels=[
                {"name": "triton_per_fused_add_native_layer_norm_0", "calls": 1, "us": 3.0},
                {"name": "nvjet_hsh_64x32_64x16_1x1_v_bz_TNN", "calls": 2, "us": 9.0},
            ],
            launches=3,
            graphs=1,
            breaks=["Unsupported method call -- cython_function_or_method"],
            autotune=[
                {
                    "op": "addmm",
                    "inputs": [[64], [8, 64], [64, 64]],
                    "offered": {"ATEN": 1, "TRITON": 12},
                    "best_ms": {"ATEN": 0.01, "TRITON": 0.008},
                    "winner": "triton_mm_3",
                    "winner_backend": "TRITON",
                    "winner_ms": 0.008,
                }
            ],
        ),
    }
    return BlockResult(
        block=block,
        kind="PositionwiseFeedForward",
        batch=batch,
        seconds=4.0,
        frames=100,
        inputs=[[batch, 100, 64]],
        copies=1,
        arms=arms,
    )


class TestVerdict:
    def test_ahead_even_behind(self):
        assert _result(30.0, 20.0).verdict() == "inductor-ahead"
        assert _result(30.0, 29.0).verdict() == "even"
        assert _result(20.0, 30.0).verdict() == "oasr-ahead"
        assert _result(30.0, 20.0).speedup() == pytest.approx(1.5)

    def test_noise_is_not_a_gap(self):
        # 10 % apart, but the two sigmas make it indistinguishable.
        assert _result(22.0, 20.0, sigma=1.0).verdict() == "even"
        assert _result(22.0, 20.0, sigma=0.1).verdict() == "inductor-ahead"

    def test_a_numerics_failure_is_never_a_win(self):
        r = _result(30.0, 10.0, ind_ok=False)
        assert r.best_compiled() is None and r.speedup() is None
        assert r.verdict() == "numerics"

    def test_best_compiled_takes_the_faster_passing_arm(self):
        r = _result(30.0, 25.0)
        r.arms["inductor"] = ArmResult("inductor", us=20.0, sigma_us=0.1, numerics_ok=True)
        assert r.best_compiled() == "inductor"
        r.arms["inductor"].numerics_ok = False
        assert r.best_compiled() == "inductor-max"

    def test_oasr_error_is_an_error(self):
        r = _result(30.0, 20.0)
        r.arms["oasr"].status, r.arms["oasr"].us = "error", math.inf
        assert r.verdict() == "error"


class TestReport:
    @pytest.mark.parametrize(
        "kernel, ops",
        [
            ("triton_per_fused_add_native_layer_norm_0", ["add", "native_layer_norm"]),
            ("triton_tem_fused_addmm_silu_1", ["addmm", "silu"]),
            ("triton_poi_fused_glu_masked_fill_12", ["glu", "masked_fill"]),
            ("triton_red_fused__to_copy_mean_pow_3", ["_to_copy", "mean", "pow"]),
            (
                "triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_"
                "bmm_constant_pad_nd_logical_not_masked_fill_mul_4",
                [
                    "_scaled_dot_product_efficient_attention",
                    "_unsafe_view",
                    "add",
                    "bmm",
                    "constant_pad_nd",
                    "logical_not",
                    "masked_fill",
                    "mul",
                ],
            ),
            ("nvjet_hsh_64x32_64x16_1x1_v_bz_TNN", []),
            ("void oasr::layer_norm_kernel<half>(...)", []),
        ],
    )
    def test_fused_ops(self, kernel, ops):
        assert ind.fused_ops(kernel) == ops

    def test_a_name_cxxfilt_cannot_read_still_reads(self):
        # CUPTI truncates long CUTLASS 2.x names; the identifiers are still there.
        name = (
            "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_"
            "9GemmShapeILi32ELi64ELi64EEENS_9transform11threadblock28PredicatedTileAcc"
        )
        hint = ind._mangled_hint(name)
        assert hint.startswith("~cutlass::Kernel::gemm::kernel::Gemm::threadblock::MmaMultistage")
        assert hint.endswith("GemmShape<32, 64, 64>")
        assert (
            ind._mangled_hint(
                "_ZN4oasr4norm18layerNormActKernelI6__halfLi8ENS_15SwishActivationEEEvPKT_"
            )
            == "~oasr::norm::layerNormActKernel::SwishActivation"
        )

    def test_markdown_lists_the_gap_with_its_kernels(self):
        meta = {"model": "tiny", "backends": ["ATEN", "TRITON"], "sm": 120}
        md = ind.render_markdown([_result(30.0, 20.0), _result(20.0, 30.0, batch=1)], meta)
        todo = md.split("## Kernel to-do")[1].split("## Inductor autotune")[0]
        assert "`enc.0.ffn` (PositionwiseFeedForward): 1.50x at B8" in todo
        assert "fuses add + native_layer_norm" in todo
        assert "graph breaks" in todo
        assert (
            "torch kernels still on the OASR path: 2 launches, 3.0 us (aten::masked_fill_)" in todo
        )
        assert "`at::ew` (aten::masked_fill_) x2" in todo
        auto = md.split("## Inductor autotune")[1]
        assert "ATEN 1, TRITON 12" in auto and "(TRITON)" in auto

    def test_markdown_says_none_when_oasr_is_ahead(self):
        md = ind.render_markdown([_result(20.0, 30.0)], {"model": "tiny"})
        assert "None: Inductor is not 5% ahead" in md

    def test_json_round_trip(self, tmp_path):
        rs = [_result(30.0, 20.0), _result(20.0, 30.0, batch=1)]
        rs[1].arms["torch"].us = math.inf
        jpath, mpath = ind.save_report(rs, {"model": "tiny"}, tmp_path, "r")
        meta, back = ind.load_report(jpath)
        assert meta == {"model": "tiny"} and mpath.read_text().startswith("# Inductor")
        assert [r.verdict() for r in back] == [r.verdict() for r in rs]
        assert math.isinf(back[1].arms["torch"].us)
        assert back[0].arms["inductor-max"].autotune == rs[0].arms["inductor-max"].autotune


class TestCli:
    def test_parser(self):
        args = cli.build_parser().parse_args(
            ["compare-inductor", "--ckpt-dir", "x", "--blocks", "ConvolutionModule", "--quick"]
        )
        assert args.func is cli._cmd_compare_inductor
        assert args.batches == "1,8,32" and args.seconds == "4,16"
        assert args.arms == "oasr,torch,inductor,inductor-max"
        assert args.backends == "ATEN,TRITON,CUTLASS,CUTEDSL"
        assert args.blocks == ["ConvolutionModule"] and args.quick

    def test_isolation_flags(self):
        args = cli.build_parser().parse_args(["compare-inductor"])
        assert args.autotune_subproc == "auto" and not args.in_process
        assert args.child_out is None and args.paths is None

    def test_autotune_subprocess_by_free_memory(self, monkeypatch):
        assert ind.autotune_subproc("on") is True and ind.autotune_subproc("off") is False
        monkeypatch.setattr(ind, "_free_device_bytes", lambda: 2 << 30)
        assert ind.autotune_subproc("auto") is True  # a shared GPU: keep candidates out
        monkeypatch.setattr(ind, "_free_device_bytes", lambda: 24 << 30)
        assert ind.autotune_subproc("auto") is False
        with pytest.raises(ValueError):
            ind.autotune_subproc("sometimes")

    def test_a_dead_child_costs_one_block(self, tmp_path, monkeypatch):
        """A child that dies (a sticky CUDA error) loses the block it was on; a
        fresh child measures the rest."""
        import subprocess
        import types

        enc = _tiny_conformer()
        paths = [p for p, _ in ind.discover_blocks(enc)]
        calls = []

        def fake_run(cmd, stderr=None, **_):
            out = Path(cmd[cmd.index("--child-out") + 1])
            todo = cmd[cmd.index("--paths") + 1].split(",")
            calls.append(todo)
            with open(out, "a") as f:
                for path in todo:
                    f.write(json.dumps({"start": path}) + "\n")
                    if len(calls) == 1 and path == paths[1]:
                        stderr.write("torch.AcceleratorError: CUDA error: misaligned address\n")
                        return types.SimpleNamespace(returncode=1)
                    r = _result(30.0, 20.0, block=path)
                    f.write(json.dumps({"result": r.to_json()}) + "\n")
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        args = cli.build_parser().parse_args(
            ["compare-inductor", "--batches", "2", "--seconds", "1"]
        )
        log = tmp_path / "children.log"
        res = cli._compare_isolated(
            args, "ckpt", ind, enc, _Spec(), [2], [1.0], ["oasr", "inductor-max"], False,
            tmp_path, log,
        )  # fmt: skip
        assert calls == [paths, paths[2:]]
        assert [r.block for r in res] == [paths[0], paths[1], paths[2], paths[3]]
        dead = res[1]
        assert dead.copies == 0 and dead.verdict() == "error"
        assert "misaligned address" in dead.arms["inductor-max"].error
        assert all(r.verdict() == "inductor-ahead" for r in res if r is not dead)

    def test_needs_a_checkpoint(self, monkeypatch):
        monkeypatch.delenv("CKPT_DIR", raising=False)
        args = cli.build_parser().parse_args(["compare-inductor"])
        with pytest.raises(SystemExit, match="--ckpt-dir is required"):
            args.func(args)

    def test_cutlass_import_leaves_no_file_behind(self, tmp_path, monkeypatch):
        # cutlass_cppgen opens compiled_cache.db in the working directory when
        # imported, even when the import then fails: the probe runs elsewhere.
        cdir = ind._cutlass_dir()
        if cdir is None:
            pytest.skip("no CUTLASS checkout with python/cutlass_library")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ind, "_CUTLASS_IMPORT", None)
        status = ind._preload_cutlass(cdir)
        assert isinstance(status, str)
        assert list(tmp_path.iterdir()) == []
        assert ind._preload_cutlass(cdir) == status  # once per process

    def test_backend_status(self):
        st = ind.available_backends(["aten", "TRITON", "CUTEDSL", "NVGEMM", "XYZ"])
        assert st["ATEN"] == st["TRITON"] == "ok"
        assert set(st) == {"ATEN", "TRITON", "CUTEDSL", "NVGEMM", "XYZ"}
        assert "unknown" in st["XYZ"]


# =============================================================================
# Measurement (CUDA)
# =============================================================================

_FAST = bench.BenchPolicy(solo=False, stage_rounds=(1, 1, 2), max_calls=8, separator=False)


def _captured(block_path):
    enc = _tiny_conformer()
    blocks = ind.discover_blocks(enc)
    xs, lens, frames = ind.encoder_inputs(_Spec(), 2, 1.0)
    args, kwargs = ind.capture_calls(enc, blocks, xs, lens)[block_path]
    return dict(blocks)[block_path], args, kwargs, frames


@needs_cuda
class TestMeasure:
    def test_every_arm_times_and_agrees(self):
        mod, args, kwargs, frames = _captured("encoders.0.conv_module")
        r = ind.measure_block(
            "encoders.0.conv_module",
            mod,
            args,
            kwargs,
            batch=2,
            seconds=1.0,
            frames=frames,
            arms=("oasr", "torch", "inductor"),
            policy=_FAST,
            max_copy_bytes=1 << 20,
        )
        assert set(r.arms) == {"oasr", "torch", "inductor"}
        for name, a in r.arms.items():
            assert a.status == "ok", (name, a.error)
            assert math.isfinite(a.us) and a.us > 0, name
            assert a.numerics_ok, (name, a.err)
            assert a.launches > 0 and a.kernels, name
        comp = r.arms["inductor"]
        assert comp.graphs and comp.graphs >= 1 and comp.compile_s is not None
        assert r.arms["oasr"].compile_s is None and r.copies >= 1
        # Inductor's kernels are its own Triton kernels; OASR's are not.
        assert any(k["name"].startswith("triton_") for k in comp.kernels)
        assert not any(k["name"].startswith("triton_") for k in r.arms["oasr"].kernels)
        # The torch path's kernels are named by the aten op that launched them;
        # an OASR launcher is no aten op.
        assert any(k["op"].startswith("aten::") for k in r.arms["torch"].kernels)
        assert any(k["op"] == "" and "oasr" in k["label"] for k in r.arms["oasr"].kernels)
        for a in r.arms.values():
            assert sum(k["calls"] for k in a.kernels) == a.launches
        assert r.verdict() in ("inductor-ahead", "even", "oasr-ahead")

    def test_rotated_copies_share_one_compiled_graph(self):
        mod, args, kwargs, frames = _captured("encoders.0.feed_forward")
        r = ind.measure_block(
            "encoders.0.feed_forward",
            mod,
            args,
            kwargs,
            batch=2,
            seconds=1.0,
            frames=frames,
            arms=("oasr", "inductor"),
            policy=_FAST,
            working_set_bytes=1 << 34,  # a model far past L2: rotate
            max_copy_bytes=4 << 20,
        )
        assert r.copies > 1
        assert r.arms["inductor"].status == "ok" and r.arms["inductor"].graphs == 1

    def test_unknown_arm_is_refused(self):
        mod, args, kwargs, frames = _captured("encoders.0.feed_forward")
        with pytest.raises(ValueError, match="unknown arms"):
            ind.measure_block(
                "x", mod, args, kwargs, batch=2, seconds=1.0, frames=frames, arms=("tvm",)
            )

    @pytest.mark.slow
    def test_max_autotune_records_what_each_backend_offered(self):
        mod, args, kwargs, frames = _captured("encoders.0.feed_forward")
        r = ind.measure_block(
            "encoders.0.feed_forward",
            mod,
            args,
            kwargs,
            batch=2,
            seconds=1.0,
            frames=frames,
            arms=("oasr", "inductor-max"),
            backends=("ATEN", "TRITON"),
            policy=_FAST,
            max_copy_bytes=1 << 20,
        )
        a = r.arms["inductor-max"]
        assert a.status == "ok", a.error
        assert a.autotune, "no autotuned GEMM recorded"
        for t in a.autotune:
            assert set(t["offered"]) <= {"ATEN", "TRITON"}
            assert t["winner_backend"] in t["offered"]
