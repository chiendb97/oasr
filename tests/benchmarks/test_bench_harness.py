# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The unified benchmark harness (``benchmarks/run.py``): family contract, sweep
driver, workload schema and the engine footprint columns -- CPU-only.

Nothing here times a kernel.  The subject is the bookkeeping that decides what a
row *says*: which arms are timed, what they are compared against, under which
tolerance, and which configuration a row claims to describe.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import shlex
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from benchmarks.core import driver, registry  # noqa: E402
from benchmarks.core.driver import Work  # noqa: E402
from benchmarks.core.footprint import (  # noqa: E402
    model_weight_mib,
    parse_quantization,
    quantization_label,
    served_quantization,
)
from benchmarks.core.metrics import REFCHECK_FAIL, REFCHECK_PASS  # noqa: E402
from benchmarks.core.schema import (  # noqa: E402
    ENVELOPE_COLUMNS,
    WORKLOAD_COLUMNS,
    WorkloadRow,
    format_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Kernel families with no ``oasr/functionals/<name>.py``: ``composite`` times a
#: chain of several functionals (the Conformer conv module) by design.
_NOT_ONE_FUNCTIONAL = {"composite"}

_REQUIRED = ("parse_args", "resolve_configs", "build_fns", "describe")


def _family(name):
    return importlib.import_module(registry.KERNEL_FAMILIES[name])


def _main_option_strings():
    from benchmarks.run import build_parser

    return [s for a in build_parser()._actions for s in a.option_strings]


# ---------------------------------------------------------------------------
# Family contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(registry.KERNEL_FAMILIES))
def test_kernel_family_honours_the_driver_contract(name):
    module = _family(name)
    assert module.SUBROUTINES and len(set(module.SUBROUTINES)) == len(module.SUBROUTINES)
    for sub in module.SUBROUTINES:
        assert module.DEFAULT_CONFIGS.get(sub), f"{name}/{sub} has no default configs"
    for attr in _REQUIRED:
        assert callable(getattr(module, attr, None)), f"{name} lacks {attr}()"

    subs = set(module.SUBROUTINES)
    for key in getattr(module, "TOLERANCES", {}):
        sub, dtype = key if isinstance(key, tuple) else (key, None)
        assert sub in subs, f"{name}.TOLERANCES names unknown subroutine {sub!r}"
        assert dtype is None or isinstance(dtype, torch.dtype)
    assert set(getattr(module, "COMPARE_ALL_OUTPUTS", ())) <= subs
    for attr in ("REF_BACKEND", "ORACLE_BACKEND"):
        assert isinstance(getattr(module, attr, ""), str)


@pytest.mark.parametrize("name", sorted(registry.KERNEL_FAMILIES))
def test_family_flags_reach_the_family(name):
    """The top-level parser runs ``parse_known_args`` with abbreviations on, so a
    family flag that prefixes one of its options is swallowed before the family
    sees it -- and the family then sweeps its defaults under the wrong name."""
    parser = argparse.ArgumentParser(add_help=False)
    _family(name).parse_args(parser)
    main = _main_option_strings()
    for action in parser._actions:
        for opt in action.option_strings:
            clash = [m for m in main if m.startswith(opt)]
            assert not clash, f"{name} flag {opt} is consumed by run.py's {clash}"


@pytest.mark.parametrize("name", sorted(set(registry.KERNEL_FAMILIES) - _NOT_ONE_FUNCTIONAL))
def test_kernel_family_maps_onto_one_functional_module(name):
    assert (REPO_ROOT / "oasr" / "functionals" / f"{name}.py").is_file()


def test_quantization_families_are_registered():
    assert registry.KERNEL_FAMILIES["quant"] == "benchmarks.kernels.quant"
    assert registry.KERNEL_FAMILIES["scaled_mm"] == "benchmarks.kernels.scaled_mm"
    assert registry.category_of("scaled_mm") == "kernel"


# ---------------------------------------------------------------------------
# quant
# ---------------------------------------------------------------------------


def _ns(**kw):
    return argparse.Namespace(**kw)


def test_quant_configs_and_work_model():
    quant = _family("quant")
    assert quant.resolve_configs(_ns(M=64, K=256, smooth=True), "quantize_per_token") == [
        {"M": 64, "K": 256, "smooth": True}
    ]
    defaults = quant.resolve_configs(_ns(M=None, K=None, smooth=False), "quantize_per_token")
    assert defaults == quant.DEFAULT_CONFIGS["quantize_per_token"]
    smoothed = quant.resolve_configs(_ns(M=None, K=None, smooth=True), "quantize_per_token")
    assert smoothed and all(c["smooth"] for c in smoothed)
    with pytest.raises(SystemExit):
        quant.resolve_configs(_ns(M=64, K=None, smooth=False), "quantize_per_token")

    M, K = 2048, 256
    work = quant.describe("quantize_per_token", {"M": M, "K": K, "smooth": False}, torch.float16)
    assert work.bytes == M * K * 2 + M * K + 4 * M and work.flops is None
    smooth = quant.describe("quantize_per_token", {"M": M, "K": K, "smooth": True}, torch.float16)
    assert smooth.bytes == work.bytes + 4 * K
    # The quantizer and its twin are bit-exact: codes and scales, no tolerance.
    assert quant.TOLERANCES["quantize_per_token"] == (0.0, 0.0)
    assert "quantize_per_token" in quant.COMPARE_ALL_OUTPUTS


# ---------------------------------------------------------------------------
# scaled_mm
# ---------------------------------------------------------------------------


def test_scaled_mm_configs_and_work_model():
    smm = _family("scaled_mm")
    assert smm.SUBROUTINES == ["scaled_mm", "quant_scaled_mm", "ffn_chain"]
    explicit = smm.resolve_configs(
        _ns(M=16, N=768, K=256, activation="swish", bias=True), "quant_scaled_mm"
    )
    assert explicit == [{"M": 16, "N": 768, "K": 256, "activation": "swish", "bias": True}]
    plain = smm.resolve_configs(_ns(M=16, N=768, K=256, activation=None, bias=False), "scaled_mm")
    assert plain[0]["activation"] == "none" and plain[0]["bias"] is False
    with pytest.raises(SystemExit):
        smm.resolve_configs(_ns(M=16, N=None, K=256, activation=None, bias=False), "scaled_mm")
    for sub in smm.SUBROUTINES:
        for cfg in smm.resolve_configs(_ns(M=None, N=None, K=None, activation=None), sub):
            # The kernel's alignment contract, so a default never fails setup.
            assert cfg["K"] % 16 == 0 and cfg["N"] % 8 == 0, cfg

    work = smm.describe("scaled_mm", explicit[0], torch.float16)
    assert work.flops == 2.0 * 16 * 768 * 256
    assert "swish" in work.shape and "+bias" in work.shape
    # The FFN chain is two GEMMs: [M, d] x [ffn, d]^T, then [M, ffn] x [d, ffn]^T.
    chain = smm.resolve_configs(_ns(M=None, N=None, K=None, activation=None), "ffn_chain")[0]
    work = smm.describe("ffn_chain", chain, torch.float16)
    assert work.flops == 2 * 2.0 * chain["M"] * chain["N"] * chain["K"]


def test_scaled_mm_compares_speed_against_fp16_and_correctness_against_the_twin():
    smm = _family("scaled_mm")
    # Speed: the half-precision GEMM a quantized layer replaces.
    assert smm.REF_BACKEND == "fp16" and "fp16" in smm.NON_GATING_BACKENDS
    # Correctness: the exact twin, never timed by default.
    assert smm.ORACLE_BACKEND == "reference"
    assert driver.default_backends(smm, dict.fromkeys(["cutlass", "torch", "fp16", "reference"]))
    assert "reference" not in driver.default_backends(
        smm, dict.fromkeys(["cutlass", "torch", "fp16", "reference"])
    )
    # One GEMM: one output ulp, per dtype -- a bf16 ulp is eight fp16 ones.
    for sub in ("scaled_mm", "quant_scaled_mm"):
        fp16 = driver.tolerance_for(smm, sub, torch.float16)
        bf16 = driver.tolerance_for(smm, sub, torch.bfloat16)
        assert fp16[1] >= 2.0**-10 and bf16[1] >= 2.0**-7 and fp16 < bf16
    # The chain: one int8 step of the intermediate at a tie, whatever the dtype.
    for dtype in (torch.float16, torch.bfloat16):
        assert driver.tolerance_for(smm, "ffn_chain", dtype)[1] >= 4e-3


def test_conformer_scaled_mm_testlist_parses_and_covers_the_grid():
    from benchmarks.run import build_parser

    path = REPO_ROOT / "benchmarks" / "testlists" / "conformer_scaled_mm.txt"
    lines = [
        shlex.split(raw)
        for raw in path.read_text().splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]
    main = build_parser()
    smm = _family("scaled_mm")
    seen = set()
    for tokens in lines:
        args, extra = main.parse_known_args(tokens)
        assert registry.resolve(args.family) == "scaled_mm"
        assert args.subroutine in smm.SUBROUTINES
        family_parser = argparse.ArgumentParser(add_help=False)
        smm.parse_args(family_parser)
        fam = family_parser.parse_args(extra)  # strict, as run.py parses it
        (cfg,) = smm.resolve_configs(fam, args.subroutine)
        seen.add((args.subroutine, cfg["M"], cfg["N"], cfg["K"]))
    want = {
        (sub, m, n, k)
        for sub in ("scaled_mm", "quant_scaled_mm")
        for n, k in ((768, 256), (256, 256), (2048, 256), (256, 2048), (512, 256))
        for m in (16, 128, 2048, 16384)
    }
    assert seen == want and len(lines) == len(want)


# ---------------------------------------------------------------------------
# Sweep driver: oracle, all-outputs refcheck, dtype-keyed tolerance
# ---------------------------------------------------------------------------


class _Collect:
    def __init__(self):
        self.rows, self.errors, self.warnings = [], [], []

    def kernel_row(self, row):
        self.rows.append(row)

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def test_refcheck_of_all_outputs_sees_what_the_first_output_hides():
    codes = torch.tensor([1, 2, 3], dtype=torch.int8)
    good = torch.tensor([0.5, 0.25])
    fns = {
        "torch": lambda: (codes.clone(), good.clone()),
        "cuda": lambda: (codes.clone(), good.flip(0)),  # right codes, rows' scales swapped
    }
    first = driver._refcheck(fns, ["cuda", "torch"], "torch", _Collect(), "s", (0.0, 0.0))
    assert first["cuda"][0] == REFCHECK_PASS
    rep = _Collect()
    every = driver._refcheck(
        fns, ["cuda", "torch"], "torch", rep, "s", (0.0, 0.0), all_outputs=True
    )
    assert every["cuda"] == (REFCHECK_FAIL, 0.25) and rep.errors


def test_tolerance_prefers_the_dtype_specific_entry():
    module = types.SimpleNamespace(TOLERANCES={"a": (1.0, 1.0), ("a", torch.bfloat16): (2.0, 2.0)})
    assert driver.tolerance_for(module, "a", torch.bfloat16) == (2.0, 2.0)
    assert driver.tolerance_for(module, "a", torch.float16) == (1.0, 1.0)
    assert driver.tolerance_for(module, "b", torch.float16) == (1e-2, 1e-2)


def _fake_family():
    exact = torch.arange(8, dtype=torch.float32)

    def build_fns(subroutine, cfg, dtype, args):
        return {
            "fast": lambda: exact.clone(),
            "base": lambda: exact + 0.5,  # a different computation, like fp16 vs int8
            "oracle": lambda: exact.clone(),
        }

    return types.SimpleNamespace(
        SUBROUTINES=["op"],
        DEFAULT_CONFIGS={"op": [{"n": 8}]},
        REF_BACKEND="base",
        ORACLE_BACKEND="oracle",
        NON_GATING_BACKENDS=frozenset({"base"}),
        TOLERANCES={("op", torch.float32): (0.0, 0.0)},
        parse_args=lambda parser: None,
        resolve_configs=lambda args, sub: [{"n": 8}],
        build_fns=build_fns,
        describe=lambda sub, cfg, dtype: Work(shape="(8,)", params="n=8", flops=8.0),
    )


def _sweep_args(**kw):
    base = {
        "subroutine": "op",
        "dtype": "float32",
        "ref_backend": None,
        "backends": None,
        "interleave": False,
        "refcheck": True,
        "allow_output_mismatch": False,
        "profile": False,
        "warmup_iters": 0,
        "iters": 1,
        "timer": "cuda_events",
        "no_flush": True,
        "autotune": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_sweep_times_against_the_baseline_and_checks_against_the_oracle(monkeypatch):
    times = {"fast": 1.0, "base": 2.0, "oracle": 100.0}

    def fake_bench(fn, **_):
        name = next(k for k, v in fns.items() if v is fn)
        return driver.timing.Sample.from_times([times[name]], 0, "fake")

    module = _fake_family()
    fns = {}
    real_build = module.build_fns

    def build_and_remember(*a):
        fns.update(real_build(*a))
        return fns

    module.build_fns = build_and_remember
    monkeypatch.setattr(driver.timing, "bench_fn", fake_bench)
    rep = _Collect()
    driver.run_sweep("fake", module, _sweep_args(), rep)

    rows = {r.backend: r for r in rep.rows}
    assert set(rows) == {"fast", "base"}  # the oracle is checked against, never timed
    assert rows["fast"].refcheck == REFCHECK_PASS and rows["fast"].max_abs_diff == 0.0
    assert rows["base"].refcheck == REFCHECK_FAIL and not rep.errors and rep.warnings
    assert rows["fast"].ref_backend == "base" and rows["fast"].speedup_vs_ref == 2.0

    # --ref-backend still names both the baseline and the oracle.
    rep = _Collect()
    driver.run_sweep("fake", module, _sweep_args(ref_backend="fast"), rep)
    rows = {r.backend: r for r in rep.rows}
    assert rows["base"].refcheck == REFCHECK_FAIL and rows["base"].speedup_vs_ref == 0.5


# ---------------------------------------------------------------------------
# Workload schema: quantization and memory columns
# ---------------------------------------------------------------------------


def test_workload_row_fields_and_columns_agree():
    fields = {f.name for f in dataclasses.fields(WorkloadRow)}
    columns = set(WORKLOAD_COLUMNS) - set(ENVELOPE_COLUMNS)
    assert fields == columns
    assert {"quantization", "weight_mib", "peak_alloc_mib"} <= columns


def test_new_workload_columns_default_to_not_applicable():
    row = WorkloadRow(harness="service", subroutine="offline")
    # Empty, not "none": a harness that cannot see the engine does not know.
    assert row.quantization == "" and row.weight_mib is None and row.peak_alloc_mib is None
    cells = {c: format_cell(c, row.as_dict().get(c)) for c in WORKLOAD_COLUMNS}
    assert cells["quantization"] == cells["weight_mib"] == cells["peak_alloc_mib"] == ""
    quantized = WorkloadRow(
        harness="engine",
        subroutine="offline",
        quantization="int8_w8a8",
        weight_mib=123.456,
        peak_alloc_mib=789.01,
    )
    assert format_cell("weight_mib", quantized.weight_mib) == "123.5"
    assert format_cell("peak_alloc_mib", quantized.peak_alloc_mib) == "789.0"


# ---------------------------------------------------------------------------
# Footprint helpers
# ---------------------------------------------------------------------------


def test_parse_quantization():
    assert parse_quantization(None) is None
    for none in ("", "none", "NONE", " off "):
        assert parse_quantization(none) is None
    assert parse_quantization("int8_w8a8") == "int8_w8a8"
    spec = parse_quantization('{"scheme": "int8_w8a8", "ignore": "a.*,b.*"}')
    assert spec == {"scheme": "int8_w8a8", "ignore": "a.*,b.*"}
    assert quantization_label(spec) == "int8_w8a8[ignore=a.*,b.*]"
    assert quantization_label("int8_w8a8") == "int8_w8a8" and quantization_label(None) == ""
    with pytest.raises(ValueError, match="JSON"):
        parse_quantization("{scheme: int8}")


def test_served_quantization_reports_what_the_engine_serves():
    full = types.SimpleNamespace(quantization_info=lambda: None)
    int8 = types.SimpleNamespace(
        quantization_info=lambda: {"schemes": ["int8_w8a8"], "layers": 72, "lanes": ["x"]}
    )
    assert served_quantization(full) == "none"
    assert served_quantization(int8) == "int8_w8a8"
    assert served_quantization(object()) == ""  # cannot say, which is not "none"


def test_model_weight_mib_counts_weights_not_derived_tables():
    nn = torch.nn
    model = nn.Module()
    model.w = nn.Parameter(torch.zeros(256, 1024, dtype=torch.float16))
    model.tied = model.w  # shared, counted once
    # A derived table -- a positional encoding a streaming runtime grows -- is not
    # a weight, persistent or not.
    model.register_buffer("pe", torch.zeros(1, 5000, 256, dtype=torch.float16))
    quantized = nn.Module()
    quantized.weight = nn.Parameter(torch.zeros(1024, 1024, dtype=torch.int8), requires_grad=False)
    quantized.register_buffer("weight_scale", torch.zeros(1024, dtype=torch.float32))
    quantized.quant_method = types.SimpleNamespace(
        preserved_tensors=lambda layer: ("weight", "weight_scale", "smooth_inv")
    )
    model.layer = quantized  # no smooth_inv: an absent slot is skipped
    expected = 256 * 1024 * 2 + 1024 * 1024 + 1024 * 4
    assert model_weight_mib(model) == pytest.approx(expected / 2**20)
    assert model_weight_mib(None) is None


# ---------------------------------------------------------------------------
# Workload families: the flag and the row
# ---------------------------------------------------------------------------


def test_engine_and_accuracy_families_take_quantization():
    for name in ("engine", "accuracy"):
        parser = argparse.ArgumentParser(add_help=False)
        registry.get_family(name).parse_args(parser)
        opts = {s for a in parser._actions for s in a.option_strings}
        assert "--quantization" in opts, name
        assert not [m for m in _main_option_strings() if m.startswith("--quantization")]
    parser = argparse.ArgumentParser(add_help=False)
    registry.get_family("accuracy").parse_args(parser)
    assert parser.parse_args(["--quantization", "none", "int8_w8a8"]).quantization == [
        "none",
        "int8_w8a8",
    ]


def test_accuracy_row_names_a_quantized_cell_and_leaves_a_full_precision_one_alone():
    accuracy = registry.get_family("accuracy")
    base = {
        "manifest": "m.jsonl",
        "ckpt": "ckpt",
        "architecture": "conformer",
        "decode_method": "ctc",
        "decode_options": "-",
        "service_mode": "offline",
        "chunk_size": 0,
        "vad_mode": "off",
        "vad_backend": "-",
        "vad_options": "-",
        "dtype": "float16",
        "max_batch_size": 16,
        "metric": "wer",
        "normalizer": "english",
        "utterances": 20,
        "audio_seconds": 100.0,
        "error_rate_pct": 4.0,
        "substitutions": 1,
        "deletions": 1,
        "insertions": 1,
        "ref_units": 100,
        "wall_seconds": 1.0,
        "rtfx": 100.0,
        "utt_per_second": 20.0,
        "latency_p50_ms": 10.0,
        "latency_p99_ms": 20.0,
    }
    full = accuracy._to_workload_row(
        accuracy.Row(**base, quantization="none", weight_mib=100.0, peak_alloc_mib=300.0), ""
    )
    assert full.quantization == "none" and full.weight_mib == 100.0
    assert "quantization" not in full.params and full.shape == "ctc/offline/float16/bs16"
    int8 = accuracy._to_workload_row(
        accuracy.Row(**base, quantization="int8_w8a8", weight_mib=60.0, peak_alloc_mib=250.0), ""
    )
    assert "quantization=int8_w8a8" in int8.params and int8.shape.endswith("/int8_w8a8")
    assert int8.peak_alloc_mib == 250.0
