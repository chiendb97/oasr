# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The native (CUTLASS 3.x) scaled-GEMM lanes on Hopper (sm90) and Blackwell (sm100).

Two halves:

* **The config space and the lane rules** -- pure functions of an SM family and a
  compute capability, checked on any machine: every native tile is one CUTLASS
  builds (one that is not fails its whole JIT module), the default lane on each
  device, int8 on sm100 confined to compute capability 10.0 (tcgen05 ``kind::i8``
  exists for ``sm_100a`` only) and built for that target, tuning entries routed to
  the lane they were measured on.
* **The kernels** -- run only where the device's native lane exists (H100, B200),
  and skipped elsewhere: every variant against the reference, the quantizing
  epilogue against the unfused pair bit for bit, every tile computing the same
  bits (the batch invariance each format's lane promises), graph replay, and the
  ``mma.sync`` A/B arm on the same device.
"""

from __future__ import annotations

import pytest
import torch
from helpers import assert_graph_replay

import oasr
import oasr.jit.core as jcore
import oasr.jit.scaled_mm as jsm
from oasr.functionals.quant import static_multiplier
from oasr.functionals.scaled_mm import (
    _get_scaled_mm_module,
    activation_id,
    reference_scaled_mm,
    scaled_mm_quant,
)
from oasr.jit.scaled_mm import (
    LANE_ID,
    NATIVE_LANE_ID,
    NativeScaledMmConfig,
    get_production_configs,
    scaled_mm_func_name,
    scaled_mm_quant_func_name,
)

_DTYPES = (torch.float16, torch.bfloat16)
_ACTIVATIONS = (None, "relu", "swish", "gelu", "gelu_tanh")
_FP8 = torch.float8_e4m3fn


def _as_device(monkeypatch, major: int, minor: int) -> None:
    monkeypatch.setattr(jsm, "_device_capability", lambda: (major, minor))


# =============================================================================
# The config space and the lane rules (any machine)
# =============================================================================


class TestConfigSpace:
    @pytest.mark.parametrize("key", sorted(NATIVE_LANE_ID))
    def test_every_native_tile_is_buildable(self, key):
        sm, fmt = key
        cfgs = get_production_configs(sm, fmt, NATIVE_LANE_ID[key])
        assert len(cfgs) == len(jsm._NATIVE_TILES[sm]), "a production tile failed the predicate"
        for name, cfg in cfgs.items():
            assert isinstance(cfg, NativeScaledMmConfig)
            assert (
                name.startswith(f"sm{sm}_") and cfg.fmt == fmt and cfg.lane == NATIVE_LANE_ID[key]
            )
            # One summation order for every tile: one K tile is 128 elements, and
            # no native variant splits K.
            assert cfg.block_k == 128 and not cfg.stream_k

    @pytest.mark.parametrize(
        "cfg",
        [
            NativeScaledMmConfig(90, 256, 128, 2, 1, "pingpong"),  # a ping-pong WG owns <= 128 rows
            NativeScaledMmConfig(90, 64, 128, 1, 1, "cooperative"),  # two WGs need >= 128 rows
            NativeScaledMmConfig(90, 64, 128, 1, 1, "1sm"),  # not an sm90 schedule
            NativeScaledMmConfig(100, 256, 128, 1, 1, "1sm"),  # the 1-SM MMA is <= 128 rows
            NativeScaledMmConfig(100, 256, 128, 1, 1, "2sm"),  # a CTA pair needs an even cluster M
            NativeScaledMmConfig(100, 256, 160, 2, 1, "2sm"),  # 16-bit 2-SM TMA epilogue: N%64
            NativeScaledMmConfig(90, 64, 120, 1, 1, "pingpong"),  # N % 16
            NativeScaledMmConfig(90, 64, 128, 1, 1, "pingpong", block_k=64),  # BK = 128
        ],
        ids=lambda c: c.compile_name,
    )
    def test_the_predicate_refuses_what_cutlass_asserts(self, cfg):
        assert not jsm._native_buildable(cfg)

    @pytest.mark.parametrize("key", sorted(NATIVE_LANE_ID))
    def test_a_tuning_entry_round_trips(self, key):
        for cfg in get_production_configs(key[0], key[1], NATIVE_LANE_ID[key]).values():
            params = jsm.config_to_params(cfg)
            assert params["lane"] == NATIVE_LANE_ID[key]
            assert jsm.config_from_params(params) == cfg

    def test_names_are_unique_across_lanes(self):
        names = [
            scaled_mm_func_name(c)
            for key, lane in NATIVE_LANE_ID.items()
            for c in get_production_configs(key[0], key[1], lane).values()
        ]
        names += [
            scaled_mm_func_name(c)
            for fmt in jsm.FORMATS
            for c in get_production_configs(90, fmt, LANE_ID[fmt]).values()
        ]
        assert len(names) == len(set(names))


class TestDefaultLane:
    @pytest.mark.parametrize(
        "cap, int8_lane, fp8_lane",
        [
            ((9, 0), "sm90_int8_wgmma", "sm90_fp8_wgmma"),
            ((10, 0), "sm100_int8_umma", "sm100_fp8_umma"),
            # B300: the sm100 family, but no tcgen05 int8 to build for.
            ((10, 3), "sm80_int8_mma", "sm100_fp8_umma"),
            ((12, 0), "sm80_int8_mma", "sm89_fp8_mma"),
            ((8, 9), "sm80_int8_mma", "sm89_fp8_mma"),
        ],
    )
    def test_each_device_defaults_to_its_fastest_lane(self, monkeypatch, cap, int8_lane, fp8_lane):
        _as_device(monkeypatch, *cap)
        sm = jcore.sm_family(*cap)
        assert jsm.default_lane(sm, "int8") == int8_lane
        assert jsm.default_lane(sm, "fp8") == fp8_lane

    def test_naming_the_mma_sync_lane_on_hopper(self, monkeypatch):
        """The A/B arm: the 2.x configs, on a device whose default is native."""
        _as_device(monkeypatch, 9, 0)
        cfgs = get_production_configs(90, "int8", LANE_ID["int8"])
        assert cfgs and all(not isinstance(c, NativeScaledMmConfig) for c in cfgs.values())

    def test_a_lane_is_checked_against_its_family_and_format(self):
        with pytest.raises(ValueError, match="sm90's, not sm100's"):
            get_production_configs(100, "int8", "sm90_int8_wgmma")
        with pytest.raises(ValueError, match="serves fp8, not int8"):
            get_production_configs(90, "int8", "sm90_fp8_wgmma")
        with pytest.raises(ValueError, match="unknown scaled_mm lane"):
            get_production_configs(90, "int8", "sm90_int4_wgmma")

    def test_the_8bit_output_alignment(self):
        assert jsm.quant_n_alignment("sm90_int8_wgmma") == 16
        assert jsm.quant_n_alignment("sm100_fp8_umma") == 16
        assert jsm.quant_n_alignment("sm80_int8_mma") == jsm.N_ALIGNMENT


class TestHeuristic:
    @pytest.mark.parametrize("key", sorted(NATIVE_LANE_ID))
    def test_it_picks_a_production_tile_and_no_padding_tile(self, key):
        sm, fmt = key
        lane = NATIVE_LANE_ID[key]
        built = get_production_configs(sm, fmt, lane)
        for m in (1, 7, 64, 65, 200, 1000, 4096, 16384):
            for n, k in ((256, 256), (2048, 256), (256, 2048), (896, 18944), (5120, 1280)):
                cfg = jsm.heuristic_config(m, n, k, sm=sm, num_sms=132, fmt=fmt, lane=lane)
                assert built[cfg.compile_name] == cfg
                assert cfg.block_m == 64 or 2 * m > cfg.block_m, (m, n, k, cfg.compile_name)

    def test_a_decode_step_takes_the_smallest_tile(self):
        cfg = jsm.heuristic_config(1, 256, 256, sm=90, num_sms=132, lane="sm90_int8_wgmma")
        assert (cfg.block_m, cfg.block_n, cfg.schedule) == (64, 64, "pingpong")


class TestTuningEntries:
    def test_an_entry_serves_only_the_lane_it_was_measured_on(self, monkeypatch):
        _as_device(monkeypatch, 9, 0)
        native = next(iter(get_production_configs(90, "int8", "sm90_int8_wgmma").values()))
        mma = next(iter(get_production_configs(90, "int8", LANE_ID["int8"]).values()))
        assert jsm._unusable_reason("scaled_mm", native, 90, "sm90_int8_wgmma") is None
        assert jsm._unusable_reason("scaled_mm", mma, 90, LANE_ID["int8"]) is None
        assert "belongs to lane" in jsm._unusable_reason("scaled_mm", native, 90, LANE_ID["int8"])
        assert "belongs to lane" in jsm._unusable_reason("scaled_mm", mma, 90, "sm90_int8_wgmma")
        # The default lane on this (faked) Hopper is the native one.
        assert jsm._unusable_reason("scaled_mm", native, 90) is None


class TestBuild:
    def test_the_native_template_renders(self):
        from oasr.jit.templates import render_template

        for key, lane in NATIVE_LANE_ID.items():
            cfg = next(iter(get_production_configs(key[0], key[1], lane).values()))
            text = render_template(
                "scaled_mm_cutlass_template_sm90.cu.jinja",
                op_name="probe",
                fp8=cfg.fp8,
                func_name=scaled_mm_func_name(cfg),
                quant_func_name=scaled_mm_quant_func_name(cfg),
                tile_m=cfg.block_m,
                tile_n=cfg.block_n,
                tile_k=cfg.block_k,
                cluster_m=cfg.cluster_m,
                cluster_n=cfg.cluster_n,
                schedule=cfg.cpp_schedule,
                sm_version=cfg.sm,
            )
            assert f"NativeSchedule::{cfg.cpp_schedule}, {cfg.sm}>" in text
            assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({scaled_mm_quant_func_name(cfg)}" in text

    def test_int8_on_sm100_builds_for_sm_100a(self, monkeypatch):
        """``sm_100f`` defines no tcgen05 ``kind::i8``: the int8 module must name ``sm_100a``."""
        monkeypatch.setattr(jcore, "_get_cuda_arch", lambda: (10, 0))
        flags = jcore._default_cuda_cflags(jsm._NATIVE_GENCODE[(100, "int8")])
        assert "-gencode=arch=compute_100a,code=sm_100a" in flags
        assert "-gencode=arch=compute_100f,code=sm_100f" in jcore._default_cuda_cflags()
        assert (100, "fp8") not in jsm._NATIVE_GENCODE  # fp8 keeps the family target


# =============================================================================
# The kernels (H100 / B200 only)
# =============================================================================


def _native_lanes_here() -> list:
    """``[(fmt, lane), ...]`` of the native lanes this device runs."""
    if not torch.cuda.is_available():
        return []
    sm = jcore.sm_family(*torch.cuda.get_device_capability())
    return [
        (fmt, NATIVE_LANE_ID[(sm, fmt)])
        for fmt in jsm.FORMATS
        if jsm.native_lane_available(sm, fmt)
    ]


_HERE = _native_lanes_here()
_VARIANTS = [
    (fmt, lane, name)
    for fmt, lane in _HERE
    for name in sorted(get_production_configs(fmt=fmt, lane=lane))
]
needs_native = pytest.mark.skipif(not _HERE, reason="no native scaled-GEMM lane on this device")


def _operands(fmt: str, m: int, n: int, k: int, dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if fmt == "int8":
        a = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8, device="cuda")
        b = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8, device="cuda")
        a_scale = torch.rand(m, generator=g, device="cuda") * 0.01 + 1e-3
        b_scale = torch.rand(n, generator=g, device="cuda") * 0.01 + 1e-3
    else:
        a = (torch.randn(m, k, generator=g, device="cuda") * 64.0).clamp(-448, 448).to(_FP8)
        b = (torch.randn(n, k, generator=g, device="cuda") * 64.0).clamp(-448, 448).to(_FP8)
        a_scale = torch.rand(m, generator=g, device="cuda") * 1e-3 + 1e-4
        b_scale = torch.rand(n, generator=g, device="cuda") * 1e-3 + 1e-4
    bias = torch.randn(n, generator=g, device="cuda").to(dtype)
    return a, b, a_scale, b_scale, bias


def _assert_close(fmt: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    """An output ulp (int8: the accumulator is exact), plus fp8's fp32 summation."""
    assert not torch.isnan(got).any()
    if fmt == "int8":
        rel, atol = (2.0**-9 if got.dtype == torch.float16 else 2.0**-6), 1e-3
    else:
        rel, atol = (2.0**-8 if got.dtype == torch.float16 else 2.0**-6), 2e-3
    diff = (got.float() - ref.float()).abs()
    assert bool((diff <= ref.float().abs() * rel + atol).all()), f"max diff {float(diff.max())}"


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


@needs_native
@pytest.mark.cuda
class TestEveryNativeVariant:
    @pytest.mark.parametrize("fmt,lane,name", _VARIANTS)
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("m,n,k", [(1710, 128, 384), (200, 264, 384), (1, 256, 2048)])
    def test_variant(self, fmt, lane, name, dtype, m, n, k):
        cfg = get_production_configs(fmt=fmt, lane=lane)[name]
        fn = getattr(_get_scaled_mm_module(fmt, lane), scaled_mm_func_name(cfg))
        a, b, a_scale, b_scale, bias = _operands(fmt, m, n, k, dtype, seed=m + n)
        for act in _ACTIVATIONS:
            out = torch.full((m, n), float("nan"), dtype=dtype, device="cuda")
            fn(out, a, b, a_scale, b_scale, bias, activation_id(act))
            _assert_close(fmt, out, reference_scaled_mm(a, b, a_scale, b_scale, bias, act, dtype))

    @pytest.mark.parametrize("fmt,lane,name", _VARIANTS)
    def test_a_static_per_tensor_scale(self, fmt, lane, name):
        """The activation-scale leaf's runtime switch: one scale for every row."""
        cfg = get_production_configs(fmt=fmt, lane=lane)[name]
        fn = getattr(_get_scaled_mm_module(fmt, lane), scaled_mm_func_name(cfg))
        a, b, _, b_scale, bias = _operands(fmt, 300, 512, 256, torch.float16)
        scale = torch.tensor([3e-3], device="cuda")
        out = torch.empty(300, 512, dtype=torch.float16, device="cuda")
        fn(out, a, b, scale, b_scale, None, activation_id("swish"))
        _assert_close(fmt, out, reference_scaled_mm(a, b, scale, b_scale, None, "swish"))

    @pytest.mark.parametrize("fmt,lane,name", _VARIANTS)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_quantizing_epilogue_is_the_unfused_pair(self, fmt, lane, name, dtype):
        """The 8-bit-output epilogue == ``quantize_static(scaled_mm(...))``, bit for bit."""
        cfg = get_production_configs(fmt=fmt, lane=lane)[name]
        m, n, k = 333, 1024, 512
        qdtype = torch.int8 if fmt == "int8" else _FP8
        a, b, a_scale, b_scale, bias = _operands(fmt, m, n, k, dtype)
        out_mul = static_multiplier(torch.tensor([0.05], device="cuda"), None, n)
        fp = torch.empty(m, n, dtype=dtype, device="cuda")
        getattr(_get_scaled_mm_module(fmt, lane), scaled_mm_func_name(cfg))(
            fp, a, b, a_scale, b_scale, bias, activation_id("swish")
        )
        unfused = oasr.quantize_static(fp, out_mul, dtype=qdtype)
        q = torch.empty(m, n, dtype=qdtype, device="cuda")
        getattr(_get_scaled_mm_module(fmt, lane), scaled_mm_quant_func_name(cfg))(
            q,
            a,
            b,
            a_scale,
            b_scale,
            bias,
            out_mul,
            activation_id("swish"),
            int(dtype != torch.float16),
        )
        assert torch.equal(_bytes(q), _bytes(unfused)), cfg.compile_name


@needs_native
@pytest.mark.cuda
class TestNativeLaneInvariants:
    @pytest.mark.parametrize("fmt,lane", _HERE)
    @pytest.mark.parametrize("m,n,k", [(1000, 768, 2048), (300, 896, 18944)])
    def test_every_tile_computes_the_same_bits(self, fmt, lane, m, n, k):
        """int8: an exact accumulator; fp8: one summation order for every tile."""
        a, b, a_scale, b_scale, bias = _operands(fmt, m, n, k, torch.float16)
        mod = _get_scaled_mm_module(fmt, lane)
        outs = {}
        for name, cfg in get_production_configs(fmt=fmt, lane=lane).items():
            out = torch.empty(m, n, dtype=torch.float16, device="cuda")
            getattr(mod, scaled_mm_func_name(cfg))(out, a, b, a_scale, b_scale, bias, 3)
            outs[name] = out
        first = next(iter(outs.values()))
        assert all(torch.equal(o, first) for o in outs.values()), "a tile computed other bits"

    @pytest.mark.parametrize("fmt,lane", _HERE)
    def test_rows_are_bitwise_independent_of_the_batch(self, fmt, lane):
        n, k = 768, 2048
        a, b, a_scale, b_scale, bias = _operands(fmt, 1000, n, k, torch.float16)
        cuts = (0, 1, 17, 81, 1000)
        full = oasr.scaled_mm(a, b, a_scale, b_scale, bias, "swish", lane=lane)
        parts = [
            oasr.scaled_mm(
                a[lo:hi].contiguous(),
                b,
                a_scale[lo:hi].contiguous(),
                b_scale,
                bias,
                "swish",
                lane=lane,
            )
            for lo, hi in zip(cuts, cuts[1:])
        ]
        assert torch.equal(torch.cat(parts), full)

    @pytest.mark.parametrize("fmt,lane", _HERE)
    def test_an_unaligned_quantizing_launch_unfuses(self, fmt, lane):
        """N % 16 != 0 has no 8-bit TMA store: the functional runs the unfused pair."""
        m, n, k = 64, 264, 256
        qdtype = torch.int8 if fmt == "int8" else _FP8
        a, b, a_scale, b_scale, bias = _operands(fmt, m, n, k, torch.float16)
        out_mul = static_multiplier(torch.tensor([0.05], device="cuda"), None, n)
        q = scaled_mm_quant(a, b, a_scale, b_scale, bias, "relu", out_mul, torch.float16, lane=lane)
        y = oasr.scaled_mm(a, b, a_scale, b_scale, bias, "relu", lane=lane)
        assert torch.equal(_bytes(q), _bytes(oasr.quantize_static(y, out_mul, dtype=qdtype)))

    @pytest.mark.parametrize("fmt,lane", _HERE)
    def test_the_mma_sync_arm_agrees(self, fmt, lane):
        """The A/B arm on the same device: both lanes within the reference's tolerance."""
        a, b, a_scale, b_scale, bias = _operands(fmt, 512, 1024, 512, torch.bfloat16)
        ref = reference_scaled_mm(a, b, a_scale, b_scale, bias, "gelu", torch.bfloat16)
        for which in (lane, LANE_ID[fmt]):
            got = oasr.scaled_mm(
                a, b, a_scale, b_scale, bias, "gelu", out_dtype=torch.bfloat16, lane=which
            )
            _assert_close(fmt, got, ref)

    @pytest.mark.parametrize("fmt,lane", _HERE)
    def test_graph_replay_recomputes(self, fmt, lane):
        a, b, a_scale, b_scale, bias = _operands(fmt, 256, 512, 256, torch.float16)
        out = torch.empty(256, 512, dtype=torch.float16, device="cuda")

        def launch():
            return oasr.scaled_mm(a, b, a_scale, b_scale, bias, "relu", out=out, lane=lane)

        def mutate():
            a.copy_((-a.float()).to(a.dtype))
            a_scale.mul_(2.0)

        assert_graph_replay(
            launch,
            out=out,
            mutate=mutate,
            expected=lambda: reference_scaled_mm(a, b, a_scale, b_scale, bias, "relu"),
            rtol=2.0**-8,
            atol=2e-3,
        )
