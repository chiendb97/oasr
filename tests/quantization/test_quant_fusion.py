# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Quantize launches removed: fused norms, GEMM chains and static input scales.

Three mechanisms let a quantized layer receive its input already in int8:

* a pre-norm declared ``fuse_output_quant`` emits per-token int8 itself
  (``LayerNorm.quantize_output``);
* a declared :class:`~oasr.quantization.GemmChain` whose consumer has a static
  input scale makes the producer's epilogue quantize for it
  (``W8A8Method.emit``);
* the ``static_inputs`` algorithm calibrates that static scale.

The property that makes all three safe to ship: every fused kernel is
bit-identical to the unfused composition it replaces, so a whole model with its
links armed returns exactly what it returns with them disarmed.  That is the
oracle here -- ``torch.equal`` on the encoder output, not a tolerance.
"""

from __future__ import annotations

import copy

import pytest
import torch
from helpers import tiny_conformer

from oasr.layers import Linear, QuantizedActivation
from oasr.quantization import (
    INT8_W8A8,
    INT8_W8A8_STATIC,
    GemmChain,
    QuantGroup,
    QuantizationSpec,
    QuantLayout,
    SmoothGroup,
    apply_quantization,
    bind_quantization,
    quantized_layers,
)
from oasr.quantization.algorithms import build_algorithm
from oasr.quantization.export import quantize_model
from oasr.quantization.recipes import get_recipe

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernels are CUDA")


def _loop(model: torch.nn.Module) -> None:
    """A calibration loop: random features through the encoder."""
    param = next(p for p in model.parameters() if p.is_floating_point())
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for t in (40, 64):
            feats = (torch.randn(1, t, 80, generator=g) * 2).to(param.device, param.dtype)
            model.encode_offline(feats, torch.tensor([t], device=param.device))


def _feats(device="cpu", dtype=torch.float32):
    g = torch.Generator().manual_seed(2)
    feats = torch.randn(3, 70, 80, generator=g).to(device, dtype)
    return feats, torch.tensor([70, 47, 12], device=device)


def _fused_model(recipe: str = "smoothquant_w8a8_fused"):
    model = tiny_conformer()
    quantize_model(model, get_recipe(recipe), _loop)
    return model


def _served(model, dtype):
    model = model.to(device="cuda", dtype=dtype)
    bind_quantization(model)
    return model


def _disarm(model) -> None:
    """Every link off: each quantized layer quantizes its own input again."""
    for layer in quantized_layers(model).values():
        layer.quant_method.emit = None
        layer.quant_method.input_source = None
    for module in model.modules():
        if "quantize_output" in vars(module):
            del module.quantize_output


def _armed(model):
    norms = [m for m in model.modules() if vars(m).get("quantize_output")]
    layers = quantized_layers(model)
    producers = [p for p, layer in layers.items() if layer.quant_method.emit is not None]
    return norms, producers


# -----------------------------------------------------------------------------
# The static scheme on one layer
# -----------------------------------------------------------------------------


def _static_linear(scale: float = 0.05) -> Linear:
    torch.manual_seed(0)
    holder = torch.nn.Module()
    holder.layer = Linear(64, 96)
    holder.layer.register_buffer("input_scale", torch.tensor([scale]))
    spec = QuantizationSpec((QuantGroup(INT8_W8A8_STATIC, ("layer",)),))
    apply_quantization(holder, spec, quantize=True)
    bind_quantization(holder)
    return holder.layer


class TestStaticScheme:
    def test_storage_keeps_the_calibrated_scale(self):
        layer = _static_linear(0.05)
        sd = layer.state_dict()
        assert set(sd) == {"weight", "bias", "weight_scale", "input_scale"}
        assert torch.equal(sd["input_scale"], torch.tensor([0.05], dtype=torch.float32))
        layer.to(torch.bfloat16)
        assert layer.input_scale.dtype == torch.float32

    def test_runs_on_cpu_through_the_reference(self):
        layer = _static_linear(0.02)
        with torch.no_grad():
            out = layer(torch.randn(5, 64))
        assert out.shape == (5, 96) and torch.isfinite(out).all()

    @cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_kernel_agrees_with_the_reference(self, dtype):
        layer = _static_linear(0.02).cuda().to(dtype)
        bind_quantization(layer)
        assert layer.quant_method.lane is not None
        x = torch.randn(300, 64, device="cuda").to(dtype)
        with torch.no_grad():
            got = layer(x)
            ref = layer.quant_method.apply_reference(layer, x)
        rel = 2.0**-9 if dtype == torch.float16 else 2.0**-6
        assert bool(((got.float() - ref.float()).abs() <= ref.float().abs() * rel + 1e-3).all())


# -----------------------------------------------------------------------------
# Links between layers
# -----------------------------------------------------------------------------


class TestLinks:
    @cuda
    def test_armed_exactly_where_declared(self):
        model = _served(_fused_model(), torch.float16)
        layout = model.quantization_layout()
        norms, producers = _armed(model)
        fused_groups = [g for g in layout.groups if g.fuse_output_quant]
        assert len(norms) == len(fused_groups) == 6  # 3 pre-norms x 2 blocks
        assert sorted(producers) == sorted(c.producer for c in layout.chains)
        layers = quantized_layers(model)
        for chain in layout.chains:
            assert "producer GEMM" in layers[chain.consumer].quant_method.input_source

    @cuda
    def test_dynamic_consumers_take_fused_norms_but_no_chain(self):
        """Online RTN: every scale per-token, so norms fuse and no producer can."""
        model = tiny_conformer()
        quantize_model(model, get_recipe("int8_w8a8_rtn"))
        model = _served(model, torch.float16)
        norms, producers = _armed(model)
        assert len(norms) == 6 and producers == []

    @cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("recipe", ["smoothquant_w8a8_fused", "int8_w8a8_rtn"])
    def test_a_fused_model_is_bitwise_the_unfused_model(self, recipe, dtype):
        """Every fused kernel equals the composition it replaces, bit for bit."""
        model = _served(_fused_model(recipe), dtype)
        unfused = copy.deepcopy(model)
        bind_quantization(unfused)
        _disarm(unfused)
        assert _armed(model)[0] and not _armed(unfused)[0]
        feats, lens = _feats("cuda", dtype)
        with torch.no_grad():
            got, got_len = model.encode_offline(feats, lens)
            want, want_len = unfused.encode_offline(feats, lens)
        assert torch.equal(got_len, want_len)
        assert torch.equal(got, want)

    @cuda
    def test_the_fused_path_actually_runs(self, monkeypatch):
        """No per-token quantize for a norm-fed GEMM, no static quantize for a chain consumer."""
        import oasr.quantization.methods.w8a8 as w8a8

        model = _served(_fused_model(), torch.float16)
        calls = {"per_token": 0, "static": 0}
        per_token, static = w8a8.quantize_per_token, w8a8.quantize_static

        def count(name, fn):
            def wrapped(*a, **k):
                calls[name] += 1
                return fn(*a, **k)

            return wrapped

        monkeypatch.setattr(w8a8, "quantize_per_token", count("per_token", per_token))
        monkeypatch.setattr(w8a8, "quantize_static", count("static", static))
        with torch.no_grad():
            model.encode_offline(*_feats("cuda", torch.float16))
        # Per block: linear_out, pointwise_conv1 and pointwise_conv2 still quantize.
        assert calls == {"per_token": 3 * 2, "static": 0}

    @cuda
    def test_the_fusion_switch_disarms_every_link(self, monkeypatch):
        import oasr.quantization.apply as apply_mod

        monkeypatch.setattr(apply_mod, "_FUSION_ENABLED", False)
        model = _served(_fused_model(), torch.float16)
        assert _armed(model) == ([], [])

    @cuda
    def test_a_rebind_on_the_host_disarms_everything(self):
        model = _served(_fused_model(), torch.float16)
        assert all(_armed(model))
        model = model.cpu().float()
        bind_quantization(model)
        assert _armed(model) == ([], [])
        with torch.no_grad():
            out, _ = model.encode_offline(*_feats())
        assert torch.isfinite(out).all()

    def test_a_producer_without_a_lane_refuses_to_arm(self):
        model = _fused_model()
        bind_quantization(model)  # on the host: no lane anywhere
        layers = quantized_layers(model)
        chain = model.quantization_layout().chains[0]
        producer, consumer = layers[chain.producer], layers[chain.consumer]
        mul, scale = consumer.quant_method._multiplier(consumer), consumer.input_scale
        assert producer.quant_method.emit_for(producer, chain.consumer, mul, scale) is False

    def test_the_reference_honours_an_armed_chain_bitwise(self):
        """Armed by hand on the host: the pair exchanges int8 and changes no bit."""
        from oasr.quantization.methods.w8a8 import OutputQuant

        model = _fused_model()
        bind_quantization(model)
        layers = quantized_layers(model)
        chain = model.quantization_layout().chains[0]
        producer, consumer = layers[chain.producer], layers[chain.consumer]
        x = torch.randn(2, 9, producer.in_features)
        with torch.no_grad():
            want = consumer(producer(x))
            mul = consumer.quant_method._multiplier(consumer)
            producer.quant_method.emit = OutputQuant(mul, consumer.input_scale, chain.consumer)
            mid = producer(x)
            got = consumer(mid)
        assert isinstance(mid, QuantizedActivation) and not mid.per_token
        assert torch.equal(got, want)


# -----------------------------------------------------------------------------
# The static_inputs algorithm
# -----------------------------------------------------------------------------


class TestStaticInputs:
    def test_chain_consumers_get_a_static_group_first(self):
        model = _fused_model()
        spec = model.quantization_spec
        layout = model.quantization_layout()
        for chain in layout.chains:
            assert spec.resolve(chain.consumer) == INT8_W8A8_STATIC
            assert spec.resolve(chain.producer) == INT8_W8A8
        assert spec.producer["static_inputs"]["layers"] == len(layout.chains)

    def test_the_scale_is_the_smoothed_absmax_step(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        layout = model.quantization_layout()
        spec = build_algorithm("smoothquant").apply(model, layout, spec, _loop)
        modules = dict(model.named_modules())
        seen = {}

        def watch(path):
            def hook(module, args):
                x = args[0].float() * module.smooth_inv
                seen[path] = max(seen.get(path, 0.0), float(x.abs().max()))

            return hook

        handles = [
            modules[c.consumer].register_forward_pre_hook(watch(c.consumer)) for c in layout.chains
        ]
        _loop(model)
        for h in handles:
            h.remove()
        build_algorithm("static_inputs", {"observer": "absmax"}).apply(model, layout, spec, _loop)
        for path, amax in seen.items():
            got = float(modules[path].input_scale)
            assert got == pytest.approx(amax / 127, rel=1e-6), path

    @pytest.mark.parametrize("observer", ["percentile", "mse"])
    def test_a_searched_clip_sits_at_or_below_the_maximum(self, observer):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        layout = model.quantization_layout()
        spec = build_algorithm("smoothquant").apply(model, layout, spec, _loop)
        clipped = copy.deepcopy(model)
        build_algorithm("static_inputs", {"observer": "absmax"}).apply(model, layout, spec, _loop)
        out = build_algorithm("static_inputs", {"observer": observer}).apply(
            clipped, layout, spec, _loop
        )
        lo, mid = out.producer["static_inputs"]["clip_over_max"]
        assert 0.0 < lo <= mid <= 1.0
        a, b = dict(model.named_modules()), dict(clipped.named_modules())
        for chain in layout.chains:
            assert float(b[chain.consumer].input_scale) <= float(a[chain.consumer].input_scale)

    def test_unknown_observers_are_refused(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        with pytest.raises(ValueError, match="unknown static_inputs observer"):
            build_algorithm("static_inputs", {"observer": "kl"}).apply(
                model, model.quantization_layout(), spec, _loop
            )

    def test_refuses_a_model_without_chains(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        with pytest.raises(ValueError, match="no GEMM chains"):
            build_algorithm("static_inputs").apply(model, QuantLayout(), spec, _loop)

    def test_refuses_without_calibration(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        with pytest.raises(ValueError, match="calibration"):
            build_algorithm("static_inputs").apply(model, model.quantization_layout(), spec, None)

    def test_a_chain_with_a_full_precision_end_is_skipped(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(
            model, INT8_W8A8, ignore=("encoder.encoders.0.feed_forward.w_1",)
        )
        out = build_algorithm("static_inputs").apply(
            model, model.quantization_layout(), spec, _loop
        )
        assert out.resolve("encoder.encoders.0.feed_forward.w_2") == INT8_W8A8
        assert out.resolve("encoder.encoders.1.feed_forward.w_2") == INT8_W8A8_STATIC


class TestLayout:
    def test_a_quantized_conformer_pins_its_fp_projections(self):
        """embed.out and linear_pos stay fp16 ahead of int8 layers: pinned, nothing else."""
        model = _fused_model("int8_w8a8_rtn")
        pinned = sorted(p for p, m in model.named_modules() if vars(m).get("batch_invariant"))
        n = len(model.encoder.encoders)
        assert pinned == sorted(
            ["encoder.embed.out.0"]
            + [f"encoder.encoders.{i}.self_attn.linear_pos" for i in range(n)]
        )
        assert not any(vars(m).get("batch_invariant") for m in tiny_conformer().modules())

    def test_a_pin_on_a_module_without_a_pinned_path_is_refused(self):
        model = tiny_conformer()
        layout = QuantLayout(batch_invariant=("encoder.embed.out.1",))  # the LayerNorm
        model.quantization_layout = lambda: layout
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        with pytest.raises(ValueError, match="cannot be pinned"):
            apply_quantization(model, spec, quantize=True)

    def test_only_a_norm_producer_may_fuse_its_output(self):
        with pytest.raises(ValueError, match="norm producer"):
            SmoothGroup(None, ("a",), "runtime", fuse_output_quant=True)

    def test_conformer_declares_one_chain_per_ffn(self):
        model = tiny_conformer()
        layout = model.quantization_layout()
        n = len(model.encoder.encoders)
        assert len(layout.chains) == 2 * n
        assert all(isinstance(c, GemmChain) for c in layout.chains)
        assert all(
            c.producer.endswith(".w_1") and c.consumer.endswith(".w_2") for c in layout.chains
        )


def test_a_quantized_activation_reports_its_scale_kind():
    q = torch.zeros(2, 3, 8, dtype=torch.int8)
    assert QuantizedActivation(q, torch.ones(2, 3), torch.float16).per_token
    assert not QuantizedActivation(q, torch.ones(1), torch.float16).per_token


class TestClipSearch:
    def test_percentile_is_the_bin_edge_where_the_share_is_reached(self):
        from oasr.quantization.calibration import clip_percentile

        counts = torch.zeros(100)
        counts[:90] = 1.0  # 90 % of the mass in the first 90 bins
        counts[99] = 10.0
        assert clip_percentile(counts, 100.0, 90.0) == pytest.approx(90.0)
        assert clip_percentile(counts, 100.0, 100.0) == pytest.approx(100.0)

    def test_mse_clips_a_lone_outlier_but_keeps_a_dense_tail(self):
        from oasr.quantization.calibration import clip_mse

        lone = torch.zeros(1000)
        lone[:100] = 1e6  # a hundred million values below 10 % of the range ...
        lone[999] = 1.0  # ... and one at the very top: saturating it is cheaper
        assert clip_mse(lone, 1.0, 127) < 0.2
        few = lone.clone()
        few[:100] = 1e3  # a hundred thousand: rounding all of them costs more
        assert clip_mse(few, 1.0, 127) > 0.5
        dense = torch.ones(1000)  # uniform up to the maximum: clipping only hurts
        assert clip_mse(dense, 1.0, 127) > 0.95

    def test_the_histogram_observer_counts_magnitudes(self):
        from oasr.quantization.calibration import HistogramPerTensor

        obs = HistogramPerTensor(4.0, bins=4)
        obs.update(torch.tensor([[-3.5, 0.5], [1.5, 9.0]]))  # 9.0 lands in the last bin
        assert obs.result().tolist() == [1.0, 1.0, 0.0, 2.0]
