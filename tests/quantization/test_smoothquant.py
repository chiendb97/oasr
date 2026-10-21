# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/quantization/algorithms/smoothquant.py`` and the Conformer's declared layout.

The oracle is **invariance**: a fold rewrites ``X @ Wᵀ`` as ``(X / s) @ (W·s)ᵀ``,
which is exact in real arithmetic, so smoothing a model *without quantizing it*
must leave its output unchanged.  That catches what a WER gate would only blur:
a consumer missing from a group, a non-commuting op between a norm and its GEMM
(the conv module's mask, pad and cache ``cat``), V rows that do not line up with
``linear_out``.  The scales are forced to random values in ``[0.1, 10]`` so every
channel moves, and the check runs on every path the encoder has: per-op, the
fused add+norm path, sequence packing, and paged streaming with a live CNN cache.

The oracle is itself checked: a layout whose consumer is wrong must fail it.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from helpers import tiny_conformer

import oasr.quantization.algorithms.smoothquant as sq
from oasr.quantization import INT8_W8A8, QuantizationSpec, QuantLayout, SmoothGroup
from oasr.quantization.algorithms import SmoothQuantOptions, build_algorithm
from oasr.quantization.calibration import AbsMaxPerChannel

_EXACT_FOLDS = "norm_affine,v_to_o,producer_rows"


def _feed(model: torch.nn.Module) -> None:
    """A calibration loop: random features through the encoder (statistics only)."""
    param = next(p for p in model.parameters() if p.is_floating_point())
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for t in (40, 64):
            feats = (torch.randn(1, t, 80, generator=g) * 2).to(param.device, param.dtype)
            model.encode_offline(feats, torch.tensor([t], device=param.device))


@pytest.fixture
def random_scales(monkeypatch):
    """Replace the computed ``s`` with log-uniform values in [0.1, 10]."""
    g = torch.Generator().manual_seed(11)

    def fake(group, act_amax, modules, opts):
        n = act_amax.numel()
        return (10.0 ** (torch.rand(n, generator=g) * 2 - 1)).to(act_amax.device)

    monkeypatch.setattr(sq, "_smoothing_vector", fake)


def _smoothed(model: torch.nn.Module, folds: str = _EXACT_FOLDS, layout=None):
    out = copy.deepcopy(model)
    spec = QuantizationSpec.for_model(out, INT8_W8A8)
    algo = build_algorithm("smoothquant", {"folds": folds})
    algo.apply(out, layout or out.quantization_layout(), spec, _feed)
    return out


def _inputs(device="cpu", dtype=torch.float32):
    g = torch.Generator().manual_seed(2)
    feats = torch.randn(2, 70, 80, generator=g).to(device, dtype)
    return feats, torch.tensor([70, 47], device=device)


def _set_fused(model, enabled: bool) -> None:
    for layer in model.encoder.encoders:
        layer._fused_addnorm = enabled


class TestInvariance:
    """fp32 on CPU: smoothing without quantizing changes the output by rounding only."""

    @pytest.mark.parametrize("path", ["per_op", "fused", "packed"])
    def test_every_offline_path(self, random_scales, path):
        model = tiny_conformer()
        if path == "per_op":
            _set_fused(model, False)
        elif path == "fused":
            assert all(layer._fused_addnorm for layer in model.encoder.encoders)
        smoothed = _smoothed(model)
        feats, lens = _inputs()
        forward = "forward_offline_packed" if path == "packed" else "forward_offline"
        with torch.no_grad():
            want, _ = getattr(model, forward)(feats, lens)
            got, _ = getattr(smoothed, forward)(feats, lens)
        assert not torch.equal(
            smoothed.encoder.encoders[0].feed_forward.w_1.weight,
            model.encoder.encoders[0].feed_forward.w_1.weight,
        ), "nothing was smoothed"
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_causal_conv_cache_path(self, random_scales):
        """The causal conv module left-pads; the smoothed values sit in its cache."""
        model = tiny_conformer(causal=True)
        smoothed = _smoothed(model)
        feats, lens = _inputs()
        with torch.no_grad():
            want, _ = model.forward_offline(feats, lens)
            got, _ = smoothed.forward_offline(feats, lens)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_the_oracle_catches_a_wrong_consumer(self, random_scales):
        """Fold norm_conv's scale into the *second* pointwise conv: must not be invariant."""
        model = tiny_conformer()
        p = "encoder.encoders.0"
        layout = QuantLayout(
            groups=(
                SmoothGroup(f"{p}.norm_conv", (f"{p}.conv_module.pointwise_conv2",), "norm_affine"),
            )
        )
        broken = _smoothed(model, folds="norm_affine", layout=layout)
        feats, lens = _inputs()
        with torch.no_grad():
            want, _ = model.forward_offline(feats, lens)
            got, _ = broken.forward_offline(feats, lens)
        assert (got - want).abs().max() > 1e-2

    def test_runtime_fold_is_the_smoother(self, random_scales):
        """A runtime group stores 1/s; applying it to the input undoes W·s exactly."""
        model = tiny_conformer()
        smoothed = _smoothed(model, folds="runtime")
        w2 = smoothed.encoder.encoders[0].feed_forward.w_2
        ref = model.encoder.encoders[0].feed_forward.w_2
        x = torch.randn(9, w2.in_features)
        torch.testing.assert_close(
            F.linear(x * w2.smooth_inv, w2.weight, w2.bias),
            F.linear(x, ref.weight, ref.bias),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="paged streaming needs CUDA")
    def test_paged_streaming_with_kv_and_cnn_caches(self, random_scales):
        """Chunk by chunk through the engine's paged backend: K/V and the CNN cache
        hold smoothed values, read back only by the smoothed model.  fp16 rounding
        differs between the two weight sets, so the bound is fp16 noise — a
        missing fold is orders of magnitude above it."""
        from oasr.cache.types import CacheConfig
        from oasr.engine.request import Request
        from oasr.engine.streaming_backend.paged import PagedStreamingBackend

        dtype, chunk = torch.float16, 16
        base = tiny_conformer(causal=True, output_size=256, attention_heads=4, linear_units=256)
        smoothed = _smoothed(base)

        def run(model):
            model = model.to(device="cuda", dtype=dtype)
            spec = model.cache_spec
            cache_cfg = CacheConfig(
                num_layers=spec.num_layers,
                n_kv_head=spec.n_kv_head,
                head_dim=spec.head_dim,
                hidden_dim=spec.hidden_dim,
                kernel_size=spec.conv_kernel_size,
                chunk_size=chunk,
                num_left_chunks=-1,
                block_size_frames=chunk,
                max_num_blocks=64,
                max_blocks_per_seq=16,
                max_batch_size=1,
                device=torch.device("cuda"),
                dtype=dtype,
            )
            cfg = SimpleNamespace(
                device="cuda", dtype=dtype, chunk_size=chunk, use_cuda_graphs=False
            )
            backend = PagedStreamingBackend(model, cfg, cache_cfg)
            g = torch.Generator().manual_seed(4)
            feats = (torch.randn(backend.decoding_window * 4, 80, generator=g) * 0.5).to(
                "cuda", dtype
            )
            req = Request(None, streaming=True)
            req.stream_id, req.feature_buffer = 0, feats
            req.feature_frames, req.feature_cursor, req.offset = int(feats.size(0)), 0, 0
            backend.allocate(req)
            outs = [backend.forward_step([req])[req.request_id].float().clone() for _ in range(4)]
            backend.free(req)
            return torch.cat(outs, dim=1)

        want, got = run(base), run(smoothed)
        assert want.shape == got.shape and want.shape[1] > chunk
        torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)


class TestScales:
    def test_formula_matches_an_independent_implementation(self):
        """llm-compressor's ``s = amax_x^a / amax_w^(1-a)`` over every consumer."""
        model = tiny_conformer()
        group = SmoothGroup(
            "encoder.encoders.0.norm_mha",
            ("encoder.encoders.0.self_attn.linear_qkv",),
            "norm_affine",
        )
        modules = dict(model.named_modules())
        act = torch.rand(32) * 5 + 0.01
        opts = SmoothQuantOptions(alpha=0.6)
        s = sq._smoothing_vector(group, act, modules, opts)
        w = modules[group.consumers[0]].weight.detach().abs().amax(dim=0)
        expected = (act**0.6 / w**0.4).clamp(min=opts.s_floor, max=opts.s_max)
        bound = modules[group.producer].weight.abs().clamp_min(
            modules[group.producer].bias.abs()
        ) / (opts.fp16_headroom * 65504.0 / (1.0 + 1e-6))
        torch.testing.assert_close(s, torch.maximum(expected, bound))

    def test_a_dead_weight_column_leaves_the_channel_alone(self):
        model = tiny_conformer()
        p = "encoder.encoders.0"
        w1 = dict(model.named_modules())[f"{p}.feed_forward.w_1"]
        with torch.no_grad():
            w1.weight[:, 3] = 0.0
        group = SmoothGroup(f"{p}.norm_ff", (f"{p}.feed_forward.w_1",), "norm_affine")
        s = sq._smoothing_vector(
            group, torch.full((32,), 4.0), dict(model.named_modules()), SmoothQuantOptions()
        )
        assert float(s[3]) == 1.0

    def test_folded_norms_stay_inside_fp16(self):
        """A tiny s would push γ/s past fp16's range; the lower clamp prevents it."""
        model = tiny_conformer()
        p = "encoder.encoders.0"
        group = SmoothGroup(f"{p}.norm_ff", (f"{p}.feed_forward.w_1",), "norm_affine")
        modules = dict(model.named_modules())
        s = sq._smoothing_vector(group, torch.full((32,), 1e-20), modules, SmoothQuantOptions())
        with torch.no_grad():
            sq._fold(group, s, modules, SmoothQuantOptions())
        assert float(modules[group.producer].weight.abs().max()) <= 0.5 * 65504.0


class TestRecipe:
    def test_refuses_a_model_with_no_declared_groups(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        with pytest.raises(ValueError, match="declares no smoothing groups"):
            build_algorithm("smoothquant").apply(model, QuantLayout(), spec, _feed)

    def test_refuses_without_calibration(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        with pytest.raises(ValueError, match="calibration"):
            build_algorithm("smoothquant").apply(model, model.quantization_layout(), spec, None)

    def test_unknown_options_are_refused(self):
        with pytest.raises(ValueError, match="unknown smoothquant option"):
            build_algorithm("smoothquant", {"beta": 1})

    def test_string_options_are_typed(self):
        assert build_algorithm("smoothquant", {"alpha": "0.7"}).options.alpha == 0.7

    def test_runtime_groups_are_recorded_in_the_spec(self):
        model = tiny_conformer()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        out = build_algorithm("smoothquant").apply(model, model.quantization_layout(), spec, _feed)
        assert "encoder.encoders.0.feed_forward.w_2" in out.runtime_smoothed
        assert "encoder.encoders.1.conv_module.pointwise_conv2" in out.runtime_smoothed
        assert out.producer["algorithm"] == "smoothquant"


class TestObservers:
    def test_valid_mask_excludes_padding(self):
        obs = AbsMaxPerChannel()
        x = torch.tensor([[[1.0, -2.0], [100.0, 100.0]]])
        obs.update(x, valid=torch.tensor([[True, False]]))
        torch.testing.assert_close(obs.result(), torch.tensor([1.0, 2.0]))
        assert obs.rows == 1

    def test_an_unfed_observer_raises(self):
        with pytest.raises(RuntimeError, match="no calibration"):
            AbsMaxPerChannel().result()
