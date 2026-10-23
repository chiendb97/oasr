# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Every architecture's ``quantization_layout()``, checked against its forward.

A layout is data a model declares about its own forward: which norm feeds which
GEMMs, where V goes, which GEMM's output only one other GEMM reads.  A wrong
declaration is silently wrong -- the smoothed model still produces a plausible
transcript -- so the checks here are structural and numerical, per architecture:

* **it resolves**: every path names a module of the declared kind, every
  quantized layer is smoothed by exactly one group (none twice), every glob
  matches;
* **SmoothQuant invariance**: smoothing with random scales in ``[0.1, 10]`` and
  random norm affines, *without* quantizing, leaves every forward path's output
  unchanged in fp32 -- offline, streaming, teacher-forced and incremental
  decode, preallocated and grown KV caches.  Runtime smoothers are emulated by
  multiplying each layer's input by its ``smooth_inv``, as the quantize kernel
  does, so all four fold kinds are checked together and must commute.  The
  calibration loop is each model's own ``calibration_forward``, so this also
  proves the loop reaches every declared group (an unobserved group raises);
* **the oracle is seen to fail**: a deliberately wrong group (a consumer
  missing, V rows that are K, a fold through a non-homogeneous activation) must
  move the output;
* **online int8** runs every forward path on the CPU reference, and on CUDA in
  fp16 / bf16 every quantized layer's kernel output agrees with its torch
  reference on the activations the real forward hands it.
"""

from __future__ import annotations

import copy
import dataclasses
import fnmatch
from typing import Callable, Dict

import pytest
import torch
from helpers import randomize_norms
from test_layer_waist import _tiny_configs

import oasr.quantization.algorithms.smoothquant as sq
from oasr.layers import QuantizableGemm, QuantizedActivation, layers_backend_override
from oasr.layers.norm import LayerNorm, RMSNorm
from oasr.models.registry import get_model_entry
from oasr.quantization import (
    FP8_W8A8,
    INT8_W8A8,
    QuantizationSpec,
    QuantLayout,
    SmoothGroup,
    apply_quantization,
    bind_quantization,
    quantized_layers,
)
from oasr.quantization.algorithms import build_algorithm

#: The architectures with a declared layout, i.e. everything SmoothQuant serves.
ARCHS = ("conformer", "whisper", "nemotron", "paraformer", "speech_llm")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the int8 kernels are CUDA")


# -----------------------------------------------------------------------------
# Tiny models and their forward paths
# -----------------------------------------------------------------------------


def _config(arch: str):
    cfg = _tiny_configs()[arch]
    if arch == "whisper":
        # The SOT prompt the calibration loop decodes from must be inside the
        # tiny vocabulary.
        cfg = dataclasses.replace(
            cfg, decoder_start_token_id=1, eos_token_id=2, forced_decoder_ids=[(1, 3), (2, 4)]
        )
    elif arch == "nemotron":
        # A trained-chunk geometry the streaming harness can step (as
        # tests/models/arch/test_nemotron.py's streaming rig).
        cfg = dataclasses.replace(
            cfg,
            default_prompt_id=3,
            encoder=dataclasses.replace(
                cfg.encoder,
                num_hidden_layers=2,
                conv_kernel_size=9,
                default_num_lookahead_tokens=1,
            ),
        )
    elif arch == "speech_llm":
        # Grouped KV, so the v_to_o tie across query heads is exercised.
        cfg = dataclasses.replace(
            cfg, text_num_attention_heads=4, text_num_key_value_heads=2, text_num_hidden_layers=2
        )
    return cfg


def _model(arch: str) -> torch.nn.Module:
    torch.manual_seed(0)
    model = get_model_entry(arch).model_cls.from_config(_config(arch)).eval()
    if arch == "nemotron":
        # Default inits leave the Transformer-XL biases at zero; give every
        # parameter something to carry.
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    randomize_norms(model)
    return model


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _feats(arch: str, model, *, batch: int = 2, device="cpu", dtype=torch.float32):
    """``(features, lengths)`` the arch's encoder accepts (second row shorter)."""
    g = _gen(7)
    if arch == "whisper":
        t, f = 2 * model.config.max_source_positions, model.config.num_mel_bins
        lens = torch.full((batch,), t)
    elif arch == "speech_llm":
        t, f = 2 * model.config.audio_max_source_positions, model.config.audio_num_mel_bins
        lens = torch.tensor([t, t - 13][:batch])
    elif arch == "nemotron":
        t, f = model.encoder.subsampling_rate * 24, model.encoder.config.num_mel_bins
        lens = torch.tensor([t, t - 40][:batch])
    else:
        t, f = 70, 80
        lens = torch.tensor([t, 47][:batch])
    feats = torch.randn(batch, t, f, generator=g)
    return feats.to(device, dtype), lens.to(device)


def _calibrate(arch: str) -> Callable[[torch.nn.Module], None]:
    """A calibration loop: the model's own ``calibration_forward``, one utterance at a time."""

    def loop(model: torch.nn.Module) -> None:
        param = next(p for p in model.parameters() if p.is_floating_point())
        feats, lens = _feats(arch, model, device=param.device, dtype=param.dtype)
        with torch.no_grad():
            for b in range(feats.size(0)):
                # A fixed-window frontend (Whisper's log-mel) pads every utterance
                # to the window and reports the real length; the others are cut.
                n = feats.size(1) if arch in ("whisper", "speech_llm") else int(lens[b])
                model.calibration_forward(feats[b : b + 1, :n], lens[b : b + 1])

    return loop


def _tokens(n: int, length: int, vocab: int, seed: int = 9) -> torch.Tensor:
    return torch.randint(5, vocab, (n, length), generator=_gen(seed))


def _paths_conformer(model, device="cpu", dtype=torch.float32) -> Dict[str, torch.Tensor]:
    from oasr.models.decoders.transformer_decoder import add_sos_eos, reverse_pad_list

    feats, lens = _feats("conformer", model, device=device, dtype=dtype)
    out: Dict[str, torch.Tensor] = {}
    out["ctc"], _ = model.forward_offline(feats, lens)
    hidden, hl = model.encode_offline(feats, lens)
    cfg = model.decoder.config
    ys = _tokens(2, 6, cfg.vocab_size - 1).to(device)
    ys[1, 4:] = -1
    ys_lens = (ys != -1).sum(dim=1)
    sos = eos = cfg.vocab_size - 1
    ys_in, _ = add_sos_eos(ys, sos, eos, -1)
    r_in, _ = add_sos_eos(reverse_pad_list(ys, ys_lens, -1), sos, eos, -1)
    out["rescore_l"], out["rescore_r"] = model.decoder(hidden, hl, ys_in, ys_lens + 1, r_in)
    # Incremental: the AR entry, its cache the pre-norm history.
    left = model.decoder.left_decoder
    caches, steps = None, []
    for offset in range(4):
        logits, caches = left.forward_one_step(hidden, hl, ys_in[:, offset], offset, caches)
        steps.append(logits)
    out["one_step"] = torch.stack(steps, dim=1)
    return out


def _paths_whisper(model, device="cpu", dtype=torch.float32) -> Dict[str, torch.Tensor]:
    feats, lens = _feats("whisper", model, device=device, dtype=dtype)
    out: Dict[str, torch.Tensor] = {}
    # The encoder output itself is *not* an invariant: its final norm absorbs
    # the scale of every decoder cross-attention K/V projection, so it is
    # handed over in the smoothed domain -- correct only because nothing but
    # those projections reads it.  Everything downstream of them is compared.
    enc, _ = model.encode_offline(feats, lens)
    prompt = torch.tensor([[1, 3, 4]] * 2, device=device)
    feed = _tokens(2, 5, model.config.vocab_size).to(device)
    for name, kwargs in (("grown", {}), ("capacity", {"capacity": 12})):
        logits, state = model.decoder.prefill(enc, prompt, **kwargs)
        steps = [logits]
        for t in range(feed.size(1)):
            logits, state = model.decoder.step(feed[:, t], state)
            steps.append(logits)
        out[f"decode_{name}"] = torch.stack(steps, dim=1)
    attn, logits = model.decoder.cross_attention(
        enc, torch.cat([prompt, feed], dim=1), heads=[(0, 0), (0, 1)]
    )
    out["align_attn"], out["align_logits"] = attn, logits
    return out


def _paths_nemotron(model, device="cpu", dtype=torch.float32) -> Dict[str, torch.Tensor]:
    feats, lens = _feats("nemotron", model, device=device, dtype=dtype)
    out: Dict[str, torch.Tensor] = {}
    out["offline"], _ = model.encode_offline(feats, lens)
    if device == "cpu":
        out["streaming"] = _stream_nemotron(model, feats[:1], chunk_size=4)
    return out


def _stream_nemotron(model, feats: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Chunk by chunk through the paged K/V + slot-state caches (CPU fp32): the
    smoothed V and the post-GLU conv tail live in the caches across chunks."""
    from oasr.cache import AttentionCacheManager, BlockPool, CacheConfig, SlotStateCache
    from oasr.cache.cnn_cache import CONV_STATE, conv_state_spec

    enc = model.encoder
    spec = enc.cache_spec
    num_left_chunks = -(-enc.fixed_attention_window // chunk_size) + 1
    cfg = CacheConfig(
        num_layers=spec.num_layers,
        n_kv_head=spec.n_kv_head,
        head_dim=spec.head_dim,
        hidden_dim=spec.hidden_dim,
        kernel_size=spec.conv_kernel_size,
        stream_states=spec.stream_states,
        prefill_kv_window=True,
        chunk_size=chunk_size,
        num_left_chunks=num_left_chunks,
        block_size_frames=chunk_size,
        max_num_blocks=num_left_chunks + 8,
        max_blocks_per_seq=num_left_chunks + 4,
        max_batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    att = AttentionCacheManager(BlockPool(cfg), cfg)
    states = SlotStateCache(
        [conv_state_spec(cfg), *cfg.stream_states],
        max_batch_size=1,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    att.allocate_stream(0, slot_id=0)
    states.allocate_stream(0, slot_id=0)
    window = chunk_size * enc.subsampling_rate
    slot = torch.tensor([0], dtype=torch.long)
    outs, offset = [], 0
    for step in range(feats.size(1) // window):
        att.prepare_chunks_batched([0])
        caches, _, _ = att.get_batched_paged_caches(slot)
        views = states.views(slot)
        chunk = model.encode_chunk_paged(
            feats[:, step * window : (step + 1) * window],
            torch.tensor([offset], dtype=torch.int32),
            caches,
            views[CONV_STATE],
            cache_t1=cfg.prefilled_cache_frames,
            states=views,
        )
        att.commit_chunks_paged_batched([0], chunk.size(1))
        offset += chunk.size(1)
        outs.append(chunk)
    assert len(outs) > 2, "the streaming path must cross several chunks"
    return torch.cat(outs, dim=1)


def _paths_paraformer(model, device="cpu", dtype=torch.float32) -> Dict[str, torch.Tensor]:
    feats, lens = _feats("paraformer", model, device=device, dtype=dtype)
    out: Dict[str, torch.Tensor] = {}
    enc, el = model.encode_offline(feats, lens)
    out["encoder"] = enc
    # The NAR decoder over fixed acoustic embeddings: the CIF token count is a
    # floor of a sum, which a rounding-level difference could tip, and the
    # predictor itself is full precision.
    embeds = torch.randn(2, 5, enc.size(-1), generator=_gen(4)).to(device, dtype)
    token_lens = torch.tensor([5, 3], device=device, dtype=torch.int32)
    out["decoder"] = model.nar_decode(enc, el, embeds, token_lens)
    return out


def _paths_speech_llm(model, device="cpu", dtype=torch.float32) -> Dict[str, torch.Tensor]:
    feats, lens = _feats("speech_llm", model, device=device, dtype=dtype)
    out: Dict[str, torch.Tensor] = {}
    audio, _ = model.encode_offline(feats, lens)
    out["audio"] = audio
    lm = model.language_model
    inputs = torch.randn(2, 7, audio.size(-1), generator=_gen(5)).to(device, dtype)
    inputs[:, 2:5] = audio[:, :3]
    valid = torch.ones(2, 7, dtype=torch.bool, device=device)
    valid[1, :2] = False  # left padding
    feed = _tokens(2, 4, model.config.vocab_size).to(device)
    for name, kwargs in (("grown", {}), ("capacity", {"capacity": 16})):
        logits, state = lm.prefill(inputs, valid, **kwargs)
        steps = [logits]
        for t in range(feed.size(1)):
            logits, state = lm.step(feed[:, t], state)
            steps.append(logits)
        out[f"decode_{name}"] = torch.stack(steps, dim=1)
    return out


PATHS = {
    "conformer": _paths_conformer,
    "whisper": _paths_whisper,
    "nemotron": _paths_nemotron,
    "paraformer": _paths_paraformer,
    "speech_llm": _paths_speech_llm,
}


# -----------------------------------------------------------------------------
# Structure
# -----------------------------------------------------------------------------


class TestLayoutsResolve:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_path_names_a_module_of_the_declared_kind(self, arch):
        model = _model(arch)
        modules = dict(model.named_modules())
        layout = model.quantization_layout()
        spec = QuantizationSpec.for_model(model, INT8_W8A8)
        assert layout.groups and layout.chains
        for group in layout.groups:
            sq._validate_group(group, modules, spec)  # the algorithm's own checks
            producer = modules.get(group.producer) if group.producer else None
            if group.fold == "norm_affine" or group.fuse_output_quant:
                assert isinstance(producer, (LayerNorm, RMSNorm)), group
            if group.fold in ("v_to_o", "producer_rows"):
                assert isinstance(producer, QuantizableGemm), group
        for chain in layout.chains:
            for path in (chain.producer, chain.consumer):
                assert isinstance(modules.get(path), QuantizableGemm), (arch, path)
        for glob in layout.keep_full_precision:
            assert any(fnmatch.fnmatchcase(p, glob) for p in modules), (arch, glob)
        for glob in layout.batch_invariant:
            hits = [p for p in modules if fnmatch.fnmatchcase(p, glob)]
            assert hits, (arch, glob)
            for path in hits:
                assert hasattr(type(modules[path]), "batch_invariant"), (arch, path)
                assert spec.resolve(path) is None, f"{arch}: pinned {path} is quantized"

    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_quantized_layer_is_smoothed_exactly_once(self, arch):
        """A layer missing from every group is never smoothed; one in two groups
        would have its input columns scaled twice."""
        model = _model(arch)
        layout = model.quantization_layout()
        targets = set(QuantizationSpec.for_model(model, INT8_W8A8).groups[0].targets)
        seen: Dict[str, int] = {}
        for group in layout.groups:
            for path in group.consumers:
                seen[path] = seen.get(path, 0) + 1
        assert {p for p, n in seen.items() if n > 1} == set()
        assert targets - set(seen) == set(), f"{arch}: quantized but never smoothed"
        assert set(seen) - targets == set(), f"{arch}: smoothed but kept full precision"

    def test_raw_weight_readers_and_heads_stay_full_precision(self):
        llm = set(QuantizationSpec.for_model(_model("speech_llm"), INT8_W8A8).groups[0].targets)
        assert "language_model.lm_head" not in llm
        assert not any(t.startswith("multi_modal_projector") for t in llm)
        assert {"gate_proj", "up_proj", "down_proj"} <= {t.rsplit(".", 1)[-1] for t in llm}
        nem = set(QuantizationSpec.for_model(_model("nemotron"), INT8_W8A8).groups[0].targets)
        assert all(t.startswith("encoder.layers.") for t in nem)
        assert not any(t.endswith("relative_k_proj") for t in nem)
        para = set(QuantizationSpec.for_model(_model("paraformer"), INT8_W8A8).groups[0].targets)
        assert not any(t.startswith("predictor.") or t.endswith("output_layer") for t in para)
        conf = set(QuantizationSpec.for_model(_model("conformer"), INT8_W8A8).groups[0].targets)
        assert any(t.startswith("decoder.right_decoder.") for t in conf)
        assert not any(t.endswith("output_layer") or t.startswith("ctc.") for t in conf)


# -----------------------------------------------------------------------------
# SmoothQuant invariance
# -----------------------------------------------------------------------------


@pytest.fixture
def random_scales(monkeypatch):
    """Replace every computed ``s`` with log-uniform values in [0.1, 10].

    In the group's own space: V channels for a grouped-KV ``v_to_o``, the
    consumers' input channels otherwise.
    """
    g = _gen(11)

    def fake(group, act_amax, modules, opts):
        n = act_amax.numel()
        if group.fold == "v_to_o" and group.heads is not None and group.heads[0] != group.heads[1]:
            n = group.heads[1] * group.heads[2]
        return (10.0 ** (torch.rand(n, generator=g) * 2 - 1)).to(act_amax.device)

    monkeypatch.setattr(sq, "_smoothing_vector", fake)


def _emulate_runtime_smoothers(model: torch.nn.Module) -> None:
    """An unquantized layer ignores ``smooth_inv``; the quantize kernel applies it.
    Apply it to the input here, so a runtime fold is checked like the others."""
    for module in model.modules():
        if "smooth_inv" in module._buffers:
            module.register_forward_pre_hook(
                lambda m, args: (args[0] * m.smooth_inv.to(args[0].dtype),) + tuple(args[1:])
            )


def _smoothed(
    arch: str, model, layout=None, folds: str = "norm_affine,v_to_o,producer_rows,runtime"
):
    out = copy.deepcopy(model)
    spec = QuantizationSpec.for_model(out, INT8_W8A8)
    algo = build_algorithm("smoothquant", {"folds": folds})
    algo.apply(out, layout or out.quantization_layout(), spec, _calibrate(arch))
    _emulate_runtime_smoothers(out)
    return out


def _assert_paths_close(want, got, *, tol: float = 2e-4) -> None:
    assert want.keys() == got.keys()
    for name in want:
        scale = max(float(want[name].abs().max()), 1.0)
        torch.testing.assert_close(
            got[name], want[name], rtol=tol, atol=tol * scale, msg=lambda m, n=name: f"{n}: {m}"
        )


def _max_rel_diff(want, got) -> float:
    return max(
        float((got[n] - want[n]).abs().max()) / max(float(want[n].abs().max()), 1e-6) for n in want
    )


class TestSmoothQuantInvariance:
    """fp32 on CPU: smoothing without quantizing changes every output by rounding only."""

    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_forward_path(self, random_scales, arch):
        model = _model(arch)
        smoothed = _smoothed(arch, model)
        moved = [
            p
            for (p, a), (_, b) in zip(model.named_parameters(), smoothed.named_parameters())
            if not torch.equal(a, b)
        ]
        layout = model.quantization_layout()
        assert len(moved) >= len(layout.groups), "nothing (or too little) was smoothed"
        with torch.no_grad():
            want = PATHS[arch](model)
            got = PATHS[arch](smoothed)
        _assert_paths_close(want, got)

    @pytest.mark.parametrize(
        "arch, broken",
        [
            # U2++ decoder: norm1 folded, but its K projection left unscaled.
            (
                "conformer",
                SmoothGroup(
                    "decoder.left_decoder.decoders.0.norm1",
                    (
                        "decoder.left_decoder.decoders.0.self_attn.linear_q",
                        "decoder.left_decoder.decoders.0.self_attn.linear_v",
                    ),
                    "norm_affine",
                ),
            ),
            # Whisper: the encoder output folded into its norm, one decoder V
            # projection missing from the cross-attention consumers.
            (
                "whisper",
                SmoothGroup(
                    "encoder.layer_norm", ("decoder.layers.0.encoder_attn.k_proj",), "norm_affine"
                ),
            ),
            # Nemotron: norm_conv's scale folded into the *second* pointwise conv.
            (
                "nemotron",
                SmoothGroup(
                    "encoder.layers.0.norm_conv",
                    ("encoder.layers.0.conv.pointwise_conv2",),
                    "norm_affine",
                ),
            ),
            # Paraformer: the cross-attention's K rows taken for V.
            (
                "paraformer",
                SmoothGroup(
                    "decoder.decoders.0.src_attn.linear_k_v",
                    ("decoder.decoders.0.src_attn.linear_out",),
                    "v_to_o",
                    v_rows=(0, 32),
                    heads=(2, 2, 16),
                ),
            ),
            # Qwen2: down_proj's scale folded through SiLU into the gate.
            (
                "speech_llm",
                SmoothGroup(
                    "language_model.layers.0.mlp.gate_proj",
                    ("language_model.layers.0.mlp.down_proj",),
                    "producer_rows",
                ),
            ),
        ],
    )
    def test_the_oracle_catches_a_wrong_group(self, random_scales, arch, broken):
        model = _model(arch)
        smoothed = _smoothed(arch, model, layout=QuantLayout(groups=(broken,)), folds=broken.fold)
        with torch.no_grad():
            want = PATHS[arch](model)
            got = PATHS[arch](smoothed)
        assert _max_rel_diff(want, got) > 1e-2

    def test_grouped_kv_ties_the_scale_across_query_heads(self, random_scales):
        """Under GQA one V channel feeds ``h / h_kv`` output channels: o_proj's
        columns are scaled by the V scale repeated per query head."""
        model = _model("speech_llm")
        smoothed = _smoothed("speech_llm", model, folds="v_to_o")
        p = "language_model.layers.0.self_attn"
        v0 = model.get_submodule(f"{p}.v_proj").weight
        v1 = smoothed.get_submodule(f"{p}.v_proj").weight
        o0 = model.get_submodule(f"{p}.o_proj").weight
        o1 = smoothed.get_submodule(f"{p}.o_proj").weight
        s_v = (v0[:, 0] / v1[:, 0]).reshape(2, 1, 8)  # (h_kv, 1, d)
        s_o = (o1[0] / o0[0]).reshape(2, 2, 8)  # (h_kv, h / h_kv, d)
        torch.testing.assert_close(s_o, s_v.expand(2, 2, 8), rtol=1e-5, atol=0)


# -----------------------------------------------------------------------------
# Online int8
# -----------------------------------------------------------------------------


def _quantize_online(model: torch.nn.Module, scheme=INT8_W8A8) -> torch.nn.Module:
    spec = QuantizationSpec.for_model(model, scheme)
    apply_quantization(model, spec, quantize=True)
    model.quantization_spec = spec
    return model


class TestOnlineInt8:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_forward_path_runs_on_the_cpu_reference(self, arch):
        model = _model(arch)
        with torch.no_grad():
            want = PATHS[arch](model)
        _quantize_online(model)
        bind_quantization(model)
        assert all(layer.quant_method.lane is None for layer in quantized_layers(model).values())
        with torch.no_grad():
            got = PATHS[arch](model)
        for name in want:
            assert got[name].shape == want[name].shape and torch.isfinite(got[name]).all(), name
            cos = torch.nn.functional.cosine_similarity(
                got[name].flatten().double(), want[name].flatten().double(), dim=0
            )
            assert float(cos) > 0.98, f"{arch}/{name}: cosine {float(cos):.4f} to fp32"

    @cuda
    @pytest.mark.parametrize("scheme", [INT8_W8A8, FP8_W8A8], ids=lambda s: s.name)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_kernel_agrees_with_its_reference_in_the_model(self, arch, dtype, scheme):
        """Each quantized layer, on the input the real forward hands it (fused-norm
        int8 included), against its torch emulation; then the whole model against
        the all-torch path."""
        if scheme is FP8_W8A8 and torch.cuda.get_device_capability() < (8, 9):
            pytest.skip("fp8 tensor cores need sm89+")
        model = _quantize_online(_model(arch), scheme).to(device="cuda", dtype=dtype)
        bind_quantization(model)
        layers = quantized_layers(model)
        missing = [p for p, layer in layers.items() if layer.quant_method.lane is None]
        assert not missing, f"no kernel lane for {missing}"
        rel = 2.0**-9 if dtype == torch.float16 else 2.0**-6
        checked: Dict[str, int] = {}

        def check(path):
            def hook(module, args, output):
                ref = module.quant_method.apply_reference(module, args[0])
                if isinstance(output, QuantizedActivation):
                    output, ref = output.data.float(), ref.data.float()
                got, ref = output.float(), ref.float()
                bad = (got - ref).abs() > ref.abs() * rel + 1e-3
                assert not bool(bad.any()), f"{path}: {int(bad.sum())} elements off"
                checked[path] = checked.get(path, 0) + 1

            return hook

        handles = [layer.register_forward_hook(check(p)) for p, layer in layers.items()]
        try:
            with torch.no_grad():
                got = PATHS[arch](model, device="cuda", dtype=dtype)
        finally:
            for h in handles:
                h.remove()
        assert set(checked) == set(layers), f"never reached: {sorted(set(layers) - set(checked))}"
        with torch.no_grad(), layers_backend_override("torch"):
            ref = PATHS[arch](model, device="cuda", dtype=dtype)
        for name in got:
            assert torch.isfinite(got[name]).all(), name
            cos = torch.nn.functional.cosine_similarity(
                got[name].flatten().double(), ref[name].flatten().double(), dim=0
            )
            assert float(cos) > 0.99, f"{arch}/{name}: cosine {float(cos):.4f}"


# -----------------------------------------------------------------------------
# Calibrated recipes: SmoothQuant in fp8, and the fused links
# -----------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("arch", ARCHS)
def test_calibration_forward_runs_the_fp32_master_on_cuda(arch):
    """``oasr quantize`` calibrates the fp32 master *on the device*: every layer a
    model's ``calibration_forward`` reaches must route fp32 to torch.  The CPU
    invariance tests cannot see a kernel that accepts only fp16 (the CTC head's
    fused ``gemm_log_softmax`` did, and U2++ calibration then raised)."""
    model = _model(arch).cuda()
    _calibrate(arch)(model)


def _calibrated(arch: str, recipe: str) -> torch.nn.Module:
    from oasr.quantization.export import quantize_model
    from oasr.quantization.recipes import get_recipe

    model = _model(arch)
    quantize_model(model, get_recipe(recipe), _calibrate(arch))
    return model


def _disarm(model: torch.nn.Module) -> None:
    """Every link off: each quantized layer quantizes its own input again."""
    for layer in quantized_layers(model).values():
        layer.quant_method.emit = None
    for module in model.modules():
        if "quantize_output" in vars(module):
            del module.quantize_output


class TestCalibratedRecipes:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_smoothquant_fp8_runs_every_path_on_the_cpu_reference(self, arch):
        model = _model(arch)
        with torch.no_grad():
            want = PATHS[arch](model)
        model = _calibrated(arch, "smoothquant_fp8_w8a8")
        layers = quantized_layers(model)
        assert {layer.weight.dtype for layer in layers.values()} == {torch.float8_e4m3fn}
        bind_quantization(model)
        with torch.no_grad():
            got = PATHS[arch](model)
        for name in want:
            cos = torch.nn.functional.cosine_similarity(
                got[name].flatten().double(), want[name].flatten().double(), dim=0
            )
            assert float(cos) > 0.95, f"{arch}/{name}: cosine {float(cos):.4f} to fp32"

    @cuda
    @pytest.mark.parametrize("arch", ARCHS)
    def test_armed_links_change_no_bit_on_any_path(self, arch):
        """``smoothquant_w8a8_fused``: every declared fusable norm and GEMM chain
        armed, on every forward path, against the same model with them disarmed.
        A norm declared fusable whose output something else reads raw would hand
        that reader a ``QuantizedActivation`` and fail here, not just differ."""
        model = _calibrated(arch, "smoothquant_w8a8_fused").to(device="cuda", dtype=torch.float16)
        bind_quantization(model)
        norms = [m for m in model.modules() if vars(m).get("quantize_output")]
        producers = [p for p, q in quantized_layers(model).items() if q.quant_method.emit]
        assert producers, f"{arch}: no GEMM chain armed"
        if arch != "speech_llm":  # the LM's RMSNorm has no int8-emitting kernel
            assert norms, f"{arch}: no fused norm armed"
        with torch.no_grad():
            armed = PATHS[arch](model, device="cuda", dtype=torch.float16)
            _disarm(model)
            plain = PATHS[arch](model, device="cuda", dtype=torch.float16)
        for name in armed:
            assert torch.equal(armed[name], plain[name]), f"{arch}/{name}"
