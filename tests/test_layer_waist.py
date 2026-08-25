#!/usr/bin/env python3
"""Conformance test for the ``oasr.layers`` narrow waist (architecture review H1).

Every registered architecture is built tiny on CPU and its module tree walked:
no bare ``nn.Linear`` / ``nn.LayerNorm`` / ``nn.Embedding`` / ``nn.*Norm`` may
appear.  That is the whole point of the test — the waist was *already there*
and unused, and nothing stopped a new model from reaching past it.  Before this
existed, four of six architectures were plain PyTorch: kernels, CUDA-graph
capture and any future quantization applied to one and a half of them.

Two properties make it a ratchet rather than a snapshot:

* the tiny-config table is keyed off :func:`list_models`, so registering an
  architecture without adding one **fails** here instead of being skipped;
* exemptions are named individually with a reason (:data:`ALLOWED_BARE`), so a
  gap is a line of code somebody has to write, not an omission.

Also checks the waist's own contract: every layer module runs on CPU/fp32, and
the kernel and torch paths agree on CUDA.  The equivalence is what makes the
CPU parity oracles meaningful evidence about the GPU serving path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from oasr.layers import layers_backend_override
from oasr.models.registry import get_model_entry, list_models

# ---------------------------------------------------------------------------
# What the waist replaces
# ---------------------------------------------------------------------------

#: torch modules an architecture must not instantiate directly.  Each has an
#: ``oasr.layers`` counterpart (or, for parameter-free padding, an operation
#: already expressible by a waist layer), so the fix never changes a checkpoint.
BANNED = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.AvgPool1d,
    nn.ConstantPad1d,
    nn.LayerNorm,
    nn.Embedding,
    nn.RMSNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.MultiheadAttention,
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.LSTM,
    nn.RNN,
)

#: Deliberate exemptions, as ``(architecture, dotted module path) -> reason``.
#: Dense, depthwise, and pointwise ``nn.Conv1d`` plus ``nn.AvgPool1d`` now have
#: BTC-native waist counterparts, so the torch classes are banned alongside the
#: other bypasses. ``nn.ConstantPad1d`` is banned with pooling so model code
#: cannot reconstruct an explicit pad + pool chain around the kernel.
ALLOWED_BARE: dict = {}


def _tiny_configs():
    """One small, CPU-buildable config per registered architecture.

    Deliberately not derived from the config defaults: those are the *release*
    recipes (Paraformer's is 220M parameters), and a conformance test that
    allocates a real model per architecture stops being run.
    """
    from oasr.models.conformer.config import ConformerEncoderConfig, ConformerModelConfig
    from oasr.models.decoders.transformer_decoder import TransformerDecoderConfig
    from oasr.models.nemotron.config import NemotronEncoderConfig, NemotronModelConfig
    from oasr.models.paraformer.config import ParaformerModelConfig
    from oasr.models.speech_llm.config import SpeechLlmModelConfig
    from oasr.models.transducer.config import TransducerModelConfig
    from oasr.models.whisper.config import WhisperModelConfig
    from oasr.models.zipformer.config import ZipformerEncoderConfig, ZipformerModelConfig

    zip_enc = {
        "feature_dim": 80,
        "downsampling_factor": (1, 2),
        "encoder_dim": (64, 96),
        "num_encoder_layers": (1, 1),
        "query_head_dim": (8,),
        "pos_head_dim": (4,),
        "value_head_dim": (6,),
        "num_heads": (4, 4),
        "feedforward_dim": (64, 96),
        "cnn_module_kernel": (15, 15),
        "pos_dim": 16,
        "causal": False,
    }
    return {
        # Conformer carries the optional U2++ AED branch, so configure it here:
        # ``models/decoders/transformer_decoder.py`` is only reachable through
        # a model, and it is one of the migrated files.
        "conformer": ConformerModelConfig(
            vocab_size=32,
            encoder=ConformerEncoderConfig(
                input_size=80, output_size=32, attention_heads=2, linear_units=64, num_blocks=2
            ),
            decoder=TransformerDecoderConfig(
                vocab_size=32,
                encoder_output_size=32,
                attention_heads=2,
                linear_units=64,
                num_blocks=1,
                r_num_blocks=1,
            ),
        ),
        "zipformer": ZipformerModelConfig(vocab_size=32, encoder=ZipformerEncoderConfig(**zip_enc)),
        "whisper": WhisperModelConfig(
            vocab_size=64,
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            num_mel_bins=80,
            max_source_positions=50,
            max_target_positions=32,
        ),
        "paraformer": ParaformerModelConfig(
            vocab_size=64,
            input_size=80,
            encoder_output_size=32,
            encoder_attention_heads=2,
            encoder_linear_units=64,
            encoder_num_blocks=2,
            decoder_attention_heads=2,
            decoder_linear_units=64,
            decoder_num_blocks=1,
            decoder_att_layer_num=1,
            predictor_idim=32,
        ),
        "speech_llm": SpeechLlmModelConfig(
            vocab_size=64,
            audio_d_model=32,
            audio_encoder_layers=1,
            audio_encoder_attention_heads=2,
            audio_encoder_ffn_dim=64,
            audio_num_mel_bins=128,
            audio_max_source_positions=25,
            text_hidden_size=32,
            text_num_hidden_layers=1,
            text_num_attention_heads=2,
            text_num_key_value_heads=2,
            text_intermediate_size=64,
        ),
        "transducer": TransducerModelConfig(
            encoder_type="zipformer",
            encoder=ZipformerEncoderConfig(**zip_enc),
            vocab_size=30,
            decoder_dim=24,
            joiner_dim=24,
            context_size=2,
            blank_id=0,
        ),
        # Keep the 8x subsampling stack at three real stages so its depthwise
        # and pointwise Conv2d kernel paths remain represented in the tiny model.
        "nemotron": NemotronModelConfig(
            vocab_size=32,
            blank_token_id=31,
            decoder_hidden_size=24,
            num_decoder_layers=1,
            num_prompts=8,
            prompt_intermediate_size=48,
            encoder=NemotronEncoderConfig(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                intermediate_size=64,
                num_mel_bins=32,
                subsampling_conv_channels=16,
                sliding_window=13,
            ),
        ),
    }


def _build(arch):
    cfg = _tiny_configs()[arch]
    return get_model_entry(arch).model_cls.from_config(cfg)


def test_every_architecture_has_a_tiny_config():
    """A new architecture must be added here, not silently skipped.

    This is the assertion that turns the test into a ratchet: without it, the
    parametrized test below would just cover fewer models over time.
    """
    missing = sorted(set(list_models()) - set(_tiny_configs()))
    assert not missing, (
        f"architectures with no tiny config in tests/test_layer_waist.py: {missing}; "
        "add one so the waist conformance check actually covers them"
    )


@pytest.mark.parametrize("arch", list_models())
def test_architecture_uses_the_layer_waist(arch):
    """No bare torch layer anywhere in a registered architecture's tree."""
    model = _build(arch)
    offenders = [
        f"{name or '<root>'}: {type(mod).__name__}"
        for name, mod in model.named_modules()
        if isinstance(mod, BANNED) and ALLOWED_BARE.get((arch, name)) is None
    ]
    assert not offenders, (
        f"{arch} reaches past oasr.layers:\n  " + "\n  ".join(offenders) + "\n"
        "Every one of these has an oasr.layers counterpart with the same parameter "
        "layout — swapping it changes no checkpoint key."
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def test_models_do_not_call_bare_torch_activations():
    """Standalone activations belong to the waist, not model code."""
    models_dir = Path(__file__).resolve().parents[1] / "oasr" / "models"
    banned = {
        f"{prefix}.{name}"
        for prefix in ("F", "torch", "torch.functional", "torch.nn.functional")
        for name in ("gelu", "relu", "sigmoid", "tanh")
    }
    offenders = []
    for path in models_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted_name(node.func) in banned:
                offenders.append(f"{path.relative_to(models_dir)}:{node.lineno}")
    assert not offenders, (
        "model code reaches past oasr.layers for standalone activation:\n  "
        + "\n  ".join(offenders)
        + "\nUse Gelu/Relu/Sigmoid/Tanh for standalone activation or a fused waist "
        "layer when the activation immediately follows its projection."
    )


#: Deliberate ``torch.matmul`` / ``torch.bmm`` sites in model code, as
#: ``(path relative to oasr/models, symbol) -> reason``.  Keyed by file rather
#: than line so the ratchet survives an edit above it, and by an explicit count
#: so a *second* bypass in the same file still fails.
ALLOWED_TORCH_MATMUL = {
    ("whisper/model.py", "torch.matmul"): (
        1,
        "the word-timing cross-attention probe forms its scores in fp32 on "
        "purpose; fp32 is outside the OASR BMM kernel contract (fp16/bf16)",
    ),
}


def test_models_do_not_bypass_bmm_with_torch_matmul():
    """KG5 ratchet: batched matrix products belong to ``oasr.bmm``.

    Zipformer's five attention products per layer were the reason this rule
    could not exist before: its head dims (query 32, pos 4, value 12) and its
    always-odd relative-position extent had no kernel, so the model file called
    ``torch.matmul`` and no counter anywhere could see it.  The general BMM lane
    closed that, and this keeps it closed.
    """
    models_dir = Path(__file__).resolve().parents[1] / "oasr" / "models"
    banned = {"torch.matmul", "torch.bmm"}
    found: dict = {}
    for path in models_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted_name(node.func) in banned:
                relative = path.relative_to(models_dir).as_posix()
                key = (relative, _dotted_name(node.func))
                found.setdefault(key, []).append(node.lineno)

    offenders = []
    for key, lines in sorted(found.items()):
        allowed = ALLOWED_TORCH_MATMUL.get(key)
        budget = allowed[0] if allowed else 0
        if len(lines) > budget:
            offenders.append(
                f"{key[0]}:{','.join(str(n) for n in lines)} {key[1]} "
                f"({len(lines)} call(s), {budget} allowed)"
            )
    assert not offenders, (
        "model code bypasses oasr.bmm:\n  "
        + "\n  ".join(offenders)
        + "\nUse oasr.bmm — it takes 3-D/4-D operands with broadcasting batch axes, "
        "either memory layout for B, and arbitrary N/K. Add an entry to "
        "ALLOWED_TORCH_MATMUL only for a dtype the kernel does not serve."
    )


#: Deliberate ``masked_fill`` sites in model code, as ``(path relative to
#: oasr/models, count) -> reason``.  A ``masked_fill`` on an *attention score*
#: tensor belongs to ``oasr.masked_softmax``; one on an activation is a
#: different (still open) gap and is allowed here by name.
ALLOWED_MASKED_FILL = {
    "conformer/model.py": (
        2,
        "zeroing the hidden state outside the utterance, not a score floor — "
        "KG12 (broadcast gate / mask-multiply), no kernel yet",
    ),
    "zipformer/encoder.py": (
        2,
        "the two NonlinAttention activation masks — KG12, not scores",
    ),
    "nemotron/encoder.py": (
        4,
        "two silence gates on activations (KG12) and two floors applied to the "
        "rel-pos bias handed to oasr.fmha as attn_bias — that kernel never "
        "materializes scores, so there is no softmax to fuse into; removing "
        "the bias pass is KG8 (in-kernel Transformer-XL rel-pos)",
    ),
    "decoders/transformer_decoder.py": (
        2,
        "token-id padding on the decoder input, not a float score tensor",
    ),
}


def test_models_do_not_hand_roll_a_masked_softmax():
    """KG6 ratchet: an attention score's bias + mask + softmax is one kernel.

    Zipformer walked its ``(H, B, T, T)`` scores eight times per layer across
    three kernels — add the shifted relative-position bias, floor the key
    padding, softmax — where ``oasr.masked_softmax`` reads the bias and both
    masks through their own strides in one.  Each entry in
    :data:`ALLOWED_MASKED_FILL` is a site that is *not* a score floor; a new
    ``masked_fill`` in any of these files fails until it is classified.
    """
    models_dir = Path(__file__).resolve().parents[1] / "oasr" / "models"
    found: dict = {}
    for path in models_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "masked_fill":
                    relative = path.relative_to(models_dir).as_posix()
                    found.setdefault(relative, []).append(node.lineno)

    offenders = []
    for relative, lines in sorted(found.items()):
        allowed = ALLOWED_MASKED_FILL.get(relative)
        budget = allowed[0] if allowed else 0
        if len(lines) > budget:
            offenders.append(
                f"{relative}:{','.join(str(n) for n in lines)} "
                f"({len(lines)} masked_fill call(s), {budget} allowed)"
            )
    assert not offenders, (
        "model code hand-rolls a masked softmax:\n  "
        + "\n  ".join(offenders)
        + "\nUse oasr.masked_softmax — it takes an additive bias and two boolean "
        "masks, each broadcast against the scores through its own strides, so a "
        "shifted or step-sliced view needs no copy. Add an entry to "
        "ALLOWED_MASKED_FILL only for a mask that is not an attention-score floor."
    )


def test_zipformer_attention_weights_use_the_fused_masked_softmax():
    """The offline and streaming score paths both have to reach the kernel."""
    encoder = Path(__file__).resolve().parents[1] / "oasr" / "models" / "zipformer" / "encoder.py"
    tree = ast.parse(encoder.read_text(), filename=str(encoder))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == "oasr.masked_softmax"
    ]
    assert len(calls) == 2, (
        f"expected oasr.masked_softmax on both Zipformer attention-weight paths "
        f"(forward and streaming_forward), found {len(calls)}"
    )


def test_eligible_residual_norm_paths_use_fused_waist():
    """Keep KG14's model wiring from silently regressing to separate adds."""
    models_dir = Path(__file__).resolve().parents[1] / "oasr" / "models"
    expected = {
        "whisper/model.py": {"forward_add": 2, "forward_add_residual": 5},
        "speech_llm/audio_tower.py": {"forward_add_residual": 2},
        "speech_llm/llm.py": {"forward_add_residual": 2},
        "paraformer/encoder.py": {"forward_add": 2, "forward_add_residual": 3},
        "paraformer/decoder.py": {"forward_add": 1, "forward_add_residual": 2},
        "nemotron/encoder.py": {"forward_add": 2, "forward_add_residual": 6},
    }
    missing = []
    for relative, minimums in expected.items():
        tree = ast.parse((models_dir / relative).read_text(), filename=relative)
        counts = dict.fromkeys(minimums, 0)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in counts:
                    counts[node.func.attr] += 1
        for name, minimum in minimums.items():
            if counts[name] < minimum:
                missing.append(f"{relative}: {name}={counts[name]} (expected >= {minimum})")
    assert not missing, (
        "eligible residual + norm sites no longer use the fused waist:\n  "
        + "\n  ".join(missing)
        + "\nZipformer is intentionally absent: its learned lerp bypass + BiasNorm belongs to KG12."
    )


@pytest.mark.parametrize(
    "oasr_cls,torch_cls,args",
    [
        ("Linear", nn.Linear, (8, 16)),
        ("ColumnParallelLinear", nn.Linear, (8, 16)),
        ("RowParallelLinear", nn.Linear, (8, 16)),
        ("LinearActivation", nn.Linear, (8, 16)),
        ("LayerNorm", nn.LayerNorm, (8,)),
        ("Embedding", nn.Embedding, (10, 8)),
        ("LSTM", nn.LSTM, (8, 16)),
        ("RNN", nn.RNN, (8, 16)),
    ],
)
def test_layer_parameter_layout_matches_torch(oasr_cls, torch_cls, args):
    """Same parameter names and shapes as the ``nn.*`` module each one replaces.

    This is what makes a migration a one-line import change: a checkpoint loads
    by key *and* shape, so a layer that reorganized its weights would break
    loading without breaking any forward — and no parity test would catch it,
    because parity tests copy state dicts between two already-agreeing trees.
    """
    import oasr.layers as L

    ours = {k: tuple(v.shape) for k, v in getattr(L, oasr_cls)(*args).named_parameters()}
    theirs = {k: tuple(v.shape) for k, v in torch_cls(*args).named_parameters()}
    assert ours == theirs


@pytest.mark.parametrize("arch", list_models())
def test_bias_free_layers_register_bias_as_none(arch):
    """``bias=False`` must leave a registered ``None``, not a stray attribute:
    ``load_state_dict`` reports an unexpected key either way, but only the
    registered form keeps ``named_parameters()`` honest."""
    model = _build(arch)
    for name, mod in model.named_modules():
        if hasattr(mod, "bias") and mod.bias is None:
            assert "bias" in mod._parameters or "bias" in mod._buffers, (
                f"{arch}.{name} has bias=None as a plain attribute; "
                "use register_parameter('bias', None)"
            )


# ---------------------------------------------------------------------------
# The waist's own contract
# ---------------------------------------------------------------------------


class TestLayersRunOnCpu:
    """Every layer module works on CPU/fp32 — the property the parity oracles
    and the whole CPU test suite depend on."""

    def test_linear(self):
        from oasr.layers import ColumnParallelLinear, Linear, RowParallelLinear

        x = torch.randn(2, 3, 8)
        for cls in (Linear, ColumnParallelLinear, RowParallelLinear):
            m = cls(8, 16)
            out = m(x)
            torch.testing.assert_close(out, torch.nn.functional.linear(x, m.weight, m.bias))

    def test_linear_unaligned_shape(self):
        """N or K not divisible by 8 is a torch-path shape, not an error.
        ``oasr.gemm`` raises on it (CUTLASS alignment-8 iterators), which is
        exactly why the decision lives in the layer: Paraformer's 8404-token
        vocabulary and icefall's 500 both land here."""
        from oasr.layers import Linear

        m = Linear(8, 500)
        assert m(torch.randn(2, 8)).shape == (2, 500)

    def test_dense_conv1d(self):
        from oasr.layers import Conv1d

        m = Conv1d(8, 16, 3, padding=2, stride=2, dilation=2)
        x = torch.randn(2, 11, 8)
        got = m(x)
        ref = torch.nn.functional.conv1d(
            x.transpose(1, 2),
            m.weight.permute(0, 2, 1),
            m.bias,
            stride=2,
            padding=2,
            dilation=2,
        ).transpose(1, 2)
        torch.testing.assert_close(got, ref)

    def test_dense_conv1d_loads_torch_weight_layout(self):
        from oasr.layers import Conv1d

        source = nn.Conv1d(8, 16, 3)
        target = Conv1d(8, 16, 3)
        target.load_state_dict(source.state_dict())
        assert target.weight.shape == (16, 3, 8)
        torch.testing.assert_close(target.weight, source.weight.permute(0, 2, 1))

    def test_dense_conv1d_activation_refuses_erf_gelu(self):
        from oasr.layers import Conv1dActivation

        with pytest.raises(ValueError, match="not fusable"):
            Conv1dActivation(8, 16, 3, activation_type="gelu")
        assert Conv1dActivation(8, 16, 3, activation_type="gelu_tanh")(
            torch.randn(2, 7, 8)
        ).shape == (2, 5, 16)

    def test_depthwise_conv1d_asymmetric_masked_residual(self):
        from oasr.layers import DepthwiseConv1d

        torch.manual_seed(0)
        m = DepthwiseConv1d(8, 5, padding=(3, 1), bias=False)
        x = torch.randn(2, 9, 8)
        mask = (torch.rand(2, 9, 1) > 0.3).to(x.dtype)

        got = m(x, mask=mask, add_input=True)
        masked = x * mask
        conv = torch.nn.functional.conv1d(
            torch.nn.functional.pad(masked.transpose(1, 2), (3, 1)),
            m.weight.permute(2, 1, 0),
            groups=8,
        ).transpose(1, 2)
        torch.testing.assert_close(got, (conv + masked) * mask)

    @pytest.mark.parametrize(
        "in_channels,out_channels,kernel,groups",
        [(8, 8, 3, 8), (8, 16, 3, 4), (8, 16, 1, 1)],
    )
    def test_grouped_and_pointwise_conv2d(self, in_channels, out_channels, kernel, groups):
        from oasr.layers import Conv2d

        padding = kernel // 2
        m = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            padding=padding,
            groups=groups,
        )
        x = torch.randn(2, 11, 7, in_channels)
        got = m(x)
        ref = torch.nn.functional.conv2d(
            x.permute(0, 3, 1, 2),
            m.weight.permute(0, 3, 1, 2),
            m.bias,
            padding=padding,
            groups=groups,
        ).permute(0, 2, 3, 1)
        torch.testing.assert_close(got, ref)

    def test_avg_pool1d(self):
        from oasr.layers import AvgPool1d

        x = torch.randn(2, 11, 8)
        for kwargs in (
            {"kernel_size": 2, "stride": 2},
            {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": False,
            },
        ):
            got = AvgPool1d(**kwargs)(x)
            ref = torch.nn.functional.avg_pool1d(x.transpose(1, 2), **kwargs).transpose(1, 2)
            torch.testing.assert_close(got, ref)

    def test_norms(self):
        from oasr.layers import BiasNorm, LayerNorm, RMSNorm

        x = torch.randn(2, 3, 8)
        torch.testing.assert_close(
            LayerNorm(8)(x), torch.nn.functional.layer_norm(x, (8,), torch.ones(8), torch.zeros(8))
        )
        assert RMSNorm(8, bias=False)(x).shape == x.shape
        assert BiasNorm(8)(x).shape == x.shape

    def test_mlp_blocks(self):
        from oasr.layers import FeedForward, GatedMLP

        x = torch.randn(2, 3, 8)
        ff = FeedForward(8, 16, activation="relu", names=("w_1", "w_2"))
        assert set(dict(ff.named_parameters())) == {
            "w_1.weight",
            "w_1.bias",
            "w_2.weight",
            "w_2.bias",
        }
        assert ff(x).shape == x.shape

        sanm = FeedForward(
            8, 16, activation="relu", names=("w_1", "w_2"), out_bias=False, inner_norm_eps=1e-12
        )
        assert "norm.weight" in dict(sanm.named_parameters())
        assert "w_2.bias" not in dict(sanm.named_parameters())

        mlp = GatedMLP(8, 16)
        assert set(dict(mlp.named_parameters())) == {
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        }
        assert mlp(x).shape == x.shape

    def test_feedforward_rejects_unknown_activation(self):
        from oasr.layers import FeedForward

        with pytest.raises(ValueError, match="not known"):
            FeedForward(8, 16, activation="mish")

    def test_standalone_and_linear_activation(self):
        from oasr.layers import Gelu, LinearActivation, Relu, Sigmoid, Tanh

        x = torch.randn(2, 8)
        torch.testing.assert_close(Gelu()(x), torch.nn.functional.gelu(x))
        torch.testing.assert_close(Relu()(x), torch.relu(x))
        torch.testing.assert_close(Sigmoid()(x), torch.sigmoid(x))
        torch.testing.assert_close(Tanh()(x), torch.tanh(x))
        assert LinearActivation(8, 16, activation_type="gelu")(x).shape == (2, 16)
        assert LinearActivation(8, 16, activation_type="gelu_tanh")(x).shape == (2, 16)

    def test_attention_mask_forms(self):
        from oasr.layers import Attention

        a = Attention(2, 4)
        q = a.split_heads(torch.randn(2, 5, 8))
        k = a.split_heads(torch.randn(2, 7, 8))
        v = a.split_heads(torch.randn(2, 7, 8))
        assert a(q, k, v).shape == (2, 2, 5, 4)
        assert a(q, k, v, kv_lens=torch.tensor([7, 3])).shape == (2, 2, 5, 4)
        assert a(q, k, v, attn_mask=torch.ones(2, 1, 5, 7, dtype=torch.bool)).shape == (2, 2, 5, 4)
        assert a(q, q, q, is_causal=True).shape == (2, 2, 5, 4)

    def test_attention_kv_lens_equals_explicit_mask(self):
        """The two spellings of right padding must agree, since models now use
        whichever reaches the kernel."""
        from oasr.layers import Attention

        torch.manual_seed(0)
        a = Attention(2, 4)
        q, k, v = (a.split_heads(torch.randn(2, 5, 8)) for _ in range(3))
        lens = torch.tensor([5, 2])
        mask = (torch.arange(5).unsqueeze(0) < lens.unsqueeze(1)).view(2, 1, 1, 5)
        torch.testing.assert_close(a(q, k, v, kv_lens=lens), a(q, k, v, attn_mask=mask))

    def test_attention_gqa(self):
        from oasr.layers import Attention

        a = Attention(4, 4, num_kv_heads=2)
        q = a.split_heads(torch.randn(2, 3, 16))
        k = a.split_kv_heads(torch.randn(2, 3, 8))
        v = a.split_kv_heads(torch.randn(2, 3, 8))
        assert a(q, k, v).shape == (2, 4, 3, 4)

    def test_rotary_matches_hf_formulation(self):
        from oasr.layers import NeoxRotaryEmbedding, apply_rotary_pos_emb

        rope = NeoxRotaryEmbedding(8, theta=10000.0)
        # Per-row positions: the case the complex freqs_cis API cannot express.
        pos = torch.tensor([[0, 1, 2], [0, 0, 1]])
        cos, sin = rope(pos)
        assert cos.shape == (2, 3, 8) and cos.dtype == torch.float32
        q = torch.randn(2, 2, 3, 8)
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
        # Position 0 is the identity rotation.
        torch.testing.assert_close(q_rot[:, :, 0], q[:, :, 0])


class TestBackendSelection:
    """OASR is the backend; torch is an optional backend you *select*.

    The distinction is the point of this class.  A framework that silently
    slides into torch when a kernel is missing can never tell you that the
    kernel is missing.
    """

    def test_oasr_is_the_default(self):
        from oasr.layers import layers_backend

        assert layers_backend() == "oasr"

    def test_torch_is_selectable(self):
        from oasr.layers import Linear, layers_backend

        with layers_backend_override("torch"):
            assert layers_backend() == "torch"
            assert Linear(8, 16)(torch.randn(2, 8)).shape == (2, 16)
        assert layers_backend() == "oasr"

    def test_there_is_no_auto_mode(self):
        """``auto`` was the old name and the old idea; both are gone."""
        from oasr.layers import set_layers_backend

        for bad in ("auto", "fastest"):
            with pytest.raises(ValueError):
                set_layers_backend(bad)

    def test_cpu_is_out_of_scope_not_a_gap(self):
        """The framework targets GPU inference, so a CPU tensor is served by
        torch and reported — but it is not kernel debt, and it must not be
        counted as one."""
        from oasr.layers import Linear
        from oasr.layers._backend import gap_hits, reset_backend_stats

        reset_backend_stats()
        Linear(8, 16)(torch.randn(2, 8))
        assert gap_hits() == {}


class TestKernelGapRegistry:
    """Missing kernels are declared debt, not invisible fallbacks."""

    #: Removing a pinned gap passes; adding one requires a deliberate update so
    #: new coverage debt is visible in review.
    #:
    #: The paged loader requires a head dimension aligned to its 32-element MMA
    #: stride and a page size that divides every K tile.  Production configs meet
    #: both constraints; small test configs may not.
    PINNED = {"fmha-head-dim", "fmha-paged-config"}

    def test_declared_gap_set_only_shrinks(self):
        from oasr.layers._backend import KERNEL_GAPS

        new = set(KERNEL_GAPS) - self.PINNED
        assert not new, (
            f"new kernel gap(s) declared: {sorted(new)}. A gap is coverage debt — "
            f"if it is genuinely unavoidable, add it to PINNED in this test so the "
            f"declaration is visible in review; otherwise fix it at the kernel or "
            f"model layer."
        )

    def test_every_declared_gap_says_where_to_fix_it(self):
        from oasr.layers._backend import KERNEL_GAPS

        for gid, gap in KERNEL_GAPS.items():
            assert gap.id == gid
            assert gap.what, f"{gid} does not say what is missing"
            assert gap.fix.startswith(("kernel:", "model:")), (
                f"{gid}.fix must name the layer that has to fix it "
                f"('kernel:' or 'model:'), got {gap.fix[:40]!r}"
            )

    def test_undeclared_refusal_raises(self):
        """A kernel that cannot run a shape nobody wrote down is a bug or an
        unfinished kernel.  It must stop the run, not cost throughput forever."""
        from oasr.layers._backend import take_gap

        with pytest.raises(RuntimeError, match="no declared gap"):
            take_gap("gemm-unaligned", "Linear(7 -> 13)")

    def test_declared_gap_is_counted(self):
        from oasr.layers._backend import gap_hits, reset_backend_stats, take_gap

        reset_backend_stats()
        assert take_gap("fmha-head-dim", "head_dim=128") is False
        assert gap_hits() == {"fmha-head-dim": 1}

    def test_report_separates_debt_from_choice(self):
        from oasr.layers._backend import (
            format_gap_report,
            reset_backend_stats,
            take_gap,
            take_policy,
        )

        reset_backend_stats()
        take_gap("fmha-head-dim", "head_dim=128")
        take_policy("gemm-below-work-floor")
        report = format_gap_report()
        assert "kernel gaps taken" in report and "fmha-head-dim" in report
        assert "performance grounds" in report
        reset_backend_stats()

    @pytest.mark.parametrize("arch", list_models())
    def test_no_architecture_needs_an_unaligned_gemm(self, arch):
        """Every output projection is allocated at a width the kernels can
        address.  An unpadded vocabulary head is fixable at the model layer
        (``oasr.models.base.align_out_features``), so it is deliberately *not*
        a declared gap — this asserts none crept back."""
        from oasr.layers.linear import Linear as OasrLinear

        offenders = [
            f"{name}: {mod.in_features} -> {mod.out_features}"
            for name, mod in _build(arch).named_modules()
            if isinstance(mod, OasrLinear) and (mod.in_features % 8 or mod.out_features % 8)
        ]
        assert not offenders, (
            f"{arch} has projections the GEMM kernels cannot address:\n  "
            + "\n  ".join(offenders)
            + "\nPad them with align_out_features() and widen the checkpoint in "
            "load_weights (see oasr/models/base.py::pad_output_projection)."
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel path needs CUDA")
class TestKernelAndTorchPathsAgree:
    """The two paths of each layer must compute the same thing.

    Without this the CPU suite proves nothing about the served model: every
    parity oracle in the repo runs the torch path.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_linear(self, dtype):
        from oasr.layers import Linear

        m = Linear(64, 128).cuda().to(dtype)
        x = torch.randn(4, 16, 64, device="cuda", dtype=dtype)
        got = m(x)
        with layers_backend_override("torch"):
            ref = m(x)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("activation", ["relu", "swish", "gelu", "gelu_tanh"])
    def test_linear_activation(self, activation):
        from oasr.layers import LinearActivation

        m = LinearActivation(64, 128, activation_type=activation).cuda().half()
        x = torch.randn(4, 16, 64, device="cuda", dtype=torch.float16)
        got = m(x)
        with layers_backend_override("torch"):
            ref = m(x)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gelu(self, dtype):
        from oasr.layers import Gelu

        x = torch.linspace(-5.0, 5.0, 4096, device="cuda", dtype=dtype)
        got = Gelu()(x)
        with layers_backend_override("torch"):
            ref = Gelu()(x)
        torch.testing.assert_close(got, ref, rtol=0, atol=2e-3)

    @pytest.mark.parametrize("module_name", ["Relu", "Sigmoid", "Tanh"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_standalone_activation(self, module_name, dtype):
        import oasr.layers as layers

        module = getattr(layers, module_name)()
        x = torch.linspace(-8.0, 8.0, 4096, device="cuda", dtype=dtype)
        got = module(x)
        with layers_backend_override("torch"):
            ref = module(x)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=2e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_dense_conv1d(self, dtype, stride):
        from oasr.layers import Conv1d

        m = Conv1d(80, 384, 3, padding=1, stride=stride).cuda().to(dtype)
        x = torch.randn(2, 127, 80, device="cuda", dtype=dtype)
        got = m(x)
        with layers_backend_override("torch"):
            ref = m(x)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("activation", ["relu", "swish", "gelu_tanh"])
    def test_dense_conv1d_activation(self, activation):
        from oasr.layers import Conv1dActivation

        m = Conv1dActivation(64, 128, 3, padding=1, activation_type=activation).cuda().half()
        x = torch.randn(2, 63, 64, device="cuda", dtype=torch.float16)
        got = m(x)
        with layers_backend_override("torch"):
            ref = m(x)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("mask_dtype", [torch.bool, None])
    def test_depthwise_conv1d_asymmetric_masked_residual(self, dtype, mask_dtype):
        from oasr.layers import DepthwiseConv1d

        m = DepthwiseConv1d(128, 11, padding=(7, 3), bias=False).cuda().to(dtype)
        x = torch.randn(2, 63, 128, device="cuda", dtype=dtype)
        bool_mask = torch.rand(2, 63, 1, device="cuda") > 0.2
        mask = bool_mask if mask_dtype is torch.bool else bool_mask.to(dtype)

        got = m(x, mask=mask, add_input=True)
        with layers_backend_override("torch"):
            ref = m(x, mask=mask, add_input=True)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_depthwise_conv1d_rejects_a_row_strided_input(self, dtype):
        """The kernel path asserts contiguity instead of quietly copying.

        A last-dim slice of a fused QKV projection — Paraformer's ``v``, from
        ``torch.split(linear_q_k_v(x), C, dim=-1)`` — is last-dim contiguous
        with a row stride of ``3 * C``, which the launcher rejects.  The layer
        does not paper over it: the caller owns the copy, so the cost sits where
        it is caused (see ``paraformer/modules.py::SanmSelfAttention._forward_fsmn``).

        This is asserted rather than merely documented because the same fused
        call came back empty for every Paraformer request at fp16 *and* bf16
        while the suite stayed green — the functional-API tests pass contiguous
        tensors and the fp32 parity oracles route to torch, where no launcher
        check is reached.  Needs no asset, so it runs unmarked rather than
        behind ``-m slow``.
        """
        from oasr.layers import DepthwiseConv1d

        channels, seq_len = 128, 63
        m = DepthwiseConv1d(channels, 11, padding=(7, 3), bias=False).cuda().to(dtype)
        qkv = torch.randn(2, seq_len, 3 * channels, device="cuda", dtype=dtype)
        _, _, x = torch.split(qkv, channels, dim=-1)
        assert not x.is_contiguous() and x.stride(1) == 3 * channels
        mask = torch.rand(2, seq_len, 1, device="cuda") > 0.2

        with pytest.raises(AssertionError, match="contiguous input"):
            m(x, mask=mask, add_input=True)

        # A row-strided mask is the same hazard, one argument over.  Slice after
        # the dtype cast: casting a view materialises a contiguous copy, which
        # would leave this half asserting nothing.
        strided_mask = (torch.rand(2, seq_len, 4, device="cuda") > 0.2)[:, :, :1]
        assert not strided_mask.is_contiguous()
        with pytest.raises(AssertionError, match="contiguous mask"):
            m(x.contiguous(), mask=strided_mask, add_input=True)

        # And the contract the caller is told to satisfy does hold.
        got = m(x.contiguous(), mask=mask, add_input=True)
        with layers_backend_override("torch"):
            ref = m(x.contiguous(), mask=mask, add_input=True)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

        # The torch path has no such requirement — a strided view is fine there,
        # which is exactly why an fp32/CPU oracle cannot catch the kernel case.
        with layers_backend_override("torch"):
            torch.testing.assert_close(
                m(x, mask=strided_mask, add_input=True),
                m(x.contiguous(), mask=strided_mask.contiguous(), add_input=True),
                rtol=2e-2,
                atol=2e-2,
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "in_channels,out_channels,kernel,groups",
        [(128, 128, 7, 128), (16, 32, 3, 4), (128, 384, 1, 1)],
    )
    def test_grouped_and_pointwise_conv2d(self, dtype, in_channels, out_channels, kernel, groups):
        from oasr.layers import Conv2d

        m = (
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=groups,
            )
            .cuda()
            .to(dtype)
        )
        x = torch.randn(2, 19, 11, in_channels, device="cuda", dtype=dtype)
        got = m(x)
        with layers_backend_override("torch"):
            ref = m(x)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kernel_size": 2, "stride": 2},
            {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True},
        ],
    )
    def test_avg_pool1d(self, dtype, kwargs):
        from oasr.layers import AvgPool1d

        m = AvgPool1d(**kwargs).cuda()
        x = torch.randn(2, 31, 128, device="cuda", dtype=dtype)
        got = m(x)
        with layers_backend_override("torch"):
            ref = m(x)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("hidden", [64, 100])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_norms(self, hidden, dtype):
        """``hidden=100`` is not a multiple of the fp16 vector width (8).  The
        launchers used to take the vectorized path anyway and fault the CUDA
        context with a misaligned address; they now drop to the scalar kernel."""
        from oasr.layers import AddLayerNorm, AddRMSNorm, BiasNorm, LayerNorm, RMSNorm

        x = torch.randn(4, 8, hidden, device="cuda", dtype=dtype)
        for m in (
            LayerNorm(hidden).cuda().to(dtype),
            RMSNorm(hidden, bias=False).cuda().to(dtype),
            BiasNorm(hidden).cuda().to(dtype),
        ):
            got = m(x)
            with layers_backend_override("torch"):
                ref = m(x)
            torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

        r = torch.randn_like(x)
        for add in (
            AddLayerNorm(hidden).cuda().to(dtype),
            AddRMSNorm(hidden, bias=False).cuda().to(dtype),
        ):
            got = add(x, r)
            got_norm, got_residual = add.forward_residual(x, r, alpha=0.5)
            with layers_backend_override("torch"):
                ref = add(x, r)
                ref_norm, ref_residual = add.forward_residual(x, r, alpha=0.5)
            torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(got_norm, ref_norm, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(got_residual, ref_residual, rtol=0, atol=0)

    def test_norm_refuses_non_contiguous_input(self):
        """A conv encoder's ``transpose(1, 2)`` output is not row-contiguous and
        the kernel addresses rows arithmetically, so it must take the torch
        path rather than read the wrong memory."""
        from oasr.layers import LayerNorm

        m = LayerNorm(64).cuda().half()
        x = torch.randn(4, 64, 8, device="cuda", dtype=torch.float16).transpose(1, 2)
        assert not x.is_contiguous()
        got = m(x)
        with layers_backend_override("torch"):
            ref = m(x.contiguous())
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_attention(self, head_dim):
        """``head_dim=128`` is a shape the CuteDSL kernel cannot implement on
        sm_120; the waist asks before dispatching, so it must degrade rather
        than raise."""
        from oasr.layers import Attention

        torch.manual_seed(0)
        a = Attention(4, head_dim)
        shape = (2, 4, 64, head_dim)
        q, k, v = (torch.randn(*shape, device="cuda", dtype=torch.float16) for _ in range(3))
        lens = torch.tensor([64, 20], device="cuda", dtype=torch.int32)
        for kwargs in ({}, {"kv_lens": lens}):
            got = a(q, k, v, **kwargs)
            with layers_backend_override("torch"):
                ref = a(q, k, v, **kwargs)
            assert not torch.isnan(got).any()
            torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestLeftPaddedWindowRouting:
    """``kv_starts`` is a kernel-eligible mask form, and the causal combination
    is routed on measured work — the two halves of what closed
    ``fmha-mask-form``."""

    @staticmethod
    def _qkv(B, H, T_q, T_k, D, dtype=torch.float16):
        torch.manual_seed(0)
        q = torch.randn(B, H, T_q, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, T_k, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, T_k, D, device="cuda", dtype=dtype)
        return q, k, v

    def test_left_padding_reaches_the_kernel(self):
        from oasr.layers import Attention
        from oasr.layers._backend import gap_hits, policy_hits, reset_backend_stats

        a = Attention(4, 64)
        q, k, v = self._qkv(2, 4, 96, 128, 64)
        lens = torch.tensor([128, 128], device="cuda", dtype=torch.int32)
        starts = torch.tensor([16, 48], device="cuda", dtype=torch.int32)
        reset_backend_stats()
        a(q, k, v, kv_lens=lens, kv_starts=starts)
        assert not gap_hits(), "left padding is no longer a declared gap"
        assert not policy_hits(), "a plain left-padded window should fuse"

    def test_kernel_and_torch_agree_on_a_left_padded_window(self):
        from oasr.layers import Attention
        from oasr.layers._backend import layers_backend_override

        a = Attention(4, 64)
        q, k, v = self._qkv(3, 4, 128, 160, 64)
        lens = torch.tensor([160, 120, 160], device="cuda", dtype=torch.int32)
        starts = torch.tensor([32, 8, 100], device="cuda", dtype=torch.int32)
        got = a(q, k, v, kv_lens=lens, kv_starts=starts)
        with layers_backend_override("torch"):
            ref = a(q, k, v, kv_lens=lens, kv_starts=starts)
        finite = torch.isfinite(ref)
        torch.testing.assert_close(got[finite], ref[finite], atol=2e-2, rtol=2e-2)

    def test_causal_window_fuses_above_the_work_floor(self):
        """Causal + a window costs SDPA a materialized mask *and* its flash
        path, so above the measured floor the fused kernel is the right call."""
        from oasr.layers import Attention
        from oasr.layers._backend import (
            FMHA_CAUSAL_WINDOW_MIN_MACS,
            gap_hits,
            policy_hits,
            reset_backend_stats,
        )

        B, H, T, D = 4, 28, 512, 128
        assert B * H * T * T * D >= FMHA_CAUSAL_WINDOW_MIN_MACS
        a = Attention(H, D)
        q, k, v = self._qkv(B, H, T, T, D, dtype=torch.bfloat16)
        lens = torch.full((B,), T, device="cuda", dtype=torch.int32)
        starts = torch.tensor([0, 32, 64, 96], device="cuda", dtype=torch.int32)
        reset_backend_stats()
        out = a(q, k, v, kv_lens=lens, kv_starts=starts, is_causal=True)
        assert not gap_hits()
        assert not policy_hits(), f"expected the fused path, got {policy_hits()}"
        assert torch.isfinite(out).all()

    def test_causal_window_stays_on_sdpa_below_the_work_floor(self):
        """Below it the fused path's fixed stride-copy cost dominates — measured
        0.34x at 4 MMACs.  A capability that is not always a win."""
        from oasr.layers import Attention
        from oasr.layers._backend import (
            FMHA_CAUSAL_WINDOW_MIN_MACS,
            policy_hits,
            reset_backend_stats,
        )

        B, H, T, D = 1, 4, 128, 64
        assert B * H * T * T * D < FMHA_CAUSAL_WINDOW_MIN_MACS
        a = Attention(H, D)
        q, k, v = self._qkv(B, H, T, T, D)
        lens = torch.full((B,), T, device="cuda", dtype=torch.int32)
        starts = torch.tensor([16], device="cuda", dtype=torch.int32)
        reset_backend_stats()
        a(q, k, v, kv_lens=lens, kv_starts=starts, is_causal=True)
        assert policy_hits().get("fmha-causal-window-small") == 1

    def test_sdpa_side_keeps_a_fully_padded_row_finite(self):
        """The SDPA path must not hand back NaN pad rows: a masked key still
        contributes ``0 * NaN`` in the next layer, so one NaN pad row poisons
        every real row.  The kernel clamps empty rows to zero; the torch path
        keeps the diagonal open to the same end."""
        from oasr.layers import Attention
        from oasr.layers._backend import layers_backend_override

        a = Attention(4, 64)
        q, k, v = self._qkv(2, 4, 64, 64, 64)
        lens = torch.full((2,), 64, device="cuda", dtype=torch.int32)
        starts = torch.tensor([32, 40], device="cuda", dtype=torch.int32)
        with layers_backend_override("torch"):
            out = a(q, k, v, kv_lens=lens, kv_starts=starts, is_causal=True)
        assert torch.isfinite(out).all()

    def test_start_without_length_raises(self):
        from oasr.layers import Attention

        a = Attention(4, 64)
        q, k, v = self._qkv(1, 4, 8, 8, 64)
        with pytest.raises(ValueError, match="requires kv_lens"):
            a(q, k, v, kv_starts=torch.zeros(1, device="cuda", dtype=torch.int32))
