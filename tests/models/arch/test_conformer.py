"""
Unit test for OASR Conformer encoder, using the WeNet SDPA implementation as
ground truth.

This test compares the outputs of the OASR ``ConformerEncoder`` under
`oasr.models.conformer` against the original WeNet ``ConformerEncoder`` from
`wenet.models.transformer.encoder`, which itself uses SDPA-based multi-head
attention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
import yaml

# This module *is* the comparison against upstream WeNet, so it genuinely needs
# the `wenet` package. Skip the module when it is absent instead of failing
# collection: a bare `from wenet...` here made `pytest tests/` error out
# entirely, which is why every verification command in the repo's docs carried
# `--ignore=tests/test_conformer.py`. Install it (`pip install wenet`) to enable
# the conformer parity oracle.
pytest.importorskip(
    "wenet",
    reason="upstream WeNet reference not installed; conformer parity oracle skipped",
)

from wenet.models.transformer import encoder as wenet_encoder  # type: ignore  # noqa: E402
from wenet.utils.common import mask_to_bias  # type: ignore  # noqa: E402
from wenet.utils.init_model import init_model  # type: ignore  # noqa: E402

from oasr.models.conformer import ConformerEncoder, load_wenet_checkpoint  # noqa: E402
from oasr.models.conformer.convert import build_config_from_wenet  # noqa: E402


def make_encoder_config(
    output_size: int = 64,
    num_blocks: int = 1,
    attention_heads: int = 2,
    linear_units: int = 128,
    use_cnn_module: bool = True,
    cnn_module_kernel: int = 15,
    use_sdpa: bool = True,
):
    """Build ConformerEncoder kwargs matching OASR ConformerEncoderConfig."""
    return {
        "input_size": 80,
        "output_size": output_size,
        "attention_heads": attention_heads,
        "linear_units": linear_units,
        "num_blocks": num_blocks,
        "dropout_rate": 0.0,
        "positional_dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
        "input_layer": "conv2d",
        "pos_enc_layer_type": "rel_pos",
        "normalize_before": True,
        "macaron_style": True,
        "use_cnn_module": use_cnn_module,
        "cnn_module_kernel": cnn_module_kernel,
        "causal": False,
        "activation_type": "swish",
        "use_sdpa": use_sdpa,
    }


def perturb_norms(module: torch.nn.Module) -> None:
    """Give every norm non-trivial parameters and statistics.

    At initialisation a ``BatchNorm1d`` in eval mode is close to the identity and
    a LayerNorm's affine part is, so swapping one norm for the other moved the
    output by less than the tolerance: this test passed while ``norm_conv`` was
    the wrong kind of norm.
    """
    with torch.no_grad():
        for mod in module.modules():
            if isinstance(mod, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-1.0, 1.0)
                mod.running_var.uniform_(0.5, 2.0)


@pytest.mark.parametrize("output_size,num_blocks", [(64, 1), (128, 1), (64, 4), (128, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_conformer_encoder_matches_wenet(output_size: int, num_blocks: int, dtype: torch.dtype):
    """Conformer encoder output matches the WeNet SDPA implementation."""
    torch.manual_seed(2)

    wenet_encoder_config = make_encoder_config(
        output_size=output_size,
        num_blocks=num_blocks,
        attention_heads=2,
        linear_units=128,
        use_cnn_module=True,
        cnn_module_kernel=15,
        use_sdpa=True,
    )

    ref_encoder = wenet_encoder.ConformerEncoder(**wenet_encoder_config)
    perturb_norms(ref_encoder)

    # Through the converter, as a checkpoint is: a config built by hand here
    # cannot see a field the converter drops or misroutes.
    raw = {"input_dim": 80, "output_dim": 32, "encoder_conf": wenet_encoder_config}
    impl_encoder = ConformerEncoder(build_config_from_wenet(raw).encoder)

    # Pass full WeNet state dict; _load_from_state_dict hooks in OASR modules
    # handle key remapping (conv.0 → conv1) and weight layout conversion.
    ref_sd = ref_encoder.state_dict()
    result = impl_encoder.load_state_dict(ref_sd, strict=False)
    assert result.missing_keys == []

    ref_encoder = ref_encoder.eval().to(dtype=dtype, device=torch.device("cuda"))
    impl_encoder = impl_encoder.eval().to(dtype=dtype, device=torch.device("cuda"))

    batch, time_in = 2, 40
    xs = torch.randn(batch, time_in, 80, dtype=dtype, device=torch.device("cuda"))
    T = xs.size(1)
    xs_lens = torch.full((batch,), T, dtype=torch.long, device=torch.device("cuda"))
    mask = wenet_encoder.make_pad_mask(xs_lens, T).unsqueeze(1)

    with torch.no_grad():
        x_embed, pos_emb, mask_embed = ref_encoder.embed(xs, mask)
        mask_pad = mask_embed
        x_embed_impl = x_embed.clone()

        mask_embed_ref = mask_to_bias(mask_embed, x_embed.dtype)
        mask_embed_impl = mask_to_bias(mask_embed.clone(), x_embed.dtype)

        # Use empty cache for each ref layer so key length stays equal to
        # pos_emb length (required by WeNet rel_pos attention).
        for i in range(num_blocks):
            att_cache = (
                torch.zeros(0, 0, 0, 0, device=torch.device("cuda")),
                torch.zeros(0, 0, 0, 0, device=torch.device("cuda")),
            )
            cnn_cache = torch.zeros(0, 0, 0, device=torch.device("cuda"))
            x_embed, mask_embed_ref, _, _ = ref_encoder.encoders[i](
                x_embed, mask_embed_ref, pos_emb, mask_pad, att_cache, cnn_cache
            )
        ref_out = x_embed

        # Same for impl: empty cache per layer so outputs match ref.
        for i in range(num_blocks):
            att_cache2 = (
                torch.zeros(0, 0, 0, 0, device=torch.device("cuda")),
                torch.zeros(0, 0, 0, 0, device=torch.device("cuda")),
            )
            cnn_cache2 = torch.zeros(0, 0, 0, device=torch.device("cuda"))
            x_embed_impl, _, _ = impl_encoder.encoders[i](
                x_embed_impl, mask_embed_impl, pos_emb, mask_pad, att_cache2, cnn_cache2
            )
        impl_out = x_embed_impl

    assert ref_out.shape == impl_out.shape
    torch.testing.assert_close(impl_out, ref_out, rtol=5e-2, atol=5e-2)


# NB: dense ``forward_chunk`` tests have been removed; streaming is now
# paged-only and is covered by tests/test_pipeline.py and tests/test_engine.py.


def load_wenet_model_from_ckpt_dir(ckpt_dir: Path, device: str):
    """Load full WeNet ASR model from checkpoint dir (train.yaml + final.pt).

    Deleted as collateral by ``c4ff9db`` (which removed the dense chunk tests)
    while ``test_load_checkpoint_matches_wenet`` below kept calling it, so that
    test raised ``NameError`` on every run it was not skipped for.  Restored
    verbatim; ``ruff``'s F821 is what surfaced it.
    """
    yaml_path = ckpt_dir / "train.yaml"
    with open(yaml_path, "r") as f:
        configs = yaml.safe_load(f)
    # Point cmvn to the checkpoint dir (train.yaml may use relative paths).
    if configs.get("cmvn") == "global_cmvn" and "cmvn_conf" in configs:
        configs["cmvn_conf"] = dict(configs["cmvn_conf"])
        configs["cmvn_conf"]["cmvn_file"] = str(ckpt_dir / "global_cmvn")
    # Use SDPA so encoder matches OASR attention path for numerical comparison.
    if "encoder_conf" in configs:
        configs["encoder_conf"] = dict(configs["encoder_conf"])
        configs["encoder_conf"]["use_sdpa"] = True
    args = argparse.Namespace(checkpoint=str(ckpt_dir / "final.pt"))
    model, _ = init_model(args, configs)
    model = model.to(device).eval()
    return model


@pytest.mark.parametrize("batch,time_in,feat_dim", [(2, 80, 80), (1, 60, 80)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_load_checkpoint_matches_wenet(
    ckpt_dir: str, batch: int, time_in: int, feat_dim: int, dtype: torch.dtype
):
    """Encoder inference outputs match between OASR and WeNet for the same checkpoint."""
    # The `ckpt_dir` fixture gates on CKPT_DIR through tests/assets.py (marker
    # `final.pt`); `train.yaml` is the WeNet-specific extra this oracle needs.
    if not (Path(ckpt_dir) / "train.yaml").exists():
        pytest.skip(f"{ckpt_dir} has no train.yaml — not a WeNet checkpoint dir")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    oasr_model, _ = load_wenet_checkpoint(str(ckpt_dir), device=device, dtype=dtype)
    wenet_model = load_wenet_model_from_ckpt_dir(Path(ckpt_dir), device)
    wenet_model = wenet_model.to(dtype=dtype)

    torch.manual_seed(42)
    feats = torch.randn(batch, time_in, feat_dim, dtype=dtype, device=device)
    lengths = torch.full((batch,), time_in, dtype=torch.long, device=device)

    with torch.no_grad():
        wenet_encoder_out, wenet_masks = wenet_model.encoder(
            feats, lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1
        )
        wenet_probs = wenet_model.ctc.log_softmax(wenet_encoder_out)

        oasr_probs = oasr_model(feats, lengths)
        oasr_probs = oasr_probs[:, :, : wenet_probs.shape[2]]

    assert (
        oasr_probs.shape == wenet_probs.shape
    ), f"Shape mismatch: OASR {oasr_probs.shape} vs WeNet {wenet_probs.shape}"
    torch.testing.assert_close(
        oasr_probs,
        wenet_probs,
        rtol=5e-2,
        atol=5e-2,
        msg="OASR and WeNet probs should match for the same checkpoint and input.",
    )
