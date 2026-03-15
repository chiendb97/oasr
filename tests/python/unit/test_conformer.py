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

from wenet.models.transformer import encoder as wenet_encoder  # type: ignore  # noqa: E402
from wenet.utils.common import mask_to_bias  # type: ignore
from wenet.utils.init_model import init_model  # type: ignore

from oasr.models.conformer import (  # noqa: E402
    ConformerEncoder,
    ConformerEncoderConfig,
    ConformerModel,
    ConformerModelConfig,
    load_wenet_checkpoint,
)


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
    return dict(
        input_size=80,
        output_size=output_size,
        attention_heads=attention_heads,
        linear_units=linear_units,
        num_blocks=num_blocks,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_layer_type="rel_pos",
        normalize_before=True,
        macaron_style=True,
        use_cnn_module=use_cnn_module,
        cnn_module_kernel=cnn_module_kernel,
        causal=False,
        activation_type="swish",
        use_sdpa=use_sdpa,
    )


@pytest.mark.parametrize("output_size,num_blocks", [(64, 1), (128, 1), (64, 4), (128, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_conformer_encoder_matches_wenet(
    output_size: int, num_blocks: int, dtype: torch.dtype
):
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

    ref_encoder = wenet_encoder.ConformerEncoder(
        **wenet_encoder_config
    )

    impl_encoder = ConformerEncoder(ConformerEncoderConfig(
        input_size=wenet_encoder_config["input_size"],
        output_size=wenet_encoder_config["output_size"],
        num_blocks=wenet_encoder_config["num_blocks"],
        attention_heads=wenet_encoder_config["attention_heads"],
        linear_units=wenet_encoder_config["linear_units"],
        use_cnn_module=wenet_encoder_config["use_cnn_module"],
        cnn_module_kernel=wenet_encoder_config["cnn_module_kernel"],
    ))

    ref_sd = ref_encoder.state_dict()
    load_sd = {k: ref_sd[k] for k in impl_encoder.state_dict() if k in ref_sd}
    impl_encoder.load_state_dict(load_sd, strict=False)

    ref_encoder = ref_encoder.eval().to(dtype=dtype, device=torch.device("cuda"))
    impl_encoder = impl_encoder.eval().to(dtype=dtype, device=torch.device("cuda"))

    batch, time_in = 2, 40
    xs = torch.randn(batch, time_in, 80, dtype=dtype,
                     device=torch.device("cuda"))
    T = xs.size(1)
    xs_lens = torch.full((batch,), T, dtype=torch.long,
                         device=torch.device("cuda"))
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
            att_cache = (torch.zeros(0, 0, 0, 0, device=torch.device("cuda")),
                         torch.zeros(0, 0, 0, 0, device=torch.device("cuda")))
            cnn_cache = torch.zeros(0, 0, 0, device=torch.device("cuda"))
            x_embed, mask_embed_ref, _, _ = ref_encoder.encoders[i](
                x_embed, mask_embed_ref, pos_emb, mask_pad, att_cache, cnn_cache
            )
        ref_out = x_embed

        # Same for impl: empty cache per layer so outputs match ref.
        for i in range(num_blocks):
            att_cache2 = (torch.zeros(0, 0, 0, 0, device=torch.device("cuda")),
                          torch.zeros(0, 0, 0, 0, device=torch.device("cuda")))
            cnn_cache2 = torch.zeros(0, 0, 0, device=torch.device("cuda"))
            x_embed_impl, _, _ = impl_encoder.encoders[i](
                x_embed_impl, mask_embed_impl, pos_emb, mask_pad, att_cache2, cnn_cache2
            )
        impl_out = x_embed_impl

    assert ref_out.shape == impl_out.shape
    torch.testing.assert_close(impl_out, ref_out, rtol=5e-2, atol=5e-2)


def load_wenet_model_from_ckpt_dir(ckpt_dir: Path, device: str):
    """Load full WeNet ASR model from checkpoint dir (train.yaml + final.pt)."""
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
def test_load_checkpoint_matches_wenet(ckpt_dir: str, batch: int, time_in: int, feat_dim: int, dtype: torch.dtype):
    """Encoder inference outputs match between OASR and WeNet for the same checkpoint."""
    if not Path(ckpt_dir).exists():
        pytest.skip(
            "WeNet checkpoint dir not set or not found; set WENET_CKPT_DIR env var or --wenet-ckpt-dir"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    oasr_model, _ = load_wenet_checkpoint(
        str(ckpt_dir), device=device, dtype=dtype)
    wenet_model = load_wenet_model_from_ckpt_dir(Path(ckpt_dir), device)
    wenet_model = wenet_model.to(dtype=dtype)

    torch.manual_seed(42)
    feats = torch.randn(batch, time_in, feat_dim, dtype=dtype, device=device)
    lengths = torch.full((batch,), time_in, dtype=torch.long, device=device)

    with torch.no_grad():
        wenet_encoder_out, wenet_masks = wenet_model.encoder(
            feats, lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1
        )

        oasr_encoder_out, oasr_masks = oasr_model(feats, lengths)

    assert oasr_encoder_out.shape == wenet_encoder_out.shape, (
        f"Shape mismatch: OASR {oasr_encoder_out.shape} vs WeNet {wenet_encoder_out.shape}"
    )
    torch.testing.assert_close(
        oasr_encoder_out,
        wenet_encoder_out,
        rtol=5e-2,
        atol=5e-2,
        msg="OASR and WeNet encoder outputs should match for the same checkpoint and input.",
    )
