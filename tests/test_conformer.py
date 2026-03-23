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

import oasr
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

    # Pass full WeNet state dict; _load_from_state_dict hooks in OASR modules
    # handle key remapping (conv.0 → conv1) and weight layout conversion.
    ref_sd = ref_encoder.state_dict()
    impl_encoder.load_state_dict(ref_sd, strict=False)

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


@pytest.mark.parametrize("chunk_size", [2, 4])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_chunk_encoder_matches_wenet(
    chunk_size: int,
    num_blocks: int,
    dtype: torch.dtype,
):
    """Encoder forward_chunk output matches the WeNet ConformerEncoder.forward_chunk."""
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bfloat16 requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    wenet_encoder_config = make_encoder_config(
        output_size=64,
        num_blocks=num_blocks,
        attention_heads=2,
        linear_units=128,
        use_cnn_module=True,
        cnn_module_kernel=15,
        use_sdpa=True,
    )

    ref_encoder = wenet_encoder.ConformerEncoder(**wenet_encoder_config)
    impl_encoder = ConformerEncoder(ConformerEncoderConfig(
        input_size=wenet_encoder_config["input_size"],
        output_size=wenet_encoder_config["output_size"],
        num_blocks=wenet_encoder_config["num_blocks"],
        attention_heads=wenet_encoder_config["attention_heads"],
        linear_units=wenet_encoder_config["linear_units"],
        use_cnn_module=wenet_encoder_config["use_cnn_module"],
        cnn_module_kernel=wenet_encoder_config["cnn_module_kernel"],
        embed_layer_norm=False,  # match WeNet default (no LayerNorm in embed)
    ))

    # Pass full WeNet state dict; _load_from_state_dict hooks in OASR modules
    # handle key remapping (conv.0 → conv1) and weight layout conversion.
    ref_sd = ref_encoder.state_dict()
    impl_encoder.load_state_dict(ref_sd, strict=False)

    ref_encoder = ref_encoder.eval().to(dtype=dtype, device=device)
    impl_encoder = impl_encoder.eval().to(dtype=dtype, device=device)

    time_in = chunk_input_time(chunk_size)
    xs = torch.randn(1, time_in, 80, dtype=dtype, device=device)
    offset = 0
    required_cache_size = -1
    att_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
    cnn_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
    # Both WeNet and OASR attention expect additive bias (float), not bool. First chunk: key_size = chunk_size.
    att_mask_bias = torch.zeros(
        1, chunk_size, chunk_size, dtype=dtype, device=device)

    with torch.no_grad():
        ref_out, ref_att_cache, ref_cnn_cache = ref_encoder.forward_chunk(
            xs,
            offset,
            required_cache_size,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            att_mask=att_mask_bias,
        )
        impl_out, impl_att_cache, impl_cnn_cache = impl_encoder.forward_chunk(
            xs,
            offset,
            required_cache_size,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            att_mask=att_mask_bias,
        )

    assert ref_out.shape == impl_out.shape, (
        f"output shape: WeNet {ref_out.shape} vs OASR {impl_out.shape}"
    )
    torch.testing.assert_close(impl_out, ref_out, rtol=5e-2, atol=5e-2)
    assert ref_att_cache.shape == impl_att_cache.shape, (
        f"att_cache shape: WeNet {ref_att_cache.shape} vs OASR {impl_att_cache.shape}"
    )
    torch.testing.assert_close(
        impl_att_cache, ref_att_cache, rtol=5e-2, atol=5e-2)
    assert ref_cnn_cache.shape == impl_cnn_cache.shape, (
        f"cnn_cache shape: WeNet {ref_cnn_cache.shape} vs OASR {impl_cnn_cache.shape}"
    )
    if ref_cnn_cache.numel() > 0:
        torch.testing.assert_close(
            impl_cnn_cache, ref_cnn_cache, rtol=5e-2, atol=5e-2)


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


def chunk_input_time(chunk_size: int, subsample_rate: int = 4, right_context: int = 6) -> int:
    """Input time length for one chunk so that after subsampling we get chunk_size frames."""
    return (chunk_size - 1) * subsample_rate + right_context + 1


@pytest.mark.parametrize("chunk_size", [2, 4, 8])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_chunk_encoder_shapes(
    chunk_size: int,
    num_blocks: int,
    dtype: torch.dtype,
):
    """Encoder forward_chunk returns correct output and cache shapes (b=1, empty caches)."""
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bfloat16 requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3)

    config = ConformerEncoderConfig(
        input_size=80,
        output_size=64,
        num_blocks=num_blocks,
        attention_heads=2,
        linear_units=128,
        use_cnn_module=True,
        cnn_module_kernel=15,
        causal=False,
    )
    encoder = ConformerEncoder(config).eval().to(device=device, dtype=dtype)

    time_in = chunk_input_time(chunk_size)
    xs = torch.randn(1, time_in, 80, dtype=dtype, device=device)
    offset = 0
    required_cache_size = -1  # keep full history
    att_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
    cnn_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)

    with torch.no_grad():
        out, r_att_cache, r_cnn_cache = encoder.forward_chunk(
            xs, offset, required_cache_size, att_cache, cnn_cache
        )

    assert out.shape == (1, chunk_size, 64), f"out shape {out.shape}"
    assert r_att_cache.dim() == 4
    assert r_att_cache.size(0) == num_blocks
    assert r_att_cache.size(1) == 2  # attention_heads
    # required_cache_size < 0 -> keep all
    assert r_att_cache.size(2) == chunk_size
    assert r_att_cache.size(3) == 64  # d_k * 2 for key+value
    assert r_cnn_cache.dim() == 4
    assert r_cnn_cache.size(0) == num_blocks
    # causal=False -> lorder=0; cnn_cache can be (elayers, 0, 0, 0) when no conv cache
    assert r_cnn_cache.size(2) == 0
    assert r_cnn_cache.size(3) == 0 or r_cnn_cache.size(3) == 64


@pytest.mark.parametrize("chunk_size", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_chunk_encoder_first_chunk_matches_full(chunk_size: int, dtype: torch.dtype):
    """First chunk with empty cache should match full encoder on the same input (single chunk)."""
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bfloat16 requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(4)

    config = ConformerModelConfig(
        encoder=ConformerEncoderConfig(
            input_size=80,
            output_size=64,
            num_blocks=1,
            attention_heads=2,
            linear_units=128,
            use_cnn_module=True,
            cnn_module_kernel=15,
            causal=False,
        ),
        vocab_size=128,
    )
    model = ConformerModel(config).eval().to(device=device, dtype=dtype)

    time_in = chunk_input_time(chunk_size)
    xs = torch.randn(1, time_in, 80, dtype=dtype, device=device)
    xs_lens = torch.full((1,), time_in, dtype=torch.long, device=device)

    with torch.no_grad():
        full_out = model(xs, xs_lens)
        chunk_out, _, _ = model.forward_chunk(
            xs,
            offset=0,
            required_cache_size=-1,
            att_cache=torch.zeros(0, 0, 0, 0, dtype=dtype, device=device),
            cnn_cache=torch.zeros(0, 0, 0, 0, dtype=dtype, device=device),
        )

    assert full_out.shape == chunk_out.shape
    torch.testing.assert_close(chunk_out, full_out, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_chunk_encoder_two_chunks(chunk_size: int, dtype: torch.dtype):
    """Two consecutive chunks with cache: shapes and no crash."""
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bfloat16 requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(5)

    config = ConformerEncoderConfig(
        input_size=80,
        output_size=64,
        num_blocks=1,
        attention_heads=2,
        linear_units=128,
        use_cnn_module=True,
        cnn_module_kernel=15,
        causal=False,
    )
    encoder = ConformerEncoder(config).eval().to(device=device, dtype=dtype)

    time_in = chunk_input_time(chunk_size)
    xs1 = torch.randn(1, time_in, 80, dtype=dtype, device=device)
    xs2 = torch.randn(1, time_in, 80, dtype=dtype, device=device)

    with torch.no_grad():
        out1, att_cache, cnn_cache = encoder.forward_chunk(
            xs1, offset=0, required_cache_size=chunk_size,
            att_cache=torch.zeros(0, 0, 0, 0, dtype=dtype, device=device),
            cnn_cache=torch.zeros(0, 0, 0, 0, dtype=dtype, device=device),
        )
    assert out1.shape == (1, chunk_size, 64)
    assert att_cache.shape[0] == 1 and att_cache.shape[1] == 2
    assert cnn_cache.dim() == 4 and cnn_cache.size(0) == 1

    with torch.no_grad():
        out2, att_cache2, cnn_cache2 = encoder.forward_chunk(
            xs2, offset=chunk_size, required_cache_size=chunk_size,
            att_cache=att_cache, cnn_cache=cnn_cache,
        )
    assert out2.shape == (1, chunk_size, 64)
    assert att_cache2.shape[0] == 1
    assert cnn_cache2.shape[0] == 1


@pytest.mark.parametrize("chunk_size", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_chunk_model_shapes(chunk_size: int, dtype: torch.dtype):
    """Conformer encoder forward_chunk + CTC produce expected shapes (logits, att_cache, cnn_cache).
    Logits are computed via F.linear to avoid custom GEMM NOT_SUPPORTED for small chunk sizes.
    """
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bfloat16 requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6)

    config = ConformerModelConfig(
        encoder=ConformerEncoderConfig(
            input_size=80,
            output_size=64,
            num_blocks=1,
            attention_heads=2,
            linear_units=128,
            use_cnn_module=True,
            cnn_module_kernel=15,
            causal=False,
        ),
        vocab_size=128,
    )
    model = ConformerModel(config).eval().to(device=device, dtype=dtype)

    time_in = chunk_input_time(chunk_size)
    xs = torch.randn(1, time_in, 80, dtype=dtype, device=device)
    att_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
    cnn_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)

    with torch.no_grad():
        probs, att_cache, cnn_cache = model.forward_chunk(
            xs,
            offset=0,
            required_cache_size=-1,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
        )

    assert probs.shape == (1, chunk_size, 128)
    assert att_cache.dim() == 4 and att_cache.size(0) == 1
    assert cnn_cache.dim() == 4 and cnn_cache.size(0) == 1


@pytest.mark.parametrize("batch,time_in,feat_dim", [(2, 80, 80), (1, 60, 80)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_load_checkpoint_matches_wenet(ckpt_dir: str, batch: int, time_in: int, feat_dim: int, dtype: torch.dtype):
    """Encoder inference outputs match between OASR and WeNet for the same checkpoint."""
    # Skip this test when no valid WeNet checkpoint directory is provided.
    # The ``ckpt_dir`` fixture returns an empty string when the option/env var
    # is not set, and ``Path('')`` resolves to the current working directory,
    # which incorrectly makes this test run and then fail with a missing
    # ``train.yaml`` error.  Instead, require that the directory exists and
    # contains the expected WeNet files.
    if not ckpt_dir:
        pytest.skip(
            "WeNet checkpoint dir not set; set WENET_CKPT_DIR env var or --ckpt-dir/--wenet-ckpt-dir"
        )
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists() or not (ckpt_path / "train.yaml").exists() or not (ckpt_path / "final.pt").exists():
        pytest.skip(
            "WeNet checkpoint dir not found or incomplete; set WENET_CKPT_DIR env var or --ckpt-dir/--wenet-ckpt-dir "
            "to a directory containing train.yaml and final.pt"
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
        wenet_probs = wenet_model.ctc.log_softmax(wenet_encoder_out)

        oasr_probs = oasr_model(feats, lengths)
        oasr_probs = oasr_probs[:, :, :wenet_probs.shape[2]]

    assert oasr_probs.shape == wenet_probs.shape, (
        f"Shape mismatch: OASR {oasr_probs.shape} vs WeNet {wenet_probs.shape}"
    )
    torch.testing.assert_close(
        oasr_probs,
        wenet_probs,
        rtol=5e-2,
        atol=5e-2,
        msg="OASR and WeNet probs should match for the same checkpoint and input.",
    )
