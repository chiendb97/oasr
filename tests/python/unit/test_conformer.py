"""
Unit tests for OASR Conformer model, using WeNet as ground truth.

These tests compare the outputs of the OASR Conformer implementations under
`oasr.models.conformer` against the original WeNet reference implementations
in `wenet.models.transformer`.
"""

from __future__ import annotations

import pytest
import torch

from wenet.models.transformer import encoder as wenet_encoder  # type: ignore  # noqa: E402
from wenet.models.transformer import (  # type: ignore  # noqa: E402
    convolution as wenet_conv,
    positionwise_feed_forward as wenet_ff,
)

from oasr.models.conformer import (  # noqa: E402
    ConformerEncoder,
    ConformerEncoderConfig,
    ConvolutionModule,
    PositionwiseFeedForward,
)
from oasr.models.conformer.model import RelPositionalEncoding  # noqa: E402


def _randn(*shape: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=torch.float32)


# -----------------------------------------------------------------------------
# PositionwiseFeedForward
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("idim,hidden_units", [(64, 128), (256, 2048)])
def test_positionwise_feed_forward_matches_wenet(idim: int, hidden_units: int):
    """PositionwiseFeedForward outputs should match WeNet implementation."""
    torch.manual_seed(0)
    dropout = 0.0
    # Conformer uses swish (SiLU); pass explicitly so ref and impl match
    activation = torch.nn.SiLU()

    ref = wenet_ff.PositionwiseFeedForward(
        idim=idim,
        hidden_units=hidden_units,
        dropout_rate=dropout,
        activation=activation,
    )
    impl = PositionwiseFeedForward(
        idim=idim,
        hidden_units=hidden_units,
        dropout_rate=dropout,
        activation=activation,
    )
    impl.load_state_dict(ref.state_dict())

    ref.eval()
    impl.eval()

    batch, time = 2, 10
    xs = _randn(batch, time, idim)

    with torch.no_grad():
        out_ref = ref(xs)
        out_impl = impl(xs)

    assert out_ref.shape == out_impl.shape
    torch.testing.assert_close(out_impl, out_ref, rtol=1e-5, atol=1e-6)


# -----------------------------------------------------------------------------
# ConvolutionModule
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("channels,kernel_size", [(64, 15), (256, 31)])
@pytest.mark.parametrize("causal", [False, True])
def test_convolution_module_matches_wenet(
    channels: int, kernel_size: int, causal: bool
):
    """ConvolutionModule outputs should match WeNet implementation."""
    torch.manual_seed(1)
    # Use SiLU to match Conformer's swish
    activation = torch.nn.SiLU()

    ref = wenet_conv.ConvolutionModule(
        channels=channels,
        kernel_size=kernel_size,
        activation=activation,
        norm="batch_norm",
        causal=causal,
        bias=True,
        norm_eps=1e-5,
        conv_inner_factor=2,
    )
    impl = ConvolutionModule(
        channels=channels,
        kernel_size=kernel_size,
        activation=activation,
        norm="batch_norm",
        causal=causal,
        bias=True,
        norm_eps=1e-5,
        conv_inner_factor=2,
    )
    impl.load_state_dict(ref.state_dict())

    ref.eval()
    impl.eval()

    batch, time = 2, 20
    x = _randn(batch, time, channels)
    mask_pad = torch.ones(batch, 1, time, dtype=torch.bool)
    if time > 1:
        mask_pad[:, :, -1] = 0
    # ConvolutionModule expects mask_pad (B, 1, T) for masking; WeNet uses (B, 1, T) in conv
    # OASR model uses mask_pad (B, 1, T) in conv_module forward
    cache = torch.zeros(0, 0, 0)

    with torch.no_grad():
        out_ref, cache_ref = ref(x, mask_pad, cache)
        out_impl, cache_impl = impl(x, mask_pad, cache)

    assert out_ref.shape == out_impl.shape
    torch.testing.assert_close(out_impl, out_ref, rtol=1e-5, atol=1e-6)
    assert cache_ref.shape == cache_impl.shape
    if cache_ref.numel() > 0:
        torch.testing.assert_close(cache_impl, cache_ref, rtol=1e-5, atol=1e-6)


# -----------------------------------------------------------------------------
# ConformerEncoderLayer (same embedded input)
# -----------------------------------------------------------------------------


def _make_wenet_encoder_config(
    output_size: int = 64,
    num_blocks: int = 1,
    attention_heads: int = 2,
    linear_units: int = 128,
    dropout_rate: float = 0.0,
    use_cnn_module: bool = True,
    cnn_module_kernel: int = 15,
):
    """Build WeNet ConformerEncoder kwargs matching OASR ConformerEncoderConfig."""
    return dict(
        input_size=80,
        output_size=output_size,
        attention_heads=attention_heads,
        linear_units=linear_units,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        positional_dropout_rate=dropout_rate,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_layer_type="rel_pos",
        normalize_before=True,
        macaron_style=True,
        use_cnn_module=use_cnn_module,
        cnn_module_kernel=cnn_module_kernel,
        causal=False,
        activation_type="swish",
    )


@pytest.mark.parametrize("output_size,num_blocks", [(64, 1), (128, 1)])
def test_conformer_encoder_layer_matches_wenet(
    output_size: int, num_blocks: int
):
    """ConformerEncoderLayer output matches WeNet when given same embedded input."""
    torch.manual_seed(2)

    ref_encoder = wenet_encoder.ConformerEncoder(
        **_make_wenet_encoder_config(
            output_size=output_size,
            num_blocks=num_blocks,
        )
    )
    impl_encoder = ConformerEncoder(
        ConformerEncoderConfig(
            input_size=80,
            output_size=output_size,
            num_blocks=num_blocks,
            attention_heads=2,
            linear_units=128,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
            use_cnn_module=True,
            cnn_module_kernel=15,
        )
    )
    ref_sd = ref_encoder.state_dict()
    load_sd = {k: ref_sd[k] for k in impl_encoder.state_dict() if k in ref_sd}
    impl_encoder.load_state_dict(load_sd, strict=False)

    ref_encoder.eval()
    impl_encoder.eval()

    batch, time_in = 2, 40
    xs = _randn(batch, time_in, 80)
    # Build mask for subsampling: (B, 1, T) valid positions
    T = xs.size(1)
    xs_lens = torch.full((batch,), T, dtype=torch.long)
    mask = wenet_encoder.make_pad_mask(xs_lens, T).unsqueeze(1)

    with torch.no_grad():
        # Single embedded input from WeNet embed; feed through ref layers and impl layers
        x_embed, pos_emb, mask_embed = ref_encoder.embed(xs, mask)
        mask_pad = mask_embed
        x_embed_impl = x_embed.clone()
        mask_embed_impl = mask_embed.clone()

        att_cache = (torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0))
        cnn_cache = torch.zeros(0, 0, 0)
        for i in range(num_blocks):
            x_embed, mask_embed, att_cache, cnn_cache = ref_encoder.encoders[i](
                x_embed, mask_embed, pos_emb, mask_pad, att_cache, cnn_cache
            )
        ref_out = x_embed

        att_cache2 = (torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0))
        cnn_cache2 = torch.zeros(0, 0, 0)
        for i in range(num_blocks):
            x_embed_impl, mask_embed_impl, att_cache2, cnn_cache2 = impl_encoder.encoders[i](
                x_embed_impl, mask_embed_impl, pos_emb, mask_pad, att_cache2, cnn_cache2
            )
        impl_out = x_embed_impl

    assert ref_out.shape == impl_out.shape
    torch.testing.assert_close(impl_out, ref_out, rtol=1e-5, atol=1e-6)


# -----------------------------------------------------------------------------
# ConformerEncoder forward shape and sanity
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("input_size,output_size,num_blocks", [(80, 64, 1), (80, 256, 2)])
def test_conformer_encoder_forward_shape(
    input_size: int, output_size: int, num_blocks: int
):
    """ConformerEncoder forward returns correct shapes."""
    torch.manual_seed(3)
    config = ConformerEncoderConfig(
        input_size=input_size,
        output_size=output_size,
        num_blocks=num_blocks,
        attention_heads=4,
        linear_units=256,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        use_cnn_module=True,
        cnn_module_kernel=15,
    )
    encoder = ConformerEncoder(config)
    encoder.eval()

    batch, time_in = 2, 80
    xs = _randn(batch, time_in, input_size)
    xs_lens = torch.full((batch,), time_in, dtype=torch.long)

    with torch.no_grad():
        out, masks = encoder(xs, xs_lens)

    # After conv2d 4x subsampling: time roughly (time_in - 6) / 4
    expected_time = (time_in - 1) // 2
    expected_time = (expected_time - 1) // 2
    assert out.shape == (batch, expected_time, output_size)
    assert masks.shape == (batch, 1, expected_time)


@pytest.mark.parametrize("batch", [1, 2])
def test_conformer_encoder_variable_length(batch: int):
    """ConformerEncoder handles variable sequence lengths via mask."""
    torch.manual_seed(4)
    config = ConformerEncoderConfig(
        input_size=80,
        output_size=64,
        num_blocks=1,
        attention_heads=2,
        linear_units=128,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        use_cnn_module=True,
    )
    encoder = ConformerEncoder(config)
    encoder.eval()

    max_len = 40
    xs = _randn(batch, max_len, 80)
    # Different lengths per batch
    xs_lens = torch.tensor([max_len, max_len - 10][:batch], dtype=torch.long)

    with torch.no_grad():
        out, masks = encoder(xs, xs_lens)

    expected_time = (max_len - 1) // 2
    expected_time = (expected_time - 1) // 2
    assert out.shape[0] == batch
    assert out.shape[1] == expected_time
    assert out.shape[2] == 64
    assert masks.shape == (batch, 1, expected_time)


# -----------------------------------------------------------------------------
# RelPositionalEncoding (OASR vs WeNet-style behaviour)
# -----------------------------------------------------------------------------


def test_rel_positional_encoding_shape():
    """RelPositionalEncoding returns scaled x and pos_emb with correct shapes."""
    torch.manual_seed(5)
    d_model = 64
    pos_enc = RelPositionalEncoding(d_model, dropout_rate=0.0, max_len=5000)

    batch, time = 2, 20
    x = _randn(batch, time, d_model)

    with torch.no_grad():
        x_out, pos_emb = pos_enc(x, offset=0)

    assert x_out.shape == (batch, time, d_model)
    # pos_emb is (1, time, d_model) from buffer slice, broadcasts over batch
    assert pos_emb.shape[0] in (1, batch)
    assert pos_emb.shape[1] == time
    assert pos_emb.shape[2] == d_model
