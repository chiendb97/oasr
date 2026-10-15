# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Positional-encoding tables past their end, and under graph capture.

WeNet sizes the sinusoid table at 5000 rows -- 200 s of audio at 40 ms frames --
and the table is a cache of a formula, not a property of the weights.  Reading
past it was a shape error offline and, in streaming, an out-of-bounds gather: a
device-side assert that took the CUDA context, and every stream on it, down once
a stream with bounded history ran long enough.  A position past the table is now
computed.  These pin that it is computed as a larger table would have held it,
and that a row the table does hold still comes from it.

The rows past a table must also never be *stored*.  Zipformer grew its table in
place, as icefall does, and Paraformer kept a process-wide one that grew with the
longest request; either way the old storage was freed under every captured
offline graph that read it, so one long request left each earlier capture
replaying positional embeddings out of whatever the allocator put there next.
The last section pins that a long request leaves a captured encoder intact.
"""

from __future__ import annotations

import pytest
import torch
from helpers import capture_graph, occupy_free_memory, written_sentinels

from oasr.models.conformer import ConformerEncoder, ConformerEncoderConfig
from oasr.models.conformer.model import RelPositionalEncoding
from oasr.models.registry import get_model_entry
from oasr.models.zipformer.encoder import CompactRelPositionalEncoding

D = 64
ROWS = 100
#: A few fp32 ulps of a value in [-1, 1]: CPU and GPU ``sin`` may differ in the
#: last bit, which is all a computed row is allowed to differ by.
SIN_ATOL = 1e-6

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _encodings(device: str, dtype: torch.dtype = torch.float32):
    """The table under test and one four times its size, as the reference."""
    small = RelPositionalEncoding(D, max_len=ROWS).to(device=device, dtype=dtype)
    large = RelPositionalEncoding(D, max_len=4 * ROWS).to(device=device, dtype=dtype)
    return small, large


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "offset,size",
    [(0, 3 * ROWS), (ROWS - 10, 30), (ROWS + 50, 40), (10, 20)],
    ids=["from-zero", "straddling", "past", "inside"],
)
def test_host_offset_past_the_table_matches_a_larger_table(device, offset, size):
    small, large = _encodings(device)
    got = small.position_encoding(offset, size)
    assert got.shape == (1, size, D)
    torch.testing.assert_close(got, large.position_encoding(offset, size), rtol=0, atol=SIN_ATOL)
    if offset + size <= ROWS:
        assert torch.equal(got, small.pe[:, offset : offset + size])  # still the table's rows


@pytest.mark.parametrize("device", DEVICES)
def test_per_stream_offsets_past_the_table_match_a_larger_table(device):
    """The streaming form: one device offset per stream, so no host-side bound."""
    small, large = _encodings(device)
    offsets = torch.tensor([0, ROWS - 20, 3 * ROWS, -5], device=device)
    size = 50
    got = small.position_encoding(offsets, size)
    want = large.position_encoding(offsets, size)
    assert got.shape == (4, size, D)
    torch.testing.assert_close(got, want, rtol=0, atol=SIN_ATOL)
    index = (offsets.unsqueeze(1) + torch.arange(size, device=device)).clamp(min=0)
    inside = index < ROWS
    assert torch.equal(got[inside], want[inside])  # bit for bit where the table has the row


def test_half_precision_keeps_full_precision_frequencies():
    """``model.to(float16)`` casts floating buffers; the frequencies must survive it.

    A position ``p`` in fp16 phase arithmetic is ``p * 2^-11`` radians off, so a
    cast frequency would make every computed row past a few thousand garbage.
    """
    pe = RelPositionalEncoding(D, max_len=ROWS)
    bits = pe.div_term_bits.clone()
    pe = pe.to(torch.float16)
    assert pe.div_term_bits.dtype == torch.int32 and torch.equal(pe.div_term_bits, bits)
    assert "div_term_bits" not in pe.state_dict()  # computed, never serialized
    got = pe.position_encoding(0, 3 * ROWS)
    assert got.dtype == torch.float16
    want = RelPositionalEncoding(D, max_len=4 * ROWS).to(torch.float16).position_encoding(0, 300)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-3)  # one fp16 ulp near 1


@pytest.mark.cuda
def test_encoder_decodes_past_the_table_as_with_a_larger_one(device):
    """End to end through the encoder: a 4-row table, an input three times longer."""
    torch.manual_seed(0)
    cfg = ConformerEncoderConfig(
        input_size=80, output_size=D, num_blocks=2, attention_heads=2, linear_units=128
    )
    encoder = ConformerEncoder(cfg).eval().to(device)
    xs = torch.randn(2, 4 * 12 + 6, 80, device=device)  # 12 frames after 4x subsampling
    lens = torch.tensor([xs.size(1), xs.size(1) - 20], device=device)
    with torch.no_grad():
        encoder.embed.pos_enc = RelPositionalEncoding(D, max_len=1000).to(device)
        want, _ = encoder(xs, lens)
        encoder.embed.pos_enc = RelPositionalEncoding(D, max_len=4).to(device)
        got, _ = encoder(xs, lens)
    assert got.shape == want.shape and got.size(1) == 12
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_a_declared_bound_keeps_the_plain_gather(monkeypatch):
    """A runtime that keeps whole histories declares its bound and pays a gather.

    Computing every row costs several times the gather, and the default
    streaming runtime never passes the table, so it must not pay for that.
    """
    pe = RelPositionalEncoding(D, max_len=ROWS)
    offsets = torch.tensor([0, 10, ROWS - 30])
    computed = pe.position_encoding(offsets, 30)
    pe.bound_offsets(ROWS)

    def no_formula(positions):
        raise AssertionError("a bounded per-stream lookup computed the sinusoid")

    monkeypatch.setattr(pe, "_sinusoid", no_formula)
    assert torch.equal(pe.position_encoding(offsets, 30), computed)
    monkeypatch.undo()

    pe.bound_offsets(None)  # no bound: every row is computed, past the table too
    far = torch.tensor([3 * ROWS])
    want = RelPositionalEncoding(D, max_len=4 * ROWS).position_encoding(far, 5)
    torch.testing.assert_close(pe.position_encoding(far, 5), want, rtol=0, atol=SIN_ATOL)


def test_a_bound_past_the_table_extends_the_table(monkeypatch):
    """A bound the table does not cover grows the table, so the gather stays exact.

    The gather clamps its index.  The paged runtime once declared a bound below
    what its streams could reach, and every position past the table was read as
    the table's last row: a wrong encoding, and no error.  A stream can cache
    8192 frames at the default block-table width; WeNet's table has 5000 rows.
    """
    pe = RelPositionalEncoding(D, max_len=ROWS)
    table = pe.pe
    held = table.clone()
    pe.bound_offsets(3 * ROWS + 5)
    assert pe.pe.size(1) == 3 * ROWS + 5 and pe.max_len == 3 * ROWS + 5
    assert torch.equal(pe.pe[:, :ROWS], held)  # the rows it had are untouched

    def no_formula(positions):
        raise AssertionError("a bounded per-stream lookup computed the sinusoid")

    monkeypatch.setattr(pe, "_sinusoid", no_formula)
    offsets = torch.tensor([0, ROWS - 3, 3 * ROWS])
    got = pe.position_encoding(offsets, 5)
    want = RelPositionalEncoding(D, max_len=4 * ROWS).position_encoding(offsets, 5)
    torch.testing.assert_close(got, want, rtol=0, atol=SIN_ATOL)
    # The replaced table stays allocated: a graph captured against it reads it
    # by address.
    assert any(t is table for t in pe._retired_tables)


# ---------------------------------------------------------------------------
# Under graph capture: the table a replay reads must outlive every request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_zipformer_rows_past_the_table_are_computed_not_stored(device):
    """Past its table, Zipformer's encoding matches a larger table and leaves its own alone."""
    small = CompactRelPositionalEncoding(16, max_len=ROWS).to(device)
    large = CompactRelPositionalEncoding(16, max_len=4 * ROWS).to(device)
    table, address = small.pe, small.pe.data_ptr()
    for size, left in ((ROWS // 2, 0), (2 * ROWS, 0), (30, ROWS)):
        x = torch.zeros(size, 1, 16, device=device)
        got = small(x, left)
        assert got.shape == (1, 2 * size + left - 1, 16)
        torch.testing.assert_close(got, large(x, left), rtol=0, atol=SIN_ATOL)
        assert small.pe is table and small.pe.data_ptr() == address


def _tiny(arch: str):
    """The ``test_layer_waist`` tiny config, for the two architectures with a growing table."""
    from oasr.models.paraformer.config import ParaformerModelConfig
    from oasr.models.zipformer.config import ZipformerEncoderConfig, ZipformerModelConfig

    configs = {
        "zipformer": ZipformerModelConfig(
            vocab_size=32,
            encoder=ZipformerEncoderConfig(
                feature_dim=80,
                downsampling_factor=(1, 2),
                encoder_dim=(64, 96),
                num_encoder_layers=(1, 1),
                query_head_dim=(8,),
                pos_head_dim=(4,),
                value_head_dim=(6,),
                num_heads=(4, 4),
                feedforward_dim=(64, 96),
                cnn_module_kernel=(15, 15),
                pos_dim=16,
                causal=False,
            ),
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
    }
    return get_model_entry(arch).model_cls.from_config(configs[arch])


@pytest.mark.cuda
@pytest.mark.parametrize(
    "arch,short,long",
    # Zipformer's first stack runs at half the feature rate and its table holds
    # 1000 frames; Paraformer's process-wide table grew in 512-frame steps.
    [("zipformer", 300, 2200), ("paraformer", 200, 700)],
)
def test_a_long_request_leaves_a_captured_encoder_intact(arch, short, long):
    """Capture a short shape, run one long request eagerly, replay the short shape.

    The replay must still equal the eager forward.  Before the fix the long
    request freed the table the capture read, and :func:`occupy_free_memory`
    makes the freed storage read as NaN rather than as whatever stale bytes it
    happened to keep.
    """
    torch.manual_seed(0)
    model = _tiny(arch).to(dtype=torch.float16).to("cuda").eval()

    def inputs(frames: int):
        feats = torch.randn(2, frames, 80, device="cuda", dtype=torch.float16)
        lens = torch.tensor([frames, frames - 17], dtype=torch.int32, device="cuda")
        return feats, lens

    feats, lens = inputs(short)
    with torch.no_grad():
        want = model.encode_offline(feats, lens)[0].clone()
        graph, (got, _) = capture_graph(lambda: model.encode_offline(feats, lens))
        model.encode_offline(*inputs(long))
        torch.cuda.synchronize()
        sentinels = occupy_free_memory()
        graph.replay()
        torch.cuda.synchronize()
    assert torch.isfinite(got).all(), "the replay read freed memory"
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert written_sentinels(sentinels) == 0, "the replay wrote into freed memory"
