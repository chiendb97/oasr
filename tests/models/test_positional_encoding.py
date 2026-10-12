# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer's relative positional encoding past its table.

WeNet sizes the sinusoid table at 5000 rows -- 200 s of audio at 40 ms frames --
and the table is a cache of a formula, not a property of the weights.  Reading
past it was a shape error offline and, in streaming, an out-of-bounds gather: a
device-side assert that took the CUDA context, and every stream on it, down once
a stream with bounded history ran long enough.  A position past the table is now
computed.  These pin that it is computed as a larger table would have held it,
and that a row the table does hold still comes from it.
"""

from __future__ import annotations

import pytest
import torch

from oasr.models.conformer import ConformerEncoder, ConformerEncoderConfig
from oasr.models.conformer.model import RelPositionalEncoding

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

    pe.bound_offsets(ROWS + 1)  # a bound past the table is no bound at all
    far = torch.tensor([3 * ROWS])
    want = RelPositionalEncoding(D, max_len=4 * ROWS).position_encoding(far, 5)
    torch.testing.assert_close(pe.position_encoding(far, 5), want, rtol=0, atol=SIN_ATOL)
