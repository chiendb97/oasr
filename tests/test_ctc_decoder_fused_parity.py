# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""A/B parity tests: fused single-kernel CTC beam-search step vs legacy pipeline.

The fused step (``include/oasr/ctc_decoder.cuh``, ``fused`` namespace) replaces
the 5-kernel prob-matrix/merge/top-k pipeline for ``beam <= 32``.  These tests
decode identical inputs through both compiled module variants (the legacy one
is selected with ``OASR_CTC_FUSED=0``, which builds with
``-DOASR_CTC_DISABLE_FUSED``) and require matching beams.

Comparison is permutation-tolerant: beams with bit-identical scores may be
ordered differently by the two top-k implementations (the legacy radix sort
and the fused (score, id) composite key break exact float ties differently),
so each batch row is compared as a multiset of token sequences plus sorted
scores.

Marked ``slow``: the first run JIT-compiles a second decoder module.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from oasr.functionals.ctc_decode import (
    GpuDecoderConfig,
    GpuStreamingDecoder,
    ctc_beam_search_decode,
)

pytestmark = pytest.mark.slow

SCORE_FLOOR = -1.0e30  # beams below this are NEG_INF filler


def _make_lp(batch, seq, vocab, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, device="cuda", generator=gen)
    logits[:, :, 0] += 2.0  # blank-heavy frames, ASR-like
    return F.log_softmax(logits * 2.0, dim=-1)


def _row_repr(tokens_row, scores_row):
    """Order-insensitive representation of one batch row's live beams."""
    live = [
        (tuple(tokens_row[k]), round(scores_row[k].item(), 3))
        for k in range(len(tokens_row))
        if scores_row[k].item() > SCORE_FLOOR
    ]
    return sorted(live)


def _assert_results_match(res_a, res_b, ctx):
    assert res_a.lengths.shape == res_b.lengths.shape, ctx
    batch = res_a.lengths.shape[0]
    for b in range(batch):
        row_a = _row_repr(res_a.tokens[b], res_a.scores[b])
        row_b = _row_repr(res_b.tokens[b], res_b.scores[b])
        assert row_a == row_b, f"{ctx} batch={b}:\n fused={row_a}\nlegacy={row_b}"
        sa = sorted(x.item() for x in res_a.scores[b] if x.item() > SCORE_FLOOR)
        sb = sorted(x.item() for x in res_b.scores[b] if x.item() > SCORE_FLOOR)
        assert sa == pytest.approx(sb, abs=1e-4), f"{ctx} batch={b} scores"


class _ForcedVariant:
    """Context manager that pins the decoder module variant via OASR_CTC_FUSED."""

    def __init__(self, use_fused: bool):
        self._value = "1" if use_fused else "0"
        self._saved = None

    def __enter__(self):
        self._saved = os.environ.get("OASR_CTC_FUSED")
        os.environ["OASR_CTC_FUSED"] = self._value
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            os.environ.pop("OASR_CTC_FUSED", None)
        else:
            os.environ["OASR_CTC_FUSED"] = self._saved
        return False


# Paged cases use short sequences: the legacy paged pipeline has a pre-existing
# free-pool race that needs page-recycling pressure (long sequences, large
# batch) to manifest; small shapes keep the legacy reference deterministic.
OFFLINE_CASES = [
    # (batch, seq, vocab, beam, blank_threshold, paged)
    (2, 80, 1000, 10, 1.0, False),
    (2, 80, 1000, 10, 0.95, False),
    (4, 200, 5000, 10, 0.95, False),
    (3, 64, 512, 16, 1.0, False),
    (1, 50, 100, 5, 1.0, False),
    (2, 60, 5000, 4, 0.95, False),
    (3, 100, 1000, 20, 1.0, False),
    (2, 48, 1000, 10, 1.0, True),
    (2, 48, 1000, 10, 0.95, True),
    (3, 40, 512, 16, 1.0, True),
    (1, 64, 5000, 8, 0.95, True),
]


class TestFusedParityOffline:
    @pytest.mark.parametrize("case", OFFLINE_CASES)
    def test_offline_matches_legacy(self, device, case):
        batch, seq, vocab, beam, thresh, paged = case
        lp = _make_lp(batch, seq, vocab, seed=hash(case) % (2**31))
        seq_lengths = torch.full((batch,), seq, dtype=torch.int32, device=device)
        if batch > 1:
            seq_lengths[1] = max(2, seq // 2)

        kwargs = {
            "beam_size": beam,
            "blank_id": 0,
            "blank_threshold": thresh,
            "max_seq_len": seq,
            "use_paged_memory": paged,
        }
        with _ForcedVariant(use_fused=True):
            res_fused = ctc_beam_search_decode(lp, seq_lengths, **kwargs)
        with _ForcedVariant(use_fused=False):
            res_legacy = ctc_beam_search_decode(lp, seq_lengths, **kwargs)
        _assert_results_match(res_fused, res_legacy, f"offline {case}")


STREAM_CASES = [
    # (vocab, beam, blank_threshold, paged, use_cuda_graphs)
    (1000, 10, 1.0, False, False),
    (1000, 10, 0.95, False, True),
    (5000, 10, 0.95, False, True),
    (512, 16, 1.0, True, False),
    (1000, 8, 0.95, True, True),
]


class TestFusedParityStreaming:
    @pytest.mark.parametrize("case", STREAM_CASES)
    def test_streaming_matches_legacy(self, device, case):
        vocab, beam, thresh, paged, graphs = case
        chunk_t, n_chunks = 16, 5
        chunks = [
            _make_lp(1, chunk_t, vocab, seed=(hash(case) + c) % (2**31)) for c in range(n_chunks)
        ]

        results = {}
        for use_fused in (True, False):
            with _ForcedVariant(use_fused=use_fused):
                cfg = GpuDecoderConfig(
                    beam_size=beam,
                    blank_id=0,
                    blank_threshold=thresh,
                    max_seq_len=128,
                    use_paged_memory=paged,
                )
                dec = GpuStreamingDecoder(cfg, use_cuda_graphs=graphs)
                dec.init_stream(1, vocab)
                for chunk in chunks:
                    dec.decode_chunk(chunk)
                results[use_fused] = dec.finalize_stream()
        _assert_results_match(results[True], results[False], f"streaming {case}")


class TestLargeBeamFallback:
    def test_beam_above_fused_cap_uses_legacy_pipeline(self, device):
        """beam > 32 falls back to the legacy kernels inside the default build."""
        lp = _make_lp(2, 40, 500, seed=11)
        seq_lengths = torch.full((2,), 40, dtype=torch.int32, device=device)
        flat = ctc_beam_search_decode(
            lp, seq_lengths, beam_size=40, blank_id=0, blank_threshold=1.0, max_seq_len=40
        )
        paged = ctc_beam_search_decode(
            lp,
            seq_lengths,
            beam_size=40,
            blank_id=0,
            blank_threshold=1.0,
            max_seq_len=40,
            use_paged_memory=True,
        )
        _assert_results_match(flat, paged, "beam=40 flat-vs-paged")
