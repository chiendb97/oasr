#!/usr/bin/env python3
"""OASR CTC Decoder Benchmark — OASR vs torchaudio CUCTCDecoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from benchmarks.routines.bench_utils import bench_fn

# (batch, seq_len, vocab_size, beam_size)
SHAPES = [
    (1, 200, 100, 10),
    (4, 200, 100, 10),
    (16, 200, 100, 10),
    (32, 200, 100, 10),
    (1, 200, 1000, 10),
    (4, 200, 1000, 10),
    (16, 200, 1000, 10),
    (32, 200, 1000, 10),
    (1, 200, 5000, 10),
    (4, 200, 5000, 10),
    (16, 200, 5000, 10),
    (16, 50, 1000, 10),
    (16, 100, 1000, 10),
    (16, 500, 1000, 10),
    (16, 200, 1000, 5),
    (16, 200, 1000, 20),
]

BLANK_THRESHOLD = 1.0
BLANK_ID = 0
WARMUP = 5
ITERS = 30

_COL_SHAPE = 26
_COL_TIME = 14
_COL_SPEEDUP = 10

_HEADER = (
    f"{'(B,T,V,beam)':>{_COL_SHAPE}}"
    f"  {'OASR':>{_COL_TIME}}"
    f"  {'torchaudio':>{_COL_TIME}}"
    f"  {'Speedup':>{_COL_SPEEDUP}}"
)
_SEP = "-" * len(_HEADER)
_TITLE = "OASR CTC Decoder Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _fmt_time(ms):
    return f"{ms:.4f}ms"


def _fmt_speedup(speedup):
    return f"{speedup:.2f}x"


def _fmt_na():
    return "N/A"


def _make_inputs(batch, seq_len, vocab_size):
    logits = torch.randn(batch, seq_len, vocab_size, device="cuda")
    log_prob = F.log_softmax(logits, dim=-1)
    seq_lengths = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
    return log_prob, seq_lengths


def _make_oasr_fn(log_prob, seq_lengths, beam_size, max_seq_len):
    from oasr.ctc_decode import ctc_beam_search_decode

    # Warm up once to trigger JIT compilation before timed runs.
    ctc_beam_search_decode(
        log_prob,
        seq_lengths,
        beam_size=beam_size,
        blank_id=BLANK_ID,
        blank_threshold=BLANK_THRESHOLD,
        max_seq_len=max_seq_len,
    )
    torch.cuda.synchronize()

    def fn():
        ctc_beam_search_decode(
            log_prob,
            seq_lengths,
            beam_size=beam_size,
            blank_id=BLANK_ID,
            blank_threshold=BLANK_THRESHOLD,
            max_seq_len=max_seq_len,
        )

    return fn


def _make_torchaudio_fn(log_prob, seq_lengths, vocab_size, beam_size):
    from torchaudio.models.decoder import CUCTCDecoder

    # Keep one extra sentinel token to avoid sporadic out-of-range ids produced
    # by torchaudio's CUDA decoder in high-beam settings.
    vocab_list = ["<blank>"] + [f"t{i}" for i in range(1, vocab_size + 1)]
    decoder = CUCTCDecoder(
        vocab_list=vocab_list,
        blank_id=BLANK_ID,
        beam_size=beam_size,
        nbest=beam_size,
        blank_skip_threshold=BLANK_THRESHOLD,
    )

    # Warm up once so setup allocations do not affect timed runs.
    decoder(log_prob, seq_lengths)
    torch.cuda.synchronize()

    def fn():
        decoder(log_prob, seq_lengths)

    return fn


def _row(shape_str, oasr_ms, torchaudio_ms):
    oasr_col = _fmt_time(oasr_ms) if oasr_ms is not None else _fmt_na()
    torchaudio_col = _fmt_time(torchaudio_ms) if torchaudio_ms is not None else _fmt_na()
    speedup_col = (
        _fmt_speedup(torchaudio_ms / oasr_ms)
        if oasr_ms is not None and torchaudio_ms is not None
        else _fmt_na()
    )
    return (
        f"{shape_str:>{_COL_SHAPE}}"
        f"  {oasr_col:>{_COL_TIME}}"
        f"  {torchaudio_col:>{_COL_TIME}}"
        f"  {speedup_col:>{_COL_SPEEDUP}}"
    )


def _run_shape(batch, seq_len, vocab_size, beam_size):
    log_prob, seq_lengths = _make_inputs(batch, seq_len, vocab_size)
    max_seq_len = seq_len

    oasr_ms = None
    torchaudio_ms = None

    try:
        oasr_fn = _make_oasr_fn(log_prob, seq_lengths, beam_size, max_seq_len)
        oasr_ms, _ = bench_fn(
            oasr_fn,
            dry_run_iters=WARMUP,
            num_iters=ITERS,
            use_cuda_events=True,
        )
    except Exception as err:
        print(f"[WARN] OASR failed for ({batch},{seq_len},{vocab_size},{beam_size}): {err}")

    try:
        torchaudio_fn = _make_torchaudio_fn(log_prob, seq_lengths, vocab_size, beam_size)
        torchaudio_ms, _ = bench_fn(
            torchaudio_fn,
            dry_run_iters=WARMUP,
            num_iters=ITERS,
            use_cuda_events=True,
        )
    except Exception as err:
        print(
            f"[WARN] torchaudio failed for ({batch},{seq_len},{vocab_size},{beam_size}): {err}"
        )

    return oasr_ms, torchaudio_ms


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(_TITLE)
    print(_TITLE_SEP)
    print(_HEADER)
    print(_SEP)

    for batch, seq_len, vocab_size, beam_size in SHAPES:
        oasr_ms, torchaudio_ms = _run_shape(batch, seq_len, vocab_size, beam_size)
        shape_str = f"({batch},{seq_len},{vocab_size},{beam_size})"
        print(_row(shape_str, oasr_ms, torchaudio_ms))

    print()


if __name__ == "__main__":
    main()
