#!/usr/bin/env python3
"""OASR CTC Decoder Benchmark — fused vs legacy step pipeline (and torchaudio).

By default compares the fused single-kernel beam-search step (the production
path for beam <= 32) against OASR's legacy multi-kernel pipeline, for both
offline batch decode and chunked streaming decode.

torchaudio's CUCTCDecoder comparison is opt-in via ``--torchaudio``: its CUDA
decoder hard-crashes (illegal address + core dump) on SM120 GPUs, taking the
whole benchmark process down.

Usage:
  python benchmarks/bench_ctc_decoder.py                 # fused vs legacy
  python benchmarks/bench_ctc_decoder.py --no-legacy     # fused only
  python benchmarks/bench_ctc_decoder.py --torchaudio    # also CUCTCDecoder
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from benchmarks.routines.bench_utils import bench_fn

# (batch, seq_len, vocab_size, beam_size)
OFFLINE_SHAPES = [
    (1, 200, 100, 10),
    (16, 200, 100, 10),
    (1, 200, 1000, 10),
    (16, 200, 1000, 10),
    (1, 200, 5000, 10),
    (4, 200, 5000, 10),
    (16, 200, 5000, 10),
    (64, 200, 5000, 10),
    (16, 50, 1000, 10),
    (16, 500, 1000, 10),
    (16, 200, 5000, 4),
    (16, 200, 5000, 20),
]

# (vocab_size, beam_size, n_streams)
STREAMING_SHAPES = [
    (5000, 10, 1),
    (5000, 10, 8),
    (1000, 10, 8),
]
STREAM_CHUNK_T = 16
STREAM_CHUNKS_PER_ITER = 2

BLANK_THRESHOLD = 1.0
BLANK_ID = 0
WARMUP = 5
ITERS = 30

_COL_SHAPE = 26
_COL_TIME = 14
_COL_SPEEDUP = 10


def _fmt_time(ms):
    return f"{ms:.4f}ms" if ms is not None else "N/A"


def _fmt_speedup(base_ms, ms):
    if base_ms is None or ms is None:
        return "N/A"
    return f"{ms / base_ms:.2f}x"


def _make_inputs(batch, seq_len, vocab_size, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(batch, seq_len, vocab_size, device="cuda", generator=gen)
    log_prob = F.log_softmax(logits, dim=-1)
    seq_lengths = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
    return log_prob, seq_lengths


class _ForcedVariant:
    """Pin the OASR decoder module variant (fused / legacy) via OASR_CTC_FUSED."""

    def __init__(self, use_fused):
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


def _bench_oasr_offline(log_prob, seq_lengths, beam_size, max_seq_len, use_fused):
    from oasr.functionals.ctc_decode import ctc_beam_search_decode

    with _ForcedVariant(use_fused):

        def fn():
            ctc_beam_search_decode(
                log_prob,
                seq_lengths,
                beam_size=beam_size,
                blank_id=BLANK_ID,
                blank_threshold=BLANK_THRESHOLD,
                max_seq_len=max_seq_len,
            )

        fn()
        torch.cuda.synchronize()
        ms, _ = bench_fn(fn, dry_run_iters=WARMUP, num_iters=ITERS, use_cuda_events=True)
    return ms


def _bench_oasr_streaming(vocab_size, beam_size, n_streams, use_fused):
    from oasr.functionals.ctc_decode import GpuDecoderConfig, GpuStreamingDecoder

    with _ForcedVariant(use_fused):
        cfg = GpuDecoderConfig(
            beam_size=beam_size,
            blank_id=BLANK_ID,
            blank_threshold=BLANK_THRESHOLD,
            max_seq_len=200,
        )
        dec = GpuStreamingDecoder(cfg)
        states = [dec.create_state(1, vocab_size) for _ in range(n_streams)]
        chunks = [
            _make_inputs(1, STREAM_CHUNK_T, vocab_size, seed=i)[0]
            for i in range(STREAM_CHUNKS_PER_ITER)
        ]

        def fn():
            for state in states:
                for chunk in chunks:
                    dec.decode_chunk(chunk, state=state)

        fn()
        torch.cuda.synchronize()
        ms, _ = bench_fn(fn, dry_run_iters=3, num_iters=ITERS, use_cuda_events=True)
    frames = n_streams * STREAM_CHUNKS_PER_ITER * STREAM_CHUNK_T
    return 1000.0 * ms / frames  # us per decoded frame


def _bench_torchaudio(log_prob, seq_lengths, vocab_size, beam_size):
    from torchaudio.models.decoder import CUCTCDecoder

    vocab_list = ["<blank>"] + [f"t{i}" for i in range(1, vocab_size + 1)]
    decoder = CUCTCDecoder(
        vocab_list=vocab_list,
        blank_id=BLANK_ID,
        beam_size=beam_size,
        nbest=beam_size,
        blank_skip_threshold=BLANK_THRESHOLD,
    )
    decoder(log_prob, seq_lengths)
    torch.cuda.synchronize()

    def fn():
        decoder(log_prob, seq_lengths)

    ms, _ = bench_fn(fn, dry_run_iters=WARMUP, num_iters=ITERS, use_cuda_events=True)
    return ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="skip the legacy OASR pipeline column (avoids a second JIT compile)",
    )
    parser.add_argument(
        "--torchaudio",
        action="store_true",
        help="also benchmark torchaudio CUCTCDecoder (crashes on SM120 — opt-in)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    cols = ["OASR fused"]
    if not args.no_legacy:
        cols.append("OASR legacy")
    if args.torchaudio:
        cols.append("torchaudio")

    header = f"{'(B,T,V,beam)':>{_COL_SHAPE}}" + "".join(f"  {c:>{_COL_TIME}}" for c in cols)
    if len(cols) > 1:
        header += f"  {'Speedup':>{_COL_SPEEDUP}}"
    print("OASR CTC Decoder Benchmark — offline batch decode")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for batch, seq_len, vocab_size, beam_size in OFFLINE_SHAPES:
        log_prob, seq_lengths = _make_inputs(batch, seq_len, vocab_size)
        fused_ms = _bench_oasr_offline(log_prob, seq_lengths, beam_size, seq_len, True)
        row = [fused_ms]
        if not args.no_legacy:
            row.append(_bench_oasr_offline(log_prob, seq_lengths, beam_size, seq_len, False))
        if args.torchaudio:
            try:
                row.append(_bench_torchaudio(log_prob, seq_lengths, vocab_size, beam_size))
            except Exception as err:  # pragma: no cover - external decoder
                print(f"[WARN] torchaudio failed: {err}")
                row.append(None)
        line = f"{f'({batch},{seq_len},{vocab_size},{beam_size})':>{_COL_SHAPE}}"
        line += "".join(f"  {_fmt_time(ms):>{_COL_TIME}}" for ms in row)
        if len(row) > 1:
            line += f"  {_fmt_speedup(row[0], row[1]):>{_COL_SPEEDUP}}"
        print(line)

    print()
    header2 = f"{'(V,beam,streams)':>{_COL_SHAPE}}" + "".join(
        f"  {c:>{_COL_TIME}}" for c in cols if not c.startswith("torchaudio")
    )
    if not args.no_legacy:
        header2 += f"  {'Speedup':>{_COL_SPEEDUP}}"
    print("Streaming chunked decode (us / decoded frame, batch=1 states)")
    print("=" * len(header2))
    print(header2)
    print("-" * len(header2))
    for vocab_size, beam_size, n_streams in STREAMING_SHAPES:
        fused_us = _bench_oasr_streaming(vocab_size, beam_size, n_streams, True)
        line = f"{f'({vocab_size},{beam_size},{n_streams})':>{_COL_SHAPE}}"
        line += f"  {f'{fused_us:.2f}us':>{_COL_TIME}}"
        if not args.no_legacy:
            legacy_us = _bench_oasr_streaming(vocab_size, beam_size, n_streams, False)
            line += f"  {f'{legacy_us:.2f}us':>{_COL_TIME}}"
            line += f"  {f'{legacy_us / fused_us:.2f}x':>{_COL_SPEEDUP}}"
        print(line)
    print()


if __name__ == "__main__":
    main()
