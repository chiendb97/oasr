# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CTC beam search -- ``oasr/functionals/ctc_decode.py``.

Two subroutines, and between them four arms that used to live in two places:
``routines/ctc_decoder.py`` compared OASR against torchaudio and
``bench_ctc_decoder.py`` compared OASR's fused single-kernel step against its
own legacy multi-kernel pipeline.  Neither was a superset of the other, and only
the first could write a CSV row.

``OASR_CTC_FUSED`` selects the variant, and it is read when the decode runs --
so the environment has to be pinned around the timed call, not only around the
construction of the closure.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

from benchmarks.core import timing
from benchmarks.core.driver import params_of
from benchmarks.core.schema import WorkloadRow

SUBROUTINES = ["beam_search", "streaming"]

BLANK_ID = 0
BLANK_THRESHOLD = 1.0
STREAM_CHUNK_T = 16
STREAM_CHUNKS_PER_ITER = 2

DEFAULT_CONFIGS: Dict[str, list] = {
    "beam_search": [
        {"batch": b, "seq_len": t, "vocab_size": v, "beam_size": k}
        for b, t, v, k in (
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
        )
    ],
    "streaming": [
        {"vocab_size": v, "beam_size": k, "n_streams": n}
        for v, k, n in ((5000, 10, 1), (5000, 10, 8), (1000, 10, 8))
    ],
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=None, help="Frames")
    parser.add_argument("--vocab", dest="vocab_size", type=int, default=None, help="Vocab size")
    parser.add_argument("--beam", dest="beam_size", type=int, default=None, help="Beam size")
    parser.add_argument("--n-streams", type=int, default=None, help="Concurrent streams")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    if subroutine == "streaming":
        if all(getattr(args, k) is not None for k in ("vocab_size", "beam_size", "n_streams")):
            return [
                {
                    "vocab_size": args.vocab_size,
                    "beam_size": args.beam_size,
                    "n_streams": args.n_streams,
                }
            ]
        return DEFAULT_CONFIGS["streaming"]
    keys = ("batch", "seq_len", "vocab_size", "beam_size")
    if any(getattr(args, k) is not None for k in keys):
        defaults = {"batch": 16, "seq_len": 200, "vocab_size": 1000, "beam_size": 10}
        return [
            {k: getattr(args, k) if getattr(args, k) is not None else defaults[k] for k in keys}
        ]
    return DEFAULT_CONFIGS["beam_search"]


@contextmanager
def _forced_variant(use_fused: bool):
    """Pin ``OASR_CTC_FUSED`` for the duration of the block."""
    saved = os.environ.get("OASR_CTC_FUSED")
    os.environ["OASR_CTC_FUSED"] = "1" if use_fused else "0"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("OASR_CTC_FUSED", None)
        else:
            os.environ["OASR_CTC_FUSED"] = saved


def _inputs(batch: int, seq_len: int, vocab: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(batch, seq_len, vocab, device="cuda", generator=gen)
    return (
        F.log_softmax(logits, dim=-1),
        torch.full((batch,), seq_len, dtype=torch.int32, device="cuda"),
    )


def _offline_arms(cfg: dict) -> Dict[str, Callable[[], Any]]:
    from oasr.functionals.ctc_decode import ctc_beam_search_decode

    log_prob, lengths = _inputs(cfg["batch"], cfg["seq_len"], cfg["vocab_size"])
    beam, max_t = cfg["beam_size"], cfg["seq_len"]

    def oasr_arm(fused: bool):
        def fn():
            with _forced_variant(fused):
                ctc_beam_search_decode(
                    log_prob,
                    lengths,
                    beam_size=beam,
                    blank_id=BLANK_ID,
                    blank_threshold=BLANK_THRESHOLD,
                    max_seq_len=max_t,
                )

        return fn

    arms: Dict[str, Callable] = {"fused": oasr_arm(True), "legacy": oasr_arm(False)}
    try:
        from torchaudio.models.decoder import CUCTCDecoder

        vocab_list = ["<blank>"] + [f"t{i}" for i in range(1, cfg["vocab_size"] + 1)]
        decoder = CUCTCDecoder(
            vocab_list=vocab_list,
            blank_id=BLANK_ID,
            beam_size=beam,
            nbest=beam,
            blank_skip_threshold=BLANK_THRESHOLD,
        )
        arms["torchaudio"] = lambda: decoder(log_prob, lengths)
    except Exception:
        # torchaudio's CUDA decoder aborts the process on some SM versions;
        # its absence is a missing column, not a failed run.
        pass
    return arms


def _streaming_arms(cfg: dict) -> Dict[str, Callable[[], Any]]:
    from oasr.functionals.ctc_decode import GpuDecoderConfig, GpuStreamingDecoder

    vocab, beam, n_streams = cfg["vocab_size"], cfg["beam_size"], cfg["n_streams"]
    chunks = [_inputs(1, STREAM_CHUNK_T, vocab, seed=i)[0] for i in range(STREAM_CHUNKS_PER_ITER)]

    def arm(fused: bool):
        with _forced_variant(fused):
            dec = GpuStreamingDecoder(
                GpuDecoderConfig(
                    beam_size=beam,
                    blank_id=BLANK_ID,
                    blank_threshold=BLANK_THRESHOLD,
                    max_seq_len=200,
                )
            )
            states = [dec.create_state(1, vocab) for _ in range(n_streams)]

        def fn():
            with _forced_variant(fused):
                for state in states:
                    for chunk in chunks:
                        dec.decode_chunk(chunk, state=state)

        return fn

    return {"fused": arm(True), "legacy": arm(False)}


def run(args: argparse.Namespace, reporter, repro_command: str = "") -> None:
    subroutine = args.subroutine or SUBROUTINES[0]
    ref = args.ref_backend or "fused"

    for cfg in resolve_configs(args, subroutine):
        if subroutine == "streaming":
            arms = _streaming_arms(cfg)
            frames = cfg["n_streams"] * STREAM_CHUNKS_PER_ITER * STREAM_CHUNK_T
            shape = f"(V={cfg['vocab_size']},beam={cfg['beam_size']},streams={cfg['n_streams']})"
        else:
            arms = _offline_arms(cfg)
            frames = cfg["batch"] * cfg["seq_len"]
            shape = (
                f"(B={cfg['batch']},T={cfg['seq_len']},"
                f"V={cfg['vocab_size']},beam={cfg['beam_size']})"
            )

        wanted = args.backends or list(arms)
        selected = [b for b in wanted if b in arms]
        for missing in [b for b in wanted if b not in arms]:
            reporter.warn(f"backend '{missing}' unavailable for ctc_decoder/{subroutine}")
        if not selected:
            continue

        measured = timing.interleaved_rounds(
            {b: arms[b] for b in selected},
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            timer=args.timer,
            flush=not args.no_flush,
        )
        ref_ms = measured[ref].median_ms if ref in measured else None
        for backend in selected:
            sample = measured[backend]
            reporter.workload_row(
                WorkloadRow(
                    harness="decoder",
                    subroutine=subroutine,
                    backend=backend,
                    ref_backend=ref if ref_ms else "",
                    speedup_vs_ref=(ref_ms / sample.median_ms if ref_ms else None),
                    shape=shape,
                    params=params_of(cfg),
                    decode_options=f"blank_id={BLANK_ID};blank_threshold={BLANK_THRESHOLD}",
                    frames=frames,
                    median_ms=sample.median_ms,
                    std_ms=sample.std_ms,
                    iters=sample.iters,
                    wall_s=sample.median_ms / 1000.0,
                    repro_command=repro_command,
                )
            )
