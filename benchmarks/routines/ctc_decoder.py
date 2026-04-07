"""CTC decoder benchmark routine (OASR vs torchaudio CUCTCDecoder)."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F

from benchmarks.routines.bench_utils import BenchResult, OutputWriter, bench_fn

SUBROUTINES = ["beam_search"]

# (batch, seq_len, vocab_size, beam_size)
DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "beam_search": [
        {"batch": 1, "seq_len": 200, "vocab_size": 100, "beam_size": 10},
        {"batch": 4, "seq_len": 200, "vocab_size": 100, "beam_size": 10},
        {"batch": 16, "seq_len": 200, "vocab_size": 100, "beam_size": 10},
        {"batch": 32, "seq_len": 200, "vocab_size": 100, "beam_size": 10},
        {"batch": 1, "seq_len": 200, "vocab_size": 1000, "beam_size": 10},
        {"batch": 4, "seq_len": 200, "vocab_size": 1000, "beam_size": 10},
        {"batch": 16, "seq_len": 200, "vocab_size": 1000, "beam_size": 10},
        {"batch": 32, "seq_len": 200, "vocab_size": 1000, "beam_size": 10},
        {"batch": 1, "seq_len": 200, "vocab_size": 5000, "beam_size": 10},
        {"batch": 4, "seq_len": 200, "vocab_size": 5000, "beam_size": 10},
        {"batch": 16, "seq_len": 200, "vocab_size": 5000, "beam_size": 10},
        {"batch": 16, "seq_len": 50, "vocab_size": 1000, "beam_size": 10},
        {"batch": 16, "seq_len": 100, "vocab_size": 1000, "beam_size": 10},
        {"batch": 16, "seq_len": 500, "vocab_size": 1000, "beam_size": 10},
        {"batch": 16, "seq_len": 200, "vocab_size": 1000, "beam_size": 5},
        {"batch": 16, "seq_len": 200, "vocab_size": 1000, "beam_size": 20},
        {"batch": 16, "seq_len": 200, "vocab_size": 1000, "beam_size": 50},
    ]
}

BLANK_ID = 0
BLANK_THRESHOLD = 1.0


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, dest="seq_len", help="Sequence length")
    parser.add_argument("--vocab", type=int, default=None, dest="vocab_size", help="Vocabulary size")
    parser.add_argument("--beam", type=int, default=None, dest="beam_size", help="Beam size")


def _make_inputs(batch: int, seq_len: int, vocab_size: int):
    logits = torch.randn(batch, seq_len, vocab_size, device="cuda")
    log_prob = F.log_softmax(logits, dim=-1)
    seq_lengths = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
    return log_prob, seq_lengths


def _make_oasr_fn(log_prob, seq_lengths, beam_size, max_seq_len):
    from oasr.ctc_decode import ctc_beam_search_decode

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

    return fn


def _resolve_configs(args: argparse.Namespace, subroutine: str) -> list[dict[str, Any]]:
    batch = getattr(args, "batch", None)
    seq_len = getattr(args, "seq_len", None)
    vocab_size = getattr(args, "vocab_size", None)
    beam_size = getattr(args, "beam_size", None)
    if any(v is not None for v in [batch, seq_len, vocab_size, beam_size]):
        return [{
            "batch": batch if batch is not None else 16,
            "seq_len": seq_len if seq_len is not None else 200,
            "vocab_size": vocab_size if vocab_size is not None else 1000,
            "beam_size": beam_size if beam_size is not None else 10,
        }]
    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["beam_search"])


def _setup_for_config(subroutine: str, cfg: dict[str, Any], dtype=None):
    _ = subroutine, dtype
    log_prob, seq_lengths = _make_inputs(cfg["batch"], cfg["seq_len"], cfg["vocab_size"])
    oasr_fn = _make_oasr_fn(log_prob, seq_lengths, cfg["beam_size"], cfg["seq_len"])
    torchaudio_fn = _make_torchaudio_fn(log_prob, seq_lengths, cfg["vocab_size"], cfg["beam_size"])
    return oasr_fn, torchaudio_fn


def get_fn_map(subroutine, oasr_fn, torch_fn):
    _ = subroutine
    return {"oasr": oasr_fn, "torchaudio": torch_fn}


def _shape_str(cfg: dict[str, Any]) -> str:
    return f"(B={cfg['batch']},T={cfg['seq_len']},V={cfg['vocab_size']},beam={cfg['beam_size']})"


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "beam_search")
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)
    backends = getattr(args, "backends", None) or ["oasr", "torchaudio"]

    configs = _resolve_configs(args, subroutine)
    for cfg in configs:
        shape_str = _shape_str(cfg)
        log_prob, seq_lengths = _make_inputs(cfg["batch"], cfg["seq_len"], cfg["vocab_size"])

        fn_map = {}
        try:
            fn_map["oasr"] = _make_oasr_fn(log_prob, seq_lengths, cfg["beam_size"], cfg["seq_len"])
        except Exception as err:
            print(f"[WARNING] Failed to setup oasr for {shape_str}: {err}")

        try:
            fn_map["torchaudio"] = _make_torchaudio_fn(
                log_prob, seq_lengths, cfg["vocab_size"], cfg["beam_size"]
            )
        except Exception as err:
            print(f"[WARNING] Failed to setup torchaudio for {shape_str}: {err}")

        for backend in backends:
            if backend not in fn_map:
                continue
            try:
                median_ms, std_ms = bench_fn(
                    fn_map[backend],
                    dry_run_iters=dry_run_iters,
                    num_iters=num_iters,
                    use_cuda_events=use_cuda_events,
                )
                output.write_result(BenchResult(
                    routine="ctc_decoder",
                    subroutine=subroutine,
                    backend=backend,
                    shape=shape_str,
                    dtype="float32",
                    median_ms=median_ms,
                    std_ms=std_ms,
                ))
            except Exception as err:
                print(f"[WARNING] {backend} failed for {shape_str}: {err}")


def run_standalone() -> None:
    """Backwards-compatible standalone entry."""
    parser = argparse.ArgumentParser(description="CTC decoder benchmark")
    parse_args(parser)
    parser.add_argument("--backends", nargs="+", default=["oasr", "torchaudio"])
    parser.add_argument("--dry_run_iters", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--use_cuda_events", action="store_true")
    args = parser.parse_args()
    run_test(args, OutputWriter())
