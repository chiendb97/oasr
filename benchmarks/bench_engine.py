#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""OASR ASR Engine Benchmark — offline and streaming transcription.

Measures Real-Time Factor (RTF), latency, and throughput for the ASR
inference engine across offline and streaming decoding modes.

Batch-size argument
-------------------
--max-batch-size   Encoder forward batch size — in streaming mode caps the
                   concurrent running pool in the ASREngine scheduler; in
                   offline mode is the GPU forward width of each pipeline
                   micro-batch.  The service runs one mode at a time.

Examples
--------
# Offline benchmark — sweep default batch sizes (1 / 4 / 8)
python benchmarks/bench_engine.py \\
    --ckpt-dir CKPT_DIR \\
    --audio-dir WAV_DIR

# Offline — explicit batch size
python benchmarks/bench_engine.py \\
    --ckpt-dir CKPT_DIR \\
    --audio-dir WAV_DIR \\
    --subroutines offline \\
    --max-batch-size MAX_BATCH_SIZE \\
    --num-utterances NUM_UTTERANCES

# Streaming — custom chunk + concurrency
python benchmarks/bench_engine.py \\
    --ckpt-dir CKPT_DIR \\
    --audio-dir WAV_DIR \\
    --subroutines streaming \\
    --chunk-size CHUNK_SIZE \\
    --max-batch-size MAX_BATCH_SIZE \\
    --num-utterances NUM_UTTERANCES

# All modes including WFST
python benchmarks/bench_engine.py \\
    --ckpt-dir CKPT_DIR \\
    --audio-dir WAV_DIR \\
    --wfst-path WFST_DIR \\
    --subroutines offline streaming offline_wfst streaming_wfst \\
    --num-utterances NUM_UTTERANCES

# Export results to CSV
python benchmarks/bench_engine.py \\
    --ckpt-dir CKPT_DIR \\
    --audio-dir WAV_DIR \\
    --output-path engine_results.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.routines.bench_utils import OutputWriter
from benchmarks.routines.engine import SUBROUTINES, _resolve_fst_file, _run_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OASR ASR Engine Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Required inputs ---
    parser.add_argument(
        "--ckpt-dir",
        required=True,
        metavar="DIR",
        help="WeNet checkpoint directory (contains final.pt, train.yaml, units.txt)",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        metavar="DIR",
        help="Directory containing .wav files to benchmark",
    )
    parser.add_argument(
        "--wfst-path",
        default=None,
        metavar="DIR|FILE",
        help="WFST directory (containing HLG.pt) or direct path to HLG.pt; "
             "required for *_wfst subroutines",
    )

    # --- Subroutine selection ---
    parser.add_argument(
        "--subroutines",
        nargs="+",
        default=["offline", "streaming"],
        choices=SUBROUTINES,
        metavar="MODE",
        help="Subroutines to run. Choices: %(choices)s  (default: offline streaming)",
    )

    # --- Batch / concurrency ---
    batch_group = parser.add_argument_group(
        "batch / concurrency",
        description="Control how many utterances are processed together.",
    )
    batch_group.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Encoder forward batch size — streaming concurrent pool cap "
             "and offline pipeline micro-batch width. "
             "Default: 32 for streaming, 4 for offline.",
    )

    # --- Streaming / chunking ---
    stream_group = parser.add_argument_group("streaming")
    stream_group.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Encoder output frames per streaming chunk "
             "(default: sweep 8 / 16 / 32 when unset)",
    )
    stream_group.add_argument(
        "--num-left-chunks",
        type=int,
        default=-1,
        metavar="N",
        help="Left-context chunks kept in attention cache; -1 = unlimited (default: -1)",
    )

    # --- Dataset ---
    data_group = parser.add_argument_group("dataset")
    data_group.add_argument(
        "--num-utterances",
        type=int,
        default=10,
        metavar="N",
        help="Number of .wav files per benchmark run (default: 10)",
    )

    # --- Precision ---
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model and cache precision (default: float16)",
    )

    # --- Timing ---
    parser.add_argument(
        "--num-iters",
        type=int,
        default=5,
        metavar="N",
        help="Number of timed measurement iterations (default: 5)",
    )

    # --- Output ---
    parser.add_argument(
        "--output-path",
        default=None,
        metavar="FILE",
        help="CSV file path for results (optional)",
    )

    parser.add_argument(
        "--cuda-graphs",
        choices=["on", "off"],
        default=None,
        help="Enable / disable CUDA Graph capture for the streaming encoder "
             "forward. Defaults to the engine config setting "
             "(``EngineConfig.use_cuda_graphs``).",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    fst_file = _resolve_fst_file(args.wfst_path)
    output = OutputWriter(output_path=args.output_path)
    output.write_header("OASR ASR Engine Benchmark")

    import torch
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # When the user does not override --max-batch-size, the sweep configs in
    # routines/engine.py drive the value per-config (offline: 1/4/8, streaming: 32).
    explicit_mbs = args.max_batch_size is not None

    for sub in args.subroutines:
        output.write_header(f"--- {sub} ---")
        is_streaming = sub.startswith("streaming")
        default_mbs = 32 if is_streaming else 4
        max_batch_size = args.max_batch_size if explicit_mbs else default_mbs
        _run_config(
            subroutine=sub,
            ckpt_dir=args.ckpt_dir,
            audio_dir=args.audio_dir,
            fst_file=fst_file,
            num_utterances=args.num_utterances,
            chunk_size=args.chunk_size or 16,
            num_left_chunks=args.num_left_chunks,
            max_batch_size=max_batch_size,
            num_iters=args.num_iters,
            dtype=dtype,
            output=output,
            use_cuda_graphs=(args.cuda_graphs == "on" if args.cuda_graphs else None),
        )

    output.finalize()


if __name__ == "__main__":
    main()
