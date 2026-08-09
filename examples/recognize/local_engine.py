#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transcribe with the Python API — no server, no network.

Every other example here talks to a running ``oasr-server`` over HTTP or gRPC,
which left the API the README actually documents (``ASREngine`` /
``EngineConfig`` / ``transcribe_offline``) undemonstrated.  This is that path:
one process, one engine, files in, transcripts out.

Reach for it when OASR is a *library* inside your own program — a batch job, a
notebook, an evaluation harness.  Reach for the server when something else has
to call you, or when you want many clients sharing one GPU.

Dependencies::

    pip install -e ".[audio]"       # soundfile, for reading the audio files

Usage::

    python examples/recognize/local_engine.py --ckpt-dir /path/to/ckpt audio.mp3
    python examples/recognize/local_engine.py --ckpt-dir /path/to/ckpt --words audio.mp3
    python examples/recognize/local_engine.py --ckpt-dir /path/to/whisper \\
        --task translate --language fr entretien.m4a

Equivalent one-liner once installed: ``oasr transcribe --ckpt-dir DIR audio.mp3``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List


def load_audio(path: Path, sample_rate: int):
    """Read one file as a mono waveform at ``sample_rate``.

    The engine is **waveform-only** and accepts exactly one rate — its own, from
    the checkpoint's feature spec.  It does not resample and it ignores a
    request's declared rate, so converting here is not optional: audio at
    another rate decodes at the wrong speed, confidently and silently.  (The
    server does this conversion for you; a library caller owns it.)
    """
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1)))
    if sr != sample_rate:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", nargs="+", type=Path, help="audio file(s)")
    p.add_argument("--ckpt-dir", required=True, help="checkpoint directory or HF repo id")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="compute dtype (default: bfloat16)",
    )
    p.add_argument("--decode-method", help="e.g. ctc_aed_rescoring on a U2++ hybrid")
    p.add_argument("--language", help='ISO-639 tag, e.g. "fr" (Whisper-family models)')
    p.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        help="Whisper-family models only; the checkpoint's own task by default",
    )
    p.add_argument(
        "--words",
        action="store_true",
        help=(
            "per-word times and confidences.  A decode family that cannot align "
            "refuses the request rather than returning nothing"
        ),
    )
    args = p.parse_args(argv)

    import torch

    from oasr.engine import ASREngine, DecodingOptions, EngineConfig

    cfg = EngineConfig(
        ckpt_dir=args.ckpt_dir,
        service_mode="offline",  # batched, and strictly faster than streaming here
        device=args.device,
        # ``EngineConfig.dtype`` is a real ``torch.dtype``, not its name — a
        # string reaches ``model.to(dtype=...)`` and raises there.
        dtype=getattr(torch, args.dtype),
        decode_method=args.decode_method,
    )
    t0 = time.perf_counter()
    engine = ASREngine(cfg)
    print(
        f"loaded {engine.decode_method} engine in {time.perf_counter() - t0:.1f}s", file=sys.stderr
    )

    # Per-request options; `None` everywhere keeps the checkpoint's defaults.
    decoding = None
    if args.task or args.language or args.words:
        decoding = DecodingOptions(
            task=args.task, language=args.language, word_timestamps=args.words
        )

    sample_rate = cfg.feature_config.sample_rate
    audios = [load_audio(path, sample_rate) for path in args.audio]

    # One call, all files: the offline executor batches them into length-bucketed
    # micro-batches, which is where OASR's throughput comes from.  Feeding files
    # one at a time would leave most of the GPU idle.
    t0 = time.perf_counter()
    # ``transcribe_offline`` returns strings; ``transcribe_outputs`` returns the
    # whole ``RequestOutput``, which is where ``words`` / ``confidence`` /
    # ``timestamps`` / ``finish_reason`` live.
    outputs = engine.transcribe_outputs(audios, streaming=False, decoding=decoding)
    elapsed = time.perf_counter() - t0

    total_s = sum(len(a) for a in audios) / sample_rate
    for path, out in zip(args.audio, outputs):
        print(f"{path.name}: {out.text}")
        for w in out.words or ():
            print(f"    {w.start:7.2f} - {w.end:7.2f}  {w.confidence:.3f}  {w.word}")
    print(
        f"\n{len(audios)} file(s), {total_s:.1f}s of audio in {elapsed:.2f}s "
        f"(RTFx {total_s / max(elapsed, 1e-9):.1f})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
