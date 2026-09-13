# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Op-level profiler for the non-GEMM 'glue' ops around the encoder.

Runs the engine offline + streaming (CUDA graphs OFF so ops are visible to
the torch profiler) and prints the aten ops ranked by self-CUDA time, plus a
roll-up into GEMM / elementwise / copy-index / norm / conv / other buckets.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import torch

from oasr.engine import ASREngine, EngineConfig


def _load(audio_dir: str, n: int):
    import torchaudio

    paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))[:n]
    wavs = []
    for p in paths:
        w, sr = torchaudio.load(p)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        wavs.append(w.squeeze(0).contiguous())
    return wavs


# aten op name -> coarse bucket
def _bucket(name: str) -> str:
    n = name.lower()
    if any(
        k in n
        for k in (
            "gemm",
            "cutlass",
            "addmm",
            "matmul",
            "bmm",
            "linear",
            "mm",
            "sm90",
            "sm80",
            "implicit",
            "conv",
        )
    ):
        if "conv" in n:
            return "conv"
        return "gemm"
    if any(k in n for k in ("layer_norm", "rms_norm", "batch_norm", "group_norm", "norm", "cmvn")):
        return "norm"
    if any(
        k in n
        for k in (
            "cat",
            "copy",
            "index",
            "gather",
            "scatter",
            "slice",
            "select",
            "stack",
            "narrow",
            "pad",
            "contiguous",
            "clone",
            "to_copy",
            "permute",
            "transpose",
            "reshape",
            "view",
            "expand",
            "masked_fill",
            "zero",
            "fill",
            "arange",
            "empty",
            "set_",
        )
    ):
        return "copy/index"
    if any(
        k in n
        for k in (
            "add",
            "mul",
            "sub",
            "div",
            "elementwise",
            "softmax",
            "exp",
            "rsqrt",
            "sigmoid",
            "silu",
            "glu",
            "where",
            "clamp",
            "neg",
        )
    ):
        return "elementwise"
    return "other"


def _profile(engine, run_fn, label: str, topk: int = 35):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        run_fn()
        torch.cuda.synchronize()

    evts = prof.key_averages()
    rows = []
    for e in evts:
        cuda_us = float(
            getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        )
        if cuda_us <= 0:
            continue
        rows.append((e.key, cuda_us, e.count))
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)

    bucket = defaultdict(float)
    for name, us, _ in rows:
        bucket[_bucket(name)] += us

    print(f"\n{'='*78}\n{label}  (total self-CUDA {total/1000:.2f} ms over profiled run)\n{'='*78}")
    print(f"{'bucket':<14}{'self-CUDA ms':>14}{'% total':>10}")
    for b in sorted(bucket, key=lambda k: -bucket[k]):
        print(f"{b:<14}{bucket[b]/1000:>14.3f}{100*bucket[b]/total:>9.1f}%")

    print(f"\n  top {topk} ops by self-CUDA:")
    print(f"  {'op':<48}{'ms':>10}{'%':>8}{'count':>9}{'bucket':>12}")
    for name, us, cnt in rows[:topk]:
        print(f"  {name[:46]:<48}{us/1000:>10.3f}{100*us/total:>7.1f}%{cnt:>9}{_bucket(name):>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR"))
    ap.add_argument("--audio-dir", default=os.environ.get("AUDIO_DIR"))
    ap.add_argument("--num-utterances", type=int, default=16)
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--mode", choices=["offline", "streaming", "both"], default="both")
    args = ap.parse_args()

    wavs = _load(args.audio_dir, args.num_utterances)
    print(f"loaded {len(wavs)} utts")

    if args.mode in ("offline", "both"):
        cfg = EngineConfig(
            ckpt_dir=args.ckpt_dir,
            device="cuda",
            dtype=torch.float16,
            service_mode="offline",
            decoder_type="ctc_cuda",
            max_batch_size=args.max_batch_size,
        )
        eng = ASREngine(cfg)
        eng.transcribe_offline(wavs[:4])  # warmup
        _profile(eng, lambda: eng.transcribe_offline(wavs), "OFFLINE")
        del eng
        torch.cuda.empty_cache()

    if args.mode in ("streaming", "both"):
        cfg = EngineConfig(
            ckpt_dir=args.ckpt_dir,
            device="cuda",
            dtype=torch.float16,
            service_mode="streaming",
            decoder_type="ctc_cuda",
            chunk_size=args.chunk_size,
            max_batch_size=args.max_batch_size,
            use_cuda_graphs=False,  # expose ops to the profiler
        )
        eng = ASREngine(cfg)
        eng.transcribe(wavs[:4])  # warmup
        _profile(eng, lambda: eng.transcribe(wavs), "STREAMING (cuda graphs OFF)")


if __name__ == "__main__":
    main()
