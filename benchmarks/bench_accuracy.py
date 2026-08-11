#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""OASR Accuracy Benchmark — WER/CER against ground truth, with speed alongside.

The companion to ``bench_engine.py``: that one answers "how fast", this one
answers "how correct", and a row here carries both so the trade is visible in
one table.  Every accuracy-affecting change before this had to hand-build a
transcript comparison and throw it away.

Manifests, not audio
--------------------
A manifest is JSON Lines, one utterance per line::

    {"id": "LJ001-0001", "audio": "LJ001-0001.wav", "text": "printing in the ..."}

``audio`` is resolved against ``--audio-root``, so a manifest is a few tens of
KB of text that anyone can check in while the corpus stays on disk.  Build one
for a corpus you have with ``--build-manifest``.

Examples
--------
    # WER on the shipped 200-utterance LJSpeech subset
    python benchmarks/bench_accuracy.py \\
        --ckpt-dir  $CKPT_DIR \\
        --manifest  benchmarks/manifests/ljspeech_200.jsonl \\
        --audio-root $AUDIO_DIR

    # Sweep decode methods, write a CSV
    python benchmarks/bench_accuracy.py --ckpt-dir $CKPT_DIR \\
        --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root $AUDIO_DIR \\
        --decode-method ctc ctc_aed_rescoring --dtype float16 float32 \\
        --output-path accuracy.csv

    # Offline vs. streaming on one table, and the chunk-size trade within streaming
    python benchmarks/bench_accuracy.py --ckpt-dir $CKPT_DIR \\
        --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root $AUDIO_DIR \\
        --service-mode offline streaming --chunk-size 8 16 32

    # CER for a Chinese model
    python benchmarks/bench_accuracy.py --ckpt-dir $OASR_PARAFORMER_CKPT \\
        --manifest my_zh.jsonl --audio-root /data/zh --metric cer --normalizer basic

    # Turn a corpus into a manifest (LJSpeech metadata.csv, or a dir of .txt/.lab)
    python benchmarks/bench_accuracy.py --build-manifest out.jsonl \\
        --audio-root $AUDIO_DIR --transcripts $AUDIO_DIR/../metadata.csv --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasr.testing.accuracy import Entry, load_audio, load_manifest, transcribe  # noqa: E402
from oasr.testing.wer import Result, compute, normalizer  # noqa: E402

# One row per (model, decode_method, dtype, batch size) — the shape the review
# asked for, and the shape a published accuracy-and-speed table needs.
CSV_COLUMNS = [
    "manifest",
    "ckpt",
    "architecture",
    "decode_method",
    "service_mode",
    "chunk_size",
    "dtype",
    "max_batch_size",
    "metric",
    "normalizer",
    "utterances",
    "audio_seconds",
    "error_rate_pct",
    "substitutions",
    "deletions",
    "insertions",
    "ref_units",
    "wall_seconds",
    "rtfx",
    "utt_per_second",
    "latency_p50_ms",
    "latency_p99_ms",
]


@dataclass
class Row:
    manifest: str
    ckpt: str
    architecture: str
    decode_method: str
    service_mode: str
    chunk_size: int
    dtype: str
    max_batch_size: int
    metric: str
    normalizer: str
    utterances: int
    audio_seconds: float
    error_rate_pct: float
    substitutions: int
    deletions: int
    insertions: int
    ref_units: int
    wall_seconds: float
    rtfx: float
    utt_per_second: float
    latency_p50_ms: float
    latency_p99_ms: float
    #: Not a CSV column — carried so the caller can print the worst utterances.
    result: Optional[Result] = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

#: Bracket *characters*, not their contents — see build_manifest().
_BRACKETS = re.compile(r"[()\[\]]")


def build_manifest(
    out_path: Path,
    audio_root: Path,
    transcripts: Optional[Path],
    limit: Optional[int],
    strip_brackets: bool = True,
) -> None:
    """Write a manifest from a corpus on disk.

    Two layouts are understood, which covers most released corpora:

    * a delimited index (LJSpeech ``metadata.csv`` and friends) — any file with
      an ``id``-like and a text-like column, or the classic ``id|text|...``;
    * one sidecar ``.txt`` / ``.lab`` next to each ``.wav``.

    ``strip_brackets`` unwraps ``(...)`` and ``[...]`` in the reference, keeping
    the words inside.  This is not cosmetic.  Whisper's ``EnglishTextNormalizer``
    *deletes* bracketed spans, because in many corpora they annotate non-speech
    events — but in LJSpeech (and most read-speech sets) the parenthetical is
    read aloud.  Left in, the reference silently loses those words while the
    hypothesis keeps them, so a correct transcription is scored as a run of
    insertions.  Measured on the shipped 200-utterance subset that was ~40
    spurious insertions per model and about +1.2 points of WER, identically for
    all three architectures — which is what gave it away.  Pass
    ``--keep-brackets`` for a corpus where brackets really do mark non-speech.
    """
    wavs = sorted(audio_root.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav files under {audio_root}")

    texts: Dict[str, str] = {}
    if transcripts is not None:
        texts = _read_transcript_index(transcripts)
        if not texts:
            raise SystemExit(f"no transcripts parsed out of {transcripts}")
    else:
        for w in wavs:
            for suffix in (".txt", ".lab"):
                side = w.with_suffix(suffix)
                if side.exists():
                    texts[w.stem] = side.read_text(encoding="utf-8").strip()
                    break
        if not texts:
            raise SystemExit(
                f"no sidecar .txt/.lab next to the wavs in {audio_root}; "
                "pass --transcripts with an index file instead"
            )

    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for w in wavs:
            text = texts.get(w.stem)
            if not text:
                continue
            if strip_brackets:
                text = _BRACKETS.sub(" ", text)
                text = " ".join(text.split())
            f.write(
                json.dumps({"id": w.stem, "audio": w.name, "text": text}, ensure_ascii=False) + "\n"
            )
            written += 1
            if limit and written >= limit:
                break
    print(f"wrote {written} entries to {out_path} (audio paths are relative to --audio-root)")


def _read_transcript_index(path: Path) -> Dict[str, str]:
    """Parse ``id -> text`` out of a delimited index file."""
    raw = path.read_text(encoding="utf-8").splitlines()
    if not raw:
        return {}
    # Pipe-delimited (canonical LJSpeech): id|text|normalized_text
    if "|" in raw[0]:
        out = {}
        for line in raw:
            parts = line.split("|")
            if len(parts) >= 2 and parts[0]:
                # Prefer the normalized column when present, as LJSpeech ships it.
                out[parts[0]] = (parts[2] if len(parts) > 2 and parts[2] else parts[1]).strip()
        return out
    # CSV with a header: find an id-ish column and a text-ish one.
    rows = list(csv.DictReader(raw))
    if not rows:
        return {}
    cols = [c for c in rows[0] if c]
    id_col = _pick(cols, ("id", "utt_id", "utterance_id", "name", "file_name"))
    text_col = _pick(cols, ("sentence", "text", "transcript", "transcription", "normalized"))
    if id_col is None or text_col is None:
        raise SystemExit(
            f"{path}: could not find id/text columns among {cols}; "
            "convert it to `id|text` per line instead"
        )
    out = {}
    for r in rows:
        uid = (r.get(id_col) or "").strip()
        if uid:
            out[Path(uid).stem] = (r.get(text_col) or "").strip()
    return out


def _pick(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in cols}
    for want in candidates:
        if want in lowered:
            return lowered[want]
    for c in cols:  # substring fallback: "normalized_transcription" etc.
        if any(want in c.lower() for want in candidates):
            return c
    return None


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_one(
    entries: Sequence[Entry],
    *,
    ckpt_dir: str,
    architecture: Optional[str],
    decode_method: Optional[str],
    service_mode: str,
    chunk_size: Optional[int],
    dtype: str,
    max_batch_size: int,
    metric: str,
    normalizer_kind: str,
    manifest_name: str,
    save_transcripts: Optional[Path],
) -> Row:
    import torch

    from oasr.engine import ASREngine, EngineConfig

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype
    ]
    streaming = service_mode == "streaming"
    cfg_kwargs = {
        "ckpt_dir": ckpt_dir,
        "service_mode": service_mode,
        "dtype": torch_dtype,
        "max_batch_size": max_batch_size,
    }
    # Only for an explicit-only converter (``transducer``): an icefall RNN-T dir
    # sniffs as ``zipformer``, so without this the sweep silently measures the
    # CTC branch — or fails on a transducer-only export.
    if architecture:
        cfg_kwargs["architecture"] = architecture
    if decode_method:
        cfg_kwargs["decode_method"] = decode_method
    # Only forward a chunk size when asked for one: the engine's default is
    # model-aware, and an encoder that cannot serve a given chunk refuses it at
    # construction rather than decoding something subtly wrong.
    if streaming and chunk_size:
        cfg_kwargs["chunk_size"] = chunk_size
    cfg = EngineConfig(**cfg_kwargs)
    engine = ASREngine(cfg)
    # Report the chunk actually run, not the flag: 0 on the CLI means "whatever
    # the model defaults to", and a table that says 0 explains nothing.
    effective_chunk = int(cfg.chunk_size) if streaming else 0

    try:
        waves, audio_seconds = load_audio(entries, engine.sample_rate)
        t0 = time.perf_counter()
        hyps, per_utt_ms = transcribe(engine, waves, max_batch_size, streaming=streaming)
        wall = time.perf_counter() - t0
    finally:
        del engine
        import torch as _t

        if _t.cuda.is_available():
            _t.cuda.empty_cache()

    result = compute(
        [e.text for e in entries],
        hyps,
        unit="word" if metric == "wer" else "char",
        normalizer=normalizer(normalizer_kind),
        uids=[e.uid for e in entries],
    )

    if save_transcripts:
        save_transcripts.parent.mkdir(parents=True, exist_ok=True)
        with open(save_transcripts, "w", encoding="utf-8") as f:
            for e, h in zip(entries, hyps):
                f.write(
                    json.dumps({"id": e.uid, "ref": e.text, "hyp": h}, ensure_ascii=False) + "\n"
                )
        print(f"[INFO] transcripts written to {save_transcripts}")

    ordered = sorted(per_utt_ms)
    return Row(
        manifest=manifest_name,
        ckpt=Path(ckpt_dir).name,
        architecture=architecture or "(detected)",
        decode_method=decode_method or "(model default)",
        service_mode=service_mode,
        chunk_size=effective_chunk,
        dtype=dtype,
        max_batch_size=max_batch_size,
        metric=metric,
        normalizer=normalizer_kind,
        utterances=len(entries),
        audio_seconds=round(audio_seconds, 2),
        error_rate_pct=round(result.percent, 3),
        substitutions=result.substitutions,
        deletions=result.deletions,
        insertions=result.insertions,
        ref_units=result.counts.ref_len,
        wall_seconds=round(wall, 3),
        rtfx=round(audio_seconds / wall, 1) if wall else 0.0,
        utt_per_second=round(len(entries) / wall, 2) if wall else 0.0,
        latency_p50_ms=round(_pct(ordered, 0.50), 2),
        latency_p99_ms=round(_pct(ordered, 0.99), 2),
        result=result,
    )


def _pct(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _envstr(name: str, default):
    return os.environ.get(name) or default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OASR Accuracy Benchmark (WER/CER + speed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt-dir", default=_envstr("CKPT_DIR", None), metavar="DIR")
    p.add_argument(
        "--architecture",
        metavar="NAME",
        help="Force a registered architecture instead of checkpoint detection. "
        "Needed for `transducer`, whose converter is explicit-only because an "
        "icefall pruned-RNNT dir sniffs as `zipformer`.",
    )
    p.add_argument("--manifest", metavar="FILE", help="JSON Lines {id, audio, text}")
    p.add_argument(
        "--audio-root",
        default=_envstr("AUDIO_DIR", None),
        metavar="DIR",
        help="Root for relative audio paths in the manifest (default: $AUDIO_DIR)",
    )
    p.add_argument("--limit", type=int, default=None, metavar="N", help="Use only the first N")

    p.add_argument(
        "--decode-method",
        nargs="+",
        default=[""],
        metavar="M",
        help="One row per method (e.g. ctc ctc_aed_rescoring); default: the model's",
    )
    p.add_argument(
        "--service-mode",
        nargs="+",
        default=["offline"],
        choices=["offline", "streaming"],
        metavar="M",
        help="One row per mode. `streaming` feeds each utterance chunk by chunk "
        "through the streaming runtime instead of one padded offline forward — "
        "same manifest and same denominator, so the two rates are directly "
        "comparable. A model whose encoder declares no streaming support fails "
        "this row at engine construction and the sweep carries on.",
    )
    p.add_argument(
        "--chunk-size",
        nargs="+",
        type=int,
        default=[0],
        metavar="N",
        help="Encoder chunk size (frames) for --service-mode streaming; 0 keeps "
        "the model's default. Only expanded into rows when streaming is in the "
        "sweep, so an offline run is not multiplied by an axis it ignores.",
    )
    p.add_argument(
        "--dtype",
        nargs="+",
        default=["float16"],
        choices=["float16", "bfloat16", "float32"],
        metavar="D",
    )
    p.add_argument("--max-batch-size", nargs="+", type=int, default=[16], metavar="N")

    p.add_argument("--metric", default="wer", choices=["wer", "cer"])
    p.add_argument(
        "--normalizer",
        default=None,
        choices=["english", "basic", "none"],
        help="Default: english for --metric wer, basic for cer",
    )

    p.add_argument("--output-path", metavar="FILE", help="Write results as CSV")
    p.add_argument("--save-transcripts", metavar="FILE", help="Dump ref/hyp JSON Lines")
    p.add_argument(
        "--show-worst", type=int, default=5, metavar="N", help="Print the N worst utterances"
    )

    g = p.add_argument_group("manifest building")
    g.add_argument("--build-manifest", metavar="OUT", help="Write a manifest and exit")
    g.add_argument("--transcripts", metavar="FILE", help="Index file (id|text or CSV) to read from")
    g.add_argument(
        "--keep-brackets",
        action="store_true",
        help="Keep (...) / [...] verbatim. Default unwraps them, keeping the words: "
        "the English normalizer deletes bracketed spans, which silently drops "
        "read-aloud parentheticals from the reference. See build_manifest().",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.build_manifest:
        if not args.audio_root:
            raise SystemExit("--build-manifest needs --audio-root")
        build_manifest(
            Path(args.build_manifest),
            Path(args.audio_root),
            Path(args.transcripts) if args.transcripts else None,
            args.limit,
            strip_brackets=not args.keep_brackets,
        )
        return 0

    if not args.manifest:
        raise SystemExit("--manifest is required (or --build-manifest to make one)")
    if not args.ckpt_dir:
        raise SystemExit("--ckpt-dir is required (or export CKPT_DIR)")

    norm_kind = args.normalizer or ("english" if args.metric == "wer" else "basic")
    manifest = Path(args.manifest)
    try:
        entries = load_manifest(
            manifest, Path(args.audio_root) if args.audio_root else None, args.limit
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"[INFO] {len(entries)} utterance(s) from {manifest}")

    rows: List[Row] = []
    failures: List[str] = []
    for method in args.decode_method:
        # An offline row ignores the chunk size, so folding it into the product
        # would report the same measurement N times under different labels.
        for mode, chunk in [
            (m, c)
            for m in args.service_mode
            for c in (args.chunk_size if m == "streaming" else [0])
        ]:
            for dtype in args.dtype:
                for bs in args.max_batch_size:
                    tag = mode if mode == "offline" else f"{mode}{chunk or ''}"
                    label = f"{method or 'default'}/{tag}/{dtype}/bs{bs}"
                    print(f"[INFO] running {label} ...", flush=True)
                    try:
                        row = run_one(
                            entries,
                            ckpt_dir=args.ckpt_dir,
                            architecture=args.architecture,
                            decode_method=method or None,
                            service_mode=mode,
                            chunk_size=chunk or None,
                            dtype=dtype,
                            max_batch_size=bs,
                            metric=args.metric,
                            normalizer_kind=norm_kind,
                            manifest_name=manifest.name,
                            save_transcripts=(
                                Path(args.save_transcripts) if args.save_transcripts else None
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 — a sweep must not lose earlier rows
                        # Not every configuration is supported by every model — the
                        # conformer's conv2d is fp16/bf16 only, for instance, and an
                        # offline-only encoder refuses `--service-mode streaming`.
                        # Report it and carry on rather than discarding rows already
                        # measured, which is the point of running a sweep at all.
                        msg = f"{label}: {type(exc).__name__}: {exc}".splitlines()[0]
                        print(f"       FAILED — {msg}", flush=True)
                        failures.append(msg)
                        continue
                    rows.append(row)
                    print(
                        f"       {row.result.summary()}  RTFx {row.rtfx}  "
                        f"p50 {row.latency_p50_ms} ms"
                    )

    _print_table(rows)

    if failures:
        print(f"{len(failures)} configuration(s) failed:")
        for f in failures:
            print("  " + f)

    if args.show_worst and rows:
        print(f"\nworst {args.show_worst} utterance(s) of the last configuration:")
        for line in rows[-1].result.worst(args.show_worst):
            print("  " + line)

    if args.output_path:
        with open(args.output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for r in rows:
                d = asdict(r)
                d.pop("result", None)
                w.writerow(d)
        print(f"[INFO] Results saved to: {args.output_path}")
    return 0 if rows else 1


def _print_table(rows: Sequence[Row]) -> None:
    hdr = f"{'config':<44} {'metric':>7} {'rate %':>8} {'RTFx':>8} {'p50 ms':>8} {'p99 ms':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        mode = r.service_mode if r.service_mode == "offline" else f"streaming{r.chunk_size}"
        cfg = f"{r.decode_method}/{mode}/{r.dtype}/bs{r.max_batch_size}"
        print(
            f"{cfg:<44} {r.metric:>7} {r.error_rate_pct:>8.2f} "
            f"{r.rtfx:>8.1f} {r.latency_p50_ms:>8.1f} {r.latency_p99_ms:>8.1f}"
        )
    print()
    print(
        f"({len(rows)} configuration(s); "
        f"{rows[0].utterances if rows else 0} utterances, "
        f"{rows[0].audio_seconds if rows else 0:.0f} s of audio)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
