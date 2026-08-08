# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The ``oasr`` command line.

    oasr transcribe meeting.mp3                     # against a running server
    oasr transcribe meeting.mp3 --ckpt-dir ./ckpt   # in-process, no server
    oasr translate  entretien.m4a --language fr
    oasr models
    oasr serve --ckpt-dir ./ckpt
    oasr convert /path/to/wenet /path/to/native

Two transcription paths, because the two audiences want different things: an
evaluator wants ``oasr transcribe file.mp3`` to work against a server they just
started, and a scripter wants the engine in-process with no server at all.
``--ckpt-dir`` selects the second; everything else uses the first.

``serve`` and ``convert`` forward to the existing entry points so there is one
command to learn rather than three.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from oasr.client import DEFAULT_BASE_URL, OASRClient, OASRClientError, Transcription

#: Response formats the server renders.
RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")


def _add_common_transcribe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio", nargs="+", help="audio file(s); any container the server decodes")
    p.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"oasr-server base URL (default: {DEFAULT_BASE_URL})",
    )
    p.add_argument(
        "--ckpt-dir",
        help="transcribe in-process with this checkpoint instead of calling a server",
    )
    p.add_argument("--model", help="model name to request (servers with --served-model-name)")
    p.add_argument("--language", help='source language, e.g. "en" or "fr-FR"')
    p.add_argument("--prompt", help="prompt override (speech-LLM families)")
    p.add_argument("--temperature", type=float, help="0 = greedy (default)")
    p.add_argument(
        "--response-format",
        default="text",
        choices=RESPONSE_FORMATS,
        help="server-side rendering (default: text)",
    )
    p.add_argument(
        "--timestamp-granularity",
        action="append",
        dest="timestamp_granularities",
        choices=("segment", "word"),
        help="request timestamps (repeatable); needs --response-format verbose_json",
    )
    p.add_argument("-o", "--output", help="write to this file instead of stdout")
    p.add_argument("--timeout", type=float, default=300.0, help="request timeout in seconds")
    p.add_argument(
        "--device",
        default="cuda",
        help="device for --ckpt-dir transcription (default: cuda)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oasr",
        description="OASR — high-performance ASR inference and serving.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("transcribe", help="transcribe audio")
    _add_common_transcribe_args(p)
    p.set_defaults(func=lambda a: _cmd_transcribe(a, task="transcribe"))

    p = sub.add_parser("translate", help="translate speech to English (Whisper-family models)")
    _add_common_transcribe_args(p)
    p.set_defaults(func=lambda a: _cmd_transcribe(a, task="translate"))

    p = sub.add_parser("models", help="list the models a server is serving")
    p.add_argument("--url", default=DEFAULT_BASE_URL)
    p.set_defaults(func=_cmd_models)

    p = sub.add_parser(
        "serve",
        help="run the HTTP + gRPC server (all flags are forwarded to oasr-server)",
        add_help=False,
    )
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser(
        "convert",
        help="convert a checkpoint to the native OASR format",
        add_help=False,
    )
    p.set_defaults(func=_cmd_convert)
    return parser


# ---------------------------------------------------------------------------
# transcribe / translate
# ---------------------------------------------------------------------------


def _render(result: Transcription, response_format: str) -> str:
    """The bytes to print for one result."""
    if response_format in ("text", "srt", "vtt"):
        return result.text
    if response_format == "json":
        return json.dumps({"text": result.text}, ensure_ascii=False)
    return json.dumps(result.raw or {"text": result.text}, ensure_ascii=False, indent=2)


def _cmd_transcribe(args: argparse.Namespace, *, task: str) -> int:
    args.task = task
    if args.ckpt_dir:
        results = _transcribe_locally(args)
    else:
        results = _transcribe_remotely(args, task=task)

    rendered = "\n".join(_render(r, args.response_format) for r in results)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(rendered)
    return 0


def _transcribe_remotely(args: argparse.Namespace, *, task: str) -> List[Transcription]:
    client = OASRClient(args.url, model=args.model, timeout=args.timeout)
    call = client.translate if task == "translate" else client.transcribe
    out: List[Transcription] = []
    for path in args.audio:
        kwargs = {
            "language": args.language,
            "prompt": args.prompt,
            "response_format": args.response_format,
            "temperature": args.temperature,
        }
        if task != "translate":
            kwargs["timestamp_granularities"] = args.timestamp_granularities
        out.append(call(path, **kwargs))
    return out


def _transcribe_locally(args: argparse.Namespace) -> List[Transcription]:
    """Build an in-process engine and run the files through it.

    This is the path the README's Python API describes and that no example
    demonstrated; having it behind one command is also what makes
    ``pip install oasr && oasr transcribe audio.mp3 --ckpt-dir ...`` a complete
    story on a machine with a GPU and no server.
    """
    from oasr.engine import ASREngine, DecodingOptions, EngineConfig

    if args.timestamp_granularities:
        raise SystemExit(
            "--timestamp-granularity needs the server path; the in-process API "
            "returns RequestOutput.timestamps directly"
        )
    cfg = EngineConfig(ckpt_dir=args.ckpt_dir, service_mode="offline", device=args.device)
    engine = ASREngine(cfg)
    try:
        # `EngineConfig.__post_init__` always materializes a feature config, and
        # `ASREngine` then resolves it against the checkpoint's FeatureSpec — so
        # by here this is the rate the engine will actually accept.
        assert cfg.feature_config is not None
        sample_rate = cfg.feature_config.sample_rate
        audios = [_load_audio(p, sample_rate) for p in args.audio]
        decoding = None
        if args.language or args.temperature or args.prompt or args.task != "transcribe":
            decoding = DecodingOptions(
                task=args.task if args.task != "transcribe" else None,
                language=(args.language or "").split("-")[0].lower() or None,
                temperature=args.temperature or 0.0,
                prompt=args.prompt or None,
            )
        texts = engine.transcribe_offline(audios, decoding=decoding)
        return [Transcription(text=t, raw={"text": t}) for t in texts]
    finally:
        del engine


def _load_audio(path: str, sample_rate: int):
    """Load one file as a mono waveform at ``sample_rate``.

    Uses ``soundfile`` (the ``audio`` extra), which reads every container the
    server does. Resampling goes through ``torchaudio`` when the rates differ —
    submitting at the wrong rate is the failure mode C2 was about, so it is
    never skipped silently.
    """
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:  # pragma: no cover - optional dep
        raise SystemExit(
            "in-process transcription needs the audio extra: " 'pip install "oasr[audio]"'
        ) from exc

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1)))
    if sr != sample_rate:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


# ---------------------------------------------------------------------------
# models / serve / convert
# ---------------------------------------------------------------------------


def _cmd_models(args: argparse.Namespace) -> int:
    for model in OASRClient(args.url).models():
        info = model.get("info") or {}
        detail = ", ".join(
            f"{k}={info[k]}"
            for k in ("decode_method", "service_mode", "device", "dtype", "sample_rate")
            if info.get(k) is not None
        )
        print(f"{model.get('id')}  {detail}" if detail else str(model.get("id")))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Forward every remaining argument to the Rust server entry point."""
    from oasr._server_cli import main as serve_main

    sys.argv = ["oasr-server", *args.rest]
    # Both entry points signal failure by raising / SystemExit, never by a code.
    serve_main()
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from oasr.checkpoints.convert import main as convert_main

    sys.argv = ["oasr-convert", *args.rest]
    convert_main()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # `serve` and `convert` are pass-throughs: parse only the subcommand name
    # and hand the rest over untouched, so their own `--help` and flags work.
    if argv and argv[0] in ("serve", "convert"):
        args = argparse.Namespace(rest=argv[1:])
        return _cmd_serve(args) if argv[0] == "serve" else _cmd_convert(args)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except OASRClientError as exc:
        hint = ""
        if exc.status is None:
            hint = (
                "\nIs a server running? Start one with:\n"
                "  oasr serve --ckpt-dir <dir> --service-mode offline\n"
                "or transcribe in-process with --ckpt-dir."
            )
        print(f"error: {exc}{hint}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
