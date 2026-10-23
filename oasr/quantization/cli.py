# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr quantize``: write a quantized native checkpoint."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    from .recipes import RECIPES

    parser = argparse.ArgumentParser(
        prog="oasr quantize",
        description=(
            "Quantize a checkpoint (WeNet / icefall / native) and write it in the native "
            "format with a `quantization` section; the engine then serves it quantized "
            "with no extra flag."
        ),
    )
    parser.add_argument("src", help="Source checkpoint directory")
    parser.add_argument("dst", help="Destination directory for the quantized native bundle")
    parser.add_argument(
        "--recipe",
        default="smoothquant_w8a8",
        choices=sorted(RECIPES),
        help="Scheme + algorithms (default: smoothquant_w8a8)",
    )
    parser.add_argument(
        "--calib-manifest",
        default=None,
        help=(
            "jsonl manifest ({id, audio, ...}) of calibration audio, e.g. "
            "$LIBRISPEECH_DEV_DIR/manifest.jsonl (python -m benchmarks.asrbench prepare "
            "librispeech --subset dev-clean --download)"
        ),
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Directory the manifest's relative audio paths resolve against "
        "(default: the manifest's directory)",
    )
    parser.add_argument(
        "--calib-utterances",
        type=int,
        default=128,
        help="Utterances to calibrate on, evenly spread over the manifest (default 128)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="SmoothQuant migration strength (default 0.5)"
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="ALGO.KEY=VALUE",
        help="Algorithm option, e.g. smoothquant.s_max=1000 (repeatable)",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="GLOB",
        help="Extra module-path glob to keep at full precision (repeatable)",
    )
    parser.add_argument("--architecture", default=None, help="Registry key override")
    parser.add_argument("--checkpoint-name", default="final.pt")
    parser.add_argument("--device", default="cuda", help="Calibration device (default cuda)")
    parser.add_argument(
        "--calib-dtype",
        default=None,
        choices=("float32", "bfloat16", "float16"),
        help=(
            "Run the calibration audio through a copy of the model in this dtype on "
            "--device while the fp32 master (every scale, fold and weight quantization) "
            "stays on the host -- for a model whose fp32 master does not fit the device "
            "(default: calibrate the fp32 master on --device)"
        ),
    )
    return parser


def _algorithm_options(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    options: Dict[str, Dict[str, Any]] = {}
    if args.alpha is not None:
        options.setdefault("smoothquant", {})["alpha"] = args.alpha
    for item in args.option:
        key, sep, value = item.partition("=")
        algo, dot, name = key.partition(".")
        if not sep or not dot:
            raise SystemExit(f"--option expects ALGO.KEY=VALUE, got {item!r}")
        options.setdefault(algo, {})[name] = value
    return options


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    from .export import quantize_checkpoint

    ignore: List[str] = list(args.ignore)
    out = quantize_checkpoint(
        args.src,
        args.dst,
        recipe=args.recipe,
        calib_manifest=args.calib_manifest,
        audio_root=args.audio_root,
        calib_utterances=args.calib_utterances,
        algorithm_options=_algorithm_options(args),
        ignore=ignore,
        architecture=args.architecture,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        calibration_dtype=args.calib_dtype,
    )
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
