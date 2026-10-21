# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr convert`` — materialize any supported checkpoint as a native bundle.

Usage::

    python -m oasr.checkpoints.convert /path/to/wenet_exp /path/to/native_out
    oasr-convert /path/to/icefall_exp out/ --architecture zipformer

Runs the format converter, loads the weights through the architecture's
``load_weights`` (printing the :class:`~oasr.models.base.LoadReport`), and
writes the round-trippable native format (``oasr_config.json`` +
``model.safetensors`` + tokenizer assets).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .native import save_native

logger = logging.getLogger(__name__)


def convert_to_native(
    src_dir: str,
    dst_dir: str,
    *,
    architecture: Optional[str] = None,
    checkpoint_name: str = "final.pt",
) -> Path:
    """Convert *src_dir* (WeNet / icefall / native) into a native bundle at *dst_dir*."""
    from oasr.models.registry import load_checkpoint_bundle, materialize_model

    arch, bundle = load_checkpoint_bundle(
        Path(src_dir), checkpoint_name=checkpoint_name, architecture=architecture
    )
    # The shared load sequence, on the host, with no dtype cast: the native copy
    # keeps the source's precision (a quantized source stays quantized).
    model, _config, report = materialize_model(arch, bundle)
    if report is not None:
        print(report.summary())

    return save_native(
        Path(dst_dir),
        architecture=arch,
        model=model,
        model_config=bundle.model_config,
        aux=bundle.aux,
        tokenizer=bundle.tokenizer,
        features=bundle.features,
        decoding=bundle.decoding,
        quantization=model.quantization_spec,
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="oasr-convert",
        description="Convert a WeNet / icefall checkpoint dir to the native OASR format.",
    )
    parser.add_argument("src", help="Source checkpoint directory")
    parser.add_argument("dst", help="Destination directory for the native bundle")
    parser.add_argument(
        "--architecture",
        default=None,
        help="Registry key override (skips format auto-detection)",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="final.pt",
        help="Weights filename inside src (default: final.pt)",
    )
    args = parser.parse_args(argv)

    out = convert_to_native(
        args.src,
        args.dst,
        architecture=args.architecture,
        checkpoint_name=args.checkpoint_name,
    )
    print(f"Native checkpoint written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
