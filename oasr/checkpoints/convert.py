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
    from oasr.models.registry import get_model_entry, load_checkpoint_bundle

    arch, bundle = load_checkpoint_bundle(
        Path(src_dir), checkpoint_name=checkpoint_name, architecture=architecture
    )
    entry = get_model_entry(arch)
    model = entry.model_cls.from_config(bundle.model_config, **bundle.aux)

    if bundle.source_format == "native":
        from .native import load_native_weights

        load_native_weights(model, dict(bundle.state_dict))
    else:
        report = model.load_weights(bundle.state_dict)
        if report is not None:
            print(report.summary())

    model.eval()
    return save_native(
        Path(dst_dir),
        architecture=arch,
        model=model,
        model_config=bundle.model_config,
        aux=bundle.aux,
        tokenizer=bundle.tokenizer,
        features=bundle.features,
        decoding=bundle.decoding,
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
