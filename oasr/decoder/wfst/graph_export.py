#!/usr/bin/env python3
"""Export a k2 HLG.pt to the binary graph image (hlg.img).

Preserves k2 arc order exactly (arc index == k2 graph arc index → lattice arc_map_a is
identity). Detects final-arc placement, validates ranges, flattens ragged aux_labels into
a CSR pool. Run offline once per graph; k2 is only needed here, never at decode time.

Usage: python -m oasr.decoder.wfst.graph_export --hlg HLG.pt --out hlg.img
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from oasr.decoder.wfst.graph_image import build_image, write_image


def export_hlg(
    hlg_pt: str, out_path: str, vocab_size: int | None = None, epsilon_id: int | None = None
) -> dict:
    import k2

    hlg = k2.Fsa.from_dict(torch.load(hlg_pt, map_location="cpu", weights_only=False))
    arcs = hlg.arcs.values()  # [A, 4] int32: src, dest, label, score-bits
    row_splits = hlg.arcs.row_splits(1).numpy().astype(np.int32)
    dest = arcs[:, 1].numpy().astype(np.int32)
    ilabel = hlg.labels.numpy().astype(np.int32)
    weight = hlg.scores.numpy().astype(np.float32)

    aux = hlg.aux_labels
    if isinstance(aux, k2.RaggedTensor):
        aux_row_splits = aux.shape.row_splits(1).numpy().astype(np.int32)
        aux_pool = aux.values.numpy().astype(np.int32)
    else:
        # Flat: exactly one aux label per arc.
        aux_pool = aux.numpy().astype(np.int32)
        aux_row_splits = np.arange(len(aux_pool) + 1, dtype=np.int32)

    img = build_image(
        row_splits=row_splits,
        dest=dest,
        ilabel=ilabel,
        weight=weight,
        aux_row_splits=aux_row_splits,
        aux_pool=aux_pool,
        vocab_size=vocab_size,
        epsilon_id=epsilon_id,
    )
    write_image(img, out_path)
    stats = {
        "num_states": img.num_states,
        "num_arcs": img.num_arcs,
        "vocab_size": img.vocab_size,
        "finals_at_end": img.finals_at_end,
        "aux_pool": int(len(img.aux_pool)),
    }
    print(f"exported {hlg_pt} -> {out_path}: {stats}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hlg", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument(
        "--epsilon-id",
        type=int,
        default=None,
        help="label id to treat as epsilon (TLG graphs; NEVER set for k2 HLG where 0=blank)",
    )
    args = parser.parse_args()
    export_hlg(args.hlg, args.out, args.vocab_size, args.epsilon_id)


if __name__ == "__main__":
    main()
