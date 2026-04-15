#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Compile an LG graph (without the CTC H transducer).

    LG = L ∘ G

Some decoders and rescoring techniques apply the CTC topology separately and
only need LG.  Outputs ``{lang_dir}/LG.pt``.

Usage::

    python scripts/compile_lg.py \\
        --lang-dir data/lang_char \\
        --lm G_3_gram \\
        --lm-dir data/lm
"""

import argparse
import logging
import sys
from pathlib import Path

# Add scripts/ to path so lang_utils can be imported.
sys.path.insert(0, str(Path(__file__).parent))

import k2
import torch

from lang_utils import LangDir


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang-dir", type=str, required=True,
                        help="Lang directory with L_disambig.pt, tokens.txt, words.txt.")
    parser.add_argument("--lm", type=str, default="G_3_gram",
                        help="Stem name of the LM file (default: G_3_gram).")
    parser.add_argument("--lm-dir", type=str, default="data/lm",
                        help="Directory containing the LM file (default: data/lm).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing LG.pt if it already exists.")
    return parser.parse_args()


def compile_LG(lang_dir: Path, lm_dir: Path, lm: str) -> k2.Fsa:
    """Compose L_disambig with G and clean up into a tidy LG graph."""
    lang = LangDir(lang_dir)
    first_token_disambig_id = lang.token_table["#0"]
    first_word_disambig_id = lang.word_table["#0"]

    l_disambig_path = lang_dir / "L_disambig.pt"
    logging.info(f"Loading L from {l_disambig_path}")
    L = k2.Fsa.from_dict(torch.load(l_disambig_path, weights_only=False))

    g_pt = lm_dir / f"{lm}.pt"
    g_fst_txt = lm_dir / f"{lm}.fst.txt"

    if g_pt.is_file():
        logging.info(f"Loading pre-compiled G from {g_pt}")
        G = k2.Fsa.from_dict(torch.load(g_pt, weights_only=False))
    elif g_fst_txt.is_file():
        logging.info(f"Loading G from {g_fst_txt}")
        with open(g_fst_txt, "r", encoding="utf-8") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        logging.info(f"Caching compiled G to {g_pt}")
        torch.save(G.as_dict(), g_pt)
    else:
        raise FileNotFoundError(
            f"LM not found: expected {g_pt} or {g_fst_txt}. "
            "Run 'python -m kaldilm ...' to produce the .fst.txt file."
        )

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Composing L and G")
    LG = k2.compose(L, G)
    logging.info(f"  shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)

    logging.info("Determinizing LG")
    LG = k2.determinize(LG)
    LG = k2.connect(LG)

    logging.info("Removing disambiguation symbols")
    labels = LG.labels
    labels[labels >= first_token_disambig_id] = 0
    LG.labels = labels
    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)
    LG = k2.arc_sort(LG)

    logging.info(f"  LG shape: {LG.shape}")
    return LG


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    lm_dir = Path(args.lm_dir)

    output_path = lang_dir / "LG.pt"
    if output_path.is_file() and not args.overwrite:
        logging.info(f"{output_path} already exists — skipping (use --overwrite to force)")
        return

    LG = compile_LG(lang_dir, lm_dir, args.lm)
    logging.info(f"Saving LG to {output_path}")
    torch.save(LG.as_dict(), output_path)
    logging.info("Done.")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
