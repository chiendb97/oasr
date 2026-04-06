#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Compile an HLG decoding graph from H, L, and G.

    HLG = H ∘ L ∘ G

where:
  - H  = CTC topology (built via ``k2.ctc_topo(max_token_id)``)
  - L  = Lexicon transducer (loaded from ``{lang_dir}/L_disambig.pt``)
  - G  = N-gram language model (loaded from ``{lm_dir}/{lm}.fst.txt`` or
         the pre-compiled ``{lm_dir}/{lm}.pt``)

Adapted from the Icefall AISHELL recipe:
  egs/aishell/ASR/local/compile_hlg.py

No dependency on the ``icefall`` Python package.

Usage::

    # Convert ARPA LM to OpenFst text format first (requires kaldilm):
    python -m kaldilm \\
        --read-symbol-table="data/lang_char/words.txt" \\
        --disambig-symbol='#0' \\
        --max-order=3 \\
        data/lm/G_3_gram.arpa > data/lm/G_3_gram.fst.txt

    # Compile HLG:
    python scripts/compile_hlg.py \\
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
    parser.add_argument(
        "--lang-dir",
        type=str,
        required=True,
        help="Lang directory containing L_disambig.pt, tokens.txt, words.txt.",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="G_3_gram",
        help="Stem name of the LM file (without extension). "
        "Expects {lm_dir}/{lm}.fst.txt (OpenFst text) or {lm_dir}/{lm}.pt. "
        "(default: G_3_gram)",
    )
    parser.add_argument(
        "--lm-dir",
        type=str,
        default="data/lm",
        help="Directory containing the LM file (default: data/lm).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HLG.pt if it already exists.",
    )
    return parser.parse_args()


def compile_HLG(lang_dir: Path, lm_dir: Path, lm: str) -> k2.Fsa:
    """Compile the HLG decoding graph.

    Args:
        lang_dir: Path to the prepared lang directory.
        lm_dir: Directory containing the language model.
        lm: Stem name of the LM file (e.g. ``G_3_gram``).

    Returns:
        The compiled HLG as a k2.Fsa.
    """
    # ------------------------------------------------------------------
    # 1. Build H: CTC topology
    # ------------------------------------------------------------------
    lang = LangDir(lang_dir)
    max_token_id = max(lang.tokens)
    logging.info(f"Building CTC topology H, max_token_id={max_token_id}")
    H = k2.ctc_topo(max_token_id)

    first_token_disambig_id = lang.token_table["#0"]
    first_word_disambig_id = lang.word_table["#0"]

    # ------------------------------------------------------------------
    # 2. Load L (lexicon FST with disambiguation symbols)
    # ------------------------------------------------------------------
    l_disambig_path = lang_dir / "L_disambig.pt"
    logging.info(f"Loading L from {l_disambig_path}")
    L = k2.Fsa.from_dict(torch.load(l_disambig_path, weights_only=False))

    # ------------------------------------------------------------------
    # 3. Load G (language model FST)
    # ------------------------------------------------------------------
    g_pt = lm_dir / f"{lm}.pt"
    g_fst_txt = lm_dir / f"{lm}.fst.txt"

    # Use the cached .pt only when it is *newer* than the .fst.txt source to
    # avoid stale caches after words.txt or ARPA changes.
    g_pt_fresh = (
        g_pt.is_file()
        and (
            not g_fst_txt.is_file()
            or g_pt.stat().st_mtime >= g_fst_txt.stat().st_mtime
        )
    )
    if g_pt_fresh:
        logging.info(f"Loading pre-compiled G from {g_pt}")
        G = k2.Fsa.from_dict(torch.load(g_pt, weights_only=False))
    elif g_fst_txt.is_file():
        logging.info(f"Loading G from OpenFst text file {g_fst_txt}")
        with open(g_fst_txt, "r", encoding="utf-8") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        # Cache the compiled G for faster subsequent runs.
        logging.info(f"Saving compiled G to {g_pt}")
        torch.save(G.as_dict(), g_pt)
    else:
        raise FileNotFoundError(
            f"Language model not found. Expected one of:\n"
            f"  {g_pt}\n"
            f"  {g_fst_txt}\n"
            "Run 'python -m kaldilm ...' to produce the .fst.txt file."
        )

    # ------------------------------------------------------------------
    # 4. Compose L ∘ G → LG
    # ------------------------------------------------------------------
    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Composing L and G")
    LG = k2.compose(L, G)
    logging.info(f"  LG shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(f"  LG shape after connect: {LG.shape}")

    logging.info("Determinizing LG")
    LG = k2.determinize(LG)

    logging.info("Connecting LG after determinize")
    LG = k2.connect(LG)

    # ------------------------------------------------------------------
    # 5. Remove disambiguation symbols
    # ------------------------------------------------------------------
    logging.info("Removing disambiguation symbols")
    labels = LG.labels
    labels[labels >= first_token_disambig_id] = 0
    LG.labels = labels

    assert isinstance(LG.aux_labels, k2.RaggedTensor), (
        "LG.aux_labels is expected to be a RaggedTensor after k2.determinize()."
    )
    LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    # ------------------------------------------------------------------
    # 6. Clean up: remove epsilon arcs and reconnect
    # ------------------------------------------------------------------
    LG = k2.remove_epsilon(LG)
    logging.info(f"  LG shape after remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)

    logging.info("Arc-sorting LG")
    LG = k2.arc_sort(LG)

    # ------------------------------------------------------------------
    # 7. Compose H ∘ LG → HLG
    # ------------------------------------------------------------------
    logging.info("Composing H and LG")
    # inner_labels="tokens" preserves the original token-level labels
    # as a named attribute on the resulting FSA (required by the decoder).
    HLG = k2.compose(H, LG, inner_labels="tokens")

    logging.info("Connecting HLG")
    HLG = k2.connect(HLG)

    logging.info("Arc-sorting HLG")
    HLG = k2.arc_sort(HLG)
    logging.info(f"  HLG shape: {HLG.shape}")

    return HLG


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    lm_dir = Path(args.lm_dir)

    output_path = lang_dir / "HLG.pt"
    if output_path.is_file() and not args.overwrite:
        logging.info(f"{output_path} already exists — skipping (use --overwrite to force)")
        return

    HLG = compile_HLG(lang_dir, lm_dir, args.lm)

    logging.info(f"Saving HLG to {output_path}")
    torch.save(HLG.as_dict(), output_path)
    logging.info("Done.")
    logging.info("")
    logging.info("The HLG graph can be used with the OASR WFST decoder:")
    logging.info("  from oasr.decode import Decoder, DecoderConfig")
    logging.info("  decoder = Decoder(DecoderConfig(search_type='wfst'), fst=str(output_path))")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
