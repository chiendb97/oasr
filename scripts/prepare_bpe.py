#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Prepare a BPE-based lang directory for CTC WFST decoding.

Reads a ``words.txt`` word symbol table and a SentencePiece BPE model,
segments each word into BPE pieces, and writes the standard lang outputs:

- ``tokens.txt``           — BPE token-to-ID mapping (from ``--units-file``,
                             extended with disambiguation symbols).
- ``lexicon.txt``          — word → BPE-piece sequence mapping.
- ``lexicon_disambig.txt`` — lexicon with disambiguation symbols.
- ``L.pt``                 — lexicon FST (k2 format).
- ``L_disambig.pt``        — lexicon FST with ``#0`` self-loops (k2 format).

The ``--lang-dir`` must already contain a ``words.txt`` word symbol table.

Usage::

    python scripts/prepare_bpe.py \\
        --lang-dir data/lang_bpe \\
        --bpe-model /path/to/train_960_unigram5000.model \\
        --units-file /path/to/units.txt

Typically ``--units-file`` is the ``units.txt`` from the acoustic model
directory (e.g. ``/path/to/am/units.txt``).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Add scripts/ to the path so lang_utils can be imported.
sys.path.insert(0, str(Path(__file__).parent))

import k2
import torch

from lang_utils import (
    Lexicon,
    add_disambig_symbols,
    lexicon_to_fst_no_sil,
    write_lexicon,
    write_mapping,
)


def segment_word(sp_model, word: str) -> List[str]:
    """Segment *word* into BPE pieces using *sp_model*.

    SentencePiece encodes with ``▁`` as word-boundary prefix.
    An empty segmentation falls back to ``["<unk>"]``.
    """
    pieces = sp_model.encode(word, out_type=str)
    return pieces if pieces else ["<unk>"]


def build_bpe_lexicon(
    sp_model,
    words: List[str],
    token_sym_table: Dict[str, int],
) -> Lexicon:
    """Build a lexicon mapping each word to its BPE segmentation.

    Words whose BPE segmentation contains a piece not in *token_sym_table*
    are mapped to ``["<unk>"]``.  ``<UNK>`` always maps to ``["<unk>"]``.
    ``<UNK>`` is appended at the end if it was not in *words*.

    Args:
        sp_model: A loaded ``sentencepiece.SentencePieceProcessor``.
        words: List of vocabulary words to segment.
        token_sym_table: BPE token-to-ID mapping (from units.txt).

    Returns:
        Lexicon — list of (word, [bpe_piece, ...]) pairs.
    """
    lexicon: Lexicon = []
    has_unk = False
    n_oov = 0

    for word in words:
        if word == "<UNK>":
            lexicon.append(("<UNK>", ["<unk>"]))
            has_unk = True
            continue
        pieces = segment_word(sp_model, word)
        if any(p not in token_sym_table for p in pieces):
            n_oov += 1
            pieces = ["<unk>"]
        lexicon.append((word, pieces))

    # Always guarantee <UNK> → <unk> is present.
    if not has_unk:
        lexicon.append(("<UNK>", ["<unk>"]))

    if n_oov:
        print(
            f"  WARNING: {n_oov} words had out-of-vocabulary BPE pieces; "
            "mapped to <unk>.",
            file=sys.stderr,
        )
    return lexicon


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        required=True,
        help="Input/output lang directory. Must contain 'words.txt'.",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        required=True,
        help="Path to the SentencePiece BPE model (.model file).",
    )
    parser.add_argument(
        "--units-file",
        type=str,
        required=True,
        help="Path to the AM units.txt (BPE token symbol table from the acoustic model).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    words_file = lang_dir / "words.txt"
    if not words_file.is_file():
        print(f"ERROR: {words_file} not found", file=sys.stderr)
        sys.exit(1)

    units_file = Path(args.units_file)
    if not units_file.is_file():
        print(f"ERROR: {units_file} not found", file=sys.stderr)
        sys.exit(1)

    bpe_model_path = Path(args.bpe_model)
    if not bpe_model_path.is_file():
        print(f"ERROR: BPE model not found: {bpe_model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import sentencepiece as spm
    except ImportError:
        print(
            "ERROR: sentencepiece is required. Install with: pip install sentencepiece",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print(f"Loading BPE model from {bpe_model_path}", file=sys.stderr)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(bpe_model_path))

    print(f"Loading BPE token table from {units_file}", file=sys.stderr)
    token_sym_table_k2 = k2.SymbolTable.from_file(units_file)
    token_sym_table: Dict[str, int] = {
        s: token_sym_table_k2[s] for s in token_sym_table_k2.symbols
    }
    print(f"  {len(token_sym_table)} tokens loaded", file=sys.stderr)

    print(f"Loading word symbol table from {words_file}", file=sys.stderr)
    word_sym_table = k2.SymbolTable.from_file(words_file)

    # Filter out special/reserved symbols that should not appear in the lexicon.
    excluded = {"<eps>", "!SIL", "<SPOKEN_NOISE>", "#0", "<s>", "</s>"}
    words = [w for w in word_sym_table.symbols if w not in excluded]
    print(f"  {len(words)} words to segment", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build lexicon
    # ------------------------------------------------------------------
    print("Segmenting words with BPE model ...", file=sys.stderr)
    lexicon = build_bpe_lexicon(sp, words, token_sym_table)

    print("Adding disambiguation symbols ...", file=sys.stderr)
    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    # ------------------------------------------------------------------
    # Extend token symbol table with disambiguation symbols (#0, #1, ...)
    # ------------------------------------------------------------------
    token_sym_table_extended = dict(token_sym_table)
    next_token_id = max(token_sym_table_extended.values()) + 1
    for i in range(max_disambig + 1):
        sym = f"#{i}"
        if sym not in token_sym_table_extended:
            token_sym_table_extended[sym] = next_token_id
            next_token_id += 1

    # Ensure required special symbols exist in word table.
    for w in ("#0", "<s>", "</s>"):
        if w not in word_sym_table:
            word_sym_table.add(w)

    word2id = {w: word_sym_table[w] for w in word_sym_table.symbols}

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    tokens_out = lang_dir / "tokens.txt"
    lexicon_out = lang_dir / "lexicon.txt"
    lexicon_disambig_out = lang_dir / "lexicon_disambig.txt"

    print(f"Writing outputs to {lang_dir}/", file=sys.stderr)
    write_mapping(tokens_out, token_sym_table_extended)
    write_lexicon(lexicon_out, lexicon)
    write_lexicon(lexicon_disambig_out, lexicon_disambig)

    print("Building L (lexicon FST without silence) ...", file=sys.stderr)
    L = lexicon_to_fst_no_sil(lexicon, token2id=token_sym_table_extended, word2id=word2id)

    print("Building L_disambig (lexicon FST with #0 self-loops) ...", file=sys.stderr)
    L_disambig = lexicon_to_fst_no_sil(
        lexicon_disambig,
        token2id=token_sym_table_extended,
        word2id=word2id,
        need_self_loops=True,
    )

    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")

    print("Done.", file=sys.stderr)
    print(f"  tokens.txt           : {tokens_out}", file=sys.stderr)
    print(f"  lexicon.txt          : {lexicon_out}", file=sys.stderr)
    print(f"  lexicon_disambig.txt : {lexicon_disambig_out}", file=sys.stderr)
    print(f"  L.pt                 : {lang_dir / 'L.pt'}", file=sys.stderr)
    print(f"  L_disambig.pt        : {lang_dir / 'L_disambig.pt'}", file=sys.stderr)


if __name__ == "__main__":
    main()
