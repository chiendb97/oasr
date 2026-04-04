#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Prepare a character-level lang directory for CTC decoding.

Adapted from the Icefall AISHELL recipe:
  egs/aishell/ASR/local/prepare_char.py

Given a lang directory that contains:
  - ``text``       — training transcripts (one utterance per line)
  - ``words.txt``  — word symbol table (``word id`` per line)

This script generates:
  - ``tokens.txt``          — character token-to-ID mapping
  - ``lexicon.txt``         — word-to-character mapping
  - ``lexicon_disambig.txt``— lexicon with disambiguation symbols
  - ``L.pt``                — lexicon FST (k2 format)
  - ``L_disambig.pt``       — lexicon FST with #0 self-loops (k2 format)

Usage::

    python scripts/prepare_char.py --lang-dir data/lang_char
"""

import argparse
import re
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


def generate_tokens(text_file) -> Dict[str, int]:
    """Scan *text_file* and build a character-level token table.

    Special tokens are assigned fixed IDs:
      - ``<blk>``     → 0  (CTC blank)
      - ``<sos/eos>`` → 1  (start/end of sentence)
      - ``<unk>``     → 2  (out-of-vocabulary)

    All other characters found in the transcripts are assigned IDs starting
    from 3.

    Args:
        text_file: Path to the transcript file.

    Returns:
        Dict mapping token string → integer ID.
    """
    tokens: Dict[str, int] = {"<blk>": 0, "<sos/eos>": 1, "<unk>": 2}
    whitespace = re.compile(r"([ \t\r\n]+)")
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            # Strip utterance-ID prefix if present (first field separated by space).
            line = line.rstrip("\r\n")
            for ch in list(re.sub(whitespace, "", line)):
                if ch not in tokens:
                    tokens[ch] = len(tokens)
    return tokens


def contain_oov(token_sym_table: Dict[str, int], tokens: List[str]) -> bool:
    """Return True if any token in *tokens* is not in *token_sym_table*."""
    return any(t not in token_sym_table for t in tokens)


def generate_lexicon(token_sym_table: Dict[str, int], words: List[str]) -> Lexicon:
    """Map each word to its constituent characters.

    Words containing characters not in *token_sym_table* are skipped (OOV
    characters).  The special word ``<UNK>`` maps to the ``<unk>`` token.

    Args:
        token_sym_table: Character token table.
        words: List of vocabulary words.

    Returns:
        Lexicon — list of (word, [char, ...]) pairs.
    """
    lexicon: Lexicon = []
    for word in words:
        chars = list(word.strip())
        if not chars or contain_oov(token_sym_table, chars):
            continue
        lexicon.append((word, chars))
    lexicon.append(("<UNK>", ["<unk>"]))
    return lexicon


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang-dir",
        type=str,
        required=True,
        help="Input/output lang directory. Must contain 'text' and 'words.txt'.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    text_file = lang_dir / "text"
    words_file = lang_dir / "words.txt"

    if not text_file.is_file():
        print(f"ERROR: {text_file} not found", file=sys.stderr)
        sys.exit(1)
    if not words_file.is_file():
        print(f"ERROR: {words_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading word symbol table from {words_file}")
    word_sym_table = k2.SymbolTable.from_file(words_file)

    # Filter out special/reserved symbols.
    excluded = {"<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>"}
    words = [w for w in word_sym_table.symbols if w not in excluded]

    print(f"Generating character token table from {text_file}")
    token_sym_table = generate_tokens(text_file)

    print(f"Building lexicon ({len(words)} words)")
    lexicon = generate_lexicon(token_sym_table, words)

    print("Adding disambiguation symbols")
    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    # Append disambiguation tokens (#0, #1, ..., #max_disambig) to the token table.
    next_token_id = max(token_sym_table.values()) + 1
    for i in range(max_disambig + 1):
        sym = f"#{i}"
        assert sym not in token_sym_table, f"Symbol {sym} already in token table"
        token_sym_table[sym] = next_token_id
        next_token_id += 1

    # Ensure #0, <s>, </s> are in the word symbol table (required for G composition).
    for w in ("#0", "<s>", "</s>"):
        if w not in word_sym_table:
            word_sym_table.add(w)

    # Build word2id dict from symbol table (k2.SymbolTable supports [] lookup).
    word2id = {w: word_sym_table[w] for w in word_sym_table.symbols}

    print(f"Writing outputs to {lang_dir}/")
    write_mapping(lang_dir / "tokens.txt", token_sym_table)
    write_lexicon(lang_dir / "lexicon.txt", lexicon)
    write_lexicon(lang_dir / "lexicon_disambig.txt", lexicon_disambig)

    print("Building L (lexicon FST without silence)")
    L = lexicon_to_fst_no_sil(lexicon, token2id=token_sym_table, word2id=word2id)

    print("Building L_disambig (lexicon FST with #0 self-loops)")
    L_disambig = lexicon_to_fst_no_sil(
        lexicon_disambig,
        token2id=token_sym_table,
        word2id=word2id,
        need_self_loops=True,
    )

    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")
    print("Done.")
    print(f"  tokens.txt           : {lang_dir / 'tokens.txt'}")
    print(f"  lexicon.txt          : {lang_dir / 'lexicon.txt'}")
    print(f"  lexicon_disambig.txt : {lang_dir / 'lexicon_disambig.txt'}")
    print(f"  L.pt                 : {lang_dir / 'L.pt'}")
    print(f"  L_disambig.pt        : {lang_dir / 'L_disambig.pt'}")


if __name__ == "__main__":
    main()
