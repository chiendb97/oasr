#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Prepare a phone-based lang directory for CTC decoding.

Adapted from the Icefall AISHELL recipe:
  egs/aishell/ASR/local/prepare_lang.py

Given a lang directory that contains:
  - ``lexicon.txt`` — word-to-phone mapping (``word phone1 phone2 ...`` per line)

This script generates:
  - ``tokens.txt``          — phone token-to-ID mapping
  - ``words.txt``           — word-to-ID mapping
  - ``lexicon_disambig.txt``— lexicon with disambiguation symbols
  - ``L.pt``                — lexicon FST with optional silence (k2 format)
  - ``L_disambig.pt``       — lexicon FST with #0 self-loops (k2 format)

Usage::

    python scripts/prepare_lang.py --lang-dir data/lang_phone [--sil-token SIL] [--sil-prob 0.5]
"""

import argparse
import sys
from pathlib import Path

# Add scripts/ to the path so lang_utils can be imported.
sys.path.insert(0, str(Path(__file__).parent))

import torch

from lang_utils import (
    add_disambig_symbols,
    generate_id_map,
    get_tokens,
    get_words,
    lexicon_to_fst,
    read_lexicon,
    write_lexicon,
    write_mapping,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang-dir",
        type=str,
        required=True,
        help="Input/output lang directory. Must contain 'lexicon.txt'.",
    )
    parser.add_argument(
        "--sil-token",
        type=str,
        default="SIL",
        help="Silence phone token (default: SIL).",
    )
    parser.add_argument(
        "--sil-prob",
        type=float,
        default=0.5,
        help="Probability of inserting silence at word boundaries (default: 0.5).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    lexicon_file = lang_dir / "lexicon.txt"

    if not lexicon_file.is_file():
        print(f"ERROR: {lexicon_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading lexicon from {lexicon_file}", file=sys.stderr)
    lexicon = read_lexicon(lexicon_file)

    tokens = get_tokens(lexicon)
    words = get_words(lexicon)

    if args.sil_token not in tokens:
        print(
            f"ERROR: silence token '{args.sil_token}' not found in any lexicon entry. "
            "Add a silence entry (e.g. 'SIL SIL') to lexicon.txt or use --sil-token.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Adding disambiguation symbols", file=sys.stderr)
    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    # Append disambiguation tokens (#0, #1, ...) to the token list.
    for i in range(max_disambig + 1):
        tokens.append(f"#{i}")

    # Build token table: <eps>=0, then sorted phones, then disambig symbols.
    assert "<eps>" not in tokens
    tokens = ["<eps>"] + tokens

    # Build word table: <eps>=0, words sorted, then special symbols.
    assert all(w not in words for w in ("<eps>", "#0", "<s>", "</s>"))
    words = ["<eps>"] + words + ["#0", "<s>", "</s>"]

    token2id = generate_id_map(tokens)
    word2id = generate_id_map(words)

    print(f"Writing outputs to {lang_dir}/", file=sys.stderr)
    write_mapping(lang_dir / "tokens.txt", token2id)
    write_mapping(lang_dir / "words.txt", word2id)
    write_lexicon(lang_dir / "lexicon_disambig.txt", lexicon_disambig)

    print(f"Building L (with silence, sil_prob={args.sil_prob})", file=sys.stderr)
    L = lexicon_to_fst(
        lexicon,
        token2id=token2id,
        word2id=word2id,
        sil_token=args.sil_token,
        sil_prob=args.sil_prob,
    )

    print("Building L_disambig (with #0 self-loops)", file=sys.stderr)
    L_disambig = lexicon_to_fst(
        lexicon_disambig,
        token2id=token2id,
        word2id=word2id,
        sil_token=args.sil_token,
        sil_prob=args.sil_prob,
        need_self_loops=True,
    )

    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")
    print("Done.", file=sys.stderr)
    print(f"  tokens.txt           : {lang_dir / 'tokens.txt'}", file=sys.stderr)
    print(f"  words.txt            : {lang_dir / 'words.txt'}", file=sys.stderr)
    print(f"  lexicon_disambig.txt : {lang_dir / 'lexicon_disambig.txt'}", file=sys.stderr)
    print(f"  L.pt                 : {lang_dir / 'L.pt'}", file=sys.stderr)
    print(f"  L_disambig.pt        : {lang_dir / 'L_disambig.pt'}", file=sys.stderr)


if __name__ == "__main__":
    main()
