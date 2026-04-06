#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Build a word symbol table (``words.txt``) from plain-text training data.

Reads one or more text files (one sentence per line, space-separated words,
no utterance IDs) and writes ``words.txt`` to ``--lang-dir``.

The output format is ``word id`` per line.  The layout is:

  <eps>   0           epsilon / no-symbol  (leading)
  <UNK>   1           unknown word         (leading)
  word1   2           real words, sorted alphabetically
  ...
  wordN   N+1
  #0      N+2         disambiguation symbol — MUST be after real words so that
  <s>     N+3         compile_hlg.py can use its ID as the first-disambig boundary
  </s>    N+4         sentence end

Usage::

    python scripts/prepare_words.py \\
        --text train-clean-100/text train-clean-360/text train-other-500/text \\
        --lang-dir data/lang_bpe

    # Discard rare words (appear fewer than N times)
    python scripts/prepare_words.py \\
        --text corpus/text \\
        --lang-dir data/lang_bpe \\
        --min-count 2
"""

import argparse
import sys
from pathlib import Path


# Leading symbols: fixed low IDs (must stay before real words).
LEADING_SYMBOLS = ["<eps>", "<UNK>"]
# Trailing symbols: disambiguation / sentence-boundary — MUST come after all
# real words so that compile_hlg.py can use their ID as the first_disambig
# boundary (it zeros out all aux_labels >= first_word_disambig_id).
TRAILING_SYMBOLS = ["#0", "<s>", "</s>"]


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more plain-text files (one sentence per line, no utterance IDs).",
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        required=True,
        help="Output directory; 'words.txt' will be written here.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        metavar="N",
        help="Minimum word frequency to include (default: 1 — keep all words).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    lang_dir.mkdir(parents=True, exist_ok=True)

    word_counts: dict = {}
    n_sentences = 0

    for text_file in args.text:
        path = Path(text_file)
        if not path.is_file():
            print(f"ERROR: text file not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_sentences += 1
                for word in line.split():
                    word_counts[word] = word_counts.get(word, 0) + 1

    all_special = set(LEADING_SYMBOLS + TRAILING_SYMBOLS)
    words = sorted(
        w for w, c in word_counts.items()
        if c >= args.min_count and w not in all_special
    )

    out_path = lang_dir / "words.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        # Leading specials (<eps>=0, <UNK>=1)
        for i, sym in enumerate(LEADING_SYMBOLS):
            f.write(f"{sym} {i}\n")
        # Real words (IDs start right after leading specials)
        base = len(LEADING_SYMBOLS)
        for i, word in enumerate(words, start=base):
            f.write(f"{word} {i}\n")
        # Trailing specials (#0, <s>, </s>) — placed after real words so that
        # compile_hlg.py's first_word_disambig_id boundary is above all real words.
        trailing_start = base + len(words)
        for i, sym in enumerate(TRAILING_SYMBOLS, start=trailing_start):
            f.write(f"{sym} {i}\n")

    total = len(LEADING_SYMBOLS) + len(words) + len(TRAILING_SYMBOLS)
    print(
        f"Wrote {out_path}  "
        f"({total} entries, "
        f"{len(words)} real words from {n_sentences} sentences)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
