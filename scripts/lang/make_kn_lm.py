#!/usr/bin/env python3
# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
#           2018  Ruizhe Huang
#           2024  OASR Authors
# Apache 2.0.
"""
Train a Kneser-Ney smoothed n-gram language model and write ARPA output.

This is a pure-Python implementation of unmodified Kneser-Ney smoothing,
equivalent to the following SRILM command::

    ngram-count -order 3 -kn-modify-counts-at-end -ukndiscount \\
        -gt1min 0 -gt2min 0 -gt3min 0 \\
        -text corpus.txt -lm lm.arpa

Adapted from ``icefall/shared/make_kn_lm.py`` (self-contained, no external
dependencies beyond the Python standard library).

Usage::

    # Single file
    python scripts/make_kn_lm.py \\
        -ngram-order 3 \\
        -text data/lang/text \\
        -lm data/lm/G_3_gram.arpa

    # Multiple files (e.g. LibriSpeech train splits)
    python scripts/make_kn_lm.py \\
        -ngram-order 3 \\
        -text train-clean-100/text train-clean-360/text train-other-500/text \\
        -lm data/lm/G_3_gram.arpa

Input text files must contain plain text with one sentence per line
(space-separated words).  Utterance IDs must be stripped beforehand.
"""

import argparse
import io
import math
import os
import re
import sys
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "-ngram-order",
    type=int,
    default=3,
    choices=[1, 2, 3, 4, 5, 6, 7],
    help="Order of n-gram (default: 3).",
)
parser.add_argument(
    "-text",
    type=str,
    nargs="*",
    default=None,
    metavar="FILE",
    help="One or more corpus files (one sentence per line, plain text). "
    "Reads from stdin if not specified.",
)
parser.add_argument(
    "-lm",
    type=str,
    default=None,
    help="Path to output ARPA file. Writes to stdout if not specified.",
)
parser.add_argument(
    "-verbose",
    type=int,
    default=0,
    choices=[0, 1, 2, 3, 4, 5],
    help="Verbosity level (default: 0).",
)
args = parser.parse_args()

# Use latin-1 for encoding-agnostic byte-stream I/O (matches Kaldi convention).
default_encoding = "latin-1"
strip_chars = " \t\r\n"
whitespace = re.compile(r"[ \t]+")


class CountsForHistory:
    """Counts and statistics for a single n-gram history state."""

    def __init__(self):
        self.word_to_count: dict = defaultdict(int)
        self.word_to_context: dict = defaultdict(set)
        self.word_to_f: dict = {}   # discounted probability
        self.word_to_bow: dict = {} # back-off weight
        self.total_count: int = 0

    def words(self):
        return self.word_to_count.keys()

    def add_count(self, predicted_word, context_word, count):
        assert count >= 0
        self.total_count += count
        self.word_to_count[predicted_word] += count
        if context_word is not None:
            self.word_to_context[predicted_word].add(context_word)


class NgramCounts:
    """Accumulates n-gram counts from a corpus and computes KN-smoothed LM."""

    def __init__(self, ngram_order, bos_symbol="<s>", eos_symbol="</s>"):
        assert ngram_order >= 1
        self.ngram_order = ngram_order
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol
        self.counts = [defaultdict(lambda: CountsForHistory()) for _ in range(ngram_order)]
        self.d: list = []

    def add_count(self, history, predicted_word, context_word, count):
        self.counts[len(history)][history].add_count(predicted_word, context_word, count)

    def add_raw_counts_from_line(self, line):
        if line == "":
            words = [self.bos_symbol, self.eos_symbol]
        else:
            words = [self.bos_symbol] + whitespace.split(line) + [self.eos_symbol]

        for i in range(len(words)):
            for n in range(1, self.ngram_order + 1):
                if i + n > len(words):
                    break
                ngram = words[i : i + n]
                predicted_word = ngram[-1]
                history = tuple(ngram[:-1])
                context_word = None if (i == 0 or n == self.ngram_order) else words[i - 1]
                self.add_count(history, predicted_word, context_word, 1)

    def add_raw_counts_from_standard_input(self):
        lines = 0
        infile = io.TextIOWrapper(sys.stdin.buffer, encoding=default_encoding)
        for line in infile:
            self.add_raw_counts_from_line(line.strip(strip_chars))
            lines += 1
        if lines == 0 or args.verbose > 0:
            print(f"make_kn_lm.py: processed {lines} lines from stdin", file=sys.stderr)

    def add_raw_counts_from_file(self, filename):
        lines = 0
        with open(filename, encoding=default_encoding) as fp:
            for line in fp:
                line = line.strip(strip_chars)
                if self.ngram_order == 1:
                    self.add_raw_counts_from_line(line.split()[0] if line else "")
                else:
                    self.add_raw_counts_from_line(line)
                lines += 1
        if lines == 0 or args.verbose > 0:
            print(f"make_kn_lm.py: processed {lines} lines from {filename}", file=sys.stderr)

    def cal_discounting_constants(self):
        """Compute per-order KN discounting constant D_n = n1/(n1 + 2*n2)."""
        self.d = [0]  # unigrams: no discounting
        for n in range(1, self.ngram_order):
            n1 = n2 = 0
            for counts_for_hist in self.counts[n].values():
                stat = Counter(counts_for_hist.word_to_count.values())
                n1 += stat[1]
                n2 += stat[2]
            assert n1 + 2 * n2 > 0, f"No counts at order {n+1}"
            self.d.append(max(0.1, float(n1)) / (n1 + 2 * n2))

    def cal_f(self):
        """Compute discounted probability f for each n-gram."""
        # Highest order: f(a_z) = max(c(a_z) - D, 0) / c(a_)
        n = self.ngram_order - 1
        for counts_for_hist in self.counts[n].values():
            for w, c in counts_for_hist.word_to_count.items():
                counts_for_hist.word_to_f[w] = (
                    max(c - self.d[n], 0) / counts_for_hist.total_count
                )

        # Lower orders: f(_z) = max(n(*_z) - D, 0) / n(*_*)
        for n in range(self.ngram_order - 1):
            for counts_for_hist in self.counts[n].values():
                n_star_star = sum(
                    len(counts_for_hist.word_to_context[w])
                    for w in counts_for_hist.word_to_count
                )
                for w in counts_for_hist.word_to_count:
                    if n_star_star != 0:
                        n_star_z = len(counts_for_hist.word_to_context[w])
                        counts_for_hist.word_to_f[w] = (
                            max(n_star_z - self.d[n], 0) / n_star_star
                        )
                    else:
                        # Histories beginning with <s> have no "modified count".
                        n_star_z = counts_for_hist.word_to_count[w]
                        counts_for_hist.word_to_f[w] = (
                            max(n_star_z - self.d[n], 0) / counts_for_hist.total_count
                        )

    def cal_bow(self):
        """Compute back-off weights: bow(a_) = (1 - sum_z f(a_z)) / (1 - sum_z f(_z))."""
        # Highest order: no BOW needed.
        n = self.ngram_order - 1
        for counts_for_hist in self.counts[n].values():
            for w in counts_for_hist.word_to_count:
                counts_for_hist.word_to_bow[w] = None

        # Lower orders.
        for n in range(self.ngram_order - 1):
            for hist, counts_for_hist in self.counts[n].items():
                for w in counts_for_hist.word_to_count:
                    if w == self.eos_symbol:
                        counts_for_hist.word_to_bow[w] = None
                        continue

                    a_ = hist + (w,)
                    a_counts = self.counts[len(a_)][a_]
                    sum_z1_f_a_z = sum(a_counts.word_to_f.values())

                    _ = a_[1:]
                    _counts = self.counts[len(_)][_]
                    sum_z1_f_z = sum(_counts.word_to_f.get(u, 0) for u in a_counts.word_to_f)

                    if sum_z1_f_z < 1:
                        counts_for_hist.word_to_bow[w] = (1.0 - sum_z1_f_a_z) / (
                            1.0 - sum_z1_f_z
                        )
                    else:
                        counts_for_hist.word_to_bow[w] = None

    def print_as_arpa(
        self,
        fout=io.TextIOWrapper(sys.stdout.buffer, encoding="latin-1"),
    ):
        """Write the LM in ARPA format to *fout*."""
        print("\\data\\", file=fout)
        for hist_len in range(self.ngram_order):
            total = sum(
                len(ch.word_to_f) for ch in self.counts[hist_len].values()
            )
            print(f"ngram {hist_len + 1}={total}", file=fout)
        print("", file=fout)

        for hist_len in range(self.ngram_order):
            print(f"\\{hist_len + 1}-grams:", file=fout)
            for hist, counts_for_hist in self.counts[hist_len].items():
                for word in counts_for_hist.word_to_count:
                    ngram = hist + (word,)
                    prob = counts_for_hist.word_to_f[word]
                    bow = counts_for_hist.word_to_bow[word]
                    prob = max(prob, 1e-99)  # avoid log(0)
                    line = f"{'%.7f' % math.log10(prob)}\t{' '.join(ngram)}"
                    if bow is not None:
                        line += f"\t{'%.7f' % math.log10(bow)}"
                    print(line, file=fout)
            print("", file=fout)

        print("\\end\\", file=fout)


if __name__ == "__main__":
    ngram_counts = NgramCounts(args.ngram_order)

    if not args.text:
        ngram_counts.add_raw_counts_from_standard_input()
    else:
        for text_file in args.text:
            if not os.path.isfile(text_file):
                print(f"ERROR: text file not found: {text_file}", file=sys.stderr)
                sys.exit(1)
            ngram_counts.add_raw_counts_from_file(text_file)

    ngram_counts.cal_discounting_constants()
    ngram_counts.cal_f()
    ngram_counts.cal_bow()

    if args.lm is None:
        ngram_counts.print_as_arpa()
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.lm)), exist_ok=True)
        with open(args.lm, "w", encoding=default_encoding) as f:
            ngram_counts.print_as_arpa(fout=f)
        print(f"Wrote ARPA LM to {args.lm}", file=sys.stderr)
