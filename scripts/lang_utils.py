#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for language directory preparation and HLG graph building.

Adapted from the Icefall AISHELL recipe:
  https://github.com/k2-fsa/icefall/tree/master/egs/aishell/ASR

No dependency on the icefall Python package — all logic is self-contained.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import k2
import torch

# Type alias: a lexicon is a list of (word, [token, ...]) pairs.
Lexicon = List[Tuple[str, List[str]]]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_lexicon(filename) -> Lexicon:
    """Read a lexicon file.

    Each line has the format ``word token1 token2 ...`` (space separated).

    Args:
        filename: Path to the lexicon file.

    Returns:
        List of (word, [tokens]) tuples.
    """
    ans: Lexicon = []
    whitespace = re.compile(r"[ \t]+")
    with open(filename, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            a = whitespace.split(line.strip())
            if not a:
                continue
            if len(a) < 2:
                print(
                    f"WARNING: bad line {lineno} in {filename}: {line.rstrip()!r} "
                    "(expected at least 2 fields, skipping)",
                    file=sys.stderr,
                )
                continue
            word = a[0]
            if word == "<eps>":
                print(
                    f"WARNING: skipping <eps> entry in {filename}",
                    file=sys.stderr,
                )
                continue
            ans.append((word, a[1:]))
    return ans


def write_lexicon(filename, lexicon: Lexicon) -> None:
    """Write a lexicon to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        for word, tokens in lexicon:
            f.write(f"{word} {' '.join(tokens)}\n")


def write_mapping(filename, sym2id: Dict[str, int]) -> None:
    """Write a symbol-to-ID mapping (one ``symbol id`` per line)."""
    with open(filename, "w", encoding="utf-8") as f:
        for sym, idx in sym2id.items():
            f.write(f"{sym} {idx}\n")


# ---------------------------------------------------------------------------
# Symbol table helpers
# ---------------------------------------------------------------------------


def get_tokens(lexicon: Lexicon) -> List[str]:
    """Return a sorted list of unique tokens from *lexicon*."""
    tokens: set = set()
    for _, toks in lexicon:
        tokens.update(toks)
    return sorted(tokens)


def get_words(lexicon: Lexicon) -> List[str]:
    """Return a sorted list of unique words from *lexicon*."""
    return sorted({word for word, _ in lexicon})


def generate_id_map(symbols: List[str]) -> Dict[str, int]:
    """Map each symbol to its index in *symbols* (0-based)."""
    return {sym: i for i, sym in enumerate(symbols)}


# ---------------------------------------------------------------------------
# Disambiguation symbols
# ---------------------------------------------------------------------------


def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    """Add pseudo-token disambiguation symbols (#1, #2, ...) to a lexicon.

    Ensures that no pronunciation is a prefix of another and that all
    pronunciations are unique.  Mirrors Kaldi's ``add_lex_disambig.pl``.

    Args:
        lexicon: Input lexicon (list of (word, [tokens]) pairs).

    Returns:
        Tuple of:
        - Modified lexicon with disambiguation symbols appended where needed.
        - The maximum disambiguation index used (0 means only #0 is needed,
          which is reserved for the composition self-loop; -1 means none used).
    """
    # Count occurrences of each token sequence.
    count: Dict[str, int] = defaultdict(int)
    for _, tokens in lexicon:
        count[" ".join(tokens)] += 1

    # Record which token sub-sequences appear as a prefix of a longer sequence.
    issubseq: Dict[str, int] = defaultdict(int)
    for _, tokens in lexicon:
        tokens = tokens.copy()
        tokens.pop()
        while tokens:
            issubseq[" ".join(tokens)] = 1
            tokens.pop()

    ans: Lexicon = []
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used: Dict[str, int] = defaultdict(int)

    for word, tokens in lexicon:
        tokenseq = " ".join(tokens)
        assert tokenseq != "", f"Word {word!r} has empty pronunciation"
        if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
            ans.append((word, tokens))
            continue

        cur_disambig = last_used[tokenseq]
        cur_disambig = cur_disambig + 1 if cur_disambig > 0 else first_allowed_disambig
        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used[tokenseq] = cur_disambig
        ans.append((word, (tokenseq + f" #{cur_disambig}").split()))

    return ans, max_disambig


# ---------------------------------------------------------------------------
# Lexicon FST construction
# ---------------------------------------------------------------------------


def add_self_loops(
    arcs: List[List[Any]],
    disambig_token: int,
    disambig_word: int,
) -> List[List[Any]]:
    """Add disambiguation self-loops to states with non-epsilon output arcs.

    Required so that disambiguation symbols propagate correctly through the
    H o L o G composition.  Mirrors Kaldi's ``fstaddselfloops.pl``.

    The self-loop input label is ``disambig_token`` (#0 in tokens.txt) and
    the output label is ``disambig_word`` (#0 in words.txt).

    Args:
        arcs: List of ``[src, dst, ilabel, olabel, score]`` arcs.
        disambig_token: Integer ID of token ``#0``.
        disambig_word: Integer ID of word ``#0``.

    Returns:
        New arc list with self-loops added.
    """
    needs_loop: set = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            needs_loop.add(src)

    loops = [[s, s, disambig_token, disambig_word, 0] for s in needs_loop]
    return arcs + loops


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Build a lexicon FST (k2 format) without silence transitions.

    Used for character- or BPE-based lexicons where inter-word silence is
    not modelled.

    Args:
        lexicon: List of (word, [tokens]) pairs.
        token2id: Mapping from token strings to integer IDs.
        word2id: Mapping from word strings to integer IDs.
        need_self_loops: If True, add ``#0`` disambiguation self-loops.

    Returns:
        k2.Fsa representing the lexicon transducer.
    """
    loop_state = 0
    next_state = 1
    arcs: List[List[Any]] = []
    eps = 0

    for word, pieces in lexicon:
        assert len(pieces) > 0, f"Word {word!r} has no pronunciation"
        cur_state = loop_state
        word_id = word2id[word]
        piece_ids = [
            token2id[p] if p in token2id else token2id["<unk>"] for p in pieces
        ]

        for i in range(len(piece_ids) - 1):
            w = word_id if i == 0 else eps
            arcs.append([cur_state, next_state, piece_ids[i], w, 0])
            cur_state = next_state
            next_state += 1

        i = len(piece_ids) - 1
        w = word_id if i == 0 else eps
        arcs.append([cur_state, loop_state, piece_ids[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(arcs, disambig_token=disambig_token, disambig_word=disambig_word)

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda a: a[0])
    arcs_str = "\n".join(" ".join(str(x) for x in a) for a in arcs)
    return k2.Fsa.from_str(arcs_str, acceptor=False)


def lexicon_to_fst(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    sil_token: str = "SIL",
    sil_prob: float = 0.5,
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Build a lexicon FST (k2 format) with optional silence transitions.

    Used for phone-based lexicons where inter-word silence is modelled.

    Args:
        lexicon: List of (word, [tokens]) pairs.
        token2id: Mapping from token strings to integer IDs.
        word2id: Mapping from word strings to integer IDs.
        sil_token: The silence phone token string.
        sil_prob: Probability of inserting silence at word boundaries.
        need_self_loops: If True, add ``#0`` disambiguation self-loops.

    Returns:
        k2.Fsa representing the lexicon transducer.
    """
    import math

    assert 0.0 < sil_prob < 1.0, "sil_prob must be in (0, 1)"
    sil_score = math.log(sil_prob)
    no_sil_score = math.log(1.0 - sil_prob)

    start_state = 0
    loop_state = 1
    sil_state = 2
    next_state = 3
    arcs: List[List[Any]] = []
    eps = 0

    sil_id = token2id[sil_token]

    arcs.append([start_state, loop_state, eps, eps, no_sil_score])
    arcs.append([start_state, sil_state, eps, eps, sil_score])
    arcs.append([sil_state, loop_state, sil_id, eps, 0])

    for word, tokens in lexicon:
        assert len(tokens) > 0, f"Word {word!r} has no pronunciation"
        cur_state = loop_state
        word_id = word2id[word]
        token_ids = [token2id[t] for t in tokens]

        for i in range(len(token_ids) - 1):
            w = word_id if i == 0 else eps
            arcs.append([cur_state, next_state, token_ids[i], w, 0])
            cur_state = next_state
            next_state += 1

        i = len(token_ids) - 1
        w = word_id if i == 0 else eps
        arcs.append([cur_state, loop_state, token_ids[i], w, no_sil_score])
        arcs.append([cur_state, sil_state, token_ids[i], w, sil_score])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(arcs, disambig_token=disambig_token, disambig_word=disambig_word)

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda a: a[0])
    arcs_str = "\n".join(" ".join(str(x) for x in a) for a in arcs)
    return k2.Fsa.from_str(arcs_str, acceptor=False)


# ---------------------------------------------------------------------------
# LangDir — lightweight helper for compile_hlg.py (replaces icefall.Lexicon)
# ---------------------------------------------------------------------------

_DISAMBIG_RE = re.compile(r"^#\d+$")


class LangDir:
    """Loads a prepared lang directory for HLG graph compilation.

    This is a lightweight replacement for ``icefall.lexicon.Lexicon`` that
    loads ``tokens.txt`` and ``words.txt`` from *lang_dir* and exposes the
    ``token_table``, ``word_table``, and ``tokens`` properties used by
    ``compile_hlg.py``.

    Args:
        lang_dir: Path to the prepared lang directory.
    """

    def __init__(self, lang_dir) -> None:
        lang_dir = Path(lang_dir)
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

    @property
    def tokens(self) -> List[int]:
        """Non-disambiguation, non-epsilon token IDs (sorted ascending).

        Excludes token 0 (epsilon/<blk>) and any disambiguation symbol (``#N``).
        """
        ans = [
            self.token_table[s]
            for s in self.token_table.symbols
            if not _DISAMBIG_RE.match(s) and self.token_table[s] != 0
        ]
        ans.sort()
        return ans
