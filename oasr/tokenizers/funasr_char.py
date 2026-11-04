# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""FunASR char tokenizer (Paraformer ``tokens.json`` vocabularies).

Decoding mirrors FunASR's pipeline exactly: ids → token strings →
``sentence_postprocess`` (ported from ``funasr.utils.postprocess_utils``),
which joins Chinese characters directly, merges English ``@@`` subword pieces
into words separated by single spaces, and collapses single-letter runs into
abbreviations (``b b c`` → ``BBC``).

Encoding is best-effort (hotword boosting / prompts): whitespace-split words
are looked up whole, falling back to per-character lookup with ``<unk>`` for
misses.  The FunASR ``seg_dict`` word-segmentation used at training time is
not applied.
"""

from __future__ import annotations

import functools
import json
from typing import FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .base import DEFAULT_SPECIAL_IDS, Tokenizer, TokenizerSpec
from .registry import register_tokenizer

_STRIP_TOKENS = {"<s>", "</s>", "<unk>", "<OOV>"}


# --- faithful ports of funasr.utils.postprocess_utils -----------------------


def _is_chinese(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "0" <= ch <= "9" or ch == "@"


_JUNK = (" ", "</s>", "<s>", "<unk>", "<OOV>")


def _clean(word: str) -> str:
    # Every junk string but the space contains "<", so a word holding neither
    # has nothing to remove and comes back unchanged.  Nearly every token and
    # every single character take this exit.  The five ``replace`` calls,
    # made per token and then again per *character* in the mixed branch, were
    # most of the decode's time (audit L7).
    if " " not in word and "<" not in word:
        return word
    for junk in _JUNK:
        word = word.replace(junk, "")
    return word


def _is_all_chinese(word: Iterable[str]) -> bool:
    # NB: iterating a *string* argument visits characters, a *list* visits
    # tokens — the FunASR reference relies on exactly this duality (token
    # lists at the top level, per-token strings inside the mixed branch).
    cleaned = [_clean(w) for w in word]
    if not cleaned:
        return False
    return all(_is_chinese(w) for w in cleaned)


def _is_all_alpha(word: Iterable[str]) -> bool:
    cleaned = [_clean(w) for w in word]
    if not cleaned:
        return False
    for w in cleaned:
        if not w.isalpha() and w != "'":
            return False
        if w.isalpha() and _is_chinese(w):
            return False
    return True


# --- the same predicates, cached per token, for the decode path (audit L7) ----
#
# ``sentence_postprocess`` classifies each token twice. ``_is_all_chinese`` and
# ``_is_all_alpha`` run once over the token list, then once more per token in
# the mixed branch, where a string argument is walked character by character.
# Both are pure functions of the string, and the strings come from a finite
# vocabulary, so each answer is computed once per token and then looked up.
# The ports above stay the reference; ``tests/models/arch/test_paraformer.py``
# decodes through both and requires the same text.

_TOKEN_CACHE = 1 << 16


@functools.lru_cache(maxsize=_TOKEN_CACHE)
def _chinese_token(token: str) -> bool:
    """One element of ``_is_all_chinese(list)``: the cleaned token is "Chinese"."""
    return _is_chinese(_clean(token))


@functools.lru_cache(maxsize=_TOKEN_CACHE)
def _alpha_token(token: str) -> bool:
    """One element of ``_is_all_alpha(list)``: the cleaned token passes both tests."""
    w = _clean(token)
    if not w.isalpha() and w != "'":
        return False
    return not (w.isalpha() and _is_chinese(w))


def _tokens_all_chinese(tokens: Sequence[str]) -> bool:
    """``_is_all_chinese`` over a token list."""
    return bool(tokens) and all(_chinese_token(t) for t in tokens)


def _tokens_all_alpha(tokens: Sequence[str]) -> bool:
    """``_is_all_alpha`` over a token list."""
    return bool(tokens) and all(_alpha_token(t) for t in tokens)


#: ``_is_all_chinese`` / ``_is_all_alpha`` of one token *string*, which they walk
#: character by character.
_token_all_chinese = functools.lru_cache(maxsize=_TOKEN_CACHE)(_is_all_chinese)
_token_all_alpha = functools.lru_cache(maxsize=_TOKEN_CACHE)(_is_all_alpha)


def _abbr_spans(words: Sequence[str]) -> Tuple[Set[int], Set[int]]:
    """Where :func:`_abbr_dispose` starts and ends each abbreviation, as index sets.

    FunASR keeps these as lists and tests membership per word, which is
    quadratic in the number of abbreviations; the sets answer the same question.
    """
    words_size = len(words)
    abbr_begin: List[int] = []
    abbr_end: List[int] = []
    last_num = -1
    for num in range(words_size):
        if num <= last_num:
            continue
        if len(words[num]) == 1 and words[num].encode("utf-8").isalpha():
            if (
                num + 1 < words_size
                and words[num + 1] == " "
                and num + 2 < words_size
                and len(words[num + 2]) == 1
                and words[num + 2].encode("utf-8").isalpha()
            ):
                abbr_begin.append(num)
                num += 2
                abbr_end.append(num)
                while True:
                    num += 1
                    if num < words_size and words[num] == " ":
                        num += 1
                        if (
                            num < words_size
                            and len(words[num]) == 1
                            and words[num].encode("utf-8").isalpha()
                        ):
                            abbr_end.pop()
                            abbr_end.append(num)
                            last_num = num
                        else:
                            break
                    else:
                        break
    return set(abbr_begin), set(abbr_end)


def _abbr_dispose(words: List[str]) -> List[str]:
    """Collapse spelled-out single-letter runs into abbreviations (b b c → BBC)."""
    words_size = len(words)
    abbr_begin, abbr_end = _abbr_spans(words)
    out: List[str] = []
    last_num = -1
    for num in range(words_size):
        if num <= last_num:
            continue
        if num in abbr_begin:
            abbr_word = words[num].upper()
            num += 1
            while num < words_size:
                if num in abbr_end:
                    abbr_word += words[num].upper()
                    last_num = num
                    break
                if words[num].encode("utf-8").isalpha():
                    abbr_word += words[num].upper()
                num += 1
            out.append(abbr_word)
        else:
            out.append(words[num])
    return out


def sentence_postprocess(tokens: Sequence[str]) -> str:
    """FunASR ``sentence_postprocess`` (text-only path)."""
    middle = [t for t in tokens if t not in _STRIP_TOKENS]

    word_lists: List[str] = []
    word_item = ""
    if _tokens_all_chinese(middle):
        word_lists = [ch.replace(" ", "") for ch in middle]
    elif _tokens_all_alpha(middle):
        for ch in middle:
            if "@@" in ch:
                word_item += ch.replace("@@", "")
            else:
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(" ")
                word_item = ""
    else:
        alpha_blank = False
        for ch in middle:
            if _token_all_chinese(ch):
                if alpha_blank:
                    word_lists.pop()
                word_lists.append(ch)
                alpha_blank = False
            elif "@@" in ch:
                word_item += ch.replace("@@", "")
                alpha_blank = False
            elif _token_all_alpha(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(" ")
                word_item = ""
                alpha_blank = True
            else:
                word_lists.append(ch)

    word_lists = _abbr_dispose(word_lists)
    return "".join(word_lists).strip()


# --- per-token pieces ----------------------------------------------------------

#: ``(text, token)``: a run of output text and the position of the token that
#: produced it, or ``None`` for a separator space the pipeline inserted.
_Fragment = Tuple[str, Optional[int]]


def sentence_pieces(tokens: Sequence[str]) -> List[str]:
    """Each token's share of :func:`sentence_postprocess`'s text, in one pass.

    ``"".join(sentence_pieces(tokens)) == sentence_postprocess(tokens)``.  The
    same pipeline, with every string replaced by fragments that remember which
    token produced them, so an abbreviation merge (``b b c`` → ``BBC``) gives
    each letter to its own token instead of rewriting the text before it.

    The pieces run in token order and the text does not always: a pending
    ``@@`` piece is written out after a Chinese character that arrived later
    (``i@@ 打 pho`` → ``打ipho``).  Such a fragment, like a separator space,
    joins the next fragment that *is* in order, which is the token whose
    arrival put it in the text.
    """
    pieces = [""] * len(tokens)
    kept = [i for i, t in enumerate(tokens) if t not in _STRIP_TOKENS]
    middle = [tokens[i] for i in kept]

    words: List[List[_Fragment]] = []
    item: List[_Fragment] = []
    if _tokens_all_chinese(middle):
        words = [[(tokens[i].replace(" ", ""), i)] for i in kept]
    elif _tokens_all_alpha(middle):
        for i in kept:
            ch = tokens[i]
            if "@@" in ch:
                item.append((ch.replace("@@", ""), i))
            else:
                item.append((ch, i))
                words += [item, [(" ", None)]]
                item = []
    else:
        alpha_blank = False
        for i in kept:
            ch = tokens[i]
            if _token_all_chinese(ch):
                if alpha_blank:
                    words.pop()
                words.append([(ch, i)])
                alpha_blank = False
            elif "@@" in ch:
                item.append((ch.replace("@@", ""), i))
                alpha_blank = False
            elif _token_all_alpha(ch):
                item.append((ch, i))
                words += [item, [(" ", None)]]
                item = []
                alpha_blank = True
            else:
                words.append([(ch, i)])

    fragments = [f for word in _abbr_dispose_fragments(words) for f in word]
    _strip_fragments(fragments)

    latest = -1
    pending = ""
    for text, token in fragments:
        if token is None or token < latest:
            pending += text
            continue
        latest = token
        pieces[token] += pending + text
        pending = ""
    if pending and latest >= 0:
        pieces[latest] += pending
    return pieces


def _abbr_dispose_fragments(words: List[List[_Fragment]]) -> List[List[_Fragment]]:
    """:func:`_abbr_dispose` over fragment lists; each letter keeps its token."""
    texts = ["".join(text for text, _ in word) for word in words]
    abbr_begin, abbr_end = _abbr_spans(texts)

    def upper(word: List[_Fragment]) -> List[_Fragment]:
        return [(text.upper(), token) for text, token in word]

    out: List[List[_Fragment]] = []
    last_num = -1
    for num in range(len(words)):
        if num <= last_num:
            continue
        if num in abbr_begin:
            abbr_word = upper(words[num])
            num += 1
            while num < len(words):
                if num in abbr_end:
                    abbr_word += upper(words[num])
                    last_num = num
                    break
                if texts[num].encode("utf-8").isalpha():
                    abbr_word += upper(words[num])
                num += 1
            out.append(abbr_word)
        else:
            out.append(words[num])
    return out


def _strip_fragments(fragments: List[_Fragment]) -> None:
    """``str.strip`` of the joined text, applied to the fragments in place."""
    for order in (range(len(fragments)), range(len(fragments) - 1, -1, -1)):
        for i in order:
            text, token = fragments[i]
            kept = text.lstrip() if order.step == 1 else text.rstrip()
            fragments[i] = (kept, token)
            if kept:
                break


# --- tokenizer ---------------------------------------------------------------


class FunASRCharTokenizer(Tokenizer):
    """Char/subword tokenizer over a FunASR ``tokens.json`` vocabulary."""

    def __init__(
        self,
        tokens_file: str,
        special_ids: FrozenSet[int] = DEFAULT_SPECIAL_IDS,
        unk_symbol: str = "<unk>",
    ) -> None:
        with open(tokens_file, "r", encoding="utf-8") as f:
            self._tokens: List[str] = json.load(f)
        self._ids = {tok: i for i, tok in enumerate(self._tokens)}
        self._special_ids = frozenset(special_ids)
        self._unk_id = self._ids.get(unk_symbol)

    @classmethod
    def from_spec(cls, spec: TokenizerSpec) -> "FunASRCharTokenizer":
        return cls(
            tokens_file=spec.files["tokens"],
            special_ids=spec.special_ids(),
            unk_symbol=spec.options.get("unk_symbol", "<unk>"),
        )

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    @property
    def special_ids(self) -> FrozenSet[int]:
        return self._special_ids

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [
            self._tokens[i]
            for i in ids
            if i not in self._special_ids and 0 <= i < len(self._tokens)
        ]
        return sentence_postprocess(tokens)

    def token_pieces(self, ids: Sequence[int]) -> List[str]:
        """:func:`sentence_pieces` over the ids ``decode`` keeps; the rest own ``""``.

        The base class drives :meth:`decode_incremental`, which re-renders the
        whole prefix per id (quadratic) and, when an abbreviation merge
        rewrites it, hands the rewritten prefix to the merging id: ``b b c``
        came out as ``['b', 'BB', 'C']``, joining to ``bBBC``.
        """
        kept = [
            pos
            for pos, i in enumerate(ids)
            if i not in self._special_ids and 0 <= i < len(self._tokens)
        ]
        pieces = [""] * len(ids)
        for pos, piece in zip(kept, sentence_pieces([self._tokens[ids[pos]] for pos in kept])):
            pieces[pos] = piece
        return pieces

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in text.split():
            hit = self._ids.get(word)
            if hit is not None:
                ids.append(hit)
                continue
            for ch in word:
                cid = self._ids.get(ch, self._unk_id)
                if cid is not None:
                    ids.append(cid)
        return ids


register_tokenizer("funasr_char", FunASRCharTokenizer.from_spec)
