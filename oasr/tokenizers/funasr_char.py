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

import json
from typing import FrozenSet, Iterable, List, Sequence

from .base import DEFAULT_SPECIAL_IDS, Tokenizer, TokenizerSpec
from .registry import register_tokenizer

_STRIP_TOKENS = {"<s>", "</s>", "<unk>", "<OOV>"}


# --- faithful ports of funasr.utils.postprocess_utils -----------------------


def _is_chinese(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "0" <= ch <= "9" or ch == "@"


def _clean(word: str) -> str:
    for junk in (" ", "</s>", "<s>", "<unk>", "<OOV>"):
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


def _abbr_dispose(words: List[str]) -> List[str]:
    """Collapse spelled-out single-letter runs into abbreviations (b b c → BBC)."""
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
    if _is_all_chinese(middle):
        word_lists = [ch.replace(" ", "") for ch in middle]
    elif _is_all_alpha(middle):
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
            if _is_all_chinese(ch):
                if alpha_blank:
                    word_lists.pop()
                word_lists.append(ch)
                alpha_blank = False
            elif "@@" in ch:
                word_item += ch.replace("@@", "")
                alpha_blank = False
            elif _is_all_alpha(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(" ")
                word_item = ""
                alpha_blank = True
            else:
                word_lists.append(ch)

    word_lists = _abbr_dispose(word_lists)
    return "".join(word_lists).strip()


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
