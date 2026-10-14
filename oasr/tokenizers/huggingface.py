# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace ``tokenizer.json`` tokenizer (AED / LLM checkpoints).

Wraps the ``tokenizers`` fast-tokenizer runtime directly (no ``transformers``
dependency).  Emitted by HF-format converters (Whisper / speech-LLM).

Some HF checkpoints (Qwen2-Audio among them) declare added special tokens only
in ``tokenizer_config.json``'s ``added_tokens_decoder`` — ``transformers``
merges them into the fast tokenizer at load time, and prompt encoding breaks
without them (``<|audio_bos|>`` would BPE-split into plain text).  When the
spec carries a ``tokenizer_config`` file, the missing entries are added here
the same way: in declared-id order, relying on the backend's sequential id
assignment, with a loud warning if any id lands somewhere else.
"""

from __future__ import annotations

import json
import logging
from typing import FrozenSet, List, Optional, Sequence

from .base import Tokenizer, TokenizerSpec
from .registry import register_tokenizer

logger = logging.getLogger(__name__)

#: What a decoder renders an incomplete UTF-8 sequence as.
_REPLACEMENT = "\ufffd"


class HuggingFaceTokenizer(Tokenizer):
    """Wraps a ``tokenizers.Tokenizer`` loaded from a ``tokenizer.json``."""

    def __init__(
        self,
        tokenizer_json_path: str,
        special_ids: Optional[FrozenSet[int]] = None,
        tokenizer_config_path: Optional[str] = None,
    ) -> None:
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "the `tokenizers` package is required for "
                "TokenizerSpec(kind='huggingface'); install it with "
                "`pip install oasr[tokenizers]` or `pip install tokenizers`"
            ) from exc
        self._tok = HFTokenizer.from_file(tokenizer_json_path)
        if tokenizer_config_path:
            self._merge_added_tokens(tokenizer_config_path)
        # HF tokenizers know their own added special tokens; the explicit set is
        # an additional strip list on top of skip_special_tokens=True.
        self._special_ids = frozenset() if special_ids is None else special_ids

    def _merge_added_tokens(self, tokenizer_config_path: str) -> None:
        """Add ``added_tokens_decoder`` entries missing from ``tokenizer.json``."""
        from tokenizers import AddedToken

        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        entries = sorted((cfg.get("added_tokens_decoder") or {}).items(), key=lambda kv: int(kv[0]))
        for tid, info in entries:
            content = info["content"]
            if self._tok.token_to_id(content) is not None:
                continue
            token = AddedToken(
                content,
                single_word=bool(info.get("single_word", False)),
                lstrip=bool(info.get("lstrip", False)),
                rstrip=bool(info.get("rstrip", False)),
                normalized=bool(info.get("normalized", False)),
                special=bool(info.get("special", True)),
            )
            if token.special:
                self._tok.add_special_tokens([token])
            else:
                self._tok.add_tokens([token])
            got = self._tok.token_to_id(content)
            if got != int(tid):
                logger.warning(
                    "added token %r landed at id %s (tokenizer_config.json "
                    "declares %s) — encode/decode ids may not match the "
                    "checkpoint",
                    content,
                    got,
                    tid,
                )

    @classmethod
    def from_spec(cls, spec: TokenizerSpec) -> "HuggingFaceTokenizer":
        ids = spec.options.get("special_ids")
        return cls(
            spec.files["tokenizer"],
            special_ids=None if ids is None else frozenset(int(i) for i in ids),
            tokenizer_config_path=spec.files.get("tokenizer_config"),
        )

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size())

    @property
    def special_ids(self) -> FrozenSet[int]:
        return self._special_ids

    def decode(self, ids: Sequence[int]) -> str:
        filtered = [int(t) for t in ids if t not in self._special_ids]
        if not filtered:
            return ""
        return self._tok.decode(filtered, skip_special_tokens=True)

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    #: Ids re-decoded on each incremental call, before the newly added ones.
    #: Byte-level BPE means one token can be a fragment of a UTF-8 sequence, so
    #: a token cannot be decoded alone; a bounded left window is enough because
    #: no BPE merge spans more than a few tokens.  Large enough to be safe,
    #: small enough that per-partial cost stops growing with the transcript.
    _INCREMENTAL_WINDOW = 16

    def decode_incremental(self, new_ids, state):
        """Windowed incremental decode (HF ``TextIteratorStreamer`` strategy).

        Decoding each token alone is wrong here: byte-level BPE emits fragments
        of multi-byte characters, which render as U+FFFD in isolation.  Decoding
        the full prefix every tick is correct but Θ(n²) over a generation.  So
        decode a bounded suffix window, diff it against the same window's
        previous rendering, and append the difference — cost per partial is
        constant in the transcript length instead of linear.

        A window that ends partway through a character renders that tail as
        U+FFFD until a later token completes it, so the tail is held back from
        the returned delta, as ``TextIteratorStreamer`` holds it back from a
        stream.  Diffing it in made the completing token's rendering disagree
        with the previous one, which sent every CJK character and emoji down
        the full re-decode and returned the whole transcript as that token's
        piece.  ``state["text"]`` still includes the tail, so it always equals
        ``decode(all_ids)``; :meth:`token_pieces` flushes it.
        """
        ids = state.setdefault("ids", [])
        ids.extend(int(i) for i in new_ids)
        # The window is anchored at a fixed index, not a fixed length — a window
        # that slides every call re-bases the string it is being diffed against,
        # so the "new suffix" would be computed against a different span each
        # time.  With a stable anchor, ``decode(ids[anchor:])`` only ever grows.
        anchor = state.get("anchor", 0)
        rendered = self.decode(ids[anchor:])
        stable = rendered.rstrip(_REPLACEMENT)
        tail = rendered[len(stable) :]
        prev_window = state.get("window_text", "")
        if not stable.startswith(prev_window):
            # Appending re-rendered part of the window (a merge absorbed an
            # earlier token).  Recompute in full rather than emit a wrong delta.
            full = self.decode(ids)
            full_stable = full.rstrip(_REPLACEMENT)
            prev = state.get("stable", "")
            state["text"] = full
            state["stable"] = full_stable
            state["window_text"] = stable
            return full_stable[len(prev) :] if full_stable.startswith(prev) else full_stable

        delta = stable[len(prev_window) :]
        # ``pop`` then ``+=``, not ``get`` then ``+``.  ``a + b`` always
        # allocates and copies both operands, so the windowed decode above was
        # constant-cost while the accumulation beneath it stayed linear in the
        # transcript.  Growing in place needs the string to be the sole
        # reference, which is what popping it out of the dict gives — both
        # keys, since ``text`` is the same object whenever nothing is held back.
        state.pop("text", None)
        emitted = state.pop("stable", "")
        emitted += delta
        state["stable"] = emitted
        state["text"] = emitted + tail if tail else emitted
        if len(ids) - anchor > 2 * self._INCREMENTAL_WINDOW:
            # Re-anchor, keeping a window of left context so the next decode
            # still has enough neighbours to render byte-BPE fragments.  The
            # accumulated ``text`` is authoritative and unaffected.
            anchor = len(ids) - self._INCREMENTAL_WINDOW
            state["anchor"] = anchor
            state["window_text"] = self.decode(ids[anchor:]).rstrip(_REPLACEMENT)
        else:
            state["window_text"] = stable
        return delta

    def token_pieces(self, ids: Sequence[int]) -> List[str]:
        """:meth:`decode_incremental`'s deltas, one id at a time, plus the tail.

        A sequence that stops partway through a character (a generation cut at
        its cap) still renders that tail in ``decode``; it belongs to the last
        token, so the pieces concatenate to ``decode(ids)`` either way.
        """
        state = self.new_decode_state()
        pieces = [self.decode_incremental((int(i),), state) for i in ids]
        tail = state.get("text", "")[len(state.get("stable", "")) :]
        if tail and pieces:
            pieces[-1] += tail
        return pieces


register_tokenizer("huggingface", HuggingFaceTokenizer.from_spec)
