# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer contract + spec (the sixth registry axis).

A :class:`Tokenizer` owns both directions of the text boundary: ``decode``
(token ids → text, every decode family) and ``encode`` (text → token ids,
needed by AED prompts / LLM decoding / hotword boosting).  Which tokenizer a
checkpoint uses is described by a :class:`TokenizerSpec` **emitted by the
checkpoint converter** and carried in the
:class:`~oasr.checkpoints.ConvertedCheckpoint` bundle — the engine builds the
tokenizer from the spec instead of sniffing ``ckpt_dir`` paths.

Implementations live in this package (``symbol_table``, ``sentencepiece``,
``huggingface``; ``whisper`` lands with the Whisper model package) and register
themselves in :mod:`oasr.tokenizers.registry`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, FrozenSet, List, Sequence

# Historical engine-wide default: <blank>=0, <unk>=1, <sos/eos>=2 are stripped
# from decoded output.  Kept as the default for spec-less construction so the
# symbol-table tokenizer is bit-compatible with the legacy ``Detokenizer``.
DEFAULT_SPECIAL_IDS = frozenset([0, 1, 2])


@dataclass
class TokenizerSpec:
    """Serializable description of a checkpoint's tokenizer.

    Parameters
    ----------
    kind : str
        Registry key selecting the implementation (``"symbol_table"``,
        ``"sentencepiece"``, ``"huggingface"``, ...).
    files : dict
        Named asset paths, e.g. ``{"table": "/ckpt/units.txt"}`` or
        ``{"model": "/ckpt/bpe.model"}``.  Absolute when emitted by a
        converter; the native checkpoint format stores them relative to the
        checkpoint dir and resolves on load.
    options : dict
        Implementation options, e.g. ``{"special_ids": [0, 1, 2]}``.
    """

    kind: str
    files: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    def special_ids(self) -> FrozenSet[int]:
        """Special-token ids declared by this spec (falls back to {0, 1, 2})."""
        ids = self.options.get("special_ids")
        return DEFAULT_SPECIAL_IDS if ids is None else frozenset(int(i) for i in ids)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "files": dict(self.files), "options": dict(self.options)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenizerSpec":
        return cls(
            kind=d["kind"], files=dict(d.get("files", {})), options=dict(d.get("options", {}))
        )


class Tokenizer(ABC):
    """Both directions of the text boundary for one checkpoint's vocabulary.

    ``decode`` is universal; ``encode`` is not.  A ``symbol_table`` tokenizer
    (every WeNet checkpoint, most icefall ones) holds only an id→piece map, so it
    can render output but cannot turn a prompt back into ids.  Callers that need
    the encode direction — the speech-LLM prompt template, hotword boosting — must
    test :attr:`supports_encode`, **not** ``hasattr(tok, "encode")``: ``encode`` is
    abstract here, so the attribute always exists and the check always passes.
    """

    #: Whether :meth:`encode` is implemented.  ``False`` means decode-only.
    supports_encode: ClassVar[bool] = True

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def special_ids(self) -> FrozenSet[int]:
        """Token ids :meth:`decode` strips from its output.

        The contract is exactly that — *what decode removes* — so a caller can
        filter a hypothesis itself and get the same tokens ``decode`` would keep.
        Kinds that delegate stripping to an upstream tokenizer still have to
        report the ids it will drop.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Token ids → text (special ids stripped)."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Text → token ids (no special ids added)."""
        raise NotImplementedError

    # Incremental decoders receive only new ids. The default re-decodes and diffs
    # the accumulated prefix, preserving context-sensitive spacing and merges.

    def new_decode_state(self) -> Dict[str, Any]:
        """Fresh per-request state for :meth:`decode_incremental`."""
        return {"ids": [], "text": ""}

    def decode_incremental(self, new_ids: Sequence[int], state: Dict[str, Any]) -> str:
        """Append ids and return text added since the previous call.

        Mutates ``state``; ``state["text"]`` must equal ``decode(all_ids)``, and so
        must the concatenated deltas, except that a byte-level tokenizer may hold
        back a trailing incomplete character until a later id completes it.
        """
        ids = state.setdefault("ids", [])
        ids.extend(int(i) for i in new_ids)
        full = self.decode(ids)
        prev = state.get("text", "")
        state["text"] = full
        # A re-render can rewrite the prefix (e.g. a subword merge).  Emitting
        # the suffix only would silently drop that; report the whole text as the
        # delta in that case and let the caller replace rather than append.
        if full.startswith(prev):
            return full[len(prev) :]
        return full

    def token_pieces(self, ids: Sequence[int]) -> List[str]:
        """Per-token text contributions for one whole sequence, in order.

        ``"".join(token_pieces(ids)) == decode(ids)``: the same contract the
        deltas of :meth:`decode_incremental` obey, materialised in one call.
        Word timings need exactly this — to cut a word out of the *rendered*
        transcript rather than reassembling it from pieces, the grouping has to
        know which token produced each character.  A token that renders to
        nothing (a special id, a byte-BPE fragment absorbed by its neighbour)
        yields ``""`` and owns no characters.

        The default drives :meth:`decode_incremental` one id at a time, which is
        correct for every kind and inherits that method's complexity — constant
        per token where it is overridden, quadratic where it is not.  A
        tokenizer whose rendering is piece-local overrides this with a single
        pass; the rest are correct as they stand.
        """
        state = self.new_decode_state()
        return [self.decode_incremental((int(i),), state) for i in ids]


#: What a decoder renders an incomplete UTF-8 sequence as.
REPLACEMENT_CHAR = "\ufffd"


class WindowedIncrementalTokenizer(Tokenizer):
    """Incremental decoding for a tokenizer whose rendering is local to a few tokens.

    Byte-level BPE emits fragments of multi-byte characters, and SentencePiece's
    byte-fallback pieces do the same, so a token cannot be rendered on its own;
    re-rendering the whole prefix on every call is correct but makes a partial
    linear in the transcript and :meth:`token_pieces` quadratic.  These kinds
    re-render a bounded window instead.  A subclass whose rendering couples to
    something further back than the window -- SentencePiece's leading-space
    rule does -- supplies that through :meth:`_render_window`.
    """

    #: Ids re-decoded on each incremental call, before the newly added ones.
    #: A bounded left window is enough because no merge or byte sequence spans
    #: more than a few tokens.  Large enough to be safe, small enough that
    #: per-partial cost stops growing with the transcript.
    _INCREMENTAL_WINDOW: ClassVar[int] = 16

    def _render_window(self, ids: List[int], anchor: int, state: Dict[str, Any]) -> str:
        """``ids[anchor:]`` rendered as that span renders inside ``decode(ids)``.

        Only the part after the span's first visible character has to agree:
        what came before is already in the window's previous rendering.  The
        default decodes the span on its own.  ``anchor == 0`` is always plain
        :meth:`decode`.
        """
        return self.decode(ids[anchor:])

    def _may_reanchor(self, ids: List[int], anchor: int, state: Dict[str, Any]) -> bool:
        """Whether :meth:`_render_window` can render a window starting at ``anchor``."""
        return True

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
        rendered = self._render_window(ids, anchor, state)
        stable = rendered.rstrip(REPLACEMENT_CHAR)
        tail = rendered[len(stable) :]
        prev_window = state.get("window_text", "")
        if not stable.startswith(prev_window):
            # Appending re-rendered part of the window (a merge absorbed an
            # earlier token).  Recompute in full rather than emit a wrong delta.
            full = self.decode(ids)
            full_stable = full.rstrip(REPLACEMENT_CHAR)
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
        new_anchor = len(ids) - self._INCREMENTAL_WINDOW
        if len(ids) - anchor > 2 * self._INCREMENTAL_WINDOW and self._may_reanchor(
            ids, new_anchor, state
        ):
            # Re-anchor, keeping a window of left context so the next decode
            # still has enough neighbours to render byte-BPE fragments.  The
            # accumulated ``text`` is authoritative and unaffected.
            state["anchor"] = new_anchor
            state["window_text"] = self._render_window(ids, new_anchor, state).rstrip(
                REPLACEMENT_CHAR
            )
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
