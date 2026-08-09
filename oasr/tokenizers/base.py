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

    # -- incremental decode -------------------------------------------------
    #
    # Streaming and AR families emit a partial per chunk / per tick, and each
    # one used to re-decode the whole prefix from scratch: Θ(n²) token-decodes
    # over a generation (~3.1k for a 448-token LLM run at 32 tokens/tick), worst
    # on ``funasr_char``, whose ``decode`` runs two all-CJK/all-alpha probes plus
    # a list-membership abbreviation pass over every token.
    #
    # ``decode_incremental`` amortises that: feed it only the ids produced since
    # the last call and it returns the **new text** plus carry-over state.  The
    # default below is a correct-but-quadratic fallback (decode the accumulated
    # prefix, diff off what was already emitted), so a tokenizer opts in to the
    # fast path by overriding, and none has to.
    #
    # Diffing rather than concatenating per-chunk decodes is what keeps this
    # equal to a full decode: joining is wrong for every kind in the tree —
    # sentencepiece word boundaries, ``@@`` subword merges and CJK spacing all
    # depend on neighbouring tokens.

    def new_decode_state(self) -> Dict[str, Any]:
        """Fresh per-request state for :meth:`decode_incremental`."""
        return {"ids": [], "text": ""}

    def decode_incremental(self, new_ids: Sequence[int], state: Dict[str, Any]) -> str:
        """Append ``new_ids`` to the running hypothesis; return the text delta.

        ``state`` is mutated in place.  ``state["text"]`` always holds the full
        transcript so far, so a caller wanting the whole thing need not
        accumulate deltas itself.

        The contract is that concatenating every delta equals
        ``decode(all_ids)`` — a subclass overriding this for speed must preserve
        that, including when a later token changes how an earlier one renders
        (the reason the default diffs full decodes instead of joining pieces).
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
