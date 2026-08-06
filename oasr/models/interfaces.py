# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""What a checkpoint must expose to serve each decode family (keystone K1).

``model.capabilities`` says *what a checkpoint claims it can do*; this module says
*what that claim requires*, in one place, as data.  Before this existed each
autoregressive strategy hand-rolled its own ``hasattr`` gauntlet with its own
error message — five copies, no shared vocabulary, and the transducer strategy
checked nothing at all (it reached for ``model.joiner`` and got an
``AttributeError`` at first decode).  Worse, nothing verified that an advertised
capability was actually backed by the surface it needs, so a converter could
advertise ``ctc_aed_rescoring`` and fail at the first request.

Why a declarative table rather than :class:`typing.Protocol`
------------------------------------------------------------
The design note called for ``runtime_checkable`` Protocols.  Protocols cannot
express what these requirements actually are: the surfaces live **nested** on
sub-objects (``model.decoder.prefill``, ``model.config.decoder.sos_id``), and
``isinstance`` against a Protocol only inspects members of the object itself.
Splitting the check between Protocols (flat members) and something else (nested
paths) would mean two sources of truth for one contract — the exact failure mode
called out elsewhere in this codebase.  So the table is authoritative, dotted
paths and all, and it doubles as the answer to "what must I implement to add
family X".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CapabilitySpec",
    "CAPABILITIES",
    "missing_members",
    "require_capability",
]


@dataclass(frozen=True)
class CapabilitySpec:
    """The model surface one decode family drives.

    Attributes
    ----------
    capability : str
        Name as it appears in ``model.capabilities`` (and in
        ``EngineConfig.decode_method``).
    requires : tuple of str
        Dotted attribute paths that must resolve on the model instance.
    why : str
        What the family does with them — quoted back in the error, so a failure
        tells the reader which capability they are missing rather than which
        attribute.
    """

    capability: str
    requires: Tuple[str, ...]
    why: str


#: One entry per decode family.  ``ctc`` resolves to the ``ctc_cuda`` /
#: ``ctc_wfst`` strategies via ``EngineConfig.decoder_type``; every other
#: capability keys its strategy directly.
CAPABILITIES: Dict[str, CapabilitySpec] = {
    "ctc": CapabilitySpec(
        capability="ctc",
        requires=("head", "forward_offline"),
        why="frame-synchronous CTC decoding over the fused encoder+head log-probs",
    ),
    "transducer": CapabilitySpec(
        capability="transducer",
        requires=(
            "encode_offline",
            "blank_id",
            # The predictor-state protocol
            # (``oasr.models.decoders.base.TransducerPredictor``): the strategy
            # treats the state as opaque, so these three are what make a
            # stateless label window and a recurrent LSTM interchangeable.
            "decoder.init_state",
            "decoder.predict",
            "decoder.advance",
            "joiner.encoder_proj",
            "joiner.decoder_proj",
        ),
        why="frame-synchronous RNNT greedy decoding (label predictor + joiner)",
    ),
    "ctc_aed_rescoring": CapabilitySpec(
        capability="ctc_aed_rescoring",
        requires=(
            "head",
            "encode_offline",
            "decoder",
            "config.decoder.sos_id",
            "config.decoder.eos_id",
            "config.decoder.vocab_size",
            "config.decoder.reverse_weight",
        ),
        why=(
            "CTC n-best plus one teacher-forced attention-decoder pass "
            "(hybrid U2/U2++ checkpoints)"
        ),
    ),
    "aed": CapabilitySpec(
        capability="aed",
        requires=(
            "encode_offline",
            "decoder.prefill",
            "decoder.step",
            "decoder.select",
            "config.sot_sequence",
            "config.eos_token_id",
            "config.suppress_tokens",
            "config.begin_suppress_tokens",
            "config.max_target_positions",
        ),
        why="label-synchronous AED generation over the batched incremental decoder surface",
    ),
    "llm": CapabilitySpec(
        capability="llm",
        requires=(
            "encode_offline",
            "decoder.prefill",
            "decoder.step",
            "decoder.select",
            "decoder.embed_tokens",
            "config.prompt_prefix",
            "config.prompt_suffix",
            "config.default_user_prompt",
            "config.eos_token_ids",
            "config.text_max_position_embeddings",
        ),
        why=(
            "speech-LLM generation: audio embeddings spliced into the checkpoint's "
            "chat template, then label-synchronous decoding"
        ),
    ),
    "paraformer": CapabilitySpec(
        capability="paraformer",
        requires=(
            "encode_offline",
            "predict",
            "nar_decode",
            "config.blank_id",
            "config.sos_id",
            "config.eos_id",
        ),
        why="one-shot non-autoregressive decoding (CIF predictor + parallel NAR decoder)",
    ),
}


def _resolve(obj: Any, path: str) -> Tuple[bool, Any]:
    """Walk a dotted attribute path; ``(found, value)``.

    ``None`` at any step counts as missing: the models declare optional slots
    (``decoder``) as bare annotations that read back ``None`` when unset, and a
    ``None`` decoder is exactly as unusable as an absent one.
    """
    if obj is None:
        return False, None
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return False, None
        cur = getattr(cur, part)
    return cur is not None, cur


def missing_members(model: Any, capability: str) -> List[str]:
    """Dotted paths ``capability`` needs that ``model`` does not provide.

    An unknown capability returns ``[]`` — capability *names* are validated
    against ``model.capabilities`` by the engine; this function only answers the
    surface question, so a new family without a spec yet is not blocked by it.
    """
    spec = CAPABILITIES.get(capability)
    if spec is None:
        return []
    return [path for path in spec.requires if not _resolve(model, path)[0]]


def require_capability(
    model: Any,
    capability: str,
    *,
    decode_method: Optional[str] = None,
) -> None:
    """Raise ``ValueError`` unless ``model`` exposes everything ``capability`` needs.

    ``model=None`` is rejected like any other model that lacks the surface — it
    lacks all of it.  (The five gauntlets this replaced disagreed on that point:
    ``aed`` / ``llm`` / ``paraformer`` raised on ``None`` while ``transducer`` and
    the CTC strategies accepted it and failed later, or not at all.)
    ``decode_method`` is the name the caller selected, quoted in the error when it
    differs from ``capability``.
    """
    missing = missing_members(model, capability)
    if not missing:
        return
    spec = CAPABILITIES[capability]
    selected = decode_method or capability
    subject = "no model was supplied" if model is None else f"{type(model).__name__} is missing"
    raise ValueError(
        f"decode_method={selected!r} needs {spec.why}, but {subject}: "
        f"{', '.join(missing)}. Required surface: {', '.join(spec.requires)}."
    )
