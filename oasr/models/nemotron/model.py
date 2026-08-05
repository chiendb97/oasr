# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron ASR: FastConformer encoder + language prompt + RNN-T.

Engine integration: ``decode_type`` is ``"transducer"``, so the engine selects
:class:`~oasr.engine.decode.TransducerDecodeStrategy`, which consumes raw encoder
hidden states (``consumes="hidden"``) and drives ``model.decoder`` +
``model.joiner`` frame-synchronously.  Offline only — see
:class:`~oasr.models.nemotron.encoder.NemotronEncoder` for why
``streaming_kind == "none"`` on a model whose name says streaming.

Two things this model does that the icefall transducer does not:

**The prompt fusion is part of encoding.**  ``encode_offline`` returns the
encoder output *after* the language-prompt projector, because that is what the
joint's encoder projection consumes.  The projector has no residual — its output
replaces the hidden state — so it is not something a caller can opt out of.

**The predictor is recurrent.**  ``model.decoder`` is a 2-layer LSTM exposing the
:class:`~oasr.models.decoders.base.TransducerPredictor` protocol, and beam search
(which reorders a label-window state across the beam) does not apply; the decode
strategy refuses ``beam_size > 1`` at construction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import torch

from ..base import BaseAsrModel, LoadReport, align_out_features, pad_output_projection
from .config import NemotronEncoderConfig, NemotronModelConfig
from .encoder import NemotronEncoder
from .predictor import NemotronPromptProjector, NemotronRnntJoint, NemotronRnntPredictor
from .subsampling import NemotronSubsampling

logger = logging.getLogger(__name__)

#: Upstream key prefix → this model's, for the two modules OASR relocates.  The
#: joint's projections live upstream as a top-level ``encoder_projector`` and as
#: the predictor's ``decoder_projector``; putting both on the joiner is what lets
#: the shared transducer strategy project the encoder once and the predictor
#: per emission.
_RENAMES = {
    "encoder_projector.": "joint.encoder_proj.",
    "decoder.decoder_projector.": "joint.decoder_proj.",
}


class NemotronModel(BaseAsrModel):
    """Encoder + prompt projector + LSTM predictor (``decoder``) + ``joint``."""

    @property
    def default_decode_type(self) -> str:
        return "transducer"

    @property
    def capabilities(self) -> frozenset:
        """Declared rather than derived, so ``tests/test_model_contract.py`` can
        check it against ``oasr.models.interfaces.CAPABILITIES`` without building
        the model."""
        return frozenset({"transducer"})

    def __init__(
        self,
        encoder: NemotronEncoder,
        decoder: NemotronRnntPredictor,
        joint: NemotronRnntJoint,
        prompt_projector: Optional[NemotronPromptProjector] = None,
        *,
        blank_id: int,
        prompt_id: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.prompt_projector = prompt_projector
        self._blank_id = int(blank_id)
        #: Language-prompt slot applied to every request.  Settable: the released
        #: checkpoint's default is the ``"auto"`` slot (its own language ID), and
        #: naming a language instead is a per-deployment choice.  Per-*request*
        #: conditioning would need the id to travel with the batch through the
        #: offline executor, which nothing in the engine does today.
        self.prompt_id = int(prompt_id)
        self._joint_out = align_out_features(int(joint.vocab_size))

    @property
    def blank_id(self) -> int:
        return self._blank_id

    @property
    def joiner(self) -> NemotronRnntJoint:
        """The name the transducer decode strategy reaches for.

        A property rather than a second attribute: assigning the module twice
        would register it twice and duplicate every joint tensor in the state
        dict.  Upstream calls it ``joint``; keeping that as the registered name
        is what makes ``joint.head.*`` load 1:1.
        """
        return self.joint

    @property
    def head(self):  # type: ignore[override]
        """Transducers have no CTC head; ``None`` keeps generic probes safe."""
        return None

    @classmethod
    def from_config(  # type: ignore[override]  # narrower config, as every arch does
        cls, config: NemotronModelConfig, **aux: Any
    ) -> "NemotronModel":
        del aux
        encoder = NemotronEncoder(config.encoder)
        decoder = NemotronRnntPredictor(
            config.vocab_size,
            config.decoder_hidden_size,
            config.num_decoder_layers,
            blank_id=config.blank_token_id,
        )
        joint = NemotronRnntJoint(
            config.encoder.hidden_size,
            config.decoder_hidden_size,
            config.vocab_size,
            activation=config.hidden_act,
        )
        prompt_projector = (
            NemotronPromptProjector(
                config.encoder.hidden_size,
                config.num_prompts,
                config.prompt_intermediate_size,
            )
            if config.num_prompts > 0
            else None
        )
        return cls(
            encoder,
            decoder,
            joint,
            prompt_projector,
            blank_id=config.blank_token_id,
            prompt_id=config.default_prompt_id,
        )

    # -- engine-facing forward -----------------------------------------------

    def encode_offline(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, T, n_mels)`` → ``(prompt-fused hidden (B, T', C), out_lengths)``.

        The prompt projector runs here, not in the joint, because its output *is*
        the encoder representation from the joint's point of view (upstream's
        ``get_audio_features``).
        """
        hidden, masks = self.encoder(features, lengths)
        if self.prompt_projector is not None:
            prompt_ids = torch.full(
                (hidden.size(0),), self.prompt_id, dtype=torch.long, device=hidden.device
            )
            hidden = self.prompt_projector(hidden, prompt_ids)
        return hidden, self._lengths_from_mask(masks)

    # -- weights --------------------------------------------------------------

    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> LoadReport:
        """Map an HF ``Nemotron3_5AsrForRNNT`` state dict into this model.

        Almost everything is 1:1 — the layer, convolution, LSTM and prompt keys
        keep their upstream names.  Three things are not:

        * the two joint projections move onto the joiner (:data:`_RENAMES`);
        * the subsampling projection's **input axis is permuted**, because
          upstream flattens the conv output ``(channel, freq)`` and this
          implementation stays in NHWC and flattens ``(freq, channel)``.  Doing
          it here rather than per forward costs one permute at load and nothing
          afterwards; doing it in a module hook would double-apply on a native
          reload, whose state dict is already in this key space;
        * the vocabulary head is widened if its width is not GEMM-aligned (a
          no-op for the released 13088).

        The Conv2d NCHW→NHWC and depthwise-conv1d layout conversions are handled
        by the layers' own ``_load_from_state_dict`` hooks, which are shape-guarded
        and therefore idempotent across a native round trip.
        """
        remapped: Dict[str, torch.Tensor] = {}
        dropped = []
        for key, value in state_dict.items():
            for old, new in _RENAMES.items():
                if key.startswith(old):
                    remapped[new + key[len(old) :]] = value
                    break
            else:
                if key.startswith(("encoder.", "decoder.", "joint.", "prompt_projector.")):
                    remapped[key] = value
                else:
                    dropped.append(key)

        self._permute_subsampling_projection(remapped)
        pad_output_projection(remapped, "joint.head.", self._joint_out)

        missing, unexpected = self.load_state_dict(remapped, strict=strict)
        if missing:
            logger.warning("Missing keys when loading nemotron weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading nemotron weights: %s", unexpected)
        return LoadReport.build(remapped, missing, unexpected, dropped)

    def _permute_subsampling_projection(self, sd: Dict[str, torch.Tensor]) -> None:
        """Reorder ``encoder.subsampling.linear.weight`` columns ``(c, f)`` → ``(f, c)``."""
        key = "encoder.subsampling.linear.weight"
        weight = sd.get(key)
        if weight is None:
            return
        subsampling = cast(NemotronSubsampling, self.encoder.subsampling)
        channels, freq = subsampling.flatten_order()
        if weight.shape[1] != channels * freq:
            raise ValueError(
                f"{key} has {weight.shape[1]} input features but the subsampling "
                f"stack flattens to {channels} channels x {freq} frequency bins "
                f"({channels * freq}); check num_mel_bins / subsampling_factor"
            )
        sd[key] = (
            weight.view(weight.shape[0], channels, freq)
            .transpose(1, 2)
            .reshape(weight.shape[0], freq * channels)
            .contiguous()
        )


__all__ = ["NemotronEncoderConfig", "NemotronModel", "NemotronModelConfig"]
