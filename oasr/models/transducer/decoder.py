# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Stateless transducer predictor (icefall ``Decoder``).

Conditions only on the last ``context_size`` labels (no recurrent state): an
embedding followed by a grouped 1-D convolution over the label window.  Param
names (``embedding`` / ``conv``) mirror icefall ``egs/.../ASR/*/decoder.py`` so
the converter loads icefall pruned-transducer checkpoints 1:1.

The conv's **group size** is the one thing that cannot be inferred from the
layout: icefall writes ``nn.Conv1d(C, C, context_size, groups=C // group_size)``
with group size 1 in the old ``pruned_transducer_stateless2/3/5`` recipes and 4
in every Zipformer one.  Those are different operators — 4x the parameters — so
assuming depthwise made every real icefall release fail to load on this single
tensor.  See ``conv_group_size`` below and
``TransducerModelConfig.decoder_conv_group_size``.

**The window before the first label.**  icefall trains the predictor on
``[blank] + y`` with the embeddings left-padded by ``context_size - 1`` *zero
vectors*, and its decoders (``greedy_search``, ``modified_beam_search``, the
streaming ``DecodeStream``, and sherpa-onnx's) start from the window
``[-1] * (context_size - 1) + [blank]``, where a negative label embeds to zero.
A window of ``context_size`` blanks is a different input wherever blank's row is
not zero, and only the oldest recipe (``transducer_stateless``) declares
``padding_idx=blank``: a Zipformer predictor learns a nonzero blank row.
:meth:`StatelessDecoder.init_state` returns icefall's start window, with
:data:`NO_LABEL` for ``-1``, and a negative label embeds to zero both in
:meth:`StatelessDecoder.forward` and in the fused decode kernels.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Sequence, Union

import torch

from oasr.layers import Conv2d, DepthwiseConv1d, Embedding, Relu

from ..decoders.base import StatelessPredictorTensors, TransducerPredictor

#: The label of a window slot before the utterance start.  Any negative label
#: embeds to zero, as in icefall's ``Decoder.forward``.
NO_LABEL = -1


class StatelessDecoder(TransducerPredictor):
    """Label predictor: ``(B, context_size)`` token window → ``(B, decoder_dim)``.

    The transducer "decode state" is just the last ``context_size`` emitted
    labels (see :meth:`init_state` for the window before the first one);
    :class:`~oasr.engine.decode.TransducerDecodeStrategy` shifts a new label in
    after each emission via :meth:`advance`.
    """

    decode_type = "transducer"
    #: The state *is* a label window, so beam search can gather-reorder it.
    label_window_state: ClassVar[bool] = True

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int = 0,
        context_size: int = 2,
        conv_group_size: int = 1,
    ) -> None:
        super().__init__()
        assert context_size >= 1, context_size
        if conv_group_size < 1 or decoder_dim % conv_group_size:
            raise ValueError(
                f"conv_group_size must be >= 1 and divide decoder_dim, got "
                f"{conv_group_size=} with {decoder_dim=}"
            )
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.blank_id = blank_id
        self.context_size = context_size
        self.conv_group_size = conv_group_size
        # No ``padding_idx``: icefall's Zipformer and pruned-transducer predictors
        # do not declare one, so blank's row is learned, and a test model built
        # here must not have it zeroed either.
        self.embedding = Embedding(vocab_size, decoder_dim)
        self.relu = Relu()
        if context_size > 1:
            # Convolve the unpadded label window to one frame. Grouped predictors
            # are distinct operators, not depthwise layouts, and use the shared
            # layer abstraction through a one-row convolution.
            self.conv: Union[DepthwiseConv1d, Conv2d]
            if conv_group_size == 1:
                self.conv = DepthwiseConv1d(decoder_dim, kernel_size=context_size, bias=False)
            else:
                self.conv = Conv2d(
                    decoder_dim,
                    decoder_dim,
                    kernel_size=(1, context_size),
                    groups=decoder_dim // conv_group_size,
                    bias=False,
                )

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """The start window ``[NO_LABEL] * (context_size - 1) + [blank]`` per row (int64).

        This is the window icefall decodes from and the input its training saw
        before the first label (see the module docstring); ``blank`` in every
        slot is not.
        """
        del dtype  # labels are always int64
        state = torch.full(
            (batch_size, self.context_size), NO_LABEL, dtype=torch.long, device=device
        )
        state[:, -1:].fill_(self.blank_id)
        return state

    def _label_conv(self, emb: torch.Tensor) -> torch.Tensor:
        """``(B, U, dim)`` → ``(B, U - context_size + 1, dim)``."""
        if self.conv_group_size == 1:
            out: torch.Tensor = self.conv(emb)
            return out
        # The grouped branch is a conv2d: BTC → a single NHWC row and back.
        grouped: torch.Tensor = self.conv(emb.unsqueeze(1)).squeeze(1)
        return grouped

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """``(B, context_size)`` label window → ``(B, decoder_dim)`` prediction."""
        # A slot before the utterance start (NO_LABEL) embeds to zero.
        emb = self.embedding(y.clamp(min=0)).masked_fill((y < 0).unsqueeze(-1), 0.0)
        if self.context_size > 1:
            emb = self._label_conv(emb)  # (B, U - context_size + 1, dim) == (B, 1, dim)
        activated: torch.Tensor = self.relu(emb)
        # (B, U_out, dim) -> (B, dim): decode feeds exactly one window per step.
        return activated[:, -1, :]

    # -- TransducerPredictor protocol ---------------------------------------
    #
    # These are the lines the greedy loop used to inline.  Stateless is the
    # degenerate case of the protocol: the state carries no history beyond the
    # window, so ``predict`` is a full recompute and ``advance`` is a shift.

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Recompute the prediction from the label window."""
        return self(state)

    def advance(
        self, state: torch.Tensor, tokens: torch.Tensor, emit: torch.Tensor
    ) -> torch.Tensor:
        """Shift ``tokens`` into the window of every row where ``emit``.

        Rows that did not emit keep their window untouched, which is what makes
        the batched :meth:`predict` that follows reproduce their previous
        projection exactly rather than approximately.
        """
        shifted = torch.cat([state[:, 1:], tokens.unsqueeze(1)], dim=1)
        merged: torch.Tensor = torch.where(emit.unsqueeze(1), shifted, state)
        return merged

    def stack_states(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(states), dim=0)

    def stateless_tensors(self) -> StatelessPredictorTensors:
        """The embedding and window convolution, exactly as :meth:`forward` uses them.

        Both convolution branches already hold the layout their kernels read --
        ``DepthwiseConv1d`` keeps ``(K, 1, C)``, the grouped ``Conv2d`` keeps KRSC
        ``(C, 1, K, group)`` -- which is the layout the fused greedy decode reads.
        The kernels embed a negative label as zero, as :meth:`forward` does.
        """
        conv_weight = self.conv.weight if self.context_size > 1 else None
        return StatelessPredictorTensors(
            embedding=self.embedding.weight,
            conv_weight=conv_weight,
            context_size=self.context_size,
            group_size=self.conv_group_size,
            blank_id=self.blank_id,
        )

    def unstack_states(self, state: torch.Tensor) -> List[torch.Tensor]:
        return [state[b : b + 1] for b in range(state.size(0))]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Reshape icefall's grouped ``Conv1d`` predictor weight into NHWC KRSC.

        Only the grouped branch needs this: at ``conv_group_size == 1`` the child
        ``DepthwiseConv1d`` owns its own ``(C, 1, K) -> (K, 1, C)`` hook.  Gated
        on ``ndim == 3`` so a native OASR checkpoint — already 4-D KRSC — round
        trips untouched, and so a genuinely wrong shape still reaches
        ``load_state_dict`` as the size mismatch it is rather than being bent
        into silence here.
        """
        key = prefix + "conv.weight"
        if self.conv_group_size > 1 and key in state_dict:
            w = state_dict[key]
            expected = (self.decoder_dim, self.conv_group_size, self.context_size)
            if isinstance(w, torch.Tensor) and w.ndim == 3 and tuple(w.shape) == expected:
                # icefall (C, group_size, K) -> KRSC [C, R=1, S=K, group_size]
                state_dict[key] = w.permute(0, 2, 1).unsqueeze(1).contiguous()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
