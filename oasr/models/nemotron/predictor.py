# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron RNN-T prediction network, joint network and prompt projector.

The prediction network is a **2-layer LSTM** over the emitted labels, not the
stateless label-window convolution the icefall transducers use.  That difference
is the whole reason
:class:`~oasr.models.decoders.base.TransducerPredictor` exists: an LSTM state
cannot be recomputed from the last ``k`` labels, so the decode strategy has to
treat it as opaque and ask the predictor to fold each emission in.

The joint network is additive with no projections of its own:
``head(relu(enc_proj + dec_proj))``.  Upstream keeps the two projections
elsewhere (a top-level ``encoder_projector`` and the predictor's
``decoder_projector``); this module holds them as ``encoder_proj`` /
``decoder_proj``, which is the layout
:class:`~oasr.engine.decode.TransducerDecodeStrategy` already drives — it
projects the encoder output once per utterance and only re-runs the predictor
side per emission.  ``load_weights`` remaps the two keys; the arithmetic is
identical.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Sequence, Tuple, cast

import torch
from torch import nn

from oasr.layers import (
    LSTM,
    ColumnParallelLinear,
    Embedding,
    Linear,
    Relu,
    RowParallelLinear,
)

from ..base import align_out_features, init_pad_rows
from ..decoders.base import Joiner, TransducerPredictor

__all__ = ["NemotronPromptProjector", "NemotronRnntJoint", "NemotronRnntPredictor"]

#: ``(prediction (B, H), h (L, B, H), c (L, B, H))``.
LstmState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class NemotronRnntPredictor(TransducerPredictor):
    """LSTM label predictor: emitted-label history → ``(B, decoder_hidden)``.

    The start-of-sequence state is **not** zeros.  NeMo (and the HF port) run the
    LSTM once on the blank token from a zero hidden state before the first frame,
    and because the blank row of the embedding table is trained with
    ``padding_idx`` it is all zeros — so the first prediction is the LSTM's
    response to a zero input, i.e. its biases.  :meth:`init_state` reproduces
    that step rather than handing back zeros, which would drop a constant out of
    every first-frame joint.
    """

    decode_type = "transducer"
    #: Recurrent state, so modified beam search (which gather-reorders a label
    #: window across the beam) does not apply; greedy is the supported mode.
    label_window_state: ClassVar[bool] = False

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        blank_id: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blank_id = blank_id
        self.embedding = Embedding(vocab_size, hidden_size)
        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def _step(self, tokens: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> LstmState:
        """One LSTM step on ``tokens (B,)`` → ``(prediction, h, c)``."""
        embedded = self.embedding(tokens.unsqueeze(1))  # (B, 1, H)
        out, (new_h, new_c) = self.lstm(embedded, (h, c))
        return out.squeeze(1), new_h, new_c

    # -- TransducerPredictor protocol ---------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> LstmState:
        if dtype is None:
            dtype = cast(torch.Tensor, self.lstm.weight_ih_l0).dtype
        zeros = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype
        )
        sos = torch.full((batch_size,), self.blank_id, dtype=torch.long, device=device)
        return self._step(sos, zeros, zeros.clone())

    def predict(self, state: LstmState) -> torch.Tensor:
        """The state already carries the prediction; this is a read, not a rerun."""
        return state[0]

    def advance(self, state: LstmState, tokens: torch.Tensor, emit: torch.Tensor) -> LstmState:
        """Step the LSTM for every row, keep the result only where ``emit``.

        Stepping all rows and masking (rather than gathering the emitting subset)
        is what upstream does too, and it keeps the op batched: the discarded rows
        cost one wider LSTM step, a gather would cost a device sync.
        """
        out, h, c = state
        new_out, new_h, new_c = self._step(tokens, h, c)
        keep_out = emit.view(-1, 1)
        keep_state = emit.view(1, -1, 1)
        return (
            torch.where(keep_out, new_out, out),
            torch.where(keep_state, new_h, h),
            torch.where(keep_state, new_c, c),
        )

    def stack_states(self, states: Sequence[LstmState]) -> LstmState:
        return (
            torch.cat([s[0] for s in states], dim=0),
            torch.cat([s[1] for s in states], dim=1),
            torch.cat([s[2] for s in states], dim=1),
        )

    def unstack_states(self, state: LstmState) -> List[LstmState]:
        out, h, c = state
        return [(out[b : b + 1], h[:, b : b + 1], c[:, b : b + 1]) for b in range(out.size(0))]


class NemotronRnntJoint(Joiner):
    """``head(relu(encoder_proj(enc) + decoder_proj(dec)))``.

    ``relu`` sits *before* the vocabulary GEMM, so it cannot fold into that
    GEMM's epilogue; it is one standalone elementwise op per decode step over a
    ``(B, decoder_hidden)`` tensor (KG10 in the kernel-coverage inventory).
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        vocab_size: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation != "relu":
            raise ValueError(
                f"joint activation {activation!r} is not implemented; the released "
                "Nemotron checkpoints use relu"
            )
        self.vocab_size = vocab_size
        self.encoder_proj = ColumnParallelLinear(encoder_dim, decoder_dim)
        self.decoder_proj = ColumnParallelLinear(decoder_dim, decoder_dim)
        # Aligned so the vocabulary GEMM has a kernel at all; 13088 already is,
        # but a future vocabulary need not be and the padding rows are made
        # unwinnable at construction as well as at load.
        self.head = Linear(decoder_dim, align_out_features(vocab_size))
        self.relu = Relu()
        init_pad_rows(self.head, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        if project_input:
            x = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            x = encoder_out + decoder_out
        logits: torch.Tensor = self.head(self.relu(x))
        return logits[..., : self.vocab_size]


class NemotronPromptProjector(nn.Module):
    """Language-prompt conditioning: ``linear_2(relu(linear_1([hidden, one_hot])))``.

    Upstream concatenates a one-hot language slot onto **every** encoder frame and
    projects the pair back to the encoder width.  There is no residual: the
    projector's output *replaces* the encoder hidden state, so this is not an
    optional adapter that can be skipped when no language is named — omitting it
    changes the encoder output entirely.

    **The one-hot is folded, not concatenated.**  ``linear_1`` splits by its input
    columns into ``W_h (I, C)`` and ``W_p (I, P)``, and a one-hot selects one
    column of the second, so ``linear_1([h, onehot(id)]) = W_h h + (W_p[:, id] +
    b)``.  The bracket is a per-prompt bias, precomputed at load as the ``(P, I)``
    table :attr:`prompt_bias` (in the checkpoint's fp32, rounded once into the
    model dtype), and each row looks its own up.  That removes the
    ``(B, T, P)`` one-hot and the ``(B, T, C + P)`` concat a forward otherwise
    materialises, and shrinks the GEMM's K from ``C + P`` to ``C``.

    It is not bit-identical to the concat: the GEMM's output is rounded before
    the bias is added, where the concat added it inside the accumulator, so the
    hidden state moves by about one ulp.  The bias is per *row* because a batch
    can mix languages; a per-batch scalar would have let it ride in the GEMM's
    epilogue, at the price of tying every row to one language.

    Upstream's ``linear_1.{weight,bias}`` (and a native checkpoint written before
    the fold) are split by :meth:`_load_from_state_dict`.
    """

    def __init__(self, hidden_size: int, num_prompts: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prompts = num_prompts
        self.intermediate_size = intermediate_size
        self.linear_h = Linear(hidden_size, intermediate_size, bias=False)
        #: ``(P, I)``: ``W_p^T + b``, the first projection's prompt columns plus
        #: its bias, one row per prompt slot.
        self.prompt_bias = Embedding(num_prompts, intermediate_size)
        self.relu = Relu()
        self.linear_2 = RowParallelLinear(intermediate_size, hidden_size)

    def forward(self, hidden: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        """``hidden (B, T, C)`` + ``prompt_ids (B,)`` int64 → ``(B, T, C)``."""
        projected = self.linear_h(hidden)
        projected.add_(self.prompt_bias(prompt_ids).unsqueeze(1))
        fused: torch.Tensor = self.linear_2(self.relu(projected))
        return fused

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
        """Split an unfolded ``linear_1`` into ``linear_h`` and the prompt-bias table.

        Keyed on the old names rather than a module version, so it is idempotent: a
        native checkpoint written after the fold has no ``linear_1`` keys, and the
        native loader stamps the model's own versions onto every state dict, which
        would hide a version-gated remap from a pre-fold native checkpoint.
        """
        weight_key, bias_key = prefix + "linear_1.weight", prefix + "linear_1.bias"
        weight = state_dict.get(weight_key)
        expected = (self.intermediate_size, self.hidden_size + self.num_prompts)
        if isinstance(weight, torch.Tensor) and tuple(weight.shape) == expected:
            del state_dict[weight_key]
            bias = state_dict.pop(bias_key, None)
            c = self.hidden_size
            # fp32 for the sum, so the table is rounded once, when it is copied
            # into the model's dtype, not once here and again there.
            table = weight[:, c:].t().float()
            if bias is not None:
                table = table + bias.float()
            state_dict[prefix + "linear_h.weight"] = weight[:, :c].contiguous()
            state_dict[prefix + "prompt_bias.weight"] = table.contiguous()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
