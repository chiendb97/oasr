# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""A transducer whose greedy-decode arithmetic is exact, for bit-level oracles.

The fused greedy-decode kernel and the strategy's op-by-op loop cannot agree bit
for bit on arbitrary weights: the kernel's GEMVs accumulate in a different order
than the library GEMMs the loop calls.  These builders choose data that makes
every sum exact instead -- integer encoder frames, weights in ``{-1, 0, 1}``, an
identity ``encoder_proj`` -- so the joiner input is ``act(integer)``: for
``tanh`` one of a handful of half-precision values, all multiples of ``2**-8``;
for ReLU an integer.  Every partial sum in either path is then exactly
representable in fp32, any accumulation order gives the same logits, and a test
can demand identical tokens, frames and final predictor state.

Both the kernel tests (``tests/kernels``) and the decode-strategy tests
(``tests/decoders``) build on it, which is why it lives here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch

__all__ = [
    "BLANK",
    "DEC_DIM",
    "JOINER_DIM",
    "VOCAB",
    "exact_transducer",
    "integer_frames",
    "transducer_strategy",
]

JOINER_DIM = DEC_DIM = 64
VOCAB = 37
BLANK = 0


def _relu_joiner_cls():
    from oasr.layers import Relu
    from oasr.models.decoders.base import AdditiveJoinerTensors
    from oasr.models.transducer import TransducerJoiner

    class ReluJoiner(TransducerJoiner):
        """The icefall joiner with ReLU in place of tanh -- the kernel's other activation."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.tanh = Relu()

        def additive_tensors(self) -> AdditiveJoinerTensors:
            t = super().additive_tensors()
            return AdditiveJoinerTensors(
                output_weight=t.output_weight,
                output_bias=t.output_bias,
                vocab_size=t.vocab_size,
                activation="relu",
                decoder_proj_weight=t.decoder_proj_weight,
                decoder_proj_bias=t.decoder_proj_bias,
            )

    return ReluJoiner


def _ternary(shape, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(-1, 2, shape, generator=gen).float()


def exact_transducer(
    dtype: torch.dtype,
    *,
    context: int = 2,
    group: int = 4,
    activation: str = "tanh",
    seed: int = 0,
    blank_bias: Optional[float] = None,
    device: str = "cuda",
):
    """A :class:`~oasr.models.transducer.TransducerModel` with exact decode arithmetic.

    ``blank_bias`` makes blanks common enough that rows both emit and advance,
    which is what exercises the bookkeeping; its default is sized to the
    activation (``tanh`` bounds the joiner input to [-1, 1], ReLU does not, and
    its logits run about twice as large on this data).
    """
    from oasr.models.transducer import StatelessDecoder, TransducerJoiner, TransducerModel

    if blank_bias is None:
        blank_bias = 10.0 if activation == "tanh" else 22.0
    gen = torch.Generator().manual_seed(seed)
    decoder = StatelessDecoder(
        VOCAB, DEC_DIM, blank_id=BLANK, context_size=context, conv_group_size=group
    )
    joiner_cls = _relu_joiner_cls() if activation == "relu" else TransducerJoiner
    joiner = joiner_cls(JOINER_DIM, DEC_DIM, JOINER_DIM, VOCAB)
    with torch.no_grad():
        decoder.embedding.weight.copy_(_ternary(decoder.embedding.weight.shape, gen))
        decoder.embedding.weight[BLANK].zero_()
        if context > 1:
            decoder.conv.weight.copy_(_ternary(decoder.conv.weight.shape, gen))
        joiner.encoder_proj.weight.copy_(torch.eye(JOINER_DIM))
        joiner.encoder_proj.bias.zero_()
        joiner.decoder_proj.weight.copy_(_ternary(joiner.decoder_proj.weight.shape, gen))
        joiner.decoder_proj.bias.copy_(_ternary(joiner.decoder_proj.bias.shape, gen))
        joiner.output_linear.weight.copy_(_ternary(joiner.output_linear.weight.shape, gen))
        joiner.output_linear.bias.copy_(_ternary(joiner.output_linear.bias.shape, gen) / 4)
        joiner.output_linear.bias[BLANK] += blank_bias
    model = TransducerModel(torch.nn.Identity(), decoder, joiner, blank_id=BLANK)
    return model.to(device=device, dtype=dtype).eval()


def integer_frames(
    batch: int, frames: int, dtype: torch.dtype, *, seed: int = 1, device: str = "cuda"
) -> torch.Tensor:
    """``(batch, frames, JOINER_DIM)`` encoder output with values in ``{-2..2}``."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(-2, 3, (batch, frames, JOINER_DIM), generator=gen).to(
        device=device, dtype=dtype
    )


def transducer_strategy(model, *, max_sym: int = 3, partial_interval: int = 1, **options: Any):
    """A :class:`~oasr.engine.decode.transducer.TransducerDecodeStrategy` over ``model``.

    No CUDA graphs (the config carries no ``use_cuda_graphs``), so the op-by-op
    loop it falls back to is the eager one -- the oracle the fused kernel is held
    to.  ``options`` are decode options (``fused=False`` and so on).
    """
    from oasr.engine.decode import Detokenizer
    from oasr.engine.decode.transducer import TransducerDecodeStrategy

    cfg = SimpleNamespace(
        transducer_max_sym_per_frame=max_sym,
        partial_decode_interval=partial_interval,
        decode_options=dict(options),
    )
    return TransducerDecodeStrategy(cfg, Detokenizer(None, None), model)
