# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer greedy decode -- ``oasr/functionals/transducer.py``.

Three arms decode the same batch on the same model:

* ``cuda``  -- :func:`oasr.transducer_greedy_decode`, the fused one-launch kernel;
* ``graph`` -- the strategy's op-by-op loop replayed from CUDA graphs, 16 steps per
  replay (what served transducer decode before the kernel);
* ``torch`` -- the same loop eager.

Each arm returns its hypotheses as a ``(B, max_len)`` tensor padded with ``-1``,
so ``--refcheck`` compares tokens exactly.  The weights are random, which makes
bit-agreement likely but not guaranteed -- a GEMV's accumulation order can move a
logit by an ulp, and that only matters at a near-tie -- so the ``cuda`` arm is
informational there; ``tests/kernels/test_transducer_greedy.py`` holds the kernel
to exact agreement on data chosen to make every sum exact.

The blank bias is calibrated so that about four steps in five are blank at the
starting state, which is close to what a real BPE transducer does (~0.2 emissions
per encoder frame); the decode's cost is its step count, so a model that emitted
everywhere or nowhere would measure a different workload.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import torch

from benchmarks.core.driver import Work, params_of

SUBROUTINES = ["greedy"]

#: Shapes of the icefall Zipformer transducer (joiner/decoder 512, BPE-500,
#: context 2, group 4).  ``frames`` 250 is a ~10 s utterance at 25 Hz; 16 is a
#: streaming chunk.
DEFAULT_CONFIGS: Dict[str, list] = {
    "greedy": [
        {"batch": 1, "frames": 250, "joiner": 512, "decoder": 512, "vocab": 500},
        {"batch": 16, "frames": 250, "joiner": 512, "decoder": 512, "vocab": 500},
        {"batch": 64, "frames": 250, "joiner": 512, "decoder": 512, "vocab": 500},
        {"batch": 64, "frames": 16, "joiner": 512, "decoder": 512, "vocab": 500},
    ],
}

REF_BACKEND = "torch"
TOLERANCES = {"greedy": (0.0, 0.0)}
NON_GATING_BACKENDS = frozenset({"cuda"})

_CONTEXT = 2
_GROUP = 4
_MAX_SYM = 10


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Rows decoded together")
    parser.add_argument("--frames", type=int, default=None, help="Encoder frames per row")
    parser.add_argument("--joiner", type=int, default=None, help="Joiner dim")
    parser.add_argument("--decoder", type=int, default=None, help="Predictor dim")
    parser.add_argument("--vocab", type=int, default=None, help="Vocabulary size")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    dims = (args.batch, args.frames, args.joiner, args.decoder, args.vocab)
    if all(v is not None for v in dims):
        b, t, j, d, v = dims
        return [{"batch": b, "frames": t, "joiner": j, "decoder": d, "vocab": v}]
    return DEFAULT_CONFIGS[subroutine]


def _model(cfg: dict, dtype: torch.dtype):
    from oasr.models.transducer import StatelessDecoder, TransducerJoiner, TransducerModel

    torch.manual_seed(0)
    decoder = StatelessDecoder(
        cfg["vocab"], cfg["decoder"], blank_id=0, context_size=_CONTEXT, conv_group_size=_GROUP
    )
    joiner = TransducerJoiner(cfg["joiner"], cfg["decoder"], cfg["joiner"], cfg["vocab"])
    model = TransducerModel(torch.nn.Identity(), decoder, joiner, blank_id=0)
    return model.to(device="cuda", dtype=dtype).eval()


def _calibrate_blank(model, enc: torch.Tensor) -> None:
    """Bias blank so it wins ~80% of steps (see the module docstring).

    Measured over random label windows, not just the start state: a random-weight
    predictor drifts once it has emitted, and a bias tuned on the start state
    alone let rows emit at the per-frame cap for the rest of the utterance --
    past the kernel's emission buffer, which then (correctly) re-decoded the batch
    on the loop and the arm measured the wrong path.
    """
    joiner, decoder = model.joiner, model.decoder
    gen = torch.Generator(device=enc.device).manual_seed(1)
    with torch.no_grad():
        b, t = enc.size(0), enc.size(1)
        windows = torch.randint(
            1, decoder.vocab_size, (b * t, decoder.context_size), generator=gen, device=enc.device
        )
        dec_proj = joiner.decoder_proj(decoder.predict(windows)).view(b, t, -1)
        logits = joiner(joiner.encoder_proj(enc), dec_proj, project_input=False).float()
        margin = logits[..., 1:].amax(-1) - logits[..., 0]
        joiner.output_linear.bias[0] += torch.quantile(margin.flatten(), 0.8).to(
            joiner.output_linear.bias.dtype
        )


def _padded(hyps: List[List[int]]) -> torch.Tensor:
    width = max([len(h) for h in hyps] + [1])
    out = torch.full((len(hyps), width), -1, dtype=torch.int32)
    for b, h in enumerate(hyps):
        out[b, : len(h)] = torch.tensor(h, dtype=torch.int32)
    return out


def _strategy(model, **cfg: Any):
    from oasr.engine.decode import Detokenizer
    from oasr.engine.decode.transducer import TransducerDecodeStrategy

    config = SimpleNamespace(transducer_max_sym_per_frame=_MAX_SYM, **cfg)
    return TransducerDecodeStrategy(config, Detokenizer(None, None), model)


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if dtype not in (torch.float16, torch.bfloat16):
        # The strategy would quietly run its loop in the ``cuda`` arm instead.
        raise ValueError("the fused transducer decode is half precision only (--dtype)")
    model = _model(cfg, dtype)
    enc = torch.randn(cfg["batch"], cfg["frames"], cfg["joiner"], device="cuda", dtype=dtype)
    _calibrate_blank(model, enc)
    lengths = torch.full((cfg["batch"],), cfg["frames"], dtype=torch.long, device="cuda")

    eager = _strategy(model, decode_options={"fused": False})
    graphed = _strategy(
        model,
        use_cuda_graphs=True,
        use_transducer_cuda_graphs=True,
        decode_options={"fused": False},
    )
    fused = _strategy(model)

    def run(strat) -> torch.Tensor:
        state, dec_proj = strat._init_state(cfg["batch"], enc.device)
        with torch.no_grad():
            hyps, _, _, _ = strat._greedy_loop(enc, lengths, state, dec_proj)
        return _padded(hyps)

    def run_fused() -> torch.Tensor:
        before = dict(fused.fused_stats)
        out = run(fused)
        # The strategy falls back to its loop rather than fail; a benchmark arm
        # that did so would report the loop's time under the kernel's name.
        if fused.fused_stats["hits"] == before["hits"] or (
            fused.fused_stats["overflows"] != before["overflows"]
        ):
            raise RuntimeError(f"fused decode did not serve this batch: {fused.fused_stats}")
        return out

    return {"cuda": run_fused, "graph": lambda: run(graphed), "torch": lambda: run(eager)}


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    b, t, j, d, v = cfg["batch"], cfg["frames"], cfg["joiner"], cfg["decoder"], cfg["vocab"]
    # A step is latency-bound, not FLOP- or byte-bound (the joiner head is
    # re-read every step); report the shape only.
    return Work(shape=f"B={b} T={t} J={j} D={d} V={v}", params=params_of(cfg))
