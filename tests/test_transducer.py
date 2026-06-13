# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) components + greedy decode-loop tests (CPU, no checkpoint).

The batched greedy in ``TransducerDecodeStrategy.decode_offline`` is validated
against an obviously-correct per-utterance reference loop on the same tiny model.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from oasr.engine.decode import Detokenizer
from oasr.engine.decode.transducer import TransducerDecodeStrategy
from oasr.models.transducer import StatelessDecoder, TransducerJoiner, TransducerModel

VOCAB, ENC_DIM, DEC_DIM, JOINER_DIM = 20, 16, 12, 14


def _build_model(context_size=2, blank=0):
    decoder = StatelessDecoder(VOCAB, DEC_DIM, blank_id=blank, context_size=context_size)
    joiner = TransducerJoiner(ENC_DIM, DEC_DIM, JOINER_DIM, VOCAB)
    return TransducerModel(nn.Identity(), decoder, joiner, blank_id=blank).eval()


def _ref_greedy(enc_b, length, model, max_sym):
    decoder, joiner = model.decoder, model.joiner
    blank = model.blank_id
    device = enc_b.device
    context = torch.full((1, decoder.context_size), blank, dtype=torch.long, device=device)
    enc_proj = joiner.encoder_proj(enc_b)  # (T, J)
    dec_proj = joiner.decoder_proj(decoder(context))  # (1, J)
    hyp, t, sym = [], 0, 0
    while t < length:
        logits = joiner(enc_proj[t : t + 1], dec_proj, project_input=False)
        tok = int(logits.argmax(-1))
        if tok == blank or sym >= max_sym:
            t += 1
            sym = 0
        else:
            hyp.append(tok)
            context = torch.roll(context, -1, dims=1)
            context[0, -1] = tok
            dec_proj = joiner.decoder_proj(decoder(context))
            sym += 1
    return hyp


def test_component_shapes():
    dec = StatelessDecoder(VOCAB, DEC_DIM, blank_id=0, context_size=2)
    st = dec.init_state(4, torch.device("cpu"))
    assert st.shape == (4, 2) and st.dtype == torch.long
    assert (st == 0).all()  # blank-filled
    out = dec(st)
    assert out.shape == (4, DEC_DIM)
    join = TransducerJoiner(ENC_DIM, DEC_DIM, JOINER_DIM, VOCAB)
    logits = join(torch.randn(4, ENC_DIM), out)
    assert logits.shape == (4, VOCAB)


def test_transducer_decode_type_and_consumes():
    model = _build_model()
    assert model.decode_type == "transducer"
    strat = TransducerDecodeStrategy(SimpleNamespace(), Detokenizer(None, None), model)
    assert strat.consumes == "hidden"


def test_greedy_batched_matches_reference():
    torch.manual_seed(0)
    model = _build_model(context_size=2)
    cfg = SimpleNamespace(transducer_max_sym_per_frame=5)
    strat = TransducerDecodeStrategy(cfg, Detokenizer(None, None), model)

    B, T = 4, 20
    hidden = torch.randn(B, T, ENC_DIM)
    lengths = torch.tensor([20, 15, 8, 1], dtype=torch.int32)
    with torch.no_grad():
        outs = strat.decode_offline(hidden, lengths)

    assert len(outs) == B
    for b in range(B):
        ref = _ref_greedy(hidden[b], int(lengths[b]), model, max_sym=5)
        assert outs[b].tokens[0] == ref, (b, outs[b].tokens[0], ref)
        assert outs[b].finished


def test_greedy_context_size_1():
    torch.manual_seed(1)
    model = _build_model(context_size=1)
    cfg = SimpleNamespace(transducer_max_sym_per_frame=3)
    strat = TransducerDecodeStrategy(cfg, Detokenizer(None, None), model)
    hidden = torch.randn(2, 12, ENC_DIM)
    lengths = torch.tensor([12, 7])
    with torch.no_grad():
        outs = strat.decode_offline(hidden, lengths)
    for b in range(2):
        assert outs[b].tokens[0] == _ref_greedy(hidden[b], int(lengths[b]), model, 3)


def test_offline_executor_takes_hidden_branch_for_transducer():
    """End-to-end offline wiring: OutputProcessor(transducer) + OfflineExecutor
    must call encode_offline (not forward_offline) and emit greedy tokens."""
    from oasr.engine.executor.offline import OfflineExecutor
    from oasr.engine.output_processor import OutputProcessor
    from oasr.engine.request import Request

    torch.manual_seed(2)
    model = _build_model(context_size=2)
    cfg = SimpleNamespace(sentencepiece_model=None, unit_table=None, transducer_max_sym_per_frame=5)
    op = OutputProcessor(cfg, decode_type="transducer", model=model)
    assert op.strategy.consumes == "hidden"
    assert type(op.strategy).__name__ == "TransducerDecodeStrategy"

    B, T = 2, 16
    hidden = torch.randn(B, T, ENC_DIM)
    lengths = torch.tensor([T, 9])

    class StubMR:
        def encode_offline(self, feats, lens):
            return feats, lens  # treat the collated "features" as encoder hidden

        def forward_offline(self, feats, lens):
            raise AssertionError("CTC path must not run for a transducer")

        forward_offline_packed = forward_offline

    class StubInp:
        def collate(self, chunk):
            return hidden, lengths

    class StubSched:
        def split_offline_batch(self, batch):
            return [batch], None

    reqs = [Request(None, streaming=False) for _ in range(B)]
    ex = OfflineExecutor(
        scheduler=StubSched(),
        input_processor=StubInp(),
        model_runner=StubMR(),
        output_processor=op,
        device=torch.device("cpu"),
        enable_packing=False,
    )
    outs = ex.run(reqs)
    assert len(outs) == B
    for b in range(B):
        ref = _ref_greedy(hidden[b], int(lengths[b]), model, 5)
        assert outs[b].tokens[0] == ref
        assert outs[b].request_id == reqs[b].request_id
        assert outs[b].finished
