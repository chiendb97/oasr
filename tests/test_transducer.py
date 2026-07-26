# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Transducer (RNNT) components, greedy decode, converter + engine-wiring tests.

The batched greedy in ``TransducerDecodeStrategy`` (offline *and* streaming) is
validated against an obviously-correct per-utterance reference loop on the same
tiny model; the icefall pruned-transducer converter is validated on a synthetic
icefall-format checkpoint (CPU, no real checkpoint needed).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
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


# --------------------------------------------------------------------------- #
# Streaming greedy (session state threaded across chunks)
# --------------------------------------------------------------------------- #


def _cfg(max_sym=5, partial_interval=1):
    return SimpleNamespace(
        transducer_max_sym_per_frame=max_sym,
        partial_decode_interval=partial_interval,
    )


def _stream(strat, rid, hidden, chunk_sizes):
    """Feed ``hidden`` (1, T, D) to ``strat`` in chunks; return the final tokens."""
    req = SimpleNamespace(request_id=rid)
    strat.create_session(req)
    t = 0
    for cs in chunk_sizes:
        chunk = hidden[:, t : t + cs]
        if chunk.size(1):
            strat.decode_streaming_batch([req], {rid: chunk})
        t += cs
    final = strat.finalize(req)
    strat.free_session(req)
    return final


def test_streaming_matches_offline_tokens():
    """Chunked greedy with carried predictor state == one-shot offline greedy."""
    torch.manual_seed(3)
    model = _build_model(context_size=2)
    T = 32
    hidden = torch.randn(1, T, ENC_DIM)

    strat = TransducerDecodeStrategy(_cfg(), Detokenizer(None, None), model)
    with torch.no_grad():
        offline = strat.decode_offline(hidden, torch.tensor([T]))[0].tokens[0]

    for chunks in ([8, 8, 8, 8], [16, 16], [5, 11, 9, 7], [1] * T):
        strat2 = TransducerDecodeStrategy(_cfg(), Detokenizer(None, None), model)
        with torch.no_grad():
            final = _stream(strat2, "s", hidden, chunks)
        assert final.tokens[0] == offline, chunks
        assert final.finished


def test_streaming_batch_mixed_chunk_lengths():
    """One tick with different chunk lengths (grouped) == each stream alone."""
    torch.manual_seed(4)
    model = _build_model(context_size=2)
    strat = TransducerDecodeStrategy(_cfg(), Detokenizer(None, None), model)
    h_a = torch.randn(1, 16, ENC_DIM)
    h_b = torch.randn(1, 12, ENC_DIM)

    ra, rb = SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")
    strat.create_session(ra)
    strat.create_session(rb)
    with torch.no_grad():
        strat.decode_streaming_batch([ra, rb], {"a": h_a[:, :8], "b": h_b[:, :6]})
        strat.decode_streaming_batch([ra, rb], {"a": h_a[:, 8:], "b": h_b[:, 6:]})
    fa, fb = strat.finalize(ra), strat.finalize(rb)

    solo = TransducerDecodeStrategy(_cfg(), Detokenizer(None, None), model)
    with torch.no_grad():
        assert fa.tokens[0] == _stream(solo, "sa", h_a, [8, 8]).tokens[0]
        assert fb.tokens[0] == _stream(solo, "sb", h_b, [6, 6]).tokens[0]


def test_streaming_partial_cadence():
    torch.manual_seed(5)
    model = _build_model()
    hidden = torch.randn(1, 8, ENC_DIM)
    req = SimpleNamespace(request_id="p")

    # interval 2: partial on every 2nd chunk only.
    strat = TransducerDecodeStrategy(_cfg(partial_interval=2), Detokenizer(None, None), model)
    strat.create_session(req)
    with torch.no_grad():
        assert strat.decode_streaming_batch([req], {"p": hidden[:, :4]}) == []
        outs = strat.decode_streaming_batch([req], {"p": hidden[:, 4:]})
    assert len(outs) == 1 and not outs[0].finished

    # interval 0: partials disabled; state still advances and finalize works.
    strat0 = TransducerDecodeStrategy(_cfg(partial_interval=0), Detokenizer(None, None), model)
    req0 = SimpleNamespace(request_id="p0")
    strat0.create_session(req0)
    with torch.no_grad():
        assert strat0.decode_streaming_batch([req0], {"p0": hidden[:, :4]}) == []
        assert strat0.decode_streaming_batch([req0], {"p0": hidden[:, 4:]}) == []
        final0 = strat0.finalize(req0)
    strat1 = TransducerDecodeStrategy(_cfg(partial_interval=1), Detokenizer(None, None), model)
    with torch.no_grad():
        assert final0.tokens[0] == _stream(strat1, "x", hidden, [4, 4]).tokens[0]


def test_finalize_without_chunks_is_empty():
    model = _build_model()
    strat = TransducerDecodeStrategy(_cfg(), Detokenizer(None, None), model)
    req = SimpleNamespace(request_id="empty")
    strat.create_session(req)
    out = strat.finalize(req)
    assert out.finished and out.tokens == [[]] and out.text == ""


# --------------------------------------------------------------------------- #
# Icefall pruned-transducer converter (synthetic checkpoint)
# --------------------------------------------------------------------------- #


def _tiny_zipformer_transducer():
    from oasr.models.transducer import TransducerModelConfig
    from oasr.models.zipformer.config import ZipformerEncoderConfig

    enc_cfg = ZipformerEncoderConfig(
        feature_dim=80,
        downsampling_factor=(1, 2),
        encoder_dim=(64, 96),
        num_encoder_layers=(1, 1),
        query_head_dim=(8,),
        pos_head_dim=(4,),
        value_head_dim=(6,),
        num_heads=(4, 4),
        feedforward_dim=(64, 96),
        cnn_module_kernel=(15, 15),
        pos_dim=16,
        causal=False,
    )
    cfg = TransducerModelConfig(
        encoder_type="zipformer",
        encoder=enc_cfg,
        vocab_size=30,
        decoder_dim=24,
        joiner_dim=20,
        context_size=2,
        blank_id=0,
    )
    torch.manual_seed(6)
    return TransducerModel.from_config(cfg), cfg


def _to_icefall_sd(model):
    """OASR transducer state dict → icefall AsrModel layout (reverse of load).

    Reverses both the key remap (``encoder.encoder_embed.*`` →
    ``encoder_embed.*`` etc.) and the depthwise-conv weight transpose: OASR's
    ``DepthwiseConv1d`` stores ``(K, 1, C)`` and its shape-gated
    ``_load_from_state_dict`` hook converts icefall's ``(C, 1, K)`` on load.
    """
    sd = {}
    for k, v in model.state_dict().items():
        ik = k[len("encoder.") :] if k.startswith("encoder.") else k
        if ik.endswith("depthwise_conv.weight") and v.ndim == 3 and v.shape[1] == 1:
            v = v.permute(2, 1, 0).contiguous()
        sd[ik] = v
    return sd


@pytest.fixture(scope="module")
def icefall_transducer_dir(tmp_path_factory):
    """Synthetic icefall pruned-transducer experiment dir (tiny random model)."""
    model, cfg = _tiny_zipformer_transducer()
    sd = _to_icefall_sd(model)
    # Pruned-RNNT training heads (declared-expected drops) + a hybrid CTC branch.
    sd["simple_am_proj.weight"] = torch.randn(30, 96)
    sd["simple_lm_proj.weight"] = torch.randn(30, 24)
    sd["ctc_output.1.weight"] = torch.randn(30, 96)
    sd["ctc_output.1.bias"] = torch.randn(30)

    d = tmp_path_factory.mktemp("icefall_transducer")
    torch.save({"model": sd}, d / "pretrained.pt")
    (d / "tokens.txt").write_text("<blk> 0\n<sos/eos> 1\n<unk> 2\nHE 3\nLO 4\n")
    return d, model, cfg


class TestIcefallTransducerConverter:
    def test_not_auto_detected(self, icefall_transducer_dir):
        """The dir sniffs as zipformer (CTC); transducer is explicit-only."""
        from oasr.models import resolve_architecture

        d, _, _ = icefall_transducer_dir
        assert resolve_architecture(d) == "zipformer"
        assert resolve_architecture(d, architecture="transducer") == "transducer"

    def test_config_inference_and_weight_round_trip(self, icefall_transducer_dir):
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        d, src_model, src_cfg = icefall_transducer_dir
        arch, bundle = load_checkpoint_bundle(d, architecture="transducer")
        assert arch == "transducer" and bundle.source_format == "icefall"
        cfg = bundle.model_config
        assert cfg.encoder_type == "zipformer"
        assert (cfg.vocab_size, cfg.decoder_dim, cfg.joiner_dim, cfg.context_size) == (
            30,
            24,
            20,
            2,
        )
        assert bundle.decoding.default_decode_type == "transducer"
        assert bundle.tokenizer is not None and bundle.tokenizer.kind == "symbol_table"

        model, _, report = instantiate_from_bundle(arch, bundle)
        assert model.decode_type == "transducer"
        # Every source tensor round-trips exactly.
        src_sd, dst_sd = src_model.state_dict(), model.state_dict()
        assert set(src_sd) == set(dst_sd)
        for k in src_sd:
            assert torch.equal(src_sd[k], dst_sd[k]), k
        # Drops are fully accounted: simple_* expected, ctc_output a named hint.
        assert report is not None and not report.missing
        assert {k.split(".")[0] for k in report.dropped} == {
            "simple_am_proj",
            "simple_lm_proj",
            "ctc_output",
        }

    def test_ctc_hint_warns(self, icefall_transducer_dir, caplog):
        import logging

        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        d, _, _ = icefall_transducer_dir
        arch, bundle = load_checkpoint_bundle(d, architecture="transducer")
        with caplog.at_level(logging.WARNING, logger="oasr.models.registry"):
            instantiate_from_bundle(arch, bundle)
        joined = " ".join(r.message for r in caplog.records)
        assert "ctc_output.*" in joined and "architecture='zipformer'" in joined
        assert "simple_am_proj" not in joined  # declared-expected: silent

    def test_native_round_trip(self, icefall_transducer_dir, tmp_path):
        pytest.importorskip("safetensors")
        from oasr.checkpoints.convert import convert_to_native
        from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

        d, src_model, _ = icefall_transducer_dir
        native = tmp_path / "native"
        convert_to_native(str(d), str(native), architecture="transducer")
        arch, bundle = load_checkpoint_bundle(native)
        assert arch == "transducer" and bundle.source_format == "native"
        model, cfg, _ = instantiate_from_bundle(arch, bundle)
        assert cfg.encoder_type == "zipformer" and cfg.context_size == 2
        src_sd, dst_sd = src_model.state_dict(), model.state_dict()
        assert set(src_sd) == set(dst_sd)
        for k in src_sd:
            assert torch.equal(src_sd[k], dst_sd[k]), k


# --------------------------------------------------------------------------- #
# Engine end-to-end (native conformer-transducer checkpoint, GPU)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
class TestEngineTransducer:
    @pytest.fixture(scope="class")
    def native_ckpt(self, tmp_path_factory):
        pytest.importorskip("safetensors")
        from oasr.checkpoints import save_native
        from oasr.checkpoints.bundle import DecodingDefaults
        from oasr.models.conformer.config import ConformerEncoderConfig
        from oasr.models.transducer import TransducerModelConfig

        enc = ConformerEncoderConfig(
            input_size=80,
            # head_dim 64 (128 / 2).  Bounded on both sides: the CuteDSL FMHA
            # cannot implement head_dim 16, and at head_dim **32** its paged
            # streaming path reads stale memory under CUDA-graph capture — a
            # pre-existing defect that also hits CTC, measured as a run-to-run
            # varying ~1e-1 log-prob delta vs eager at every model width.  The
            # old 64/2 fixture sat exactly there; it went unnoticed only
            # because hidden mode used to be excluded from capture.
            output_size=128,
            num_blocks=2,
            attention_heads=2,
            linear_units=128,
            cnn_module_kernel=15,
            # Streaming conformers are causal (the paged CNN cache carries the
            # depthwise conv's left context; non-causal convs keep none).
            causal=True,
            embed_layer_norm=False,
        )
        cfg = TransducerModelConfig(
            encoder_type="conformer",
            encoder=enc,
            vocab_size=32,
            decoder_dim=24,
            joiner_dim=20,
            context_size=2,
        )
        torch.manual_seed(7)
        model = TransducerModel.from_config(cfg).eval()
        d = tmp_path_factory.mktemp("native_transducer")
        save_native(
            d,
            architecture="transducer",
            model=model,
            model_config=cfg,
            decoding=DecodingDefaults(default_decode_type="transducer"),
        )
        return d

    def test_offline_engine(self, native_ckpt):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(ckpt_dir=str(native_ckpt), service_mode="offline", max_batch_size=2)
        engine = ASREngine(cfg)
        torch.manual_seed(8)
        wavs = [torch.randn(16000), torch.randn(24000)]
        texts = engine.transcribe_offline(wavs)
        assert len(texts) == 2 and all(isinstance(t, str) for t in texts)

    def test_streaming_engine(self, native_ckpt):
        """Streaming path: hidden-mode paged backend + session greedy."""
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=str(native_ckpt),
            service_mode="streaming",
            max_batch_size=2,
            max_num_blocks=256,
            max_blocks_per_seq=64,
        )
        engine = ASREngine(cfg)
        # Hidden mode must have routed the paged backend off the fused head...
        backend = engine._model_runner.streaming_backend  # noqa: SLF001
        assert backend._consumes == "hidden"  # noqa: SLF001
        # ...and must still get CUDA-graph capture: the capture takes whichever
        # chunk-forward the strategy routed to, so hidden mode is not a
        # second-class streaming path (H3).
        assert backend._graph_cache is not None  # noqa: SLF001
        torch.manual_seed(9)
        text = engine.transcribe(torch.randn(32000))
        assert isinstance(text, str)

    def test_streaming_graph_capture_is_token_identical_to_eager(self, native_ckpt):
        """Capturing the encoder-only chunk forward must not move a single token.

        The gate for H3: graph replay reuses pre-allocated input/output buffers
        and a private memory pool, so a capture that baked in the wrong buffer
        or missed a cache write shows up here as a token divergence rather than
        as a plausible-looking transcript.
        """
        from oasr.engine import ASREngine, EngineConfig

        def run(use_graphs: bool):
            cfg = EngineConfig(
                ckpt_dir=str(native_ckpt),
                service_mode="streaming",
                max_batch_size=4,
                max_num_blocks=256,
                max_blocks_per_seq=64,
                use_cuda_graphs=use_graphs,
            )
            engine = ASREngine(cfg)
            assert (
                engine._model_runner.streaming_backend._graph_cache is not None  # noqa: SLF001
            ) is use_graphs
            torch.manual_seed(21)
            wavs = [torch.randn(16000 + 4000 * i) for i in range(4)]
            ids = [engine.add_streaming_request(request_id=f"g{i}") for i in range(4)]
            for rid, w in zip(ids, wavs):
                engine.feed_chunk(rid, w, is_last=True)
            finals = {}
            for _ in range(2000):
                for out in engine.step():
                    if out.finished:
                        finals[out.request_id] = list(out.tokens[0]) if out.tokens else []
                if not (engine.num_running or engine.num_waiting):
                    break
            return finals

        eager = run(False)
        graphed = run(True)
        assert set(eager) == set(graphed) == {f"g{i}" for i in range(4)}
        for rid in eager:
            assert graphed[rid] == eager[rid], rid

    @pytest.mark.parametrize("stride", [1, 2, 3, 64])
    def test_greedy_is_invariant_to_the_termination_check_stride(self, native_ckpt, stride):
        """The greedy loop checks termination once per block, not per iteration (H7).

        Overshooting a block boundary must be **inert**: once every row has
        ``t >= its length``, ``active`` is all-false, so ``emit`` / ``advance`` are
        too and no state mutates.  ``stride=1`` reproduces the old per-iteration
        check exactly, so it is the reference — any stride must agree with it, both
        offline (one loop to the row length) and streaming (a short loop per chunk,
        where the overshoot is proportionally largest).
        """
        from oasr.engine import ASREngine, EngineConfig
        from oasr.engine.decode import transducer as transducer_mod

        torch.manual_seed(11)
        wavs = [torch.randn(n) for n in (8000, 16000, 24000)]
        original = transducer_mod._TERMINATION_CHECK_STRIDE  # noqa: SLF001

        def transcribe(value, streaming):
            transducer_mod._TERMINATION_CHECK_STRIDE = value  # noqa: SLF001
            if streaming:
                cfg = EngineConfig(
                    ckpt_dir=str(native_ckpt),
                    service_mode="streaming",
                    max_batch_size=3,
                    max_num_blocks=512,
                    max_blocks_per_seq=64,
                )
                engine = ASREngine(cfg)
                return [engine.transcribe(w) for w in wavs]
            cfg = EngineConfig(ckpt_dir=str(native_ckpt), service_mode="offline", max_batch_size=3)
            return ASREngine(cfg).transcribe_offline(wavs)

        try:
            for streaming in (False, True):
                reference = transcribe(1, streaming)
                assert transcribe(stride, streaming) == reference, (
                    f"{'streaming' if streaming else 'offline'} greedy diverged at "
                    f"stride={stride}; the block overshoot is not inert"
                )
        finally:
            transducer_mod._TERMINATION_CHECK_STRIDE = original  # noqa: SLF001


@pytest.mark.skipif(not torch.cuda.is_available(), reason="beam search runs on the model device")
class TestTransducerBeamSearch:
    """Modified beam search (P4), validated by properties rather than an oracle.

    There is no conformer-transducer checkpoint in the tree, so WER is not
    available — and a WER number on random weights would be meaningless anyway.
    What *is* available are four properties that a correct implementation must
    have and a buggy one almost certainly breaks.
    """

    @pytest.fixture(scope="class")
    def model(self):
        from oasr.models.conformer.config import ConformerEncoderConfig
        from oasr.models.transducer import TransducerModel, TransducerModelConfig

        enc = ConformerEncoderConfig(
            input_size=80,
            output_size=64,
            num_blocks=2,
            attention_heads=2,
            linear_units=128,
            cnn_module_kernel=15,
            causal=True,
            embed_layer_norm=False,
        )
        cfg = TransducerModelConfig(
            encoder_type="conformer",
            encoder=enc,
            vocab_size=48,
            decoder_dim=32,
            joiner_dim=32,
            context_size=2,
        )
        torch.manual_seed(11)
        return TransducerModel.from_config(cfg).eval().to("cuda", torch.float32)

    def _set_blank_bias(self, model, bias):
        """Steer the emission rate; random weights almost never pick blank."""
        with torch.no_grad():
            b = model.joiner.output_linear.bias
            b.zero_()
            b[int(model.blank_id)] = bias

    def _enc(self, model, B=4, T=40):
        torch.manual_seed(3)
        enc = torch.randn(B, T, model.encoder.output_size, device="cuda")
        lengths = torch.tensor([T, T - 5, T - 11, T][:B], device="cuda")
        return enc, lengths

    def _strategy(self, model, beam, max_sym):
        from oasr.engine.decode.transducer import TransducerDecodeStrategy

        cfg = SimpleNamespace(
            device="cuda",
            transducer_max_sym_per_frame=max_sym,
            decode_options={"beam_size": beam},
            partial_decode_interval=1,
        )
        detok = SimpleNamespace(
            detokenize=lambda ids: " ".join(map(str, ids)),
            new_state=lambda: {"ids": [], "text": ""},
            detokenize_incremental=lambda n, st: "",
        )
        return TransducerDecodeStrategy(cfg, detok, model)

    def _beam_rows(self, model, enc, lengths, beam):
        from oasr.engine.decode.transducer_beam import beam_search_chunk, init_beam_state

        st = init_beam_state(model.decoder, enc.size(0), beam, enc.device, capacity=enc.size(1))
        st = beam_search_chunk(model, enc, lengths, st)
        return st.hypotheses()

    @pytest.mark.parametrize("blank_bias", [2.0, 0.5, -1.0])
    def test_beam_one_reproduces_greedy(self, model, blank_bias):
        """``beam=1`` and greedy at ``max_sym_per_frame=1`` are the same algorithm.

        Both take the argmax and advance one frame, so they must agree token for
        token.  This is the exactness gate: an off-by-one in the parent gather,
        the blank handling or the token scatter shows up here immediately, where
        a WER comparison would hide it.  Parameterised over the blank bias so the
        check covers the all-blank, sparse and dense emission regimes — with
        random weights argmax almost never picks blank, and an emit-nothing run
        would pass trivially.
        """
        self._set_blank_bias(model, blank_bias)
        enc, lengths = self._enc(model)
        strat = self._strategy(model, beam=1, max_sym=1)
        ctx, dp = strat._init_state(enc.size(0), enc.device)  # noqa: SLF001
        greedy, _, _ = strat._greedy_loop(enc, lengths, ctx, dp)  # noqa: SLF001
        rows, _ = self._beam_rows(model, enc, lengths, beam=1)
        assert [r[0] for r in rows] == greedy

    def test_wider_beams_never_score_worse(self, model):
        """A wider beam explores a superset, so its best cannot be worse."""
        self._set_blank_bias(model, 0.5)
        enc, lengths = self._enc(model)
        prev = None
        for k in (1, 2, 4, 8):
            _, scores = self._beam_rows(model, enc, lengths, beam=k)
            best = [s[0] for s in scores]
            if prev is not None:
                for i, (now, before) in enumerate(zip(best, prev)):
                    assert now >= before - 1e-4, f"beam {k} regressed on row {i}"
            prev = best

    def test_nbest_is_ordered_and_sized(self, model):
        """``n_best`` finally means something for this family (T5)."""
        self._set_blank_bias(model, 0.5)
        enc, lengths = self._enc(model)
        rows, scores = self._beam_rows(model, enc, lengths, beam=4)
        for b in range(enc.size(0)):
            assert len(rows[b]) == 4 and len(scores[b]) == 4
            assert scores[b] == sorted(scores[b], reverse=True)

    def test_padding_frames_do_not_touch_a_short_row(self, model):
        """A short utterance in a mixed batch must decode as if it were alone.

        The per-row ``active`` mask is what guarantees this; without it the
        padding frames the batch forced on a short row would keep extending its
        hypothesis.
        """
        self._set_blank_bias(model, 0.5)
        enc, lengths = self._enc(model)
        rows_batched, _ = self._beam_rows(model, enc, lengths, beam=4)
        for b in range(enc.size(0)):
            L = int(lengths[b])
            solo, _ = self._beam_rows(
                model, enc[b : b + 1, :L], lengths[b : b + 1].clamp(max=L), beam=4
            )
            assert rows_batched[b] == solo[0], f"row {b} differs when batched"

    def test_streaming_beam_matches_offline_beam(self, model):
        """Chunked beam search must equal one-shot beam search over the same audio.

        The session carries the ``(1, k, ...)`` state across chunks and the group
        is re-stacked every tick (streams are grouped by chunk length), so this is
        the test that the stack/select round trip preserves the beam exactly.
        """
        from oasr.engine.decode.transducer_beam import (
            beam_search_chunk,
            init_beam_state,
            select_rows,
            stack_states,
        )

        self._set_blank_bias(model, 0.5)
        B, T, chunk = 3, 36, 12
        torch.manual_seed(5)
        enc = torch.randn(B, T, model.encoder.output_size, device="cuda")
        lengths = torch.full((B,), T, dtype=torch.long, device="cuda")
        offline, _ = self._beam_rows(model, enc, lengths, beam=4)

        # Chunked, with a stack/select round trip between every chunk.
        per_stream = [init_beam_state(model.decoder, 1, 4, enc.device) for _ in range(B)]
        for start in range(0, T, chunk):
            piece = enc[:, start : start + chunk]
            lens = torch.full((B,), piece.size(1), dtype=torch.long, device=enc.device)
            state = stack_states(per_stream)
            state = beam_search_chunk(model, piece, lens, state)
            per_stream = [
                select_rows(state, torch.tensor([b], device=enc.device)) for b in range(B)
            ]
        streamed = [s.hypotheses()[0][0] for s in per_stream]
        for b in range(B):
            assert streamed[b] == offline[b], f"stream/offline beam differ on row {b}"
