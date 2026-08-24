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

    Every ``_load_from_state_dict`` hook the load path applies has to be undone
    here, or the fixture is not the icefall checkpoint it claims to be and the
    round trip measures the hook against itself:

    * the key remap (``encoder.encoder_embed.*`` → ``encoder_embed.*`` etc.);
    * :class:`~oasr.layers.Conv2d`, which stores KRSC ``(K, R, S, C)`` against
      torch's KCRS ``(K, C, R, S)``;
    * ``Conv2dSubsampling.out``, whose input features are flattened
      ``(frequency, channel)`` in the NHWC runtime layout and
      ``(channel, frequency)`` in icefall's NCHW one;
    * ``DepthwiseConv1d``, which stores ``(K, 1, C)`` against icefall's
      ``(C, 1, K)``.

    The fixture is saved as a plain ``dict``, so it reaches ``load_state_dict``
    without ``_metadata`` — version 1, exactly like a real icefall export, which
    is what arms the version-gated projection hook.
    """
    from oasr.layers import Conv2d
    from oasr.models.zipformer.subsampling import Conv2dSubsampling

    nhwc_conv = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, Conv2d) and module.weight.ndim == 4
    }
    projections = {
        f"{name}.out.weight": (module.out_width, module.layer3_channels)
        for name, module in model.named_modules()
        if isinstance(module, Conv2dSubsampling)
    }

    sd = {}
    for k, v in model.state_dict().items():
        if k in nhwc_conv:
            v = v.permute(0, 3, 1, 2).contiguous()  # KRSC -> KCRS
        elif k in projections:
            width, channels = projections[k]
            v = (
                v.view(v.shape[0], width, channels)  # (out, F, C) -> (out, C, F)
                .transpose(1, 2)
                .reshape_as(v)
                .contiguous()
            )
        ik = k[len("encoder.") :] if k.startswith("encoder.") else k
        if ik.endswith(("depthwise_conv.weight", "decoder.conv.weight")) and v.ndim == 3:
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
        # second-class streaming path.
        assert backend._graph_cache is not None  # noqa: SLF001
        torch.manual_seed(9)
        text = engine.transcribe(torch.randn(32000))
        assert isinstance(text, str)

    def test_streaming_graph_capture_is_token_identical_to_eager(self, native_ckpt):
        """Capturing the encoder-only chunk forward must not move a single token.

        Graph replay reuses pre-allocated input/output buffers
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
        """The greedy loop checks termination once per block, not per iteration.

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
    """Modified beam search, validated by properties rather than an oracle.

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
        greedy, _, _, _ = strat._greedy_loop(enc, lengths, ctx, dp)  # noqa: SLF001
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


# --------------------------------------------------------------------------- #
# Predictor conv group size
# --------------------------------------------------------------------------- #


class TestPredictorConvGroupSize:
    """icefall's stateless predictor conv is grouped, not depthwise.

    ``nn.Conv1d(C, C, context_size, groups=C // group_size)``: group size 1 in the
    old ``pruned_transducer_stateless2/3/5`` recipes, **4** in every Zipformer one
    (``zipformer``, ``pruned_transducer_stateless7``).  Modelling it as fully
    depthwise made every real icefall release fail to load on that one tensor —
    4x the parameters, so no permute recovers it — which is why the size is
    carried in the config and read off the checkpoint rather than assumed.
    """

    @pytest.mark.parametrize("group_size", [1, 2, 4])
    def test_matches_icefall_grouped_conv1d(self, group_size):
        """The whole predictor, against icefall ``Decoder.forward`` verbatim.

        Compared with a tolerance rather than ``torch.equal`` on purpose: this is
        ``F.conv2d(groups=...)`` against ``F.conv1d(groups=...)``, two different
        algorithm choices, and their summation order need not agree.  Measured on
        this box, group sizes 1 and 4 come out bit-identical while 2 differs by
        2.4e-07 on 6 of 80 elements — so exactness here is incidental to the
        algorithm torch picks and is not the contract worth pinning.
        """
        import torch.nn.functional as F

        torch.manual_seed(0)
        vocab, dim, ctx = 32, 16, 2
        dec = StatelessDecoder(vocab, dim, context_size=ctx, conv_group_size=group_size).eval()
        # An icefall-layout weight for this group size, loaded through the hook.
        w_ice = torch.randn(dim, group_size, ctx)
        sd = dict(dec.state_dict())
        sd["conv.weight"] = w_ice
        dec.load_state_dict(sd)

        y = torch.randint(0, vocab, (5, ctx))
        with torch.inference_mode():
            got = dec(y)
            emb = F.embedding(y, dec.embedding.weight, padding_idx=0).permute(0, 2, 1)
            ref = F.conv1d(emb, w_ice, groups=dim // group_size).permute(0, 2, 1)
            ref = F.relu(ref)[:, -1, :]
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)

    def test_group_size_one_is_still_the_depthwise_layer(self):
        """The default must not change which operator the old configs get."""
        from oasr.layers import Conv2d, DepthwiseConv1d

        assert isinstance(StatelessDecoder(8, 16, context_size=2).conv, DepthwiseConv1d)
        assert isinstance(
            StatelessDecoder(8, 16, context_size=2, conv_group_size=4).conv,
            Conv2d,
        )

    def test_native_state_dict_round_trips_without_re_permuting(self):
        """The hook is gated on the 3-D icefall layout, not applied blindly.

        A native OASR checkpoint already holds the 4-D KRSC weight; re-permuting
        it on load would corrupt a round trip that looks successful.
        """
        torch.manual_seed(0)
        dec = StatelessDecoder(32, 16, context_size=2, conv_group_size=4).eval()
        reloaded = StatelessDecoder(32, 16, context_size=2, conv_group_size=4).eval()
        reloaded.load_state_dict(dec.state_dict())
        for k, v in dec.state_dict().items():
            assert torch.equal(v, reloaded.state_dict()[k]), k

    def test_bad_group_size_rejected(self):
        with pytest.raises(ValueError, match="divide decoder_dim"):
            StatelessDecoder(8, 12, context_size=2, conv_group_size=5)


# --------------------------------------------------------------------------- #
# Real icefall checkpoint
# --------------------------------------------------------------------------- #


@pytest.mark.requires_assets("TRANSDUCER_CKPT")
class TestRealIcefallTransducer:
    """The architecture's first real weights — see `.artifacts/known_issues.md` §3.

    Everything else in this file runs on random weights, which structurally
    cannot catch a converter that mis-reads a real export (the ``audio_scale``
    class of bug) or a predictor whose operator is wrong.
    """

    @pytest.fixture(scope="class")
    def bundle(self):
        import assets

        from oasr.models.registry import load_checkpoint_bundle

        return load_checkpoint_bundle(assets.require("TRANSDUCER_CKPT"), architecture="transducer")

    def test_config_is_inferred_from_the_weights(self, bundle):
        _, b = bundle
        cfg = b.model_config
        assert (cfg.vocab_size, cfg.decoder_dim, cfg.joiner_dim, cfg.context_size) == (
            500,
            512,
            512,
            2,
        )
        # The field this checkpoint exists to pin: read, not assumed.
        assert cfg.decoder_conv_group_size == 4
        assert cfg.encoder_type == "zipformer"
        assert cfg.encoder.num_encoder_layers == (2, 2, 3, 4, 3, 2)
        assert cfg.encoder.encoder_dim == (192, 256, 384, 512, 384, 256)

    def test_every_weight_loads(self, bundle):
        from oasr.models.registry import instantiate_from_bundle

        arch, b = bundle
        _, _, report = instantiate_from_bundle(arch, b)
        assert not report.missing
        # Only the pruned-RNNT training heads are dropped.
        assert {k.split(".")[0] for k in report.dropped} == {"simple_am_proj", "simple_lm_proj"}

    def test_predictor_matches_icefall_verbatim(self, bundle):
        """Bit-exact against ``icefall/.../decoder.py::Decoder.forward``.

        The grouping convention is the thing at risk: get it wrong and decoding
        degrades quietly instead of failing, because the shapes still line up.
        Observed bit-identical with these weights, but asserted with a tolerance —
        the two sides are different conv algorithms, so summation order is not
        contractual (see ``test_matches_icefall_grouped_conv1d``).
        """
        import torch.nn.functional as F

        from oasr.models.registry import instantiate_from_bundle

        arch, b = bundle
        model, cfg, _ = instantiate_from_bundle(arch, b)
        model.eval()
        w_ice = b.state_dict["decoder.conv.weight"]
        emb_w = b.state_dict["decoder.embedding.weight"]
        assert tuple(w_ice.shape) == (512, 4, 2)  # as shipped, before the hook

        torch.manual_seed(0)
        y = torch.randint(0, cfg.vocab_size, (7, cfg.context_size))
        with torch.inference_mode():
            got = model.decoder(y)
            e = F.embedding(y, emb_w, padding_idx=0).permute(0, 2, 1)
            e = F.conv1d(e, w_ice, groups=cfg.decoder_dim // cfg.decoder_conv_group_size)
            ref = F.relu(e.permute(0, 2, 1))[:, -1, :]
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
    def test_engine_transcribes_real_audio_and_beam1_is_greedy(self):
        """Also the gate that ``beam_size=1`` really is the greedy path.

        Pinned on real weights because a random-weight model emits near-uniform
        logits, where any tie-break ordering looks correct.
        """
        import assets
        import torchaudio

        from oasr.engine import ASREngine, EngineConfig

        ckpt = assets.require("TRANSDUCER_CKPT")
        wavs = assets.require_wavs(2)
        audios = [torchaudio.load(w)[0].squeeze(0) for w in wavs]

        def run(**opts):
            eng = ASREngine(
                EngineConfig(
                    ckpt_dir=ckpt,
                    architecture="transducer",
                    service_mode="offline",
                    dtype=torch.float16,
                    max_batch_size=2,
                    decode_options=opts,
                )
            )
            try:
                return eng.transcribe_offline(list(audios))
            finally:
                del eng
                torch.cuda.empty_cache()

        greedy = run()
        assert "printing in the only sense with which we are at present concerned" in (
            greedy[0].lower()
        )
        assert "in being comparatively modern" in greedy[1].lower()
        assert run(beam_size=1) == greedy


class TestPredictorStepGraph:
    """CUDA-graph capture of the predictor step (``oasr.engine.predictor_graph``).

    The step is nine launches for tens of microseconds of GPU work, so capturing
    it is worth ~1.3x on an offline Nemotron decode.  What has to be true is that
    it changes *nothing*: the captured graph runs the same kernels in the same
    order, so state, projections and transcripts must be bit-identical to the
    eager path.  Every test here is an equality, not a tolerance.
    """

    @staticmethod
    def _recurrent_model(vocab=24, enc_dim=16, hidden=32, layers=2, blank=0):
        """A tiny transducer whose predictor carries recurrent state."""
        from oasr.models.nemotron.predictor import NemotronRnntJoint, NemotronRnntPredictor

        torch.manual_seed(7)
        decoder = NemotronRnntPredictor(
            vocab_size=vocab, hidden_size=hidden, num_layers=layers, blank_id=blank
        )
        joiner = NemotronRnntJoint(enc_dim, hidden, vocab)
        return TransducerModel(nn.Identity(), decoder, joiner, blank_id=blank).eval()

    def _strategy(self, model, graphs, max_sym=3):
        cfg = SimpleNamespace(
            device="cuda",
            transducer_max_sym_per_frame=max_sym,
            partial_decode_interval=1,
            use_cuda_graphs=True,
            use_transducer_cuda_graphs=graphs,
        )
        detok = SimpleNamespace(
            detokenize=lambda ids: " ".join(map(str, ids)),
            new_state=lambda: {"ids": [], "text": ""},
            detokenize_incremental=lambda n, st: "",
        )
        return TransducerDecodeStrategy(cfg, detok, model)

    # -- contract, no CUDA needed ---------------------------------------

    def test_capturable_only_accepts_a_flat_cuda_tensor_state(self):
        from oasr.engine.predictor_graph import PredictorStepGraphCache as C

        assert not C.capturable(None)
        assert not C.capturable(())
        assert not C.capturable(torch.zeros(2))  # a bare tensor is not a state tuple
        assert not C.capturable((torch.zeros(2),))  # CPU
        assert not C.capturable(("x",))
        if torch.cuda.is_available():
            assert C.capturable((torch.zeros(2, device="cuda"),))
            assert not C.capturable((torch.zeros(2, device="cuda"), torch.zeros(2)))

    def test_detach_copies_every_tensor(self):
        from oasr.engine.predictor_graph import PredictorStepGraphCache as C

        src = (torch.zeros(2), torch.ones(3))
        out = C.detach(src)
        assert all(a is not b for a, b in zip(src, out))
        assert all(torch.equal(a, b) for a, b in zip(src, out))
        # A state shape it does not understand is handed back untouched.
        assert C.detach("opaque") == "opaque"

    # -- the graph reproduces the eager step exactly ---------------------

    @pytest.mark.cuda
    def test_graph_step_is_bit_identical_over_many_steps(self, device):
        """Replay must equal ``advance`` + ``decoder_proj``, step after step.

        Run the two arms forward together for several steps.  A single step could
        pass while the write-back into the graph's own input buffers is wrong; a
        chain cannot, because the second step would read the wrong state.
        """
        from oasr.engine.predictor_graph import PredictorStepGraphCache

        model = self._recurrent_model().to(device)
        decoder, joiner = model.decoder, model.joiner
        B = 4
        cache = PredictorStepGraphCache(decoder, joiner)
        with torch.inference_mode():
            eager = decoder.init_state(B, device)
            graphed = tuple(t.clone() for t in eager)
            for step in range(6):
                tok = torch.arange(B, device=device, dtype=torch.long) + step + 1
                emit = torch.tensor([True, False, True, True], device=device)
                eager = decoder.advance(eager, tok, emit)
                eager_proj = joiner.decoder_proj(decoder.predict(eager))
                got = cache.step(graphed, tok, emit)
                assert got is not None, "capture declined on a capturable state"
                graphed, proj = got
                for i, (a, b) in enumerate(zip(eager, graphed)):
                    assert torch.equal(a, b), f"state[{i}] diverged at step {step}"
                assert torch.equal(eager_proj, proj), f"projection diverged at step {step}"
                # Keep the eager arm's own copy: the graph's is about to be reused.
                graphed = cache.detach(graphed)
        assert cache.num_captured == 1

    @pytest.mark.cuda
    def test_steady_state_needs_no_input_copy(self, device):
        """Handing the graph's own state back must hit the copy-free path."""
        from oasr.engine.predictor_graph import PredictorStepGraphCache

        model = self._recurrent_model().to(device)
        cache = PredictorStepGraphCache(model.decoder, model.joiner)
        B = 2
        with torch.inference_mode():
            state = model.decoder.init_state(B, device)
            tok = torch.zeros(B, dtype=torch.long, device=device)
            emit = torch.ones(B, dtype=torch.bool, device=device)
            state, _ = cache.step(state, tok, emit)
            again, _ = cache.step(state, tok, emit)
        assert all(
            a is b for a, b in zip(state, again)
        ), "the cache must return its own buffers so the next call skips the copy"

    @pytest.mark.cuda
    def test_capture_budget_declines_rather_than_growing(self, device):
        from oasr.engine.predictor_graph import PredictorStepGraphCache

        model = self._recurrent_model().to(device)
        cache = PredictorStepGraphCache(model.decoder, model.joiner, max_captures=1)
        with torch.inference_mode():
            for B, expect in ((2, True), (3, False)):
                state = model.decoder.init_state(B, device)
                tok = torch.zeros(B, dtype=torch.long, device=device)
                emit = torch.ones(B, dtype=torch.bool, device=device)
                got = cache.step(state, tok, emit)
                assert (got is not None) is expect, f"B={B}"
        assert cache.num_captured == 1

    @pytest.mark.cuda
    def test_a_failed_capture_is_not_retried(self, device, monkeypatch):
        """A capture costs a warm-up forward, so a failure must be remembered."""
        from oasr.engine import predictor_graph as pg

        model = self._recurrent_model().to(device)
        cache = pg.PredictorStepGraphCache(model.decoder, model.joiner)
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise RuntimeError("nope")

        monkeypatch.setattr(model.decoder, "advance", boom)
        with torch.inference_mode():
            # Built by hand, not via ``init_state`` — that calls ``_step``, and
            # the patched ``advance`` is what has to be the thing that fails.
            state = (
                torch.zeros(2, 32, device=device),
                torch.zeros(2, 2, 32, device=device),
                torch.zeros(2, 2, 32, device=device),
            )
            tok = torch.zeros(2, dtype=torch.long, device=device)
            emit = torch.ones(2, dtype=torch.bool, device=device)
            assert cache.step(state, tok, emit) is None
            n = len(calls)
            assert cache.step(state, tok, emit) is None
        assert len(calls) == n, "a remembered failure must not pay for another warm-up"

    # -- end to end through the greedy loop ------------------------------

    @pytest.mark.cuda
    @pytest.mark.parametrize("batch", [1, 3])
    def test_offline_decode_matches_the_eager_path(self, device, batch):
        model = self._recurrent_model().to(device)
        torch.manual_seed(11)
        enc = torch.randn(batch, 9, 16, device=device)
        lengths = torch.full((batch,), 9, dtype=torch.long, device=device)
        off = self._strategy(model, graphs=False)
        on = self._strategy(model, graphs=True)
        with torch.inference_mode():
            s0, p0 = off._init_state(batch, device)
            a = off._greedy_loop(enc, lengths, s0, p0)[0]
            s1, p1 = on._init_state(batch, device)
            b = on._greedy_loop(enc, lengths, s1, p1)[0]
        assert a == b
        assert on._pred_graphs is not None and on._pred_graphs.num_captured >= 1

    @pytest.mark.cuda
    def test_the_loop_syncs_per_iteration_only_for_word_timings(self, device):
        """``bool(emit.any())`` is a host sync inside the per-frame loop.

        Dropping it lets the loop run fully async: the predictor step is masked by
        ``emit`` either way, so a no-emission iteration mutates nothing — it only
        costs one more ``(B,)`` snapshot in ``emitted``, ~1.1 us per entry at the
        single readback after the loop, against a sync that makes the host wait
        out everything the iteration queued.  Worth 1.04-1.17x on nemotron.

        It is kept for word timings, where the per-step snapshot adds four eager
        launches (clone, gather, vocab-wide logsumexp, exp) to *every* iteration
        instead of to the emitting ones, and at batch 8 that costs more than the
        sync saves (0.995x).

        Counted, not timed: ``bool(tensor)`` goes through ``Tensor.__bool__``, so
        the number of per-iteration syncs is observable exactly.  The loop keeps
        one *other* sync — the termination check, once per
        ``_TERMINATION_CHECK_STRIDE`` iterations — so the assertion is that the
        no-timing path is bounded by that, not that it is zero.
        """
        from oasr.engine.decode import transducer as td_mod

        model = self._recurrent_model().to(device)
        strat = self._strategy(model, graphs=False, max_sym=3)
        B, T = 4, 12
        torch.manual_seed(11)
        enc = torch.randn(B, T, 16, device=device)
        lengths = torch.full((B,), T, dtype=torch.long, device=device)

        counter = {"n": 0}
        real_bool = torch.Tensor.__bool__

        def counting_bool(self):
            counter["n"] += 1
            return real_bool(self)

        def count(track):
            counter["n"] = 0
            state, proj = strat._init_state(B, device)
            torch.Tensor.__bool__ = counting_bool
            try:
                with torch.inference_mode():
                    hyps = strat._greedy_loop(enc, lengths, state, proj, track=track)[0]
            finally:
                torch.Tensor.__bool__ = real_bool
            return counter["n"], hyps

        n_plain, hyps_plain = count(False)
        n_timed, hyps_timed = count(True)

        # Same tokens either way: the two paths differ only in when the host looks.
        assert hyps_plain == hyps_timed

        stride = td_mod._TERMINATION_CHECK_STRIDE
        max_steps = T * (3 + 1) + B + 1
        budget = max_steps // stride + 4  # termination checks, plus slack
        assert n_plain <= budget, (
            f"the no-word-timing path made {n_plain} host syncs for at most "
            f"{max_steps} iterations; only the termination check (every {stride}) "
            f"should sync, so at most ~{budget}"
        )
        # And the word-timing path still branches per iteration, so it syncs more.
        assert n_timed > n_plain, (
            f"word-timing path made {n_timed} syncs vs {n_plain} without — the "
            "per-iteration branch it needs is gone"
        )

    @pytest.mark.cuda
    def test_two_session_groups_in_one_tick_do_not_share_graph_state(self, device):
        """The state a group keeps must survive *another* group's replay.

        Streams are grouped by chunk length, so one tick can run
        ``_greedy_loop`` several times.  It hands back the graph's own buffers,
        and the streaming path stores per-session state sliced out with
        ``unstack_states`` — which for a recurrent predictor returns *views*.  So
        group A stores views, group B replays the same graph, and group A's
        stored state is now group B's.

        The single-group case cannot show this: ``stack_states`` copies, and it
        runs before the next replay.  Two groups per tick is what opens the
        window, which is why this keeps A's rows *unrestacked* across B's call.

        The assertion is on the **state tensors**, not on the transcript: a tiny
        randomly-initialised model's argmax barely depends on the predictor, so
        comparing hypotheses lets the corruption through.  Revert the ``detach``
        in ``_greedy_loop`` and this fails on the first tick.
        """
        model = self._recurrent_model().to(device)
        torch.manual_seed(13)
        a_chunks = [torch.randn(2, 5, 16, device=device) for _ in range(3)]
        b_chunks = [torch.randn(2, 5, 16, device=device) * 3 for _ in range(3)]
        lengths = torch.full((2,), 5, dtype=torch.long, device=device)

        def rollout(graphs):
            strat = self._strategy(model, graphs=graphs)
            with torch.inference_mode():
                sa, pa = strat._init_state(2, device)
                sb, pb = strat._init_state(2, device)
                trace = []
                for ca, cb in zip(a_chunks, b_chunks):
                    ha, _, sa, pa = strat._greedy_loop(ca, lengths, sa, pa)
                    # What the streaming path stores: per-session rows.
                    rows_a = model.decoder.unstack_states(sa)
                    proj_a = [pa[i : i + 1] for i in range(2)]
                    # ...and then another group runs in the same tick.
                    hb, _, sb, pb = strat._greedy_loop(cb, lengths, sb, pb)
                    # ...and only now does A restack what it stored.
                    sa = model.decoder.stack_states(rows_a)
                    pa = torch.cat(proj_a, dim=0)
                    trace.append(
                        (
                            [list(h) for h in ha],
                            [list(h) for h in hb],
                            [t.clone() for t in sa],
                            pa.clone(),
                        )
                    )
                return trace

        eager, graphed = rollout(False), rollout(True)
        for tick, (e, g) in enumerate(zip(eager, graphed)):
            assert e[0] == g[0], f"tick {tick}: group A hypotheses diverged"
            assert e[1] == g[1], f"tick {tick}: group B hypotheses diverged"
            for i, (x, y) in enumerate(zip(e[2], g[2])):
                assert torch.equal(x, y), f"tick {tick}: group A state[{i}] diverged"
            assert torch.equal(e[3], g[3]), f"tick {tick}: group A projection diverged"
