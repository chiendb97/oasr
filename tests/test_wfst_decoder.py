# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""In-tree GPU WFST decoder (oasr.decoder.wfst_decoder) tests.

Two groups:

* Self-contained toy-graph tests build a tiny .img in a tmp dir and exercise the full
  stack (JIT compile -> graph load -> GPU decode -> backtrack) with no external assets.
  They run whenever CUDA is present and the JIT module compiles.
* Real-graph smoke tests drive the public ``oasr.decode.Decoder`` API through the GPU
  backend; they need a decoding graph via ``OASR_TEST_FST`` (an HLG ``.pt`` or a prebuilt
  ``.img``).

The CUDA decoder is JIT-compiled on first use (``oasr.jit.wfst_decoder``); the stateful
decoder lives behind an int64 handle and results come back through CPU output tensors.
"""

import math
import os

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


def _module():
    """Return the JIT-compiled WFST decoder module, or skip if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    try:
        from oasr.decoder.wfst_decoder import _get_module

        return _get_module()
    except Exception as exc:  # pragma: no cover - JIT toolchain missing
        pytest.skip(f"WFST decoder JIT module unavailable: {exc}")


def _cpu_reference_module():
    """Return the test-only WFST CPU reference JIT module (the parity oracle), or skip.

    Built separately from the production decoder module (``gen_wfst_cpu_reference_module``)
    so the shipping decoder carries no reference-decoder code.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    try:
        from oasr.jit.wfst_decoder import gen_wfst_cpu_reference_module

        return gen_wfst_cpu_reference_module().build_and_load()
    except Exception as exc:  # pragma: no cover - JIT toolchain missing
        pytest.skip(f"WFST CPU reference JIT module unavailable: {exc}")


# ---------------------------------------------------------------------------
# Toy graph: 0 -(1)-> 1 -(2)-> 2 -(-1)-> 3, with blank (label 0) self-loops so it
# accepts [blank* 1 blank* 2 blank*] -> words [10, 20]. Epsilon-free, k2 conventions.
# ---------------------------------------------------------------------------

_TOY_VOCAB = 3
_TOY_WORDS = [10, 20]


def _write_toy_graph(path: str) -> None:
    from oasr.decoder.wfst.graph_image import build_image, write_image

    num_states = 4
    # (src, dest, ilabel, weight), grouped by src, emitting-before-final within each src.
    arcs = [
        (0, 0, 0, 0.0),   # 0: blank self-loop
        (0, 1, 1, 0.0),   # 1: 0 -> 1 on label 1  (word 10)
        (1, 1, 0, 0.0),   # 2: blank self-loop
        (1, 2, 2, 0.0),   # 3: 1 -> 2 on label 2  (word 20)
        (2, 2, 0, 0.0),   # 4: blank self-loop
        (2, 3, -1, 0.0),  # 5: 2 -> super-final (final arc)
    ]
    aux = {1: [10], 3: [20]}

    row_splits = np.zeros(num_states + 1, dtype=np.int64)
    for src, *_ in arcs:
        row_splits[src + 1] += 1
    row_splits = np.cumsum(row_splits).astype(np.int32)
    dest = np.array([a[1] for a in arcs], dtype=np.int32)
    ilabel = np.array([a[2] for a in arcs], dtype=np.int32)
    weight = np.array([a[3] for a in arcs], dtype=np.float32)
    aux_row_splits = np.zeros(len(arcs) + 1, dtype=np.int32)
    pool: list[int] = []
    for i in range(len(arcs)):
        pool.extend(aux.get(i, []))
        aux_row_splits[i + 1] = len(pool)

    img = build_image(
        row_splits, dest, ilabel, weight,
        aux_row_splits=aux_row_splits, aux_pool=np.array(pool, dtype=np.int32),
        vocab_size=_TOY_VOCAB,
    )
    write_image(img, path)


def _toy_logp(labels, device="cpu") -> torch.Tensor:
    """[T, V] log-probs where frame t strongly favors labels[t] (0 == blank)."""
    lp = torch.full((len(labels), _TOY_VOCAB), -10.0, dtype=torch.float32, device=device)
    for t, lab in enumerate(labels):
        lp[t, lab] = 0.0
    return lp


def _cpu_decode(graph_path, logp_cpu, min_active=1, max_active=100):
    """Run the test-only CPU reference oracle over the .img at *graph_path*.

    Returns (words, score, ok). The oracle loads the graph image itself, so it takes a
    path rather than a production graph handle.
    """
    mod = _cpu_reference_module()
    cap = max(1, logp_cpu.size(0) * 4)
    out_words = torch.empty((cap,), dtype=torch.int32)
    out_wlen = torch.empty((1,), dtype=torch.int32)
    out_score = torch.empty((1,), dtype=torch.float64)
    out_meta = torch.empty((3,), dtype=torch.int32)
    mod.wfst_cpu_decode(graph_path, logp_cpu, 20.0, 8.0, min_active, max_active, 1, 0, 3,
                        out_words, out_wlen, out_score, out_meta)
    length = min(int(out_wlen[0]), cap)
    return out_words[:length].tolist(), float(out_score[0]), bool(out_meta[0])


@pytest.fixture(scope="module")
def toy_fst(tmp_path_factory):
    _module()  # skip early if no CUDA / JIT unavailable
    path = str(tmp_path_factory.mktemp("wfst") / "toy.img")
    _write_toy_graph(path)
    return path


# ---------------------------------------------------------------------------
# Self-contained toy-graph tests (no external assets)
# ---------------------------------------------------------------------------


def test_toy_gpu_matches_cpu_reference(toy_fst):
    """GPU decode == CPU reference == the expected word sequence."""
    from oasr.decoder.wfst_decoder import WfstDecoderOptions, WfstDecoderSearch

    opts = WfstDecoderOptions(min_active_states=1, max_active_states=100,
                              blank_skip_thresh=1.0, max_frames=16, max_offline_lanes=2)
    searcher = WfstDecoderSearch(toy_fst, opts)
    tokens, scores = searcher.decode_offline(_toy_logp([1, 2], "cuda"))
    gpu_words, gpu_score = tokens[0], scores[0]

    cpu_words, cpu_score, cpu_ok = _cpu_decode(toy_fst, _toy_logp([1, 2]))

    assert cpu_ok
    assert cpu_words == _TOY_WORDS
    assert gpu_words == cpu_words
    assert gpu_score == pytest.approx(cpu_score, abs=1e-4)


def test_toy_batched_offline_equivalence(toy_fst):
    """decode_offline_batch on a padded, mixed-length batch == per-row decode_offline."""
    _module()
    from oasr.decoder.wfst_decoder import WfstDecoderOptions, WfstDecoderSearch

    opts = WfstDecoderOptions(min_active_states=1, max_active_states=100,
                              blank_skip_thresh=1.0, max_frames=32, max_offline_lanes=4)
    searcher = WfstDecoderSearch(toy_fst, opts)

    rows = [
        _toy_logp([1, 2], "cuda"),        # T=2
        _toy_logp([1, 0, 2], "cuda"),     # T=3, blank in the middle
        _toy_logp([1, 2, 0], "cuda"),     # T=3, trailing blank
    ]
    per_row = [searcher.decode_offline(r) for r in rows]  # each -> ([[words]], [score])

    max_t = max(r.size(0) for r in rows)
    packed = rows[0].new_full((len(rows), max_t, _TOY_VOCAB), -30.0)
    lens = []
    for i, r in enumerate(rows):
        packed[i, : r.size(0)] = r
        lens.append(r.size(0))
    b_tokens, b_scores = searcher.decode_offline_batch(packed, torch.tensor(lens))

    for i, (tok, sc) in enumerate(per_row):
        assert list(b_tokens[i]) == list(tok[0]), (i, b_tokens[i], tok[0])
        assert b_scores[i] == pytest.approx(sc[0], abs=1e-4), (i, b_scores[i], sc[0])
    # Sanity: the clean single-utterance row decodes to the expected words.
    assert list(b_tokens[0]) == _TOY_WORDS


def test_toy_winners_gc_matches_plain(toy_fst):
    """gc_interval > 0 (segmented decode + winners-log GC + host prefix merge) must
    reproduce the plain decode exactly, including across repeats on the same decoder."""
    _module()
    from oasr.decoder.wfst_decoder import WfstDecoderOptions, WfstDecoderSearch

    labels = ([1] + [0] * 29 + [2] + [0] * 29) * 1  # T=60 >> gc_interval
    logp = _toy_logp(labels, "cuda")
    plain = WfstDecoderSearch(
        toy_fst,
        WfstDecoderOptions(min_active_states=1, max_active_states=100,
                           blank_skip_thresh=1.0, max_frames=64, max_offline_lanes=2),
    )
    gc = WfstDecoderSearch(
        toy_fst,
        WfstDecoderOptions(min_active_states=1, max_active_states=100,
                           blank_skip_thresh=1.0, max_frames=64, max_offline_lanes=2,
                           gc_interval=8),
    )
    t0, s0 = plain.decode_offline(logp)
    t1, s1 = gc.decode_offline(logp)
    t2, s2 = gc.decode_offline(logp)  # repeat: window/prefix reset across batches
    assert t0[0] == _TOY_WORDS
    assert t1[0] == t0[0] and t2[0] == t0[0]
    assert s1[0] == pytest.approx(s0[0], abs=1e-6)
    assert s2[0] == s1[0]


def test_toy_streaming_gc_recovers_full_path(toy_fst):
    """Per-chunk streaming GC (ring + host-drained prefix) returns the FULL word
    sequence for streams longer than path_cap; the legacy (gc=0) decoder truncates the
    device backtrack to the newest chunk+2 arcs and loses the early words."""
    import torch

    from oasr.decoder.wfst_decoder import _get_graph
    from oasr.jit.wfst_decoder import gen_wfst_decoder_module

    mod = _module()
    graph = _get_graph(toy_fst)
    labels = [1] + [0] * 40 + [2] + [0] * 18  # T=60 >> chunk+2
    logp = _toy_logp(labels, "cuda")
    chunk = 8

    def run(gc):
        dec = int(mod.wfst_create_decoder(
            graph, 20.0, 8.0, 1, 100, 1, 1, chunk, 0, 16, 4, 1, 0, 0, 1, 0, 3, 0, 0, gc))
        ch = int(mod.wfst_create_stream(dec))
        assert ch >= 0
        for s in range(0, logp.size(0), chunk):
            piece = logp[s : s + chunk].unsqueeze(0).contiguous()
            outs = [torch.empty((1, 8), dtype=torch.int32),
                    torch.empty((1,), dtype=torch.int32),
                    torch.empty((1,), dtype=torch.int32),
                    torch.empty((1,), dtype=torch.int32)]
            mod.wfst_advance_chunk(dec, torch.tensor([ch], dtype=torch.int32), piece,
                                   torch.tensor([piece.size(1)], dtype=torch.int32), 0,
                                   *outs)
        ow = torch.empty((64,), dtype=torch.int32)
        wl = torch.empty((1,), dtype=torch.int32)
        sc = torch.empty((1,), dtype=torch.float64)
        meta = torch.empty((3,), dtype=torch.int32)
        mod.wfst_finalize_stream(dec, ch, ow, wl, sc, meta)
        words = ow[: min(int(wl[0]), 64)].tolist()
        mod.wfst_free_decoder(dec)
        return words, float(sc[0]), int(meta[0])

    w_gc, s_gc, ok_gc = run(gc=2)
    w_legacy, s_legacy, ok_legacy = run(gc=0)
    assert ok_gc and ok_legacy
    assert s_gc == pytest.approx(s_legacy, abs=1e-6)  # decode dynamics untouched
    assert w_gc == _TOY_WORDS  # full path recovered (word 10 sits on the first arc)
    assert w_legacy == w_gc[len(w_gc) - len(w_legacy):]  # legacy = truncated suffix
    assert len(w_legacy) < len(w_gc)


# ---------------------------------------------------------------------------
# Real-graph smoke tests via the public Decoder API (need OASR_TEST_FST)
# ---------------------------------------------------------------------------

FST = os.environ.get(
    "OASR_TEST_FST",
    "/data01/kilm/users/chiendb/models/asr/lm/20210610_u2pp_conformer_exp_librispeech"
    "/lang_bpe/HLG.pt",
)


@pytest.fixture(scope="module")
def wfst_decoder_cls():
    _module()  # CUDA + JIT available
    if not os.path.exists(FST):
        pytest.skip(f"no decoding graph at {FST} (set OASR_TEST_FST)")
    from oasr.decode import Decoder, DecoderConfig

    return Decoder, DecoderConfig


def _fake_logp(t: int, vocab: int, device) -> torch.Tensor:
    logp = torch.full((t, vocab), math.log(1e-6), device=device)
    logp[:, 0] = math.log(0.9)  # blank-dominated frames
    return torch.log_softmax(logp, dim=-1)


def test_offline_smoke(wfst_decoder_cls):
    Decoder, DecoderConfig = wfst_decoder_cls
    dec = Decoder(DecoderConfig(search_type="wfst", wfst_blank_skip_thresh=1.0), fst=FST)
    logp = _fake_logp(20, 5002, "cuda")
    result = dec.decode(logp)
    assert isinstance(result.tokens, list) and len(result.tokens) == 1
    assert isinstance(result.scores[0], float)


def test_streaming_matches_repeat(wfst_decoder_cls):
    Decoder, DecoderConfig = wfst_decoder_cls
    cfg = DecoderConfig(search_type="wfst", wfst_blank_skip_thresh=1.0)
    dec = Decoder(cfg, fst=FST)
    logp = _fake_logp(30, 5002, "cuda")

    def run():
        dec.init_stream()
        for s in range(0, logp.size(0), 7):
            dec.decode_chunk(logp[s : s + 7])
        return dec.finalize_stream()

    a, b = run(), run()  # channel reuse must fully reset state
    assert a.tokens == b.tokens
    assert a.scores[0] == pytest.approx(b.scores[0], abs=1e-4)


def test_channel_released_on_finalize(wfst_decoder_cls):
    Decoder, DecoderConfig = wfst_decoder_cls
    cfg = DecoderConfig(search_type="wfst", wfst_blank_skip_thresh=1.0)
    logp = _fake_logp(8, 5002, "cuda")
    # More sequential streams than max_streams: works only if channels are released.
    for _ in range(40):
        dec = Decoder(cfg, fst=FST)
        dec.init_stream()
        dec.decode_chunk(logp)
        dec.finalize_stream()
