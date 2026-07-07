# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""In-tree GPU WFST decoder (oasr.decoder.wfst_decoder) tests.

Two groups:

* Self-contained toy-graph tests build a tiny .img in a tmp dir and exercise the full
  stack (graph load -> GPU decode -> backtrack) with no external assets. They run
  whenever CUDA is present and oasr was built with ``OASR_USE_WFST_DECODER=1``.
* Real-graph smoke tests drive the public ``oasr.decode.Decoder`` API through the GPU
  backend; they need a decoding graph via ``OASR_TEST_FST`` (an HLG ``.pt`` or a prebuilt
  ``.img``).
"""

import math
import os

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


def _c_decoder():
    """Return oasr._C.decoder if the in-tree GPU WFST decoder is built, else skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    try:
        import oasr._C as _C
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"oasr._C unavailable: {exc}")
    dec = getattr(_C, "decoder", None)
    if dec is None or not getattr(dec, "wfst_decoder_available", False):
        pytest.skip("in-tree GPU WFST decoder not built (reinstall with OASR_USE_WFST_DECODER=1)")
    return dec


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


@pytest.fixture(scope="module")
def toy_fst(tmp_path_factory):
    _c_decoder()  # skip early if no CUDA / not built
    path = str(tmp_path_factory.mktemp("wfst") / "toy.img")
    _write_toy_graph(path)
    return path


# ---------------------------------------------------------------------------
# Self-contained toy-graph tests (no external assets)
# ---------------------------------------------------------------------------


def test_toy_gpu_matches_cpu_reference(toy_fst):
    """GPU decode == CPU reference == the expected word sequence."""
    dec = _c_decoder()
    g = dec.load_graph(toy_fst)
    lp = _toy_logp([1, 2])  # exactly consume labels 1, 2

    cpu = dec.cpu_decode(g, lp, search_beam=20.0, output_beam=8.0,
                         min_active=1, max_active=100, allow_partial=True)
    gd = dec.GpuDecoder(g, search_beam=20.0, output_beam=8.0, min_active=1, max_active=100,
                        allow_partial=True, max_lanes=2, max_frames=16, device=0,
                        main_q_factor=8, cand_factor=6)
    outs = gd.decode_batch(lp.unsqueeze(0).cuda().contiguous(),
                           torch.tensor([lp.size(0)], dtype=torch.int32))
    gpu = outs[0]

    assert cpu["ok"] and gpu["ok"]
    assert list(cpu["words"]) == _TOY_WORDS
    assert list(gpu["words"]) == list(cpu["words"])
    assert gpu["score"] == pytest.approx(cpu["score"], abs=1e-4)


def test_toy_batched_offline_equivalence(toy_fst):
    """decode_offline_batch on a padded, mixed-length batch == per-row decode_offline."""
    _c_decoder()
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
    _c_decoder()  # CUDA + in-tree build
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
