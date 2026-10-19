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

import assets
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
        (0, 0, 0, 0.0),  # 0: blank self-loop
        (0, 1, 1, 0.0),  # 1: 0 -> 1 on label 1  (word 10)
        (1, 1, 0, 0.0),  # 2: blank self-loop
        (1, 2, 2, 0.0),  # 3: 1 -> 2 on label 2  (word 20)
        (2, 2, 0, 0.0),  # 4: blank self-loop
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
        row_splits,
        dest,
        ilabel,
        weight,
        aux_row_splits=aux_row_splits,
        aux_pool=np.array(pool, dtype=np.int32),
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
    mod.wfst_cpu_decode(
        graph_path,
        logp_cpu,
        20.0,
        8.0,
        min_active,
        max_active,
        1,
        0,
        3,
        out_words,
        out_wlen,
        out_score,
        out_meta,
    )
    length = min(int(out_wlen[0]), cap)
    return out_words[:length].tolist(), float(out_score[0]), bool(out_meta[0])


@pytest.fixture(scope="module")
def toy_fst(tmp_path_factory):
    _module()  # skip early if no CUDA / JIT unavailable
    path = str(tmp_path_factory.mktemp("wfst") / "toy.img")
    _write_toy_graph(path)
    return path


# ---------------------------------------------------------------------------
# Loop graph: one state that accepts any label sequence and emits word 100 + k for
# each frame of label k.  Every input decodes to itself, so a caller handed another
# caller's result is visible without a reference decode.
# ---------------------------------------------------------------------------

_LOOP_VOCAB = 32


def _write_loop_graph(path: str) -> None:
    from oasr.decoder.wfst.graph_image import build_image, write_image

    # State 0: blank self-loop, one self-loop per label, then the final arc to state 1.
    labels = list(range(_LOOP_VOCAB))
    dest = [0] * len(labels) + [1]
    ilabel = labels + [-1]
    aux_row_splits = [0]
    pool: list[int] = []
    for k in ilabel:
        if k > 0:
            pool.append(100 + k)
        aux_row_splits.append(len(pool))
    img = build_image(
        np.array([0, len(ilabel), len(ilabel)], dtype=np.int32),
        np.array(dest, dtype=np.int32),
        np.array(ilabel, dtype=np.int32),
        np.zeros(len(ilabel), dtype=np.float32),
        aux_row_splits=np.array(aux_row_splits, dtype=np.int32),
        aux_pool=np.array(pool, dtype=np.int32),
        vocab_size=_LOOP_VOCAB,
    )
    write_image(img, path)


def _loop_case(seed: int, frames: int):
    """``([T, V] CUDA log-probs, the words they must decode to)``."""
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, _LOOP_VOCAB, (frames,), generator=gen).tolist()
    lp = torch.full((frames, _LOOP_VOCAB), -10.0)
    lp[torch.arange(frames), torch.tensor(labels)] = 0.0
    return lp.cuda(), [100 + k for k in labels if k > 0]


def _loop_opts(**kw):
    from oasr.decoder.wfst_decoder import WfstDecoderOptions

    fields = {
        "min_active_states": 1,
        "max_active_states": 100,
        "blank_skip_thresh": 1.0,
        "max_frames": 256,
        "max_offline_lanes": 4,
        "max_streams": 16,
        **kw,
    }
    return WfstDecoderOptions(**fields)


@pytest.fixture(scope="module")
def loop_fst(tmp_path_factory):
    _module()
    path = str(tmp_path_factory.mktemp("wfst") / "loop.img")
    _write_loop_graph(path)
    return path


# ---------------------------------------------------------------------------
# Self-contained toy-graph tests (no external assets)
# ---------------------------------------------------------------------------


def test_toy_gpu_matches_cpu_reference(toy_fst):
    """GPU decode == CPU reference == the expected word sequence."""
    from oasr.decoder.wfst_decoder import WfstDecoderOptions, WfstDecoderSearch

    opts = WfstDecoderOptions(
        min_active_states=1,
        max_active_states=100,
        blank_skip_thresh=1.0,
        max_frames=16,
        max_offline_lanes=2,
    )
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

    opts = WfstDecoderOptions(
        min_active_states=1,
        max_active_states=100,
        blank_skip_thresh=1.0,
        max_frames=32,
        max_offline_lanes=4,
    )
    searcher = WfstDecoderSearch(toy_fst, opts)

    rows = [
        _toy_logp([1, 2], "cuda"),  # T=2
        _toy_logp([1, 0, 2], "cuda"),  # T=3, blank in the middle
        _toy_logp([1, 2, 0], "cuda"),  # T=3, trailing blank
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


def _toy_strategy(toy_fst, **decoder):
    """The engine's ``ctc_wfst`` strategy over the toy graph, built as the engine builds it."""
    from types import SimpleNamespace

    from oasr.decode import DecoderConfig
    from oasr.engine.decode import Detokenizer, build_decode_strategy

    fields = {
        "search_type": "wfst",
        "wfst_min_active_states": 1,
        "wfst_max_active_states": 100,
        "wfst_blank_skip_thresh": 1.0,
        **decoder,
    }
    cfg = SimpleNamespace(
        decoder_type="ctc_wfst",
        device="cuda",
        dtype=None,
        max_batch_size=2,
        use_cuda_graphs=False,
        use_ctc_cuda_graphs=False,
        _model_config=SimpleNamespace(vocab_size=_TOY_VOCAB),
        ctc_decoder_config=None,
        wfst_decoder_config=DecoderConfig(**fields),
        fst_path=toy_fst,
    )
    # The CTC capability surface; the strategy never calls either member.
    model = SimpleNamespace(head=lambda *a: None, forward_offline=lambda *a: None)
    return build_decode_strategy("ctc", cfg, Detokenizer(None, None), model)


def _best(output) -> list:
    return list(output.tokens[0]) if output.tokens else []


def test_toy_strategy_decode_offline_does_not_fail_the_batch(toy_fst):
    """The engine's ``ctc_wfst`` strategy decodes a batch — not just the decoder.

    ``CtcWfstDecodeStrategy.decode_offline`` read ``decoder_config.blank_id``, a
    field ``DecoderConfig`` does not have (it is ``blank``).  The keyword was
    evaluated on every call, so every offline batch raised and the executor's
    failure isolation returned an empty ``finish_reason="error"`` for every
    request.  The decoder-level tests above could not see it: they never go
    through the strategy.
    """
    _module()
    strategy = _toy_strategy(toy_fst)
    assert strategy.speech_activity_kwargs() == {"blank_id": 0}

    rows = [_toy_logp([1, 2, 0], "cuda"), _toy_logp([1, 0, 2], "cuda")]
    log_probs = torch.stack(rows)
    lengths = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
    outputs = strategy.decode_offline(log_probs, lengths)
    assert [o.tokens[0] for o in outputs] == [_TOY_WORDS, _TOY_WORDS]
    assert all(o.finished for o in outputs)


def test_toy_strategy_turns_do_not_repeat_the_previous_one(toy_fst):
    """Audit #11: a ``vad.mode="segment"`` turn boundary, decode side.

    The executor finalizes the decoder, then frees and re-creates the session on
    the *same* request.  Both used to be no-ops, so the next turn searched on the
    finalized decoder.  With blank skipping on (the default), a turn of silence
    never reaches the search at all, and its partial and its final were the
    previous turn's words — so "A B" then a pause finalized as "A B A B".
    """
    from oasr.engine.request import Request

    strategy = _toy_strategy(toy_fst, wfst_blank_skip_thresh=0.98)
    speech = _toy_logp([1, 0, 2], "cuda").unsqueeze(0)
    silence = _toy_logp([0, 0, 0, 0], "cuda").unsqueeze(0)
    req = Request(request_id="turns", streaming=True)
    strategy.create_session(req)

    def close_turn():  # StreamingExecutor._close_turn, as the strategy sees it
        final = strategy.finalize(req)
        strategy.free_session(req)
        strategy.create_session(req)
        return _best(final)

    assert _best(strategy.decode_streaming_chunk(req, speech)) == _TOY_WORDS
    assert close_turn() == _TOY_WORDS

    assert _best(strategy.decode_streaming_chunk(req, silence)) == [], "partial repeated a turn"
    assert close_turn() == [], "a turn of silence finalized as the previous turn"

    assert _best(strategy.decode_streaming_chunk(req, speech)) == _TOY_WORDS
    assert close_turn() == _TOY_WORDS, "a later turn carried an earlier one's words"
    strategy.free_session(req)


def test_toy_strategy_free_session_returns_the_channel(toy_fst):
    """An aborted stream is freed without being finalized, and the GPU decoder's
    channels are shared by every stream in the process.  Releasing the channel
    only when the request is garbage-collected let aborted streams that were
    still referenced exhaust them."""
    from oasr.engine.request import Request

    strategy = _toy_strategy(toy_fst, wfst_max_streams=2)
    speech = _toy_logp([1, 2], "cuda").unsqueeze(0)
    held = []
    for i in range(5):  # more aborted streams than channels, all still referenced
        req = Request(request_id=f"aborted-{i}", streaming=True)
        strategy.create_session(req)
        assert _best(strategy.decode_streaming_chunk(req, speech)) == _TOY_WORDS
        strategy.free_session(req)
        held.append(req)


# ---------------------------------------------------------------------------
# Concurrency and ownership (audit #12)
# ---------------------------------------------------------------------------


def test_concurrent_callers_of_one_decoder_each_get_their_own_result(loop_fst):
    """Audit #12: TVM-FFI releases the GIL around every decoder call, so two engine
    workers in one process can be inside one ``GpuDecoder`` at once — and it has one
    stream, one staging buffer and one lane-length table.  Without its mutex, callers
    decoded against each other's inputs (2 of 40 streams wrong on the LibriSpeech
    graph) or crashed the process.

    Every thread here shares the process-wide decoders on purpose: the engine no
    longer does, so this is the only place the lock itself is exercised.
    """
    import threading

    from oasr.decoder.wfst_decoder import WfstDecoderSearch

    threads, rounds = 4, 8
    offline = [[_loop_case(100 * t + i, 192) for i in range(4)] for t in range(threads)]
    streams = [[_loop_case(1000 + 10 * t + i, 160) for i in range(3)] for t in range(threads)]
    wrong, errors = [], []

    def run(t):
        try:
            search = WfstDecoderSearch(loop_fst, _loop_opts())
            packed = torch.stack([lp for lp, _ in offline[t]])
            for _ in range(rounds):
                tokens, _ = search.decode_offline_batch(packed, [192] * len(offline[t]))
                wrong.extend(t for got, (_, want) in zip(tokens, offline[t]) if got != want)
                searchers = [WfstDecoderSearch(loop_fst, _loop_opts()) for _ in streams[t]]
                for start in range(0, 160, 16):
                    for s, (lp, _) in zip(searchers, streams[t]):
                        s.search(lp[start : start + 16])
                for s, (_, want) in zip(searchers, streams[t]):
                    s.finalize_search()
                    if s.outputs[0] != want:
                        wrong.append(t)
        except Exception as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(repr(exc))

    workers = [threading.Thread(target=run, args=(t,)) for t in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert not errors, errors[:3]
    assert not wrong, f"{len(wrong)} results belonged to another caller"


def test_scopes_do_not_share_decoders(loop_fst):
    """One decoder serializes its callers, so each engine gets its own; within an
    owner a decoder is reused, and without one the process-wide decoder is."""
    from oasr.decoder.wfst_decoder import (
        WfstDecoderScope,
        _get_offline_decoder,
        _get_streaming_decoder,
    )

    opts = _loop_opts()
    a, b = WfstDecoderScope(), WfstDecoderScope()
    for get in (_get_offline_decoder, _get_streaming_decoder):
        assert get(loop_fst, opts, 0, a) is get(loop_fst, opts, 0, a)
        assert get(loop_fst, opts, 0, a).handle != get(loop_fst, opts, 0, b).handle
        assert get(loop_fst, opts, 0) is get(loop_fst, opts, 0)
        assert get(loop_fst, opts, 0).handle not in {
            get(loop_fst, opts, 0, a).handle,
            get(loop_fst, opts, 0, b).handle,
        }


def test_a_closed_scope_keeps_a_decoder_until_its_last_stream_lets_go(loop_fst):
    """An engine shutting down closes its scope while a stream may still hold a
    channel.  Freeing the decoder then would leave that stream calling into freed
    memory; it is freed only once the channel is released."""
    import gc
    import weakref

    from oasr.decoder.wfst_decoder import WfstDecoderScope, WfstDecoderSearch

    lp, want = _loop_case(7, 96)
    scope = WfstDecoderScope()
    search = WfstDecoderSearch(loop_fst, _loop_opts(), scope=scope)
    search.search(lp[:48])
    decoder = weakref.ref(search._stream_dec)

    scope.close()
    gc.collect()
    assert decoder() is not None, "the scope freed a decoder a stream was still using"
    search.search(lp[48:])
    search.finalize_search()
    assert search.outputs[0] == want
    gc.collect()
    assert decoder() is None, "the decoder outlived its scope and its last stream"

    # A new owner on the same graph starts from scratch — the graph's shared device
    # copy is uploaded again if nothing else held it.
    fresh = WfstDecoderSearch(loop_fst, _loop_opts(), scope=WfstDecoderScope())
    fresh.search(lp)
    fresh.finalize_search()
    assert fresh.outputs[0] == want


def test_toy_engines_each_get_their_own_stream_channels(toy_fst):
    """Engine workers have identical configs, so on the process-wide decoders they
    shared one streaming decoder sized for *one* engine's stream pool: with two
    workers, the stream past ``max_streams`` found no free channel."""
    from oasr.engine.request import Request

    speech = _toy_logp([1, 2], "cuda").unsqueeze(0)
    workers = [_toy_strategy(toy_fst, wfst_max_streams=2) for _ in range(2)]
    live = []
    for w, strategy in enumerate(workers):
        for i in range(2):  # a full pool on each engine
            req = Request(request_id=f"w{w}-{i}", streaming=True)
            strategy.create_session(req)
            assert _best(strategy.decode_streaming_chunk(req, speech)) == _TOY_WORDS
            live.append((strategy, req))
    for strategy, req in live:
        assert _best(strategy.finalize(req)) == _TOY_WORDS
        strategy.free_session(req)


def test_toy_strategy_shutdown_releases_its_decoders(toy_fst):
    """``ASREngine.shutdown`` reaches the strategy, which drops its decoders: per
    engine they are gigabytes (the workspace plus, for the last owner of a graph, its
    device copy), and leaving them to garbage collection stranded them in every
    process that built more than one engine."""
    import gc
    import weakref

    strategy = _toy_strategy(toy_fst)
    log_probs = torch.stack([_toy_logp([1, 2, 0], "cuda")])
    strategy.decode_offline(log_probs, torch.tensor([3], dtype=torch.int32, device="cuda"))
    held = [weakref.ref(d) for d in strategy._wfst_scope._offline.values()]
    assert held

    strategy.shutdown()
    gc.collect()
    assert all(ref() is None for ref in held)


def test_toy_winners_gc_matches_plain(toy_fst):
    """gc_interval > 0 (segmented decode + winners-log GC + host prefix merge) must
    reproduce the plain decode exactly, including across repeats on the same decoder."""
    _module()
    from oasr.decoder.wfst_decoder import WfstDecoderOptions, WfstDecoderSearch

    labels = ([1] + [0] * 29 + [2] + [0] * 29) * 1  # T=60 >> gc_interval
    logp = _toy_logp(labels, "cuda")
    plain = WfstDecoderSearch(
        toy_fst,
        WfstDecoderOptions(
            min_active_states=1,
            max_active_states=100,
            blank_skip_thresh=1.0,
            max_frames=64,
            max_offline_lanes=2,
        ),
    )
    gc = WfstDecoderSearch(
        toy_fst,
        WfstDecoderOptions(
            min_active_states=1,
            max_active_states=100,
            blank_skip_thresh=1.0,
            max_frames=64,
            max_offline_lanes=2,
            gc_interval=8,
        ),
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

    mod = _module()
    graph = _get_graph(toy_fst)
    labels = [1] + [0] * 40 + [2] + [0] * 18  # T=60 >> chunk+2
    logp = _toy_logp(labels, "cuda")
    chunk = 8

    def run(gc):
        dec = int(
            mod.wfst_create_decoder(
                graph, 20.0, 8.0, 1, 100, 1, 1, chunk, 0, 16, 4, 1, 0, 0, 1, 0, 3, 0, 0, gc
            )
        )
        ch = int(mod.wfst_create_stream(dec))
        assert ch >= 0
        for s in range(0, logp.size(0), chunk):
            piece = logp[s : s + chunk].unsqueeze(0).contiguous()
            outs = [
                torch.empty((1, 8), dtype=torch.int32),
                torch.empty((1,), dtype=torch.int32),
                torch.empty((1,), dtype=torch.int32),
                torch.empty((1,), dtype=torch.int32),
            ]
            mod.wfst_advance_chunk(
                dec,
                torch.tensor([ch], dtype=torch.int32),
                piece,
                torch.tensor([piece.size(1)], dtype=torch.int32),
                0,
                *outs,
            )
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
    assert w_legacy == w_gc[len(w_gc) - len(w_legacy) :]  # legacy = truncated suffix
    assert len(w_legacy) < len(w_gc)


# ---------------------------------------------------------------------------
# Real-graph smoke tests via the public Decoder API (need OASR_TEST_FST)
# ---------------------------------------------------------------------------

# Declared in tests/assets.py so --strict-assets can make it fatal.
FST = assets.declared("OASR_TEST_FST")


@pytest.fixture(scope="module")
def wfst_decoder_cls():
    _module()  # CUDA + JIT available
    assets.require("OASR_TEST_FST")
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
