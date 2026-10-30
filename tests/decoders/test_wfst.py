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
# Fan graph: label 1 sends the start state S into N parallel states that loop on
# label 1 and never meet, until label 2 brings them all back to S; from S, labels
# 3 and 4 emit words 300 and 400.  A run of label 1 is a "noise burst" whose
# frontier is exactly N tokens wide and whose chains never converge, so it can
# overflow any capacity on demand; the words after it show whether the decode
# survived.
# ---------------------------------------------------------------------------

_FAN_VOCAB = 5
_FAN_SPEECH = [2, 0, 3, 0, 4, 0, 3, 0]
_FAN_WORDS = [300, 400, 300]


def _write_fan_graph(path: str, n: int) -> None:
    from oasr.decoder.wfst.graph_image import build_image, write_image

    final = n + 1
    arcs = [(0, 0, 0, 0)] + [(0, k, 1, 0) for k in range(1, n + 1)]
    arcs += [(0, 0, 3, 300), (0, 0, 4, 400), (0, final, -1, 0)]
    for k in range(1, n + 1):
        arcs += [(k, k, 0, 0), (k, k, 1, 0), (k, 0, 2, 0)]
    row_splits = np.zeros(n + 3, dtype=np.int64)
    for src, *_ in arcs:
        row_splits[src + 1] += 1
    aux_row_splits, pool = [0], []
    for *_, word in arcs:
        if word:
            pool.append(word)
        aux_row_splits.append(len(pool))
    img = build_image(
        np.cumsum(row_splits).astype(np.int32),
        np.array([a[1] for a in arcs], dtype=np.int32),
        np.array([a[2] for a in arcs], dtype=np.int32),
        np.zeros(len(arcs), dtype=np.float32),
        aux_row_splits=np.array(aux_row_splits, dtype=np.int32),
        aux_pool=np.array(pool, dtype=np.int32),
        vocab_size=_FAN_VOCAB,
    )
    write_image(img, path)


def _fan_logp(labels) -> torch.Tensor:
    lp = torch.full((len(labels), _FAN_VOCAB), -10.0)
    lp[torch.arange(len(labels)), torch.tensor(labels)] = 0.0
    return lp.cuda()


def _fan_opts(**kw):
    return _loop_opts(**{"max_frames": 512, "max_offline_lanes": 2, **kw})


@pytest.fixture(scope="module")
def fan_fst(tmp_path_factory):
    """2048 parallel burst states: over an offline frontier of ``32 * 48``."""
    _module()
    path = str(tmp_path_factory.mktemp("wfst") / "fan.img")
    _write_fan_graph(path, 2048)
    return path


@pytest.fixture(scope="module")
def wide_fan_fst(tmp_path_factory):
    """8192 parallel burst states: a streaming ring of one mapping chunk (4M entries
    at 32 MiB) fills in about 512 frames."""
    _module()
    path = str(tmp_path_factory.mktemp("wfst") / "wide_fan.img")
    _write_fan_graph(path, 8192)
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


def _toy_strategy(toy_fst, vocab=_TOY_VOCAB, **decoder):
    """The engine's ``ctc_wfst`` strategy over a toy graph, built as the engine builds it."""
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
        _model_config=SimpleNamespace(vocab_size=vocab),
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


# ---------------------------------------------------------------------------
# Overflow: rescue, recovery and reporting (audit #13)
# ---------------------------------------------------------------------------


def test_an_overflowed_lane_is_decoded_again_with_room_to_spare(fan_fst):
    """Offline, a lane whose frontier outgrew ``main_q`` lost tokens in claim order,
    not score order, and was returned as if complete.  Which tokens is not
    deterministic, so neither is the damage: at this capacity 7 of 50 first attempts
    dropped or *inserted* words, at a third of it 33 of 50.  (On real log-probs it is
    rarer — 15 of 2000 LJSpeech utterances overflow, none changed — which is why it
    stayed invisible.)  What is deterministic is the flag, and that the rescue, with
    room for the whole frontier, is exact."""
    from oasr.decoder.wfst_decoder import (
        OVERFLOW_KEPT,
        WfstDecoderSearch,
        _get_module,
        _get_offline_decoder,
        _run_decode_batch,
    )

    opts = _fan_opts(max_active_states=48)  # main_q 1536 < 2048; the rescue has 3072
    search = WfstDecoderSearch(fan_fst, opts)
    burst = _fan_logp([1] * 10 + _FAN_SPEECH)

    # What production returned until now: an attempt the decoder itself flags.
    first = _get_offline_decoder(fan_fst, opts, 0).handle
    _, _, _, bits = _run_decode_batch(_get_module(), first, burst.unsqueeze(0), [18])
    assert bits[0] & OVERFLOW_KEPT, "nothing overflowed, so nothing was under test"

    tokens, _ = search.decode_offline(burst)
    assert tokens[0] == _FAN_WORDS and search.overflow == 0

    # Batched, beside a row that fits: only the flagged row is decoded again.
    quiet = _fan_logp([0] * 11 + _FAN_SPEECH[1:])
    tokens, _ = search.decode_offline_batch(torch.stack([burst, quiet]), [18, 18])
    assert tokens == [_FAN_WORDS, _FAN_WORDS]
    assert search.batch_overflow == [0, 0]


def test_what_the_rescue_cannot_absorb_is_reported(fan_fst):
    """The rescue doubles every per-lane table; a lane that needs more still comes
    back — it is the best the search found — but flagged."""
    from oasr.decoder.wfst_decoder import OVERFLOW_KEPT, WfstDecoderSearch

    search = WfstDecoderSearch(fan_fst, _fan_opts(max_active_states=24))  # 768, then 1536
    search.decode_offline(_fan_logp([1] * 10 + _FAN_SPEECH))
    assert search.overflow & OVERFLOW_KEPT


def test_a_stream_outlives_a_ring_overrun(wide_fan_fst):
    """Streaming, a burst whose chains never converge fills the channel's winners
    ring.  The decoder drops that round and its collector skips the lane from then
    on, so the ring never frees again: everything after the burst was lost — the
    final here was ``[]``.  The searcher now carries on on a fresh channel."""
    from oasr.decoder.wfst_decoder import OVERFLOW_ARENA, WfstDecoderSearch

    opts = _fan_opts(max_active_states=8192, max_streams=2, stream_log_entries=1)
    lp = _fan_logp([1] * 1200 + _FAN_SPEECH)
    search = WfstDecoderSearch(wide_fan_fst, opts)
    for start in range(0, lp.size(0), 16):
        search.search(lp[start : start + 16])
    search.finalize_search()
    assert search.overflow & OVERFLOW_ARENA, "the ring never overran, so nothing was under test"
    assert search.outputs[0] == _FAN_WORDS


def test_toy_strategy_puts_what_it_could_not_recover_on_the_output(fan_fst):
    """``RequestOutput.search_overflow`` is the client's only sign that a transcript
    may be short; a lane the rescue fixed must not set it."""
    from oasr.decoder.wfst_decoder import OVERFLOW_KEPT

    burst = torch.stack([_fan_logp([1] * 10 + _FAN_SPEECH)])
    lengths = torch.tensor([18], dtype=torch.int32, device="cuda")

    rescued = _toy_strategy(fan_fst, vocab=_FAN_VOCAB, wfst_max_active_states=48)
    (out,) = rescued.decode_offline(burst, lengths)
    assert out.tokens[0] == _FAN_WORDS and out.search_overflow is None

    short = _toy_strategy(fan_fst, vocab=_FAN_VOCAB, wfst_max_active_states=24)
    (out,) = short.decode_offline(burst, lengths)
    assert out.search_overflow & OVERFLOW_KEPT


def test_toy_strategy_keeps_a_turns_overflow_on_the_streams_final(wide_fan_fst):
    """A segment-mode turn boundary restarts the search, clearing its bits; the
    stream's final must still say an earlier turn overran."""
    from oasr.decoder.wfst_decoder import OVERFLOW_ARENA
    from oasr.engine.request import Request

    strategy = _toy_strategy(
        wide_fan_fst,
        vocab=_FAN_VOCAB,
        wfst_max_active_states=8192,
        wfst_max_streams=2,
        wfst_stream_log_entries=1,
    )
    req = Request(request_id="turns", streaming=True)
    strategy.create_session(req)
    burst = _fan_logp([1] * 1200 + [2, 0])
    for start in range(0, burst.size(0), 16):
        strategy.decode_streaming_chunk(req, burst[start : start + 16].unsqueeze(0))
    turn = strategy.finalize(req)  # StreamingExecutor._close_turn, decode side
    strategy.free_session(req)
    strategy.create_session(req)
    assert turn.search_overflow & OVERFLOW_ARENA

    speech = _fan_logp(_FAN_SPEECH[1:]).unsqueeze(0)
    partial = strategy.decode_streaming_chunk(req, speech)
    final = strategy.finalize(req)
    assert final.tokens[0] == _FAN_WORDS
    assert partial.search_overflow & OVERFLOW_ARENA and final.search_overflow & OVERFLOW_ARENA
    strategy.free_session(req)


# ---------------------------------------------------------------------------
# Capacity limits that bound without an error (audit M15, M16, M17, L6)
# ---------------------------------------------------------------------------

#: Epsilon graph: state 0 emits word 100 + k on label k into state 1, and an epsilon
#: arc returns to 0, so every word frame puts two arcs on the best path -- what a TLG's
#: word boundaries do.  Label 5 is the epsilon id; log-prob rows keep a column for it.
_EPS_VOCAB = 6


def _write_eps_graph(path: str) -> None:
    from oasr.decoder.wfst.graph_image import build_image, write_image

    arcs = [(0, 0, 0, 0)] + [(0, 1, k, 100 + k) for k in range(1, 5)] + [(0, 2, -1, 0)]
    arcs += [(1, 0, 5, 0)]
    row_splits = np.zeros(4, dtype=np.int64)
    for src, *_ in arcs:
        row_splits[src + 1] += 1
    aux_row_splits, pool = [0], []
    for *_, word in arcs:
        if word:
            pool.append(word)
        aux_row_splits.append(len(pool))
    img = build_image(
        np.cumsum(row_splits).astype(np.int32),
        np.array([a[1] for a in arcs], dtype=np.int32),
        np.array([a[2] for a in arcs], dtype=np.int32),
        np.zeros(len(arcs), dtype=np.float32),
        aux_row_splits=np.array(aux_row_splits, dtype=np.int32),
        aux_pool=np.array(pool, dtype=np.int32),
        vocab_size=_EPS_VOCAB,
        epsilon_id=5,
    )
    write_image(img, path)


def _eps_case(seed: int, frames: int):
    """``([T, V] log-probs, the words they decode to)``: every frame a word."""
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, 5, (frames,), generator=gen).tolist()
    lp = torch.full((frames, _EPS_VOCAB), -10.0)
    lp[torch.arange(frames), torch.tensor(labels)] = 0.0
    return lp, [100 + k for k in labels]


@pytest.fixture(scope="module")
def eps_fst(tmp_path_factory):
    _module()
    path = str(tmp_path_factory.mktemp("wfst") / "eps.img")
    _write_eps_graph(path)
    return path


def test_an_epsilon_heavy_path_keeps_its_first_words(eps_fst):
    """The best-path buffer held ``max_frames + 2`` arcs, but on an epsilon graph a
    path runs up to one arc per closure pass on top of each frame's.  The backtrack
    walks newest to oldest, so the first words were the ones dropped, silently: here
    the first 32 of 64."""
    from oasr.decoder.wfst_decoder import WfstDecoderSearch

    lp, words = _eps_case(0, 64)
    search = WfstDecoderSearch(eps_fst, _loop_opts(max_frames=64))
    tokens, _ = search.decode_offline(lp.cuda())
    assert tokens[0] == words
    assert search.overflow == 0
    assert _cpu_decode(eps_fst, lp)[0] == words


def test_a_lane_whose_path_outgrows_its_buffer_is_flagged(toy_fst):
    """Where the buffer still binds -- the gc-off streaming decoder sizes it for one
    chunk -- the cut is now reported rather than returned as a whole transcript."""
    from oasr.decoder.wfst_decoder import OVERFLOW_PATH, _get_graph

    mod = _module()
    graph = _get_graph(toy_fst)
    logp = _toy_logp([1] + [0] * 40 + [2] + [0] * 18, "cuda")

    def final_bits(gc):
        dec = int(
            mod.wfst_create_decoder(
                graph, 20.0, 8.0, 1, 100, 1, 1, 8, 0, 16, 4, 1, 0, 0, 1, 0, 3, 0, 0, gc
            )
        )
        ch = int(mod.wfst_create_stream(dec))
        for s in range(0, logp.size(0), 8):
            piece = logp[s : s + 8].unsqueeze(0).contiguous()
            outs = [torch.empty((1, 8), dtype=torch.int32)]
            outs += [torch.empty((1,), dtype=torch.int32) for _ in range(3)]
            lens = torch.tensor([piece.size(1)], dtype=torch.int32)
            mod.wfst_advance_chunk(
                dec, torch.tensor([ch], dtype=torch.int32), piece, lens, 0, *outs
            )
        meta = torch.empty((3,), dtype=torch.int32)
        mod.wfst_finalize_stream(
            dec,
            ch,
            torch.empty((64,), dtype=torch.int32),
            torch.empty((1,), dtype=torch.int32),
            torch.empty((1,), dtype=torch.float64),
            meta,
        )
        mod.wfst_free_decoder(dec)
        return int(meta[2])

    assert final_bits(gc=0) & OVERFLOW_PATH
    assert final_bits(gc=2) == 0


@pytest.mark.parametrize("gc_interval", [0, 8], ids=["whole", "gc"])
def test_an_utterance_longer_than_max_frames_decodes(loop_fst, gc_interval):
    """``max_frames`` was a hard limit that nothing declared: a longer utterance
    raised ``T > max_frames`` from the decoder, after its whole encoder forward.  It is
    now the starting capacity, grown to fit, and a batch beyond the whole-batch graphs
    runs in segments -- with the same result as a decoder built large enough."""
    from oasr.decoder.wfst_decoder import WfstDecoderScope, WfstDecoderSearch

    long_lp, long_words = _loop_case(1, 700)
    short_lp, short_words = _loop_case(2, 50)
    grown = WfstDecoderSearch(
        loop_fst, _loop_opts(max_frames=256, gc_interval=gc_interval), WfstDecoderScope()
    )
    roomy = WfstDecoderSearch(
        loop_fst, _loop_opts(max_frames=1024, gc_interval=gc_interval), WfstDecoderScope()
    )
    tokens, scores = grown.decode_offline(long_lp)
    assert tokens[0] == long_words and grown.overflow == 0
    assert scores[0] == pytest.approx(roomy.decode_offline(long_lp)[1][0], abs=1e-4)

    # Batched with a short row, and again after the growth dropped the old graphs.
    packed = torch.full((2, 700, _LOOP_VOCAB), -30.0, device="cuda")
    packed[0], packed[1, :50] = long_lp, short_lp
    for _ in range(2):
        tokens, _ = grown.decode_offline_batch(packed, [700, 50])
        assert tokens == [long_words, short_words]
        assert grown.decode_offline(short_lp)[0][0] == short_words


def _fan_lap_stream(fst, n, ring_entries, blocks):
    """Stream ``blocks`` of ``[1, 2, word]`` through an ``n``-state fan graph in
    128-frame chunks; returns (words, the words it must decode to, overflow bits).
    Label 1 fans out to every burst state and label 2 converges them again, so the
    collector advances every chunk while the ring takes up to ``n + 1`` entries a
    frame.  The words are random and dense: a walk that reads the wrong ring slot
    follows a stale chain only a few frames before an old cut stops it, and it shows
    only if those frames held a word."""
    from oasr.decoder.wfst_decoder import WfstDecoderScope, WfstDecoderSearch

    gen = torch.Generator().manual_seed(n)
    words = torch.randint(3, 5, (blocks,), generator=gen).tolist()
    labels = [lab for w in words for lab in (1, 2, w)]
    opts = _fan_opts(max_active_states=n, max_streams=1, stream_log_entries=ring_entries)
    search = WfstDecoderSearch(fst, opts, WfstDecoderScope())
    lp = _fan_logp(labels + _FAN_SPEECH[1:])
    for start in range(0, lp.size(0), 128):
        search.search(lp[start : start + 128])
    search.finalize_search()
    return search.outputs[0], [100 * w for w in words] + _FAN_WORDS, search.overflow


def test_streaming_ids_rebased_every_ring_lap_keep_the_transcript(wide_fan_fst):
    """Streaming winner ids only grow, so past 2^31 appends they went negative.  They
    are now moved down by whole ring laps each time the collector passes one.  A
    stream that laps the smallest ring (one 4Mi-entry mapping chunk) three times keeps
    every word, and there is one every three frames."""
    words, expected, bits = _fan_lap_stream(wide_fan_fst, 8192, 1, 1000)
    assert words == expected and bits == 0


_PAST_INT32_SCRIPT = r"""
import json, os, sys
sys.path[:0] = [sys.argv[2], os.path.dirname(sys.argv[2])]  # tests/decoders, tests/
from test_wfst import _fan_lap_stream, _write_fan_graph

_write_fan_graph(sys.argv[1], 65536)
# ~117k entries a block (measured): 2^31 of them in ~18.3k blocks, 22k for a margin.
# A 16Mi-entry ring holds two 128-frame chunks of them.
words, expected, bits = _fan_lap_stream(sys.argv[1], 65536, 16 << 20, 22000)
print(json.dumps({"complete": words == expected, "overflow": bits}))
"""


@pytest.mark.slow
def test_a_stream_lives_past_two_to_the_31_winner_entries(tmp_path):
    """How long the ids lasted depends on the frontier: here 2^31 entries take ~55k
    frames and ~130 ring laps (~25 s).  Before the rebase the next id wrapped negative and indexed the
    ring out of bounds.  In a subprocess: a fault there is sticky for the process."""
    import json
    import os
    import subprocess
    import sys

    _module()
    script = tmp_path / "run.py"
    script.write_text(_PAST_INT32_SCRIPT)
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "fan64k.img"), tests_dir],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"complete": True, "overflow": 0}


#: Deep-epsilon graph: label 1 leads to state 1, then an epsilon chain 1 -> 2 -> 3 -> 4
#: -> 5.  Three closure passes a frame reach state 4, whose label-2 arc emits word 40;
#: only a fourth pass reaches state 5, whose label-2 arc (word 50) scores higher.
_DEEP_VOCAB = 4


def _write_deep_eps_graph(path: str) -> None:
    from oasr.decoder.wfst.graph_image import build_image, write_image

    arcs = [(0, 0, 0, 0.0, 0), (0, 1, 1, 0.0, 10)]
    arcs += [(s, s + 1, 3, 0.0, 0) for s in range(1, 4)]
    arcs += [(4, 6, 2, 0.0, 40), (4, 5, 3, 0.0, 0), (5, 6, 2, 1.0, 50)]
    arcs += [(6, 6, 0, 0.0, 0), (6, 7, -1, 0.0, 0)]
    row_splits = np.zeros(9, dtype=np.int64)
    for src, *_ in arcs:
        row_splits[src + 1] += 1
    aux_row_splits, pool = [0], []
    for *_, word in arcs:
        if word:
            pool.append(word)
        aux_row_splits.append(len(pool))
    img = build_image(
        np.cumsum(row_splits).astype(np.int32),
        np.array([a[1] for a in arcs], dtype=np.int32),
        np.array([a[2] for a in arcs], dtype=np.int32),
        np.array([a[3] for a in arcs], dtype=np.float32),
        aux_row_splits=np.array(aux_row_splits, dtype=np.int32),
        aux_pool=np.array(pool, dtype=np.int32),
        vocab_size=_DEEP_VOCAB,
        epsilon_id=3,
    )
    write_image(img, path)


def _deep_logp(labels) -> torch.Tensor:
    lp = torch.full((len(labels), _DEEP_VOCAB), -10.0)
    lp[torch.arange(len(labels)), torch.tensor(labels)] = 0.0
    return lp


def test_opening_a_stream_leaves_the_others_alone(tmp_path):
    """Opening a channel runs the epsilon closure of its start frontier, and that
    launch took every lane still in a real or epsilon phase -- which a live stream is
    after a chunk that ended on a real frame (8 frames here; 128 in the wrapper).  It
    got three more closure passes, reached state 5, and decoded ``[10, 50]`` instead of
    ``[10, 40]``: its transcript depended on when another stream opened."""
    from oasr.decoder.wfst_decoder import WfstDecoderScope, WfstDecoderSearch

    _module()
    fst = str(tmp_path / "deep.img")
    _write_deep_eps_graph(fst)
    first, second = _deep_logp([0] * 7 + [1]), _deep_logp([2] + [0] * 7)
    opts = _loop_opts(max_streams=2)
    assert _cpu_decode(fst, torch.cat([first, second]))[0] == [10, 40]

    def run(open_another: bool):
        scope = WfstDecoderScope()
        live = WfstDecoderSearch(fst, opts, scope)
        live.search(first.cuda())
        other = WfstDecoderSearch(fst, opts, scope)
        if open_another:
            other.search(second.cuda())
        live.search(second.cuda())
        live.finalize_search()
        other.reset()
        return live.outputs[0]

    assert run(open_another=False) == [10, 40]
    assert run(open_another=True) == [10, 40]


def _patch_image(src: str, dst: str, patch) -> None:
    """Copy a graph image, letting ``patch(header, arcs)`` edit it (both writable)."""
    data = bytearray(open(src, "rb").read())
    header = np.frombuffer(data, dtype=np.int64, count=1, offset=48 + 8 * 2)  # offsets[2]
    arcs_off = int(header[0])
    meta = np.frombuffer(data, dtype=np.int32, count=2, offset=32)  # vocab, start
    num_arcs = int(np.frombuffer(data, dtype=np.int64, count=1, offset=24)[0])
    arcs = np.frombuffer(data, dtype=np.int32, count=2 * num_arcs, offset=arcs_off)
    patch(meta, arcs)
    open(dst, "wb").write(bytes(data))


@pytest.mark.parametrize(
    "what, patch",
    [
        ("start state", lambda meta, arcs: meta.__setitem__(1, 99)),
        ("destination", lambda meta, arcs: arcs.__setitem__(0, 99)),
        ("label", lambda meta, arcs: arcs.__setitem__(3, 7)),
    ],
)
def test_a_corrupt_graph_image_is_refused_at_load(toy_fst, tmp_path, what, patch):
    """The loader checked the counts and the two ends of ``row_splits``; a state, arc
    destination or label out of range loaded, and the decoder read past its arrays."""
    mod = _module()
    bad = str(tmp_path / "bad.img")
    _patch_image(toy_fst, bad, patch)
    with pytest.raises(RuntimeError, match="corrupt graph image"):
        mod.wfst_load_graph(bad)


def test_the_launchers_refuse_buffers_they_would_overrun(toy_fst):
    """Each of these was read or written out of bounds: a length past the tensor, a
    log-prob tensor narrower than the graph's labels or in the other dtype, and
    output buffers with fewer rows than the batch."""
    from oasr.decoder.wfst_decoder import (
        WfstDecoderOptions,
        _get_module,
        _get_offline_decoder,
        _run_decode_batch,
    )

    mod = _get_module()
    dec = _get_offline_decoder(toy_fst, WfstDecoderOptions(max_frames=64), 0).handle
    lp = _toy_logp([1, 0, 2], "cuda").unsqueeze(0).contiguous()
    with pytest.raises(RuntimeError, match="outside"):
        _run_decode_batch(mod, dec, lp, [4])
    with pytest.raises(RuntimeError, match="columns"):
        _run_decode_batch(mod, dec, lp[..., :2].contiguous(), [3])
    with pytest.raises(RuntimeError, match="float32"):
        _run_decode_batch(mod, dec, lp.half(), [3])
    with pytest.raises(RuntimeError, match="rows"):
        mod.wfst_decode_batch(
            dec,
            torch.cat([lp, lp]),
            torch.tensor([3, 3], dtype=torch.int32),
            torch.empty((1, 8), dtype=torch.int32),
            torch.empty((2,), dtype=torch.int32),
            torch.empty((2,), dtype=torch.float64),
            torch.empty((2, 3), dtype=torch.int32),
        )
    words, _, oks, _ = _run_decode_batch(mod, dec, lp, [3])
    assert oks == [True] and words == [_TOY_WORDS]


_LEAK_SCRIPT = r"""
import sys
import torch
from oasr.decoder.wfst_decoder import _get_graph, _get_module

mod, graph = _get_module(), _get_graph(sys.argv[1])
torch.cuda.init()
free0 = torch.cuda.mem_get_info()[0]
# 2^28-token frontiers: nine 1 GiB arrays per lane before the candidate buffers, and
# enough lanes that the whole workspace cannot fit, so construction fails part-way.
lanes = int(free0 // (2**28 * 4 * 20)) + 1
try:
    mod.wfst_create_decoder(graph, 20.0, 8.0, 1, 2**23, 1, lanes, 64, 0, 32, 2, 1,
                            0, 0, 0, 0, 3, 0, 0, 0)
    print("constructed")
except Exception as exc:
    print("failed", str(exc).splitlines()[0][:200])
print(free0 - torch.cuda.mem_get_info()[0])
"""


def test_a_decoder_that_fails_to_construct_frees_what_it_allocated(toy_fst, tmp_path):
    """The constructor allocates as it goes, and a C++ object whose constructor throws
    is never destroyed: everything allocated before an out-of-memory stayed allocated
    for the life of the process.  In a subprocess, where nothing else holds memory."""
    import subprocess
    import sys

    _module()
    script = tmp_path / "leak.py"
    script.write_text(_LEAK_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script), toy_fst], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    status, held = proc.stdout.strip().splitlines()[-2:]
    assert status.startswith("failed"), "construction did not fail, so nothing was under test"
    assert int(held) < 64 << 20, f"{int(held) >> 20} MiB still held after the failure"
