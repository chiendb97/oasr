# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The backend contract, and the network clients against in-process fakes."""

from __future__ import annotations

import asyncio
import io
import json
import wave

import numpy as np
import pytest

from benchmarks.asrbench.backends import (
    BACKENDS,
    Capabilities,
    RequestFailed,
    Unsupported,
    backend_class,
    create,
)
from benchmarks.asrbench.backends.base import STREAMING_MODES, Backend
from benchmarks.asrbench.workloads import WorkItem


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_registry_entries_honour_the_contract(name):
    cls = backend_class(name)
    assert issubclass(cls, Backend) and cls.name == name and cls.description
    assert isinstance(cls.OPTIONS, dict) and cls.OPTIONS
    ok, why = cls.available()
    assert isinstance(ok, bool) and (ok or why)
    for mode in ("offline", "streaming"):
        reason = cls.mode_unsupported(mode, {})
        assert reason is None or isinstance(reason, str)
    with pytest.raises(ValueError, match="unknown option"):
        cls({"definitely_not_an_option": 1})


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        backend_class("nope")


def test_mock_contract():
    async def go():
        b = create("mock", {"base_ms": 1})
        await b.load()
        try:
            caps = b.capabilities()
            assert isinstance(caps, Capabilities) and caps.streaming in STREAMING_MODES
            item = WorkItem(uid="x", source_id="x", audio_path="", duration_s=1.0, text="a b c")
            p = b.prepare(np.zeros(16000, np.float32), item)
            tr = await b.transcribe(p)
            assert tr.text == "a b c" and tr.t_done >= tr.t_submit
            outs = await b.transcribe_batch([p, p])
            assert len(outs) == 2 and outs[0].t_done == outs[1].t_done
            s = await b.open_stream(word_timestamps=True)
            for k in range(3):
                await s.send(b.prepare_chunk(np.zeros(8000, np.float32)), is_last=k == 2)
            final = await s.wait_final(2.0)
            assert final.kind == "final" and final.words
            assert [e.kind for e in s.events][-1] == "final"
            assert b.info().to_json()["model"]["simulated"] is True
        finally:
            await b.close()

    asyncio.run(go())


def test_mock_without_streaming_refuses():
    async def go():
        b = create("mock", {"streaming": "none"})
        await b.load()
        assert b.capabilities().streaming is None
        with pytest.raises(Unsupported):
            await b.open_stream()
        await b.close()

    asyncio.run(go())


def test_parse_prometheus_sums_series():
    from benchmarks.asrbench.backends.openai_http import parse_prometheus

    text = (
        "# HELP x\n"
        'oasr_engine_queue_wait_seconds_sum{mode="offline",engine="0"} 1.5\n'
        'oasr_engine_queue_wait_seconds_sum{mode="streaming",engine="0"} 0.5\n'
        "oasr_engine_queue_wait_seconds_count 4\n"
        "garbage line\n"
    )
    m = parse_prometheus(text)
    assert m["oasr_engine_queue_wait_seconds_sum"] == 2.0
    assert m["oasr_engine_queue_wait_seconds_count"] == 4.0


def test_openai_http_maps_responses_and_encodes_wav():
    httpx = pytest.importorskip("httpx")
    from benchmarks.asrbench.backends.openai_http import OpenAIHttpBackend

    seen = []

    def handler(request):
        seen.append(request)
        n = len(seen)
        if n == 1:
            return httpx.Response(200, json={"text": "hello there"})
        if n == 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(500, text="kaput")

    async def go():
        b = OpenAIHttpBackend({"base_url": "http://fake", "model": "m1"})
        await b.load()
        await b._client.aclose()
        b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = WorkItem(uid="x", source_id="x", audio_path="", duration_s=0.5)
        body = b.prepare(np.zeros(8000, np.float32), item)
        with wave.open(io.BytesIO(body)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (
                16000,
                1,
                2,
                8000,
            )
        tr = await b.transcribe(body)
        assert tr.text == "hello there"
        with pytest.raises(RequestFailed) as rej:
            await b.transcribe(body)
        assert rej.value.kind == "rejected"
        with pytest.raises(RequestFailed) as fail:
            await b.transcribe(body)
        assert fail.value.kind == "failed" and fail.value.status == 500
        await b.close()

    asyncio.run(go())
    assert seen[0].url.path == "/v1/audio/transcriptions"
    assert b'name="model"' in seen[0].content and b"m1" in seen[0].content
    assert backend_class("openai_http").mode_unsupported("streaming", {}) is not None
    assert backend_class("openai_http").mode_unsupported("streaming", {"realtime": True}) is None


def test_realtime_websocket_session():
    websockets = pytest.importorskip("websockets")
    pytest.importorskip("httpx")
    from benchmarks.asrbench.backends.openai_http import OpenAIHttpBackend

    received = {"bytes": 0, "update": None}

    async def server(ws):
        received["update"] = json.loads(await ws.recv())
        async for msg in ws:
            if isinstance(msg, bytes):
                received["bytes"] += len(msg)
                await ws.send(json.dumps({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "a", "text": "a" * (received["bytes"] // 3200),
                }))  # fmt: skip
            elif json.loads(msg).get("type") == "input_audio_buffer.commit":
                await ws.send(json.dumps({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "done",
                }))  # fmt: skip
                return

    async def go():
        async with websockets.serve(server, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            b = OpenAIHttpBackend({"base_url": f"http://127.0.0.1:{port}", "realtime": True})
            await b.load()
            s = await b.open_stream()
            for k in range(3):
                await s.send(b.prepare_chunk(np.zeros(1600, np.float32)), is_last=k == 2)
            final = await s.wait_final(5.0)
            await b.close()
            return s, final

    s, final = asyncio.run(go())
    assert final.text == "done"
    assert received["bytes"] == 3 * 3200  # PCM16 binary frames
    assert received["update"]["type"] == "session.update"
    assert received["update"]["session"]["encoding"] == "LINEAR16"
    kinds = [e.kind for e in s.events]
    assert kinds.count("partial") == 3 and kinds[-1] == "final"
    assert [e.t for e in s.events] == sorted(e.t for e in s.events)


def test_oasr_server_attach_takes_the_servers_mode_not_the_requested_one(monkeypatch):
    # One running server is one --service-mode, and a run attaches both of its
    # instances to it.  Labelling by the requested mode would call a buffered
    # stream "native", and turn an offline kind into failures, not "unsupported".
    websockets = pytest.importorskip("websockets")
    pytest.importorskip("httpx")
    from benchmarks.asrbench.backends.openai_http import OasrServerBackend

    handshakes = []

    async def server(ws):
        handshakes.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({
            "type": "transcription_session.created",
            "session": {"model": "m", "mode": "offline"},
        }))  # fmt: skip
        await ws.wait_closed()

    async def attach(mode, url):
        b = OasrServerBackend({"base_url": url}, mode)
        await b.load()
        await b.close()
        return b

    async def go():
        # /v1/models names the mode: no session is opened.
        served = {"id": "m", "service_mode": "streaming", "dtype": "torch.float16"}
        monkeypatch.setattr(OasrServerBackend, "_served_model", lambda self: dict(served))
        on_streaming = await attach("offline", "http://127.0.0.1:9")
        # It does not: the realtime handshake does.
        monkeypatch.setattr(OasrServerBackend, "_served_model", lambda self: {})
        async with websockets.serve(server, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            on_offline = await attach("streaming", f"http://127.0.0.1:{port}")
        return on_streaming, on_offline

    on_streaming, on_offline = asyncio.run(go())
    caps = on_streaming.capabilities()
    assert not caps.offline and caps.streaming == "native" and caps.partials
    with pytest.raises(Unsupported):
        asyncio.run(on_streaming.transcribe(b""))
    info = on_streaming.info().to_json()
    assert info["config"]["service_mode"] == "streaming" and info["dtype"] == "float16"
    assert info["model"]["id"] == "m" and info["config"]["spawned"] is False
    caps = on_offline.capabilities()
    assert caps.offline and caps.streaming == "buffered" and not caps.partials
    assert len(handshakes) == 1 and handshakes[0]["type"] == "session.update"
    assert on_offline.info().to_json()["config"]["service_mode"] == "offline"


def test_oasr_server_grpc_transport(monkeypatch):
    # Recognize and StreamingRecognize against an in-process Speech servicer:
    # what goes on the wire, how statuses classify, and the stream's events.
    grpc = pytest.importorskip("grpc")
    pytest.importorskip("grpc_tools")
    pytest.importorskip("httpx")
    pytest.importorskip("websockets")
    from benchmarks.asrbench.audio import encode_wav
    from benchmarks.asrbench.backends import grpc_speech
    from benchmarks.asrbench.backends.openai_http import OasrServerBackend

    pb, pb_grpc = grpc_speech.stubs()
    seen = {"unary": [], "stream_cfg": None, "stream_bytes": 0}

    class Speech(pb_grpc.SpeechServicer):
        async def Recognize(self, request, context):
            seen["unary"].append(request)
            n = len(seen["unary"])
            if n == 2:
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "busy")
            if n == 3:
                await context.abort(grpc.StatusCode.UNAVAILABLE, "worker lost")
            alt = pb.SpeechRecognitionAlternative(transcript="hello there")
            return pb.RecognizeResponse(results=[pb.SpeechRecognitionResult(alternatives=[alt])])

        async def StreamingRecognize(self, request_iterator, context):
            text = ""
            async for req in request_iterator:
                if req.HasField("streaming_config"):
                    seen["stream_cfg"] = req.streaming_config
                    continue
                seen["stream_bytes"] += len(req.audio_content)
                text += "a"
                alt = pb.SpeechRecognitionAlternative(transcript=text)
                yield pb.StreamingRecognizeResponse(
                    results=[pb.StreamingRecognitionResult(alternatives=[alt])]
                )
            alt = pb.SpeechRecognitionAlternative(transcript="done")
            yield pb.StreamingRecognizeResponse(
                results=[pb.StreamingRecognitionResult(alternatives=[alt], is_final=True)]
            )

    def backend(mode, port, **extra):
        opts = {"base_url": "http://127.0.0.1:9", "transport": "grpc"}
        opts.update(grpc_addr=f"127.0.0.1:{port}", **extra)
        return OasrServerBackend(opts, mode)

    async def go():
        server = grpc.aio.server()
        pb_grpc.add_SpeechServicer_to_server(Speech(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            monkeypatch.setattr(
                OasrServerBackend, "_served_model", lambda self: {"service_mode": "offline"}
            )
            b = backend("offline", port)
            await b.load()
            audio = np.zeros(8000, np.float32)
            item = WorkItem(uid="x", source_id="x", audio_path="", duration_s=0.5)
            req = b.prepare(audio, item)
            assert (await b.transcribe(req)).text == "hello there"
            with pytest.raises(RequestFailed) as rej:
                await b.transcribe(req)
            assert rej.value.kind == "rejected"
            with pytest.raises(RequestFailed) as fail:
                await b.transcribe(req)
            assert fail.value.kind == "failed" and "UNAVAILABLE" in str(fail.value)
            info = b.info().to_json()["config"]
            await b.close()
            pcm = backend("offline", port, grpc_encoding="linear16")
            await pcm.load()
            pcm_req = pcm.prepare(audio, item)
            await pcm.close()

            monkeypatch.setattr(
                OasrServerBackend, "_served_model", lambda self: {"service_mode": "streaming"}
            )
            s_backend = backend("streaming", port)
            await s_backend.load()
            with pytest.raises(Unsupported):
                await s_backend.transcribe(req)
            s = await s_backend.open_stream()
            for k in range(3):
                await s.send(s_backend.prepare_chunk(np.zeros(1600, np.float32)), is_last=k == 2)
            final = await s.wait_final(5.0)
            await s_backend.close()
            return req, pcm_req, info, s, final
        finally:
            await server.stop(None)

    req, pcm_req, info, s, final = asyncio.run(go())
    # Unary: by default the very bytes the HTTP backend uploads.
    assert req.config.encoding == pb.RecognitionConfig.WAV
    assert req.audio.content == encode_wav(np.zeros(8000, np.float32), 16000)
    assert pcm_req.config.encoding == pb.RecognitionConfig.LINEAR16
    assert len(pcm_req.audio.content) == 2 * 8000
    assert info["transport"] == "grpc" and "Recognize WAV" in info["wire"]
    # Streaming: config first, PCM16 chunks, cumulative partials, then the final.
    assert seen["stream_cfg"].interim_results
    assert seen["stream_cfg"].config.encoding == pb.RecognitionConfig.LINEAR16
    assert seen["stream_bytes"] == 3 * 3200
    assert [e.kind for e in s.events] == ["partial"] * 3 + ["final"]
    assert [e.text for e in s.events] == ["a", "aa", "aaa", "done"] and final.text == "done"


def test_oasr_server_grpc_options_are_validated():
    from benchmarks.asrbench.backends.openai_http import OasrServerBackend

    with pytest.raises(ValueError, match="transport"):
        OasrServerBackend({"transport": "carrier-pigeon"})
    with pytest.raises(ValueError, match="grpc_encoding"):
        OasrServerBackend({"transport": "grpc", "grpc_encoding": "mp3"})
    attached = OasrServerBackend({"transport": "grpc", "base_url": "http://127.0.0.1:9"})
    if backend_class("oasr_server").available()[0]:
        with pytest.raises((ValueError, RuntimeError), match="grpc"):
            asyncio.run(attached.load())


def test_engine_helpers_without_a_gpu(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from benchmarks.asrbench.backends.oasr_engine import OasrEngineBackend, checkpoint_fingerprint

    snap = tmp_path / "models--x" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"0" * 10)
    fp = checkpoint_fingerprint(str(snap))
    assert fp["revision"] == "abc123" and fp["bytes"] == 10
    assert fp == checkpoint_fingerprint(str(snap))

    monkeypatch.delenv("CKPT_DIR", raising=False)
    with pytest.raises(ValueError, match="ckpt_dir"):
        OasrEngineBackend({}, "offline")._engine_kwargs()
    with pytest.raises(ValueError, match="dtype"):
        OasrEngineBackend({"ckpt_dir": str(snap), "dtype": "int8"})._engine_kwargs()
    kw = OasrEngineBackend(
        {"ckpt_dir": str(snap), "chunk_size": 32, "engine": {"max_wait_time": 0.01}}, "streaming"
    )._engine_kwargs()
    assert kw["service_mode"] == "streaming" and kw["max_wait_time"] == 0.01
    assert kw["block_size_frames"] >= 32  # one paged block holds one encoder chunk


def test_transformers_backend_refuses_streaming_statically():
    cls = backend_class("transformers")
    assert cls.mode_unsupported("streaming", {})
    assert cls.mode_unsupported("offline", {}) is None
