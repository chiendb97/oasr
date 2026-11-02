#!/usr/bin/env python3
"""Tests for ``oasr.client`` and the ``oasr`` command line.

Both are exercised against a **stub HTTP server** that speaks the same shapes
``oasr-server`` does.  That is the honest boundary for a client library: what it
sends (multipart field names, the filename the server uses as a container hint)
and what it makes of a response — including the two different error envelopes a
caller can hit, which is exactly the part a hand-rolled client gets wrong.

Nothing here needs a checkpoint, a GPU, or the Rust extension.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from oasr.client import (
    AsyncOASRClient,
    OASRClient,
    OASRClientError,
    Transcription,
    _read_audio,
)

httpx = pytest.importorskip("httpx", reason="the OASR client needs httpx")


# --------------------------------------------------------------------------- #
# Stub server
# --------------------------------------------------------------------------- #


class _Stub(BaseHTTPRequestHandler):
    """Answers the routes the client calls; records the last request."""

    #: Filled per request so a test can assert what the client sent.
    last = {}
    #: Set by a test to force an error response.
    fail_with = None

    def log_message(self, *args):  # noqa: D102 - silence the default stderr spam
        pass

    def _send(self, status, body, content_type="application/json"):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/readyz":
            return self._send(200, "ready", "text/plain")
        if self.path == "/v1/models":
            return self._send(
                200,
                json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "/ckpt/u2pp",
                                "object": "model",
                                "info": {"decode_method": "ctc", "sample_rate": 16000},
                            }
                        ],
                    }
                ),
            )
        self._send(404, json.dumps({"error": {"message": "no"}}))

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Stub.last = {
            "path": self.path,
            "content_type": self.headers.get("Content-Type", ""),
            "fields": _multipart_fields(raw),
            "authorization": self.headers.get("Authorization"),
            "size": len(raw),
        }
        if _Stub.fail_with is not None:
            status, body = _Stub.fail_with
            return self._send(status, body)
        fmt = _Stub.last["fields"].get("response_format", "json")
        if fmt == "text":
            return self._send(200, "hello world", "text/plain; charset=utf-8")
        if fmt == "verbose_json":
            return self._send(
                200,
                json.dumps(
                    {
                        "task": "transcribe",
                        "duration": 1.5,
                        "text": "hello world",
                        "segments": [{"id": 0, "start": 0.0, "end": 1.5, "text": "hello world"}],
                        "request_id": "rid-1",
                    }
                ),
            )
        self._send(200, json.dumps({"text": "hello world"}))


def _multipart_fields(raw: bytes) -> dict:
    """Pull `name -> value` (and the file's declared filename) out of a body."""
    fields = {}
    for part in re.split(rb"--[-\w]+", raw):
        m = re.search(rb'name="([^"]+)"', part)
        if not m:
            continue
        name = m.group(1).decode()
        fname = re.search(rb'filename="([^"]*)"', part)
        if fname:
            fields["__filename__"] = fname.group(1).decode()
            fields[name] = f"<{len(part)} bytes>"
            continue
        body = part.split(b"\r\n\r\n", 1)
        if len(body) == 2:
            fields[name] = body[1].strip(b"\r\n").decode()
    return fields


@pytest.fixture
def server():
    _Stub.last = {}
    _Stub.fail_with = None
    httpd = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def wav(tmp_path):
    """A tiny real WAV file on disk."""
    path = tmp_path / "speech.wav"
    n = 160
    data = b"".join(int(i * 100).to_bytes(2, "little", signed=True) for i in range(n))
    header = (
        b"RIFF"
        + (36 + len(data)).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (16000).to_bytes(4, "little")
        + (32000).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + len(data).to_bytes(4, "little")
    )
    path.write_bytes(header + data)
    return path


# --------------------------------------------------------------------------- #
# What the client sends
# --------------------------------------------------------------------------- #


class TestRequestShape:
    def test_transcribe_posts_the_openai_form(self, server, wav):
        client = OASRClient(server)
        result = client.transcribe(wav, model="whisper-1", language="fr-FR")
        assert isinstance(result, Transcription)
        assert result.text == "hello world"

        sent = _Stub.last
        assert sent["path"] == "/v1/audio/transcriptions"
        assert sent["content_type"].startswith("multipart/form-data")
        assert sent["fields"]["model"] == "whisper-1"
        # The tag is forwarded as given; the *server* reduces it to a primary
        # subtag, so the client must not silently rewrite the caller's value.
        assert sent["fields"]["language"] == "fr-FR"
        assert sent["fields"]["response_format"] == "json"

    def test_the_filename_is_forwarded_as_a_container_hint(self, server, tmp_path):
        """The server sniffs the body, but falls back to the filename's
        extension — so a client that posts every upload as "audio.wav" makes
        that fallback useless."""
        mp3 = tmp_path / "podcast.mp3"
        mp3.write_bytes(b"ID3\x04\x00rest of a file")
        OASRClient(server).transcribe(mp3)
        assert _Stub.last["fields"]["__filename__"] == "podcast.mp3"

    def test_translate_hits_the_other_route(self, server, wav):
        OASRClient(server).translate(wav, language="fr")
        assert _Stub.last["path"] == "/v1/audio/translations"

    def test_a_default_model_is_applied_to_every_call(self, server, wav):
        OASRClient(server, model="whisper-1").transcribe(wav)
        assert _Stub.last["fields"]["model"] == "whisper-1"
        # An explicit argument still wins.
        OASRClient(server, model="whisper-1").transcribe(wav, model="other")
        assert _Stub.last["fields"]["model"] == "other"

    def test_an_api_key_is_sent_as_a_bearer_token(self, server, wav):
        OASRClient(server, api_key="sk-test").transcribe(wav)
        assert _Stub.last["authorization"] == "Bearer sk-test"

    def test_bytes_and_file_objects_are_accepted(self, server, wav):
        client = OASRClient(server)
        assert client.transcribe(wav.read_bytes()).text == "hello world"
        with open(wav, "rb") as fh:
            assert client.transcribe(fh).text == "hello world"

    def test_a_text_source_is_refused_with_a_clear_message(self):
        with pytest.raises(OASRClientError, match="binary mode"):
            import io

            _read_audio(io.StringIO("not bytes"))


# --------------------------------------------------------------------------- #
# What the client makes of a response
# --------------------------------------------------------------------------- #


class TestResponseHandling:
    def test_non_json_formats_come_back_as_the_document(self, server, wav):
        result = OASRClient(server).transcribe(wav, response_format="text")
        assert result.text == "hello world"

    def test_verbose_json_populates_the_structured_fields(self, server, wav):
        result = OASRClient(server).transcribe(wav, response_format="verbose_json")
        assert result.duration == 1.5
        assert result.request_id == "rid-1"
        assert len(result.segments) == 1
        # Nothing is dropped: a field this dataclass does not model stays in raw.
        assert result.raw["task"] == "transcribe"

    def test_the_openai_error_envelope_is_surfaced(self, server, wav):
        _Stub.fail_with = (
            400,
            json.dumps({"error": {"message": "word-level timestamps are not available yet"}}),
        )
        with pytest.raises(OASRClientError, match="word-level timestamps") as exc:
            OASRClient(server).transcribe(wav)
        assert exc.value.status == 400

    def test_the_google_error_envelope_is_surfaced_too(self, server, wav):
        """A client should not have to know which route produced the failure."""
        _Stub.fail_with = (
            503,
            json.dumps({"error": {"code": 503, "status": "RESOURCE_EXHAUSTED", "message": "busy"}}),
        )
        with pytest.raises(OASRClientError, match="busy"):
            OASRClient(server).transcribe(wav)

    def test_an_unreachable_server_says_so_without_a_status(self, wav):
        with pytest.raises(OASRClientError, match="could not reach") as exc:
            # Port 1 is reserved and never listening.
            OASRClient("http://127.0.0.1:1", timeout=1.0).transcribe(wav)
        assert exc.value.status is None, "a transport failure has no HTTP status"

    def test_models_and_readiness(self, server):
        client = OASRClient(server)
        models = client.models()
        assert [m["id"] for m in models] == ["/ckpt/u2pp"]
        assert client.is_ready() is True
        assert OASRClient("http://127.0.0.1:1").is_ready() is False


class TestAsyncClient:
    # Driven with `asyncio.run` rather than a plugin: the suite has no async
    # test framework, and an `async def` test without one is silently *skipped*
    # — which is how a broken async client stays green.
    def test_async_transcribe_matches_the_sync_client(self, server, wav):
        import asyncio

        async def go():
            async with AsyncOASRClient(server) as client:
                return await client.transcribe(wav, language="de")

        result = asyncio.run(go())
        assert result.text == "hello world"
        assert _Stub.last["fields"]["language"] == "de"

    def test_async_translate_hits_the_other_route(self, server, wav):
        import asyncio

        async def go():
            async with AsyncOASRClient(server) as client:
                return await client.translate(wav)

        assert asyncio.run(go()).text == "hello world"
        assert _Stub.last["path"] == "/v1/audio/translations"


class TestRealtimeStreaming:
    """The ``/v1/realtime`` client, against a stub WebSocket speaking the
    server's protocol.

    Worth a real socket rather than a mocked one: the session handshake, the
    binary audio frames and the commit are three separate things the client has
    to get right, and none of them shows up in a response body.
    """

    @staticmethod
    def _run(chunks, **kwargs):
        import asyncio

        websockets = pytest.importorskip("websockets")
        seen = {"session": None, "audio": bytearray(), "committed": False}

        async def handler(ws):
            async for message in ws:
                if isinstance(message, bytes):
                    seen["audio"].extend(message)
                    continue
                event = json.loads(message)
                if event["type"] == "session.update":
                    seen["session"] = event["session"]
                elif event["type"] == "input_audio_buffer.commit":
                    seen["committed"] = True
                    break
            # Two interim updates, then the final -- the server's own shape:
            # `delta` is the increment and nothing else.
            for delta in ("hello", " world"):
                await ws.send(_delta(delta))
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "hello world",
                    }
                )
            )

        async def go():
            async with websockets.serve(handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                client = AsyncOASRClient(f"http://127.0.0.1:{port}")
                return [e async for e in client.stream(chunks, **kwargs)]

        return asyncio.run(go()), seen

    def test_chunks_are_sent_as_binary_and_the_final_ends_the_iterator(self):
        chunks = [b"\x01\x02" * 80, b"\x03\x04" * 80]
        events, seen = self._run(chunks, sample_rate=8000, language="fr", task="translate")

        # Binary frames, byte-exact — base64 would have inflated them ~33%.
        assert bytes(seen["audio"]) == b"".join(chunks)
        assert seen["committed"], "the client must commit, or the server never decodes"
        assert seen["session"]["sample_rate"] == 8000
        assert seen["session"]["language"] == "fr"
        assert seen["session"]["task"] == "translate"

        assert [e.is_final for e in events] == [False, False, True]
        assert [e.text for e in events] == ["hello", "hello world", "hello world"]
        assert [e.delta for e in events] == ["hello", " world", ""]

    def test_an_async_chunk_source_works_too(self):
        async def mic():
            for _ in range(3):
                yield b"\x00\x01" * 40

        events, seen = self._run(mic())
        assert len(seen["audio"]) == 3 * 80
        assert events[-1].is_final and events[-1].text == "hello world"

    def test_a_server_error_event_raises(self):
        import asyncio

        websockets = pytest.importorskip("websockets")

        async def handler(ws):
            async for message in ws:
                if isinstance(message, str) and "commit" in message:
                    break
            await ws.send(
                json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "no audio received"},
                    }
                )
            )

        async def go():
            async with websockets.serve(handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                client = AsyncOASRClient(f"http://127.0.0.1:{port}")
                return [e async for e in client.stream([b""])]

        with pytest.raises(OASRClientError, match="no audio received"):
            asyncio.run(go())

    @staticmethod
    def _session(handler, consume):
        """Serve ``handler`` on a local socket and run ``consume(client)``."""
        import asyncio

        websockets = pytest.importorskip("websockets")

        async def go():
            async with websockets.serve(handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                return await asyncio.wait_for(
                    consume(AsyncOASRClient(f"http://127.0.0.1:{port}")), 10
                )

        return asyncio.run(go())

    def test_a_revision_replaces_the_transcript(self):
        """A partial that revises the transcript has no increment: the server
        sends the corrected text as ``text``, including a revision to nothing."""

        async def handler(ws):
            async for message in ws:
                if isinstance(message, str) and "commit" in message:
                    break
            for event in (
                _delta("their"),
                _delta("", text="there are"),
                _delta(" two"),
                _delta("", text=""),
                _delta("two"),
            ):
                await ws.send(event)
            await ws.send(_completed("two"))

        async def consume(client):
            return [(e.text, e.delta) async for e in client.stream([b"\x00\x01"])]

        assert self._session(handler, consume) == [
            ("their", "their"),
            ("there are", ""),
            ("there are two", " two"),
            ("", ""),
            ("two", "two"),
            ("two", ""),
        ]

    def test_events_arrive_while_the_audio_is_still_being_sent(self):
        """Interim results are read as the audio goes out, not after it.

        The source does not produce its second chunk until the caller has seen
        40 events -- more than the 16 that `websockets` queues unread before it
        stops reading the socket, pongs included.  Read only after the commit,
        this waited forever; against a live server, a session longer than ~40 s
        died of its own keepalive timeout.
        """
        import asyncio

        async def handler(ws):
            sent = False
            async for message in ws:
                if isinstance(message, bytes) and not sent:
                    sent = True
                    for i in range(40):
                        await ws.send(_delta(f" w{i}"))
                elif isinstance(message, str) and "commit" in message:
                    break
            await ws.send(_completed("done"))

        async def consume(client):
            seen = asyncio.Event()
            events = []

            async def source():
                yield b"\x00\x01" * 40
                await asyncio.wait_for(seen.wait(), 5)
                yield b"\x00\x01" * 40

            async for e in client.stream(source()):
                events.append(e)
                if len(events) == 40:
                    seen.set()
            return events

        events = self._session(handler, consume)
        assert len(events) == 41 and events[-1].is_final
        assert events[39].text == "".join(f" w{i}" for i in range(40))

    def test_a_sync_source_that_never_suspends_still_lets_events_in(self):
        """A list or a plain generator never yields to the loop, and a send
        that the socket takes at once does not either, so the sender gives the
        reader a turn per chunk.  Without it the first event waited for the
        last chunk, exactly as when nothing read until the commit."""
        seen = {}

        async def handler(ws):
            answered = False
            async for message in ws:
                if isinstance(message, bytes) and not answered:
                    answered = True
                    await ws.send(_delta("hi"))
                elif isinstance(message, str) and "commit" in message:
                    break
            await ws.send(_completed("hi"))

        def source():
            for i in range(5000):
                seen["pending"] = i
                yield b"\x00\x01" * 40

        async def consume(client):
            async for _ in client.stream(source()):
                seen.setdefault("first_event_at_chunk", seen["pending"])
            return seen

        got = self._session(handler, consume)
        assert got["first_event_at_chunk"] < 4999, got

    def test_a_failing_source_raises_its_own_error(self):
        async def handler(ws):
            async for _ in ws:
                pass

        async def consume(client):
            async def source():
                yield b"\x00\x01" * 40
                raise ValueError("microphone unplugged")

            return [e async for e in client.stream(source())]

        with pytest.raises(ValueError, match="microphone unplugged"):
            self._session(handler, consume)

    def test_a_session_closed_without_a_final_raises(self):
        """The iterator ending normally means the final arrived; a session the
        server dropped must not look like one that finished."""

        async def handler(ws):
            async for message in ws:
                if isinstance(message, str) and "commit" in message:
                    break
            await ws.send(_delta("partial"))

        async def consume(client):
            return [e async for e in client.stream([b"\x00\x01"])]

        with pytest.raises(OASRClientError, match="closed before the final"):
            self._session(handler, consume)

    def test_a_final_before_the_audio_ends_finishes_the_stream(self):
        """A server may close the turn itself (an endpointer, a length cap)
        while audio is still going out.  That is the end of the stream, not a
        failure to send the rest."""
        import asyncio

        async def handler(ws):
            async for message in ws:
                if isinstance(message, bytes):
                    break
            await ws.send(_completed("short"))

        async def consume(client):
            async def endless():
                while True:
                    yield b"\x00\x01" * 40
                    await asyncio.sleep(0.01)

            return [e async for e in client.stream(endless())]

        events = self._session(handler, consume)
        assert [(e.text, e.is_final) for e in events] == [("short", True)]

    def test_a_caller_that_stops_reading_ends_the_session(self):
        import asyncio
        import contextlib

        closed = []

        async def handler(ws):
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        await ws.send(_delta("x"))
            finally:
                closed.append(True)

        async def consume(client):
            async def endless():
                while True:
                    yield b"\x00\x01" * 40
                    await asyncio.sleep(0.01)

            async with contextlib.aclosing(client.stream(endless())) as events:
                async for _ in events:
                    break
            for _ in range(100):
                if closed:
                    break
                await asyncio.sleep(0.02)
            return closed

        assert self._session(handler, consume) == [True], "the socket outlived the caller"


def _delta(delta, **extra):
    return json.dumps(
        {"type": "conversation.item.input_audio_transcription.delta", "delta": delta, **extra}
    )


def _completed(transcript):
    return json.dumps(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": transcript}
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCli:
    def test_transcribe_prints_the_transcript(self, server, wav, capsys):
        from oasr.cli import main

        assert main(["transcribe", str(wav), "--url", server]) == 0
        assert capsys.readouterr().out.strip() == "hello world"
        # The CLI defaults to `text`, which is what a terminal wants.
        assert _Stub.last["fields"]["response_format"] == "text"

    def test_translate_uses_the_translations_route(self, server, wav, capsys):
        from oasr.cli import main

        assert main(["translate", str(wav), "--url", server]) == 0
        capsys.readouterr()
        assert _Stub.last["path"] == "/v1/audio/translations"

    def test_several_files_in_one_invocation(self, server, wav, capsys):
        from oasr.cli import main

        main(["transcribe", str(wav), str(wav), "--url", server])
        assert capsys.readouterr().out.strip().splitlines() == ["hello world", "hello world"]

    def test_output_file(self, server, wav, tmp_path, capsys):
        from oasr.cli import main

        out = tmp_path / "t.txt"
        main(["transcribe", str(wav), "--url", server, "-o", str(out)])
        capsys.readouterr()
        assert out.read_text().strip() == "hello world"

    def test_models_lists_what_the_server_serves(self, server, capsys):
        from oasr.cli import main

        assert main(["models", "--url", server]) == 0
        assert "/ckpt/u2pp" in capsys.readouterr().out

    def test_a_dead_server_exits_nonzero_with_a_hint(self, wav, capsys):
        """The first thing a new user hits is "no server running"; the error has
        to say what to do about it."""
        from oasr.cli import main

        assert main(["transcribe", str(wav), "--url", "http://127.0.0.1:1"]) == 1
        err = capsys.readouterr().err
        assert "error:" in err and "oasr serve" in err

    def test_a_server_error_is_reported(self, server, wav, capsys):
        from oasr.cli import main

        _Stub.fail_with = (400, json.dumps({"error": {"message": "unsupported codec: Opus"}}))
        assert main(["transcribe", str(wav), "--url", server]) == 1
        assert "unsupported codec" in capsys.readouterr().err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
