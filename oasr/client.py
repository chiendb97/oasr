# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""A first-class client for a running ``oasr-server``.

Before this module the only "clients" in the tree were two example scripts and
a benchmark harness, so the first thing anyone evaluating OASR had to do was
write one.  The API here is deliberately the shape people already know — the
same call names and arguments as OpenAI's audio client — over OASR's
OpenAI-compatible endpoints:

.. code-block:: python

    from oasr.client import OASRClient

    client = OASRClient("http://127.0.0.1:8080")
    print(client.transcribe("meeting.mp3").text)
    print(client.translate("entretien.m4a", language="fr").text)

and, for live audio, an async iterator over the ``/v1/realtime`` WebSocket:

.. code-block:: python

    async with AsyncOASRClient("http://127.0.0.1:8080") as client:
        async for event in client.stream(mic_chunks(), sample_rate=16000):
            print(event.text, end="\\r" if not event.is_final else "\\n")

``httpx`` (and ``websockets`` for streaming) are needed at call time and come
from the ``serving`` extra: ``pip install "oasr[serving]"``.  They are imported
lazily so ``import oasr.client`` costs nothing on a machine that only serves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, Iterable, List, Optional, Tuple, Union

__all__ = [
    "AsyncOASRClient",
    "OASRClient",
    "Transcription",
    "TranscriptEvent",
    "OASRClientError",
]

#: Where a local ``oasr-server`` listens by default.
DEFAULT_BASE_URL = "http://127.0.0.1:8080"

#: Anything acceptable as audio: a path, raw encoded bytes, or an open file.
AudioSource = Union[str, Path, bytes, bytearray, BinaryIO]


class OASRClientError(RuntimeError):
    """A request the server rejected, or a transport failure.

    ``status`` is the HTTP status when the server answered; ``None`` when the
    request never got that far.
    """

    def __init__(self, message: str, *, status: Optional[int] = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class Transcription:
    """One completed transcription.

    ``text`` is what nearly every caller wants; the rest is present when the
    server produced it. ``raw`` keeps the full decoded response so a field this
    dataclass does not model is still reachable rather than lost.
    """

    text: str
    request_id: Optional[str] = None
    duration: Optional[float] = None
    language: Optional[str] = None
    finish_reason: Optional[str] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    #: ``[{"word", "start", "end", "confidence"}]`` — present only when the
    #: request asked (``timestamp_granularities=["word"]`` with
    #: ``response_format="verbose_json"``) and the decode family produced them.
    #: Empty is the honest answer for a request that did not ask; a family that
    #: *cannot* align rejects the request instead, so an empty list here never
    #: means "unsupported".
    words: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # `print(client.transcribe(...))` should work
        return self.text


@dataclass
class TranscriptEvent:
    """One update from a live session.

    ``text`` is always the transcript **so far**, not the increment: a caller
    rendering a caption wants to overwrite, and a caller accumulating deltas can
    do so from ``delta``. Getting this backwards duplicates text on screen.
    """

    text: str
    delta: str = ""
    is_final: bool = False


def _read_audio(audio: AudioSource) -> Tuple[str, bytes]:
    """``(filename, bytes)`` for any accepted audio source.

    The filename matters: the server uses it as a container hint when the
    upload's content type says nothing useful (which is what ``curl -F`` and
    most HTTP libraries send).
    """
    if isinstance(audio, (bytes, bytearray)):
        return "audio.wav", bytes(audio)
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        try:
            return path.name, path.read_bytes()
        except OSError as exc:
            raise OASRClientError(f"could not read {path}: {exc}") from exc
    if hasattr(audio, "read"):
        data = audio.read()
        if not isinstance(data, (bytes, bytearray)):
            raise OASRClientError("the file object must be opened in binary mode ('rb')")
        name = getattr(audio, "name", "audio.wav")
        return Path(str(name)).name, bytes(data)
    raise OASRClientError(f"unsupported audio source {type(audio).__name__}")


def _import_httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dep
        raise OASRClientError(
            "the OASR client needs `httpx`; install it with "
            '`pip install "oasr[serving]"` or `pip install httpx`'
        ) from exc
    return httpx


def _form(
    *,
    model: Optional[str],
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    temperature: Optional[float],
    timestamp_granularities: Optional[Iterable[str]],
) -> List[Tuple[str, str]]:
    """The multipart text parts, in the shape the server (and OpenAI) expect."""
    parts: List[Tuple[str, str]] = [("response_format", response_format)]
    if model:
        parts.append(("model", model))
    if language:
        parts.append(("language", language))
    if prompt:
        parts.append(("prompt", prompt))
    if temperature is not None:
        parts.append(("temperature", str(temperature)))
    for g in timestamp_granularities or ():
        parts.append(("timestamp_granularities[]", g))
    return parts


def _to_transcription(status: int, content_type: str, body: bytes) -> Transcription:
    """Turn a 2xx response into a [`Transcription`], whatever format it is in."""
    text_body = body.decode("utf-8", errors="replace")
    if "json" not in content_type:
        # text / srt / vtt come back as the document itself.
        return Transcription(text=text_body.strip(), raw={"text": text_body})
    try:
        payload = json.loads(text_body)
    except json.JSONDecodeError as exc:
        raise OASRClientError(
            f"server returned {status} with unparseable JSON: {exc}", status=status
        ) from exc
    return Transcription(
        text=payload.get("text", ""),
        request_id=payload.get("request_id"),
        duration=payload.get("duration"),
        language=payload.get("language"),
        finish_reason=payload.get("finish_reason"),
        segments=payload.get("segments") or [],
        words=payload.get("words") or [],
        raw=payload,
    )


def _raise_for_status(status: int, body: bytes) -> None:
    """Turn a non-2xx response into an [`OASRClientError`] carrying the reason.

    Both server envelopes are understood — OpenAI's ``error.message`` and the
    Google-shaped ``error.status`` — because a client should not have to know
    which route produced the failure.
    """
    if 200 <= status < 300:
        return
    detail: Any = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(detail)
        err = payload.get("error", payload)
        detail = err.get("message", detail) if isinstance(err, dict) else detail
    except (json.JSONDecodeError, AttributeError):
        payload = None
    raise OASRClientError(f"server returned {status}: {detail}", status=status, body=payload)


class _BaseClient:
    """Shared configuration for the sync and async clients."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Accepted for compatibility with OpenAI-shaped tooling that always
        # sends one. `oasr-server` has no auth today, so it is simply forwarded.
        self.api_key = api_key or os.environ.get("OASR_API_KEY")
        self.model = model
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _ws_url(self, path: str = "/v1/realtime") -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[-1]
        return f"{scheme}://{host}{path}"


class OASRClient(_BaseClient):
    """Blocking client for ``oasr-server``'s OpenAI-compatible endpoints."""

    def transcribe(
        self,
        audio: AudioSource,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[Iterable[str]] = None,
        timeout: Optional[float] = None,
    ) -> Transcription:
        """Transcribe one audio file.

        ``audio`` may be a path, raw encoded bytes, or an open binary file. Any
        container the server can decode works — WAV, MP3, FLAC, M4A, OGG, AIFF —
        so callers no longer transcode first.
        """
        return self._post(
            "/v1/audio/transcriptions",
            audio,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            timeout=timeout,
        )

    def translate(
        self,
        audio: AudioSource,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Transcription:
        """Translate speech to English.

        Needs a checkpoint whose decode family has a task control (Whisper);
        anything else answers ``400`` naming the limitation rather than
        transcribing and calling it a translation.  ``language`` is the
        **source** language hint.
        """
        return self._post(
            "/v1/audio/translations",
            audio,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=None,
            timeout=timeout,
        )

    def models(self) -> List[Dict[str, Any]]:
        """The models this server serves (`GET /v1/models`)."""
        httpx = _import_httpx()
        try:
            resp = httpx.get(self._url("/v1/models"), headers=self._headers(), timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise OASRClientError(f"could not reach {self.base_url}: {exc}") from exc
        _raise_for_status(resp.status_code, resp.content)
        payload: Dict[str, Any] = resp.json()
        return list(payload.get("data", []))

    def is_ready(self) -> bool:
        """Whether the engine is loaded and serving (`GET /readyz`)."""
        httpx = _import_httpx()
        try:
            return bool(httpx.get(self._url("/readyz"), timeout=5.0).status_code == 200)
        except Exception:
            return False

    def _post(
        self,
        path: str,
        audio: AudioSource,
        *,
        timeout: Optional[float],
        **kwargs: Any,
    ) -> Transcription:
        httpx = _import_httpx()
        filename, blob = _read_audio(audio)
        kwargs["model"] = kwargs.get("model") or self.model
        try:
            resp = httpx.post(
                self._url(path),
                headers=self._headers(),
                files={"file": (filename, blob)},
                data=dict(_form(**kwargs)),
                timeout=timeout if timeout is not None else self.timeout,
            )
        except httpx.HTTPError as exc:
            raise OASRClientError(f"could not reach {self.base_url}: {exc}") from exc
        _raise_for_status(resp.status_code, resp.content)
        return _to_transcription(
            resp.status_code, resp.headers.get("content-type", ""), resp.content
        )


class AsyncOASRClient(_BaseClient):
    """Async client, including the live ``/v1/realtime`` session."""

    async def transcribe(
        self,
        audio: AudioSource,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[Iterable[str]] = None,
        timeout: Optional[float] = None,
    ) -> Transcription:
        """:meth:`OASRClient.transcribe`, awaited."""
        return await self._post(
            "/v1/audio/transcriptions",
            audio,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            timeout=timeout,
        )

    async def translate(
        self,
        audio: AudioSource,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Transcription:
        """:meth:`OASRClient.translate`, awaited."""
        return await self._post(
            "/v1/audio/translations",
            audio,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=None,
            timeout=timeout,
        )

    async def _post(
        self,
        path: str,
        audio: AudioSource,
        *,
        timeout: Optional[float],
        **kwargs: Any,
    ) -> Transcription:
        httpx = _import_httpx()
        filename, blob = _read_audio(audio)
        kwargs["model"] = kwargs.get("model") or self.model
        async with httpx.AsyncClient(
            timeout=timeout if timeout is not None else self.timeout
        ) as http:
            try:
                resp = await http.post(
                    self._url(path),
                    headers=self._headers(),
                    files={"file": (filename, blob)},
                    data=dict(_form(**kwargs)),
                )
            except httpx.HTTPError as exc:
                raise OASRClientError(f"could not reach {self.base_url}: {exc}") from exc
        _raise_for_status(resp.status_code, resp.content)
        return _to_transcription(
            resp.status_code, resp.headers.get("content-type", ""), resp.content
        )

    async def stream(
        self,
        chunks: Union[Iterable[bytes], AsyncIterator[bytes]],
        *,
        sample_rate: int = 16000,
        encoding: str = "LINEAR16",
        language: Optional[str] = None,
        task: Optional[str] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        interim_results: bool = True,
    ) -> AsyncIterator[TranscriptEvent]:
        """Stream audio to ``/v1/realtime`` and yield transcript updates.

        ``chunks`` yields raw PCM in ``encoding`` at ``sample_rate``; they are
        sent as **binary** frames, which skips base64's ~33% inflation.  The
        iterator ends after the final event, whose ``is_final`` is ``True``.

        Works against either service mode.  A streaming engine decodes as the
        audio arrives; an offline one buffers it and streams the *text* back, so
        the interim events appear only once the audio is complete.
        """
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - optional dep
            raise OASRClientError(
                "streaming needs `websockets`; install it with "
                '`pip install "oasr[serving]"` or `pip install websockets`'
            ) from exc

        session = {
            "sample_rate": sample_rate,
            "encoding": encoding,
            "interim_results": interim_results,
        }
        for key, value in (
            ("language", language),
            ("task", task),
            ("prompt", prompt),
            ("model", model or self.model),
        ):
            if value:
                session[key] = value

        async with websockets.connect(self._ws_url(), **_ws_header_kwarg(self._headers())) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": session}))
            async for chunk in _aiter(chunks):
                if chunk:
                    await ws.send(chunk)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            text = ""
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                kind = event.get("type", "")
                if kind.endswith("input_audio_transcription.delta"):
                    delta = event.get("delta", "")
                    # `text` is OASR's extension carrying the authoritative
                    # transcript so far.  Prefer it: a frame-synchronous family
                    # can *revise* a partial, which arrives as an empty delta
                    # plus corrected text, and accumulating deltas alone would
                    # keep the superseded words on screen until the final.
                    text = event.get("text") or (text + delta)
                    yield TranscriptEvent(text=text, delta=delta, is_final=False)
                elif kind.endswith("input_audio_transcription.completed"):
                    text = event.get("transcript", text)
                    yield TranscriptEvent(text=text, delta="", is_final=True)
                    return
                elif kind == "error":
                    detail = event.get("error", {})
                    raise OASRClientError(
                        f"realtime session failed: {detail.get('message', detail)}"
                    )

    async def __aenter__(self) -> "AsyncOASRClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _ws_header_kwarg(headers: Dict[str, str]) -> Dict[str, Any]:
    """The keyword ``websockets.connect`` takes for extra headers, this version.

    It was renamed ``extra_headers`` → ``additional_headers`` in websockets 14,
    and the ``serving`` extra floors at 12.  Passing the wrong one is a
    ``TypeError`` at connect time — an error that would only ever show up for
    someone using an API key, which is the least-tested path.
    """
    if not headers:
        return {}
    import inspect

    import websockets

    params = inspect.signature(websockets.connect).parameters
    name = "additional_headers" if "additional_headers" in params else "extra_headers"
    return {name: headers}


async def _aiter(chunks: Union[Iterable[bytes], AsyncIterator[bytes]]) -> AsyncIterator[bytes]:
    """Iterate either a sync or an async source of chunks."""
    if hasattr(chunks, "__aiter__"):
        async for c in chunks:  # type: ignore[union-attr]
            yield c
    else:
        for c in chunks:  # type: ignore[union-attr]
            yield c
