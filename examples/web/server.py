#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Same-origin front door for the OASR web demo: static page + reverse proxy.

    python examples/web/server.py --oasr-server http://asr-host:8080
    # then open http://localhost:8000

Serves ``static/`` and forwards the API paths to a running ``oasr-server``, so the
browser only ever talks to *this* origin.  That is the whole point of the process:

* the server needs **no** ``--cors-allow-origin``, because nothing is cross-origin;
* the page needs no ``?server=`` and no SSH tunnel — one flag names the host;
* ``oasr-server`` needs no network exposure of its own.  It has no authentication,
  so being reachable only by this relay is a feature.

Not the bridge this repo used to ship
-------------------------------------

The previous ``server.py`` was a *protocol* bridge: a browser could not reach OASR
at all, so a FastAPI app translated ``WS /api/stream`` ⇆ gRPC ``StreamingRecognize``
and ``POST /api/recognize`` ⇆ ``Recognize``, framing raw f32 PCM itself.  It needed
fastapi, uvicorn and grpcio, and it had to understand every message on the wire —
so each protocol change broke it.

``oasr-server`` now speaks ``POST /v1/audio/transcriptions`` and ``WS /v1/realtime``
natively, so this file is a **byte relay instead of a translator**: the WebSocket
handshake is forwarded verbatim and the connection is then spliced raw, and nothing
here parses a frame or a JSON event.  A new realtime event type needs no change.
The page keeps talking to the real OASR API, which is also what makes it a working
example rather than a demo-only protocol.  Consequences worth knowing:

* the standard library is the only dependency — no ``requirements.txt``;
* per-request logs count *bytes*, not chunks and partials.  A relay that does not
  parse cannot report what it did not read; the server's own ``rid=`` log lines
  have the decode detail.

Microphone
----------

Browsers expose ``getUserMedia`` only in a secure context — HTTPS, or a loopback
host — and that is the browser's decision, so nothing this process sends can grant
it.  ``--tls-self-signed`` is the one-flag answer: HTTPS with a cached self-signed
certificate, warned about once per browser.  To stay on plain HTTP instead, the
origin has to be allowlisted in the browser; the startup warning spells out how
per vendor.  File upload never needs any of this.

Logging
-------

Every request gets a correlation id (``[http-1a2b3c4d]`` / ``[ws-…]``), also
returned as the ``x-oasr-trace`` response header so browser devtools joins these
logs.  ``--log-level info`` is one line per API request (and one per realtime
session, with its byte counts); ``debug`` adds static-asset requests, upstream
response headers and the reason a relay tore down::

    python examples/web/server.py --log-level debug --log-file /tmp/bridge.log
"""

import argparse
import ipaddress
import json
import logging
import os
import selectors
import shutil
import socket
import ssl
import subprocess
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import SplitResult, urlsplit

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

# Where --tls-self-signed keeps its pair.  Alongside the JIT cache, by the same
# convention, and *persistent* on purpose: see ensure_self_signed().
TLS_DIR = Path.home() / ".cache" / "oasr" / "web-demo"

# What the upstream owns; everything else is a file in ``static/``.  An explicit
# list, not "proxy whatever is not on disk": a typo in an asset path should 404
# here rather than turn into a puzzling error from the inference server.
PROXY_PREFIXES = ("/v1/", "/healthz", "/readyz", "/metrics")

# Headers that describe *this* hop and must not be handed to the next one.  The
# WebSocket path deliberately ignores this set: there, ``connection`` and
# ``upgrade`` are the payload of the hop rather than a description of it.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

BUF = 64 * 1024
CONNECT_TIMEOUT = 10.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG = logging.getLogger("oasr.webdemo")
LOG_HTTP = LOG.getChild("http")  # browser <-> relay, request/response pairs
LOG_WS = LOG.getChild("ws")  # browser <-> relay, realtime sessions
LOG_UP = LOG.getChild("upstream")  # relay <-> oasr-server

#: Correlation id of the in-flight request, stamped onto every record by
#: ``_TraceFilter``.  A ContextVar rather than a parameter, so that a handler deep
#: in a relay loop logs under the right id without it being passed down to it.
_TRACE: ContextVar[str] = ContextVar("oasr_webdemo_trace", default="-")

_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(trace)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


class _TraceFilter(logging.Filter):
    """Stamp each record with the current correlation id (``-`` when idle)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace = _TRACE.get()
        return True


def setup_logging(level: str, log_file: Optional[str] = None) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    trace = _TraceFilter()

    root = logging.getLogger()
    for stale in root.handlers[:]:
        root.removeHandler(stale)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
        handler.addFilter(trace)  # on the handler, so it covers every logger
        root.addHandler(handler)
    root.setLevel(lvl)


def _trace_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _kib(n: int) -> str:
    return f"{n / 1024:.1f} KiB" if n >= 1024 else f"{n} B"


def _quiet_shutdown(sock: socket.socket) -> None:
    """Half-close, ignoring the errors a peer that already left is entitled to."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _recv_available(sock: socket.socket) -> bytes:
    """One ``recv``, plus anything already decrypted inside the TLS layer.

    ``select`` reports on the *kernel* buffer, so a TLS socket holding a whole
    decrypted record would never wake it again — the classic SSL-with-select
    stall.  ``pending()`` exists only on ``SSLSocket``, hence the ``getattr``.
    """
    data = sock.recv(BUF)
    pending = getattr(sock, "pending", None)
    while data and pending is not None and pending():
        more = sock.recv(BUF)
        if not more:
            break
        data += more
    return data


@dataclass
class _RelayStats:
    """Byte counters for one spliced ``/v1/realtime`` session."""

    up: int = 0  # browser -> oasr-server (audio)
    down: int = 0  # oasr-server -> browser (events)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class Handler(SimpleHTTPRequestHandler):
    """Static files for the page, transparent proxy for the API paths."""

    #: Set once by ``main()``; every instance reads the same upstream.
    upstream: SplitResult = urlsplit("http://127.0.0.1:8080")

    # 1.1 so a page load is one keep-alive connection, and so a relayed response
    # may be delimited by close when the upstream gave no Content-Length.
    protocol_version = "HTTP/1.1"
    server_version = "oasr-web-demo"

    # ---- plumbing ----------------------------------------------------------

    def handle_one_request(self) -> None:
        # One correlation id per request, opened before the request line is even
        # parsed so that a malformed one is still attributable.  A realtime
        # session re-stamps itself as `ws-…` once the headers say so, which
        # cannot be known this early.
        token = _TRACE.set(_trace_id("http"))
        try:
            super().handle_one_request()
        finally:
            _TRACE.reset(token)

    def _wants_websocket(self) -> bool:
        return "websocket" in self.headers.get("Upgrade", "").lower()

    def end_headers(self) -> None:
        # Every response we generate — static, proxied or error — carries the id,
        # so a devtools entry can be matched against a log line.
        self.send_header("x-oasr-trace", _TRACE.get())
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # Route the base class's stderr chatter into logging. Static assets are
        # DEBUG so that INFO reads as one pair of lines per real API request.
        LOG_HTTP.debug(fmt, *args)

    def log_error(self, fmt: str, *args) -> None:
        LOG_HTTP.warning(fmt, *args)

    def _client(self) -> str:
        return f"{self.client_address[0]}:{self.client_address[1]}"

    # ---- routing -----------------------------------------------------------

    def _proxied(self) -> bool:
        return self.path.startswith(PROXY_PREFIXES)

    def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler's naming
        if not self._proxied():
            return super().do_GET()
        if self._wants_websocket():
            return self._splice_websocket()
        self._proxy("GET")

    def do_HEAD(self):  # noqa: N802
        return super().do_HEAD() if not self._proxied() else self._proxy("HEAD")

    def do_POST(self):  # noqa: N802
        if not self._proxied():
            return self.send_error(405, "Only the API paths accept POST")
        self._proxy("POST")

    def do_OPTIONS(self):  # noqa: N802
        # A same-origin fetch never preflights, so this exists only so a
        # non-browser client pointed at the relay behaves as it would upstream.
        return self._proxy("OPTIONS") if self._proxied() else self.send_error(405)

    # ---- plain HTTP --------------------------------------------------------

    def _open_upstream(self) -> HTTPConnection:
        cls = HTTPSConnection if self.upstream.scheme == "https" else HTTPConnection
        return cls(self.upstream.netloc, timeout=CONNECT_TIMEOUT)

    def _proxy(self, method: str) -> None:
        t0 = time.perf_counter()
        body_len = int(self.headers.get("Content-Length") or 0)
        LOG_HTTP.info("-> %s %s from %s (%s)", method, self.path, self._client(), _kib(body_len))
        try:
            conn = self._open_upstream()
            conn.putrequest(method, self.path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", self.upstream.netloc)
            for key, value in self.headers.items():
                if key.lower() not in HOP_BY_HOP and key.lower() != "host":
                    conn.putheader(key, value)
            conn.endheaders()
            if conn.sock is not None:
                # The connect deadline must not become a *body* deadline: an SSE
                # response (``stream=true``) is idle between events by design,
                # and a long upload outlasts any sane connect timeout.
                conn.sock.settimeout(None)
            self._forward_request_body(conn, body_len)
            resp = conn.getresponse()
        except OSError as exc:
            LOG_UP.error("%s %s failed after %.1f ms: %s", method, self.path, _ms(t0), exc)
            self._unreachable(exc)
            return

        LOG_UP.debug(
            "upstream %d %s, headers=%s", resp.status, resp.reason, dict(resp.getheaders())
        )
        try:
            sent = self._relay_response(resp, head_only=(method == "HEAD"))
        finally:
            resp.close()
            conn.close()
        LOG_HTTP.info(
            "<- %s %s %d in %.1f ms (%s)", method, self.path, resp.status, _ms(t0), _kib(sent)
        )

    def _forward_request_body(self, conn: HTTPConnection, body_len: int) -> None:
        """Stream the request body upstream in blocks.

        A transcription POST is a whole audio file, so buffering it here would
        double peak RSS for nothing.  A body with no ``Content-Length`` is not
        forwarded: ``fetch`` with a ``FormData`` always sets one, and a guess
        would be worse than the honest omission.
        """
        remaining = body_len
        while remaining > 0:
            chunk = self.rfile.read(min(BUF, remaining))
            if not chunk:
                break
            conn.send(chunk)
            remaining -= len(chunk)

    def _relay_response(self, resp: HTTPResponse, head_only: bool) -> int:
        self.send_response(resp.status, resp.reason)
        for key, value in resp.getheaders():
            if key.lower() not in HOP_BY_HOP:
                self.send_header(key, value)
        if resp.getheader("Content-Length") is None:
            # No length means chunked upstream — an SSE transcription stream, in
            # practice.  ``http.client`` has already de-chunked it, so rather
            # than re-frame, delimit by closing: legal HTTP/1.1, and it keeps the
            # relay from reproducing chunk boundaries it can no longer see.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if head_only:
            return 0
        sent = 0
        try:
            while True:
                # read1, not read: ``read`` loops to fill the buffer, which would
                # hold an SSE event back until enough of the *next* one arrived.
                chunk = resp.read1(BUF)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                sent += len(chunk)
        except OSError as exc:
            LOG_HTTP.debug("client left mid-response after %s: %s", _kib(sent), exc)
            self.close_connection = True
        return sent

    # ---- WebSocket ---------------------------------------------------------

    def _splice_websocket(self) -> None:
        """Forward the handshake verbatim, then relay bytes both ways.

        Nothing here understands WebSocket framing, and that is the design: the
        realtime protocol (``session.update``, binary audio, ``commit``) can grow
        without this file knowing.  ``Upgrade`` and ``Connection`` are forwarded
        on purpose — this hop *is* the upgrade — and so are the ``Sec-WebSocket-*``
        headers, which is what lets client and server agree end to end.
        """
        _TRACE.set(_trace_id("ws"))  # a session, not a request/response pair
        t0 = time.perf_counter()
        host = self.upstream.hostname or "127.0.0.1"
        port = self.upstream.port or (443 if self.upstream.scheme == "https" else 80)
        LOG_WS.info("open %s from %s -> %s:%d", self.path, self._client(), host, port)
        try:
            up: socket.socket = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
            if self.upstream.scheme == "https":
                up = ssl.create_default_context().wrap_socket(up, server_hostname=host)
        except OSError as exc:
            # A 502 to the handshake surfaces in the page as ``ws.onerror``,
            # which is where a refused connection lands too.
            LOG_UP.error("realtime connect failed: %s", exc)
            self._unreachable(exc)
            return
        up.settimeout(None)

        head = [f"GET {self.path} HTTP/1.1", f"Host: {self.upstream.netloc}"]
        head += [f"{k}: {v}" for k, v in self.headers.items() if k.lower() != "host"]
        try:
            up.sendall(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
        except OSError as exc:
            _quiet_shutdown(up)
            LOG_UP.error("realtime handshake failed: %s", exc)
            self._unreachable(exc)
            return

        # The 101 needs no special case: it is simply the first thing upstream
        # sends, and it rides the same relay as every frame after it.
        self.close_connection = True
        stats = _RelayStats()
        try:
            head_tail = self._buffered_tail()
            if head_tail:
                up.sendall(head_tail)
                stats.up += len(head_tail)
            self._relay(up, stats)
        finally:
            _quiet_shutdown(up)
            _quiet_shutdown(self.connection)
            # Bytes, not chunks and partials: see the module docstring — a relay
            # that does not parse cannot count what it never read.
            LOG_WS.info(
                "closed after %.1f ms, up %s / down %s",
                _ms(t0),
                _kib(stats.up),
                _kib(stats.down),
            )

    def _buffered_tail(self) -> bytes:
        """Whatever the header read pulled in past the end of the handshake.

        Normally nothing: a client waits for the 101 before sending a frame.  But
        ``rfile`` is buffered, and bytes already sitting in it would be invisible
        to a selector that watches the socket, so they are drained here — without
        blocking, since usually there are none.
        """
        self.connection.setblocking(False)
        try:
            return self.rfile.read1(BUF) or b""
        except OSError:  # nothing buffered (incl. ssl.SSLWantReadError)
            return b""
        finally:
            self.connection.setblocking(True)

    def _relay(self, up: socket.socket, stats: _RelayStats) -> None:
        """Shuttle both directions from this one thread.

        One thread rather than a pump per direction, because with
        ``--ssl-certfile`` the browser side is an ``SSLSocket``: OpenSSL requires
        that one SSL object be used by one thread at a time, and a concurrent
        read/write pair is exactly what a TLS 1.3 KeyUpdate mid-session turns
        into corruption.  It also makes the half-close below simple to get right.
        """
        client = self.connection
        with selectors.DefaultSelector() as sel:
            sel.register(client, selectors.EVENT_READ, "up")
            sel.register(up, selectors.EVENT_READ, "down")
            while sel.get_map():
                for key, _ in sel.select():
                    upward = key.data == "up"
                    src, dst = (client, up) if upward else (up, client)
                    try:
                        data = _recv_available(src)
                    except OSError as exc:
                        LOG_WS.debug("relay %s ended: %s", key.data, exc)
                        data = b""
                    if not data:
                        # Half-close rather than tear down: a server still
                        # sending its final transcript must not be cut off by a
                        # client that has finished speaking.
                        sel.unregister(src)
                        try:
                            dst.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        continue
                    try:
                        dst.sendall(data)
                    except OSError as exc:
                        LOG_WS.debug("relay %s send failed: %s", key.data, exc)
                        return
                    if upward:
                        stats.up += len(data)
                    else:
                        stats.down += len(data)

    # ---- errors ------------------------------------------------------------

    def _unreachable(self, exc: OSError) -> None:
        """502 in the shape the page already parses (``data.error.message``)."""
        message = f"web demo cannot reach oasr-server at {self.upstream.geturl()}: {exc}"
        body = json.dumps({"error": {"message": message, "type": "upstream_unreachable"}}).encode()
        self.send_response(502, "Bad Gateway")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Microphone / secure context
# ---------------------------------------------------------------------------
#
# Browsers expose getUserMedia only in a "secure context": HTTPS, or a loopback
# host.  That is decided by the *browser*, so no flag or header on this side can
# grant it — `--tls-self-signed` below removes the certificate work instead, and
# the warning names the browser-side allowlist for anyone who must stay on plain
# HTTP.  File upload is unaffected either way.


def _tls_names(bind_host: str, extra: List[str]) -> List[str]:
    """Names a self-signed cert should cover: whatever the user might type."""
    names = {"localhost", "127.0.0.1", "::1"}
    if bind_host not in ("0.0.0.0", "::"):
        names.add(bind_host)
    try:
        hostname = socket.gethostname()
        names.update({hostname, socket.getfqdn()})
        names.update(socket.gethostbyname_ex(hostname)[2])
    except OSError:  # unresolvable hostname is not fatal — a SAN miss is one click
        pass
    try:
        # The address a *remote* browser would use, which is usually the one
        # typed into the address bar and often not what the hostname resolves to
        # (plenty of hosts map their own name to 127.0.0.1). Connecting a UDP
        # socket sends nothing; it only asks the kernel which source address it
        # would pick, so this needs a route rather than a reachable peer.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1: reserved, never routed
            names.add(probe.getsockname()[0])
    except OSError:
        pass
    names.update(extra)
    return sorted(n for n in names if n)


def _san_arg(names: List[str]) -> str:
    parts = []
    for name in names:
        try:
            ipaddress.ip_address(name)
            parts.append(f"IP:{name}")
        except ValueError:
            parts.append(f"DNS:{name}")
    return "subjectAltName=" + ",".join(parts)


def ensure_self_signed(names: List[str]) -> tuple[str, str]:
    """Reuse, or generate, a self-signed pair covering ``names``.

    Cached under ``~/.cache/oasr/web-demo`` rather than made fresh per start, and
    that is the point of the flag: a browser pins its "proceed anyway" exception
    to one certificate, so a new certificate on every start would mean clicking
    through the warning on every start.  Regenerated only when it is missing,
    within a week of expiry, or no longer covers the names being served.
    """
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError(
            "--tls-self-signed needs the `openssl` command on PATH; pass "
            "--ssl-certfile/--ssl-keyfile with your own pair instead"
        )
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    cert, key, stamp = TLS_DIR / "cert.pem", TLS_DIR / "key.pem", TLS_DIR / "sans.txt"
    san = _san_arg(names)
    covered = stamp.is_file() and stamp.read_text() == san
    unexpired = (
        cert.is_file()
        and not subprocess.run(
            [openssl, "x509", "-in", str(cert), "-noout", "-checkend", "604800"],
            capture_output=True,
        ).returncode
    )

    if not (covered and unexpired and key.is_file()):
        # The CN is legacy; browsers read subjectAltName, which is why a name the
        # user actually types has to be in there.
        primary = next(
            (n for n in names if n not in ("localhost", "127.0.0.1", "::1")), "localhost"
        )
        base = [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-days", "825",
            "-nodes", "-keyout", str(key), "-out", str(cert), "-subj", f"/CN={primary}",
        ]  # fmt: skip
        done = subprocess.run(base + ["-addext", san], capture_output=True, text=True)
        if done.returncode:
            # -addext wants OpenSSL >= 1.1.1. Without it the cert has a CN only,
            # which still serves TLS; it just adds a name-mismatch click.
            LOG.debug("openssl -addext rejected (%s); retrying without SANs", done.stderr.strip())
            done = subprocess.run(base, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"openssl failed to generate a certificate: {done.stderr.strip()}")
        stamp.write_text(san)
        key.chmod(0o600)
        LOG.info("generated a self-signed certificate in %s", TLS_DIR)
    else:
        LOG.info("reusing the self-signed certificate in %s", TLS_DIR)

    shown = subprocess.run(
        [openssl, "x509", "-in", str(cert), "-noout", "-fingerprint", "-sha256"],
        capture_output=True,
        text=True,
    )
    if not shown.returncode:
        LOG.info("  %s", shown.stdout.strip())
    LOG.info("  browsers will warn once — accept it, and the microphone works")
    return str(cert), str(key)


def warn_insecure_context(port: int) -> None:
    """Explain the secure-context gate, and every way around it."""
    LOG.warning(
        "the microphone will be unavailable: a plain-HTTP page on a non-loopback "
        "address is not a secure context, and no server-side flag or header can "
        "change that — the browser decides. File upload works regardless. Options:"
    )
    LOG.warning("  easiest      restart with --tls-self-signed, accept the warning once")
    LOG.warning("  no TLS       tell one browser to trust this origin:")
    LOG.warning(
        "               Chrome/Edge  chrome://flags/#unsafely-treat-insecure-origin-as-secure"
    )
    LOG.warning("                            → add the full origin, scheme and port included")
    LOG.warning("               Firefox      about:config → media.devices.insecure.enabled = true")
    LOG.warning("                            and media.getusermedia.insecure.enabled = true")
    LOG.warning("               Safari       no equivalent; use --tls-self-signed")
    LOG.warning(
        "  or           reach the page on loopback: ssh -L %d:localhost:%d <host>", port, port
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Serve the OASR web demo and proxy its API calls to an oasr-server.",
        epilog="example: python examples/web/server.py --oasr-server http://gpu-box:8080",
    )
    p.add_argument(
        "--oasr-server",
        default=os.environ.get("OASR_SERVER_URL", "http://127.0.0.1:8080"),
        metavar="URL",
        help="base URL of a running oasr-server's HTTP listener, or "
        "$OASR_SERVER_URL (default: %(default)s)",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host (default: %(default)s)")
    p.add_argument("--port", type=int, default=8000, help="bind port (default: %(default)s)")
    p.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="relay log verbosity; debug adds static-asset requests, upstream "
        "response headers and relay teardown reasons (default: %(default)s)",
    )
    p.add_argument("--log-file", default=None, help="also append logs to this file")
    p.add_argument(
        "--tls-self-signed",
        action="store_true",
        help="serve HTTPS with a self-signed certificate generated and cached in "
        f"{TLS_DIR} (needs `openssl` on PATH). The one-flag way to get the "
        "microphone working on a remote host: browsers only expose it in a secure "
        "context. They warn once about the certificate; accepting it is remembered.",
    )
    p.add_argument(
        "--tls-san",
        action="append",
        default=[],
        metavar="NAME",
        help="extra hostname or IP for the generated certificate to cover; repeatable. "
        "The bind address, this machine's hostname and its local IPs are included "
        "already — add anything else users will type in the address bar.",
    )
    p.add_argument(
        "--ssl-certfile",
        default=None,
        help="TLS cert (PEM) to serve the page over HTTPS, instead of --tls-self-signed. "
        "Independent of the upstream either way: this relay may serve HTTPS while "
        "talking plain HTTP to oasr-server.",
    )
    p.add_argument("--ssl-keyfile", default=None, help="TLS private key paired with the cert")
    args = p.parse_args(argv)

    setup_logging(args.log_level, args.log_file)

    upstream = urlsplit(args.oasr_server.rstrip("/"))
    if upstream.scheme not in ("http", "https") or not upstream.hostname:
        p.error(f"--oasr-server must be an http(s) URL with a host, got {args.oasr_server!r}")
    if args.tls_self_signed and args.ssl_certfile:
        p.error("--tls-self-signed and --ssl-certfile are alternatives; pass one")
    if args.ssl_certfile and not args.ssl_keyfile:
        p.error("--ssl-certfile also needs --ssl-keyfile")
    if args.tls_san and not args.tls_self_signed:
        p.error("--tls-san only applies to the certificate --tls-self-signed generates")
    Handler.upstream = upstream

    if not STATIC_DIR.is_dir():
        LOG.error("static dir not found: %s", STATIC_DIR)
        return 1

    handler = partial(Handler, directory=str(STATIC_DIR))
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        LOG.error("cannot bind %s:%d: %s", args.host, args.port, exc)
        return 1

    certfile, keyfile = args.ssl_certfile, args.ssl_keyfile
    if args.tls_self_signed:
        try:
            certfile, keyfile = ensure_self_signed(_tls_names(args.host, args.tls_san))
        except RuntimeError as exc:
            LOG.error("%s", exc)
            return 1
    scheme = "http"
    if certfile:
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(certfile, keyfile)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    wildcard = args.host in ("0.0.0.0", "::")
    shown = "localhost" if wildcard or args.host == "127.0.0.1" else args.host
    LOG.info(
        "OASR web demo on %s://%s:%d%s",
        scheme,
        shown,
        args.port,
        f" (bound to {args.host} — every interface)" if wildcard else "",
    )
    routed = ", ".join(p_ + "*" if p_.endswith("/") else p_ for p_ in PROXY_PREFIXES)
    LOG.info("  %s → %s", routed, upstream.geturl())
    LOG.info("  same origin: oasr-server needs no --cors-allow-origin")
    if scheme == "http" and args.host not in ("127.0.0.1", "localhost", "::1"):
        warn_insecure_context(args.port)
    with httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            LOG.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
