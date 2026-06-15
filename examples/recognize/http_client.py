#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Minimal HTTP client for the OASR `oasr.speech.v1` Speech service.

Sends a single audio file to ``POST /v1/speech:recognize`` and prints the
JSON response.  Requires the server to be running in ``--service-mode
offline``.

The request **body is the raw audio payload** (no base64, no JSON envelope) and
the config rides in the query string — this skips the ~33% base64 inflation +
JSON build/parse (~2× the throughput of a base64-JSON body under load).  The
response is a small JSON ``RecognizeResponse``.

Dependencies::

    pip install requests

Usage::

    # WAV container (server reads the embedded sample rate)
    python examples/recognize/http_client.py \\
        --server-url http://127.0.0.1:8080 \\
        --wav tests/fixtures/hello.wav

    # Headerless raw PCM
    python examples/recognize/http_client.py \\
        --server-url http://127.0.0.1:8080 \\
        --wav audio.f32 --encoding LINEAR32F --sample-rate 16000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-url", default="http://127.0.0.1:8080",
                   help="Base URL of the oasr-server HTTP listener")
    p.add_argument("--wav", required=True, type=Path,
                   help="Path to the audio file to transcribe")
    p.add_argument("--encoding", default="WAV",
                   choices=("WAV", "LINEAR16", "LINEAR32F"),
                   help="Audio encoding sent in the `encoding` query parameter "
                        "(default: WAV; sample rate is taken from the header)")
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="Sample rate for raw PCM payloads "
                        "(ignored when encoding=WAV)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="HTTP request timeout in seconds")
    args = p.parse_args(argv)

    if not args.wav.is_file():
        print(f"audio file not found: {args.wav}", file=sys.stderr)
        return 1

    url = f"{args.server_url.rstrip('/')}/v1/speech:recognize"
    # Body is the raw audio bytes; recognition config travels in the query string.
    resp = requests.post(
        url,
        params={"encoding": args.encoding, "sample_rate": args.sample_rate},
        data=args.wav.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=args.timeout,
    )
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        return 2

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # Highlight the top transcript for convenience.
    for result in data.get("results", []):
        for alt in result.get("alternatives", []):
            transcript = alt.get("transcript", "")
            if transcript:
                print(f"\ntranscript: {transcript}")
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
