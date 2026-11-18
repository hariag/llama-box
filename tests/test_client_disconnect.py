#!/usr/bin/env python3
"""Verify cancelled requests release every HTTP worker on an isolated Linux server.

Start llama-box with a small model, --parallel 2, --ctx-size 4096,
--tokens-per-second 25, and --log-verbosity 2. Redirect its logs to a file.
This test cancels one request per worker and verifies that new requests complete.
"""

import argparse
from contextlib import closing
import http.client
import json
from pathlib import Path
import re
import socket
import struct
import time
from urllib.parse import urlsplit


def post(target, payload, timeout):
    with closing(http.client.HTTPConnection(target.hostname, target.port, timeout=timeout)) as conn:
        conn.request("POST", "/v1/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read()
        assert response.status == 200, (response.status, body)
        return body


def wait_for_log(log_path, offset, pattern, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with log_path.open("rb") as log:
            log.seek(offset)
            match = re.search(pattern, log.read().decode(errors="replace"))
            if match:
                return match
        time.sleep(0.05)
    raise TimeoutError(f"server log did not match {pattern!r}")


def abort_request(target, payload, log_path, timeout):
    log_offset = log_path.stat().st_size
    body = json.dumps(payload).encode()
    with socket.create_connection((target.hostname, target.port), timeout=timeout) as sock:
        headers = (f"POST /v1/completions HTTP/1.1\r\nHost: {target.netloc}\r\n"
                   "Content-Type: application/json\r\nX-Request-Tokens-Per-Second: 25\r\n"
                   f"Content-Length: {len(body)}\r\n\r\n")
        sock.sendall(headers.encode() + body)
        if payload["stream"]:
            received = bytearray()
            while b"data:" not in received:
                part = sock.recv(4096)
                if not part:
                    raise RuntimeError("stream ended before its first SSE frame")
                received.extend(part)
        else:
            wait_for_log(log_path, log_offset, r"decode_budget = 1000", timeout)
            time.sleep(0.4)
        # Reset the TCP connection while the model still has tokens to produce.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    wait_for_log(log_path, log_offset, r"decode_t = [1-9]\d*, [^\n]+stop_r = closed", timeout)


def check_followup(target, model, stream, timeout):
    payload = {"model": model, "prompt": "The opposite of hot is", "max_tokens": 8,
               "temperature": 0, "ignore_eos": True, "stream": stream}
    body = post(target, payload, timeout)
    if stream:
        frames = [line[6:] for line in body.splitlines() if line.startswith(b"data: ")]
        assert frames and frames[-1] == b"[DONE]", body
        records = [json.loads(frame) for frame in frames[:-1]]
        assert len({record["id"] for record in records}) == 1, records
        assert records[-1]["choices"][0]["finish_reason"] == "length", records
    else:
        record = json.loads(body)
        choice = record["choices"][0]
        assert choice["text"].strip() and choice["finish_reason"] == "length", record
        assert record["usage"]["completion_tokens"] == 8, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--mode", choices=("stream", "non-stream", "both"), default="both")
    args = parser.parse_args()
    target = urlsplit(args.url)
    if target.scheme != "http" or target.hostname not in ("localhost", "127.0.0.1", "::1") or not target.port:
        parser.error("use a local HTTP server with an explicit port")
    if args.workers <= 0:
        parser.error("--workers must match the isolated server's positive --parallel setting")

    for mode in ("non-stream", "stream"):
        if args.mode not in (mode, "both"):
            continue
        stream = mode == "stream"
        check_followup(target, args.model, stream, args.timeout)
        for index in range(args.workers):
            payload = {"model": args.model, "prompt": f"{index + 1}. A long list of facts about trees, each described:",
                       "max_tokens": 1000, "ignore_eos": True, "stream": stream}
            abort_request(target, payload, args.server_log, args.timeout)
        check_followup(target, args.model, stream, args.timeout)
        print(f"PASS: {mode} requests complete after {args.workers} client disconnects", flush=True)


if __name__ == "__main__":
    main()
