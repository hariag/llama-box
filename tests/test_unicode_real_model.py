#!/usr/bin/env python3
"""Exercise Unicode streaming and stops against an isolated llama-box server.

Example: python3 test_utf8_sse_regression.py --server http://127.0.0.1:19682
The server must run a model supporting these literal grammar outputs.  This
test makes real inference requests; do not point it at a production instance.
"""
import argparse
import json
import urllib.request


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body, ensure_ascii=False).encode(),
                                 {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=120) as response:
        raw = response.read().decode("utf-8", errors="strict")
    if body["stream"]:
        lines = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        assert lines and lines[-1] == "[DONE]", raw
        records = [json.loads(line) for line in lines[:-1]]
    else:
        records = [json.loads(raw)]
    choices = [record["choices"][0] for record in records if record.get("choices")]
    chat = "/chat/" in url
    pieces = [(choice.get("delta", {}) if body["stream"] else choice.get("message", {})).get("content") or ""
              if chat else choice.get("text", "") for choice in choices]
    # Each response must itself contain valid UTF-8; a replacement character
    # here cannot be repaired by concatenating subsequent chunks.
    assert all("\ufffd" not in piece for piece in pieces), pieces
    return "".join(pieces), choices[-1]["finish_reason"], choices


def complete(server, expected, *, stream, stop=None, limit=64, chat=False, model=None, **extra):
    body = {"temperature": 0, "max_tokens": limit, "stream": stream,
            "samplers": ["temperature"], "grammar": "root ::= " + json.dumps(expected, ensure_ascii=False)}
    if chat:
        body["messages"] = [{"role": "user", "content": "x"}]
    else:
        body["prompt"] = "x"
    if stop is not None:
        body["stop"] = stop
    if model is not None:
        body["model"] = model
    body.update(extra)
    return post(server + ("/v1/chat/completions" if chat else "/v1/completions"), body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="isolated server base URL")
    parser.add_argument("--model", help="optional model name to send in requests")
    parser.add_argument("--qwen-byte-logprobs", action="store_true",
                        help="also test byte-level logprobs using Qwen3.5 token ID 169286")
    args = parser.parse_args()
    server = args.server.rstrip("/")

    for expected in ["✅", "🦙", "🚀", "𠮷", "a✅b🦙c", "中文✅字", "🧑🏽‍💻"]:
        for chat in [False, True]:
            for stream in [False, True]:
                actual, reason, _ = complete(server, expected, stream=stream, chat=chat, model=args.model)
                assert (actual, reason) == (expected, "stop"), (expected, chat, stream, actual, reason)
        print("PASS unicode", expected, flush=True)

    for stop, expected in [("✅", "abc"), ("end", "abc✅"), ("✅end", "abc"), ("bc", "a"),
                           (["end", "✅end"], "abc"), (["✅end", "end"], "abc")]:
        for chat in [False, True]:
            for stream in [False, True]:
                actual, reason, _ = complete(server, "abc✅end", stream=stream, chat=chat, stop=stop, model=args.model)
                assert (actual, reason) == (expected, "stop"), (stop, chat, stream, actual, reason)
        print("PASS stop", stop, flush=True)

    # Tokenization differs among models. A truncated literal may be either a
    # proper Unicode prefix or empty, but never corrupted or differ by mode.
    for limit in range(1, 7):
        a, ar, _ = complete(server, "✅", stream=False, limit=limit, model=args.model)
        b, br, _ = complete(server, "✅", stream=True, limit=limit, model=args.model)
        assert a == b and ar == br and "✅".startswith(a), (limit, a, ar, b, br)
        print("PASS length", limit, repr(a), ar, flush=True)

    if args.qwen_byte_logprobs:
        for stream in [False, True]:
            text, _, choices = complete(server, "✅", stream=stream, chat=True,
                                        model=args.model, logprobs=True, top_logprobs=5,
                                        logit_bias={"169286": -100})
            entries = [entry for choice in choices for entry in
                       (choice.get("logprobs") or {}).get("content", [])]
            byte_values = [entry["bytes"] for entry in entries]
            assert text == "✅" and byte_values[:3] == [[226], [156], [133]], (text, byte_values)
            tops = [top for entry in entries for top in entry["top_logprobs"]]
            partial = [top for top in tops if top["token"].startswith("byte: ")]
            assert partial, tops
            for top in partial:
                assert top["bytes"] == [int(top["token"].rsplit("x", 1)[1], 16)], top
            print("PASS byte logprobs", stream, flush=True)


if __name__ == "__main__":
    main()
