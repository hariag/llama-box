#!/usr/bin/env python3
"""Validate chat cache reuse for synthetic tool history against the rendered token prefix.

Use a single-slot loopback server with --log-verbosity 3 and pass its log
path. Run the same fixture once with --no-cache-prompt to build a reference,
then pass that JSONL file to the candidate with --reference-jsonl.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError

from benchmark_long_context import emit, open_api, shared_prefix, tokenize


TOOL_NAME = "count_open_followups"
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Read a count from a synthetic record.",
        "parameters": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["pending"]}},
            "required": ["status"],
            "additionalProperties": False,
        },
    },
}


def rendered_prompt(path, offset, request_id):
    marker = f"rid {request_id} | formatted prompt\n"
    end = re.compile(r"\n\d+\.\d{2}\.\d{3}\.\d{3} [A-Z] ")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with path.open("rb") as log:
            log.seek(offset)
            chunk = log.read().decode("utf-8")
        _, found, contents = chunk.partition(marker)
        boundary = end.search(contents) if found else None
        if boundary:
            return contents[:boundary.start()]
        time.sleep(0.05)
    raise RuntimeError("formatted prompt missing from the isolated server log")


def chat(args, messages, tools):
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "samplers": ["temperature"],
        "seed": 27041,
        "stream": False,
        "max_tokens": 256,
        "parallel_tool_calls": False,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    if tools:
        payload.update(tools=tools, tool_choice="auto")
    offset = args.server_log.stat().st_size
    with open_api(args, "/v1/chat/completions", payload) as response:
        request_id = response.headers["X-Request-ID"]
        result = json.load(response)
    tokens = tokenize(args, rendered_prompt(args.server_log, offset, request_id), add_special=True)
    usage = result["usage"]
    if usage["prompt_tokens"] != len(tokens):
        raise RuntimeError("retokenized server prompt differs from reported prompt usage")
    answer = result["choices"][0]
    message = answer["message"]
    tool_calls = message.get("tool_calls") or []
    output = {
        "finish": answer["finish_reason"],
        "text": (message.get("content") or "").strip(),
        "tools": [{
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"]),
        } for call in tool_calls],
    }
    if any(not call.get("id") for call in tool_calls):
        raise RuntimeError("tool call is missing an ID")
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    return tokens, cached, output


def run(args):
    with open_api(args, "/v1/models") as response:
        metadata = next(model["meta"] for model in json.load(response)["data"] if model["id"] == args.model)
    if metadata["n_slot"] != 1:
        raise RuntimeError("chat cache validation requires an isolated single-slot instance")
    if args.expect_cache and not metadata["prompt_cache"]:
        raise RuntimeError("the candidate reports prompt caching disabled")

    reference = None
    if args.reference_jsonl:
        reference = {}
        reference_cache_disabled = False
        for line in args.reference_jsonl.read_text().splitlines():
            row = json.loads(line)
            if row.get("record") == "chat_run":
                reference_cache_disabled = row.get("prompt_cache") is False
            if row.get("record") == "chat_request" and row["cached_tokens"] == 0 and not row["errors"]:
                reference[row["case"]] = row
        if not reference_cache_disabled:
            raise RuntimeError("chat reference JSONL must come from a server with prompt caching disabled")
    emit("chat_run", model=args.model, prompt_cache=metadata["prompt_cache"],
         thinking=args.thinking, filler_repeats=args.filler_repeats)

    instruction = ("Ignore these garden notes. " + " The oak trees shade a pond." * args.filler_repeats +
                   "\nCall count_open_followups with status pending. When the result arrives, output only its number.")
    start = [{"role": "user", "content": instruction}]
    followed = [
        *start,
        {"role": "assistant", "content": "", "reasoning_content": "I need to query the count before answering.",
         "tool_calls": [{"id": "call_synthetic_0", "type": "function",
                         "function": {"name": TOOL_NAME, "arguments": '{"status":"pending"}'}}]},
        {"role": "tool", "tool_call_id": "call_synthetic_0", "content": "73"},
    ]
    edited_result = [*followed[:-1], {**followed[-1], "content": "47"}]
    reset = [{"role": "user", "content": "Start a new conversation. Reply with only the integer 19."}]
    tool_output = {"finish": "tool_calls", "text": "",
                   "tools": [{"name": TOOL_NAME, "arguments": {"status": "pending"}}]}
    final_output = {"finish": "stop", "text": "73", "tools": []}
    cases = [
        ("tool-first", start, [TOOL], tool_output),
        ("tool-exact", start, [TOOL], tool_output),
        ("tool-result", followed, [TOOL], final_output),
        ("tool-result-exact", followed, [TOOL], final_output),
        ("tool-result-edited", edited_result, [TOOL], {**final_output, "text": "47"}),
        ("tool-result-edited-exact", edited_result, [TOOL], {**final_output, "text": "47"}),
        ("request-reset", reset, [], {"finish": "stop", "text": "19", "tools": []}),
    ]
    previous = None
    failures = []
    for case, messages, tools, expected in cases:
        tokens, cached, output = chat(args, messages, tools)
        lcp = shared_prefix(previous, tokens) if previous is not None else None
        signature = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        errors = []
        if not isinstance(cached, int) or cached < 0 or cached >= len(tokens) or (lcp is not None and cached > lcp):
            errors.append("reported reuse exceeds the actual rendered token prefix")
        elif args.expect_cache and case.endswith("-exact") and cached != len(tokens) - 1:
            errors.append("exact rendered history was not reused")
        elif args.expect_cache and case in ("tool-result", "tool-result-edited"):
            # A short append can retain the previous prompt's near-tail anchor
            # instead of copying another nearby checkpoint for the new prompt.
            tail_replay = 256 if case.endswith("-edited") else 128
            if lcp > tail_replay and cached < lcp - tail_replay:
                errors.append("tool continuation missed its recent shared prefix")
        visible = output
        if args.model.startswith("gemma4"):
            # Normalize the empty reasoning wrapper for the answer assertion.
            # The no-cache comparison below still checks the raw output.
            text = re.sub(r"^<\|channel>thought\s*<channel\|>", "", output["text"]).strip()
            visible = {**output, "text": text}
        if visible != expected:
            errors.append("tool call or visible answer differs from the synthetic assertion")
        if reference is not None:
            baseline = reference.get(case, {})
            if baseline.get("prompt_sha256") != signature or baseline.get("output") != output:
                errors.append("prompt or visible output differs from the independent cold reference")
        emit("chat_request", case=case, input_tokens=len(tokens), cached_tokens=cached,
             previous_lcp_tokens=lcp, prompt_sha256=signature, output=output, errors=errors)
        failures.extend(f"{case} {error}" for error in errors)
        previous = tokens
    emit("finished", ok=not failures, errors=failures)
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="base URL without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--expect-cache", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--filler-repeats", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    if args.filler_repeats < 1 or args.timeout <= 0:
        parser.error("filler repeats and timeout must be positive")
    try:
        return run(args)
    except HTTPError as error:
        emit("fatal", error=f"HTTP {error.code}: {error.read(400).decode(errors='replace')}")
    except (OSError, ValueError, RuntimeError, StopIteration, KeyError) as error:
        emit("fatal", error=str(error))
    return 1


if __name__ == "__main__":
    sys.exit(main())
