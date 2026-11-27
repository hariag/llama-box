#!/usr/bin/env python3
"""Measure long-context performance and cache reuse on one isolated llama-box slot.

Run --cases cold while tuning batch sizes. Use --expect-cache for regression
checks. For cold-output comparison of each variant, run the same command
against a server started with --no-cache-prompt, capture its JSONL output, and
pass that file to the candidate with --reference-jsonl.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def emit(record, **fields):
    print(json.dumps({"record": record, **fields}, ensure_ascii=False), flush=True)


def mem_available_mib():
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def monitor_memory(done, samples):
    while not done.wait(0.5):
        samples.append(mem_available_mib())


def open_api(args, path, payload=None):
    data = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    request = Request(args.url + path, data=data, headers={"Content-Type": "application/json"})
    return build_opener(ProxyHandler({})).open(request, timeout=args.timeout)


def tokenize(args, content, add_special=False):
    with open_api(args, "/tokenize", {"content": content, "add_special": add_special}) as response:
        return json.load(response)["tokens"]


def expand(tokens, count):
    if not tokens or count < 0:
        raise ValueError("token sequence cannot fill the requested prompt")
    return (tokens * ((count + len(tokens) - 1) // len(tokens)))[:count]


def fixtures(args):
    document = "The following notes describe a botanical garden in four seasons.\n"
    head = tokenize(args, f"{args.seed} | " + document, add_special=True)
    paragraph = tokenize(args, " Oak leaves fell along the shaded path. The mist cleared above a pond. "
                               " Gardeners observed migrating birds and marked each sighting.")
    tail = tokenize(args, "\nContinue the botanical field notes with several numbered paragraphs.\n1.")
    prompt = head + expand(paragraph, args.input_tokens - len(head) - len(tail)) + tail
    extension = tokenize(args, "\nAdditional notes on the diversity of flowering plants. ")
    appended = prompt + expand(extension, args.append_tokens)
    alternatives = tokenize(args, " Compare another unrelated record of winter rainfall.")
    edit_at = len(appended) - args.diverge_tokens
    divergent = appended.copy()
    divergent[edit_at] = next(token for token in alternatives if token != appended[edit_at])
    shorter = divergent[:-max(2, args.append_tokens // 2)]
    unrelated = expand(alternatives, min(args.input_tokens, 512))
    unrelated[0] = next(token for token in alternatives if token != prompt[0])
    return {
        "cold": prompt,
        "exact": prompt,
        "append": appended,
        "return_to_original": prompt,
        "diverge": divergent,
        "truncate": shorter,
        "unrelated": unrelated,
    }


def shared_prefix(previous, current):
    for index, (left, right) in enumerate(zip(previous, current)):
        if left != right:
            return index
    return min(len(previous), len(current))


def completion(args, prompt, output_tokens):
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "samplers": ["temperature"],
        "seed": args.seed,
        "max_tokens": output_tokens,
        "ignore_eos": True,
    }
    start = time.monotonic()
    text = []
    usage = None
    first = None
    finish = None
    done = False
    with open_api(args, "/v1/completions", payload) as response:
        for raw in response:
            if raw.startswith(b"error:"):
                raise RuntimeError(raw[6:].decode().strip())
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
                if fragment := choice.get("text"):
                    if first is None:
                        first = time.monotonic()
                    text.append(fragment)
    elapsed = (time.monotonic() - start) * 1_000
    if not done or not usage or first is None:
        raise RuntimeError("missing SSE completion, terminal usage, or output text")
    output = "".join(text)
    return {
        **({"output": output} if args.capture_output else {}),
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "draft_tokens": usage.get("draft_tokens"),
        "draft_acceptance": usage.get("draft_tokens_acceptance"),
        "client_ttft_ms": (first - start) * 1_000,
        "server_ttft_ms": usage.get("time_to_first_token_ms"),
        "server_reported_prefill_tok_s": usage.get("prompt_tokens_per_second"),
        "server_decode_tok_s": usage.get("tokens_per_second"),
        "elapsed_ms": elapsed,
        "finish_reason": finish,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "unicode_replacements": output.count("\ufffd"),
    }


def verify(args, case, row, expected, original_hash, reference):
    errors = []
    cached = row["cached_tokens"]
    lcp = row["previous_lcp_tokens"]
    if row["prompt_tokens"] != len(expected):
        errors.append("server did not process the exact input token count")
    if row["completion_tokens"] != args.output_tokens or row["finish_reason"] != "length":
        errors.append("generation did not consume the output budget")
    if not isinstance(cached, int) or not 0 <= cached <= lcp:
        errors.append("reported cache reuse exceeds the verified prefix")
    elif args.expect_cache:
        if (case in ("exact", "append", "return_to_original") or
            (case == "diverge" and args.diverge_tokens == 1)) and cached < lcp - 1:
            errors.append("near-complete prefix reuse is missing")
        if case == "truncate" and cached < max(1, min(lcp, args.input_tokens) - 129):
            errors.append("truncation missed the retained checkpoint near the original prompt tail")
    if case in ("exact", "return_to_original") and row["output_sha256"] != original_hash:
        errors.append("cached output for the original prompt differs from cold output")
    if args.require_mtp and not row["draft_tokens"]:
        errors.append("MTP submitted no speculative tokens")
    if row["unicode_replacements"]:
        errors.append("output contains replacement characters")
    if reference is not None:
        expected_row = reference.get(case, {})
        if (expected_row.get("prompt_sha256") != row["prompt_sha256"]
                or expected_row.get("completion_tokens") != row["completion_tokens"]):
            errors.append("missing matching reference fixture")
        elif expected_row["output_sha256"] != row["output_sha256"]:
            errors.append("output differs from separately measured cold reference")
    return errors


def workload(args):
    with open_api(args, "/v1/models") as response:
        models = json.load(response)["data"]
    metadata = next(model["meta"] for model in models if model["id"] == args.model)
    if metadata["n_slot"] != 1:
        raise RuntimeError("use an isolated one-slot instance to validate cache provenance")
    if args.expect_cache and not metadata["prompt_cache"]:
        raise RuntimeError("the candidate reports prompt caching disabled")
    if args.reference_jsonl:
        reference = {}
        reference_cache_disabled = False
        for line in args.reference_jsonl.read_text().splitlines():
            row = json.loads(line)
            if row.get("record") == "run":
                reference_cache_disabled = row.get("prompt_cache") is False
            if row.get("record") == "request" and row["cached_tokens"] == 0 and not row["errors"]:
                reference[row["case"]] = row
        if not reference_cache_disabled:
            raise RuntimeError("reference JSONL must come from a server with prompt caching disabled")
    else:
        reference = None

    case_prompts = fixtures(args)
    if args.cases == "cold":
        case_prompts = {"cold": case_prompts["cold"]}
    if max(map(len, case_prompts.values())) + args.output_tokens >= metadata["n_ctx_single"]:
        raise RuntimeError("server context must leave room for all input and output tokens")
    emit("run", model=args.model, cases=args.cases, input_tokens=args.input_tokens,
         output_tokens=args.output_tokens, context=metadata["n_ctx_single"],
         prompt_cache=metadata["prompt_cache"], seed=args.seed, diverge_tokens=args.diverge_tokens)

    warmup = tokenize(args, "Warmup request for the isolated benchmark instance.")
    first = case_prompts["cold"][0]
    warmup[0] = next(token for token in warmup if token != first)
    completion(args, warmup, 8)
    previous = warmup
    cold_hash = None
    failures = []
    samples = [mem_available_mib()]
    done = threading.Event()
    sampler = threading.Thread(target=monitor_memory, args=(done, samples))
    sampler.start()
    try:
        for case, prompt in case_prompts.items():
            row = completion(args, prompt, args.output_tokens)
            row["prompt_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            row["previous_lcp_tokens"] = shared_prefix(previous, prompt)
            if isinstance(row["cached_tokens"], int):
                row["uncached_prompt_tokens"] = row["prompt_tokens"] - row["cached_tokens"]
            errors = verify(args, case, row, prompt, cold_hash, reference)
            emit("request", case=case, **row, errors=errors)
            if case == "cold":
                cold_hash = row["output_sha256"]
            failures.extend(f"{case} {error}" for error in errors)
            previous = prompt
    finally:
        done.set()
        sampler.join()
        samples.append(mem_available_mib())
        emit("memory", before_mib=samples[0], minimum_mib=min(samples),
             median_mib=statistics.median(samples), after_mib=samples[-1], samples=len(samples))
    if args.min_mem_available_mib and min(samples) < args.min_mem_available_mib:
        failures.append("measured MemAvailable fell below the requested floor")
    emit("finished", ok=not failures, errors=failures)
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="base URL without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--append-tokens", type=int, default=128)
    parser.add_argument("--diverge-tokens", type=int, default=1, help="distance of the edited token from the end")
    parser.add_argument("--seed", type=int, default=27111)
    parser.add_argument("--cases", choices=("cold", "cache"), default="cache")
    parser.add_argument("--expect-cache", action="store_true")
    parser.add_argument("--require-mtp", action="store_true")
    parser.add_argument("--capture-output", action="store_true", help="retain synthetic output for diagnostics")
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--min-mem-available-mib", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    if args.input_tokens < 128 or args.output_tokens < 1 or args.append_tokens < 4 or args.timeout <= 0:
        parser.error("input must be at least 128 tokens and append budget at least 4 tokens")
    if not 1 <= args.diverge_tokens < args.input_tokens + args.append_tokens:
        parser.error("divergence must edit a token after the first one")
    try:
        return workload(args)
    except HTTPError as error:
        emit("fatal", error=f"HTTP {error.code}: {error.read(400).decode(errors='replace')}")
    except (OSError, ValueError, RuntimeError, StopIteration, KeyError) as error:
        emit("fatal", error=str(error))
    return 1


if __name__ == "__main__":
    sys.exit(main())
