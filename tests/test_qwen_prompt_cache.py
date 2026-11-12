#!/usr/bin/env python3
"""Compare single-slot cached Qwen inference with a no-cache reference.

Start two independent servers with the same model, context and sampling config;
add --no-cache-prompt on reference. For MTP use --spec-type draft-mtp on both.
Run with --candidate URL --reference URL --require-hit; add --require-mtp
when both endpoints use MTP. Use --concurrency with at least two candidate slots.
Both endpoints must serve the same Qwen model with deterministic sampling.
The reference endpoint must have prompt caching disabled.
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--candidate', required=True)
p.add_argument('--reference', required=True)
p.add_argument('--model', help='override model ID reported by candidate')
p.add_argument('--require-hit', action='store_true')
p.add_argument('--concurrency', action='store_true')
p.add_argument('--require-mtp', action='store_true')
p.add_argument('--timeout', type=float, default=240)
p.add_argument('--length', type=int, default=512)
p.add_argument('--max-tokens', type=int, default=12)
a = p.parse_args()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def req(url, path, body=None):
    r = urllib.request.Request(url.rstrip('/') + path, headers={'Content-Type': 'application/json'}, data=None if body is None else json.dumps(body).encode())
    t = time.monotonic()
    try:
        with opener.open(r, timeout=a.timeout) as f:
            return (json.load(f), round(time.monotonic() - t, 3))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:500]}')

def tok(s):
    return req(a.candidate, '/tokenize', {'content': s, 'add_special': False})[0]['tokens']

def lcp(x, y):
    for (i, (a, b)) in enumerate(zip(x, y)):
        if a != b:
            return i
    return min(len(x), len(y))
block = tok('The quick brown fox is quietly reading an old book. ')
head = tok('<|im_start|>user\nThe following is context material.\n')
tail = tok('\nQuestion: What is 2+2? Answer with a single digit.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
assert a.length > len(head) + len(tail) + 10
n = a.length - len(head) - len(tail)
prompt = head + (block * ((n + len(block) - 1) // len(block)))[:n] + tail
alt = tok(' ocean universe random ')[0]

def edit(i):
    x = prompt.copy()
    x[i] = alt if x[i] != alt else block[0]
    return x
branch = edit(max(len(head) + 1, a.length * 3 // 4))
distinct = edit(len(head) + 1)
cases = [('cold', prompt, False, None), ('repeat', prompt, True, None), ('diverge-late', branch, False, None), ('repeat-branch', branch, True, None), ('diverge-early', distinct, False, None), ('repeat-early', distinct, True, None), ('return-original', prompt, False, None), ('repeat-original', prompt, True, None)]
secret_tail = tok('\nRepeat the secret passcode exactly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
secret_fill = tok('Ignore this filler text. ')

def secret_prompt(value):
    secret_head = tok(f'<|im_start|>user\nThe secret passcode is {value}. Remember it.\n')
    m = a.length - len(secret_head) - len(secret_tail)
    assert m > 0
    return secret_head + (secret_fill * ((m + len(secret_fill) - 1) // len(secret_fill)))[:m] + secret_tail
secret_a = secret_prompt('4927')
secret_b = secret_prompt('1358')
cases += [('secret-a', secret_a, False, '4927'), ('repeat-secret-a', secret_a, True, '4927'), ('secret-b', secret_b, False, '1358'), ('repeat-secret-b', secret_b, True, '1358')]
failed = []
last = None
reference_outputs = {}
candidate_model = req(a.candidate, '/v1/models')[0]['data'][0]
reference_model = req(a.reference, '/v1/models')[0]['data'][0]
model_id = a.model or candidate_model['id']
candidate_meta = candidate_model['meta']
reference_meta = reference_model['meta']
print(json.dumps({'candidate_meta': candidate_meta, 'reference_meta': reference_meta}, ensure_ascii=False), flush=True)
if reference_meta.get('prompt_cache') is not False:
    raise SystemExit('Reference must have prompt caching disabled')
if a.require_hit and candidate_meta.get('prompt_cache') is not True:
    raise SystemExit('Candidate must have prompt caching enabled')
for field in ('n_vocab', 'n_params', 'size'):
    if candidate_meta.get(field) != reference_meta.get(field):
        raise SystemExit(f'Model metadata differs for {field}')
for (name, tokens, musthit, expected) in cases:
    body = {'model': model_id, 'prompt': tokens, 'max_tokens': a.max_tokens, 'temperature': 0, 'seed': 42, 'stream': False}
    try:
        (ref, rt) = req(a.reference, '/v1/completions', body)
        (got, ct) = req(a.candidate, '/v1/completions', body)
        (ru, cu) = (ref['usage'], got['usage'])
        cached = cu.get('prompt_tokens_details', {}).get('cached_tokens')
        rtext = ref['choices'][0]['text']
        text = got['choices'][0]['text']
        errors = []
        if ru['prompt_tokens'] != len(tokens) or cu['prompt_tokens'] != len(tokens):
            errors.append('prompt size mismatch')
        if cached is None or cached < 0 or cached > len(tokens):
            errors.append('invalid cached count')
        if candidate_meta.get('n_slot') == 1 and last is not None and (cached is not None) and (cached > lcp(tokens, last)):
            errors.append('cached beyond previous matching prefix (single-slot)')
        if rtext != text:
            errors.append('output differs from no-cache reference')
        if not text.strip():
            errors.append('empty output')
        if expected is not None and (expected not in text or expected not in rtext):
            errors.append('secret recall failed')
        if a.require_hit and musthit and (not cached):
            errors.append('repeat did not hit')
        if a.require_mtp and (not (cu.get('draft_tokens', 0) > 0 and ru.get('draft_tokens', 0) > 0)):
            errors.append('MTP did not draft')
        row = {'case': name, 'tokens': len(tokens), 'lcp_previous': None if last is None else lcp(tokens, last), 'cached': cached, 'candidate_seconds': ct, 'reference_seconds': rt, 'candidate_text': text, 'reference_text': rtext, 'draft_tokens': cu.get('draft_tokens'), 'errors': errors}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        failed.extend((f'{name}: {x}' for x in errors))
        last = tokens
        if expected is not None:
            reference_outputs[expected] = rtext
    except Exception as e:
        failed.append(f'{name}: {e!r}')
        print(json.dumps({'case': name, 'error': repr(e)}), flush=True)
        break
if a.concurrency and (not failed):
    meta = req(a.candidate, '/v1/models')[0]['data'][0]['meta']
    if meta.get('n_slot', 0) < 2:
        failed.append('concurrency test requires at least two candidate slots')
    else:

        def concurrent_request(item):
            (expected, tokens) = item
            body = {'model': model_id, 'prompt': tokens, 'max_tokens': a.max_tokens, 'temperature': 0, 'seed': 42, 'stream': False}
            (result, seconds) = req(a.candidate, '/v1/completions', body)
            return (expected, result, seconds)
        for turn in range(2):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(concurrent_request, [('4927', secret_a), ('1358', secret_b)]))
                for (expected, result, seconds) in results:
                    usage = result['usage']
                    output = result['choices'][0]['text']
                    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens')
                    errors = []
                    if output != reference_outputs[expected] or expected not in output:
                        errors.append('concurrent output differs from reference')
                    if usage['prompt_tokens'] != a.length or cached is None or (not 0 <= cached <= a.length):
                        errors.append('invalid token accounting')
                    if a.require_mtp and (not usage.get('draft_tokens', 0)):
                        errors.append('MTP did not draft')
                    failed.extend((f'concurrent {turn} {expected}: {error}' for error in errors))
                    print(json.dumps({'case': 'concurrent', 'turn': turn, 'secret': expected, 'cached': cached, 'seconds': seconds, 'errors': errors}, ensure_ascii=False), flush=True)
            except Exception as e:
                failed.append(f'concurrent {turn}: {e!r}')
                break
print(json.dumps({'ok': not failed, 'errors': failed}, ensure_ascii=False), flush=True)
sys.exit(bool(failed))
