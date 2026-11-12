#!/usr/bin/env python3
"""Stress shared KV budget on an isolated Qwen server.

Example server: llama-box -m MODEL -c 1024 -np 8 --kv-unified -t 1
  --spec-type draft-mtp --spec-draft-n-max 3 -v > server.log 2>&1
Run: python tests/test_qwen_cache_saturation.py --url http://127.0.0.1:8080
  --server-log server.log
This test fills the server's cache; never run it against a production instance.
"""
import argparse
import concurrent.futures
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import urllib.parse

parser = argparse.ArgumentParser()
parser.add_argument('--url', required=True)
parser.add_argument('--server-log', help='verbose server log used to verify all slots')
parser.add_argument('--timeout', type=float, default=180)
parser.add_argument('--disconnect', action='store_true', help='test cancellation of an admission-waiting non-streaming client')
args = parser.parse_args()
base = args.url.rstrip('/')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
log_offset = 0
if args.server_log:
    with open(args.server_log, 'rb') as log:
        log.seek(0, 2)
        log_offset = log.tell()


def request(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    try:
        with opener.open(req, timeout=args.timeout) as response:
            return response.status, json.load(response), round(time.monotonic() - start, 3)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(), round(time.monotonic() - start, 3)


def tokenize(text):
    status, data, _ = request('/tokenize', {'content': text, 'add_special': False})
    assert status == 200, data
    return data['tokens']

status, models, _ = request('/v1/models')
assert status == 200
model = models['data'][0]
meta = model['meta']
assert meta['n_ctx'] == 1024 and meta['n_slot'] == 8 and meta.get('dynamic_context'), meta
assert meta['prompt_cache'], meta
filler = tokenize('There is an old library with many books. ')
suffix = tokenize('\nGive exactly one number, 2 + 2 = ')
assert filler


def make_prompt(identity, size):
    head = tokenize('Request ' + str(identity) + ' A sequence of records. ')
    n = size - len(head) - len(suffix)
    assert n > 0
    return head + (filler * ((n + len(filler) - 1) // len(filler)))[:n] + suffix


def complete(identity, length, max_tokens, barrier=None):
    prompt = make_prompt(identity, length)
    if barrier is not None:
        barrier.wait(timeout=args.timeout)
    code, data, elapsed = request('/v1/completions', {
        'model': model['id'], 'prompt': prompt, 'max_tokens': max_tokens,
        'temperature': 0, 'ignore_eos': True,
    })
    print(json.dumps({'request': identity, 'status': code, 'seconds': elapsed, 'result': data}), flush=True)
    assert code == 200, (identity, code, data)
    usage = data['usage']
    assert usage['prompt_tokens'] == length, (identity, usage)
    assert data['choices'][0]['finish_reason'] == 'length', (identity, data)
    return usage

barrier = threading.Barrier(8)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(complete, i, 100, 20, barrier) for i in range(8)]
    for f in futures:
        f.result()

if args.server_log:
    with open(args.server_log, 'rb') as log:
        log.seek(log_offset)
        lines = log.read().decode(errors='replace')
    slots = set(re.findall(r'released cache prompt, seq = (\d+)', lines))
    assert len(slots) >= 8, ('test did not fill all 8 cache slots', slots)

# This fits on its own but requires the server to reclaim many inactive slots.
complete('large', 900, 16)
if args.server_log:
    with open(args.server_log, 'rb') as log:
        log.seek(log_offset)
        lines = log.read().decode(errors='replace')
    evicted = set(re.findall(r'evicted idle hybrid cache, seq = (\d+)', lines))
    assert len(evicted) >= 4, ('expected at least four idle slots evicted', evicted)
complete('recovery', 50, 8)
usage = complete('recovery', 50, 8)
assert usage['prompt_tokens_details']['cached_tokens'] == 49, usage
# With 8 requests, each 100 prompt tokens and 40 requested output tokens,
# the 128-token initial share cannot hold all requested outputs. Verify that
# every request terminates and some output is bounded by the shared quota.
barrier = threading.Barrier(8)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(complete, i + 1000, 100, 40, barrier) for i in range(8)]
    active = [f.result() for f in futures]
assert any(u['completion_tokens'] < 40 for u in active), 'did not reach dynamic output budget'
complete('post-budget', 50, 8)

if args.disconnect:
    assert args.server_log, '--disconnect requires --server-log'
    target = urllib.parse.urlparse(base)
    assert target.scheme == 'http', 'disconnect test requires HTTP'
    payload = json.dumps({'model': model['id'], 'prompt': make_prompt('disconnected', 400),
                          'max_tokens': 40, 'temperature': 0, 'ignore_eos': True}).encode()
    with open(args.server_log, 'rb') as log:
        log.seek(0, 2)
        cancel_offset = log.tell()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        long_request = pool.submit(complete, 'long-for-cancel', 700, 40)
        time.sleep(0.5)
        with socket.create_connection((target.hostname, target.port or 80), timeout=args.timeout) as client:
            client.sendall(b'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n'
                           b'Content-Type: application/json\r\nContent-Length: ' +
                           str(len(payload)).encode() + b'\r\n\r\n' + payload)
            deadline = time.monotonic() + args.timeout
            request_id = None
            while time.monotonic() < deadline:
                with open(args.server_log, 'rb') as log:
                    log.seek(cancel_offset)
                    lines = log.read().decode(errors='replace')
                match = re.search(r'rid (\S+) \| batching, waiting dynamic context admission', lines)
                if match:
                    request_id = match.group(1)
                    break
                time.sleep(0.1)
            assert request_id, 'did not observe request waiting for admission'
        long_request.result()
    with open(args.server_log, 'rb') as log:
        log.seek(cancel_offset)
        lines = log.read().decode(errors='replace')
    assert f'rid {request_id} | cancelled while waiting for dynamic context admission' in lines, lines[-5000:]
    assert f'rid {request_id} | prefix cache' not in lines, 'disconnected request performed prefill'
    complete('post-disconnect', 50, 8)

print('PASS: concurrent quota, multi-slot eviction and post-pressure requests')
