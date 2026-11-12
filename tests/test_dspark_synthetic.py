#!/usr/bin/env python3
"""Exercise DSpark verification using a synthetic llama.cpp test-arch GGUF.

Generate targets with test-llama-archs -a '^llama$' -s 43 -o DIR or
-a '^deepseek41$' -s 42 -o DIR, then use --server PATH --target DIR/MODEL.gguf. Requires numpy and gguf-py in
PYTHONPATH. The synthetic vocab emits each token as two hexadecimal digits.
"""
import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request

import gguf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--server', required=True)
parser.add_argument('--target', required=True)
args = parser.parse_args()
target = os.path.abspath(args.target)
server = os.path.abspath(args.server)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
body = json.dumps({'model': 'test', 'prompt': 'tok_1tok_2tok_3', 'max_tokens': 6,
                   'temperature': 0, 'seed': 42, 'logprobs': 1}).encode()


def make_draft(path, transitions):
    source = gguf.GGUFReader(target)
    writer = gguf.GGUFWriter(path, 'dflash')
    arch = source.get_field('general.architecture').contents()
    if arch not in ('llama', 'deepseek41'):
        raise ValueError(f'unsupported synthetic target {arch}')
    model_keys = {'vocab_size', 'context_length', 'embedding_length', 'block_count',
                  'feed_forward_length', 'attention.head_count', 'attention.head_count_kv',
                  'attention.layer_norm_rms_epsilon', 'rope.dimension_count', 'rope.freq_base'}
    for key, field in source.fields.items():
        if key.startswith('tokenizer.'):
            if field.types[0] == gguf.GGUFValueType.ARRAY and len(field.types) == 1:
                continue
            name = key
        elif arch == 'llama' and key.startswith('llama.') and key[6:] in model_keys:
            name = 'dflash.' + key[6:]
        else:
            continue
        writer.add_key_value(name, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    if arch == 'deepseek41':
        for key, value in (('vocab_size', 128), ('context_length', 256), ('embedding_length', 512),
                           ('block_count', 1), ('feed_forward_length', 1024), ('attention.head_count', 8),
                           ('attention.head_count_kv', 8), ('rope.dimension_count', 64)):
            writer.add_uint32('dflash.' + key, value)
        writer.add_float32('dflash.attention.layer_norm_rms_epsilon', 1e-5)
        writer.add_float32('dflash.rope.freq_base', 10000.0)
    writer.add_uint32('dflash.block_size', 4)
    writer.add_array('dflash.target_layers', [0])
    writer.add_uint32('tokenizer.ggml.mask_token_id', 3)
    for tensor in source.tensors:
        if tensor.name in ('token_embd.weight', 'output_norm.weight', 'output.weight') or (
            arch == 'llama' and tensor.name.startswith('blk.') and tensor.name.endswith('.weight') and any(
                name in tensor.name for name in ('attn_norm', 'attn_q.', 'attn_k.', 'attn_v.',
                                                  'attn_output.', 'ffn_norm', 'ffn_gate.',
                                                  'ffn_down.', 'ffn_up.'))):
            writer.add_tensor(tensor.name, np.array(tensor.data))
    if arch == 'llama':
        for i in range(2):
            writer.add_tensor(f'blk.{i}.attn_q_norm.weight', np.ones(128, dtype=np.float32))
            writer.add_tensor(f'blk.{i}.attn_k_norm.weight', np.ones(128, dtype=np.float32))
        writer.add_tensor('fc.weight', np.eye(256, dtype=np.float32))
        writer.add_tensor('enc.output_norm.weight', np.ones(256, dtype=np.float32))
    else:
        rng = np.random.default_rng(123)
        writer.add_tensor('fc.weight', np.eye(512, dtype=np.float32))
        writer.add_tensor('enc.output_norm.weight', np.ones(512, dtype=np.float32))
        for name, size in (('attn_norm', 512), ('ffn_norm', 512), ('attn_q_norm', 64), ('attn_k_norm', 64)):
            writer.add_tensor(f'blk.0.{name}.weight', np.ones(size, dtype=np.float32))
        for name, shape in (('attn_q', (512, 512)), ('attn_k', (512, 512)),
                            ('attn_v', (512, 512)), ('attn_output', (512, 512)),
                            ('ffn_gate', (1024, 512)), ('ffn_up', (1024, 512)),
                            ('ffn_down', (512, 1024))):
            writer.add_tensor(f'blk.0.{name}.weight', rng.normal(0, 0.02, shape).astype(np.float32))
    w1 = np.eye(128, dtype=np.float32)
    w2 = np.zeros((128, 128), dtype=np.float32)
    for prev, token in transitions.items():
        w2[token, prev] = 1000
    writer.add_tensor('markov_w1.weight', w1)
    writer.add_tensor('markov_w2.weight', w2)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def copy_target_with_eos(path, eos_token):
    source = gguf.GGUFReader(target)
    arch = source.get_field('general.architecture').contents()
    writer = gguf.GGUFWriter(path, arch)
    for key, field in source.fields.items():
        if key.startswith('GGUF.') or key in ('general.architecture', 'tokenizer.ggml.eos_token_id'):
            continue
        if field.types[0] == gguf.GGUFValueType.ARRAY and len(field.types) == 1:
            continue
        writer.add_key_value(key, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.add_uint32('tokenizer.ggml.eos_token_id', eos_token)
    for tensor in source.tensors:
        writer.add_tensor(tensor.name, np.array(tensor.data), raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def start(port, log, draft=None):
    argv = [server, '-m', target, '--no-warmup', '-ngl', '0', '-c', '128', '-t', '2',
            '--threads-http', '2', '--port', str(port)]
    if draft:
        argv += ['--spec-type', 'draft-dspark', '-md', draft]
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode}, inspect {log.name}')
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return proc
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError('server startup timed out')


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def request(port):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=240) as response:
        return json.load(response)


def check(response, expected):
    assert response['choices'][0]['text'] == expected, response
    assert response['usage']['completion_tokens'] == 6, response


with tempfile.TemporaryDirectory(prefix='llama-box-dspark-') as directory:
    processes = []
    try:
        with open(os.path.join(directory, 'server.log'), 'w') as log:
            port = free_port()
            processes.append(start(port, log))
            reference = request(port)
            stop(processes.pop())
            expected = reference['choices'][0]['text']
            check(reference, expected)
            pieces = reference['choices'][0]['logprobs']['tokens']
            assert len(pieces) == 6, reference
            tokens = [int(piece, 16) for piece in pieces]
            if len(set(tokens[:3])) < 3:
                raise ValueError('fixture must generate three distinct initial tokens; use the documented seed')
            # Force at least the first proposed token to match the reference.
            draft = os.path.join(directory, 'draft.gguf')
            make_draft(draft, {tokens[0]: tokens[1]})
            port = free_port()
            processes.append(start(port, log, draft))
            first = request(port)
            check(first, expected)
            assert first['usage']['completion_tokens_details']['accepted_prediction_tokens'] > 0, first
            repeat = request(port)
            check(repeat, expected)
            if gguf.GGUFReader(target).get_field('general.architecture').contents() == 'llama':
                assert repeat['usage']['prompt_tokens_details']['cached_tokens'] > 0, repeat
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                concurrent = list(pool.map(lambda _: request(port), range(2)))
            for response in concurrent:
                check(response, expected)
            stop(processes.pop())
            # Force the first prediction away from the target's next token.
            reject = os.path.join(directory, 'reject.gguf')
            make_draft(reject, {tokens[0]: (tokens[1] + 1) % 128})
            port = free_port()
            processes.append(start(port, log, reject))
            rejected = request(port)
            check(rejected, expected)
            assert rejected['usage']['completion_tokens_details']['rejected_prediction_tokens'] > 0, rejected
            stop(processes.pop())
            # The third predicted token is EOS. A verifier must stop immediately,
            # discard later speculative KV, and leave a reusable prompt cache.
            eos_target = os.path.join(directory, 'eos-target.gguf')
            copy_target_with_eos(eos_target, tokens[2])
            target = eos_target
            port = free_port()
            processes.append(start(port, log))
            eos_reference = request(port)
            stop(processes.pop())
            eos_text = eos_reference['choices'][0]['text']
            assert eos_reference['choices'][0]['finish_reason'] == 'stop', eos_reference
            eos_draft = os.path.join(directory, 'eos-draft.gguf')
            make_draft(eos_draft, {tokens[0]: tokens[1], tokens[1]: tokens[2]})
            port = free_port()
            processes.append(start(port, log, eos_draft))
            for _ in range(2):
                eos_response = request(port)
                assert eos_response['choices'][0]['text'] == eos_text, eos_response
                assert eos_response['choices'][0]['finish_reason'] == 'stop', eos_response
                assert eos_response['usage']['completion_tokens'] == eos_reference['usage']['completion_tokens'], eos_response
                assert eos_response['usage']['completion_tokens_details']['accepted_prediction_tokens'] > 0, eos_response
            arch = gguf.GGUFReader(target).get_field('general.architecture').contents()
            cache_check = 'cache hit' if arch == 'llama' else 'repeat (prompt cache unavailable)'
            print(f'DSpark accepted/rejected verification, budget, EOS, {cache_check} and concurrency passed')
    finally:
        for proc in processes:
            stop(proc)
