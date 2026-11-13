#!/usr/bin/env python3
"""Black-box regression for UTF-8 split across generated tokens.

Generate the fixtures first using make_unicode_fixtures.py. Requires only the
Python standard library for test execution. Starts isolated loopback servers.

python test_unicode_http.py --server /path/to/llama-box --seq /path/seq.gguf --prefix /path/prefix.gguf
"""
import argparse, json, socket, subprocess, tempfile, time, urllib.error, urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--server', required=True)
parser.add_argument('--seq', required=True)
parser.add_argument('--prefix', required=True)
parser.add_argument('--chat', help='optional chat fixture')
parser.add_argument('--chat-template', help='chat template emitted with chat fixture')
a = parser.parse_args()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

class Server:
    def __init__(self, model, template=None):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); self.port = s.getsockname()[1]
        self.log = tempfile.TemporaryFile()
        argv = [a.server, '-m', model, '--no-warmup', '-ngl', '0', '-c', '128',
                '-t', '2', '--threads-http', '2', '--port', str(self.port)]
        if template: argv += ['--jinja', '--chat-template-file', template]
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.log.seek(0); raise RuntimeError(self.log.read().decode(errors='replace'))
            try:
                with socket.create_connection(('127.0.0.1', self.port), timeout=1): return
            except OSError: time.sleep(0.05)
        self.close(); raise RuntimeError('server startup timeout')
    def close(self):
        self.proc.terminate()
        try: self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
        self.log.close()
    def request(self, stream, n, stop=None, chat=False):
        body = {'model':'test', 'prompt':[1,2,3], 'max_tokens':n, 'temperature':0, 'seed':42, 'stream':stream}
        if stop is not None: body['stop'] = stop
        endpoint = '/v1/completions'
        if chat:
            body.pop('prompt'); body['messages'] = [{'role':'user','content':'hello'}]
            endpoint = '/v1/chat/completions'
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}{endpoint}', data=json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=30) as resp: raw=resp.read().decode('utf-8',errors='strict')
        if not stream:
            choice = json.loads(raw)['choices'][0]
            return (choice['message'].get('content') or '' if chat else choice['text']), choice['finish_reason']
        chunks, finishes, saw_done = [], [], False
        for event in raw.split('\n\n'):
            if not event: continue
            assert event.startswith('data: '), repr(event)
            data=event[6:]
            if data == '[DONE]': saw_done=True; continue
            choice=json.loads(data)['choices'][0]
            text=choice.get('delta',{}).get('content','') if chat else choice.get('text','')
            text=text or ''
            assert '\ufffd' not in text, f'invalid UTF-8 replacement in event {event!r}'
            chunks.append(text)
            if choice.get('finish_reason') is not None: finishes.append(choice['finish_reason'])
        assert saw_done and len(finishes)==1, (saw_done,finishes,raw)
        return ''.join(chunks),finishes[0]

def run(model, cases, template=None):
    s=Server(model,template)
    try:
        for n,stop,expected,reason in cases:
            for stream in (False,True):
                got=s.request(stream,n,stop,chat=bool(template))
                assert got == (expected,reason), f'n={n} stop={stop!r} stream={stream}: got {got!r}, expected {(expected,reason)!r}'
                print(f'PASS stream={stream} n={n} stop={stop!r}: {got!r}')
    finally: s.close()

run(a.seq, [(1,None,'','length'), (2,None,'','length'), (3,None,'✅','length'),
            (4,None,'✅tok_117','length'), (5,'tok_1','✅','stop'),
            (5,'✅','','stop'), (5,'✅tok','','stop')])
run(a.prefix, [(1,None,'A','length'), (2,None,'A','length'), (3,None,'A','length'),
               (4,None,'A✅','length'), (5,None,'A✅tok_86','length'),
               (8,'✅','A','stop'), (8,'A✅tok','','stop'), (8,'tok_8','A✅','stop')])

if bool(a.chat) != bool(a.chat_template):
    parser.error('--chat and --chat-template must be provided together')
if a.chat:
    run(a.chat, [(1,None,'','length'), (2,None,'','length'), (4,None,'✅tok_117','length'),
                 (5,'tok_1','✅','stop'), (5,'✅','','stop')], a.chat_template)

def check_chat_logprobs(model,template):
    server=Server(model,template)
    try:
        for stream in (False,True):
            body={'model':'test','messages':[{'role':'user','content':'hello'}], 'max_tokens':4,
                  'temperature':0,'seed':42,'stream':stream,'logprobs':True,'top_logprobs':1}
            req=urllib.request.Request(f'http://127.0.0.1:{server.port}/v1/chat/completions',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with opener.open(req,timeout=30) as resp: raw=resp.read().decode('utf-8',errors='strict')
            choices=[]
            if stream:
                for event in raw.split('\n\n'):
                    if event.startswith('data: ') and event[6:] != '[DONE]': choices.extend(json.loads(event[6:])['choices'])
            else: choices=json.loads(raw)['choices']
            content=[]; items=[]
            for c in choices:
                ctext=(c.get('delta') or {}).get('content') if stream else c['message']['content']
                if ctext: content.append(ctext)
                items.extend((c.get('logprobs') or {}).get('content') or [])
            assert ''.join(content)=='✅tok_117', (stream,content)
            expected=[[0xe2],[0x9c],[0x85],list(b'tok_117')]
            got=[entry['bytes'] for entry in items]
            assert got == expected, f'raw token bytes: stream={stream} got {got}, expected {expected}'
            for entry, raw_bytes in zip(items,expected):
                assert entry['top_logprobs'][0]['bytes'] == raw_bytes, (stream, entry)
            print(f'PASS chat logprobs raw bytes stream={stream}')
    finally: server.close()

if a.chat:
    check_chat_logprobs(a.chat,a.chat_template)
