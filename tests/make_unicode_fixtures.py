#!/usr/bin/env python3
"""Make tiny deterministic byte-token GGUFs from seed-43 test-llama-archs output.

Example: test-llama-archs -a '^llama$' -s 43 -o /tmp/models
PYTHONPATH=.../gguf-py python make_unicode_fixtures.py /tmp/models/llama-dense.gguf /tmp/utf8-fixtures
Requires gguf-py and numpy.
"""
import os,sys
import gguf
import numpy as np
src, outdir = sys.argv[1:]
os.makedirs(outdir, exist_ok=True)
source=gguf.GGUFReader(src)

def make(name,mapping,prefix=False,chat=False):
 writer=gguf.GGUFWriter(os.path.join(outdir,name),'llama')
 for key, field in source.fields.items():
  if key.startswith('GGUF.') or key in ('general.architecture','tokenizer.ggml.model','tokenizer.ggml.tokens','tokenizer.ggml.token_type'): continue
  if field.types[0] == gguf.GGUFValueType.ARRAY and len(field.types)==1: continue
  writer.add_key_value(key,field.contents(),field.types[0],field.types[1] if len(field.types)>1 else None)
 writer.add_string('tokenizer.ggml.model','llama')
 words=[f'tok_{i}' for i in range(128)]; types=[1]*128
 if prefix: words[33]='A'
 if chat: types[1:4]=[4]*3
 for token,byte in mapping:
  words[token]=f'<0x{byte:02X}>'; types[token]=6
 writer.add_array('tokenizer.ggml.tokens',words); writer.add_array('tokenizer.ggml.token_type',types)
 for tensor in source.tensors: writer.add_tensor(tensor.name,np.array(tensor.data),raw_dtype=tensor.tensor_type)
 writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
make('seq.gguf',[(33,0xE2),(90,0x9C),(4,0x85)])
make('prefix.gguf',[(90,0xE2),(4,0x9C),(117,0x85)],True)

make('chat.gguf',[(33,0xE2),(90,0x9C),(4,0x85)],chat=True)
with open(os.path.join(outdir,'chat.jinja'),'w') as f: f.write('tok_1tok_2tok_3')
