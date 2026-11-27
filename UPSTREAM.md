# Upstream dependencies

The checked-in submodule gitlinks are the authoritative dependency revisions.
The llama.cpp submodule is a fork because it also supplies the ggml operations
used by stable-diffusion.cpp, including INT8 convrot, FP8 E5M2 and SageAttention.
Do not replace it with an unmodified upstream checkout.

The November 2026 CUDA update merges ggml-org/llama.cpp at
`5d4c8eeb24db049acf74bd6f354bed6eaeffc7dc` into the fork. The multimodal
integration uses `tools/mtmd` from upstream
`b65bf69213d72a1bb0c9648fad26ac7adf9e73ea`, immediately before the removal
of the `clip` interfaces used by llama-box. It also carries the small
llama-box compatibility helpers. Selected later changes, including Gemma 4
streaming audio encoding, have been backported to that interface. Other later
mtmd changes require migration to the public mtmd API.

The fork adds ggml operations and uses 160-byte tensor names. Its RPC protocol
uses fork-specific major version 11, and the patch field identifies the tensor
name size. Clients reject mismatched protocol versions and name sizes. Upgrade
all llama-box RPC workers and the main server together.

The November 18 model sync backports the following changes without replacing
the fork's ggml or multimodal APIs:

- llama.cpp: Gemma 4 tool calls from an unclosed thought (#32232),
  Qwen3.5 combined tensor conversion and Decider metadata (#32102, #32156),
  Qwen3-Next output scale and Qwen-VL tensor and projector metadata
  (#32393, #32241, #32349, #32130), and Qwen4Exp FP8 conversion and graph reuse
  (#32476, #32483).
- stable-diffusion.cpp: Qwen Image 2.1 flow schedule (#2048), fused MLP LoRA
  mapping (#2057), and VAE FP16 scaling adapted to the fork's existing 3D
  convolution and singleton temporal weight handling (#2054).

The November 27 sync uses ggml-org/llama.cpp `a07831a008` and the Unsloth
inference fork as review sources.

- Gemma 4 E2B and E4B with shared KV layers prune unneeded query rows during
  prefill in a multi-slot context. [llama.cpp #32342](https://github.com/ggml-org/llama.cpp/pull/32342)
- KV sequence removal looks up the sequence's positions instead of scanning
  every cell. [llama.cpp #32380](https://github.com/ggml-org/llama.cpp/pull/32380)
- CUDA fuses RMS normalization, weight multiplication, and MROPE/IMROPE,
  including partial rotation in Qwen. CUDA also checks Q/K strides before
  accepting a Gated Delta Net operation. [llama.cpp #32500](https://github.com/ggml-org/llama.cpp/pull/32500), [llama.cpp #32660](https://github.com/ggml-org/llama.cpp/pull/32660)
- Unsloth indexes sampling penalties by token ID when candidates are in
  full vocabulary order. Reordered and filtered candidates use the existing
  path. [Unsloth #95](https://github.com/unslothai/llama.cpp/pull/95)

The CUDA build was tested on RTX 3090 with driver 580.173.02 and CUDA 13.0
Update 1. With Gemma 4 E2B Q4_0, four concurrent HTTP requests of about 2,056
prompt tokens each, a 16K context, batch 2048, microbatch 512 and prompt
caching disabled, median aggregate prompt throughput rose from 7,180 to
9,886 tok/s with split KV and from 7,364 to 10,147 tok/s with unified KV.
Each result is the median of three request batches. Reserved KV buffers
were unchanged in each mode. The unified CUDA compute buffer rose from
566.31 to 581.82 MiB. All 436 CUDA RMSNorm/RoPE cases passed, and Qwen3.8-27B
completed streaming and non-streaming chat with MTP predictions accepted.

The Qwen recurrent replay update changes runtime state and several backends.
The Unsloth quantized-KV updates depend on a different CUDA attention layout.
Both remain separate from this sync. [llama.cpp #31669](https://github.com/ggml-org/llama.cpp/pull/31669), [Unsloth #303](https://github.com/unslothai/llama.cpp/pull/303)

The November 27 follow-up brings over additional performance and input
validation fixes from the same upstream revision.

- BPE tokenization reduces merge-queue work and allocations. Large CPU
  `GET_ROWS` operations use all available worker threads. [llama.cpp #30962](https://github.com/ggml-org/llama.cpp/pull/30962), [#32859](https://github.com/ggml-org/llama.cpp/pull/32859), [#32514](https://github.com/ggml-org/llama.cpp/pull/32514)
- The C++ GGUF reader validates tensor types before conversion, and the Python
  reader checks tensor extents using Python integer offsets. GBNF repetition counts are range-checked.
  Buffered model reads resume after a partial direct read, and CUDA matrix-vector
  kernels compute matrix offsets with 64-bit arithmetic. [llama.cpp #32641](https://github.com/ggml-org/llama.cpp/pull/32641), [#32495](https://github.com/ggml-org/llama.cpp/pull/32495), [#32551](https://github.com/ggml-org/llama.cpp/pull/32551), [#32545](https://github.com/ggml-org/llama.cpp/pull/32545), [#32965](https://github.com/ggml-org/llama.cpp/pull/32965)
- cpp-httplib advances to 0.67.0. The retained mtmd implementation validates
  required HunyuanVL and LLaVA tensors and uses the MiniCPM-V projector's
  output width for its embedding size. [llama.cpp #32563](https://github.com/ggml-org/llama.cpp/pull/32563), [#32131](https://github.com/ggml-org/llama.cpp/pull/32131), [#32129](https://github.com/ggml-org/llama.cpp/pull/32129), [#32132](https://github.com/ggml-org/llama.cpp/pull/32132)

All 15 CPU tokenizer suites and 261 CPU `GET_ROWS` cases passed. Token IDs
matched the baseline with both special-token modes on a 149,010-byte corpus
of Chinese text, code and emoji. Median isolated tokenization time fell from
55.11 to 36.93 ms for Qwen3.5 and from 35.90 to 17.62 ms for Gemma 4.
The integrated CUDA build on the RTX 3090 passed Gemma 4 E2B single- and
four-slot chat and Qwen3.8-27B MTP streaming and non-streaming chat. The
three projector changes have build coverage.

The GB10 sync adds a Linux ARM64 job for CUDA 13.0.3 with `121a-real`,
the SM121a target recommended for DGX Spark. Its CUDA runtime matches the
DGX OS 7.6.0 software stack with driver 580.178.04. [NVIDIA build guide](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/llama-cpp), [DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)

- CUDA adds few-row MMQ for dense and MoE matrix multiplication, including
  zero-block quantization and GB10 dispatch starting at two tokens for
  eligible quantizations. [llama.cpp #31533](https://github.com/ggml-org/llama.cpp/pull/31533), [#32065](https://github.com/ggml-org/llama.cpp/pull/32065), [#32566](https://github.com/ggml-org/llama.cpp/pull/32566)
- Quantized cuBLAS operations on GB10 prefer BF16 inputs with FP32
  accumulation in automatic mode. The backport takes the GB10 branch of
  the upstream policy and uses the existing precision overrides. CUDA
  packs eight NVFP4 blocks per dequantization thread block. [llama.cpp #30672](https://github.com/ggml-org/llama.cpp/pull/30672), [#32875](https://github.com/ggml-org/llama.cpp/pull/32875)
- Multi-stage Flash Attention rotates the order of K/V reads for better L2
  reuse. MMQ sets the shared-memory limit for the selected kernel. [llama.cpp #32407](https://github.com/ggml-org/llama.cpp/pull/32407), [#32392](https://github.com/ggml-org/llama.cpp/pull/32392)
- On WSL2 with an integrated GPU such as RTX Spark N1X, CUDA supplies free
  GPU memory. On native Linux UMA devices such as DGX Spark, the backend
  uses host memory information. The WSL2 backport comes from an open
  upstream pull request. [llama.cpp #31670](https://github.com/ggml-org/llama.cpp/issues/31670), [#31673](https://github.com/ggml-org/llama.cpp/pull/31673)

CUDA 13.0 generated native SM121a cubins for the affected kernels. On the
RTX 3090, 33 matrix multiplication, 21 NVFP4 dequantization and 248 Flash
Attention cases passed against CPU reference results. The WSL2 memory
policy passed all eight host-side test cases. GB10 and RTX Spark runtime
performance remains to be measured on the target devices.

Some historical patches in `llama-box/patches` are already applied in the
fork; an apply warning by itself does not mean a change is missing. Check both
the fork contents and the relevant build configuration before changing them.
The build scripts modify submodule worktrees and, unless `BOX_PATCH_DEBUG` is
enabled, reset them after building. Build from clean, disposable checkouts.

The model update also includes Gemma 4 chat fixes, Qwen4exp MTP support,
DeepSeek V4.1 legacy MXFP8 loading and DFlash/DSpark server integration.
DeepSeek V4.1 prompt checkpoint reuse is not implemented. DFlash/DSpark
requires a compatible draft model, sufficient sequence rollback, and a
draft size smaller than the microbatch; multimodal prefill is not supported.

Before refreshing the fork, check ggml operation numbers, RPC compatibility,
model and mtmd APIs, and stable-diffusion.cpp's use of the shared ggml backend.
Validate CPU and CUDA builds, model inference, speculative decoding, images,
audio, and the backends changed by the update. Record the tested GPU, driver,
CUDA toolkit, model, and runtime settings with the results.

For a standalone llama.cpp build with shared ggml libraries, the fork-specific
GPU checks can be run against the same backend libraries as follows (select the
intended GPU with `CUDA_VISIBLE_DEVICES`):

```sh
c++ -std=c++17 -Illama.cpp/ggml/include tests/test_ggml_custom_ops.cpp \
  -L"$BUILD/bin" -Wl,-rpath,"$BUILD/bin" -lggml -lggml-base -o /tmp/test-ggml-custom
/tmp/test-ggml-custom CUDA0
c++ -std=c++17 -Illama.cpp/ggml/include tests/test_ggml_sage_attention.cpp \
  -L"$BUILD/bin" -Wl,-rpath,"$BUILD/bin" -lggml -lggml-base -o /tmp/test-ggml-sage
/tmp/test-ggml-sage
```

Qwen35 hybrid prompt caching retains one checkpoint per idle slot for an
identical text-only prompt, without moving KV positions. The last prompt token
is evaluated again for logits. Different prompts, multimodal input, LoRA and
embedding mode do not use this checkpoint. MTP target, draft and speculative
state are restored together; separate draft models are excluded. Each slot's
checkpoint is capped at 768 MiB, after which the server falls back to prefill.
The server-wide `--no-cache-prompt` switch disables caching. Checkpoints are
in-memory and can be evicted under capacity pressure. The current implementation
does not reuse a merely shared prefix or honor a per-request `cache_prompt`
override.

To compare a candidate against an otherwise identical server started with
`--no-cache-prompt`, run `tests/test_qwen_prompt_cache.py --candidate URL
--reference URL --require-hit`. Add `--require-mtp` for MTP and `--concurrency`
when the candidate has at least two slots. The test sends requests to both
servers and should be run against isolated test instances.

For shared-KV saturation and eviction, run an isolated Qwen35 server with a
1024-token context, eight slots, `--kv-unified`, and verbose logging, then run
`tests/test_qwen_cache_saturation.py --url URL --server-log LOG`. The test
checks that all eight slots were used, at least four idle slots were evicted,
and a subsequent request can complete and be cached. Add `--disconnect` to
exercise non-streaming cancellation while waiting for dynamic admission.
