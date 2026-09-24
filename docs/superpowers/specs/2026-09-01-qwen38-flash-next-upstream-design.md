# Qwen3.8-Flash-Next upstream support

## Goal

Upgrade the self-maintained `llama-box` integration so it can load and serve
the `Qwen3.8-Flash-Next` GGUF release, including text generation, MTP
speculative decoding, and vision requests when a compatible projector is
available. The only full-model runtime validation target is host `10.10.0.180`
because the model does not fit the available test hardware elsewhere.

## Current evidence

- `llama-box` pins `llama.cpp` at `eb25b7263e1604b4382295563f5a924002d6f87c`.
- Upstream commit `6c84c7d5d8833c6e0df69628f75a0f599797934e` adds the
  `qwen4exp` GGUF conversion, loader, hybrid graph, PLE n-gram embedding,
  QSA sparse attention, and model tests for Qwen3.8-Flash-Next.
- Upstream master currently resolves to `0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0`
  and contains subsequent Qwen4Exp fixes for recurrent rollback, QSA, cache
  state, and backend behavior.
- The test model is stored on 180 as four GGUF shards under
  `/data16t/qwen38-flash-next/models/UD-Q4_K_XL/`, totaling about 104 GB.
- The current 180 worker container does not bind-mount that host directory;
  tests must use a temporary explicit model mount or an equivalent temporary
  staging path.

## Design

### Dependency boundary

Move only the `llama.cpp` submodule pointer to the reviewed upstream master
revision. Do not fork or reimplement Qwen4Exp inside `llama-box`. Preserve the
existing `stable-diffusion.cpp`, queue submodules, dynamic context quota, and
AIStack changes.

The build patch remains the compatibility boundary. Run its application from a
clean isolated build directory. If upstream API changes make a patch fail,
port only the affected integration shim into
`llama-box/patches/llama.cpp/zz_compat.patch` or the llama-box call site. Do not
edit upstream submodule files to hide a patch conflict.

### Runtime integration

Use the upstream `qwen4exp` model architecture and its GGUF metadata as the
source of truth. Verify that the existing llama-box argument forwarding,
chat-template handling, MTP path, unified KV path, and multimodal assembly
work with the new architecture. Add a llama-box-side compatibility change only
if a concrete compile or runtime failure is reproduced.

The model directory on 180 is supplied to the test process through an explicit
temporary mount. No production container, model registry tag, or 179 service is
changed as part of this validation.

### Validation

Run, in order:

1. `git diff --check`, submodule cleanliness, and patch-application checks.
2. CPU configure/build and `llama-box --help`/version checks.
3. CUDA build on the supported 180 environment.
4. GGUF metadata/load test against all four shards.
5. Text request with normal sampling and a short deterministic response.
6. MTP request using the model's compatible draft checkpoint/options.
7. `--parallel=1` and `--parallel=4`, including `--kv-unified`.
8. Vision request only when a compatible projector file is available; record
   the exact projector provenance and do not call the test passed without it.
9. Existing llama-box API, dynamic-context, chat-template, Qwen3.5 MTP, and
   stable-diffusion regression tests.

For each model run, retain the binary revision, submodule revision, command
arguments, exit status, and relevant startup/error lines. Do not retain
credentials or tokens in logs.

### Rollback

The rollback point is the current `llama-box` `main` commit and the current
`llama.cpp` submodule pointer. A failed validation must leave the main worktree
and all submodules recoverable by reverting only the intended submodule pointer
and compatibility patch changes.

## Out of scope

- Deploying or restarting 179.
- Changing AIStack model catalog entries or resource estimation.
- Replacing the self-developed llama-box implementation with upstream server.
- Updating stable-diffusion.cpp or unrelated upstream dependencies.
- Claiming VLM success when no compatible projector is present on 180.
