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
llama-box compatibility helpers. Later mtmd changes are not part of this
update and require migration to the public mtmd API.

The fork adds ggml operations and uses 160-byte tensor names. Its RPC protocol
uses a fork-specific major version, and the patch field identifies the tensor
name size. Clients reject mismatched protocol versions and name sizes. Upgrade
all llama-box RPC workers and the main server together.
The pinned stable-diffusion.cpp remains `e29085a5cdf6365dee4cbacddf4d889b104bb498`.

Some historical patches in `llama-box/patches` are already applied in the
fork; an apply warning by itself does not mean a change is missing. Check both
the fork contents and the relevant build configuration before changing them.
The build scripts modify submodule worktrees and, unless `BOX_PATCH_DEBUG` is
enabled, reset them after building. Build from clean, disposable checkouts.

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
