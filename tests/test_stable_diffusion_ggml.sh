#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
patch_file="$repo_root/llama-box/patches/ggml/stable-diffusion-int8-convrot.patch"

test -f "$patch_file"
git -C "$repo_root/llama.cpp" apply --check "$patch_file"
rg -q '^diff --git a/ggml/include/ggml.h b/ggml/include/ggml.h$' "$patch_file"
rg -q '^diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c$' "$patch_file"
rg -q 'GGML_OP_QUANTIZE_I8_CONVROT' "$patch_file"
rg -q 'ggml_mul_mat_i8_tensorwise' "$patch_file"
rg -q 'ggml_quantize_i8_convrot' "$patch_file"

echo "STABLE_DIFFUSION_GGML_TEST_PASS"
