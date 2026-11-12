#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ci_workflow="$repo_root/.github/workflows/ci.yml"
sync_workflow="$repo_root/.github/workflows/sync.yml"

# The Windows runner includes a complete OpenSSL SDK. Chocolatey's pinned
# 4.0.1 installer URL disappeared, so verify all six jobs use the runner SDK.
! rg -q '^\s+choco install openssl' "$ci_workflow"
[[ $(rg -c 'runner OpenSSL development headers are missing' "$ci_workflow") -eq 6 ]]
[[ $(rg -c 'runner OpenSSL installation is unusable' "$ci_workflow") -eq 6 ]]

cann_patch="$repo_root/llama-box/patches/ggml/cann-compat.patch"
test -f "$cann_patch"
if rg -q '^#include <aclnnop/aclnn_recurrent_gated_delta_rule.h>$' "$repo_root/llama.cpp/ggml/src/ggml-cann/aclnn_ops.cpp"; then
    git -C "$repo_root/llama.cpp/ggml" apply --check "$cann_patch"
fi

for provider in GitCode Gitee; do
    rg -q "name: Check ${provider} credentials" "$sync_workflow"
done
credential_guard_count=$(rg -c "^        if: steps.credentials.outputs.configured == 'true'$" "$sync_workflow")
[[ "$credential_guard_count" -eq 4 ]]

echo "CI_WORKFLOWS_TEST_PASS"
