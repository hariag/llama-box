#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ci_workflow="$repo_root/.github/workflows/ci.yml"
sync_workflow="$repo_root/.github/workflows/sync.yml"

openssl_lines=$(rg '^\s+choco install openssl' "$ci_workflow")
[[ $(wc -l <<<"$openssl_lines") -eq 6 ]]
while IFS= read -r line; do
    [[ "$line" == *"--version 4.0.1 -y"* ]]
done <<<"$openssl_lines"

cann_patch="$repo_root/llama-box/patches/ggml/cann-compat.patch"
test -f "$cann_patch"
git -C "$repo_root/llama.cpp/ggml" apply --check "$cann_patch"

for provider in GitCode Gitee; do
    rg -q "name: Check ${provider} credentials" "$sync_workflow"
done
credential_guard_count=$(rg -c "^        if: steps.credentials.outputs.configured == 'true'$" "$sync_workflow")
[[ "$credential_guard_count" -eq 4 ]]

echo "CI_WORKFLOWS_TEST_PASS"
