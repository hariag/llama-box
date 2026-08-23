#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

"${CXX:-c++}" -std=c++17 \
  -I"$root_dir" \
  -I"$root_dir/llama.cpp" \
  -I"$root_dir/llama.cpp/include" \
  -I"$root_dir/llama.cpp/ggml/include" \
  -I"$root_dir/llama.cpp/vendor" \
  "$root_dir/tests/test_chat_template_request.cpp" \
  -o "$build_dir/test_chat_template_request"

"$build_dir/test_chat_template_request"
echo "CHAT_TEMPLATE_REQUEST_TEST_PASS"
