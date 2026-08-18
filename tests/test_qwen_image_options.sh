#!/usr/bin/env bash
set -euo pipefail

binary=${1:?usage: test_qwen_image_options.sh /path/to/llama-box}

help=$($binary --help 2>&1)

grep -q -- "--image-llm-model" <<<"$help"
grep -q -- "--image-qwen-image-zero-cond-t" <<<"$help"
grep -q -- "--image-flow-shift" <<<"$help"

echo "QWEN_IMAGE_OPTIONS_TEST_PASS"
