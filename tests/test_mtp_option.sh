#!/usr/bin/env bash
set -euo pipefail

binary=${1:?usage: test_mtp_option.sh /path/to/llama-box}

help_output=$("$binary" --spec-type draft-mtp --spec-draft-n-max 3 --help 2>&1)
mtp_help_output=$("$binary" --mtp --help 2>&1)

if grep -q -- "Unknown argument" <<<"$help_output"; then
    printf '%s\n' "$help_output"
    exit 1
fi

grep -q -- "--spec-type" <<<"$help_output"
grep -q -- "draft-mtp" <<<"$help_output"
grep -q -- "--spec-draft-n-max" <<<"$help_output"
grep -q -- "--mtp" <<<"$mtp_help_output"
