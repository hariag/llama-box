#!/usr/bin/env bash
# Verify patch scripts work with pre-1.8.5 Git, which has no global -C flag.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/project/llama-box/patches/llama.cpp" "$tmp/project/llama-box/patches/stable-diffusion.cpp" "$tmp/project/llama.cpp" "$tmp/project/stable-diffusion.cpp"
touch "$tmp/project/llama-box/patches/llama.cpp/test.patch" "$tmp/project/llama-box/patches/stable-diffusion.cpp/test.patch"
cat > "$tmp/bin/git" <<'GIT'
#!/usr/bin/env bash
if [[ ${1-} == --version ]]; then echo 'git version 1.8.3.1'; exit 0; fi
if [[ ${1-} == -C ]]; then exit 129; fi
printf '%s|%s\n' "$PWD" "$*" >> "$TEST_GIT_LOG"
GIT
chmod +x "$tmp/bin/git"
export TEST_GIT_LOG="$tmp/git.log"
cmake_bin=${CMAKE:-cmake}
(cd "$tmp/project/llama-box" && "$cmake_bin" -DGIT_EXECUTABLE="$tmp/bin/git" -P "$repo/llama-box/scripts/build-patch.cmake")
(cd "$tmp/project/llama-box" && "$cmake_bin" -DGIT_EXECUTABLE="$tmp/bin/git" -P "$repo/llama-box/scripts/build-patch-reset.cmake")
for vendor in llama.cpp stable-diffusion.cpp; do
    grep -Fx "$tmp/project/$vendor|apply --check $tmp/project/llama-box/patches/$vendor/test.patch" "$TEST_GIT_LOG" >/dev/null
    grep -Fx "$tmp/project/$vendor|apply --whitespace=nowarn $tmp/project/llama-box/patches/$vendor/test.patch" "$TEST_GIT_LOG" >/dev/null
    grep -Fx "$tmp/project/$vendor|reset --hard" "$TEST_GIT_LOG" >/dev/null
    grep -Fx "$tmp/project/$vendor|clean -df" "$TEST_GIT_LOG" >/dev/null
done
echo LEGACY_GIT_PATCH_TEST_PASS
