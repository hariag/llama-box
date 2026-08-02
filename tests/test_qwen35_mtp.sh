#!/usr/bin/env bash
set -euo pipefail

binary=${1:?usage: test_qwen35_mtp.sh /path/to/llama-box /path/to/qwen35.gguf}
model=${2:?usage: test_qwen35_mtp.sh /path/to/llama-box /path/to/qwen35.gguf}
port=${LLAMA_BOX_TEST_PORT:-19092}
log=$(mktemp /tmp/llama-box-qwen35-mtp.XXXXXX.log)
pid=""

cleanup() {
    if [[ -n "$pid" ]]; then
        kill -- -"$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    rm -f "$log"
}
trap cleanup EXIT

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
setsid timeout 120s "$binary" --spec-type draft-mtp --spec-draft-n-max 3 -m "$model" -c 512 -ngl 99 -np 1 \
    --host 127.0.0.1 --port "$port" -t 4 >"$log" 2>&1 &
pid=$!

ready=1
for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
        ready=0
        break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        break
    fi
    sleep 1
done

if [[ $ready -ne 0 ]]; then
    cat "$log"
    exit 1
fi

grep -q "creating MTP draft context" "$log"
response=$(curl -fsS --max-time 45 "http://127.0.0.1:$port/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen35","messages":[{"role":"user","content":"Reply with exactly MTP_OK"}],"max_tokens":16,"temperature":0,"stream":false}')
python3 -c 'import json,sys; data=json.load(sys.stdin); assert data.get("choices"), data; assert data.get("usage", {}).get("draft_tokens", 0) > 0, data' <<<"$response"
echo QWEN35_MTP_TEST_PASS
