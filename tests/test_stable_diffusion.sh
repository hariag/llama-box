#!/usr/bin/env bash
set -euo pipefail

binary=${1:?usage: test_stable_diffusion.sh /path/to/llama-box /path/to/model.gguf}
model=${2:?usage: test_stable_diffusion.sh /path/to/llama-box /path/to/model.gguf}
port=${LLAMA_BOX_TEST_PORT:-19093}
model_id=${MODEL_ID:-$(basename -- "$model")}
log=$(mktemp /tmp/llama-box-stable-diffusion.XXXXXX.log)
pid=""

cleanup() {
    if [[ -n "$pid" ]]; then
        kill -- -"$pid" 2>/dev/null || true
        for _ in $(seq 1 5); do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        kill -9 -- -"$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    rm -f "$log"
}
trap cleanup EXIT

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
setsid timeout 180s "$binary" --images -m "$model" --no-warmup \
    --image-max-width 256 --image-max-height 256 --image-sampling-steps 1 \
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

response=$(curl -fsS --max-time 180 "http://127.0.0.1:$port/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model_id\",\"prompt\":\"a simple red square on a white background\",\"n\":1,\"response_format\":\"b64_json\",\"size\":\"256x256\",\"sample_method\":\"euler\",\"sampling_steps\":1,\"stream\":false}")

python3 -c 'import base64,json,sys; data=json.load(sys.stdin); images=data.get("data", []); assert len(images) == 1, data; raw=base64.b64decode(images[0].get("b64_json", ""), validate=True); assert raw.startswith(b"\x89PNG\r\n\x1a\n"), raw[:16]; print("STABLE_DIFFUSION_TEST_PASS")' <<<"$response"
