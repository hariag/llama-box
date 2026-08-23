#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
header="$repo_root/llama-box/z_stablediffusion.hpp"
api_header="$repo_root/stable-diffusion.cpp/include/stable-diffusion.h"

rg -q 'generate_image\(sd_ctx, &generation, &stream->images, &stream->num_images\)' "$header"
rg -q 'free_sd_images\(images, num_images\)' "$header"
rg -q 'upscale\(upscaler_ctx, img, upscale_factor,' "$header"
rg -q '&upscaled_images, &upscaled_count' "$header"
rg -q 'free_sd_images\(upscaled_images, upscaled_count\)' "$header"
rg -q 'generation\.sample_params\.flow_shift' "$header"
rg -q 'generation\.ref_image_args' "$header"
rg -q 'ctx_params\.model_args' "$header"
rg -q 'ctx_params\.backend' "$header"
rg -q 'ctx_params\.params_backend' "$header"

if rg -q 'ctx_params\.(vae_decode_only|free_params_immediately|offload_params_to_cpu|keep_control_net_on_cpu|keep_vae_on_cpu|qwen_image_zero_cond_t|flow_shift)' "$header"; then
    echo "stable-diffusion wrapper still uses removed context fields" >&2
    exit 1
fi

rg -q 'SD_API bool generate_image\(' "$api_header"
rg -q 'sd_image_t\*\* images_out' "$api_header"
rg -q 'SD_API bool upscale\(' "$api_header"
rg -q 'sd_image_t\*\* images_out' "$api_header"

echo "STABLE_DIFFUSION_API_TEST_PASS"
