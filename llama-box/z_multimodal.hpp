#pragma once

// heads

#include <atomic>
#include <algorithm>
#include <csignal>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>

#include "llama.cpp/tools/mtmd/clip-impl.h"
#include "llama.cpp/tools/mtmd/mtmd-audio.h"
#include "llama.cpp/tools/mtmd/mtmd-helper.h"
#include "llama.cpp/tools/mtmd/mtmd-image.h"

// types

struct llama_multimodal_tokens {
    llama_token        dummy_token = LLAMA_TOKEN_NULL;
    int32_t            n_tokens    = 0;
    int32_t            n_pos       = 0;
    bool               is_audio    = false;
    std::vector<float> embed;
    clip_image_size    size;
    clip_image_size    grid_size;
};

// implementations

struct llama_multimodal_embed_batch {
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    std::vector<llama_pos>      pos;
    llama_batch                 temp = {};

    llama_multimodal_embed_batch() = default;

    llama_multimodal_embed_batch(float * embd, int32_t n_tokens, std::vector<llama_pos> && pos, llama_seq_id seq_id) {
        n_seq_id.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids.resize(n_tokens + 1);
        seq_ids[n_tokens] = nullptr;
        logits.resize(n_tokens);
        this->pos = std::move(pos);
        temp      = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/this->pos.data(),
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            temp.n_seq_id[i] = 1;
            temp.seq_id[i]   = seq_id_0.data();
            temp.logits[i]   = false;
        }
    }

    llama_multimodal_embed_batch(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        n_seq_id.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids.resize(n_tokens + 1);
        seq_ids[n_tokens] = nullptr;
        logits.resize(n_tokens);
        pos.resize(n_tokens);
        temp = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/pos.data(),
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            temp.pos[i]      = pos_0 + i;
            temp.n_seq_id[i] = 1;
            temp.seq_id[i]   = seq_id_0.data();
            temp.logits[i]   = false;
        }
    }
};

static std::atomic<llama_token> multimodal_dummy_token_generator{ LLAMA_TOKEN_NULL };

// llama.cpp intentionally keeps these model predicates internal to mtmd.  LLaMA
// Box needs a small compatibility surface because it performs its own prompt
// assembly and embedding cache.  Keep the mapping here so upstream mtmd API
// churn does not leak through the HTTP server.
static inline int clip_is_minicpmv(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_MINICPMV ? clip_get_hparams(ctx)->minicpmv_version : 0;
}

static inline bool clip_is_glm(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_GLM_EDGE;
}

static inline bool clip_is_qwen2vl(const clip_ctx * ctx) {
    const auto type = clip_get_projector_type(ctx);
    return type == PROJECTOR_TYPE_QWEN2VL || type == PROJECTOR_TYPE_QWEN25VL || type == PROJECTOR_TYPE_QWEN3VL;
}

static inline bool clip_is_gemma3(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_GEMMA3;
}

static inline bool clip_is_smolvlm(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_IDEFICS3;
}

static inline bool clip_is_pixtral(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_PIXTRAL;
}

static inline bool clip_is_internvl(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_INTERNVL;
}

static inline bool clip_is_llama4(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_LLAMA4;
}

static inline bool clip_is_ultravox(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_ULTRAVOX;
}

static inline bool clip_is_qwen2a(const clip_ctx * ctx) {
    const auto type = clip_get_projector_type(ctx);
    return type == PROJECTOR_TYPE_QWEN2A || type == PROJECTOR_TYPE_QWEN25O;
}

static inline bool clip_is_voxtral(const clip_ctx * ctx) {
    return clip_get_projector_type(ctx) == PROJECTOR_TYPE_VOXTRAL;
}

static inline int clip_get_patch_size(const clip_ctx * ctx) {
    return clip_get_hparams(ctx)->patch_size;
}

static inline std::unique_ptr<mtmd_image_preprocessor> make_image_preprocessor(clip_ctx * ctx) {
    const auto type = clip_get_projector_type(ctx);
    const bool has_pinpoints = !clip_get_hparams(ctx)->image_res_candidates.empty();
    switch (type) {
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_COGVLM:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_GLM_EDGE:
            if (has_pinpoints) {
                return std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx);
            }
            return std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx);
        case PROJECTOR_TYPE_MINICPMV:
        case PROJECTOR_TYPE_MINICPMV4_6:
        case PROJECTOR_TYPE_LLAMA4:
        case PROJECTOR_TYPE_INTERNVL:
            if (type == PROJECTOR_TYPE_INTERNVL) {
                return std::make_unique<mtmd_image_preprocessor_internvl>(ctx);
            }
            return std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx);
        case PROJECTOR_TYPE_IDEFICS3:
            return std::make_unique<mtmd_image_preprocessor_idefics3>(ctx);
        case PROJECTOR_TYPE_STEP3VL:
            return std::make_unique<mtmd_image_preprocessor_step3vl>(ctx);
        case PROJECTOR_TYPE_YOUTUVL:
            return std::make_unique<mtmd_image_preprocessor_youtuvl>(ctx);
        case PROJECTOR_TYPE_DEEPSEEKOCR:
        case PROJECTOR_TYPE_DEEPSEEKOCR2:
            return std::make_unique<mtmd_image_preprocessor_deepseekocr>(ctx);
        case PROJECTOR_TYPE_LIGHTONOCR:
            return std::make_unique<mtmd_image_preprocessor_longest_edge>(ctx);
        case PROJECTOR_TYPE_LFM2:
            return std::make_unique<mtmd_image_preprocessor_lfm2>(ctx);
        case PROJECTOR_TYPE_GRANITE4_VISION:
            return std::make_unique<mtmd_image_preprocessor_granite>(ctx);
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_MIMOVL:
        case PROJECTOR_TYPE_MINIMAX_M3:
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_PHI4:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_KIMIK25:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_GEMMA4UV:
        case PROJECTOR_TYPE_HUNYUANVL:
        case PROJECTOR_TYPE_EXAONE4_5:
        case PROJECTOR_TYPE_DOTS_OCR:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
        case PROJECTOR_TYPE_YASA2:
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV:
            if (type == PROJECTOR_TYPE_NEMOTRON_V2_VL || type == PROJECTOR_TYPE_YASA2 ||
                type == PROJECTOR_TYPE_GEMMA3 || type == PROJECTOR_TYPE_GEMMA3NV) {
                return std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx);
            }
            return std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx);
        default:
            throw std::runtime_error("unsupported vision projector type");
    }
}

static inline std::unique_ptr<mtmd_audio_preprocessor> make_audio_preprocessor(clip_ctx * ctx) {
    switch (clip_get_projector_type(ctx)) {
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_QWEN25O:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_MERALION:
            return std::make_unique<mtmd_audio_preprocessor_whisper>(ctx);
        case PROJECTOR_TYPE_QWEN3A:
            return std::make_unique<mtmd_audio_preprocessor_qwen3a>(ctx);
        case PROJECTOR_TYPE_LFM2A:
            return std::make_unique<mtmd_audio_preprocessor_conformer>(ctx);
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            return std::make_unique<mtmd_audio_preprocessor_granite_speech>(ctx);
        case PROJECTOR_TYPE_GEMMA4A:
            return std::make_unique<mtmd_audio_preprocessor_gemma4a>(ctx);
        case PROJECTOR_TYPE_GEMMA4UA:
            return std::make_unique<mtmd_audio_preprocessor_gemma4ua>(ctx);
        case PROJECTOR_TYPE_MIMO_AUDIO:
            return std::make_unique<mtmd_audio_preprocessor_mimo_audio>(ctx);
        case PROJECTOR_TYPE_PARAKEET:
            return std::make_unique<mtmd_audio_preprocessor_parakeet>(ctx);
        default:
            throw std::runtime_error("unsupported audio projector type");
    }
}

static inline clip_image_u8 resize_image_to_max(const clip_image_u8 & src, int max_image_size) {
    const clip_image_size source_size = src.get_size();
    if (max_image_size <= 0 || std::max(source_size.width, source_size.height) <= max_image_size) {
        return src;
    }

    const float scale = static_cast<float>(max_image_size) /
                        static_cast<float>(std::max(source_size.width, source_size.height));
    const int width  = std::max(1, static_cast<int>(std::round(source_size.width * scale)));
    const int height = std::max(1, static_cast<int>(std::round(source_size.height * scale)));
    clip_image_u8 dst;
    dst.set_size({width, height}, src.is_placeholder());
    if (src.is_placeholder()) {
        return dst;
    }

    for (int y = 0; y < height; ++y) {
        const float sy = (height == 1 ? 0.0f : static_cast<float>(y) * (source_size.height - 1) / (height - 1));
        const int y0 = static_cast<int>(sy);
        const int y1 = std::min(y0 + 1, source_size.height - 1);
        const float fy = sy - y0;
        for (int x = 0; x < width; ++x) {
            const float sx = (width == 1 ? 0.0f : static_cast<float>(x) * (source_size.width - 1) / (width - 1));
            const int x0 = static_cast<int>(sx);
            const int x1 = std::min(x0 + 1, source_size.width - 1);
            const float fx = sx - x0;
            const auto p00 = src.get_pixel(x0, y0);
            const auto p01 = src.get_pixel(x1, y0);
            const auto p10 = src.get_pixel(x0, y1);
            const auto p11 = src.get_pixel(x1, y1);
            std::array<uint8_t, 3> p{};
            for (int c = 0; c < 3; ++c) {
                const float top = p00[c] * (1.0f - fx) + p01[c] * fx;
                const float bot = p10[c] * (1.0f - fx) + p11[c] * fx;
                p[c] = static_cast<uint8_t>(std::clamp(std::round(top * (1.0f - fy) + bot * fy), 0.0f, 255.0f));
            }
            dst.set_pixel(x, y, p);
        }
    }
    return dst;
}

// tokenize_image is not thread-safe, must be called from mutex-protected context.
static inline std::vector<llama_multimodal_tokens> tokenize_image(clip_ctx * ctx_clip, const int n_threads,
                                                                  const clip_image_u8 * img,
                                                                  int max_image_size = 0) {
    auto preprocessor = make_image_preprocessor(ctx_clip);
    auto preproc_out = preprocessor->preprocess(resize_image_to_max(*img, max_image_size));
    clip_image_f32_batch batch;
    batch.entries = std::move(preproc_out.entries);
    batch.is_audio = false;
    if (preproc_out.has_overview()) {
        batch.entries.insert(batch.entries.begin(), std::move(preproc_out.overview));
    }
    const int grid_x = preproc_out.grid_x;
    const int grid_y = preproc_out.grid_y;
    if (batch.entries.empty()) {
        LOG_ERR("%s", "unable to preprocess image\n");
        return {};
    }

    const int32_t n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);

    std::vector<llama_multimodal_tokens> result;

    // minicpmv, slicing
    if (clip_is_minicpmv(ctx_clip) != 0) {
        result.resize(batch.entries.size());

        const auto & entries   = batch.entries;
        const size_t n_entries = entries.size();
        for (size_t i = 0; i < n_entries; i++) {
            // init
            result[i].n_tokens  = clip_n_output_tokens(ctx_clip, &entries[i]);
            result[i].n_pos     = result[i].n_tokens;
            result[i].size      = entries[i].get_size();
            result[i].grid_size = clip_image_size{ grid_x, grid_y };
            result[i].embed.resize(result[i].n_tokens * n_mmproj_embd);
            // encode
            const int64_t t_start = ggml_time_us();
            bool          encoded = clip_image_encode(ctx_clip, n_threads, &entries[i], result[i].embed);
            if (!encoded) {
                LOG_ERR("failed to encode image %2zu/%zu\n", i + 1, n_entries);
                return {};
            }
            if (common_log_get_verbosity_thold() >= 3) {
                LOG_INF("encoded image %2zu/%zu within %8.2f ms, n_tokens = %d\n", i + 1, n_entries,
                        (ggml_time_us() - t_start) / 1000.0, result[i].n_tokens);
            }
            result[i].dummy_token = multimodal_dummy_token_generator--;
        }
    }
    // llava / glm, non-batching
    else if (clip_is_llava(ctx_clip) || clip_is_glm(ctx_clip)) {
        result.resize(1);

        int32_t n_tokens = 0;
        for (const auto & entry : batch.entries) {
            n_tokens += clip_n_output_tokens(ctx_clip, &entry);
        }

        // init
        result[0].n_tokens  = n_tokens;
        result[0].n_pos     = result[0].n_tokens;
        result[0].size      = batch.entries[0].get_size();
        result[0].grid_size = clip_image_size{ grid_x, grid_y };
        result[0].embed.resize(result[0].n_tokens * n_mmproj_embd);
        // encode
        const auto & entries   = batch.entries;
        const size_t n_entries = entries.size();
        for (size_t i = 0; i < n_entries; i++) {
            int32_t       n_entry_tokens = clip_n_output_tokens(ctx_clip, &entries[i]);
            const int64_t t_start        = ggml_time_us();
            std::vector<float> encoded_entry;
            bool          encoded        = clip_image_encode(ctx_clip, n_threads, &entries[i], encoded_entry);
            if (!encoded) {
                LOG_ERR("failed to encode image %2zu/%zu\n", i + 1, n_entries);
                return {};
            }
            std::copy(encoded_entry.begin(), encoded_entry.end(),
                      result[0].embed.begin() + i * n_entry_tokens * n_mmproj_embd);
            if (common_log_get_verbosity_thold() >= 3) {
                LOG_INF("encoded image %2zu/%zu within %8.2f ms, n_tokens = %d\n", i + 1, n_entries,
                        (ggml_time_us() - t_start) / 1000.0, n_entry_tokens);
            }
        }
        result[0].dummy_token = multimodal_dummy_token_generator--;
    }
    // others, batching
    else {
        result.resize(1);

        // init
        result[0].n_tokens  = clip_n_output_tokens(ctx_clip, &batch.entries[0]);
        result[0].n_pos     = result[0].n_tokens;
        result[0].size      = batch.entries[0].get_size();
        result[0].grid_size = clip_image_size{ grid_x, grid_y };
        result[0].embed.resize(result[0].n_tokens * n_mmproj_embd);
        // encode
        const int64_t t_start = ggml_time_us();
        bool          encoded = clip_image_batch_encode(ctx_clip, n_threads, &batch, result[0].embed);
        if (!encoded) {
            LOG_ERR("%s", "failed to encode image in batch\n");
            return {};
        }
        if (common_log_get_verbosity_thold() >= 3) {
            LOG_INF("encoded image in batch within %8.2f ms, n_tokens = %d\n", (ggml_time_us() - t_start) / 1000.0,
                    result[0].n_tokens);
        }
        if (clip_is_qwen2vl(ctx_clip)) {
            // Qwen2VL/Qwen3VL-family models use the actual projector output
            // grid for M-RoPE.  The preprocessed image can be resized or
            // padded, so deriving the grid from its pixel dimensions and
            // patch size produces positions that are larger than the
            // embedding sequence (notably for small images).
            const int32_t grid_x = clip_n_output_tokens_x(ctx_clip, &batch.entries[0]);
            const int32_t grid_y = clip_n_output_tokens_y(ctx_clip, &batch.entries[0]);
            result[0].grid_size = clip_image_size{ grid_x, grid_y };
            // M-RoPE advances the sequential position by the larger image
            // grid dimension, not by the height alone.  This is important
            // for non-square images and matches mtmd's decoder contract.
            result[0].n_pos     = std::max(grid_x, grid_y);
        }
        result[0].dummy_token = multimodal_dummy_token_generator--;
    }

    return result;
}

static inline std::vector<llama_multimodal_tokens> tokenize_audio(clip_ctx * ctx_clip, const int n_threads,
                                                                  const std::vector<float> & samples) {
    auto preprocessor = make_audio_preprocessor(ctx_clip);
    preprocessor->initialize();
    std::vector<mtmd_audio_mel> entries;
    if (!preprocessor->preprocess(samples.data(), samples.size(), entries)) {
        LOG_ERR("%s", "unable to preprocess audio\n");
        return {};
    }
    if (entries.empty()) {
        LOG_ERR("%s", "no audio chunks after preprocessing\n");
        return {};
    }

    const int32_t n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);

    std::vector<llama_multimodal_tokens> result;
    result.resize(entries.size());

    const size_t n_entries = entries.size();
    for (size_t i = 0; i < n_entries; i++) {
        clip_image_f32 mel_f32;
        mel_f32.set_size({static_cast<int>(entries[i].n_len), static_cast<int>(entries[i].n_mel)},
                         entries[i].data.empty(), true);
        mel_f32.cpy_buf(entries[i].data);
        // init
        result[i].n_tokens  = clip_n_output_tokens(ctx_clip, &mel_f32);
        result[i].n_pos     = result[i].n_tokens;
        result[i].is_audio  = true;
        result[i].size      = clip_image_size{ entries[i].n_len, entries[i].n_mel };
        result[i].grid_size = clip_image_size{ 1, 1 };
        result[i].embed.resize(result[i].n_tokens * n_mmproj_embd);
        // encode
        clip_image_f32_batch batch_f32;
        batch_f32.is_audio = true;
        batch_f32.entries.push_back(std::move(mel_f32));
        const int64_t t_start = ggml_time_us();
        bool          encoded = clip_image_batch_encode(ctx_clip, n_threads, &batch_f32, result[i].embed);
        if (!encoded) {
            LOG_ERR("failed to encode audio %2zu/%zu\n", i + 1, n_entries);
            return {};
        }
        if (common_log_get_verbosity_thold() >= 3) {
            LOG_INF("encoded audio %2zu/%zu within %8.2f ms, n_tokens = %d\n", i + 1, n_entries,
                    (ggml_time_us() - t_start) / 1000.0, result[i].n_tokens);
        }
        result[i].dummy_token = multimodal_dummy_token_generator--;
    }

    return result;
}
