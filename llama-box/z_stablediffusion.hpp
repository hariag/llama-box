#pragma once

// heads

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stable-diffusion.cpp/thirdparty/stb_image.h"
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stable-diffusion.cpp/thirdparty/stb_image_resize.h"
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stable-diffusion.cpp/model.h"
#include "stable-diffusion.cpp/stable-diffusion.h"
#include "stable-diffusion.cpp/thirdparty/stb_image_write.h"

#define SELF_PACKAGE 0
#include "z_utils.hpp"

// defines

using schedule_t = enum scheduler_t;

static constexpr int N_SAMPLE_METHODS        = SAMPLE_METHOD_COUNT;
static constexpr int N_SCHEDULES             = SCHEDULER_COUNT;
static constexpr sample_method_t EULER       = EULER_SAMPLE_METHOD;
static constexpr sample_method_t EULER_A     = EULER_A_SAMPLE_METHOD;
static constexpr schedule_t DISCRETE         = DISCRETE_SCHEDULER;

static inline sample_method_t sd_argument_to_sample_method(const char * value) {
    const auto method = str_to_sample_method(value == nullptr ? "" : value);
    if (method == SAMPLE_METHOD_COUNT) {
        throw std::invalid_argument("unknown stable-diffusion sample method");
    }
    return method;
}

static inline const char * sd_sample_method_to_argument(sample_method_t method) {
    return sd_sample_method_name(method);
}

static inline schedule_t sd_argument_to_schedule(const char * value) {
    if (value == nullptr || !strcmp(value, "default")) {
        return DISCRETE;
    }
    const auto scheduler = str_to_scheduler(value);
    if (scheduler == SCHEDULER_COUNT) {
        throw std::invalid_argument("unknown stable-diffusion scheduler");
    }
    return scheduler;
}

static inline const char * sd_schedule_to_argument(schedule_t scheduler) {
    return sd_scheduler_name(scheduler);
}

// types

struct stablediffusion_params_sampling {
    uint32_t         seed            = LLAMA_DEFAULT_SEED;
    int              height          = 1024;
    int              width           = 1024;
    float            guidance        = 3.5f;
    float            strength        = 0.0f;
    sample_method_t  sample_method   = static_cast<sample_method_t>(N_SAMPLE_METHODS);
    int              sampling_steps  = 0;
    float            cfg_scale       = 0.0f;
    float            slg_scale       = 0.0f;
    std::vector<int> slg_skip_layers = { 7, 8, 9 };
    float            slg_start       = 0.01;
    float            slg_end         = 0.2;
    schedule_t       schedule_method = DISCRETE;
    std::string      negative_prompt;
    float            control_strength   = 0.9f;
    bool             control_canny      = false;
    uint8_t *        control_img_buffer = nullptr;
    uint8_t *        init_img_buffer    = nullptr;
    uint8_t *        mask_img_buffer    = nullptr;
};

struct stablediffusion_params {
    stablediffusion_params_sampling sampling;

    int         max_batch_count            = 4;
    bool        text_encoder_model_offload = true;
    std::string clip_l_model;
    std::string clip_g_model;
    std::string t5xxl_model;
    std::string llm_model;
    std::string llm_vision_model;
    bool        qwen_image_zero_cond_t = false;
    float       flow_shift = 3.0f;
    bool        vae_model_offload = true;
    std::string vae_model;
    bool        vae_tiling = false;
    std::string taesd_model;
    std::string upscale_model;
    int         upscale_repeats       = 1;
    bool        control_model_offload = true;
    std::string control_net_model;
    bool        free_compute_immediately = false;

    // inherited from common_params
    std::string                           model;
    std::string                           model_alias;
    ggml_numa_strategy                    numa                    = GGML_NUMA_STRATEGY_DISABLED;
    int32_t                               n_parallel              = 1;
    uint32_t                              seed                    = LLAMA_DEFAULT_SEED;
    bool                                  warmup                  = true;
    bool                                  flash_attn              = false;
    int                                   n_threads               = 1;
    bool                                  lora_init_without_apply = false;
    std::vector<common_adapter_lora_info> lora_adapters           = {};
    float *                               tensor_split            = nullptr;
};

struct stablediffusion_sampling_stream {
    stablediffusion_sampling_stream(std::string prompt, stablediffusion_params_sampling params) :
        prompt(std::move(prompt)),
        params(std::move(params)) {}

    ~stablediffusion_sampling_stream() {
        if (images != nullptr) {
            if (images[0].data != nullptr) {
                free(images[0].data);
                images[0].data = nullptr;
            }
            free(images);
            images = nullptr;
        }
    }

    std::string                   prompt;
    stablediffusion_params_sampling params;
    std::vector<std::string>       lora_paths;
    std::vector<sd_lora_t>          loras;
    sd_image_t *                    images  = nullptr;
    int                             steps   = 0;
    bool                            sampled = false;
    bool                            failed  = false;
};

struct stablediffusion_generated_image {
    explicit stablediffusion_generated_image(int size, unsigned char * data) : size(size), data(data) {}

    ~stablediffusion_generated_image() {
        if (data != nullptr) {
            stbi_image_free(data);
            data = nullptr;
        }
    }

    int             size;
    unsigned char * data;
};

// implementations

class stablediffusion_context {
  public:
    stablediffusion_context(sd_ctx_t * sd_ctx, upscaler_ctx_t * upscaler_ctx, stablediffusion_params params) :
        sd_ctx(sd_ctx),
        upscaler_ctx(upscaler_ctx),
        params(params) {}

    ~stablediffusion_context();

    float               get_default_strength();
    sample_method_t     get_default_sample_method();
    int                 get_default_sampling_steps();
    float               get_default_cfg_scale();
    std::pair<int, int> get_default_image_size();
    void                apply_lora_adapters(std::vector<common_adapter_lora_info> & lora_adapters);
    std::unique_ptr<stablediffusion_sampling_stream> generate_stream(const char *                    prompt,
                                                                     stablediffusion_params_sampling sparams);
    bool                                             sample_stream(stablediffusion_sampling_stream * stream);
    std::pair<int, int>                              progress_stream(stablediffusion_sampling_stream * stream);
    std::unique_ptr<stablediffusion_generated_image> preview_image_stream(stablediffusion_sampling_stream * stream,
                                                                          bool faster = false);
    std::unique_ptr<stablediffusion_generated_image> result_image_stream(stablediffusion_sampling_stream * stream);

  private:
    sd_ctx_t *             sd_ctx       = nullptr;
    upscaler_ctx_t *       upscaler_ctx = nullptr;
    stablediffusion_params params;
    std::vector<std::string> active_lora_paths;
    std::vector<sd_lora_t>   active_loras;
};

stablediffusion_context::~stablediffusion_context() {
    if (sd_ctx != nullptr) {
        free_sd_ctx(sd_ctx);
        sd_ctx = nullptr;
    }
    if (upscaler_ctx != nullptr) {
        free_upscaler_ctx(upscaler_ctx);
        upscaler_ctx = nullptr;
    }
}

float stablediffusion_context::get_default_strength() {
    return 0.75f;
}

sample_method_t stablediffusion_context::get_default_sample_method() {
    const auto method = sd_get_default_sample_method(sd_ctx);
    if (method < SAMPLE_METHOD_COUNT) {
        return method;
    }
    return params.llm_model.empty() ? EULER_A : EULER;
}

int stablediffusion_context::get_default_sampling_steps() {
    return params.llm_model.empty() ? 20 : 40;
}

float stablediffusion_context::get_default_cfg_scale() {
    return params.llm_model.empty() ? 4.5f : 2.5f;
}

std::pair<int, int> stablediffusion_context::get_default_image_size() {
    return { 1024, 1024 };
}

void stablediffusion_context::apply_lora_adapters(std::vector<common_adapter_lora_info> & lora_adapters) {
    active_lora_paths.clear();
    active_loras.clear();
    active_lora_paths.reserve(lora_adapters.size());
    active_loras.reserve(lora_adapters.size());
    for (const auto & lora_adapter : lora_adapters) {
        active_lora_paths.push_back(lora_adapter.path);
    }
    for (size_t i = 0; i < lora_adapters.size(); ++i) {
        active_loras.push_back({ false, lora_adapters[i].scale, active_lora_paths[i].c_str() });
    }
}

std::unique_ptr<stablediffusion_sampling_stream> stablediffusion_context::generate_stream(
    const char * prompt, stablediffusion_params_sampling sparams) {
    auto stream = std::make_unique<stablediffusion_sampling_stream>(prompt == nullptr ? "" : prompt,
                                                                      std::move(sparams));
    stream->lora_paths = active_lora_paths;
    stream->loras.reserve(active_loras.size());
    for (size_t i = 0; i < active_loras.size(); ++i) {
        stream->loras.push_back({ active_loras[i].is_high_noise, active_loras[i].multiplier,
                                  stream->lora_paths[i].c_str() });
    }
    return stream;
}

bool stablediffusion_context::sample_stream(stablediffusion_sampling_stream * stream) {
    if (stream == nullptr || stream->sampled) {
        return false;
    }

    sd_img_gen_params_t generation;
    sd_img_gen_params_init(&generation);
    generation.loras          = stream->loras.data();
    generation.lora_count    = static_cast<uint32_t>(stream->loras.size());
    generation.prompt        = stream->prompt.c_str();
    generation.negative_prompt = stream->params.negative_prompt.c_str();
    generation.width         = stream->params.width;
    generation.height        = stream->params.height;
    generation.strength      = stream->params.strength;
    generation.seed          = stream->params.seed == LLAMA_DEFAULT_SEED ? -1 : stream->params.seed;
    generation.batch_count   = 1;
    generation.sample_params.scheduler = stream->params.schedule_method;
    generation.sample_params.sample_method = stream->params.sample_method;
    generation.sample_params.sample_steps = stream->params.sampling_steps;
    generation.sample_params.guidance.txt_cfg = stream->params.cfg_scale;
    generation.sample_params.guidance.img_cfg = stream->params.cfg_scale;
    generation.sample_params.guidance.distilled_guidance = stream->params.guidance;
    generation.sample_params.guidance.slg.scale = stream->params.slg_scale;
    generation.sample_params.guidance.slg.layer_start = stream->params.slg_start;
    generation.sample_params.guidance.slg.layer_end = stream->params.slg_end;
    generation.sample_params.guidance.slg.layers = stream->params.slg_skip_layers.data();
    generation.sample_params.guidance.slg.layer_count = stream->params.slg_skip_layers.size();
    generation.control_strength = stream->params.control_strength;
    generation.vae_tiling_params.enabled = params.vae_tiling;

    sd_image_t input_image = { uint32_t(std::max(stream->params.width, 0)),
                               uint32_t(std::max(stream->params.height, 0)), 3, stream->params.init_img_buffer };
    sd_image_t mask_image = { uint32_t(std::max(stream->params.width, 0)),
                              uint32_t(std::max(stream->params.height, 0)), 1, stream->params.mask_img_buffer };
    sd_image_t control_image = { uint32_t(std::max(stream->params.width, 0)),
                                 uint32_t(std::max(stream->params.height, 0)), 3,
                                 stream->params.control_img_buffer };
    if (!params.llm_model.empty() && input_image.data != nullptr) {
        generation.ref_images = &input_image;
        generation.ref_images_count = 1;
        generation.auto_resize_ref_image = true;
        generation.increase_ref_index = false;
    } else {
        generation.init_image = input_image;
        generation.mask_image = mask_image;
    }
    if (control_image.data != nullptr) {
        generation.control_image = control_image;
    }

    stream->images = generate_image(sd_ctx, &generation);
    stream->steps = generation.sample_params.sample_steps > 0 ? generation.sample_params.sample_steps : 1;
    stream->sampled = true;
    stream->failed = stream->images == nullptr || stream->images[0].data == nullptr;
    return false;
}

std::pair<int, int> stablediffusion_context::progress_stream(stablediffusion_sampling_stream * stream) {
    if (stream == nullptr || stream->failed) {
        return { 0, 0 };
    }

    return { stream->sampled ? stream->steps : 0, stream->steps };
}

std::unique_ptr<stablediffusion_generated_image> stablediffusion_context::preview_image_stream(
    stablediffusion_sampling_stream * stream, bool faster) {
    if (stream == nullptr || !stream->sampled || stream->failed || stream->images == nullptr) {
        return nullptr;
    }

    sd_image_t img = stream->images[0];
    if (img.data == nullptr) {
        return nullptr;
    }

    int             size = 0;
    unsigned char * data = stbi_write_png_to_mem((stbi_uc *) img.data, 0, (int) img.width, (int) img.height,
                                                 (int) img.channel, &size, nullptr);
    if (data == nullptr || size <= 0) {
        return nullptr;
    }

    return std::make_unique<stablediffusion_generated_image>(size, data);
}

std::unique_ptr<stablediffusion_generated_image> stablediffusion_context::result_image_stream(
    stablediffusion_sampling_stream * stream) {
    if (stream == nullptr || !stream->sampled || stream->failed || stream->images == nullptr) {
        return nullptr;
    }

    sd_image_t img = stream->images[0];
    if (img.data == nullptr) {
        return nullptr;
    }

    int upscale_factor = 4;
    if (upscaler_ctx != nullptr && params.upscale_repeats > 0) {
        for (int u = 0; u < params.upscale_repeats; ++u) {
            sd_image_t upscaled_img = upscale(upscaler_ctx, img, upscale_factor);
            if (upscaled_img.data == nullptr) {
                LOG_WRN("%s: failed to upscale image\n", __func__);
                break;
            }
            stbi_image_free(img.data);
            if (img.data == stream->images[0].data) {
                stream->images[0].data = nullptr;
            }
            img = upscaled_img;
        }
    }

    int             size  = 0;
    unsigned char * data  = stbi_write_png_to_mem((stbi_uc *) img.data, 0, (int) img.width, (int) img.height,
                                                  (int) img.channel, &size, nullptr);
    if (data == nullptr || size <= 0) {
        if (img.data != stream->images[0].data) {
            stbi_image_free(img.data);
        }
        return nullptr;
    }

    if (img.data != stream->images[0].data) {
        stbi_image_free(img.data);
    }

    return std::make_unique<stablediffusion_generated_image>(size, data);
}

struct common_sd_init_result {
    std::unique_ptr<stablediffusion_context> context;
};

common_sd_init_result common_sd_init_from_params(stablediffusion_params params) {
    common_sd_init_result result;

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.diffusion_model_path = params.model.c_str();
    ctx_params.clip_l_path = params.clip_l_model.c_str();
    ctx_params.clip_g_path = params.clip_g_model.c_str();
    ctx_params.t5xxl_path = params.t5xxl_model.c_str();
    ctx_params.llm_path = params.llm_model.c_str();
    ctx_params.llm_vision_path = params.llm_vision_model.c_str();
    ctx_params.vae_path = params.vae_model.c_str();
    ctx_params.taesd_path = params.taesd_model.c_str();
    ctx_params.control_net_path = params.control_net_model.c_str();
    ctx_params.vae_decode_only = false;
    ctx_params.free_params_immediately = false;
    ctx_params.n_threads = params.n_threads;
    ctx_params.rng_type = CUDA_RNG;
    ctx_params.sampler_rng_type = RNG_TYPE_COUNT;
    ctx_params.offload_params_to_cpu = params.text_encoder_model_offload;
    ctx_params.keep_control_net_on_cpu = params.control_model_offload;
    ctx_params.keep_vae_on_cpu = params.vae_model_offload;
    ctx_params.diffusion_flash_attn = params.flash_attn;
    ctx_params.qwen_image_zero_cond_t = params.qwen_image_zero_cond_t;
    ctx_params.flow_shift = params.llm_model.empty() ? INFINITY : params.flow_shift;

    sd_ctx_t * sd_ctx = new_sd_ctx(&ctx_params);
    if (sd_ctx == nullptr) {
        LOG_ERR("%s: failed to create stable diffusion context\n", __func__);
        return result;
    }

    upscaler_ctx_t * upscaler_ctx = nullptr;
    if (!params.upscale_model.empty()) {
        upscaler_ctx = new_upscaler_ctx(params.upscale_model.c_str(), params.vae_model_offload, false,
                                        params.n_threads, 128);
        if (upscaler_ctx == nullptr) {
            LOG_ERR("%s: failed to create upscaler context\n", __func__);
            free_sd_ctx(sd_ctx);
            return result;
        }
    }

    std::unique_ptr<stablediffusion_context> sc =
        std::make_unique<stablediffusion_context>(sd_ctx, upscaler_ctx, params);
    if (!params.lora_init_without_apply && !params.lora_adapters.empty()) {
        sc->apply_lora_adapters(params.lora_adapters);
    }
    if (params.warmup) {
        LOG_WRN("%s: warming up the model with an empty run - please wait ... (--no-warmup to disable)\n", __func__);

        stablediffusion_params_sampling wparams = params.sampling;
        wparams.sampling_steps                  = 1;  // sample only once
        wparams.sample_method                   = EULER;
        wparams.schedule_method                 = DISCRETE;

        std::unique_ptr<stablediffusion_sampling_stream> stream = sc->generate_stream("a lovely cat", wparams);
        sc->sample_stream(stream.get());
        sc->result_image_stream(stream.get());
    }

    result.context = std::move(sc);
    return result;
}

static void sd_log_set(sd_log_cb_t cb, void * data) {
    sd_set_log_callback(cb, data);
}

static void sd_progress_set(sd_progress_cb_t cb, void * data) {
    sd_set_progress_callback(cb, data);
}

static ggml_log_level sd_log_level_to_ggml_log_level(sd_log_level_t level) {
    switch (level) {
        case SD_LOG_INFO:
            return GGML_LOG_LEVEL_INFO;
        case SD_LOG_WARN:
            return GGML_LOG_LEVEL_WARN;
        case SD_LOG_ERROR:
            return GGML_LOG_LEVEL_ERROR;
        default:
            return GGML_LOG_LEVEL_DEBUG;
    }
}
