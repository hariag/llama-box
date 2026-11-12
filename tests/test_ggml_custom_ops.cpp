#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * name = argc > 1 ? argv[1] : "CPU";
    ggml_backend_t backend = ggml_backend_init_by_name(name, nullptr);
    if (!backend) {
        std::fprintf(stderr, "backend unavailable: %s\n", name);
        return 1;
    }

    ggml_init_params params = { 1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * hadamard = ggml_regular_hadamard(ctx, input, 256);
    ggml_tensor * quantized = ggml_quantize_i8_convrot(ctx, input, 256);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 256, 4);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * output = ggml_mul_mat_i8_tensorwise(ctx, weight, quantized, scale, nullptr, 256);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, hadamard);
    ggml_build_forward_expand(graph, output);

    if (!ggml_backend_supports_op(backend, hadamard) || !ggml_backend_supports_op(backend, quantized) ||
        !ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr, "custom op unsupported on %s\n", name);
        return 2;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 3;

    std::vector<float> x(256, 0.0f);
    std::vector<int8_t> weights(256 * 4, 1);
    float scales[] = { 0.5f, 0.5f, 0.5f, 0.5f };
    x[0] = 1.0f;
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(scale, scales, 0, sizeof(scales));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 4;

    float transformed[256], result[4];
    ggml_backend_tensor_get(hadamard, transformed, 0, sizeof(transformed));
    ggml_backend_tensor_get(output, result, 0, sizeof(result));
    for (int i = 0; i < 256; ++i) {
        int digits = i;
        int negative = 0;
        for (int j = 0; j < 4; ++j) {
            negative += (digits % 4 == 3);
            digits /= 4;
        }
        const float expected = negative % 2 ? -0.0625f : 0.0625f;
        if (std::fabs(transformed[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "hadamard[%d]=%f expected=%f\n", i, transformed[i], expected);
            return 5;
        }
    }
    for (float value : result) {
        if (std::fabs(value - 0.5f) > 0.02f) {
            std::fprintf(stderr, "matmul=%f expected=0.5\n", value);
            return 6;
        }
    }
    std::printf("custom ops passed on %s\n", name);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
