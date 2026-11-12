#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) return 1;

    ggml_init_params params = { 1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 32, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 32, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 32, 1, 1);
    ggml_tensor * output = ggml_sage_attn(ctx, q, k, v, 0.125f, GGML_SAGE_ATTN_AUTO);
    if (!ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr, "SageAttention unsupported on CUDA0\n");
        return 2;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 3;

    std::vector<float> input(64 * 32), result(64 * 32);
    std::vector<ggml_fp16_t> ones(64 * 32, ggml_fp32_to_fp16(1.0f));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = float(int(i % 23) - 11) / 11.0f;
    }
    ggml_backend_tensor_set(q, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(k, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(v, ones.data(), 0, ones.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 4;
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    for (float value : result) {
        if (!std::isfinite(value) || std::fabs(value - 1.0f) > 0.1f) {
            std::fprintf(stderr, "SageAttention output=%f expected=1\n", value);
            return 5;
        }
    }
    std::puts("SageAttention constant-value test passed on CUDA0");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
