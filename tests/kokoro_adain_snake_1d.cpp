#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/tts_model.h"

namespace {
struct adain_snake_case {
    int channels;
    int length;
    int batch;
};

std::vector<float> make_values(size_t n, float scale, int mod) {
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = scale * (float)((int)(i % (size_t)mod) - mod / 2);
    }
    return values;
}

std::vector<float> make_alpha_values(int channels) {
    std::vector<float> values((size_t)channels);
    const std::vector<float> seed = make_values((size_t)channels, 0.025f, 11);
    for (int i = 0; i < channels; ++i) {
        values[(size_t)i] = 0.5f + std::fabs(seed[(size_t)i]);
    }
    return values;
}

bool compare_vectors(const char * name, const adain_snake_case & c, const std::vector<float> & actual, const std::vector<float> & expected) {
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        if (diff > 1e-4f) {
            std::fprintf(stderr,
                         "%s mismatch at %zu: reference=%f fused=%f diff=%f "
                         "case channels=%d length=%d batch=%d\n",
                         name, i, expected[i], actual[i], diff, c.channels, c.length, c.batch);
            return false;
        }
    }
    return true;
}

bool run_cpu_composition_case(const adain_snake_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * alpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, c.channels);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.channels, c.length, c.batch);
    ggml_tensor * gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.channels, 1);
    ggml_tensor * beta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.channels, 1);

    const std::vector<float> alpha_data = make_alpha_values(c.channels);
    const std::vector<float> input_data = make_values((size_t)c.channels * c.length * c.batch, 0.125f, 17);
    const std::vector<float> gamma_data = make_values((size_t)c.channels, 0.03125f, 13);
    const std::vector<float> beta_data = make_values((size_t)c.channels, 0.0625f, 19);
    std::memcpy(alpha->data, alpha_data.data(), alpha_data.size() * sizeof(float));
    std::memcpy(input->data, input_data.data(), input_data.size() * sizeof(float));
    std::memcpy(gamma->data, gamma_data.data(), gamma_data.size() * sizeof(float));
    std::memcpy(beta->data, beta_data.data(), beta_data.size() * sizeof(float));

    ggml_tensor * adain = ggml_add(ctx, ggml_add(ctx, input, ggml_mul(ctx, input, gamma)), beta);
    ggml_tensor * reference = snake_1d(ctx, alpha, ggml_cont(ctx, ggml_transpose(ctx, adain)));
    ggml_tensor * fused = ggml_kokoro_adain_snake_1d_t(ctx, alpha, input, gamma, beta);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, fused);

    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "adain snake graph compute failed\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<float> expected((size_t)ggml_nelements(reference));
    std::vector<float> actual((size_t)ggml_nelements(fused));
    std::memcpy(expected.data(), reference->data, expected.size() * sizeof(float));
    std::memcpy(actual.data(), fused->data, actual.size() * sizeof(float));
    const bool ok = compare_vectors("cpu", c, actual, expected);

    ggml_free(ctx);
    return ok;
}

std::vector<float> run_fused_backend(ggml_backend_t backend, const adain_snake_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * alpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, c.channels);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.channels, c.length, c.batch);
    ggml_tensor * gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.channels, 1);
    ggml_tensor * beta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.channels, 1);
    ggml_tensor * fused = ggml_kokoro_adain_snake_1d_t(ctx, alpha, input, gamma, beta);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, fused);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    const std::vector<float> alpha_data = make_alpha_values(c.channels);
    const std::vector<float> input_data = make_values((size_t)c.channels * c.length * c.batch, 0.125f, 17);
    const std::vector<float> gamma_data = make_values((size_t)c.channels, 0.03125f, 13);
    const std::vector<float> beta_data = make_values((size_t)c.channels, 0.0625f, 19);
    ggml_backend_tensor_set(alpha, alpha_data.data(), 0, alpha_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(gamma, gamma_data.data(), 0, gamma_data.size() * sizeof(float));
    ggml_backend_tensor_set(beta, beta_data.data(), 0, beta_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend adain snake graph compute failed on %s\n", ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t)ggml_nelements(fused));
    ggml_backend_tensor_get(fused, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}
}

int main() {
    const adain_snake_case cases[] = {
        {4, 9, 1},
        {16, 37, 1},
        {8, 11, 2},
        {256, 1140, 1},
        {128, 6841, 1},
    };

    for (const adain_snake_case & c : cases) {
        if (!run_cpu_composition_case(c)) {
            return 1;
        }
    }

    ggml_backend_t vulkan = tts_backend_init_accelerator(false);
    if (!vulkan) {
        std::fprintf(stderr, "Vulkan backend unavailable; skipping kokoro adain snake Vulkan parity test\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    bool ok = true;
    for (const adain_snake_case & c : cases) {
        const std::vector<float> expected = run_fused_backend(cpu, c);
        const std::vector<float> actual = run_fused_backend(vulkan, c);
        ok = compare_vectors("vulkan", c, actual, expected) && ok;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(vulkan);

    return ok ? 0 : 1;
}
