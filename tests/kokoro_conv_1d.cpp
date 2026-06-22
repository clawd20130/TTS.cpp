#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/tts_model.h"

namespace {
struct conv_case {
    int kernel;
    int in_channels;
    int out_channels;
    int length;
    int batch;
    int stride;
    int padding;
    int dilation;
};

std::vector<float> make_values(size_t n, float scale, int mod) {
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = scale * (float)((int)(i % (size_t)mod) - mod / 2);
    }
    return values;
}

bool run_case(const conv_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.kernel, c.in_channels, c.out_channels);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.length, c.in_channels, c.batch);

    const std::vector<float> weight_data = make_values((size_t)c.kernel * c.in_channels * c.out_channels, 0.03125f, 19);
    const std::vector<float> input_data = make_values((size_t)c.length * c.in_channels * c.batch, 0.125f, 17);
    std::memcpy(weight->data, weight_data.data(), weight_data.size() * sizeof(float));
    std::memcpy(input->data, input_data.data(), input_data.size() * sizeof(float));

    ggml_tensor * reference = ggml_conv_1d(ctx, weight, input, c.stride, c.padding, c.dilation);
    ggml_tensor * fused = ggml_kokoro_conv_1d(ctx, weight, input, c.stride, c.padding, c.dilation);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, fused);

    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "conv graph compute failed\n");
        ggml_free(ctx);
        return false;
    }

    if (!ggml_are_same_shape(reference, fused)) {
        std::fprintf(stderr, "shape mismatch\n");
        ggml_free(ctx);
        return false;
    }

    const float * ref = static_cast<const float *>(reference->data);
    const float * got = static_cast<const float *>(fused->data);
    for (int64_t i = 0; i < ggml_nelements(reference); ++i) {
        const float diff = std::fabs(ref[i] - got[i]);
        if (diff > 1e-5f) {
            std::fprintf(stderr, "conv mismatch at %lld: reference=%f fused=%f diff=%f\n",
                         (long long)i, ref[i], got[i], diff);
            ggml_free(ctx);
            return false;
        }
    }

    ggml_free(ctx);
    return true;
}

std::vector<float> run_fused_backend(ggml_backend_t backend, const conv_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.kernel, c.in_channels, c.out_channels);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.length, c.in_channels, c.batch);
    ggml_tensor * fused = ggml_kokoro_conv_1d(ctx, weight, input, c.stride, c.padding, c.dilation);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, fused);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    const std::vector<float> weight_data = make_values((size_t)c.kernel * c.in_channels * c.out_channels, 0.03125f, 19);
    const std::vector<float> input_data = make_values((size_t)c.length * c.in_channels * c.batch, 0.125f, 17);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend conv graph compute failed on %s\n", ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t)ggml_nelements(fused));
    ggml_backend_tensor_get(fused, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

bool compare_vectors(const conv_case & c, const std::vector<float> & actual, const std::vector<float> & expected) {
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        if (diff > 1e-4f) {
            std::fprintf(stderr,
                         "vulkan conv mismatch at %zu: reference=%f fused=%f diff=%f "
                         "case k=%d ic=%d oc=%d len=%d batch=%d stride=%d pad=%d dil=%d\n",
                         i, expected[i], actual[i], diff,
                         c.kernel, c.in_channels, c.out_channels, c.length, c.batch,
                         c.stride, c.padding, c.dilation);
            return false;
        }
    }
    return true;
}
}

int main() {
    const conv_case cases[] = {
        {3, 2, 4, 8, 1, 1, 1, 1},
        {5, 3, 2, 11, 2, 1, 2, 1},
        {3, 4, 3, 13, 2, 2, 1, 1},
        {3, 3, 3, 17, 1, 1, 2, 2},
    };

    for (const conv_case & c : cases) {
        if (!run_case(c)) {
            return 1;
        }
    }

    ggml_backend_t vulkan = tts_backend_init_accelerator(false);
    if (!vulkan) {
        std::fprintf(stderr, "Vulkan backend unavailable; skipping kokoro conv Vulkan parity test\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    bool ok = true;
    for (const conv_case & c : cases) {
        const std::vector<float> expected = run_fused_backend(cpu, c);
        const std::vector<float> actual = run_fused_backend(vulkan, c);
        ok = compare_vectors(c, actual, expected) && ok;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(vulkan);

    if (!ok) {
        return 1;
    }
    return 0;
}
