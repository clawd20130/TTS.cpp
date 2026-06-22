#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/tts_model.h"

namespace {
struct transpose_case {
    const char * name;
    int input_ne0;
    int input_ne1;
};

std::vector<float> make_values(size_t n, int salt) {
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) {
        const int v = (int)((i * 131u + (size_t)salt * 17u) % 257u) - 128;
        values[i] = (float)v * 0.00390625f;
    }
    return values;
}

std::vector<float> run_case(ggml_backend_t backend, const transpose_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        std::abort();
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.input_ne0, c.input_ne1);
    ggml_tensor * output = ggml_cont(ctx, ggml_transpose(ctx, input));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate backend tensors for %s on %s\n",
                     c.name, ggml_backend_name(backend));
        std::abort();
    }

    const std::vector<float> input_data = make_values((size_t)ggml_nelements(input), c.input_ne0 + c.input_ne1);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed for %s on %s\n",
                     c.name, ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t)ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

bool compare_case(ggml_backend_t vulkan, const transpose_case & c) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    const std::vector<float> expected = run_case(cpu, c);
    ggml_backend_free(cpu);

    const std::vector<float> actual = run_case(vulkan, c);
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s size mismatch: actual=%zu expected=%zu\n",
                     c.name, actual.size(), expected.size());
        return false;
    }

    double max_abs = 0.0;
    double rms = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double diff = (double)actual[i] - (double)expected[i];
        const double abs_diff = std::fabs(diff);
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
            worst = i;
        }
        rms += diff * diff;
    }
    rms = std::sqrt(rms / (double)expected.size());
    if (max_abs != 0.0) {
        std::fprintf(stderr,
                     "%s mismatch: max_abs=%g rms=%g worst=%zu actual=%f expected=%f\n",
                     c.name, max_abs, rms, worst, actual[worst], expected[worst]);
        return false;
    }
    return true;
}
}

int main() {
    const transpose_case cases[] = {
        { "small_odd", 5, 7 },
        { "tile_edge", 17, 16 },
        { "kokoro_channels_time_256_4620", 256, 4620 },
        { "kokoro_channels_time_128_27721", 128, 27721 },
        { "kokoro_time_channels_4620_256", 4620, 256 },
        { "kokoro_time_channels_27721_128", 27721, 128 },
    };

    ggml_backend_t vulkan = tts_backend_init_accelerator(false);
    if (!vulkan) {
        std::fprintf(stderr, "Vulkan backend unavailable; skipping transpose-cont parity test\n");
        return 0;
    }

    bool ok = true;
    for (const transpose_case & c : cases) {
        ok = compare_case(vulkan, c) && ok;
    }

    ggml_backend_free(vulkan);
    return ok ? 0 : 1;
}
