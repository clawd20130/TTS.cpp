#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/tts_model.h"

namespace {
struct im2col_case {
    const char * name;
    bool is_2d;
    int input_w;
    int input_h;
    int channels;
    int batch;
    int kernel_w;
    int kernel_h;
    int out_channels;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int dilation_w;
    int dilation_h;
};

std::vector<float> make_values(size_t n, float scale, int mod, int salt) {
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) {
        const int v = (int)((i * 131u + (size_t)salt * 17u) % (size_t)mod) - mod / 2;
        values[i] = scale * (float)v;
    }
    return values;
}

std::vector<float> run_case(ggml_backend_t backend, const im2col_case & c) {
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

    ggml_tensor * input = nullptr;
    ggml_tensor * kernel = nullptr;
    if (c.is_2d) {
        input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.input_w, c.input_h, c.channels, c.batch);
        kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.kernel_w, c.kernel_h, c.channels, c.out_channels);
    } else {
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.input_w, c.channels, c.batch);
        kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.kernel_w, c.channels, c.out_channels);
    }
    ggml_tensor * output = ggml_im2col(
        ctx, kernel, input,
        c.stride_w, c.stride_h,
        c.pad_w, c.pad_h,
        c.dilation_w, c.dilation_h,
        c.is_2d, GGML_TYPE_F32);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate backend tensors for %s on %s\n",
                     c.name, ggml_backend_name(backend));
        std::abort();
    }

    const std::vector<float> input_data =
        make_values((size_t)ggml_nelements(input), 0.125f, 29, c.channels);
    const std::vector<float> kernel_data =
        make_values((size_t)ggml_nelements(kernel), 0.03125f, 31, c.kernel_w);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(kernel, kernel_data.data(), 0, kernel_data.size() * sizeof(float));

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

bool compare_case(ggml_backend_t vulkan, const im2col_case & c) {
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
    if (max_abs > 1e-6) {
        std::fprintf(stderr,
                     "%s mismatch: max_abs=%g rms=%g worst=%zu actual=%f expected=%f\n",
                     c.name, max_abs, rms, worst, actual[worst], expected[worst]);
        return false;
    }
    return true;
}
}

int main() {
    const im2col_case cases[] = {
        {
            "im2col_1d_padded_dilated",
            false,
            17, 1, 4, 2,
            5, 1, 3,
            1, 1,
            4, 0,
            2, 1,
        },
        {
            "im2col_1d_style_bert_conv_pre",
            false,
            60, 1, 192, 1,
            7, 1, 512,
            1, 1,
            3, 0,
            1, 1,
        },
        {
            "im2col_2d_padded",
            true,
            9, 7, 3, 2,
            3, 3, 5,
            2, 1,
            1, 2,
            1, 1,
        },
    };

    ggml_backend_t vulkan = tts_backend_init_accelerator(false);
    if (!vulkan) {
        std::fprintf(stderr, "Vulkan backend unavailable; skipping im2col parity test\n");
        return 0;
    }

    bool ok = true;
    for (const im2col_case & c : cases) {
        ok = compare_case(vulkan, c) && ok;
    }

    ggml_backend_free(vulkan);
    return ok ? 0 : 1;
}
