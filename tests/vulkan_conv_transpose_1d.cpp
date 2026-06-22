#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/tts_model.h"

namespace {
struct conv_case {
    int k;
    int cout;
    int cin;
    int length;
    int stride;
    int padding;
    int groups;
};

std::vector<float> make_kernel(const conv_case & c) {
    std::vector<float> values((size_t)c.k * c.cout * c.cin);
    for (size_t i = 0; i < values.size(); ++i) {
        const int centered = (int)(i % 17) - 8;
        values[i] = (float)centered * 0.03125f;
    }
    return values;
}

std::vector<float> make_input(const conv_case & c) {
    std::vector<float> values((size_t)c.length * c.cin);
    for (size_t i = 0; i < values.size(); ++i) {
        const int centered = (int)(i % 13) - 6;
        values[i] = (float)centered * 0.125f;
    }
    return values;
}

std::vector<float> make_istft_input(int n_fft, int frames) {
    const int bins = n_fft / 2 + 1;
    std::vector<float> values((size_t)bins * frames * 2);
    for (int frame = 0; frame < frames; ++frame) {
        for (int bin = 0; bin < bins; ++bin) {
            const size_t magnitude_index = (size_t)bin + (size_t)frame * bins;
            const size_t phase_index = magnitude_index + (size_t)bins * frames;
            values[magnitude_index] = 0.02f + 0.001f * (float)((bin + frame) % 17);
            values[phase_index] = -1.0f + 0.05f * (float)((bin * 3 + frame) % 41);
        }
    }
    return values;
}

std::vector<float> make_stft_input(int length) {
    std::vector<float> values((size_t)length);
    for (int i = 0; i < length; ++i) {
        const float t = (float)i;
        values[(size_t)i] = 0.05f * std::sin(t * 0.017f) + 0.03f * std::cos(t * 0.071f);
    }
    return values;
}

std::vector<float> make_window(int n_fft) {
    std::vector<float> values;
    values.reserve((size_t)n_fft);
    hann_window((size_t)n_fft, values);
    return values;
}

std::vector<float> run_case(ggml_backend_t backend, const conv_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, c.cout, c.cin);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.length, c.cin);
    ggml_tensor * output = ggml_conv_transpose_1d_ex(ctx, kernel, input, c.stride, c.padding, 1, 0, c.groups);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    const std::vector<float> kernel_data = make_kernel(c);
    const std::vector<float> input_data = make_input(c);
    ggml_backend_tensor_set(kernel, kernel_data.data(), 0, kernel_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed on backend %s\n", ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t)output->ne[0] * output->ne[1]);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

std::vector<float> run_stft_case(ggml_backend_t backend) {
    constexpr int n_fft = 20;
    constexpr int hop = 5;
    constexpr int length = 138600;

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, length, 1);
    ggml_tensor * window = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_fft);
    ggml_tensor * output = stft(ctx, input, window, n_fft, hop, true, true);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    const std::vector<float> input_data = make_stft_input(length);
    const std::vector<float> window_data = make_window(n_fft);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(window, window_data.data(), 0, window_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "stft graph compute failed on backend %s\n", ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t)ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

std::vector<float> run_istft_case(ggml_backend_t backend) {
    constexpr int n_fft = 20;
    constexpr int hop = 5;
    constexpr int frames = 130;
    constexpr int bins = n_fft / 2 + 1;

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, bins, frames, 1, 2);
    ggml_tensor * window = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_fft);
    ggml_tensor * output = ggml_istft(ctx, input, window, n_fft, hop, true);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    const std::vector<float> input_data = make_istft_input(n_fft, frames);
    const std::vector<float> window_data = make_window(n_fft);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(window, window_data.data(), 0, window_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "istft graph compute failed on backend %s\n", ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t)output->ne[0] * output->ne[1]);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

bool compare_vectors(const char * name, const std::vector<float> & actual, const std::vector<float> & expected, double tolerance) {
    double max_abs = 0.0;
    double rms = 0.0;
    int worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double diff = (double)actual[i] - (double)expected[i];
        const double abs_diff = std::fabs(diff);
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
            worst = (int)i;
        }
        rms += diff * diff;
    }
    rms = std::sqrt(rms / (double)expected.size());

    if (max_abs > tolerance) {
        std::fprintf(stderr,
                     "%s mismatch: max_abs=%g rms=%g worst=%d actual=%f expected=%f\n",
                     name,
                     max_abs,
                     rms,
                     worst,
                     actual[(size_t)worst],
                     expected[(size_t)worst]);
        return false;
    }
    return true;
}

bool compare_vectors_rms(const char * name, const std::vector<float> & actual, const std::vector<float> & expected, double tolerance) {
    double max_abs = 0.0;
    double rms = 0.0;
    int worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double diff = (double)actual[i] - (double)expected[i];
        const double abs_diff = std::fabs(diff);
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
            worst = (int)i;
        }
        rms += diff * diff;
    }
    rms = std::sqrt(rms / (double)expected.size());

    if (rms > tolerance) {
        std::fprintf(stderr,
                     "%s mismatch: max_abs=%g rms=%g worst=%d actual=%f expected=%f\n",
                     name,
                     max_abs,
                     rms,
                     worst,
                     actual[(size_t)worst],
                     expected[(size_t)worst]);
        return false;
    }
    return true;
}

bool compare_case(ggml_backend_t vulkan, const conv_case & c, const char * name) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    std::vector<float> expected = run_case(cpu, c);
    ggml_backend_free(cpu);

    std::vector<float> actual = run_case(vulkan, c);

    return compare_vectors(name, actual, expected, 1e-4);
}
}

int main() {
    conv_case dense{
        /*.k       =*/ 4,
        /*.cout    =*/ 5,
        /*.cin     =*/ 3,
        /*.length  =*/ 7,
        /*.stride  =*/ 2,
        /*.padding =*/ 1,
        /*.groups  =*/ 1,
    };
    conv_case depthwise{
        /*.k       =*/ 4,
        /*.cout    =*/ 1,
        /*.cin     =*/ 5,
        /*.length  =*/ 7,
        /*.stride  =*/ 2,
        /*.padding =*/ 1,
        /*.groups  =*/ 5,
    };
    conv_case kokoro_up0{
        /*.k       =*/ 20,
        /*.cout    =*/ 256,
        /*.cin     =*/ 512,
        /*.length  =*/ 114,
        /*.stride  =*/ 10,
        /*.padding =*/ 5,
        /*.groups  =*/ 1,
    };
    conv_case kokoro_up1{
        /*.k       =*/ 12,
        /*.cout    =*/ 128,
        /*.cin     =*/ 256,
        /*.length  =*/ 1140,
        /*.stride  =*/ 6,
        /*.padding =*/ 3,
        /*.groups  =*/ 1,
    };

    ggml_backend_t vulkan = tts_backend_init_accelerator(false);
    if (!vulkan) {
        std::fprintf(stderr, "Vulkan backend unavailable; skipping conv_transpose_1d parity test\n");
        return 0;
    }

    const bool dense_ok = compare_case(vulkan, dense, "dense");
    const bool depthwise_ok = compare_case(vulkan, depthwise, "depthwise");
    const bool kokoro_up0_ok = compare_case(vulkan, kokoro_up0, "kokoro_up0");
    const bool kokoro_up1_ok = compare_case(vulkan, kokoro_up1, "kokoro_up1");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    std::vector<float> expected_stft = run_stft_case(cpu);
    std::vector<float> expected_istft = run_istft_case(cpu);
    ggml_backend_free(cpu);
    std::vector<float> actual_stft = run_stft_case(vulkan);
    std::vector<float> actual_istft = run_istft_case(vulkan);
    const bool stft_ok = compare_vectors_rms("aa_stft_kokoro_shape", actual_stft, expected_stft, 0.05);
    const bool istft_ok = compare_vectors("aa_istft", actual_istft, expected_istft, 1e-4);
    ggml_backend_free(vulkan);

    return dense_ok && depthwise_ok && kokoro_up0_ok && kokoro_up1_ok && stft_ok && istft_ok ? 0 : 1;
}
