#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/tts_model.h"

namespace {
struct lstm_case {
    int hidden;
    int sequence;
    bool reversed;
};

std::vector<float> make_values(size_t n, float scale, int mod, int salt) {
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) {
        const int v = (int)((i * 131u + (size_t)salt * 17u) % (size_t)mod) - mod / 2;
        values[i] = scale * (float)v;
    }
    return values;
}

ggml_tensor * gate_view(ggml_context * ctx, ggml_tensor * gates, int gate, int64_t hidden, int64_t time_index) {
    const size_t gate_offset = (size_t)gate * (size_t)hidden * gates->nb[0];
    return ggml_view_3d(ctx,
                        gates,
                        hidden,
                        1,
                        gates->ne[2],
                        gates->nb[0],
                        gates->nb[1],
                        gates->nb[1] * time_index + gate_offset);
}

ggml_tensor * build_reference_lstm_scan(
        ggml_context * ctx,
        ggml_cgraph * graph,
        ggml_tensor * input_gates,
        ggml_tensor * recurrent_weights,
        ggml_tensor * recurrent_biases,
        ggml_tensor * h0,
        ggml_tensor * c0,
        const lstm_case & c) {
    ggml_tensor * h = h0;
    ggml_tensor * state = c0;
    ggml_tensor * outputs = nullptr;

    for (int index = 0; index < c.sequence; ++index) {
        const int t = c.reversed ? (c.sequence - 1 - index) : index;
        ggml_tensor * recurrent_gates = ggml_add(ctx, ggml_mul_mat(ctx, recurrent_weights, h), recurrent_biases);

        ggml_tensor * i = ggml_sigmoid(ctx, ggml_add(ctx, gate_view(ctx, input_gates, 0, c.hidden, t),
                                                     gate_view(ctx, recurrent_gates, 0, c.hidden, 0)));
        ggml_tensor * f = ggml_sigmoid(ctx, ggml_add(ctx, gate_view(ctx, input_gates, 1, c.hidden, t),
                                                     gate_view(ctx, recurrent_gates, 1, c.hidden, 0)));
        ggml_tensor * g = ggml_tanh(ctx, ggml_add(ctx, gate_view(ctx, input_gates, 2, c.hidden, t),
                                                  gate_view(ctx, recurrent_gates, 2, c.hidden, 0)));
        ggml_tensor * o = ggml_sigmoid(ctx, ggml_add(ctx, gate_view(ctx, input_gates, 3, c.hidden, t),
                                                     gate_view(ctx, recurrent_gates, 3, c.hidden, 0)));

        state = ggml_add(ctx, ggml_mul(ctx, f, state), ggml_mul(ctx, i, g));
        h = ggml_mul(ctx, ggml_tanh(ctx, state), o);

        if (outputs == nullptr) {
            outputs = h;
        } else {
            outputs = c.reversed ? ggml_concat(ctx, h, outputs, 1) : ggml_concat(ctx, outputs, h, 1);
        }
        ggml_build_forward_expand(graph, outputs);
    }
    return outputs;
}

void fill_inputs(const lstm_case & c,
                 ggml_tensor * input_gates,
                 ggml_tensor * recurrent_weights,
                 ggml_tensor * recurrent_biases,
                 ggml_tensor * h0,
                 ggml_tensor * c0) {
    const std::vector<float> input_data =
        make_values((size_t)4 * c.hidden * c.sequence, 0.03125f, 31, c.hidden + c.sequence);
    const std::vector<float> weight_data =
        make_values((size_t)c.hidden * 4 * c.hidden, 0.015625f, 37, c.hidden);
    const std::vector<float> bias_data =
        make_values((size_t)4 * c.hidden, 0.0625f, 29, c.sequence);
    const std::vector<float> h0_data =
        make_values((size_t)c.hidden, 0.03125f, 23, c.hidden * 3);
    const std::vector<float> c0_data =
        make_values((size_t)c.hidden, 0.03125f, 19, c.sequence * 5);

    std::memcpy(input_gates->data, input_data.data(), input_data.size() * sizeof(float));
    std::memcpy(recurrent_weights->data, weight_data.data(), weight_data.size() * sizeof(float));
    std::memcpy(recurrent_biases->data, bias_data.data(), bias_data.size() * sizeof(float));
    std::memcpy(h0->data, h0_data.data(), h0_data.size() * sizeof(float));
    std::memcpy(c0->data, c0_data.data(), c0_data.size() * sizeof(float));
}

bool compare_vectors(const char * name,
                     const lstm_case & c,
                     const std::vector<float> & actual,
                     const std::vector<float> & expected,
                     float tolerance) {
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
    if (max_abs > tolerance) {
        std::fprintf(stderr,
                     "%s mismatch: hidden=%d sequence=%d reversed=%d "
                     "max_abs=%g rms=%g worst=%zu actual=%f expected=%f\n",
                     name, c.hidden, c.sequence, c.reversed ? 1 : 0,
                     max_abs, rms, worst, actual[worst], expected[worst]);
        return false;
    }
    return true;
}

bool run_cpu_reference_case(const lstm_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 512 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * input_gates = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4 * c.hidden, c.sequence);
    ggml_tensor * recurrent_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden, 4 * c.hidden);
    ggml_tensor * recurrent_biases = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * c.hidden);
    ggml_tensor * h0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.hidden);
    ggml_tensor * c0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.hidden);
    fill_inputs(c, input_gates, recurrent_weights, recurrent_biases, h0, c0);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 20000, false);
    ggml_tensor * reference = build_reference_lstm_scan(ctx, graph, input_gates, recurrent_weights, recurrent_biases, h0, c0, c);
    ggml_tensor * fused = ggml_kokoro_lstm_scan(ctx, input_gates, recurrent_weights, recurrent_biases, h0, c0, c.reversed);
    ggml_build_forward_expand(graph, fused);

    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "lstm scan graph compute failed\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<float> expected((size_t)ggml_nelements(reference));
    std::vector<float> actual((size_t)ggml_nelements(fused));
    std::memcpy(expected.data(), reference->data, expected.size() * sizeof(float));
    std::memcpy(actual.data(), fused->data, actual.size() * sizeof(float));

    const bool ok = compare_vectors("cpu lstm scan", c, actual, expected, 1e-5f);
    ggml_free(ctx);
    return ok;
}

std::vector<float> run_fused_backend(ggml_backend_t backend, const lstm_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * input_gates = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4 * c.hidden, c.sequence);
    ggml_tensor * recurrent_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden, 4 * c.hidden);
    ggml_tensor * recurrent_biases = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * c.hidden);
    ggml_tensor * h0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.hidden);
    ggml_tensor * c0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.hidden);
    ggml_tensor * fused = ggml_kokoro_lstm_scan(ctx, input_gates, recurrent_weights, recurrent_biases, h0, c0, c.reversed);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, fused);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate backend tensors for %s\n", ggml_backend_name(backend));
        std::abort();
    }

    const std::vector<float> input_data =
        make_values((size_t)4 * c.hidden * c.sequence, 0.03125f, 31, c.hidden + c.sequence);
    const std::vector<float> weight_data =
        make_values((size_t)c.hidden * 4 * c.hidden, 0.015625f, 37, c.hidden);
    const std::vector<float> bias_data =
        make_values((size_t)4 * c.hidden, 0.0625f, 29, c.sequence);
    const std::vector<float> h0_data =
        make_values((size_t)c.hidden, 0.03125f, 23, c.hidden * 3);
    const std::vector<float> c0_data =
        make_values((size_t)c.hidden, 0.03125f, 19, c.sequence * 5);

    ggml_backend_tensor_set(input_gates, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(recurrent_weights, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(recurrent_biases, bias_data.data(), 0, bias_data.size() * sizeof(float));
    ggml_backend_tensor_set(h0, h0_data.data(), 0, h0_data.size() * sizeof(float));
    ggml_backend_tensor_set(c0, c0_data.data(), 0, c0_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend lstm scan graph compute failed on %s\n", ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t)ggml_nelements(fused));
    ggml_backend_tensor_get(fused, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

std::vector<float> run_reference_backend(ggml_backend_t backend, const lstm_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 512 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * input_gates = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4 * c.hidden, c.sequence);
    ggml_tensor * recurrent_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden, 4 * c.hidden);
    ggml_tensor * recurrent_biases = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * c.hidden);
    ggml_tensor * h0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.hidden);
    ggml_tensor * c0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.hidden);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 20000, false);
    ggml_tensor * reference =
        build_reference_lstm_scan(ctx, graph, input_gates, recurrent_weights, recurrent_biases, h0, c0, c);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate reference backend tensors for %s\n", ggml_backend_name(backend));
        std::abort();
    }

    const std::vector<float> input_data =
        make_values((size_t)4 * c.hidden * c.sequence, 0.03125f, 31, c.hidden + c.sequence);
    const std::vector<float> weight_data =
        make_values((size_t)c.hidden * 4 * c.hidden, 0.015625f, 37, c.hidden);
    const std::vector<float> bias_data =
        make_values((size_t)4 * c.hidden, 0.0625f, 29, c.sequence);
    const std::vector<float> h0_data =
        make_values((size_t)c.hidden, 0.03125f, 23, c.hidden * 3);
    const std::vector<float> c0_data =
        make_values((size_t)c.hidden, 0.03125f, 19, c.sequence * 5);

    ggml_backend_tensor_set(input_gates, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(recurrent_weights, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(recurrent_biases, bias_data.data(), 0, bias_data.size() * sizeof(float));
    ggml_backend_tensor_set(h0, h0_data.data(), 0, h0_data.size() * sizeof(float));
    ggml_backend_tensor_set(c0, c0_data.data(), 0, c0_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend expanded lstm graph compute failed on %s\n", ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t)ggml_nelements(reference));
    ggml_backend_tensor_get(reference, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}
}

int main() {
    const lstm_case cases[] = {
        {4, 5, false},
        {4, 5, true},
        {16, 7, false},
        {16, 7, true},
        {64, 9, false},
        {64, 9, true},
        {256, 78, false},
        {256, 78, true},
        {256, 231, false},
        {256, 231, true},
    };

    bool ok = true;
    for (const lstm_case & c : cases) {
        ok = run_cpu_reference_case(c) && ok;
    }
    if (!ok) {
        return 1;
    }

    ggml_backend_t vulkan = tts_backend_init_accelerator(false);
    if (!vulkan) {
        std::fprintf(stderr, "Vulkan backend unavailable; skipping kokoro lstm scan Vulkan parity test\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    for (const lstm_case & c : cases) {
        const std::vector<float> expected = run_fused_backend(cpu, c);
        const std::vector<float> actual = run_fused_backend(vulkan, c);
        ok = compare_vectors("vulkan lstm scan", c, actual, expected, 1e-4f) && ok;

        const std::vector<float> expanded = run_reference_backend(vulkan, c);
        ok = compare_vectors("vulkan expanded vs scan", c, actual, expanded, 1e-4f) && ok;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(vulkan);

    return ok ? 0 : 1;
}
