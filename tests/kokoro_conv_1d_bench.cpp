#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/tts_model.h"

namespace {
enum class conv_impl {
    im2col,
    fused,
};

struct conv_case {
    const char * name;
    int kernel;
    int in_channels;
    int out_channels;
    int length;
    int batch;
    int stride;
    int padding;
    int dilation;
    int trace_count;
};

struct bench_result {
    std::vector<float> output;
    std::vector<double> samples_ms;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    int graph_nodes = 0;
};

struct diff_result {
    double max_abs = 0.0;
    double rms = 0.0;
    size_t worst = 0;
};

const char * impl_name(conv_impl impl) {
    return impl == conv_impl::im2col ? "im2col" : "fused";
}

std::vector<float> make_values(size_t n, float scale, int mod, int salt) {
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) {
        const int v = (int)((i * 131u + (size_t)salt * 17u) % (size_t)mod) - mod / 2;
        values[i] = scale * (float)v;
    }
    return values;
}

ggml_tensor * build_conv(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input, const conv_case & c, conv_impl impl) {
    if (impl == conv_impl::fused) {
        return ggml_kokoro_conv_1d(ctx, weight, input, c.stride, c.padding, c.dilation);
    }
    return ggml_conv_1d(ctx, weight, input, c.stride, c.padding, c.dilation);
}

bench_result run_backend(ggml_backend_t backend, const conv_case & c, conv_impl impl, int warmup, int iterations) {
    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        std::abort();
    }

    ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.kernel, c.in_channels, c.out_channels);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.length, c.in_channels, c.batch);
    ggml_tensor * output = build_conv(ctx, weight, input, c, impl);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate backend tensors for %s %s on %s\n",
                     c.name, impl_name(impl), ggml_backend_name(backend));
        std::abort();
    }

    const std::vector<float> weight_data =
        make_values((size_t)c.kernel * c.in_channels * c.out_channels, 0.00125f, 31, c.kernel);
    const std::vector<float> input_data =
        make_values((size_t)c.length * c.in_channels * c.batch, 0.015625f, 37, c.in_channels);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    auto compute_once = [&]() {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph compute failed for %s %s on %s\n",
                         c.name, impl_name(impl), ggml_backend_name(backend));
            std::abort();
        }
    };

    for (int i = 0; i < warmup; ++i) {
        compute_once();
    }

    bench_result result;
    result.graph_nodes = graph->n_nodes;
    result.samples_ms.reserve((size_t)iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        compute_once();
        const auto end = std::chrono::steady_clock::now();
        result.samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    if (!result.samples_ms.empty()) {
        std::vector<double> sorted = result.samples_ms;
        std::sort(sorted.begin(), sorted.end());
        result.min_ms = sorted.front();
        result.max_ms = sorted.back();
        result.median_ms = sorted[sorted.size() / 2];
    }

    result.output.resize((size_t)ggml_nelements(output));
    ggml_backend_tensor_get(output, result.output.data(), 0, result.output.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

diff_result compare_outputs(const std::vector<float> & actual, const std::vector<float> & expected) {
    diff_result diff;
    if (actual.size() != expected.size()) {
        diff.max_abs = INFINITY;
        diff.rms = INFINITY;
        return diff;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double d = (double)actual[i] - (double)expected[i];
        const double a = std::fabs(d);
        if (a > diff.max_abs) {
            diff.max_abs = a;
            diff.worst = i;
        }
        sum_sq += d * d;
    }
    diff.rms = std::sqrt(sum_sq / (double)expected.size());
    return diff;
}

void print_result(const conv_case & c,
                  const char * backend_name,
                  conv_impl impl,
                  const bench_result & result,
                  const diff_result & diff,
                  int iterations) {
    std::printf(
        "KOKORO_CONV1D_BENCH case=%s backend=%s impl=%s nodes=%d output=%zu "
        "iters=%d median_ms=%.3f min_ms=%.3f max_ms=%.3f trace_count=%d trace_est_ms=%.3f "
        "max_abs=%.8g rms=%.8g worst=%zu\n",
        c.name,
        backend_name,
        impl_name(impl),
        result.graph_nodes,
        result.output.size(),
        iterations,
        result.median_ms,
        result.min_ms,
        result.max_ms,
        c.trace_count,
        result.median_ms * (double)c.trace_count,
        diff.max_abs,
        diff.rms,
        diff.worst);
}

int parse_int_arg(int argc, char ** argv, const char * name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return fallback;
}

bool has_arg(int argc, char ** argv, const char * name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return true;
        }
    }
    return false;
}
}

int main(int argc, char ** argv) {
    const int warmup = std::max(0, parse_int_arg(argc, argv, "--warmup", 2));
    const int iterations = std::max(1, parse_int_arg(argc, argv, "--iters", 5));
    const int threads = std::max(1, parse_int_arg(argc, argv, "--threads", 8));
    const bool cpu_only = has_arg(argc, argv, "--cpu-only");

    const conv_case cases[] = {
        {"noise0-k12-s6", 12, 22, 256, 6841, 1, 6, 3, 1, 1},
        {"res0-k3-d1", 3, 256, 256, 1140, 1, 1, 1, 1, 6},
        {"res0-k7-d3", 7, 256, 256, 1140, 1, 1, 9, 3, 12},
        {"res0-k11-d5", 11, 256, 256, 1140, 1, 1, 25, 5, 6},
        {"noise1-k1", 1, 22, 128, 6841, 1, 1, 0, 1, 1},
        {"res1-k3-d1", 3, 128, 128, 6841, 1, 1, 1, 1, 6},
        {"res1-k7-d3", 7, 128, 128, 6841, 1, 1, 9, 3, 6},
        {"res1-k11-d5", 11, 128, 128, 6841, 1, 1, 25, 5, 12},
        {"post-k7", 7, 128, 22, 6841, 1, 1, 3, 1, 1},
    };

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);

    ggml_backend_t vulkan = nullptr;
    if (!cpu_only) {
        vulkan = tts_backend_init_accelerator(false);
        if (!vulkan) {
            std::fprintf(stderr, "Vulkan backend unavailable; running CPU-only Kokoro Conv1D bench\n");
        }
    }

    bool ok = true;
    for (const conv_case & c : cases) {
        const bench_result cpu_im2col = run_backend(cpu, c, conv_impl::im2col, warmup, iterations);
        print_result(c, "CPU", conv_impl::im2col, cpu_im2col, {0.0, 0.0, 0}, iterations);

        const bench_result cpu_fused = run_backend(cpu, c, conv_impl::fused, warmup, iterations);
        const diff_result cpu_fused_diff = compare_outputs(cpu_fused.output, cpu_im2col.output);
        print_result(c, "CPU", conv_impl::fused, cpu_fused, cpu_fused_diff, iterations);
        ok = ok && cpu_fused_diff.max_abs <= 1e-4;

        if (vulkan) {
            const bench_result vk_im2col = run_backend(vulkan, c, conv_impl::im2col, warmup, iterations);
            const diff_result vk_im2col_diff = compare_outputs(vk_im2col.output, cpu_im2col.output);
            print_result(c, "Vulkan", conv_impl::im2col, vk_im2col, vk_im2col_diff, iterations);
            ok = ok && vk_im2col_diff.max_abs <= 1e-3;

            const bench_result vk_fused = run_backend(vulkan, c, conv_impl::fused, warmup, iterations);
            const diff_result vk_fused_diff = compare_outputs(vk_fused.output, cpu_im2col.output);
            print_result(c, "Vulkan", conv_impl::fused, vk_fused, vk_fused_diff, iterations);
            ok = ok && vk_fused_diff.max_abs <= 1e-3;
        }
    }

    ggml_backend_free(cpu);
    if (vulkan) {
        ggml_backend_free(vulkan);
    }

    return ok ? 0 : 1;
}
