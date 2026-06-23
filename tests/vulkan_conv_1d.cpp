#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../src/models/style_bert_vits2/model.h"
#include "../src/tts_model.h"

namespace {
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
};

std::vector<float> make_values(size_t n, float scale, int mod, int salt) {
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) {
        const int v = (int)((i * 131u + (size_t)salt * 17u) % (size_t)mod) - mod / 2;
        values[i] = scale * (float)v;
    }
    return values;
}

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }
    const std::streamsize size = file.tellg();
    if (size < 0 || size % (std::streamsize) sizeof(float) != 0) {
        std::fprintf(stderr, "invalid f32 file size for %s: %lld\n", path.c_str(), (long long) size);
        std::exit(1);
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values((size_t) size / sizeof(float));
    if (!file.read(reinterpret_cast<char *>(values.data()), size)) {
        std::fprintf(stderr, "failed to read %s\n", path.c_str());
        std::exit(1);
    }
    return values;
}

void restore_env(const char * name, const std::string & value) {
    if (value.empty()) {
        unsetenv(name);
    } else {
        setenv(name, value.c_str(), 1);
    }
}

std::vector<float> run_case(ggml_backend_t backend, const conv_case & c) {
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.kernel, c.in_channels, c.out_channels);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.length, c.in_channels, c.batch);
    ggml_tensor * output = ggml_conv_1d(ctx, weight, input, c.stride, c.padding, c.dilation);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    const std::vector<float> weight_data = make_values((size_t) ggml_nelements(weight), 0.03125f, 31, c.kernel);
    const std::vector<float> input_data = make_values((size_t) ggml_nelements(input), 0.125f, 29, c.in_channels);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "conv1d graph compute failed for %s on %s\n", c.name, ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

bool compare_case(ggml_backend_t vulkan, const conv_case & c, double tolerance) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    const std::vector<float> expected = run_case(cpu, c);
    ggml_backend_free(cpu);

    const std::vector<float> actual = run_case(vulkan, c);
    double max_abs = 0.0;
    double rms = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double diff = (double) actual[i] - (double) expected[i];
        const double abs_diff = std::fabs(diff);
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
            worst = i;
        }
        rms += diff * diff;
    }
    rms = std::sqrt(rms / (double) expected.size());
    std::fprintf(stdout,
                 "VULKAN_CONV_1D node=%s max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
                 c.name,
                 max_abs,
                 rms,
                 worst,
                 actual[worst],
                 expected[worst],
                 expected.size());
    return max_abs <= tolerance;
}

std::vector<float> run_real_conv_pre(ggml_backend_t backend,
                                     const std::vector<float> & weight_data,
                                     const std::vector<float> & bias_data,
                                     const std::vector<float> & z_data,
                                     int frames,
                                     int channels,
                                     bool add_bias) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, channels, 512);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, channels, 1);
    ggml_tensor * output = ggml_conv_1d(ctx, weight, input, 1, 3, 1);
    ggml_tensor * bias = nullptr;
    if (add_bias) {
        bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 512);
        output = ggml_add(ctx, output, bias);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_type_t buft = tts_backend_get_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, z_data.data(), 0, z_data.size() * sizeof(float));
    if (bias) {
        ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "real Style-Bert conv_pre graph compute failed on %s\n", ggml_backend_name(backend));
        std::abort();
    }

    std::vector<float> result((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

bool compare_real_style_bert_conv_pre(ggml_backend_t vulkan, double tolerance) {
    const char * gguf_env = std::getenv("STYLE_BERT_VITS2_GGUF");
    const char * fixture_env = std::getenv("STYLE_BERT_VITS2_FIXTURE_DIR");
    const std::filesystem::path gguf = gguf_env && gguf_env[0] ? gguf_env : "tmp/style-bert-vits2/jvnv-F1-jp-decoder.gguf";
    const std::filesystem::path fixture_dir = fixture_env && fixture_env[0]
        ? fixture_env
        : "/home/kevinzhow/github/kokoro-tts/tmp/style-bert-vits2-fixtures";
    const std::filesystem::path z_path = fixture_dir / "akai_chi_decoder.decoder_z.f32";
    if (!std::filesystem::is_regular_file(gguf) || !std::filesystem::is_regular_file(z_path)) {
        std::fprintf(stdout, "VULKAN_CONV_1D_REAL_STYLE_BERT skipped gguf=%s z=%s\n", gguf.c_str(), z_path.c_str());
        return true;
    }

    const char * original_backend = std::getenv("TTS_BACKEND");
    const char * original_strict = std::getenv("TTS_BACKEND_STRICT");
    const std::string saved_backend = original_backend ? original_backend : "";
    const std::string saved_strict = original_strict ? original_strict : "";
    setenv("TTS_BACKEND", "cpu", 1);
    unsetenv("TTS_BACKEND_STRICT");

    generation_configuration config;
    std::unique_ptr<tts_generation_runner> base = runner_from_file(gguf.c_str(), 4, config, true);

    restore_env("TTS_BACKEND", saved_backend);
    restore_env("TTS_BACKEND_STRICT", saved_strict);

    auto * runner = dynamic_cast<style_bert_vits2_runner *>(base.get());
    if (!runner) {
        std::fprintf(stderr, "loaded runner is not style_bert_vits2_runner\n");
        return false;
    }

    ggml_tensor * weight = runner->model->decoder.conv_pre.weight;
    ggml_tensor * bias = runner->model->decoder.conv_pre.bias;
    std::vector<float> weight_data((size_t) ggml_nelements(weight));
    std::vector<float> bias_data((size_t) ggml_nelements(bias));
    ggml_backend_tensor_get(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_get(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));

    const std::vector<float> z_nct = read_f32_file(z_path);
    const int channels = (int) runner->model->inter_channels;
    const int frames = (int) (z_nct.size() / (size_t) channels);
    std::vector<float> z_data((size_t) frames * (size_t) channels);
    for (int c = 0; c < channels; ++c) {
        for (int t = 0; t < frames; ++t) {
            z_data[(size_t) t + (size_t) frames * (size_t) c] = z_nct[(size_t) c * (size_t) frames + (size_t) t];
        }
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    bool ok = true;
    for (bool add_bias : {false, true}) {
        const std::vector<float> expected = run_real_conv_pre(cpu, weight_data, bias_data, z_data, frames, channels, add_bias);
        const std::vector<float> actual = run_real_conv_pre(vulkan, weight_data, bias_data, z_data, frames, channels, add_bias);
        double max_abs = 0.0;
        double rms = 0.0;
        size_t worst = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            const double diff = (double) actual[i] - (double) expected[i];
            const double abs_diff = std::fabs(diff);
            if (abs_diff > max_abs) {
                max_abs = abs_diff;
                worst = i;
            }
            rms += diff * diff;
        }
        rms = std::sqrt(rms / (double) expected.size());
        std::fprintf(stdout,
                     "VULKAN_CONV_1D_REAL_STYLE_BERT node=conv_pre%s max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
                     add_bias ? "_bias" : "",
                     max_abs,
                     rms,
                     worst,
                     actual[worst],
                     expected[worst],
                     expected.size());
        ok = (max_abs <= tolerance) && ok;
    }
    ggml_backend_free(cpu);
    return ok;
}
}

int main() {
    setenv("GGML_VK_DISABLE_F16", "1", 0);
    setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
    setenv("GGML_VK_DISABLE_COOPMAT2", "1", 0);

    const conv_case cases[] = {
        {"conv_1d_small", 3, 4, 3, 13, 2, 1, 2, 2},
        {"conv_1d_style_bert_conv_pre", 7, 192, 512, 60, 1, 1, 3, 1},
    };

    ggml_backend_t vulkan = tts_backend_init_accelerator(false);
    if (!vulkan) {
        std::fprintf(stderr, "Vulkan backend unavailable; skipping conv1d parity test\n");
        return 0;
    }

    bool ok = true;
    for (const conv_case & c : cases) {
        ok = compare_case(vulkan, c, 1e-4) && ok;
    }
    ok = compare_real_style_bert_conv_pre(vulkan, 1e-4) && ok;

    ggml_backend_free(vulkan);
    return ok ? 0 : 1;
}
