#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../src/models/style_bert_vits2/model.h"

namespace {
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

std::filesystem::path fixture_dir() {
    const char * env = std::getenv("STYLE_BERT_VITS2_FIXTURE_DIR");
    if (env && env[0]) {
        return env;
    }
    return "/home/kevinzhow/github/kokoro-tts/tmp/style-bert-vits2-fixtures";
}

std::string gguf_path() {
    const char * env = std::getenv("STYLE_BERT_VITS2_GGUF");
    if (env && env[0]) {
        return env;
    }
    return "tmp/style-bert-vits2/jvnv-F1-jp-decoder.gguf";
}

std::vector<float> run_node(style_bert_vits2_runner * runner,
                            const std::vector<float> & z,
                            const std::vector<float> & g,
                            uint32_t frames,
                            const char * node_name) {
    setenv("STYLE_BERT_VITS2_DEBUG_OUTPUT_NODE", node_name, 1);
    tts_response response{};
    std::string error;
    if (!runner->decode(z.data(), g.data(), frames, response, &error)) {
        std::fprintf(stderr, "decoder compare failed for %s: %s\n", node_name, error.c_str());
        return {};
    }
    return std::vector<float>(response.data, response.data + response.n_outputs);
}

void restore_env(const char * name, const char * value) {
    if (value && value[0]) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
}

void compare_node(const char * node_name, const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        std::fprintf(stdout,
                     "STYLE_BERT_VITS2_BACKEND_COMPARE node=%s size_mismatch actual=%zu expected=%zu\n",
                     node_name,
                     actual.size(),
                     expected.size());
        return;
    }
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
                 "STYLE_BERT_VITS2_BACKEND_COMPARE node=%s max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
                 node_name,
                 max_abs,
                 rms,
                 worst,
                 actual[worst],
                 expected[worst],
                 expected.size());
}
}

int main() {
    const std::string model_path = gguf_path();
    if (!std::filesystem::is_regular_file(model_path)) {
        std::fprintf(stderr, "Style-Bert GGUF missing; skipping backend compare: %s\n", model_path.c_str());
        return 0;
    }

    const std::filesystem::path dir = fixture_dir();
    const std::vector<float> z = read_f32_file(dir / "akai_chi_decoder.decoder_z.f32");
    const std::vector<float> g = read_f32_file(dir / "akai_chi_decoder.decoder_g.f32");

    generation_configuration config;
    const char * original_backend = std::getenv("TTS_BACKEND");
    const char * original_strict = std::getenv("TTS_BACKEND_STRICT");
    const char * original_max_frames = std::getenv("STYLE_BERT_VITS2_MAX_DECODER_FRAMES");
    std::string saved_backend = original_backend ? original_backend : "";
    std::string saved_strict = original_strict ? original_strict : "";
    std::string saved_max_frames = original_max_frames ? original_max_frames : "";

    setenv("TTS_BACKEND", "cpu", 1);
    unsetenv("TTS_BACKEND_STRICT");
    std::unique_ptr<tts_generation_runner> cpu_base = runner_from_file(model_path.c_str(), 4, config, true);
    auto * cpu_runner = dynamic_cast<style_bert_vits2_runner *>(cpu_base.get());
    if (!cpu_runner) {
        std::fprintf(stderr, "CPU runner is not style_bert_vits2_runner\n");
        return 1;
    }
    const uint32_t frames = (uint32_t) (z.size() / cpu_runner->model->inter_channels);
    if (frames > 1) {
        tts_response limited_response{};
        std::string limit_error;
        setenv("STYLE_BERT_VITS2_MAX_DECODER_FRAMES", "1", 1);
        if (cpu_runner->decode(z.data(), g.data(), frames, limited_response, &limit_error)) {
            std::fprintf(stderr, "expected decoder frame limit failure for %u frames\n", frames);
            return 1;
        }
        if (limit_error.find("STYLE_BERT_VITS2_MAX_DECODER_FRAMES") == std::string::npos) {
            std::fprintf(stderr, "unexpected decoder frame limit error: %s\n", limit_error.c_str());
            return 1;
        }
        restore_env("STYLE_BERT_VITS2_MAX_DECODER_FRAMES", saved_max_frames.c_str());
    }

    restore_env("TTS_BACKEND", saved_backend.c_str());
    restore_env("TTS_BACKEND_STRICT", saved_strict.c_str());
    std::unique_ptr<tts_generation_runner> accel_base = runner_from_file(model_path.c_str(), 4, config, false);
    auto * accel_runner = dynamic_cast<style_bert_vits2_runner *>(accel_base.get());
    if (!accel_runner) {
        std::fprintf(stderr, "accelerator runner is not style_bert_vits2_runner\n");
        return 1;
    }
    if (!accel_runner->sctx->backend) {
        std::fprintf(stderr, "accelerator backend unavailable; skipping backend compare\n");
        return 0;
    }

    const std::vector<const char *> nodes = {
        "style_bert_vits2.conv_pre",
        "style_bert_vits2.cond",
        "style_bert_vits2.pre_upsample",
        "style_bert_vits2.up.0.pre_relu",
        "style_bert_vits2.up.0.conv_transpose",
        "style_bert_vits2.up.0.res.0",
        "style_bert_vits2.up.0.res.1",
        "style_bert_vits2.up.0.res.2",
        "style_bert_vits2.up.0.average",
        "style_bert_vits2.up.1.pre_relu",
        "style_bert_vits2.up.1.conv_transpose",
        "style_bert_vits2.up.1.res.0",
        "style_bert_vits2.up.1.res.1",
        "style_bert_vits2.up.1.res.2",
        "style_bert_vits2.up.1.average",
        "style_bert_vits2.up.2.pre_relu",
        "style_bert_vits2.up.2.conv_transpose",
        "style_bert_vits2.up.2.res.0",
        "style_bert_vits2.up.2.res.1",
        "style_bert_vits2.up.2.res.2",
        "style_bert_vits2.up.2.average",
        "style_bert_vits2.up.3.pre_relu",
        "style_bert_vits2.up.3.conv_transpose",
        "style_bert_vits2.up.3.res.0",
        "style_bert_vits2.up.3.res.1",
        "style_bert_vits2.up.3.res.2",
        "style_bert_vits2.up.3.average",
        "style_bert_vits2.up.4.pre_relu",
        "style_bert_vits2.up.4.conv_transpose",
        "style_bert_vits2.up.4.res.0",
        "style_bert_vits2.up.4.res.1",
        "style_bert_vits2.up.4.res.2",
        "style_bert_vits2.up.4.average",
        "style_bert_vits2.final_relu",
        "style_bert_vits2.conv_post",
        "style_bert_vits2_decoder_output",
    };

    for (const char * node : nodes) {
        setenv("TTS_BACKEND", "cpu", 1);
        unsetenv("TTS_BACKEND_STRICT");
        std::vector<float> expected = run_node(cpu_runner, z, g, frames, node);
        restore_env("TTS_BACKEND", saved_backend.c_str());
        restore_env("TTS_BACKEND_STRICT", saved_strict.c_str());
        std::vector<float> actual = run_node(accel_runner, z, g, frames, node);
        compare_node(node, actual, expected);
    }
    return 0;
}
