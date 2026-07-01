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

bool compare_vectors(const std::vector<float> & actual, const std::vector<float> & expected, double tolerance) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
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
                 "STYLE_BERT_VITS2_DECODER_FIXTURE max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
                 max_abs,
                 rms,
                 worst,
                 actual[worst],
                 expected[worst],
                 expected.size());
    return max_abs <= tolerance;
}
}

int main() {
    const std::string model_path = gguf_path();
    if (!std::filesystem::is_regular_file(model_path)) {
        std::fprintf(stderr, "Style-Bert GGUF missing; skipping fixture test: %s\n", model_path.c_str());
        return 0;
    }

    const std::filesystem::path dir = fixture_dir();
    const std::filesystem::path z_path = dir / "akai_chi_decoder.decoder_z.f32";
    const std::filesystem::path g_path = dir / "akai_chi_decoder.decoder_g.f32";
    const std::filesystem::path expected_path = dir / "akai_chi_decoder.decoder_output.f32";
    if (!std::filesystem::is_regular_file(z_path) ||
        !std::filesystem::is_regular_file(g_path) ||
        !std::filesystem::is_regular_file(expected_path)) {
        std::fprintf(stderr, "Style-Bert fixture raw files missing; skipping fixture test in %s\n", dir.c_str());
        return 0;
    }

    generation_configuration config;
    const bool cpu_only = !(std::getenv("STYLE_BERT_VITS2_TEST_ACCELERATOR") && std::getenv("STYLE_BERT_VITS2_TEST_ACCELERATOR")[0]);
    std::unique_ptr<tts_generation_runner> base = runner_from_file(model_path.c_str(), 4, config, cpu_only);
    auto * runner = dynamic_cast<style_bert_vits2_runner *>(base.get());
    if (!runner) {
        std::fprintf(stderr, "loaded runner is not style_bert_vits2_runner\n");
        return 1;
    }

    std::vector<float> z = read_f32_file(z_path);
    std::vector<float> g = read_f32_file(g_path);
    std::vector<float> expected = read_f32_file(expected_path);
    const uint32_t frames = (uint32_t) (z.size() / runner->model->inter_channels);
    if ((size_t) frames * runner->model->inter_channels != z.size()) {
        std::fprintf(stderr, "invalid z size %zu for inter_channels=%u\n", z.size(), runner->model->inter_channels);
        return 1;
    }
    if (g.size() != runner->model->gin_channels) {
        std::fprintf(stderr, "invalid g size %zu for gin_channels=%u\n", g.size(), runner->model->gin_channels);
        return 1;
    }

    tts_response response{};
    std::string decode_error;
    if (!runner->decode(z.data(), g.data(), frames, response, &decode_error)) {
        std::fprintf(stderr, "decoder fixture failed: %s\n", decode_error.c_str());
        return 1;
    }
    std::vector<float> actual(response.data, response.data + response.n_outputs);
    if (!compare_vectors(actual, expected, 1e-4)) {
        return 1;
    }
    return 0;
}
