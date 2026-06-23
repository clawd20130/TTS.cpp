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
    const char * env = std::getenv("STYLE_BERT_VITS2_FRONT_FIXTURE_DIR");
    if (env && env[0]) {
        return env;
    }
    return "/home/kevinzhow/github/kokoro-tts/tmp/style-bert-vits2-front-fixtures";
}

std::string gguf_path() {
    const char * env = std::getenv("STYLE_BERT_VITS2_GGUF");
    if (env && env[0]) {
        return env;
    }
    return "tmp/style-bert-vits2/jvnv-F1-jp-full.gguf";
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
                 "STYLE_BERT_VITS2_FLOW_FIXTURE max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
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
        std::fprintf(stderr, "Style-Bert GGUF missing; skipping flow fixture test: %s\n", model_path.c_str());
        return 0;
    }

    const std::filesystem::path dir = fixture_dir();
    const std::filesystem::path z_p_path = dir / "akai_chi_frontend.z_p.f32";
    const std::filesystem::path y_mask_path = dir / "akai_chi_frontend.y_mask.f32";
    const std::filesystem::path g_path = dir / "akai_chi_frontend.g.f32";
    const std::filesystem::path expected_path = dir / "akai_chi_frontend.z.f32";
    if (!std::filesystem::is_regular_file(z_p_path) ||
        !std::filesystem::is_regular_file(y_mask_path) ||
        !std::filesystem::is_regular_file(g_path) ||
        !std::filesystem::is_regular_file(expected_path)) {
        std::fprintf(stderr, "Style-Bert flow fixture raw files missing; skipping fixture test in %s\n", dir.c_str());
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

    std::vector<float> z_p = read_f32_file(z_p_path);
    std::vector<float> y_mask = read_f32_file(y_mask_path);
    std::vector<float> g = read_f32_file(g_path);
    std::vector<float> expected = read_f32_file(expected_path);
    const uint32_t frames = (uint32_t) y_mask.size();
    if (z_p.size() != expected.size()) {
        std::fprintf(stderr, "input/expected size mismatch: z_p=%zu expected=%zu\n", z_p.size(), expected.size());
        return 1;
    }
    if (z_p.size() != (size_t) runner->model->inter_channels * frames) {
        std::fprintf(stderr, "invalid z_p size %zu for frames=%u inter_channels=%u\n",
                     z_p.size(),
                     frames,
                     runner->model->inter_channels);
        return 1;
    }
    if (g.size() != runner->model->gin_channels) {
        std::fprintf(stderr, "invalid g size %zu for gin_channels=%u\n", g.size(), runner->model->gin_channels);
        return 1;
    }

    std::vector<float> actual;
    runner->run_flow_reverse(z_p.data(), y_mask.data(), g.data(), frames, actual);
    if (!compare_vectors(actual, expected, 1e-4)) {
        return 1;
    }
    return 0;
}
