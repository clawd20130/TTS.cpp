#include <cmath>
#include <cstdint>
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

std::vector<int32_t> read_i64_as_i32_file(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }
    const std::streamsize size = file.tellg();
    if (size < 0 || size % (std::streamsize) sizeof(int64_t) != 0) {
        std::fprintf(stderr, "invalid i64 file size for %s: %lld\n", path.c_str(), (long long) size);
        std::exit(1);
    }
    file.seekg(0, std::ios::beg);
    std::vector<int64_t> raw((size_t) size / sizeof(int64_t));
    if (!file.read(reinterpret_cast<char *>(raw.data()), size)) {
        std::fprintf(stderr, "failed to read %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<int32_t> values(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        values[i] = (int32_t) raw[i];
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

bool compare_vectors(const char * label, const std::vector<float> & actual, const std::vector<float> & expected, double tolerance) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s size mismatch: actual=%zu expected=%zu\n", label, actual.size(), expected.size());
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
                 "STYLE_BERT_VITS2_TEXT_ENCODER_FIXTURE %s max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
                 label,
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
        std::fprintf(stderr, "Style-Bert GGUF missing; skipping TextEncoder fixture test: %s\n", model_path.c_str());
        return 0;
    }

    const std::filesystem::path dir = fixture_dir();
    const std::filesystem::path phone_path = dir / "akai_chi_frontend.phone_ids.i64";
    const std::filesystem::path tone_path = dir / "akai_chi_frontend.tone_ids.i64";
    const std::filesystem::path language_path = dir / "akai_chi_frontend.language_ids.i64";
    const std::filesystem::path bert_path = dir / "akai_chi_frontend.ja_bert.f32";
    const std::filesystem::path style_path = dir / "akai_chi_frontend.style_vector.f32";
    const std::filesystem::path x_mask_path = dir / "akai_chi_frontend.x_mask.f32";
    const std::filesystem::path g_path = dir / "akai_chi_frontend.g.f32";
    const std::filesystem::path expected_x_path = dir / "akai_chi_frontend.x.f32";
    const std::filesystem::path expected_m_path = dir / "akai_chi_frontend.m_p.f32";
    const std::filesystem::path expected_logs_path = dir / "akai_chi_frontend.logs_p.f32";
    if (!std::filesystem::is_regular_file(phone_path) ||
        !std::filesystem::is_regular_file(tone_path) ||
        !std::filesystem::is_regular_file(language_path) ||
        !std::filesystem::is_regular_file(bert_path) ||
        !std::filesystem::is_regular_file(style_path) ||
        !std::filesystem::is_regular_file(x_mask_path) ||
        !std::filesystem::is_regular_file(g_path) ||
        !std::filesystem::is_regular_file(expected_x_path) ||
        !std::filesystem::is_regular_file(expected_m_path) ||
        !std::filesystem::is_regular_file(expected_logs_path)) {
        std::fprintf(stderr, "Style-Bert TextEncoder fixture raw files missing; skipping fixture test in %s\n", dir.c_str());
        return 0;
    }

    std::vector<int32_t> phone_ids = read_i64_as_i32_file(phone_path);
    std::vector<int32_t> tone_ids = read_i64_as_i32_file(tone_path);
    std::vector<int32_t> language_ids = read_i64_as_i32_file(language_path);
    std::vector<float> bert = read_f32_file(bert_path);
    std::vector<float> style_vec = read_f32_file(style_path);
    std::vector<float> x_mask = read_f32_file(x_mask_path);
    std::vector<float> g = read_f32_file(g_path);
    std::vector<float> expected_x = read_f32_file(expected_x_path);
    std::vector<float> expected_m = read_f32_file(expected_m_path);
    std::vector<float> expected_logs = read_f32_file(expected_logs_path);
    const uint32_t tokens = (uint32_t) phone_ids.size();

    if (tone_ids.size() != tokens || language_ids.size() != tokens || x_mask.size() != tokens) {
        std::fprintf(stderr, "length mismatch: phone=%zu tone=%zu language=%zu mask=%zu\n",
                     phone_ids.size(), tone_ids.size(), language_ids.size(), x_mask.size());
        return 1;
    }
    if (bert.size() != (size_t) tokens * 1024) {
        std::fprintf(stderr, "invalid BERT fixture size: %zu for tokens=%u\n", bert.size(), tokens);
        return 1;
    }
    if (style_vec.size() != 256) {
        std::fprintf(stderr, "invalid style vector size: %zu\n", style_vec.size());
        return 1;
    }

    generation_configuration config;
    const bool cpu_only = !(std::getenv("STYLE_BERT_VITS2_TEST_ACCELERATOR") && std::getenv("STYLE_BERT_VITS2_TEST_ACCELERATOR")[0]);
    std::unique_ptr<tts_generation_runner> base = runner_from_file(model_path.c_str(), 4, config, cpu_only);
    auto * runner = dynamic_cast<style_bert_vits2_runner *>(base.get());
    if (!runner) {
        std::fprintf(stderr, "loaded runner is not style_bert_vits2_runner\n");
        return 1;
    }

    if (g.size() != runner->model->gin_channels) {
        std::fprintf(stderr, "invalid g size %zu for gin_channels=%u\n", g.size(), runner->model->gin_channels);
        return 1;
    }
    if (expected_x.size() != (size_t) runner->model->hidden_channels * tokens ||
        expected_m.size() != (size_t) runner->model->inter_channels * tokens ||
        expected_logs.size() != expected_m.size()) {
        std::fprintf(stderr, "invalid expected sizes: x=%zu m=%zu logs=%zu tokens=%u\n",
                     expected_x.size(), expected_m.size(), expected_logs.size(), tokens);
        return 1;
    }

    std::vector<float> actual_x;
    std::vector<float> actual_m;
    std::vector<float> actual_logs;
    runner->run_text_encoder(phone_ids.data(),
                             tone_ids.data(),
                             language_ids.data(),
                             bert.data(),
                             style_vec.data(),
                             x_mask.data(),
                             g.data(),
                             tokens,
                             actual_x,
                             actual_m,
                             actual_logs);
    const bool x_ok = compare_vectors("x", actual_x, expected_x, 1e-4);
    const bool m_ok = compare_vectors("m_p", actual_m, expected_m, 1e-4);
    const bool logs_ok = compare_vectors("logs_p", actual_logs, expected_logs, 1e-4);
    if (!x_ok || !m_ok || !logs_ok) {
        return 1;
    }
    return 0;
}
