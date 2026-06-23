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
                 "STYLE_BERT_VITS2_TEXT_ENCODER_INPUT_FIXTURE max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
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
        std::fprintf(stderr, "Style-Bert GGUF missing; skipping text encoder input fixture test: %s\n", model_path.c_str());
        return 0;
    }

    const std::filesystem::path dir = fixture_dir();
    const std::filesystem::path phone_path = dir / "akai_chi_frontend.phone_ids.i64";
    const std::filesystem::path tone_path = dir / "akai_chi_frontend.tone_ids.i64";
    const std::filesystem::path language_path = dir / "akai_chi_frontend.language_ids.i64";
    const std::filesystem::path bert_path = dir / "akai_chi_frontend.ja_bert.f32";
    const std::filesystem::path style_path = dir / "akai_chi_frontend.style_vector.f32";
    const std::filesystem::path expected_path = dir / "akai_chi_frontend.enc_input.f32";
    if (!std::filesystem::is_regular_file(phone_path) ||
        !std::filesystem::is_regular_file(tone_path) ||
        !std::filesystem::is_regular_file(language_path) ||
        !std::filesystem::is_regular_file(bert_path) ||
        !std::filesystem::is_regular_file(style_path) ||
        !std::filesystem::is_regular_file(expected_path)) {
        std::fprintf(stderr, "Style-Bert front fixture raw files missing; skipping fixture test in %s\n", dir.c_str());
        return 0;
    }

    std::vector<int32_t> phone_ids = read_i64_as_i32_file(phone_path);
    std::vector<int32_t> tone_ids = read_i64_as_i32_file(tone_path);
    std::vector<int32_t> language_ids = read_i64_as_i32_file(language_path);
    std::vector<float> bert = read_f32_file(bert_path);
    std::vector<float> style_vec = read_f32_file(style_path);
    std::vector<float> expected = read_f32_file(expected_path);
    const uint32_t tokens = (uint32_t) phone_ids.size();

    if (tone_ids.size() != tokens || language_ids.size() != tokens) {
        std::fprintf(stderr, "id length mismatch: phone=%zu tone=%zu language=%zu\n",
                     phone_ids.size(), tone_ids.size(), language_ids.size());
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

    std::vector<float> actual;
    runner->encode_text_encoder_input(phone_ids.data(),
                                      tone_ids.data(),
                                      language_ids.data(),
                                      bert.data(),
                                      style_vec.data(),
                                      tokens,
                                      actual);
    if (!compare_vectors(actual, expected, 1e-4)) {
        return 1;
    }
    return 0;
}
