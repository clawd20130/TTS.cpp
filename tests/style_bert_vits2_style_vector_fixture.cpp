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
                 "STYLE_BERT_VITS2_STYLE_VECTOR_FIXTURE max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
                 max_abs,
                 rms,
                 worst,
                 actual[worst],
                 expected[worst],
                 expected.size());
    return max_abs <= tolerance;
}

double vector_rms_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "size mismatch for rms diff: a=%zu b=%zu\n", a.size(), b.size());
        std::exit(1);
    }
    double rms = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = (double) a[i] - (double) b[i];
        rms += diff * diff;
    }
    return std::sqrt(rms / (double) a.size());
}
}

int main() {
    const std::string model_path = gguf_path();
    if (!std::filesystem::is_regular_file(model_path)) {
        std::fprintf(stderr, "Style-Bert GGUF missing; skipping style vector fixture test: %s\n", model_path.c_str());
        return 0;
    }

    const std::filesystem::path dir = fixture_dir();
    const std::filesystem::path expected_path = dir / "akai_chi_frontend.style_vector.f32";
    if (!std::filesystem::is_regular_file(expected_path)) {
        std::fprintf(stderr, "Style-Bert style vector fixture raw file missing; skipping fixture test in %s\n", dir.c_str());
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

    std::vector<float> expected = read_f32_file(expected_path);
    if (expected.size() != 256) {
        std::fprintf(stderr, "invalid style vector size %zu\n", expected.size());
        return 1;
    }

    std::vector<float> actual;
    runner->encode_style_vector(0, 1.0f, actual);
    if (!compare_vectors(actual, expected, 0.0)) {
        return 1;
    }

    std::vector<float> sad;
    runner->encode_style_vector(5, 1.0f, sad);
    const double sad_rms = vector_rms_diff(sad, actual);
    std::fprintf(stdout, "STYLE_BERT_VITS2_STYLE_VECTOR_SAD_DIFF rms=%g samples=%zu\n", sad_rms, sad.size());
    if (sad_rms < 0.01) {
        std::fprintf(stderr, "Sad style vector is too close to Neutral: rms=%g\n", sad_rms);
        return 1;
    }

    std::vector<float> sad_zero_weight;
    runner->encode_style_vector(5, 0.0f, sad_zero_weight);
    if (!compare_vectors(sad_zero_weight, actual, 1e-6)) {
        return 1;
    }
    return 0;
}
