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
                 "STYLE_BERT_VITS2_ALIGNMENT_FIXTURE %s max_abs=%g rms=%g worst=%zu actual=%f expected=%f samples=%zu\n",
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
        std::fprintf(stderr, "Style-Bert GGUF missing; skipping alignment fixture test: %s\n", model_path.c_str());
        return 0;
    }

    const std::filesystem::path dir = fixture_dir();
    const std::filesystem::path logw_path = dir / "akai_chi_frontend.logw.f32";
    const std::filesystem::path w_path = dir / "akai_chi_frontend.w.f32";
    const std::filesystem::path w_ceil_path = dir / "akai_chi_frontend.w_ceil.f32";
    const std::filesystem::path x_mask_path = dir / "akai_chi_frontend.x_mask.f32";
    const std::filesystem::path y_mask_path = dir / "akai_chi_frontend.y_mask.f32";
    const std::filesystem::path attn_path = dir / "akai_chi_frontend.attn.f32";
    const std::filesystem::path m_path = dir / "akai_chi_frontend.m_p.f32";
    const std::filesystem::path logs_path = dir / "akai_chi_frontend.logs_p.f32";
    const std::filesystem::path m_expanded_path = dir / "akai_chi_frontend.m_p_expanded.f32";
    const std::filesystem::path logs_expanded_path = dir / "akai_chi_frontend.logs_p_expanded.f32";
    if (!std::filesystem::is_regular_file(logw_path) ||
        !std::filesystem::is_regular_file(w_path) ||
        !std::filesystem::is_regular_file(w_ceil_path) ||
        !std::filesystem::is_regular_file(x_mask_path) ||
        !std::filesystem::is_regular_file(y_mask_path) ||
        !std::filesystem::is_regular_file(attn_path) ||
        !std::filesystem::is_regular_file(m_path) ||
        !std::filesystem::is_regular_file(logs_path) ||
        !std::filesystem::is_regular_file(m_expanded_path) ||
        !std::filesystem::is_regular_file(logs_expanded_path)) {
        std::fprintf(stderr, "Style-Bert alignment fixture raw files missing; skipping fixture test in %s\n", dir.c_str());
        return 0;
    }

    generation_configuration config;
    std::unique_ptr<tts_generation_runner> base = runner_from_file(model_path.c_str(), 4, config, true);
    auto * runner = dynamic_cast<style_bert_vits2_runner *>(base.get());
    if (!runner) {
        std::fprintf(stderr, "loaded runner is not style_bert_vits2_runner\n");
        return 1;
    }

    std::vector<float> logw = read_f32_file(logw_path);
    std::vector<float> expected_w = read_f32_file(w_path);
    std::vector<float> expected_w_ceil = read_f32_file(w_ceil_path);
    std::vector<float> x_mask = read_f32_file(x_mask_path);
    std::vector<float> expected_y_mask = read_f32_file(y_mask_path);
    std::vector<float> expected_attn = read_f32_file(attn_path);
    std::vector<float> m_p = read_f32_file(m_path);
    std::vector<float> logs_p = read_f32_file(logs_path);
    std::vector<float> expected_m_expanded = read_f32_file(m_expanded_path);
    std::vector<float> expected_logs_expanded = read_f32_file(logs_expanded_path);
    const uint32_t tokens = (uint32_t) logw.size();
    if (x_mask.size() != tokens || expected_w.size() != tokens || expected_w_ceil.size() != tokens) {
        std::fprintf(stderr, "invalid duration fixture sizes: logw=%zu x_mask=%zu w=%zu w_ceil=%zu\n",
                     logw.size(),
                     x_mask.size(),
                     expected_w.size(),
                     expected_w_ceil.size());
        return 1;
    }
    if ((size_t) runner->model->inter_channels * tokens != m_p.size() || logs_p.size() != m_p.size()) {
        std::fprintf(stderr, "invalid source stats sizes: m=%zu logs=%zu tokens=%u inter_channels=%u\n",
                     m_p.size(),
                     logs_p.size(),
                     tokens,
                     runner->model->inter_channels);
        return 1;
    }

    style_bert_vits2_alignment_result actual =
        runner->expand_alignment(logw.data(), x_mask.data(), m_p.data(), logs_p.data(), tokens, 1.0f);
    if (actual.y_mask.size() != expected_y_mask.size()) {
        std::fprintf(stderr, "frame count mismatch: actual=%zu expected=%zu\n", actual.y_mask.size(), expected_y_mask.size());
        return 1;
    }

    bool ok = true;
    ok = compare_vectors("w", actual.w, expected_w, 1e-5) && ok;
    ok = compare_vectors("w_ceil", actual.w_ceil, expected_w_ceil, 0.0) && ok;
    ok = compare_vectors("y_mask", actual.y_mask, expected_y_mask, 0.0) && ok;
    ok = compare_vectors("attn", actual.attn, expected_attn, 0.0) && ok;
    ok = compare_vectors("m_p_expanded", actual.m_p_expanded, expected_m_expanded, 0.0) && ok;
    ok = compare_vectors("logs_p_expanded", actual.logs_p_expanded, expected_logs_expanded, 0.0) && ok;
    return ok ? 0 : 1;
}
