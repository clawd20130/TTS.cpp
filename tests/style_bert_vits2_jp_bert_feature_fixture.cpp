#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "../src/models/loaders.h"
#include "../src/models/style_bert_vits2_jp_bert/model.h"

namespace {
std::string gguf_path() {
    const char * env = std::getenv("STYLE_BERT_VITS2_JP_BERT_FULL_GGUF");
    if (env && env[0]) {
        return env;
    }
    return "tmp/style-bert-vits2-jp-bert.gguf";
}
}

int main() {
    const std::string path = gguf_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::fprintf(stderr, "Style-Bert JP BERT full GGUF missing; skipping feature fixture test: %s\n", path.c_str());
        return 0;
    }

    generation_configuration config;
    const bool cpu_only = !(std::getenv("STYLE_BERT_VITS2_JP_BERT_TEST_ACCELERATOR") &&
                            std::getenv("STYLE_BERT_VITS2_JP_BERT_TEST_ACCELERATOR")[0]);
    std::unique_ptr<tts_generation_runner> base = runner_from_file(path.c_str(), 4, config, cpu_only);
    auto * runner = dynamic_cast<style_bert_vits2_jp_bert_runner *>(base.get());
    if (!runner) {
        std::fprintf(stderr, "loaded runner is not style_bert_vits2_jp_bert_runner\n");
        return 1;
    }
    if (runner->model->n_layers != 24) {
        std::fprintf(stderr, "expected full JP BERT with 24 layers, got %u\n", runner->model->n_layers);
        return 1;
    }

    const int32_t input_ids[] = {1, 880, 9, 961, 2};
    std::vector<float> features;
    runner->encode_features(input_ids, 5, features);
    if (features.size() != 5 * 1024) {
        std::fprintf(stderr, "feature size mismatch: actual=%zu expected=%d\n", features.size(), 5 * 1024);
        return 1;
    }
    double checksum = 0.0;
    for (size_t i = 0; i < features.size(); ++i) {
        if (!std::isfinite(features[i])) {
            std::fprintf(stderr, "non-finite feature value at %zu\n", i);
            return 1;
        }
        checksum += features[i];
    }
    constexpr double expected_checksum = -9.803918839;
    if (std::fabs(checksum - expected_checksum) > 0.05) {
        std::fprintf(stderr, "feature checksum mismatch: actual=%.9f expected=%.9f\n", checksum, expected_checksum);
        return 1;
    }

    std::fprintf(stdout, "STYLE_BERT_VITS2_JP_BERT_FEATURE_FIXTURE layers=%u checksum=%.9f\n",
                 runner->model->n_layers, checksum);
    return 0;
}
