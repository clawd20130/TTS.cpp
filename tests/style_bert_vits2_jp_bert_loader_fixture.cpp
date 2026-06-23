#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "../src/models/loaders.h"
#include "../src/models/style_bert_vits2_jp_bert/model.h"

namespace {
std::string gguf_path() {
    const char * env = std::getenv("STYLE_BERT_VITS2_JP_BERT_GGUF");
    if (env && env[0]) {
        return env;
    }
    return "tmp/style-bert-vits2-jp-bert-1layer.gguf";
}

bool expect_shape(const ggml_tensor * tensor, int n_dims, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1) {
    const int actual_dims = ggml_n_dims(tensor);
    if (actual_dims != n_dims) {
        std::fprintf(stderr, "%s n_dims mismatch: actual=%d expected=%d\n", tensor->name, actual_dims, n_dims);
        return false;
    }
    if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1 || tensor->ne[2] != ne2) {
        std::fprintf(stderr,
                     "%s shape mismatch: actual=[%lld,%lld,%lld] expected=[%lld,%lld,%lld]\n",
                     tensor->name,
                     (long long) tensor->ne[0],
                     (long long) tensor->ne[1],
                     (long long) tensor->ne[2],
                     (long long) ne0,
                     (long long) ne1,
                     (long long) ne2);
        return false;
    }
    return true;
}
}

int main() {
    const std::string path = gguf_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::fprintf(stderr, "Style-Bert JP BERT GGUF missing; skipping loader fixture test: %s\n", path.c_str());
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

    const style_bert_vits2_jp_bert_model * model = runner->model.get();
    if (model->vocab_size != 22012 || model->hidden_size != 1024 || model->intermediate_size != 4096 ||
        model->n_layers != 1 || model->n_attn_heads != 16 || model->head_size != 64 ||
        model->position_buckets != 256 || model->max_relative_positions != -1 ||
        !model->relative_attention || model->position_biased_input || !model->share_att_key ||
        model->feature_hidden_state_offset != 2) {
        std::fprintf(stderr,
                     "metadata mismatch: vocab=%u hidden=%u intermediate=%u layers=%u heads=%u head=%u buckets=%u max_rel=%d rel=%d pos_bias=%d share=%d offset=%u\n",
                     model->vocab_size,
                     model->hidden_size,
                     model->intermediate_size,
                     model->n_layers,
                     model->n_attn_heads,
                     model->head_size,
                     model->position_buckets,
                     model->max_relative_positions,
                     model->relative_attention,
                     model->position_biased_input,
                     model->share_att_key,
                     model->feature_hidden_state_offset);
        return 1;
    }

    if (model->tensors.size() != 26) {
        std::fprintf(stderr, "tensor count mismatch: actual=%zu expected=26\n", model->tensors.size());
        return 1;
    }

    bool ok = true;
    ok &= expect_shape(model->tensor("emb.word.weight"), 2, 1024, 22012);
    ok &= expect_shape(model->tensor("layers.0.attn.self.query.weight"), 2, 1024, 1024);
    ok &= expect_shape(model->tensor("layers.0.intermediate.dense.weight"), 2, 1024, 4096);
    ok &= expect_shape(model->tensor("enc.rel_embeddings.weight"), 2, 1024, 512);
    ok &= expect_shape(model->tensor("enc.conv.conv.weight"), 3, 3, 1024, 1024);
    if (!ok) {
        return 1;
    }

    std::vector<int32_t> c2p_indices;
    std::vector<int32_t> p2c_indices;
    style_bert_vits2_jp_bert_build_relative_position_indices(5, 5, model->position_buckets,
                                                             model->max_relative_positions < 1
                                                                 ? model->max_position_embeddings
                                                                 : (uint32_t) model->max_relative_positions,
                                                             c2p_indices, p2c_indices);
    const std::vector<int32_t> expected_c2p = {
        256, 255, 254, 253, 252,
        257, 256, 255, 254, 253,
        258, 257, 256, 255, 254,
        259, 258, 257, 256, 255,
        260, 259, 258, 257, 256,
    };
    const std::vector<int32_t> expected_p2c = {
        256, 257, 258, 259, 260,
        255, 256, 257, 258, 259,
        254, 255, 256, 257, 258,
        253, 254, 255, 256, 257,
        252, 253, 254, 255, 256,
    };
    if (c2p_indices != expected_c2p || p2c_indices != expected_p2c) {
        std::fprintf(stderr, "relative position indices mismatch\n");
        return 1;
    }

    const int32_t input_ids[] = {1, 880, 9, 961, 2};
    std::vector<float> embeddings;
    runner->encode_embeddings(input_ids, 5, embeddings);
    if (embeddings.size() != 5 * 1024) {
        std::fprintf(stderr, "embedding size mismatch: actual=%zu expected=%d\n", embeddings.size(), 5 * 1024);
        return 1;
    }
    double checksum = 0.0;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        if (!std::isfinite(embeddings[i])) {
            std::fprintf(stderr, "non-finite embedding value at %zu\n", i);
            return 1;
        }
        checksum += embeddings[i];
    }
    constexpr double expected_embedding_checksum = 29.254646301;
    if (std::fabs(checksum - expected_embedding_checksum) > 1e-3) {
        std::fprintf(stderr,
                     "embedding checksum mismatch: actual=%.9f expected=%.9f\n",
                     checksum,
                     expected_embedding_checksum);
        return 1;
    }

    std::vector<float> layer0;
    runner->encode_layer0(input_ids, 5, layer0);
    if (layer0.size() != 5 * 1024) {
        std::fprintf(stderr, "layer0 size mismatch: actual=%zu expected=%d\n", layer0.size(), 5 * 1024);
        return 1;
    }
    double layer0_checksum = 0.0;
    for (size_t i = 0; i < layer0.size(); ++i) {
        if (!std::isfinite(layer0[i])) {
            std::fprintf(stderr, "non-finite layer0 value at %zu\n", i);
            return 1;
        }
        layer0_checksum += layer0[i];
    }
    constexpr double expected_layer0_checksum = 7.514720917;
    if (std::fabs(layer0_checksum - expected_layer0_checksum) > 1e-3) {
        std::fprintf(stderr,
                     "layer0 checksum mismatch: actual=%.9f expected=%.9f\n",
                     layer0_checksum,
                     expected_layer0_checksum);
        return 1;
    }

    std::vector<float> layer0_conv;
    runner->encode_layer0_with_conv(input_ids, 5, layer0_conv);
    if (layer0_conv.size() != 5 * 1024) {
        std::fprintf(stderr, "layer0 conv size mismatch: actual=%zu expected=%d\n", layer0_conv.size(), 5 * 1024);
        return 1;
    }
    double layer0_conv_checksum = 0.0;
    for (size_t i = 0; i < layer0_conv.size(); ++i) {
        if (!std::isfinite(layer0_conv[i])) {
            std::fprintf(stderr, "non-finite layer0 conv value at %zu\n", i);
            return 1;
        }
        layer0_conv_checksum += layer0_conv[i];
    }
    constexpr double expected_layer0_conv_checksum = 52.063125610;
    if (std::fabs(layer0_conv_checksum - expected_layer0_conv_checksum) > 1e-3) {
        std::fprintf(stderr,
                     "layer0 conv checksum mismatch: actual=%.9f expected=%.9f\n",
                     layer0_conv_checksum,
                     expected_layer0_conv_checksum);
        return 1;
    }

    std::fprintf(stdout,
                 "STYLE_BERT_VITS2_JP_BERT_LOADER_FIXTURE tensors=%zu vocab=%u hidden=%u layers=%u embedding_checksum=%.9f layer0_checksum=%.9f layer0_conv_checksum=%.9f\n",
                 model->tensors.size(),
                 model->vocab_size,
                 model->hidden_size,
                 model->n_layers,
                 checksum,
                 layer0_checksum,
                 layer0_conv_checksum);
    return 0;
}
