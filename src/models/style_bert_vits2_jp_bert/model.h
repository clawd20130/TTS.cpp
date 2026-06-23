#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../../tts_model.h"
#include "../loaders.h"

void style_bert_vits2_jp_bert_build_relative_position_indices(uint32_t query_size, uint32_t key_size,
                                                              uint32_t bucket_size, uint32_t max_position,
                                                              std::vector<int32_t> & c2p_indices,
                                                              std::vector<int32_t> & p2c_indices);

struct style_bert_vits2_jp_bert_model : tts_model {
    static constexpr const char * arch = "style-bert-vits2-jp-bert";

    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t n_layers = 0;
    uint32_t n_attn_heads = 0;
    uint32_t head_size = 0;
    uint32_t max_position_embeddings = 0;
    uint32_t position_buckets = 0;
    int32_t max_relative_positions = 0;
    uint32_t type_vocab_size = 0;
    uint32_t pad_token_id = 0;
    float layer_norm_eps = 0.0f;
    std::string hidden_act;
    bool relative_attention = false;
    bool position_biased_input = false;
    bool share_att_key = false;
    uint32_t feature_hidden_state_offset = 0;

    std::unordered_map<std::string, ggml_tensor *> tensors;

    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only = true);
    void assign_weight(const char * name, ggml_tensor & tensor);
    bool has_tensor(const char * name) const;
    ggml_tensor * tensor(const char * name) const;

  private:
    void prep_constants(gguf_context * meta_ctx);
};

struct style_bert_vits2_jp_bert_context : runner_context {
    style_bert_vits2_jp_bert_context(style_bert_vits2_jp_bert_model * model, int n_threads);
    ~style_bert_vits2_jp_bert_context() override = default;

    style_bert_vits2_jp_bert_model * model;
    ggml_tensor * input_ids = nullptr;
    ggml_tensor * c2p_flat_indices = nullptr;
    ggml_tensor * p2c_flat_indices = nullptr;

    void build_schedule();
};

struct style_bert_vits2_jp_bert_runner : tts_generation_runner {
    style_bert_vits2_jp_bert_runner(std::unique_ptr<style_bert_vits2_jp_bert_model> model,
                                    style_bert_vits2_jp_bert_context * context);
    ~style_bert_vits2_jp_bert_runner() override;

    std::unique_ptr<style_bert_vits2_jp_bert_model> model;
    style_bert_vits2_jp_bert_context * bctx;

    void assign_weight(const char * name, ggml_tensor & tensor) override;
    void prepare_post_load() override;
    ggml_cgraph * build_embedding_graph(uint32_t tokens);
    void set_embedding_inputs(const int32_t * input_ids, uint32_t tokens);
    void encode_embeddings(const int32_t * input_ids, uint32_t tokens, std::vector<float> & output);
    ggml_cgraph * build_layer0_graph(uint32_t tokens);
    void set_layer_inputs(const int32_t * input_ids, uint32_t tokens);
    void encode_layer0(const int32_t * input_ids, uint32_t tokens, std::vector<float> & output);
    ggml_cgraph * build_layer0_with_conv_graph(uint32_t tokens);
    void encode_layer0_with_conv(const int32_t * input_ids, uint32_t tokens, std::vector<float> & output);
    ggml_cgraph * build_feature_graph(uint32_t tokens);
    void encode_features(const int32_t * input_ids, uint32_t tokens, std::vector<float> & output);
    void generate(const char * sentence, tts_response & output, const generation_configuration & config) override;
};

struct style_bert_vits2_jp_bert_model_loader : tts_model_loader {
    style_bert_vits2_jp_bert_model_loader();
    unique_ptr<tts_generation_runner> from_file(
        gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, bool cpu_only,
        const generation_configuration & config) const override;
};

extern const style_bert_vits2_jp_bert_model_loader style_bert_vits2_jp_bert_loader;
