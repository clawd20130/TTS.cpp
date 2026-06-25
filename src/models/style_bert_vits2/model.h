#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../../tts_model.h"
#include "models/loaders.h"

extern const struct style_bert_vits2_model_loader final : tts_model_loader {
    explicit style_bert_vits2_model_loader();

    unique_ptr<tts_generation_runner> from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads,
                                                bool cpu_only, const generation_configuration & config) const override;
} style_bert_vits2_loader;

inline float style_bert_vits2_ctx_size_offset() {
    const char * value = std::getenv("STYLE_BERT_VITS2_CTX_SIZE_OFFSET");
    if (!value || !value[0]) {
        return 4.0f;
    }
    char * end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end != value && parsed >= 2.0f) {
        return parsed;
    }
    return 4.0f;
}

struct style_bert_vits2_conv1d {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
    uint32_t kernel = 1;
    uint32_t padding = 0;
    uint32_t dilation = 1;
};

struct style_bert_vits2_upsample {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
    uint32_t kernel = 1;
    uint32_t stride = 1;
    uint32_t padding = 0;
};

struct style_bert_vits2_resblock {
    std::vector<style_bert_vits2_conv1d> convs1;
    std::vector<style_bert_vits2_conv1d> convs2;
};

struct style_bert_vits2_attention_layer {
    style_bert_vits2_conv1d conv_q;
    style_bert_vits2_conv1d conv_k;
    style_bert_vits2_conv1d conv_v;
    style_bert_vits2_conv1d conv_o;
    ggml_tensor * emb_rel_k = nullptr;
    ggml_tensor * emb_rel_v = nullptr;
};

struct style_bert_vits2_ffn_layer {
    style_bert_vits2_conv1d conv_1;
    style_bert_vits2_conv1d conv_2;
};

struct style_bert_vits2_encoder {
    std::vector<style_bert_vits2_attention_layer> attn_layers;
    std::vector<style_bert_vits2_ffn_layer> ffn_layers;
    std::vector<ggml_tensor *> norm_layers_1_gamma;
    std::vector<ggml_tensor *> norm_layers_1_beta;
    std::vector<ggml_tensor *> norm_layers_2_gamma;
    std::vector<ggml_tensor *> norm_layers_2_beta;
    ggml_tensor * spk_emb_linear_weight = nullptr;
    ggml_tensor * spk_emb_linear_bias = nullptr;
    uint32_t n_layers = 0;
    uint32_t n_heads = 2;
    uint32_t kernel = 3;
    uint32_t padding = 1;
    uint32_t window_size = 4;
    uint32_t cond_layer_idx = 2;
};

struct style_bert_vits2_decoder {
    style_bert_vits2_conv1d conv_pre;
    style_bert_vits2_conv1d cond;
    style_bert_vits2_conv1d conv_post;
    std::vector<style_bert_vits2_upsample> ups;
    std::vector<style_bert_vits2_resblock> resblocks;
};

struct style_bert_vits2_text_encoder {
    ggml_tensor * token_embedding = nullptr;
    ggml_tensor * tone_embedding = nullptr;
    ggml_tensor * language_embedding = nullptr;
    style_bert_vits2_conv1d bert_proj;
    style_bert_vits2_conv1d ja_bert_proj;
    style_bert_vits2_conv1d en_bert_proj;
    style_bert_vits2_conv1d proj;
    style_bert_vits2_encoder encoder;
    ggml_tensor * style_proj_weight = nullptr;
    ggml_tensor * style_proj_bias = nullptr;
};

struct style_bert_vits2_duration_predictor {
    style_bert_vits2_conv1d conv_1;
    style_bert_vits2_conv1d conv_2;
    style_bert_vits2_conv1d proj;
    style_bert_vits2_conv1d cond;
    ggml_tensor * norm_1_gamma = nullptr;
    ggml_tensor * norm_1_beta = nullptr;
    ggml_tensor * norm_2_gamma = nullptr;
    ggml_tensor * norm_2_beta = nullptr;
    uint32_t filter_channels = 256;
};

struct style_bert_vits2_dds_conv {
    std::vector<style_bert_vits2_conv1d> convs_sep;
    std::vector<style_bert_vits2_conv1d> convs_1x1;
    std::vector<ggml_tensor *> norms_1_gamma;
    std::vector<ggml_tensor *> norms_1_beta;
    std::vector<ggml_tensor *> norms_2_gamma;
    std::vector<ggml_tensor *> norms_2_beta;
    uint32_t n_layers = 3;
    uint32_t kernel = 3;
};

struct style_bert_vits2_elementwise_affine {
    ggml_tensor * m = nullptr;
    ggml_tensor * logs = nullptr;
};

struct style_bert_vits2_conv_flow {
    style_bert_vits2_conv1d pre;
    style_bert_vits2_dds_conv convs;
    style_bert_vits2_conv1d proj;
    uint32_t num_bins = 10;
};

struct style_bert_vits2_stochastic_duration_predictor {
    style_bert_vits2_conv1d pre;
    style_bert_vits2_conv1d proj;
    style_bert_vits2_conv1d cond;
    style_bert_vits2_dds_conv convs;
    style_bert_vits2_elementwise_affine flow_affine;
    std::vector<style_bert_vits2_conv_flow> flows;

    style_bert_vits2_conv1d post_pre;
    style_bert_vits2_conv1d post_proj;
    style_bert_vits2_dds_conv post_convs;
    style_bert_vits2_elementwise_affine post_flow_affine;
    std::vector<style_bert_vits2_conv_flow> post_flows;

    uint32_t n_flows = 4;
    uint32_t n_layers = 3;
    uint32_t kernel = 3;
    uint32_t num_bins = 10;
};

struct style_bert_vits2_flow_layer {
    style_bert_vits2_conv1d pre;
    style_bert_vits2_encoder encoder;
    style_bert_vits2_conv1d post;
};

struct style_bert_vits2_flow {
    std::vector<style_bert_vits2_flow_layer> layers;
    uint32_t use_transformer = 1;
    uint32_t n_flows = 4;
    uint32_t n_layers = 6;
    uint32_t kernel = 5;
    uint32_t padding = 2;
    uint32_t window_size = 4;
    uint32_t cond_layer_idx = 2;
};

struct style_bert_vits2_alignment_result {
    uint32_t frames = 0;
    std::vector<float> w;
    std::vector<float> w_ceil;
    std::vector<float> y_mask;
    std::vector<float> attn;
    std::vector<float> m_p_expanded;
    std::vector<float> logs_p_expanded;
};

struct style_bert_vits2_model : tts_model {
    uint32_t decoder_only = 1;
    uint32_t sample_rate = 44100;
    uint32_t inter_channels = 192;
    uint32_t hidden_channels = 192;
    uint32_t gin_channels = 512;
    uint32_t jp_extra = 0;
    uint32_t upsample_initial_channel = 512;
    uint32_t num_upsamples = 5;
    uint32_t num_kernels = 3;
    uint32_t resblock = 1;

    ggml_tensor * speaker_embedding = nullptr;
    ggml_tensor * style_vectors = nullptr;
    style_bert_vits2_text_encoder text_encoder;
    style_bert_vits2_duration_predictor duration_predictor;
    style_bert_vits2_stochastic_duration_predictor stochastic_duration_predictor;
    style_bert_vits2_flow flow;
    style_bert_vits2_decoder decoder;

    void prep_constants(gguf_context * meta);
    void prep_layers(gguf_context * meta);
    void prep_encoder_layers(style_bert_vits2_encoder & encoder, uint32_t n_layers, uint32_t kernel, uint32_t window_size, uint32_t cond_layer_idx);
    void prep_dds_conv(style_bert_vits2_dds_conv & conv, uint32_t n_layers, uint32_t kernel);
    void prep_stochastic_duration_predictor();
    void assign_text_encoder_weight(std::string name, ggml_tensor * tensor);
    void assign_encoder_weight(style_bert_vits2_encoder & encoder, std::string name, ggml_tensor * tensor);
    void assign_duration_predictor_weight(std::string name, ggml_tensor * tensor);
    void assign_dds_conv_weight(style_bert_vits2_dds_conv & conv, std::string name, ggml_tensor * tensor);
    void assign_conv_flow_weight(style_bert_vits2_conv_flow & flow, std::string name, ggml_tensor * tensor);
    void assign_elementwise_affine_weight(style_bert_vits2_elementwise_affine & affine, std::string name, ggml_tensor * tensor);
    void assign_stochastic_duration_predictor_weight(std::string name, ggml_tensor * tensor);
    void assign_flow_weight(std::string name, ggml_tensor * tensor);
    void assign_decoder_weight(std::string name, ggml_tensor * tensor);
    void assign_weight(const char * name, ggml_tensor & tensor);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only = true) {
        prep_constants(meta_ctx);
        prep_layers(meta_ctx);
        tts_model::setup_from_file(meta_ctx, load_context, cpu_only, "style_bert_vits2", style_bert_vits2_ctx_size_offset());
    }
};

struct style_bert_vits2_context : runner_context {
    style_bert_vits2_context(style_bert_vits2_model * model, int n_threads): runner_context(n_threads), model(model) {}
    ~style_bert_vits2_context() override = default;

    style_bert_vits2_model * model;
    uint32_t decoder_frames = 0;
    ggml_tensor * decoder_z = nullptr;
    ggml_tensor * decoder_g = nullptr;
    ggml_tensor * speaker_id = nullptr;
    ggml_tensor * style_id = nullptr;
    ggml_tensor * style_mean_id = nullptr;
    ggml_tensor * style_weight = nullptr;
    uint32_t text_tokens = 0;
    ggml_tensor * phone_ids = nullptr;
    ggml_tensor * tone_ids = nullptr;
    ggml_tensor * language_ids = nullptr;
    ggml_tensor * bert = nullptr;
    ggml_tensor * style_vec = nullptr;
    ggml_tensor * duration_x = nullptr;
    ggml_tensor * duration_x_mask = nullptr;
    ggml_tensor * duration_g = nullptr;
    ggml_tensor * sdp_x = nullptr;
    ggml_tensor * sdp_x_mask = nullptr;
    ggml_tensor * sdp_g = nullptr;
    ggml_tensor * sdp_reverse_z = nullptr;
    ggml_tensor * sdp_reverse_x_mask = nullptr;
    ggml_tensor * sdp_reverse_condition = nullptr;
    ggml_tensor * text_projection_x = nullptr;
    ggml_tensor * text_projection_x_mask = nullptr;
    ggml_tensor * text_encoder_x_output = nullptr;
    ggml_tensor * text_encoder_stats_output = nullptr;
    ggml_tensor * encoder_attn_x = nullptr;
    ggml_tensor * encoder_attn_x_mask = nullptr;
    ggml_tensor * encoder_layer_x = nullptr;
    ggml_tensor * encoder_layer_x_mask = nullptr;
    ggml_tensor * encoder_layer_g = nullptr;
    ggml_tensor * encoder_rel_pos_ids = nullptr;
    ggml_tensor * encoder_ffn_x = nullptr;
    ggml_tensor * encoder_ffn_x_mask = nullptr;
    uint32_t prior_frames = 0;
    ggml_tensor * prior_m_p = nullptr;
    ggml_tensor * prior_logs_p = nullptr;
    ggml_tensor * prior_noise = nullptr;
    ggml_tensor * prior_noise_scale = nullptr;
    ggml_tensor * flow_z = nullptr;
    ggml_tensor * flow_y_mask = nullptr;
    ggml_tensor * flow_g = nullptr;

    void build_schedule(size_t max_nodes = 400000) {
        runner_context::build_schedule(std::max<size_t>(model->max_nodes(), max_nodes));
    }
};

struct style_bert_vits2_runner : tts_generation_runner {
    style_bert_vits2_runner(unique_ptr<style_bert_vits2_model> model, style_bert_vits2_context * context):
        tts_generation_runner{style_bert_vits2_loader}, model{move(model)}, sctx(context) {
        tts_runner::sampling_rate = (float) this->model->sample_rate;
    }
    ~style_bert_vits2_runner() override {
        if (ctx) {
            ggml_free(ctx);
        }
        delete sctx;
        model->free();
    }

    unique_ptr<style_bert_vits2_model> model;
    style_bert_vits2_context * sctx;

    void init_build() {
        tts_runner::init_build(&sctx->buf_compute_meta);
    }

    void assign_weight(const char * name, ggml_tensor & tensor) override;
    void prepare_post_load() override;
    ggml_cgraph * build_speaker_embedding_graph();
    void set_speaker_embedding_inputs(int32_t speaker_id);
    void encode_speaker(int32_t speaker_id, std::vector<float> & output);
    ggml_cgraph * build_style_vector_graph();
    void set_style_vector_inputs(int32_t style_id, float style_weight);
    void encode_style_vector(int32_t style_id, float style_weight, std::vector<float> & output);
    ggml_cgraph * build_text_encoder_input_graph(uint32_t tokens);
    void set_text_encoder_input_inputs(const int32_t * phone_ids, const int32_t * tone_ids, const int32_t * language_ids,
                                       const float * bert_t_major, const float * style_vec, uint32_t tokens);
    void encode_text_encoder_input(const int32_t * phone_ids, const int32_t * tone_ids, const int32_t * language_ids,
                                   const float * bert_t_major, const float * style_vec, uint32_t tokens,
                                   std::vector<float> & output);
    ggml_cgraph * build_duration_predictor_graph(uint32_t tokens);
    void set_duration_predictor_inputs(const float * x_nct, const float * x_mask_nct, const float * g_nct,
                                       uint32_t tokens);
    void predict_duration(const float * x_nct, const float * x_mask_nct, const float * g_nct, uint32_t tokens,
                          std::vector<float> & output);
    ggml_cgraph * build_stochastic_duration_condition_graph(uint32_t tokens);
    void set_stochastic_duration_condition_inputs(const float * x_nct, const float * x_mask_nct, const float * g_nct,
                                                  uint32_t tokens);
    void run_stochastic_duration_condition(const float * x_nct, const float * x_mask_nct, const float * g_nct,
                                           uint32_t tokens, std::vector<float> & output);
    ggml_cgraph * build_stochastic_duration_reverse_graph(uint32_t tokens);
    void set_stochastic_duration_reverse_inputs(const float * z_nct, const float * x_mask_nct,
                                                const float * condition_nct, uint32_t tokens);
    void run_stochastic_duration_reverse(const float * z_nct, const float * x_mask_nct, const float * condition_nct,
                                         uint32_t tokens, std::vector<float> & output);
    ggml_cgraph * build_text_encoder_projection_graph(uint32_t tokens);
    void set_text_encoder_projection_inputs(const float * x_nct, const float * x_mask_nct, uint32_t tokens);
    void project_text_encoder(const float * x_nct, const float * x_mask_nct, uint32_t tokens,
                              std::vector<float> & m_p_nct, std::vector<float> & logs_p_nct);
    ggml_cgraph * build_text_encoder_attention_graph(uint32_t layer_index, uint32_t tokens);
    void set_text_encoder_attention_inputs(const float * x_nct, const float * x_mask_nct, uint32_t tokens);
    void run_text_encoder_attention(uint32_t layer_index, const float * x_nct, const float * x_mask_nct,
                                    uint32_t tokens, std::vector<float> & output);
    ggml_cgraph * build_text_encoder_layer_graph(uint32_t layer_index, uint32_t tokens);
    void set_text_encoder_layer_inputs(const float * x_nct, const float * x_mask_nct, const float * g_nct,
                                       uint32_t tokens);
    void run_text_encoder_layer(uint32_t layer_index, const float * x_nct, const float * x_mask_nct,
                                const float * g_nct, uint32_t tokens, std::vector<float> & output);
    ggml_cgraph * build_text_encoder_stack_graph(uint32_t tokens);
    void run_text_encoder_stack(const float * x_nct, const float * x_mask_nct, const float * g_nct,
                                uint32_t tokens, std::vector<float> & output);
    ggml_cgraph * build_text_encoder_graph(uint32_t tokens);
    void run_text_encoder(const int32_t * phone_ids, const int32_t * tone_ids, const int32_t * language_ids,
                          const float * bert_t_major, const float * style_vec, const float * x_mask_nct,
                          const float * g_nct, uint32_t tokens, std::vector<float> & x_nct,
                          std::vector<float> & m_p_nct, std::vector<float> & logs_p_nct);
    ggml_cgraph * build_text_encoder_ffn_graph(uint32_t layer_index, uint32_t tokens);
    void set_text_encoder_ffn_inputs(const float * x_nct, const float * x_mask_nct, uint32_t tokens);
    void run_text_encoder_ffn(uint32_t layer_index, const float * x_nct, const float * x_mask_nct, uint32_t tokens,
                              std::vector<float> & output);
    style_bert_vits2_alignment_result expand_alignment(const float * logw_nct,
                                                       const float * x_mask_nct,
                                                       const float * m_p_nct,
                                                       const float * logs_p_nct,
                                                       uint32_t tokens,
                                                       float length_scale);
    ggml_cgraph * build_prior_sample_graph(uint32_t frames);
    void set_prior_sample_inputs(const float * m_p_nct, const float * logs_p_nct, const float * noise_nct,
                                 uint32_t frames, float noise_scale);
    void sample_prior(const float * m_p_nct, const float * logs_p_nct, const float * noise_nct,
                      uint32_t frames, float noise_scale, std::vector<float> & output);
    ggml_cgraph * build_flow_reverse_step_graph(uint32_t flow_index, uint32_t frames);
    ggml_cgraph * build_flow_reverse_group_graph(uint32_t start_flow_index, uint32_t layer_count, uint32_t frames);
    ggml_cgraph * build_flow_reverse_graph(uint32_t frames);
    void set_flow_reverse_inputs(const float * z_p_nct, const float * y_mask_nct, const float * g_nct,
                                 uint32_t frames);
    void run_flow_reverse_step(uint32_t flow_index, const float * z_p_nct, const float * y_mask_nct,
                               const float * g_nct, uint32_t frames, std::vector<float> & output);
    void run_flow_reverse_group(uint32_t start_flow_index, uint32_t layer_count, const float * z_p_nct,
                                const float * y_mask_nct, const float * g_nct, uint32_t frames,
                                std::vector<float> & output);
    void run_flow_reverse_fused(const float * z_p_nct, const float * y_mask_nct, const float * g_nct,
                                uint32_t frames, std::vector<float> & output);
    void run_flow_reverse(const float * z_p_nct, const float * y_mask_nct, const float * g_nct,
                          uint32_t frames, std::vector<float> & output);
    style_bert_vits2_alignment_result build_decoder_latent(const float * logw_nct,
                                                           const float * x_mask_nct,
                                                           const float * m_p_nct,
                                                           const float * logs_p_nct,
                                                           const float * noise_nct,
                                                           const float * g_nct,
                                                           uint32_t tokens,
                                                           float length_scale,
                                                           float noise_scale,
                                                           std::vector<float> & decoder_z_nct);
    bool synthesize_latent(const std::vector<float> & logw,
                           const std::vector<float> & x_mask,
                           const std::vector<float> & m_p,
                           const std::vector<float> & logs_p,
                           const std::vector<float> & noise,
                           const std::vector<float> & g,
                           uint32_t tokens,
                           float length_scale,
                           float noise_scale,
                           tts_response & output,
                           style_bert_vits2_alignment_result & alignment,
                           std::string & error);
    bool synthesize_front(const std::vector<int32_t> & phone_ids,
                          const std::vector<int32_t> & tone_ids,
                          const std::vector<int32_t> & language_ids,
                          const std::vector<float> & bert,
                          int32_t speaker_id,
                          int32_t style_id,
                          float style_weight,
                          float sdp_ratio,
                          float length_scale,
                          float noise_scale,
                          float noise_w_scale,
                          tts_response & output,
                          style_bert_vits2_alignment_result & alignment,
                          std::string & error);
    bool synthesize_front_with_style_vec(const std::vector<int32_t> & phone_ids,
                                         const std::vector<int32_t> & tone_ids,
                                         const std::vector<int32_t> & language_ids,
                                         const std::vector<float> & bert,
                                         const std::vector<float> & style_vec,
                                         int32_t speaker_id,
                                         float sdp_ratio,
                                         float length_scale,
                                         float noise_scale,
                                         float noise_w_scale,
                                         tts_response & output,
                                         style_bert_vits2_alignment_result & alignment,
                                         std::string & error);
    ggml_cgraph * build_decoder_graph(uint32_t frames);
    void set_decoder_inputs(const float * decoder_z_nct, const float * decoder_g_nct, uint32_t frames);
    void decode(const float * decoder_z_nct, const float * decoder_g_nct, uint32_t frames, tts_response & output);
    void generate(const char * sentence, tts_response & output, const generation_configuration & config) override;
};

style_bert_vits2_context * build_new_style_bert_vits2_context(style_bert_vits2_model * model, int n_threads, bool use_cpu = true);
