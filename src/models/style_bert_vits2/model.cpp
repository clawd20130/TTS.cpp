#include "model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {
static uint32_t read_u32(gguf_context * meta, const std::string & key, uint32_t fallback) {
    const int found = gguf_find_key(meta, key.c_str());
    return found == -1 ? fallback : gguf_get_val_u32(meta, found);
}

static std::string style_shape(const ggml_tensor * tensor) {
    if (!tensor) {
        return "[]";
    }
    return "[" + std::to_string(tensor->ne[0]) + "," + std::to_string(tensor->ne[1]) + "," +
        std::to_string(tensor->ne[2]) + "," + std::to_string(tensor->ne[3]) + "]";
}

static bool debug_timings_enabled() {
    const char * env = std::getenv("STYLE_BERT_VITS2_DEBUG_TIMINGS");
    return env && env[0] && std::strcmp(env, "0") != 0;
}

static bool debug_load_enabled() {
    const char * env = std::getenv("STYLE_BERT_VITS2_DEBUG_LOAD");
    return env && env[0] && std::strcmp(env, "0") != 0;
}

static bool flow_fused_enabled() {
    const char * env = std::getenv("STYLE_BERT_VITS2_FLOW_FUSED");
    return env && env[0] && std::strcmp(env, "0") != 0;
}

static bool attention_legacy_enabled() {
    const char * env = std::getenv("STYLE_BERT_VITS2_ATTENTION_LEGACY");
    return env && env[0] && std::strcmp(env, "0") != 0;
}

static std::string_view attention_mode() {
    const char * mode = std::getenv("STYLE_BERT_VITS2_ATTENTION_MODE");
    if (mode && mode[0]) {
        return std::string_view(mode);
    }
    const char * experiment = std::getenv("STYLE_BERT_VITS2_ATTENTION_EXPERIMENT");
    if (experiment && experiment[0]) {
        return std::string_view(experiment);
    }
    return "values";
}

static uint32_t flow_group_size_env(uint32_t max_layers) {
    const char * env = std::getenv("STYLE_BERT_VITS2_FLOW_GROUP_SIZE");
    if (!env || !env[0]) {
        return 1;
    }
    char * end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || parsed <= 0) {
        return 1;
    }
    uint32_t group_size = (uint32_t) parsed;
    return std::max<uint32_t>(1, std::min<uint32_t>(group_size, max_layers));
}

static size_t flow_graph_node_capacity(uint32_t frames, uint32_t layers) {
    const size_t estimated = 2048 + (size_t) frames * 320 * std::max<uint32_t>(layers, 1);
    return std::max<size_t>(400000, estimated);
}

static size_t style_bert_vits2_graph_max_nodes(size_t model_max_nodes) {
    size_t configured = 800000;
    const char * env = std::getenv("STYLE_BERT_VITS2_GRAPH_MAX_NODES");
    if (env && env[0]) {
        char * end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end != env && parsed >= 400000) {
            configured = (size_t) parsed;
        }
    }
    return std::max<size_t>(model_max_nodes, configured);
}

static const char * debug_output_node_name() {
    const char * env = std::getenv("STYLE_BERT_VITS2_DEBUG_OUTPUT_NODE");
    return env && env[0] ? env : nullptr;
}

static double elapsed_ms(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static bool debug_node_matches(ggml_tensor * tensor, const char * target) {
    return target && tensor && std::strcmp(ggml_get_name(tensor), target) == 0;
}

static ggml_tensor * conv1d(ggml_context * ctx, const style_bert_vits2_conv1d & conv, ggml_tensor * input) {
    TTS_ASSERT(conv.weight);
    TTS_ASSERT(input);
    ggml_tensor * cur = ggml_conv_1d(ctx, conv.weight, input, 1, (int) conv.padding, (int) conv.dilation);
    if (conv.bias) {
        cur = ggml_add(ctx, cur, conv.bias);
    }
    return cur;
}

static ggml_tensor * depthwise_conv1d(ggml_context * ctx, const style_bert_vits2_conv1d & conv, ggml_tensor * input) {
    TTS_ASSERT(conv.weight);
    TTS_ASSERT(input);
    TTS_ASSERT(conv.weight->ne[1] == 1);
    TTS_ASSERT(conv.weight->ne[2] == input->ne[1]);
    ggml_tensor * cur = ggml_conv_1d_dw(ctx, conv.weight, input, 1, (int) conv.padding, (int) conv.dilation);
    if (conv.bias) {
        cur = ggml_add(ctx, cur, conv.bias);
    }
    return cur;
}

static ggml_tensor * style_layer_norm(ggml_context * ctx, ggml_tensor * input, ggml_tensor * gamma, ggml_tensor * beta) {
    TTS_ASSERT(input);
    TTS_ASSERT(gamma);
    TTS_ASSERT(beta);
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, input));
    cur = ggml_norm(ctx, cur, 1e-5f);
    cur = ggml_add(ctx, ggml_mul(ctx, cur, gamma), beta);
    return ggml_cont(ctx, ggml_transpose(ctx, cur));
}

static ggml_tensor * apply_time_mask(ggml_context * ctx, ggml_tensor * input, ggml_tensor * mask) {
    TTS_ASSERT(input);
    TTS_ASSERT(mask);
    return ggml_mul(ctx, input, mask);
}

static ggml_tensor * constant_like(ggml_context * ctx, ggml_tensor * one, ggml_tensor * target, float value) {
    return ggml_scale(ctx, ggml_cont(ctx, ggml_repeat(ctx, one, target)), value);
}

static ggml_tensor * row_slice(ggml_context * ctx, ggml_tensor * input, uint32_t start, uint32_t rows) {
    TTS_ASSERT(input);
    TTS_ASSERT((int64_t) start + (int64_t) rows <= input->ne[0]);
    return ggml_view_3d(ctx,
                        input,
                        rows,
                        input->ne[1],
                        input->ne[2],
                        input->nb[1],
                        input->nb[2],
                        (size_t) start * input->nb[0]);
}

static ggml_tensor * time_channel_slice(
    ggml_context * ctx,
    ggml_tensor * input,
    uint32_t start_channel,
    uint32_t channels) {
    TTS_ASSERT(input);
    TTS_ASSERT((int64_t) start_channel + (int64_t) channels <= input->ne[1]);
    return ggml_view_3d(ctx,
                        input,
                        input->ne[0],
                        channels,
                        input->ne[2],
                        input->nb[1],
                        input->nb[2],
                        (size_t) start_channel * input->nb[1]);
}

static ggml_tensor * bins_first(ggml_context * ctx, ggml_tensor * input) {
    return ggml_cont(ctx, ggml_transpose(ctx, input));
}

static ggml_tensor * time_first(ggml_context * ctx, ggml_tensor * input) {
    return ggml_cont(ctx, ggml_transpose(ctx, input));
}

static ggml_tensor * embedding_lookup_t_major(ggml_context * ctx, ggml_tensor * embedding, ggml_tensor * ids,
                                              uint32_t tokens, uint32_t hidden_channels) {
    TTS_ASSERT(embedding);
    TTS_ASSERT(ids);
    ggml_tensor * rows = ggml_get_rows(ctx, embedding, ids);
    ggml_tensor * transposed = ggml_cont(ctx, ggml_transpose(ctx, rows));
    return ggml_reshape_3d(ctx, transposed, tokens, hidden_channels, 1);
}

static ggml_tensor * build_speaker_embedding(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx) {
    TTS_ASSERT(model->speaker_embedding);
    TTS_ASSERT(sctx->speaker_id);
    ggml_tensor * row = ggml_get_rows(ctx, model->speaker_embedding, sctx->speaker_id);
    row = ggml_cont(ctx, row);
    row = ggml_reshape_3d(ctx, row, 1, model->gin_channels, 1);
    ggml_set_name(row, "style_bert_vits2.speaker_g");
    return row;
}

static ggml_tensor * build_style_vector(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx) {
    TTS_ASSERT(model->style_vectors);
    TTS_ASSERT(sctx->style_id);
    TTS_ASSERT(sctx->style_mean_id);
    TTS_ASSERT(sctx->style_weight);
    ggml_tensor * mean = ggml_cont(ctx, ggml_get_rows(ctx, model->style_vectors, sctx->style_mean_id));
    ggml_tensor * style = ggml_cont(ctx, ggml_get_rows(ctx, model->style_vectors, sctx->style_id));
    ggml_tensor * weight = ggml_repeat(ctx, sctx->style_weight, style);
    ggml_tensor * cur = ggml_add(ctx, mean, ggml_mul(ctx, ggml_sub(ctx, style, mean), weight));
    ggml_set_name(cur, "style_bert_vits2.style_vector");
    return cur;
}

static ggml_tensor * build_text_encoder_input(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx) {
    TTS_ASSERT(model->text_encoder.token_embedding);
    TTS_ASSERT(model->text_encoder.tone_embedding);
    TTS_ASSERT(model->text_encoder.language_embedding);
    TTS_ASSERT(model->text_encoder.bert_proj.weight);
    TTS_ASSERT(model->text_encoder.style_proj_weight);
    TTS_ASSERT(model->text_encoder.style_proj_bias);
    TTS_ASSERT(sctx->phone_ids);
    TTS_ASSERT(sctx->tone_ids);
    TTS_ASSERT(sctx->language_ids);
    TTS_ASSERT(sctx->bert);
    TTS_ASSERT(sctx->style_vec);

    ggml_tensor * token = embedding_lookup_t_major(ctx, model->text_encoder.token_embedding, sctx->phone_ids,
                                                   sctx->text_tokens, model->hidden_channels);
    ggml_set_name(token, "style_bert_vits2.text_encoder.token_embedding");
    ggml_tensor * tone = embedding_lookup_t_major(ctx, model->text_encoder.tone_embedding, sctx->tone_ids,
                                                  sctx->text_tokens, model->hidden_channels);
    ggml_set_name(tone, "style_bert_vits2.text_encoder.tone_embedding");
    ggml_tensor * language = embedding_lookup_t_major(ctx, model->text_encoder.language_embedding, sctx->language_ids,
                                                      sctx->text_tokens, model->hidden_channels);
    ggml_set_name(language, "style_bert_vits2.text_encoder.language_embedding");

    ggml_tensor * bert = conv1d(ctx, model->text_encoder.bert_proj, sctx->bert);
    ggml_set_name(bert, "style_bert_vits2.text_encoder.bert_proj");

    ggml_tensor * style = ggml_mul_mat(ctx, model->text_encoder.style_proj_weight, sctx->style_vec);
    style = ggml_add(ctx, style, model->text_encoder.style_proj_bias);
    style = ggml_reshape_3d(ctx, style, 1, model->hidden_channels, 1);
    style = ggml_repeat(ctx, style, token);
    ggml_set_name(style, "style_bert_vits2.text_encoder.style_proj");

    ggml_tensor * cur = ggml_add(ctx, token, tone);
    cur = ggml_add(ctx, cur, language);
    cur = ggml_add(ctx, cur, bert);
    cur = ggml_add(ctx, cur, style);
    cur = ggml_scale(ctx, cur, std::sqrt((float) model->hidden_channels));
    ggml_set_name(cur, "style_bert_vits2.text_encoder.enc_input");
    return cur;
}

static ggml_tensor * build_duration_predictor(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx) {
    const auto & dp = model->duration_predictor;
    TTS_ASSERT(dp.conv_1.weight);
    TTS_ASSERT(dp.conv_2.weight);
    TTS_ASSERT(dp.proj.weight);
    TTS_ASSERT(dp.cond.weight);
    TTS_ASSERT(dp.norm_1_gamma);
    TTS_ASSERT(dp.norm_1_beta);
    TTS_ASSERT(dp.norm_2_gamma);
    TTS_ASSERT(dp.norm_2_beta);
    TTS_ASSERT(sctx->duration_x);
    TTS_ASSERT(sctx->duration_x_mask);
    TTS_ASSERT(sctx->duration_g);

    ggml_tensor * cond = conv1d(ctx, dp.cond, sctx->duration_g);
    cond = ggml_repeat(ctx, cond, sctx->duration_x);
    ggml_set_name(cond, "style_bert_vits2.duration_predictor.cond");

    ggml_tensor * cur = ggml_add(ctx, sctx->duration_x, cond);
    cur = conv1d(ctx, dp.conv_1, apply_time_mask(ctx, cur, sctx->duration_x_mask));
    ggml_set_name(cur, "style_bert_vits2.duration_predictor.conv_1");
    cur = ggml_relu(ctx, cur);
    cur = style_layer_norm(ctx, cur, dp.norm_1_gamma, dp.norm_1_beta);
    ggml_set_name(cur, "style_bert_vits2.duration_predictor.norm_1");

    cur = conv1d(ctx, dp.conv_2, apply_time_mask(ctx, cur, sctx->duration_x_mask));
    ggml_set_name(cur, "style_bert_vits2.duration_predictor.conv_2");
    cur = ggml_relu(ctx, cur);
    cur = style_layer_norm(ctx, cur, dp.norm_2_gamma, dp.norm_2_beta);
    ggml_set_name(cur, "style_bert_vits2.duration_predictor.norm_2");

    cur = conv1d(ctx, dp.proj, apply_time_mask(ctx, cur, sctx->duration_x_mask));
    cur = apply_time_mask(ctx, cur, sctx->duration_x_mask);
    ggml_set_name(cur, "style_bert_vits2.duration_predictor.logw_dp");
    return cur;
}

static ggml_tensor * build_dds_conv(
    ggml_context * ctx,
    const style_bert_vits2_dds_conv & conv,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * g = nullptr) {
    TTS_ASSERT(input);
    TTS_ASSERT(mask);
    ggml_tensor * cur = input;
    if (g) {
        cur = ggml_add(ctx, cur, ggml_repeat(ctx, g, cur));
    }
    for (uint32_t i = 0; i < conv.n_layers; ++i) {
        TTS_ASSERT(conv.convs_sep.at(i).weight);
        TTS_ASSERT(conv.convs_1x1.at(i).weight);
        TTS_ASSERT(conv.norms_1_gamma.at(i));
        TTS_ASSERT(conv.norms_1_beta.at(i));
        TTS_ASSERT(conv.norms_2_gamma.at(i));
        TTS_ASSERT(conv.norms_2_beta.at(i));

        ggml_tensor * y = depthwise_conv1d(ctx, conv.convs_sep.at(i), apply_time_mask(ctx, cur, mask));
        y = style_layer_norm(ctx, y, conv.norms_1_gamma.at(i), conv.norms_1_beta.at(i));
        y = ggml_gelu(ctx, y);
        y = conv1d(ctx, conv.convs_1x1.at(i), y);
        y = style_layer_norm(ctx, y, conv.norms_2_gamma.at(i), conv.norms_2_beta.at(i));
        y = ggml_gelu(ctx, y);
        cur = ggml_add(ctx, cur, y);
    }
    return apply_time_mask(ctx, cur, mask);
}

static ggml_tensor * build_stochastic_duration_condition(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx) {
    const auto & sdp = model->stochastic_duration_predictor;
    TTS_ASSERT(sdp.pre.weight);
    TTS_ASSERT(sdp.proj.weight);
    TTS_ASSERT(sdp.cond.weight);
    TTS_ASSERT(sctx->sdp_x);
    TTS_ASSERT(sctx->sdp_x_mask);
    TTS_ASSERT(sctx->sdp_g);

    ggml_tensor * cur = conv1d(ctx, sdp.pre, sctx->sdp_x);
    ggml_tensor * cond = conv1d(ctx, sdp.cond, sctx->sdp_g);
    cur = ggml_add(ctx, cur, ggml_repeat(ctx, cond, cur));
    cur = build_dds_conv(ctx, sdp.convs, cur, sctx->sdp_x_mask);
    cur = conv1d(ctx, sdp.proj, cur);
    cur = apply_time_mask(ctx, cur, sctx->sdp_x_mask);
    ggml_set_name(cur, "style_bert_vits2.sdp.condition");
    return cur;
}

static ggml_tensor * softplus(ggml_context * ctx, ggml_tensor * input, ggml_tensor * one) {
    return ggml_log(ctx, ggml_add(ctx, constant_like(ctx, one, input, 1.0f), ggml_exp(ctx, input)));
}

static ggml_tensor * spline_cum_locations(
    ggml_context * ctx,
    ggml_tensor * unnormalized,
    ggml_tensor * one_row,
    float min_bin,
    float left,
    float right,
    uint32_t num_bins) {
    ggml_tensor * probs = ggml_soft_max(ctx, unnormalized);
    ggml_tensor * widths = ggml_scale(ctx, probs, 1.0f - min_bin * (float) num_bins);
    widths = ggml_add(ctx, widths, constant_like(ctx, one_row, widths, min_bin));

    ggml_tensor * cumulative = ggml_cumsum(ctx, widths);
    ggml_tensor * left_edge = ggml_scale(ctx, one_row, left);
    ggml_tensor * right_edge = ggml_scale(ctx, one_row, right);
    ggml_tensor * middle = row_slice(ctx, cumulative, 0, num_bins - 1);
    middle = ggml_add(ctx, ggml_scale(ctx, ggml_cont(ctx, middle), right - left), constant_like(ctx, one_row, middle, left));
    return ggml_concat(ctx, ggml_concat(ctx, left_edge, middle, 0), right_edge, 0);
}

static ggml_tensor * build_unconstrained_rational_quadratic_spline_reverse(
    ggml_context * ctx,
    ggml_tensor * input_time,
    ggml_tensor * mask_time,
    ggml_tensor * h_time,
    uint32_t filter_channels,
    uint32_t num_bins) {
    constexpr float min_bin_width = 1e-3f;
    constexpr float min_bin_height = 1e-3f;
    constexpr float min_derivative = 1e-3f;
    constexpr float tail_bound = 5.0f;
    constexpr float eps = 1e-6f;

    ggml_tensor * widths_raw = bins_first(ctx, time_channel_slice(ctx, h_time, 0, num_bins));
    ggml_tensor * heights_raw = bins_first(ctx, time_channel_slice(ctx, h_time, num_bins, num_bins));
    ggml_tensor * derivatives_raw = bins_first(ctx, time_channel_slice(ctx, h_time, num_bins * 2, num_bins - 1));
    ggml_tensor * one_row = bins_first(ctx, mask_time);
    ggml_tensor * one_time = time_first(ctx, one_row);

    const float inv_sqrt_filter = 1.0f / std::sqrt((float) filter_channels);
    widths_raw = ggml_scale(ctx, widths_raw, inv_sqrt_filter);
    heights_raw = ggml_scale(ctx, heights_raw, inv_sqrt_filter);

    ggml_tensor * cumwidths = spline_cum_locations(ctx, widths_raw, one_row, min_bin_width, -tail_bound, tail_bound, num_bins);
    ggml_tensor * cumheights = spline_cum_locations(ctx, heights_raw, one_row, min_bin_height, -tail_bound, tail_bound, num_bins);
    ggml_tensor * widths = ggml_sub(ctx, row_slice(ctx, cumwidths, 1, num_bins), row_slice(ctx, cumwidths, 0, num_bins));
    ggml_tensor * heights = ggml_sub(ctx, row_slice(ctx, cumheights, 1, num_bins), row_slice(ctx, cumheights, 0, num_bins));

    ggml_tensor * derivative_edge = ggml_scale(
        ctx,
        one_row,
        std::log(std::exp(1.0f - min_derivative) - 1.0f));
    ggml_tensor * derivatives = ggml_concat(ctx, ggml_concat(ctx, derivative_edge, derivatives_raw, 0), derivative_edge, 0);
    derivatives = ggml_add(ctx, softplus(ctx, derivatives, one_row), constant_like(ctx, one_row, derivatives, min_derivative));

    ggml_tensor * y_original = bins_first(ctx, input_time);
    ggml_tensor * y_bins = ggml_repeat(ctx, y_original, widths);
    ggml_tensor * y_clamped = ggml_clamp(ctx, y_bins, -tail_bound + eps, tail_bound - eps);

    ggml_tensor * cumheight_left = row_slice(ctx, cumheights, 0, num_bins);
    ggml_tensor * cumheight_right = row_slice(ctx, cumheights, 1, num_bins);
    ggml_tensor * lower = ggml_step(ctx, ggml_add(ctx, ggml_sub(ctx, y_bins, cumheight_left), constant_like(ctx, one_row, y_bins, eps)));
    ggml_tensor * upper = ggml_step(ctx, ggml_sub(ctx, cumheight_right, y_bins));
    ggml_tensor * bin_mask = ggml_mul(ctx, lower, upper);

    ggml_tensor * cumwidth_left = row_slice(ctx, cumwidths, 0, num_bins);
    ggml_tensor * delta = ggml_div(ctx, heights, widths);
    ggml_tensor * derivative = row_slice(ctx, derivatives, 0, num_bins);
    ggml_tensor * derivative_next = row_slice(ctx, derivatives, 1, num_bins);
    ggml_tensor * y_minus_cumheight = ggml_sub(ctx, y_clamped, cumheight_left);
    ggml_tensor * common = ggml_sub(ctx, ggml_add(ctx, derivative, derivative_next), ggml_scale(ctx, delta, 2.0f));

    ggml_tensor * a = ggml_add(
        ctx,
        ggml_mul(ctx, y_minus_cumheight, common),
        ggml_mul(ctx, heights, ggml_sub(ctx, delta, derivative)));
    ggml_tensor * b = ggml_sub(
        ctx,
        ggml_mul(ctx, heights, derivative),
        ggml_mul(ctx, y_minus_cumheight, common));
    ggml_tensor * c = ggml_neg(ctx, ggml_mul(ctx, delta, y_minus_cumheight));
    ggml_tensor * discriminant = ggml_sub(ctx, ggml_sqr(ctx, b), ggml_scale(ctx, ggml_mul(ctx, a, c), 4.0f));
    discriminant = ggml_clamp(ctx, discriminant, 0.0f, INFINITY);
    ggml_tensor * root = ggml_div(
        ctx,
        ggml_scale(ctx, c, 2.0f),
        ggml_sub(ctx, ggml_neg(ctx, b), ggml_sqrt(ctx, discriminant)));
    ggml_tensor * candidate = ggml_add(ctx, ggml_mul(ctx, root, widths), cumwidth_left);
    ggml_tensor * selected = time_first(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, candidate, bin_mask)));

    ggml_tensor * inside_lower = ggml_step(
        ctx,
        ggml_add(ctx, ggml_add(ctx, input_time, ggml_scale(ctx, one_time, tail_bound)), ggml_scale(ctx, one_time, eps)));
    ggml_tensor * inside_upper = ggml_step(
        ctx,
        ggml_add(ctx, ggml_sub(ctx, ggml_scale(ctx, one_time, tail_bound), input_time), ggml_scale(ctx, one_time, eps)));
    ggml_tensor * inside = ggml_mul(ctx, inside_lower, inside_upper);
    ggml_tensor * outside = ggml_sub(ctx, one_time, inside);
    return ggml_add(ctx, ggml_mul(ctx, selected, inside), ggml_mul(ctx, input_time, outside));
}

static ggml_tensor * build_sdp_conv_flow_reverse(
    ggml_context * ctx,
    const style_bert_vits2_conv_flow & flow,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * condition,
    uint32_t filter_channels,
    uint32_t num_bins) {
    TTS_ASSERT(input);
    TTS_ASSERT(mask);
    TTS_ASSERT(condition);
    TTS_ASSERT(flow.pre.weight);
    TTS_ASSERT(flow.proj.weight);

    ggml_tensor * x0 = time_channel_slice(ctx, input, 0, 1);
    ggml_tensor * x1 = time_channel_slice(ctx, input, 1, 1);
    ggml_tensor * h = conv1d(ctx, flow.pre, x0);
    h = build_dds_conv(ctx, flow.convs, h, mask, condition);
    h = conv1d(ctx, flow.proj, h);
    h = apply_time_mask(ctx, h, mask);
    x1 = build_unconstrained_rational_quadratic_spline_reverse(ctx, x1, mask, h, filter_channels, num_bins);
    ggml_tensor * cur = ggml_concat(ctx, x0, x1, 1);
    cur = apply_time_mask(ctx, cur, mask);
    ggml_set_name(cur, "style_bert_vits2.sdp.conv_flow.reverse");
    return cur;
}

static ggml_tensor * build_sdp_elementwise_affine_reverse(
    ggml_context * ctx,
    const style_bert_vits2_elementwise_affine & affine,
    ggml_tensor * input,
    ggml_tensor * mask) {
    TTS_ASSERT(affine.m);
    TTS_ASSERT(affine.logs);
    ggml_tensor * m = ggml_repeat(ctx, affine.m, input);
    ggml_tensor * logs = ggml_repeat(ctx, affine.logs, input);
    ggml_tensor * cur = ggml_mul(ctx, ggml_sub(ctx, input, m), ggml_exp(ctx, ggml_neg(ctx, logs)));
    cur = apply_time_mask(ctx, cur, mask);
    ggml_set_name(cur, "style_bert_vits2.sdp.elementwise_affine.reverse");
    return cur;
}

static ggml_tensor * build_stochastic_duration_reverse(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx) {
    const auto & sdp = model->stochastic_duration_predictor;
    TTS_ASSERT(sctx->sdp_reverse_z);
    TTS_ASSERT(sctx->sdp_reverse_x_mask);
    TTS_ASSERT(sctx->sdp_reverse_condition);
    TTS_ASSERT(sdp.flow_affine.m);
    TTS_ASSERT(sdp.flow_affine.logs);

    ggml_tensor * cur = sctx->sdp_reverse_z;
    for (int64_t i = (int64_t) sdp.flows.size() - 1; i >= 1; --i) {
        cur = ggml_concat(ctx, time_channel_slice(ctx, cur, 1, 1), time_channel_slice(ctx, cur, 0, 1), 1);
        ggml_set_name(cur, "style_bert_vits2.sdp.flip.reverse");
        cur = build_sdp_conv_flow_reverse(ctx,
                                          sdp.flows.at((size_t) i),
                                          cur,
                                          sctx->sdp_reverse_x_mask,
                                          sctx->sdp_reverse_condition,
                                          sdp.convs.convs_1x1.empty()
                                              ? model->hidden_channels
                                              : (uint32_t) sdp.convs.convs_1x1.front().weight->ne[1],
                                          sdp.num_bins);
    }
    cur = ggml_concat(ctx, time_channel_slice(ctx, cur, 1, 1), time_channel_slice(ctx, cur, 0, 1), 1);
    ggml_set_name(cur, "style_bert_vits2.sdp.flip.reverse.final");
    cur = build_sdp_elementwise_affine_reverse(ctx, sdp.flow_affine, cur, sctx->sdp_reverse_x_mask);
    ggml_tensor * logw = time_channel_slice(ctx, cur, 0, 1);
    logw = apply_time_mask(ctx, logw, sctx->sdp_reverse_x_mask);
    ggml_set_name(logw, "style_bert_vits2.sdp.logw");
    return logw;
}

static ggml_tensor * build_text_encoder_projection_from(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    ggml_tensor * input,
    ggml_tensor * mask) {
    TTS_ASSERT(model->text_encoder.proj.weight);
    TTS_ASSERT(input);
    TTS_ASSERT(mask);

    ggml_tensor * cur = conv1d(ctx, model->text_encoder.proj, input);
    cur = apply_time_mask(ctx, cur, mask);
    ggml_set_name(cur, "style_bert_vits2.text_encoder.stats");
    return cur;
}

static ggml_tensor * build_text_encoder_projection(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx) {
    TTS_ASSERT(sctx->text_projection_x);
    TTS_ASSERT(sctx->text_projection_x_mask);
    return build_text_encoder_projection_from(ctx, model, sctx->text_projection_x, sctx->text_projection_x_mask);
}

static ggml_tensor * build_text_encoder_ffn(
    ggml_context * ctx,
    const style_bert_vits2_ffn_layer & ffn,
    ggml_tensor * input,
    ggml_tensor * mask) {
    TTS_ASSERT(ffn.conv_1.weight);
    TTS_ASSERT(ffn.conv_2.weight);
    ggml_tensor * cur = conv1d(ctx, ffn.conv_1, apply_time_mask(ctx, input, mask));
    cur = ggml_relu(ctx, cur);
    cur = conv1d(ctx, ffn.conv_2, apply_time_mask(ctx, cur, mask));
    cur = apply_time_mask(ctx, cur, mask);
    ggml_set_name(cur, "style_bert_vits2.text_encoder.ffn");
    return cur;
}

static ggml_tensor * build_relative_embedding_with_zero_row(
    ggml_context * ctx,
    ggml_tensor * embedding) {
    TTS_ASSERT(embedding);
    ggml_tensor * first = ggml_view_2d(ctx, embedding, embedding->ne[0], 1, embedding->nb[1], 0);
    ggml_tensor * zero = ggml_scale(ctx, first, 0.0f);
    return ggml_concat(ctx, embedding, zero, 1);
}

static ggml_tensor * build_zero_rows_like(
    ggml_context * ctx,
    ggml_tensor * embedding,
    uint32_t rows) {
    TTS_ASSERT(embedding);
    TTS_ASSERT(rows > 0);
    ggml_tensor * first = ggml_view_2d(ctx, embedding, embedding->ne[0], 1, embedding->nb[1], 0);
    ggml_tensor * zero = ggml_scale(ctx, first, 0.0f);
    return ggml_repeat(ctx,
                       zero,
                       ggml_new_tensor_2d(ctx, embedding->type, embedding->ne[0], rows));
}

static ggml_tensor * build_zero_1d_like(
    ggml_context * ctx,
    ggml_tensor * input,
    uint32_t length) {
    TTS_ASSERT(input);
    TTS_ASSERT(length > 0);
    ggml_tensor * first = ggml_view_1d(ctx, input, 1, 0);
    ggml_tensor * zero = ggml_scale(ctx, first, 0.0f);
    return ggml_repeat(ctx, zero, ggml_new_tensor_1d(ctx, input->type, length));
}

static ggml_tensor * build_relative_embedding_for_length(
    ggml_context * ctx,
    ggml_tensor * embedding,
    uint32_t tokens,
    uint32_t window_size) {
    TTS_ASSERT(embedding);
    TTS_ASSERT(tokens > 0);
    const uint32_t target_rows = 2 * tokens - 1;
    if (tokens <= window_size + 1) {
        const uint32_t slice_start = window_size + 1 - tokens;
        TTS_ASSERT((int64_t) slice_start + (int64_t) target_rows <= embedding->ne[1]);
        return ggml_view_2d(ctx,
                            embedding,
                            embedding->ne[0],
                            target_rows,
                            embedding->nb[1],
                            (size_t) slice_start * embedding->nb[1]);
    }

    const uint32_t pad_rows = tokens - (window_size + 1);
    ggml_tensor * zero_pad = build_zero_rows_like(ctx, embedding, pad_rows);
    ggml_tensor * padded = ggml_concat(ctx, zero_pad, embedding, 1);
    padded = ggml_concat(ctx, padded, zero_pad, 1);
    TTS_ASSERT(padded->ne[1] == target_rows);
    return padded;
}

static ggml_tensor * view_relative_position_ids_for_query(
    ggml_context * ctx,
    ggml_tensor * rel_pos_ids,
    uint32_t tokens,
    uint32_t query_index) {
    TTS_ASSERT(rel_pos_ids);
    TTS_ASSERT(query_index < tokens);
    return ggml_view_1d(ctx,
                        rel_pos_ids,
                        tokens,
                        (size_t) query_index * rel_pos_ids->nb[1]);
}

static ggml_tensor * view_attention_query_column(
    ggml_context * ctx,
    ggml_tensor * head_channels_time,
    uint32_t query_index);

static ggml_tensor * build_relative_key_score_column(
    ggml_context * ctx,
    ggml_tensor * padded_embedding,
    ggml_tensor * rel_pos_ids,
    ggml_tensor * q_column,
    uint32_t tokens,
    uint32_t query_index) {
    ggml_tensor * relative_logits = ggml_mul_mat(ctx, padded_embedding, q_column);
    relative_logits = ggml_cont(ctx, ggml_transpose(ctx, relative_logits));
    ggml_tensor * ids = view_relative_position_ids_for_query(ctx, rel_pos_ids, tokens, query_index);
    ggml_tensor * column = ggml_get_rows(ctx, relative_logits, ids);
    return ggml_cont(ctx, ggml_transpose(ctx, column));
}

static ggml_tensor * build_relative_value_column(
    ggml_context * ctx,
    ggml_tensor * padded_embedding,
    ggml_tensor * rel_pos_ids,
    ggml_tensor * probs_column,
    uint32_t tokens,
    uint32_t query_index) {
    ggml_tensor * ids = view_relative_position_ids_for_query(ctx, rel_pos_ids, tokens, query_index);
    ggml_tensor * rel_v = ggml_get_rows(ctx, padded_embedding, ids);
    rel_v = ggml_cont(ctx, ggml_transpose(ctx, rel_v));
    return ggml_mul_mat(ctx, probs_column, rel_v);
}

static ggml_tensor * build_relative_key_scores_full(
    ggml_context * ctx,
    ggml_tensor * embedding,
    ggml_tensor * q_head,
    uint32_t tokens,
    uint32_t window_size) {
    TTS_ASSERT(embedding);
    TTS_ASSERT(q_head);
    ggml_tensor * relative_embedding = build_relative_embedding_for_length(ctx, embedding, tokens, window_size);
    ggml_tensor * rel_logits = ggml_mul_mat(ctx, relative_embedding, q_head);
    rel_logits = ggml_pad(ctx, rel_logits, 1, 0, 0, 0);
    rel_logits = ggml_reshape_1d(ctx, rel_logits, (int64_t) tokens * 2 * tokens);
    if (tokens > 1) {
        rel_logits = ggml_pad(ctx, rel_logits, tokens - 1, 0, 0, 0);
    }
    ggml_tensor * skewed = ggml_reshape_2d(ctx, rel_logits, 2 * tokens - 1, tokens + 1);
    ggml_tensor * scores = ggml_view_2d(ctx,
                                        skewed,
                                        tokens,
                                        tokens,
                                        skewed->nb[1],
                                        (size_t) (tokens - 1) * skewed->nb[0]);
    return ggml_cont(ctx, scores);
}

static ggml_tensor * build_relative_attention_weights_full(
    ggml_context * ctx,
    ggml_tensor * probs,
    uint32_t tokens) {
    TTS_ASSERT(probs);
    ggml_tensor * padded = probs;
    if (tokens > 1) {
        padded = ggml_pad(ctx, probs, tokens - 1, 0, 0, 0);
    }
    ggml_tensor * flat = ggml_reshape_1d(ctx, padded, (int64_t) tokens * (2 * tokens - 1));
    ggml_tensor * zero_prefix = build_zero_1d_like(ctx, flat, tokens);
    flat = ggml_concat(ctx, zero_prefix, flat, 0);
    ggml_tensor * skewed = ggml_reshape_2d(ctx, flat, 2 * tokens, tokens);
    ggml_tensor * weights = ggml_view_2d(ctx,
                                         skewed,
                                         2 * tokens - 1,
                                         tokens,
                                         skewed->nb[1],
                                         skewed->nb[0]);
    return ggml_cont(ctx, weights);
}

static ggml_tensor * view_attention_head_channels(
    ggml_context * ctx,
    ggml_tensor * channels_time,
    uint32_t head_index,
    uint32_t head_size) {
    TTS_ASSERT(channels_time);
    return ggml_view_2d(ctx,
                        channels_time,
                        head_size,
                        channels_time->ne[1],
                        channels_time->nb[1],
                        (size_t) head_index * head_size * channels_time->nb[0]);
}

static ggml_tensor * view_attention_value_head(
    ggml_context * ctx,
    ggml_tensor * time_channels,
    uint32_t head_index,
    uint32_t head_size) {
    TTS_ASSERT(time_channels);
    return ggml_view_2d(ctx,
                        time_channels,
                        time_channels->ne[0],
                        head_size,
                        time_channels->nb[1],
                        (size_t) head_index * head_size * time_channels->nb[1]);
}

static ggml_tensor * view_attention_query_column(
    ggml_context * ctx,
    ggml_tensor * head_channels_time,
    uint32_t query_index) {
    TTS_ASSERT(head_channels_time);
    return ggml_view_2d(ctx,
                        head_channels_time,
                        head_channels_time->ne[0],
                        1,
                        head_channels_time->nb[1],
                        (size_t) query_index * head_channels_time->nb[1]);
}

static ggml_tensor * append_attention_column(
    ggml_context * ctx,
    ggml_tensor * current,
    ggml_tensor * column) {
    return current ? ggml_concat(ctx, current, column, 1) : column;
}

static ggml_tensor * build_text_encoder_attention_legacy(
    ggml_context * ctx,
    const style_bert_vits2_encoder & encoder,
    const style_bert_vits2_attention_layer & attn,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * rel_pos_ids) {
    TTS_ASSERT(input);
    TTS_ASSERT(mask);
    TTS_ASSERT(rel_pos_ids);
    TTS_ASSERT(attn.conv_q.weight);
    TTS_ASSERT(attn.conv_k.weight);
    TTS_ASSERT(attn.conv_v.weight);
    TTS_ASSERT(attn.conv_o.weight);
    TTS_ASSERT(attn.emb_rel_k);
    TTS_ASSERT(attn.emb_rel_v);

    const uint32_t tokens = (uint32_t) input->ne[0];
    const uint32_t head_count = encoder.n_heads;
    const uint32_t head_size = (uint32_t) input->ne[1] / head_count;
    const float inv_sqrt_head = 1.0f / std::sqrt((float) head_size);

    ggml_tensor * q_time_channels = conv1d(ctx, attn.conv_q, input);
    ggml_tensor * k_time_channels = conv1d(ctx, attn.conv_k, input);
    ggml_tensor * v_time_channels = conv1d(ctx, attn.conv_v, input);
    ggml_set_name(q_time_channels, "style_bert_vits2.text_encoder.attn.q");
    ggml_set_name(k_time_channels, "style_bert_vits2.text_encoder.attn.k");
    ggml_set_name(v_time_channels, "style_bert_vits2.text_encoder.attn.v");

    ggml_tensor * q_channels_time = ggml_cont(ctx, ggml_transpose(ctx, q_time_channels));
    ggml_tensor * k_channels_time = ggml_cont(ctx, ggml_transpose(ctx, k_time_channels));

    ggml_tensor * merged_heads = nullptr;
    for (uint32_t head = 0; head < head_count; ++head) {
        ggml_tensor * q_head = view_attention_head_channels(ctx, q_channels_time, head, head_size);
        ggml_tensor * k_head = view_attention_head_channels(ctx, k_channels_time, head, head_size);
        ggml_tensor * v_head = view_attention_value_head(ctx, v_time_channels, head, head_size);

        ggml_tensor * rel_k_embedding = build_relative_embedding_with_zero_row(ctx, attn.emb_rel_k);
        ggml_tensor * rel_v_embedding = build_relative_embedding_with_zero_row(ctx, attn.emb_rel_v);
        ggml_tensor * head_output = nullptr;
        for (uint32_t query_index = 0; query_index < tokens; ++query_index) {
            ggml_tensor * q_column = view_attention_query_column(ctx, q_head, query_index);
            ggml_tensor * scores = ggml_mul_mat(ctx, k_head, q_column);
            scores = ggml_scale(ctx, scores, inv_sqrt_head);

            ggml_tensor * rel_scores = build_relative_key_score_column(ctx,
                                                                       rel_k_embedding,
                                                                       rel_pos_ids,
                                                                       q_column,
                                                                       tokens,
                                                                       query_index);
            rel_scores = ggml_scale(ctx, rel_scores, inv_sqrt_head);
            scores = ggml_add(ctx, scores, rel_scores);

            ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);
            ggml_tensor * value = ggml_mul_mat(ctx, probs, v_head);
            ggml_tensor * rel_value = build_relative_value_column(ctx,
                                                                  rel_v_embedding,
                                                                  rel_pos_ids,
                                                                  probs,
                                                                  tokens,
                                                                  query_index);
            value = ggml_add(ctx, value, rel_value);
            value = ggml_cont(ctx, ggml_transpose(ctx, value));
            head_output = head_output ? ggml_concat(ctx, head_output, value, 1) : value;
        }
        TTS_ASSERT(head_output);
        merged_heads = merged_heads ? ggml_concat(ctx, merged_heads, head_output, 0) : head_output;
    }

    TTS_ASSERT(merged_heads);
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, merged_heads));
    cur = ggml_reshape_3d(ctx, cur, tokens, input->ne[1], 1);
    cur = conv1d(ctx, attn.conv_o, cur);
    ggml_set_name(cur, "style_bert_vits2.text_encoder.attn");
    return cur;
}

static ggml_tensor * build_text_encoder_attention_scores_full_values_legacy(
    ggml_context * ctx,
    const style_bert_vits2_encoder & encoder,
    const style_bert_vits2_attention_layer & attn,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * rel_pos_ids) {
    GGML_UNUSED(mask);
    TTS_ASSERT(input);
    TTS_ASSERT(rel_pos_ids);

    const uint32_t tokens = (uint32_t) input->ne[0];
    const uint32_t head_count = encoder.n_heads;
    const uint32_t head_size = (uint32_t) input->ne[1] / head_count;
    const float inv_sqrt_head = 1.0f / std::sqrt((float) head_size);

    ggml_tensor * q_time_channels = conv1d(ctx, attn.conv_q, input);
    ggml_tensor * k_time_channels = conv1d(ctx, attn.conv_k, input);
    ggml_tensor * v_time_channels = conv1d(ctx, attn.conv_v, input);
    ggml_set_name(q_time_channels, "style_bert_vits2.text_encoder.attn.q");
    ggml_set_name(k_time_channels, "style_bert_vits2.text_encoder.attn.k");
    ggml_set_name(v_time_channels, "style_bert_vits2.text_encoder.attn.v");

    ggml_tensor * q_channels_time = ggml_cont(ctx, ggml_transpose(ctx, q_time_channels));
    ggml_tensor * k_channels_time = ggml_cont(ctx, ggml_transpose(ctx, k_time_channels));

    ggml_tensor * merged_heads = nullptr;
    for (uint32_t head = 0; head < head_count; ++head) {
        ggml_tensor * q_head = view_attention_head_channels(ctx, q_channels_time, head, head_size);
        ggml_tensor * k_head = view_attention_head_channels(ctx, k_channels_time, head, head_size);
        ggml_tensor * v_head = view_attention_value_head(ctx, v_time_channels, head, head_size);

        ggml_tensor * scores = ggml_mul_mat(ctx, k_head, q_head);
        scores = ggml_scale(ctx, scores, inv_sqrt_head);
        ggml_tensor * rel_scores = build_relative_key_scores_full(ctx,
                                                                  attn.emb_rel_k,
                                                                  q_head,
                                                                  tokens,
                                                                  encoder.window_size);
        rel_scores = ggml_scale(ctx, rel_scores, inv_sqrt_head);
        scores = ggml_add(ctx, scores, rel_scores);
        ggml_tensor * probs_full = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);

        ggml_tensor * rel_v_embedding = build_relative_embedding_with_zero_row(ctx, attn.emb_rel_v);
        ggml_tensor * head_output = nullptr;
        for (uint32_t query_index = 0; query_index < tokens; ++query_index) {
            ggml_tensor * probs = view_attention_query_column(ctx, probs_full, query_index);
            ggml_tensor * value = ggml_mul_mat(ctx, probs, v_head);
            ggml_tensor * rel_value = build_relative_value_column(ctx,
                                                                  rel_v_embedding,
                                                                  rel_pos_ids,
                                                                  probs,
                                                                  tokens,
                                                                  query_index);
            value = ggml_add(ctx, value, rel_value);
            value = ggml_cont(ctx, ggml_transpose(ctx, value));
            head_output = append_attention_column(ctx, head_output, value);
        }
        merged_heads = merged_heads ? ggml_concat(ctx, merged_heads, head_output, 0) : head_output;
    }

    TTS_ASSERT(merged_heads);
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, merged_heads));
    cur = ggml_reshape_3d(ctx, cur, tokens, input->ne[1], 1);
    cur = conv1d(ctx, attn.conv_o, cur);
    ggml_set_name(cur, "style_bert_vits2.text_encoder.attn");
    return cur;
}

static ggml_tensor * build_text_encoder_attention_scores_legacy_values_full(
    ggml_context * ctx,
    const style_bert_vits2_encoder & encoder,
    const style_bert_vits2_attention_layer & attn,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * rel_pos_ids) {
    GGML_UNUSED(mask);
    TTS_ASSERT(input);
    TTS_ASSERT(rel_pos_ids);

    const uint32_t tokens = (uint32_t) input->ne[0];
    const uint32_t head_count = encoder.n_heads;
    const uint32_t head_size = (uint32_t) input->ne[1] / head_count;
    const float inv_sqrt_head = 1.0f / std::sqrt((float) head_size);

    ggml_tensor * q_time_channels = conv1d(ctx, attn.conv_q, input);
    ggml_tensor * k_time_channels = conv1d(ctx, attn.conv_k, input);
    ggml_tensor * v_time_channels = conv1d(ctx, attn.conv_v, input);
    ggml_set_name(q_time_channels, "style_bert_vits2.text_encoder.attn.q");
    ggml_set_name(k_time_channels, "style_bert_vits2.text_encoder.attn.k");
    ggml_set_name(v_time_channels, "style_bert_vits2.text_encoder.attn.v");

    ggml_tensor * q_channels_time = ggml_cont(ctx, ggml_transpose(ctx, q_time_channels));
    ggml_tensor * k_channels_time = ggml_cont(ctx, ggml_transpose(ctx, k_time_channels));

    ggml_tensor * merged_heads = nullptr;
    for (uint32_t head = 0; head < head_count; ++head) {
        ggml_tensor * q_head = view_attention_head_channels(ctx, q_channels_time, head, head_size);
        ggml_tensor * k_head = view_attention_head_channels(ctx, k_channels_time, head, head_size);
        ggml_tensor * v_head = view_attention_value_head(ctx, v_time_channels, head, head_size);
        ggml_tensor * rel_k_embedding = build_relative_embedding_with_zero_row(ctx, attn.emb_rel_k);

        ggml_tensor * probs_full = nullptr;
        for (uint32_t query_index = 0; query_index < tokens; ++query_index) {
            ggml_tensor * q_column = view_attention_query_column(ctx, q_head, query_index);
            ggml_tensor * scores = ggml_mul_mat(ctx, k_head, q_column);
            scores = ggml_scale(ctx, scores, inv_sqrt_head);

            ggml_tensor * rel_scores = build_relative_key_score_column(ctx,
                                                                       rel_k_embedding,
                                                                       rel_pos_ids,
                                                                       q_column,
                                                                       tokens,
                                                                       query_index);
            rel_scores = ggml_scale(ctx, rel_scores, inv_sqrt_head);
            scores = ggml_add(ctx, scores, rel_scores);

            ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);
            probs_full = append_attention_column(ctx, probs_full, probs);
        }
        TTS_ASSERT(probs_full);

        ggml_tensor * value = ggml_mul_mat(ctx, probs_full, v_head);
        ggml_tensor * rel_weights = build_relative_attention_weights_full(ctx, probs_full, tokens);
        ggml_tensor * rel_v_embedding = build_relative_embedding_for_length(ctx,
                                                                           attn.emb_rel_v,
                                                                           tokens,
                                                                           encoder.window_size);
        ggml_tensor * rel_v_positions_channels = ggml_cont(ctx, ggml_transpose(ctx, rel_v_embedding));
        ggml_tensor * rel_value = ggml_mul_mat(ctx, rel_weights, rel_v_positions_channels);
        value = ggml_add(ctx, value, rel_value);
        value = ggml_cont(ctx, ggml_transpose(ctx, value));
        merged_heads = merged_heads ? ggml_concat(ctx, merged_heads, value, 0) : value;
    }

    TTS_ASSERT(merged_heads);
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, merged_heads));
    cur = ggml_reshape_3d(ctx, cur, tokens, input->ne[1], 1);
    cur = conv1d(ctx, attn.conv_o, cur);
    ggml_set_name(cur, "style_bert_vits2.text_encoder.attn");
    return cur;
}

static ggml_tensor * build_text_encoder_attention_full(
    ggml_context * ctx,
    const style_bert_vits2_encoder & encoder,
    const style_bert_vits2_attention_layer & attn,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * rel_pos_ids) {
    GGML_UNUSED(mask);
    GGML_UNUSED(rel_pos_ids);
    TTS_ASSERT(input);
    TTS_ASSERT(attn.conv_q.weight);
    TTS_ASSERT(attn.conv_k.weight);
    TTS_ASSERT(attn.conv_v.weight);
    TTS_ASSERT(attn.conv_o.weight);
    TTS_ASSERT(attn.emb_rel_k);
    TTS_ASSERT(attn.emb_rel_v);

    const uint32_t tokens = (uint32_t) input->ne[0];
    const uint32_t head_count = encoder.n_heads;
    const uint32_t head_size = (uint32_t) input->ne[1] / head_count;
    const float inv_sqrt_head = 1.0f / std::sqrt((float) head_size);

    ggml_tensor * q_time_channels = conv1d(ctx, attn.conv_q, input);
    ggml_tensor * k_time_channels = conv1d(ctx, attn.conv_k, input);
    ggml_tensor * v_time_channels = conv1d(ctx, attn.conv_v, input);
    ggml_set_name(q_time_channels, "style_bert_vits2.text_encoder.attn.q");
    ggml_set_name(k_time_channels, "style_bert_vits2.text_encoder.attn.k");
    ggml_set_name(v_time_channels, "style_bert_vits2.text_encoder.attn.v");

    ggml_tensor * q_channels_time = ggml_cont(ctx, ggml_transpose(ctx, q_time_channels));
    ggml_tensor * k_channels_time = ggml_cont(ctx, ggml_transpose(ctx, k_time_channels));

    ggml_tensor * merged_heads = nullptr;
    for (uint32_t head = 0; head < head_count; ++head) {
        ggml_tensor * q_head = view_attention_head_channels(ctx, q_channels_time, head, head_size);
        ggml_tensor * k_head = view_attention_head_channels(ctx, k_channels_time, head, head_size);
        ggml_tensor * v_head = view_attention_value_head(ctx, v_time_channels, head, head_size);

        ggml_tensor * scores = ggml_mul_mat(ctx, k_head, q_head);
        scores = ggml_scale(ctx, scores, inv_sqrt_head);
        ggml_tensor * rel_scores = build_relative_key_scores_full(ctx,
                                                                  attn.emb_rel_k,
                                                                  q_head,
                                                                  tokens,
                                                                  encoder.window_size);
        rel_scores = ggml_scale(ctx, rel_scores, inv_sqrt_head);
        scores = ggml_add(ctx, scores, rel_scores);

        ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);
        ggml_tensor * value = ggml_mul_mat(ctx, probs, v_head);

        ggml_tensor * rel_weights = build_relative_attention_weights_full(ctx, probs, tokens);
        ggml_tensor * rel_v_embedding = build_relative_embedding_for_length(ctx,
                                                                           attn.emb_rel_v,
                                                                           tokens,
                                                                           encoder.window_size);
        ggml_tensor * rel_v_positions_channels = ggml_cont(ctx, ggml_transpose(ctx, rel_v_embedding));
        ggml_tensor * rel_value = ggml_mul_mat(ctx, rel_weights, rel_v_positions_channels);
        value = ggml_add(ctx, value, rel_value);
        value = ggml_cont(ctx, ggml_transpose(ctx, value));
        merged_heads = merged_heads ? ggml_concat(ctx, merged_heads, value, 0) : value;
    }

    TTS_ASSERT(merged_heads);
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, merged_heads));
    cur = ggml_reshape_3d(ctx, cur, tokens, input->ne[1], 1);
    cur = conv1d(ctx, attn.conv_o, cur);
    ggml_set_name(cur, "style_bert_vits2.text_encoder.attn");
    return cur;
}

static ggml_tensor * build_text_encoder_attention(
    ggml_context * ctx,
    const style_bert_vits2_encoder & encoder,
    const style_bert_vits2_attention_layer & attn,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * rel_pos_ids) {
    const std::string_view mode = attention_mode();
    if (attention_legacy_enabled() || mode == "legacy") {
        return build_text_encoder_attention_legacy(ctx, encoder, attn, input, mask, rel_pos_ids);
    }
    if (mode == "full") {
        return build_text_encoder_attention_full(ctx, encoder, attn, input, mask, rel_pos_ids);
    }
    if (mode == "scores") {
        return build_text_encoder_attention_scores_full_values_legacy(ctx, encoder, attn, input, mask, rel_pos_ids);
    }
    return build_text_encoder_attention_scores_legacy_values_full(ctx, encoder, attn, input, mask, rel_pos_ids);
}

static ggml_tensor * new_relative_position_ids_input(
    ggml_context * ctx,
    style_bert_vits2_context * sctx,
    uint32_t tokens,
    const char * name) {
    sctx->encoder_rel_pos_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tokens, tokens);
    ggml_set_input(sctx->encoder_rel_pos_ids);
    sctx->set_tensor_backend(sctx->encoder_rel_pos_ids);
    ggml_set_name(sctx->encoder_rel_pos_ids, name);
    return sctx->encoder_rel_pos_ids;
}

static ggml_tensor * build_encoder_speaker_conditioning(
    ggml_context * ctx,
    const style_bert_vits2_encoder & encoder,
    ggml_tensor * input,
    ggml_tensor * g) {
    TTS_ASSERT(encoder.spk_emb_linear_weight);
    TTS_ASSERT(encoder.spk_emb_linear_bias);
    TTS_ASSERT(input);
    TTS_ASSERT(g);
    ggml_tensor * cond = ggml_mul_mat(ctx, encoder.spk_emb_linear_weight, g);
    cond = ggml_add(ctx, cond, encoder.spk_emb_linear_bias);
    cond = ggml_reshape_3d(ctx, cond, 1, input->ne[1], 1);
    cond = ggml_repeat(ctx, cond, input);
    ggml_set_name(cond, "style_bert_vits2.text_encoder.spk_emb_linear");
    return cond;
}

static ggml_tensor * build_text_encoder_layer(
    ggml_context * ctx,
    const style_bert_vits2_encoder & encoder,
    uint32_t layer_index,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * g,
    ggml_tensor * rel_pos_ids) {
    TTS_ASSERT(input);
    TTS_ASSERT(mask);
    TTS_ASSERT(layer_index < encoder.n_layers);
    TTS_ASSERT(encoder.norm_layers_1_gamma.at(layer_index));
    TTS_ASSERT(encoder.norm_layers_1_beta.at(layer_index));
    TTS_ASSERT(encoder.norm_layers_2_gamma.at(layer_index));
    TTS_ASSERT(encoder.norm_layers_2_beta.at(layer_index));

    ggml_tensor * cur = input;
    if (layer_index == encoder.cond_layer_idx && g) {
        cur = ggml_add(ctx, cur, build_encoder_speaker_conditioning(ctx, encoder, cur, g));
        cur = apply_time_mask(ctx, cur, mask);
    }

    ggml_tensor * attn = build_text_encoder_attention(ctx,
                                                      encoder,
                                                      encoder.attn_layers.at(layer_index),
                                                      cur,
                                                      mask,
                                                      rel_pos_ids);
    cur = ggml_add(ctx, cur, attn);
    cur = style_layer_norm(ctx,
                           cur,
                           encoder.norm_layers_1_gamma.at(layer_index),
                           encoder.norm_layers_1_beta.at(layer_index));
    ggml_set_name(cur, "style_bert_vits2.text_encoder.layer.norm1");

    ggml_tensor * ffn = build_text_encoder_ffn(ctx, encoder.ffn_layers.at(layer_index), cur, mask);
    cur = ggml_add(ctx, cur, ffn);
    cur = style_layer_norm(ctx,
                           cur,
                           encoder.norm_layers_2_gamma.at(layer_index),
                           encoder.norm_layers_2_beta.at(layer_index));
    ggml_set_name(cur, "style_bert_vits2.text_encoder.layer.norm2");
    return cur;
}

static ggml_tensor * build_text_encoder_stack(
    ggml_context * ctx,
    const style_bert_vits2_encoder & encoder,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * g,
    ggml_tensor * rel_pos_ids) {
    ggml_tensor * cur = apply_time_mask(ctx, input, mask);
    for (uint32_t layer_index = 0; layer_index < encoder.n_layers; ++layer_index) {
        cur = build_text_encoder_layer(ctx, encoder, layer_index, cur, mask, g, rel_pos_ids);
    }
    cur = apply_time_mask(ctx, cur, mask);
    ggml_set_name(cur, "style_bert_vits2.text_encoder.encoder");
    return cur;
}

static ggml_tensor * view_channels(
    ggml_context * ctx,
    ggml_tensor * input,
    uint32_t start_channel,
    uint32_t channels) {
    TTS_ASSERT(input);
    TTS_ASSERT((int64_t) start_channel + (int64_t) channels <= input->ne[1]);
    return ggml_view_3d(ctx,
                        input,
                        input->ne[0],
                        channels,
                        input->ne[2],
                        input->nb[1],
                        input->nb[2],
                        (size_t) start_channel * input->nb[1]);
}

static ggml_tensor * flip_channels(ggml_context * ctx, ggml_tensor * input) {
    TTS_ASSERT(input);
    ggml_tensor * output = nullptr;
    for (int64_t channel = input->ne[1] - 1; channel >= 0; --channel) {
        ggml_tensor * cur = view_channels(ctx, input, (uint32_t) channel, 1);
        output = output ? ggml_concat(ctx, output, cur, 1) : cur;
    }
    TTS_ASSERT(output);
    ggml_set_name(output, "style_bert_vits2.flow.flip");
    return output;
}

static ggml_tensor * build_flow_layer_reverse(
    ggml_context * ctx,
    const style_bert_vits2_flow_layer & layer,
    ggml_tensor * input,
    ggml_tensor * mask,
    ggml_tensor * g,
    ggml_tensor * rel_pos_ids) {
    TTS_ASSERT(input);
    TTS_ASSERT(mask);
    TTS_ASSERT(g);
    TTS_ASSERT(layer.pre.weight);
    TTS_ASSERT(layer.post.weight);

    const uint32_t half_channels = (uint32_t) input->ne[1] / 2;
    ggml_tensor * x0 = view_channels(ctx, input, 0, half_channels);
    ggml_tensor * x1 = view_channels(ctx, input, half_channels, half_channels);
    ggml_tensor * h = conv1d(ctx, layer.pre, x0);
    h = apply_time_mask(ctx, h, mask);
    h = build_text_encoder_stack(ctx, layer.encoder, h, mask, g, rel_pos_ids);
    ggml_tensor * m = conv1d(ctx, layer.post, h);
    m = apply_time_mask(ctx, m, mask);
    x1 = apply_time_mask(ctx, ggml_sub(ctx, x1, m), mask);
    ggml_tensor * cur = ggml_concat(ctx, x0, x1, 1);
    ggml_set_name(cur, "style_bert_vits2.flow.layer.reverse");
    return cur;
}

static ggml_tensor * build_prior_sample(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx) {
    TTS_ASSERT(sctx->prior_m_p);
    TTS_ASSERT(sctx->prior_logs_p);
    TTS_ASSERT(sctx->prior_noise);
    TTS_ASSERT(sctx->prior_noise_scale);

    ggml_tensor * scale = ggml_repeat(ctx, sctx->prior_noise_scale, sctx->prior_m_p);
    ggml_tensor * cur = ggml_mul(ctx, sctx->prior_noise, ggml_exp(ctx, sctx->prior_logs_p));
    cur = ggml_mul(ctx, cur, scale);
    cur = ggml_add(ctx, sctx->prior_m_p, cur);
    GGML_UNUSED(model);
    ggml_set_name(cur, "style_bert_vits2.z_p");
    return cur;
}

static ggml_tensor * crop_time_padding(ggml_context * ctx, ggml_tensor * input, uint32_t padding) {
    if (padding == 0) {
        return input;
    }
    const int64_t crop = (int64_t) padding * 2;
    if (input->ne[0] <= crop) {
        TTS_ABORT("Style-Bert decoder conv-transpose crop exceeds output length: shape=%s padding=%u\n",
                  style_shape(input).c_str(), padding);
    }
    ggml_tensor * view = ggml_view_3d(ctx,
                                      input,
                                      input->ne[0] - crop,
                                      input->ne[1],
                                      input->ne[2],
                                      input->nb[1],
                                      input->nb[2],
                                      (size_t) padding * input->nb[0]);
    return ggml_cont(ctx, view);
}

static ggml_tensor * conv_transpose1d(ggml_context * ctx, const style_bert_vits2_upsample & up, ggml_tensor * input) {
    TTS_ASSERT(up.weight);
    TTS_ASSERT(input);
    ggml_tensor * cur = tts_conv_transpose_1d(ctx, up.weight, input, (int) up.stride, 0, 1, 0, 1);
    cur = crop_time_padding(ctx, cur, up.padding);
    if (up.bias) {
        cur = ggml_add(ctx, cur, up.bias);
    }
    return cur;
}

static ggml_tensor * build_resblock(ggml_context * ctx, ggml_tensor * input, const style_bert_vits2_resblock & block) {
    ggml_tensor * x = input;
    for (size_t i = 0; i < block.convs1.size(); ++i) {
        ggml_tensor * cur = ggml_leaky_relu(ctx, x, 0.1f, false);
        cur = conv1d(ctx, block.convs1[i], cur);
        cur = ggml_leaky_relu(ctx, cur, 0.1f, false);
        cur = conv1d(ctx, block.convs2[i], cur);
        x = ggml_add(ctx, cur, x);
    }
    return x;
}

static ggml_tensor * build_decoder(
    ggml_context * ctx,
    style_bert_vits2_model * model,
    style_bert_vits2_context * sctx,
    const char * debug_stop_node,
    bool * debug_stop_found) {
    ggml_tensor * cur = conv1d(ctx, model->decoder.conv_pre, sctx->decoder_z);
    ggml_set_name(cur, "style_bert_vits2.conv_pre");
    if (debug_node_matches(cur, debug_stop_node)) {
        *debug_stop_found = true;
        return cur;
    }
    ggml_tensor * cond = conv1d(ctx, model->decoder.cond, sctx->decoder_g);
    ggml_set_name(cond, "style_bert_vits2.cond");
    if (debug_node_matches(cond, debug_stop_node)) {
        *debug_stop_found = true;
        return cond;
    }
    cur = ggml_add(ctx, cur, cond);
    ggml_set_name(cur, "style_bert_vits2.pre_upsample");
    if (debug_node_matches(cur, debug_stop_node)) {
        *debug_stop_found = true;
        return cur;
    }

    for (uint32_t i = 0; i < model->num_upsamples; ++i) {
        cur = ggml_leaky_relu(ctx, cur, 0.1f, false);
        ggml_format_name(cur, "style_bert_vits2.up.%u.pre_relu", i);
        if (debug_node_matches(cur, debug_stop_node)) {
            *debug_stop_found = true;
            return cur;
        }
        cur = conv_transpose1d(ctx, model->decoder.ups[i], cur);
        ggml_format_name(cur, "style_bert_vits2.up.%u.conv_transpose", i);
        if (debug_node_matches(cur, debug_stop_node)) {
            *debug_stop_found = true;
            return cur;
        }

        ggml_tensor * sum = nullptr;
        for (uint32_t j = 0; j < model->num_kernels; ++j) {
            ggml_tensor * res = build_resblock(ctx, cur, model->decoder.resblocks[(size_t) i * model->num_kernels + j]);
            ggml_format_name(res, "style_bert_vits2.up.%u.res.%u", i, j);
            if (debug_node_matches(res, debug_stop_node)) {
                *debug_stop_found = true;
                return res;
            }
            sum = sum ? ggml_add(ctx, sum, res) : res;
        }
        cur = ggml_scale(ctx, sum, 1.0f / (float) model->num_kernels);
        ggml_format_name(cur, "style_bert_vits2.up.%u.average", i);
        if (debug_node_matches(cur, debug_stop_node)) {
            *debug_stop_found = true;
            return cur;
        }
    }

    cur = ggml_leaky_relu(ctx, cur, 0.01f, false);
    ggml_set_name(cur, "style_bert_vits2.final_relu");
    if (debug_node_matches(cur, debug_stop_node)) {
        *debug_stop_found = true;
        return cur;
    }
    cur = conv1d(ctx, model->decoder.conv_post, cur);
    ggml_set_name(cur, "style_bert_vits2.conv_post");
    if (debug_node_matches(cur, debug_stop_node)) {
        *debug_stop_found = true;
        return cur;
    }
    cur = ggml_tanh(ctx, cur);
    ggml_set_name(cur, "style_bert_vits2_decoder_output");
    if (debug_node_matches(cur, debug_stop_node)) {
        *debug_stop_found = true;
    }
    return cur;
}

static void assign_conv(style_bert_vits2_model * model, style_bert_vits2_conv1d & conv, const std::string & leaf, ggml_tensor * tensor) {
    if (leaf == "weight") {
        conv.weight = ggml_dup_tensor(model->ctx, tensor);
        model->set_tensor(conv.weight, tensor);
    } else if (leaf == "bias") {
        conv.bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
        model->set_tensor(conv.bias, tensor);
    }
}

static void assign_up(style_bert_vits2_model * model, style_bert_vits2_upsample & up, const std::string & leaf, ggml_tensor * tensor) {
    if (leaf == "weight") {
        up.weight = ggml_dup_tensor(model->ctx, tensor);
        model->set_tensor(up.weight, tensor);
    } else if (leaf == "bias") {
        up.bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
        model->set_tensor(up.bias, tensor);
    }
}

static void assign_linear(style_bert_vits2_model * model, ggml_tensor * & dst, ggml_tensor * tensor) {
    dst = ggml_dup_tensor(model->ctx, tensor);
    model->set_tensor(dst, tensor);
}

static void assign_layer_norm_tensor(style_bert_vits2_model * model, std::vector<ggml_tensor *> & tensors,
                                     size_t index, ggml_tensor * tensor) {
    if (index >= tensors.size()) {
        TTS_ABORT("Style-Bert-VITS2 encoder norm index out of range: %zu >= %zu\n", index, tensors.size());
    }
    tensors[index] = ggml_dup_tensor(model->ctx, tensor);
    model->set_tensor(tensors[index], tensor);
}

static std::string expand_compact_leaf(const std::string & leaf) {
    if (leaf == "w") {
        return "weight";
    }
    if (leaf == "b") {
        return "bias";
    }
    return leaf;
}
}

void style_bert_vits2_model::prep_constants(gguf_context * meta) {
    const std::string arch = "style-bert-vits2";
    decoder_only = read_u32(meta, arch + ".decoder_only", decoder_only);
    sample_rate = read_u32(meta, arch + ".sample_rate", sample_rate);
    inter_channels = read_u32(meta, arch + ".inter_channels", inter_channels);
    hidden_channels = read_u32(meta, arch + ".hidden_channels", hidden_channels);
    const uint32_t filter_channels = read_u32(meta, arch + ".filter_channels", 768);
    const uint32_t n_heads = read_u32(meta, arch + ".n_heads", 2);
    text_encoder.encoder.n_heads = n_heads;
    text_encoder.encoder.n_layers = read_u32(meta, arch + ".text_encoder.n_layers", 6);
    text_encoder.encoder.kernel = read_u32(meta, arch + ".text_encoder.kernel", 3);
    text_encoder.encoder.padding = text_encoder.encoder.kernel / 2;
    text_encoder.encoder.window_size = read_u32(meta, arch + ".text_encoder.window_size", 4);
    text_encoder.encoder.cond_layer_idx = read_u32(meta, arch + ".text_encoder.cond_layer_idx", 2);
    gin_channels = read_u32(meta, arch + ".gin_channels", gin_channels);
    jp_extra = read_u32(meta, arch + ".jp_extra", jp_extra);
    duration_predictor.filter_channels = read_u32(meta, arch + ".duration_predictor.filter_channels", 256);
    const uint32_t duration_kernel = read_u32(meta, arch + ".duration_predictor.kernel", 3);
    const uint32_t duration_padding = read_u32(meta, arch + ".duration_predictor.padding", duration_kernel / 2);
    duration_predictor.conv_1.kernel = duration_kernel;
    duration_predictor.conv_1.padding = duration_padding;
    duration_predictor.conv_1.dilation = 1;
    duration_predictor.conv_2.kernel = duration_kernel;
    duration_predictor.conv_2.padding = duration_padding;
    duration_predictor.conv_2.dilation = 1;
    duration_predictor.proj.kernel = 1;
    duration_predictor.proj.padding = 0;
    duration_predictor.proj.dilation = 1;
    duration_predictor.cond.kernel = 1;
    duration_predictor.cond.padding = 0;
    duration_predictor.cond.dilation = 1;
    stochastic_duration_predictor.n_flows = read_u32(meta, arch + ".sdp.n_flows", 4);
    stochastic_duration_predictor.n_layers = read_u32(meta, arch + ".sdp.n_layers", 3);
    stochastic_duration_predictor.kernel = read_u32(meta, arch + ".sdp.kernel", duration_kernel);
    stochastic_duration_predictor.num_bins = read_u32(meta, arch + ".sdp.num_bins", 10);
    flow.use_transformer = read_u32(meta, arch + ".flow.use_transformer", 1);
    flow.n_flows = read_u32(meta, arch + ".flow.n_flows", 4);
    flow.n_layers = read_u32(meta, arch + ".flow.n_layers", 6);
    flow.kernel = read_u32(meta, arch + ".flow.kernel", 5);
    flow.padding = flow.kernel / 2;
    flow.window_size = read_u32(meta, arch + ".flow.window_size", 4);
    flow.cond_layer_idx = read_u32(meta, arch + ".flow.cond_layer_idx", 2);
    GGML_UNUSED(filter_channels);
    upsample_initial_channel = read_u32(meta, arch + ".upsample_initial_channel", upsample_initial_channel);
    num_upsamples = read_u32(meta, arch + ".decoder.num_upsamples", num_upsamples);
    num_kernels = read_u32(meta, arch + ".decoder.num_kernels", num_kernels);
    resblock = read_u32(meta, arch + ".decoder.resblock", resblock);

    decoder.conv_pre.kernel = read_u32(meta, arch + ".decoder.conv_pre.kernel", 7);
    decoder.conv_pre.padding = read_u32(meta, arch + ".decoder.conv_pre.padding", 3);
    decoder.conv_pre.dilation = 1;
    decoder.cond.kernel = 1;
    decoder.cond.padding = 0;
    decoder.cond.dilation = 1;
    decoder.conv_post.kernel = read_u32(meta, arch + ".decoder.conv_post.kernel", 7);
    decoder.conv_post.padding = read_u32(meta, arch + ".decoder.conv_post.padding", 3);
    decoder.conv_post.dilation = 1;
}

void style_bert_vits2_model::prep_layers(gguf_context * meta) {
    const std::string arch = "style-bert-vits2";
    prep_encoder_layers(text_encoder.encoder,
                        text_encoder.encoder.n_layers,
                        text_encoder.encoder.kernel,
                        text_encoder.encoder.window_size,
                        text_encoder.encoder.cond_layer_idx);
    prep_stochastic_duration_predictor();
    flow.layers.resize(flow.n_flows);
    for (uint32_t i = 0; i < flow.n_flows; ++i) {
        flow.layers[i].pre.kernel = 1;
        flow.layers[i].pre.padding = 0;
        flow.layers[i].pre.dilation = 1;
        flow.layers[i].post.kernel = 1;
        flow.layers[i].post.padding = 0;
        flow.layers[i].post.dilation = 1;
        prep_encoder_layers(flow.layers[i].encoder, flow.n_layers, flow.kernel, flow.window_size, flow.cond_layer_idx);
    }

    decoder.ups.resize(num_upsamples);
    decoder.resblocks.resize((size_t) num_upsamples * num_kernels);

    for (uint32_t i = 0; i < num_upsamples; ++i) {
        const std::string base = arch + ".decoder.ups." + std::to_string(i);
        decoder.ups[i].stride = read_u32(meta, base + ".stride", 1);
        decoder.ups[i].padding = read_u32(meta, base + ".padding", 0);
        decoder.ups[i].kernel = read_u32(meta, base + ".kernel", 1);
    }

    for (size_t block_index = 0; block_index < decoder.resblocks.size(); ++block_index) {
        style_bert_vits2_resblock & block = decoder.resblocks[block_index];
        block.convs1.resize(3);
        block.convs2.resize(3);
        for (uint32_t conv_index = 0; conv_index < 3; ++conv_index) {
            const std::string base1 = arch + ".decoder.resblocks." + std::to_string(block_index) +
                ".convs1." + std::to_string(conv_index);
            block.convs1[conv_index].padding = read_u32(meta, base1 + ".padding", 1);
            block.convs1[conv_index].dilation = read_u32(meta, base1 + ".dilation", 1);
            block.convs1[conv_index].kernel = read_u32(meta, base1 + ".kernel", 3);

            const std::string base2 = arch + ".decoder.resblocks." + std::to_string(block_index) +
                ".convs2." + std::to_string(conv_index);
            block.convs2[conv_index].padding = read_u32(meta, base2 + ".padding", 1);
            block.convs2[conv_index].dilation = read_u32(meta, base2 + ".dilation", 1);
            block.convs2[conv_index].kernel = read_u32(meta, base2 + ".kernel", 3);
        }
    }
}

void style_bert_vits2_model::prep_dds_conv(style_bert_vits2_dds_conv & conv,
                                           uint32_t n_layers,
                                           uint32_t kernel) {
    conv.n_layers = n_layers;
    conv.kernel = kernel;
    conv.convs_sep.resize(n_layers);
    conv.convs_1x1.resize(n_layers);
    conv.norms_1_gamma.assign(n_layers, nullptr);
    conv.norms_1_beta.assign(n_layers, nullptr);
    conv.norms_2_gamma.assign(n_layers, nullptr);
    conv.norms_2_beta.assign(n_layers, nullptr);
    for (uint32_t i = 0; i < n_layers; ++i) {
        uint32_t dilation = 1;
        for (uint32_t power = 0; power < i; ++power) {
            dilation *= kernel;
        }
        const uint32_t padding = (kernel * dilation - dilation) / 2;
        conv.convs_sep[i].kernel = kernel;
        conv.convs_sep[i].padding = padding;
        conv.convs_sep[i].dilation = dilation;
        conv.convs_1x1[i].kernel = 1;
        conv.convs_1x1[i].padding = 0;
        conv.convs_1x1[i].dilation = 1;
    }
}

void style_bert_vits2_model::prep_stochastic_duration_predictor() {
    style_bert_vits2_stochastic_duration_predictor & sdp = stochastic_duration_predictor;
    sdp.pre.kernel = 1;
    sdp.pre.padding = 0;
    sdp.pre.dilation = 1;
    sdp.proj.kernel = 1;
    sdp.proj.padding = 0;
    sdp.proj.dilation = 1;
    sdp.cond.kernel = 1;
    sdp.cond.padding = 0;
    sdp.cond.dilation = 1;
    sdp.post_pre.kernel = 1;
    sdp.post_pre.padding = 0;
    sdp.post_pre.dilation = 1;
    sdp.post_proj.kernel = 1;
    sdp.post_proj.padding = 0;
    sdp.post_proj.dilation = 1;
    prep_dds_conv(sdp.convs, sdp.n_layers, sdp.kernel);
    prep_dds_conv(sdp.post_convs, sdp.n_layers, sdp.kernel);
    sdp.flows.resize(sdp.n_flows);
    sdp.post_flows.resize(sdp.n_flows);
    for (uint32_t i = 0; i < sdp.n_flows; ++i) {
        for (style_bert_vits2_conv_flow * flow : {&sdp.flows[i], &sdp.post_flows[i]}) {
            flow->pre.kernel = 1;
            flow->pre.padding = 0;
            flow->pre.dilation = 1;
            flow->proj.kernel = 1;
            flow->proj.padding = 0;
            flow->proj.dilation = 1;
            flow->num_bins = sdp.num_bins;
            prep_dds_conv(flow->convs, sdp.n_layers, sdp.kernel);
        }
    }
}

void style_bert_vits2_model::prep_encoder_layers(style_bert_vits2_encoder & encoder,
                                                 uint32_t n_layers,
                                                 uint32_t kernel,
                                                 uint32_t window_size,
                                                 uint32_t cond_layer_idx) {
    encoder.n_layers = n_layers;
    encoder.kernel = kernel;
    encoder.padding = kernel / 2;
    encoder.window_size = window_size;
    encoder.cond_layer_idx = cond_layer_idx;
    encoder.attn_layers.resize(n_layers);
    encoder.ffn_layers.resize(n_layers);
    encoder.norm_layers_1_gamma.assign(n_layers, nullptr);
    encoder.norm_layers_1_beta.assign(n_layers, nullptr);
    encoder.norm_layers_2_gamma.assign(n_layers, nullptr);
    encoder.norm_layers_2_beta.assign(n_layers, nullptr);
    for (uint32_t i = 0; i < n_layers; ++i) {
        style_bert_vits2_attention_layer & attn = encoder.attn_layers[i];
        attn.conv_q.kernel = attn.conv_k.kernel = attn.conv_v.kernel = attn.conv_o.kernel = 1;
        attn.conv_q.padding = attn.conv_k.padding = attn.conv_v.padding = attn.conv_o.padding = 0;
        attn.conv_q.dilation = attn.conv_k.dilation = attn.conv_v.dilation = attn.conv_o.dilation = 1;

        style_bert_vits2_ffn_layer & ffn = encoder.ffn_layers[i];
        ffn.conv_1.kernel = kernel;
        ffn.conv_1.padding = kernel / 2;
        ffn.conv_1.dilation = 1;
        ffn.conv_2.kernel = kernel;
        ffn.conv_2.padding = kernel / 2;
        ffn.conv_2.dilation = 1;
    }
}

void style_bert_vits2_model::assign_encoder_weight(style_bert_vits2_encoder & encoder,
                                                   std::string name,
                                                   ggml_tensor * tensor) {
    const std::vector<std::string> parts = split(name, ".");
    if (parts.size() == 4 && parts[0] == "al") {
        const size_t index = (size_t) std::stoul(parts[1]);
        style_bert_vits2_attention_layer & layer = encoder.attn_layers.at(index);
        const std::string leaf = expand_compact_leaf(parts[3]);
        if (parts[2] == "q") {
            assign_conv(this, layer.conv_q, leaf, tensor);
            return;
        }
        if (parts[2] == "k") {
            assign_conv(this, layer.conv_k, leaf, tensor);
            return;
        }
        if (parts[2] == "v") {
            assign_conv(this, layer.conv_v, leaf, tensor);
            return;
        }
        if (parts[2] == "o") {
            assign_conv(this, layer.conv_o, leaf, tensor);
            return;
        }
    }
    if (parts.size() == 3 && parts[0] == "al") {
        const size_t index = (size_t) std::stoul(parts[1]);
        style_bert_vits2_attention_layer & layer = encoder.attn_layers.at(index);
        if (parts[2] == "rk") {
            assign_linear(this, layer.emb_rel_k, tensor);
            return;
        }
        if (parts[2] == "rv") {
            assign_linear(this, layer.emb_rel_v, tensor);
            return;
        }
    }
    if (parts.size() == 4 && parts[0] == "ffn") {
        const size_t index = (size_t) std::stoul(parts[1]);
        style_bert_vits2_ffn_layer & layer = encoder.ffn_layers.at(index);
        const std::string leaf = expand_compact_leaf(parts[3]);
        if (parts[2] == "c1") {
            assign_conv(this, layer.conv_1, leaf, tensor);
            return;
        }
        if (parts[2] == "c2") {
            assign_conv(this, layer.conv_2, leaf, tensor);
            return;
        }
    }
    if (parts.size() == 3 && (parts[0] == "n1" || parts[0] == "n2")) {
        const size_t index = (size_t) std::stoul(parts[1]);
        std::vector<ggml_tensor *> & gammas = parts[0] == "n1" ? encoder.norm_layers_1_gamma : encoder.norm_layers_2_gamma;
        std::vector<ggml_tensor *> & betas = parts[0] == "n1" ? encoder.norm_layers_1_beta : encoder.norm_layers_2_beta;
        if (parts[2] == "g") {
            assign_layer_norm_tensor(this, gammas, index, tensor);
            return;
        }
        if (parts[2] == "b") {
            assign_layer_norm_tensor(this, betas, index, tensor);
            return;
        }
    }
    if (parts.size() == 2 && parts[0] == "spk") {
        if (parts[1] == "w") {
            assign_linear(this, encoder.spk_emb_linear_weight, tensor);
            return;
        }
        if (parts[1] == "b") {
            assign_linear(this, encoder.spk_emb_linear_bias, tensor);
            return;
        }
    }
    if (parts.size() == 4 && parts[0] == "attn_layers") {
        const size_t index = (size_t) std::stoul(parts[1]);
        style_bert_vits2_attention_layer & layer = encoder.attn_layers.at(index);
        if (parts[2] == "conv_q") {
            assign_conv(this, layer.conv_q, parts[3], tensor);
            return;
        }
        if (parts[2] == "conv_k") {
            assign_conv(this, layer.conv_k, parts[3], tensor);
            return;
        }
        if (parts[2] == "conv_v") {
            assign_conv(this, layer.conv_v, parts[3], tensor);
            return;
        }
        if (parts[2] == "conv_o") {
            assign_conv(this, layer.conv_o, parts[3], tensor);
            return;
        }
    }
    if (parts.size() == 3 && parts[0] == "attn_layers") {
        const size_t index = (size_t) std::stoul(parts[1]);
        style_bert_vits2_attention_layer & layer = encoder.attn_layers.at(index);
        if (parts[2] == "emb_rel_k") {
            assign_linear(this, layer.emb_rel_k, tensor);
            return;
        }
        if (parts[2] == "emb_rel_v") {
            assign_linear(this, layer.emb_rel_v, tensor);
            return;
        }
    }
    if (parts.size() == 4 && parts[0] == "ffn_layers") {
        const size_t index = (size_t) std::stoul(parts[1]);
        style_bert_vits2_ffn_layer & layer = encoder.ffn_layers.at(index);
        if (parts[2] == "conv_1") {
            assign_conv(this, layer.conv_1, parts[3], tensor);
            return;
        }
        if (parts[2] == "conv_2") {
            assign_conv(this, layer.conv_2, parts[3], tensor);
            return;
        }
    }
    if (parts.size() == 3 && parts[0] == "norm_layers_1") {
        const size_t index = (size_t) std::stoul(parts[1]);
        if (parts[2] == "gamma") {
            assign_layer_norm_tensor(this, encoder.norm_layers_1_gamma, index, tensor);
            return;
        }
        if (parts[2] == "beta") {
            assign_layer_norm_tensor(this, encoder.norm_layers_1_beta, index, tensor);
            return;
        }
    }
    if (parts.size() == 3 && parts[0] == "norm_layers_2") {
        const size_t index = (size_t) std::stoul(parts[1]);
        if (parts[2] == "gamma") {
            assign_layer_norm_tensor(this, encoder.norm_layers_2_gamma, index, tensor);
            return;
        }
        if (parts[2] == "beta") {
            assign_layer_norm_tensor(this, encoder.norm_layers_2_beta, index, tensor);
            return;
        }
    }
    if (parts.size() == 2 && parts[0] == "spk_emb_linear") {
        if (parts[1] == "weight") {
            assign_linear(this, encoder.spk_emb_linear_weight, tensor);
            return;
        }
        if (parts[1] == "bias") {
            assign_linear(this, encoder.spk_emb_linear_bias, tensor);
            return;
        }
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 encoder tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_text_encoder_weight(std::string name, ggml_tensor * tensor) {
    if (name == "token_embedding.weight") {
        text_encoder.token_embedding = ggml_dup_tensor(ctx, tensor);
        set_tensor(text_encoder.token_embedding, tensor);
        return;
    }
    if (name == "tone_embedding.weight") {
        text_encoder.tone_embedding = ggml_dup_tensor(ctx, tensor);
        set_tensor(text_encoder.tone_embedding, tensor);
        return;
    }
    if (name == "language_embedding.weight") {
        text_encoder.language_embedding = ggml_dup_tensor(ctx, tensor);
        set_tensor(text_encoder.language_embedding, tensor);
        return;
    }
    if (has_prefix(name, "bert_proj.")) {
        assign_conv(this, text_encoder.bert_proj, name.substr(sizeof("bert_proj.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "ja_bert_proj.")) {
        assign_conv(this, text_encoder.ja_bert_proj, name.substr(sizeof("ja_bert_proj.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "en_bert_proj.")) {
        assign_conv(this, text_encoder.en_bert_proj, name.substr(sizeof("en_bert_proj.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "proj.")) {
        assign_conv(this, text_encoder.proj, name.substr(sizeof("proj.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "encoder.")) {
        assign_encoder_weight(text_encoder.encoder, name.substr(sizeof("encoder.") - 1), tensor);
        return;
    }
    if (name == "style_proj.weight") {
        text_encoder.style_proj_weight = ggml_dup_tensor(ctx, tensor);
        set_tensor(text_encoder.style_proj_weight, tensor);
        return;
    }
    if (name == "style_proj.bias") {
        text_encoder.style_proj_bias = ggml_dup_tensor(ctx, tensor);
        set_tensor(text_encoder.style_proj_bias, tensor);
        return;
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 text encoder tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_duration_predictor_weight(std::string name, ggml_tensor * tensor) {
    if (has_prefix(name, "conv_1.")) {
        assign_conv(this, duration_predictor.conv_1, name.substr(sizeof("conv_1.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "conv_2.")) {
        assign_conv(this, duration_predictor.conv_2, name.substr(sizeof("conv_2.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "proj.")) {
        assign_conv(this, duration_predictor.proj, name.substr(sizeof("proj.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "cond.")) {
        assign_conv(this, duration_predictor.cond, name.substr(sizeof("cond.") - 1), tensor);
        return;
    }
    if (name == "norm_1.gamma") {
        duration_predictor.norm_1_gamma = ggml_dup_tensor(ctx, tensor);
        set_tensor(duration_predictor.norm_1_gamma, tensor);
        return;
    }
    if (name == "norm_1.beta") {
        duration_predictor.norm_1_beta = ggml_dup_tensor(ctx, tensor);
        set_tensor(duration_predictor.norm_1_beta, tensor);
        return;
    }
    if (name == "norm_2.gamma") {
        duration_predictor.norm_2_gamma = ggml_dup_tensor(ctx, tensor);
        set_tensor(duration_predictor.norm_2_gamma, tensor);
        return;
    }
    if (name == "norm_2.beta") {
        duration_predictor.norm_2_beta = ggml_dup_tensor(ctx, tensor);
        set_tensor(duration_predictor.norm_2_beta, tensor);
        return;
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 duration predictor tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_dds_conv_weight(style_bert_vits2_dds_conv & conv,
                                                    std::string name,
                                                    ggml_tensor * tensor) {
    const std::vector<std::string> parts = split(name, ".");
    if (parts.size() == 3 && parts[0] == "convs_sep") {
        const size_t index = (size_t) std::stoul(parts[1]);
        assign_conv(this, conv.convs_sep.at(index), parts[2], tensor);
        return;
    }
    if (parts.size() == 3 && parts[0] == "convs_1x1") {
        const size_t index = (size_t) std::stoul(parts[1]);
        assign_conv(this, conv.convs_1x1.at(index), parts[2], tensor);
        return;
    }
    if (parts.size() == 3 && (parts[0] == "norms_1" || parts[0] == "norms_2")) {
        const size_t index = (size_t) std::stoul(parts[1]);
        std::vector<ggml_tensor *> & gammas = parts[0] == "norms_1" ? conv.norms_1_gamma : conv.norms_2_gamma;
        std::vector<ggml_tensor *> & betas = parts[0] == "norms_1" ? conv.norms_1_beta : conv.norms_2_beta;
        if (parts[2] == "gamma") {
            assign_layer_norm_tensor(this, gammas, index, tensor);
            return;
        }
        if (parts[2] == "beta") {
            assign_layer_norm_tensor(this, betas, index, tensor);
            return;
        }
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 DDSConv tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_elementwise_affine_weight(style_bert_vits2_elementwise_affine & affine,
                                                              std::string name,
                                                              ggml_tensor * tensor) {
    if (name == "m") {
        assign_linear(this, affine.m, tensor);
        return;
    }
    if (name == "logs") {
        assign_linear(this, affine.logs, tensor);
        return;
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 ElementwiseAffine tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_conv_flow_weight(style_bert_vits2_conv_flow & flow,
                                                     std::string name,
                                                     ggml_tensor * tensor) {
    if (has_prefix(name, "pre.")) {
        assign_conv(this, flow.pre, name.substr(sizeof("pre.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "proj.")) {
        assign_conv(this, flow.proj, name.substr(sizeof("proj.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "convs.")) {
        assign_dds_conv_weight(flow.convs, name.substr(sizeof("convs.") - 1), tensor);
        return;
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 ConvFlow tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_stochastic_duration_predictor_weight(std::string name, ggml_tensor * tensor) {
    style_bert_vits2_stochastic_duration_predictor & sdp = stochastic_duration_predictor;
    if (has_prefix(name, "pre.")) {
        assign_conv(this, sdp.pre, name.substr(sizeof("pre.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "proj.")) {
        assign_conv(this, sdp.proj, name.substr(sizeof("proj.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "cond.")) {
        assign_conv(this, sdp.cond, name.substr(sizeof("cond.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "convs.")) {
        assign_dds_conv_weight(sdp.convs, name.substr(sizeof("convs.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "post_pre.")) {
        assign_conv(this, sdp.post_pre, name.substr(sizeof("post_pre.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "post_proj.")) {
        assign_conv(this, sdp.post_proj, name.substr(sizeof("post_proj.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "post_convs.")) {
        assign_dds_conv_weight(sdp.post_convs, name.substr(sizeof("post_convs.") - 1), tensor);
        return;
    }
    const std::vector<std::string> parts = split(name, ".");
    if (parts.size() >= 3 && (parts[0] == "flows" || parts[0] == "post_flows")) {
        const size_t raw_index = (size_t) std::stoul(parts[1]);
        const std::string rest = name.substr(parts[0].size() + parts[1].size() + 2);
        if (raw_index == 0) {
            style_bert_vits2_elementwise_affine & affine =
                parts[0] == "flows" ? sdp.flow_affine : sdp.post_flow_affine;
            assign_elementwise_affine_weight(affine, rest, tensor);
            return;
        }
        if (raw_index % 2 == 0) {
            TTS_ABORT("Style-Bert-VITS2 SDP tensor unexpectedly targets Flip layer '%s'\n", name.c_str());
        }
        const size_t index = (raw_index - 1) / 2;
        std::vector<style_bert_vits2_conv_flow> & flows =
            parts[0] == "flows" ? sdp.flows : sdp.post_flows;
        assign_conv_flow_weight(flows.at(index), rest, tensor);
        return;
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 stochastic duration predictor tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_flow_weight(std::string name, ggml_tensor * tensor) {
    const std::vector<std::string> parts = split(name, ".");
    if (parts.size() < 4 || parts[0] != "flows") {
        TTS_ABORT("Unknown Style-Bert-VITS2 flow tensor '%s'\n", name.c_str());
    }
    const size_t raw_index = (size_t) std::stoul(parts[1]);
    if (raw_index % 2 != 0) {
        TTS_ABORT("Style-Bert-VITS2 flow tensor unexpectedly targets Flip layer '%s'\n", name.c_str());
    }
    const size_t index = raw_index / 2;
    style_bert_vits2_flow_layer & layer = flow.layers.at(index);
    const std::string rest = name.substr(parts[0].size() + parts[1].size() + 2);
    if (has_prefix(rest, "pre.")) {
        assign_conv(this, layer.pre, rest.substr(sizeof("pre.") - 1), tensor);
        return;
    }
    if (has_prefix(rest, "post.")) {
        assign_conv(this, layer.post, rest.substr(sizeof("post.") - 1), tensor);
        return;
    }
    if (has_prefix(rest, "enc.")) {
        assign_encoder_weight(layer.encoder, rest.substr(sizeof("enc.") - 1), tensor);
        return;
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 flow tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_decoder_weight(std::string name, ggml_tensor * tensor) {
    if (has_prefix(name, "conv_pre.")) {
        assign_conv(this, decoder.conv_pre, name.substr(sizeof("conv_pre.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "cond.")) {
        assign_conv(this, decoder.cond, name.substr(sizeof("cond.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "conv_post.")) {
        assign_conv(this, decoder.conv_post, name.substr(sizeof("conv_post.") - 1), tensor);
        return;
    }
    if (has_prefix(name, "ups.")) {
        const std::vector<std::string> parts = split(name, ".");
        if (parts.size() == 3) {
            const int index = std::stoi(parts[1]);
            assign_up(this, decoder.ups.at((size_t) index), parts[2], tensor);
        }
        return;
    }
    if (has_prefix(name, "resblocks.")) {
        const std::vector<std::string> parts = split(name, ".");
        if (parts.size() == 5) {
            const int block_index = std::stoi(parts[1]);
            const int conv_index = std::stoi(parts[3]);
            style_bert_vits2_resblock & block = decoder.resblocks.at((size_t) block_index);
            if (parts[2] == "convs1") {
                assign_conv(this, block.convs1.at((size_t) conv_index), parts[4], tensor);
            } else if (parts[2] == "convs2") {
                assign_conv(this, block.convs2.at((size_t) conv_index), parts[4], tensor);
            }
        }
        return;
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 decoder tensor '%s'\n", name.c_str());
}

void style_bert_vits2_model::assign_weight(const char * name, ggml_tensor & tensor) {
    const std::string_view name_sv{ name };
    if (name_sv == "speaker_embedding.weight") {
        speaker_embedding = ggml_dup_tensor(ctx, &tensor);
        set_tensor(speaker_embedding, &tensor);
        return;
    }
    if (name_sv == "style_vectors") {
        style_vectors = ggml_dup_tensor(ctx, &tensor);
        set_tensor(style_vectors, &tensor);
        return;
    }
    if (name_sv.starts_with("text_encoder.")) {
        assign_text_encoder_weight(std::string{ name_sv.substr(sizeof("text_encoder.") - 1) }, &tensor);
        return;
    }
    if (name_sv.starts_with("te.enc.")) {
        assign_encoder_weight(text_encoder.encoder, std::string{ name_sv.substr(sizeof("te.enc.") - 1) }, &tensor);
        return;
    }
    if (name_sv.starts_with("duration_predictor.")) {
        assign_duration_predictor_weight(std::string{ name_sv.substr(sizeof("duration_predictor.") - 1) }, &tensor);
        return;
    }
    if (name_sv.starts_with("sdp.")) {
        assign_stochastic_duration_predictor_weight(std::string{ name_sv.substr(sizeof("sdp.") - 1) }, &tensor);
        return;
    }
    if (name_sv.starts_with("flow.")) {
        assign_flow_weight(std::string{ name_sv.substr(sizeof("flow.") - 1) }, &tensor);
        return;
    }
    if (name_sv.starts_with("fl.")) {
        const std::vector<std::string> parts = split(std::string{ name_sv }, ".");
        if (parts.size() < 4) {
            TTS_ABORT("Unknown compact Style-Bert-VITS2 flow tensor '%s'\n", name);
        }
        const size_t index = (size_t) std::stoul(parts[1]);
        style_bert_vits2_flow_layer & layer = flow.layers.at(index);
        const std::string rest{ name_sv.substr(parts[0].size() + parts[1].size() + 2) };
        if (has_prefix(rest, "pre.")) {
            assign_conv(this, layer.pre, expand_compact_leaf(rest.substr(sizeof("pre.") - 1)), &tensor);
            return;
        }
        if (has_prefix(rest, "post.")) {
            assign_conv(this, layer.post, expand_compact_leaf(rest.substr(sizeof("post.") - 1)), &tensor);
            return;
        }
        if (has_prefix(rest, "enc.")) {
            assign_encoder_weight(layer.encoder, rest.substr(sizeof("enc.") - 1), &tensor);
            return;
        }
        TTS_ABORT("Unknown compact Style-Bert-VITS2 flow tensor '%s'\n", name);
    }
    if (name_sv.starts_with("decoder.")) {
        assign_decoder_weight(std::string{ name_sv.substr(sizeof("decoder.") - 1) }, &tensor);
        return;
    }
    TTS_ABORT("Unknown Style-Bert-VITS2 tensor '%s'\n", name);
}

void style_bert_vits2_runner::assign_weight(const char * name, ggml_tensor & tensor) {
    const std::string_view name_sv{ name };
    GGML_ASSERT(name_sv.starts_with("style_bert_vits2."));
    const std::string trimmed{ name_sv.substr(sizeof("style_bert_vits2.") - 1) };
    if (debug_load_enabled()) {
        std::cerr << "STYLE_BERT_VITS2_LOAD assign " << trimmed << " shape=" << style_shape(&tensor) << std::endl;
    }
    model->assign_weight(trimmed.c_str(), tensor);
}

void style_bert_vits2_runner::prepare_post_load() {
    if (!model->decoder_only) {
        TTS_ABORT("Only Style-Bert-VITS2 decoder-only GGUF files are currently supported.\n");
    }
}

ggml_cgraph * style_bert_vits2_runner::build_speaker_embedding_graph() {
    init_build();
    sctx->speaker_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(sctx->speaker_id);
    sctx->set_tensor_backend(sctx->speaker_id);
    ggml_set_name(sctx->speaker_id, "style_bert_vits2_speaker_id");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * output = build_speaker_embedding(ctx, &*model, sctx);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_speaker_embedding_inputs(int32_t speaker_id) {
    ggml_backend_tensor_set(sctx->speaker_id, &speaker_id, 0, sizeof(int32_t));
}

void style_bert_vits2_runner::encode_speaker(int32_t speaker_id, std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_speaker_embedding_graph();
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.speaker_embedding");
    set_speaker_embedding_inputs(speaker_id);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);
    output.assign(output_data, output_data + ggml_nelements(output_tensor));
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_style_vector_graph() {
    init_build();
    sctx->style_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(sctx->style_id);
    sctx->set_tensor_backend(sctx->style_id);
    ggml_set_name(sctx->style_id, "style_bert_vits2_style_id");

    sctx->style_mean_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(sctx->style_mean_id);
    sctx->set_tensor_backend(sctx->style_mean_id);
    ggml_set_name(sctx->style_mean_id, "style_bert_vits2_style_mean_id");

    sctx->style_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(sctx->style_weight);
    sctx->set_tensor_backend(sctx->style_weight);
    ggml_set_name(sctx->style_weight, "style_bert_vits2_style_weight");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * output = build_style_vector(ctx, &*model, sctx);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_style_vector_inputs(int32_t style_id, float style_weight) {
    const int32_t mean_id = 0;
    ggml_backend_tensor_set(sctx->style_id, &style_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(sctx->style_mean_id, &mean_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(sctx->style_weight, &style_weight, 0, sizeof(float));
}

void style_bert_vits2_runner::encode_style_vector(int32_t style_id, float style_weight, std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_style_vector_graph();
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.style_vector");
    set_style_vector_inputs(style_id, style_weight);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);
    output.assign(output_data, output_data + ggml_nelements(output_tensor));
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_text_encoder_input_graph(uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->phone_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(sctx->phone_ids);
    sctx->set_tensor_backend(sctx->phone_ids);
    ggml_set_name(sctx->phone_ids, "style_bert_vits2_phone_ids");

    sctx->tone_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(sctx->tone_ids);
    sctx->set_tensor_backend(sctx->tone_ids);
    ggml_set_name(sctx->tone_ids, "style_bert_vits2_tone_ids");

    sctx->language_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(sctx->language_ids);
    sctx->set_tensor_backend(sctx->language_ids);
    ggml_set_name(sctx->language_ids, "style_bert_vits2_language_ids");

    sctx->bert = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1024, 1);
    ggml_set_input(sctx->bert);
    sctx->set_tensor_backend(sctx->bert);
    ggml_set_name(sctx->bert, "style_bert_vits2_bert");

    sctx->style_vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_set_input(sctx->style_vec);
    sctx->set_tensor_backend(sctx->style_vec);
    ggml_set_name(sctx->style_vec, "style_bert_vits2_style_vec");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * output = build_text_encoder_input(ctx, &*model, sctx);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_text_encoder_input_inputs(const int32_t * phone_ids,
                                                            const int32_t * tone_ids,
                                                            const int32_t * language_ids,
                                                            const float * bert_t_major,
                                                            const float * style_vec,
                                                            uint32_t tokens) {
    ggml_backend_tensor_set(sctx->phone_ids, phone_ids, 0, (size_t) tokens * sizeof(int32_t));
    ggml_backend_tensor_set(sctx->tone_ids, tone_ids, 0, (size_t) tokens * sizeof(int32_t));
    ggml_backend_tensor_set(sctx->language_ids, language_ids, 0, (size_t) tokens * sizeof(int32_t));
    ggml_backend_tensor_set(sctx->bert, bert_t_major, 0, (size_t) tokens * 1024 * sizeof(float));
    ggml_backend_tensor_set(sctx->style_vec, style_vec, 0, 256 * sizeof(float));
}

void style_bert_vits2_runner::encode_text_encoder_input(const int32_t * phone_ids,
                                                        const int32_t * tone_ids,
                                                        const int32_t * language_ids,
                                                        const float * bert_t_major,
                                                        const float * style_vec,
                                                        uint32_t tokens,
                                                        std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_text_encoder_input_graph(tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.text_encoder.input");
    set_text_encoder_input_inputs(phone_ids, tone_ids, language_ids, bert_t_major, style_vec, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);
    output.assign(output_data, output_data + ggml_nelements(output_tensor));
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_duration_predictor_graph(uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->duration_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, model->hidden_channels, 1);
    ggml_set_input(sctx->duration_x);
    sctx->set_tensor_backend(sctx->duration_x);
    ggml_set_name(sctx->duration_x, "style_bert_vits2_duration_x");

    sctx->duration_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->duration_x_mask);
    sctx->set_tensor_backend(sctx->duration_x_mask);
    ggml_set_name(sctx->duration_x_mask, "style_bert_vits2_duration_x_mask");

    sctx->duration_g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, model->gin_channels, 1);
    ggml_set_input(sctx->duration_g);
    sctx->set_tensor_backend(sctx->duration_g);
    ggml_set_name(sctx->duration_g, "style_bert_vits2_duration_g");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * output = build_duration_predictor(ctx, &*model, sctx);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_duration_predictor_inputs(const float * x_nct,
                                                            const float * x_mask_nct,
                                                            const float * g_nct,
                                                            uint32_t tokens) {
    if (sctx->duration_x && sctx->duration_x->buffer) {
        std::vector<float> x((size_t) tokens * model->hidden_channels);
        for (uint32_t c = 0; c < model->hidden_channels; ++c) {
            for (uint32_t t = 0; t < tokens; ++t) {
                x[(size_t) t + (size_t) tokens * c] = x_nct[(size_t) c * tokens + t];
            }
        }
        ggml_backend_tensor_set(sctx->duration_x, x.data(), 0, x.size() * sizeof(float));
    }

    if (sctx->duration_x_mask && sctx->duration_x_mask->buffer) {
        ggml_backend_tensor_set(sctx->duration_x_mask, x_mask_nct, 0, (size_t) tokens * sizeof(float));
    }

    if (sctx->duration_g && sctx->duration_g->buffer) {
        ggml_backend_tensor_set(sctx->duration_g, g_nct, 0, (size_t) model->gin_channels * sizeof(float));
    }
}

void style_bert_vits2_runner::predict_duration(const float * x_nct,
                                               const float * x_mask_nct,
                                               const float * g_nct,
                                               uint32_t tokens,
                                               std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_duration_predictor_graph(tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.duration_predictor");
    set_duration_predictor_inputs(x_nct, x_mask_nct, g_nct, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);
    output.assign(output_data, output_data + ggml_nelements(output_tensor));
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_stochastic_duration_condition_graph(uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->sdp_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, model->hidden_channels, 1);
    ggml_set_input(sctx->sdp_x);
    sctx->set_tensor_backend(sctx->sdp_x);
    ggml_set_name(sctx->sdp_x, "style_bert_vits2_sdp_x");

    sctx->sdp_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->sdp_x_mask);
    sctx->set_tensor_backend(sctx->sdp_x_mask);
    ggml_set_name(sctx->sdp_x_mask, "style_bert_vits2_sdp_x_mask");

    sctx->sdp_g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, model->gin_channels, 1);
    ggml_set_input(sctx->sdp_g);
    sctx->set_tensor_backend(sctx->sdp_g);
    ggml_set_name(sctx->sdp_g, "style_bert_vits2_sdp_g");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * output = build_stochastic_duration_condition(ctx, &*model, sctx);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_stochastic_duration_condition_inputs(const float * x_nct,
                                                                       const float * x_mask_nct,
                                                                       const float * g_nct,
                                                                       uint32_t tokens) {
    if (sctx->sdp_x && sctx->sdp_x->buffer) {
        std::vector<float> x((size_t) tokens * model->hidden_channels);
        for (uint32_t c = 0; c < model->hidden_channels; ++c) {
            for (uint32_t t = 0; t < tokens; ++t) {
                x[(size_t) t + (size_t) tokens * c] = x_nct[(size_t) c * tokens + t];
            }
        }
        ggml_backend_tensor_set(sctx->sdp_x, x.data(), 0, x.size() * sizeof(float));
    }

    if (sctx->sdp_x_mask && sctx->sdp_x_mask->buffer) {
        ggml_backend_tensor_set(sctx->sdp_x_mask, x_mask_nct, 0, (size_t) tokens * sizeof(float));
    }

    if (sctx->sdp_g && sctx->sdp_g->buffer) {
        ggml_backend_tensor_set(sctx->sdp_g, g_nct, 0, (size_t) model->gin_channels * sizeof(float));
    }
}

void style_bert_vits2_runner::run_stochastic_duration_condition(const float * x_nct,
                                                                const float * x_mask_nct,
                                                                const float * g_nct,
                                                                uint32_t tokens,
                                                                std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_stochastic_duration_condition_graph(tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.sdp.condition");
    set_stochastic_duration_condition_inputs(x_nct, x_mask_nct, g_nct, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);

    output.assign((size_t) model->hidden_channels * tokens, 0.0f);
    for (uint32_t c = 0; c < model->hidden_channels; ++c) {
        for (uint32_t t = 0; t < tokens; ++t) {
            output[(size_t) c * tokens + t] = output_data[(size_t) t + (size_t) tokens * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_stochastic_duration_reverse_graph(uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->sdp_reverse_z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 2, 1);
    ggml_set_input(sctx->sdp_reverse_z);
    sctx->set_tensor_backend(sctx->sdp_reverse_z);
    ggml_set_name(sctx->sdp_reverse_z, "style_bert_vits2_sdp_reverse_z");

    sctx->sdp_reverse_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->sdp_reverse_x_mask);
    sctx->set_tensor_backend(sctx->sdp_reverse_x_mask);
    ggml_set_name(sctx->sdp_reverse_x_mask, "style_bert_vits2_sdp_reverse_x_mask");

    sctx->sdp_reverse_condition = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, model->hidden_channels, 1);
    ggml_set_input(sctx->sdp_reverse_condition);
    sctx->set_tensor_backend(sctx->sdp_reverse_condition);
    ggml_set_name(sctx->sdp_reverse_condition, "style_bert_vits2_sdp_reverse_condition");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, std::max<size_t>(400000, (size_t) tokens * 16000), false);
    ggml_tensor * output = build_stochastic_duration_reverse(ctx, &*model, sctx);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_stochastic_duration_reverse_inputs(const float * z_nct,
                                                                     const float * x_mask_nct,
                                                                     const float * condition_nct,
                                                                     uint32_t tokens) {
    if (sctx->sdp_reverse_z && sctx->sdp_reverse_z->buffer) {
        std::vector<float> z((size_t) tokens * 2);
        for (uint32_t c = 0; c < 2; ++c) {
            for (uint32_t t = 0; t < tokens; ++t) {
                z[(size_t) t + (size_t) tokens * c] = z_nct[(size_t) c * tokens + t];
            }
        }
        ggml_backend_tensor_set(sctx->sdp_reverse_z, z.data(), 0, z.size() * sizeof(float));
    }

    if (sctx->sdp_reverse_x_mask && sctx->sdp_reverse_x_mask->buffer) {
        ggml_backend_tensor_set(sctx->sdp_reverse_x_mask, x_mask_nct, 0, (size_t) tokens * sizeof(float));
    }

    if (sctx->sdp_reverse_condition && sctx->sdp_reverse_condition->buffer) {
        std::vector<float> condition((size_t) tokens * model->hidden_channels);
        for (uint32_t c = 0; c < model->hidden_channels; ++c) {
            for (uint32_t t = 0; t < tokens; ++t) {
                condition[(size_t) t + (size_t) tokens * c] = condition_nct[(size_t) c * tokens + t];
            }
        }
        ggml_backend_tensor_set(sctx->sdp_reverse_condition, condition.data(), 0, condition.size() * sizeof(float));
    }
}

void style_bert_vits2_runner::run_stochastic_duration_reverse(const float * z_nct,
                                                              const float * x_mask_nct,
                                                              const float * condition_nct,
                                                              uint32_t tokens,
                                                              std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_stochastic_duration_reverse_graph(tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.sdp.reverse");
    set_stochastic_duration_reverse_inputs(z_nct, x_mask_nct, condition_nct, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);

    output.assign((size_t) tokens, 0.0f);
    for (uint32_t t = 0; t < tokens; ++t) {
        output[t] = output_data[t];
    }
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_text_encoder_projection_graph(uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->text_projection_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, model->hidden_channels, 1);
    ggml_set_input(sctx->text_projection_x);
    sctx->set_tensor_backend(sctx->text_projection_x);
    ggml_set_name(sctx->text_projection_x, "style_bert_vits2_text_projection_x");

    sctx->text_projection_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->text_projection_x_mask);
    sctx->set_tensor_backend(sctx->text_projection_x_mask);
    ggml_set_name(sctx->text_projection_x_mask, "style_bert_vits2_text_projection_x_mask");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * output = build_text_encoder_projection(ctx, &*model, sctx);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_text_encoder_projection_inputs(const float * x_nct,
                                                                 const float * x_mask_nct,
                                                                 uint32_t tokens) {
    if (sctx->text_projection_x && sctx->text_projection_x->buffer) {
        std::vector<float> x((size_t) tokens * model->hidden_channels);
        for (uint32_t c = 0; c < model->hidden_channels; ++c) {
            for (uint32_t t = 0; t < tokens; ++t) {
                x[(size_t) t + (size_t) tokens * c] = x_nct[(size_t) c * tokens + t];
            }
        }
        ggml_backend_tensor_set(sctx->text_projection_x, x.data(), 0, x.size() * sizeof(float));
    }

    if (sctx->text_projection_x_mask && sctx->text_projection_x_mask->buffer) {
        ggml_backend_tensor_set(sctx->text_projection_x_mask, x_mask_nct, 0, (size_t) tokens * sizeof(float));
    }
}

static void set_relative_position_ids_input(
    ggml_tensor * rel_pos_ids,
    uint32_t tokens,
    uint32_t window_size) {
    if (!rel_pos_ids || !rel_pos_ids->buffer) {
        return;
    }
    const int32_t zero_row = (int32_t) (2 * window_size + 1);
    std::vector<int32_t> ids((size_t) tokens * tokens, zero_row);
    for (uint32_t query_index = 0; query_index < tokens; ++query_index) {
        for (uint32_t key_index = 0; key_index < tokens; ++key_index) {
            const int64_t relative_index =
                (int64_t) window_size + (int64_t) key_index - (int64_t) query_index;
            if (relative_index >= 0 && relative_index < (int64_t) zero_row) {
                ids[(size_t) key_index + (size_t) tokens * query_index] = (int32_t) relative_index;
            }
        }
    }
    ggml_backend_tensor_set(rel_pos_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
}

void style_bert_vits2_runner::project_text_encoder(const float * x_nct,
                                                   const float * x_mask_nct,
                                                   uint32_t tokens,
                                                   std::vector<float> & m_p_nct,
                                                   std::vector<float> & logs_p_nct) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_text_encoder_projection_graph(tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.text_encoder.projection");
    set_text_encoder_projection_inputs(x_nct, x_mask_nct, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);

    m_p_nct.assign((size_t) model->inter_channels * tokens, 0.0f);
    logs_p_nct.assign((size_t) model->inter_channels * tokens, 0.0f);
    for (uint32_t c = 0; c < model->inter_channels; ++c) {
        for (uint32_t t = 0; t < tokens; ++t) {
            m_p_nct[(size_t) c * tokens + t] = output_data[(size_t) t + (size_t) tokens * c];
            logs_p_nct[(size_t) c * tokens + t] =
                output_data[(size_t) t + (size_t) tokens * ((size_t) model->inter_channels + c)];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_text_encoder_attention_graph(uint32_t layer_index, uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->encoder_attn_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, model->hidden_channels, 1);
    ggml_set_input(sctx->encoder_attn_x);
    sctx->set_tensor_backend(sctx->encoder_attn_x);
    ggml_set_name(sctx->encoder_attn_x, "style_bert_vits2_encoder_attn_x");

    sctx->encoder_attn_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->encoder_attn_x_mask);
    sctx->set_tensor_backend(sctx->encoder_attn_x_mask);
    ggml_set_name(sctx->encoder_attn_x_mask, "style_bert_vits2_encoder_attn_x_mask");

    ggml_tensor * rel_pos_ids = new_relative_position_ids_input(
        ctx, sctx, tokens, "style_bert_vits2_encoder_attn_rel_pos_ids");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, false);
    ggml_tensor * output = build_text_encoder_attention(ctx,
                                                        model->text_encoder.encoder,
                                                        model->text_encoder.encoder.attn_layers.at(layer_index),
                                                        sctx->encoder_attn_x,
                                                        sctx->encoder_attn_x_mask,
                                                        rel_pos_ids);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_text_encoder_attention_inputs(const float * x_nct,
                                                                const float * x_mask_nct,
                                                                uint32_t tokens) {
    if (sctx->encoder_attn_x && sctx->encoder_attn_x->buffer) {
        std::vector<float> x((size_t) tokens * model->hidden_channels);
        for (uint32_t c = 0; c < model->hidden_channels; ++c) {
            for (uint32_t t = 0; t < tokens; ++t) {
                x[(size_t) t + (size_t) tokens * c] = x_nct[(size_t) c * tokens + t];
            }
        }
        ggml_backend_tensor_set(sctx->encoder_attn_x, x.data(), 0, x.size() * sizeof(float));
    }

    if (sctx->encoder_attn_x_mask && sctx->encoder_attn_x_mask->buffer) {
        ggml_backend_tensor_set(sctx->encoder_attn_x_mask, x_mask_nct, 0, (size_t) tokens * sizeof(float));
    }

    set_relative_position_ids_input(
        sctx->encoder_rel_pos_ids,
        tokens,
        model->text_encoder.encoder.window_size);
}

void style_bert_vits2_runner::run_text_encoder_attention(uint32_t layer_index,
                                                         const float * x_nct,
                                                         const float * x_mask_nct,
                                                         uint32_t tokens,
                                                         std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_text_encoder_attention_graph(layer_index, tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.text_encoder.attention");
    set_text_encoder_attention_inputs(x_nct, x_mask_nct, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);

    output.assign((size_t) model->hidden_channels * tokens, 0.0f);
    for (uint32_t c = 0; c < model->hidden_channels; ++c) {
        for (uint32_t t = 0; t < tokens; ++t) {
            output[(size_t) c * tokens + t] = output_data[(size_t) t + (size_t) tokens * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_text_encoder_layer_graph(uint32_t layer_index, uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->encoder_layer_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, model->hidden_channels, 1);
    ggml_set_input(sctx->encoder_layer_x);
    sctx->set_tensor_backend(sctx->encoder_layer_x);
    ggml_set_name(sctx->encoder_layer_x, "style_bert_vits2_encoder_layer_x");

    sctx->encoder_layer_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->encoder_layer_x_mask);
    sctx->set_tensor_backend(sctx->encoder_layer_x_mask);
    ggml_set_name(sctx->encoder_layer_x_mask, "style_bert_vits2_encoder_layer_x_mask");

    sctx->encoder_layer_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model->gin_channels);
    ggml_set_input(sctx->encoder_layer_g);
    sctx->set_tensor_backend(sctx->encoder_layer_g);
    ggml_set_name(sctx->encoder_layer_g, "style_bert_vits2_encoder_layer_g");

    ggml_tensor * rel_pos_ids = new_relative_position_ids_input(
        ctx, sctx, tokens, "style_bert_vits2_encoder_layer_rel_pos_ids");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, false);
    ggml_tensor * output = build_text_encoder_layer(ctx,
                                                    model->text_encoder.encoder,
                                                    layer_index,
                                                    sctx->encoder_layer_x,
                                                    sctx->encoder_layer_x_mask,
                                                    sctx->encoder_layer_g,
                                                    rel_pos_ids);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_text_encoder_layer_inputs(const float * x_nct,
                                                            const float * x_mask_nct,
                                                            const float * g_nct,
                                                            uint32_t tokens) {
    if (sctx->encoder_layer_x && sctx->encoder_layer_x->buffer) {
        std::vector<float> x((size_t) tokens * model->hidden_channels);
        for (uint32_t c = 0; c < model->hidden_channels; ++c) {
            for (uint32_t t = 0; t < tokens; ++t) {
                x[(size_t) t + (size_t) tokens * c] = x_nct[(size_t) c * tokens + t];
            }
        }
        ggml_backend_tensor_set(sctx->encoder_layer_x, x.data(), 0, x.size() * sizeof(float));
    }

    if (sctx->encoder_layer_x_mask && sctx->encoder_layer_x_mask->buffer) {
        ggml_backend_tensor_set(sctx->encoder_layer_x_mask, x_mask_nct, 0, (size_t) tokens * sizeof(float));
    }

    if (sctx->encoder_layer_g && sctx->encoder_layer_g->buffer) {
        if (g_nct) {
            ggml_backend_tensor_set(sctx->encoder_layer_g, g_nct, 0, (size_t) model->gin_channels * sizeof(float));
        } else {
            std::vector<float> zero_g(model->gin_channels, 0.0f);
            ggml_backend_tensor_set(sctx->encoder_layer_g, zero_g.data(), 0, zero_g.size() * sizeof(float));
        }
    }

    set_relative_position_ids_input(
        sctx->encoder_rel_pos_ids,
        tokens,
        model->text_encoder.encoder.window_size);
}

void style_bert_vits2_runner::run_text_encoder_layer(uint32_t layer_index,
                                                     const float * x_nct,
                                                     const float * x_mask_nct,
                                                     const float * g_nct,
                                                     uint32_t tokens,
                                                     std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_text_encoder_layer_graph(layer_index, tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.text_encoder.layer");
    set_text_encoder_layer_inputs(x_nct, x_mask_nct, g_nct, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);

    output.assign((size_t) model->hidden_channels * tokens, 0.0f);
    for (uint32_t c = 0; c < model->hidden_channels; ++c) {
        for (uint32_t t = 0; t < tokens; ++t) {
            output[(size_t) c * tokens + t] = output_data[(size_t) t + (size_t) tokens * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_text_encoder_stack_graph(uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->encoder_layer_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, model->hidden_channels, 1);
    ggml_set_input(sctx->encoder_layer_x);
    sctx->set_tensor_backend(sctx->encoder_layer_x);
    ggml_set_name(sctx->encoder_layer_x, "style_bert_vits2_encoder_stack_x");

    sctx->encoder_layer_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->encoder_layer_x_mask);
    sctx->set_tensor_backend(sctx->encoder_layer_x_mask);
    ggml_set_name(sctx->encoder_layer_x_mask, "style_bert_vits2_encoder_stack_x_mask");

    sctx->encoder_layer_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model->gin_channels);
    ggml_set_input(sctx->encoder_layer_g);
    sctx->set_tensor_backend(sctx->encoder_layer_g);
    ggml_set_name(sctx->encoder_layer_g, "style_bert_vits2_encoder_stack_g");

    ggml_tensor * rel_pos_ids = new_relative_position_ids_input(
        ctx, sctx, tokens, "style_bert_vits2_encoder_stack_rel_pos_ids");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, false);
    ggml_tensor * output = build_text_encoder_stack(ctx,
                                                    model->text_encoder.encoder,
                                                    sctx->encoder_layer_x,
                                                    sctx->encoder_layer_x_mask,
                                                    sctx->encoder_layer_g,
                                                    rel_pos_ids);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::run_text_encoder_stack(const float * x_nct,
                                                     const float * x_mask_nct,
                                                     const float * g_nct,
                                                     uint32_t tokens,
                                                     std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_text_encoder_stack_graph(tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.text_encoder.stack");
    set_text_encoder_layer_inputs(x_nct, x_mask_nct, g_nct, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);

    output.assign((size_t) model->hidden_channels * tokens, 0.0f);
    for (uint32_t c = 0; c < model->hidden_channels; ++c) {
        for (uint32_t t = 0; t < tokens; ++t) {
            output[(size_t) c * tokens + t] = output_data[(size_t) t + (size_t) tokens * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_text_encoder_graph(uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->phone_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(sctx->phone_ids);
    sctx->set_tensor_backend(sctx->phone_ids);
    ggml_set_name(sctx->phone_ids, "style_bert_vits2_phone_ids");

    sctx->tone_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(sctx->tone_ids);
    sctx->set_tensor_backend(sctx->tone_ids);
    ggml_set_name(sctx->tone_ids, "style_bert_vits2_tone_ids");

    sctx->language_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(sctx->language_ids);
    sctx->set_tensor_backend(sctx->language_ids);
    ggml_set_name(sctx->language_ids, "style_bert_vits2_language_ids");

    sctx->bert = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1024, 1);
    ggml_set_input(sctx->bert);
    sctx->set_tensor_backend(sctx->bert);
    ggml_set_name(sctx->bert, "style_bert_vits2_bert");

    sctx->style_vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_set_input(sctx->style_vec);
    sctx->set_tensor_backend(sctx->style_vec);
    ggml_set_name(sctx->style_vec, "style_bert_vits2_style_vec");

    sctx->encoder_layer_x = nullptr;
    sctx->encoder_layer_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->encoder_layer_x_mask);
    sctx->set_tensor_backend(sctx->encoder_layer_x_mask);
    ggml_set_name(sctx->encoder_layer_x_mask, "style_bert_vits2_text_encoder_x_mask");

    sctx->encoder_layer_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model->gin_channels);
    ggml_set_input(sctx->encoder_layer_g);
    sctx->set_tensor_backend(sctx->encoder_layer_g);
    ggml_set_name(sctx->encoder_layer_g, "style_bert_vits2_text_encoder_g");

    ggml_tensor * rel_pos_ids = new_relative_position_ids_input(
        ctx, sctx, tokens, "style_bert_vits2_text_encoder_rel_pos_ids");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, false);
    ggml_tensor * enc_input = build_text_encoder_input(ctx, &*model, sctx);
    ggml_tensor * x = build_text_encoder_stack(ctx,
                                               model->text_encoder.encoder,
                                               enc_input,
                                               sctx->encoder_layer_x_mask,
                                               sctx->encoder_layer_g,
                                               rel_pos_ids);
    ggml_tensor * stats = build_text_encoder_projection_from(ctx, &*model, x, sctx->encoder_layer_x_mask);
    ggml_tensor * output = ggml_concat(ctx, x, stats, 1);
    ggml_set_name(output, "style_bert_vits2.text_encoder.output");
    sctx->text_encoder_x_output = x;
    sctx->text_encoder_stats_output = output;
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::run_text_encoder(const int32_t * phone_ids,
                                               const int32_t * tone_ids,
                                               const int32_t * language_ids,
                                               const float * bert_t_major,
                                               const float * style_vec,
                                               const float * x_mask_nct,
                                               const float * g_nct,
                                               uint32_t tokens,
                                               std::vector<float> & x_nct,
                                               std::vector<float> & m_p_nct,
                                               std::vector<float> & logs_p_nct) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_text_encoder_graph(tokens);
    ggml_tensor * output_tensor = sctx->text_encoder_stats_output;
    TTS_ASSERT(output_tensor);
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.text_encoder");
    set_text_encoder_input_inputs(phone_ids, tone_ids, language_ids, bert_t_major, style_vec, tokens);
    ggml_backend_tensor_set(sctx->encoder_layer_x_mask, x_mask_nct, 0, (size_t) tokens * sizeof(float));
    if (g_nct) {
        ggml_backend_tensor_set(sctx->encoder_layer_g, g_nct, 0, (size_t) model->gin_channels * sizeof(float));
    } else {
        std::vector<float> zero_g(model->gin_channels, 0.0f);
        ggml_backend_tensor_set(sctx->encoder_layer_g, zero_g.data(), 0, zero_g.size() * sizeof(float));
    }
    set_relative_position_ids_input(
        sctx->encoder_rel_pos_ids,
        tokens,
        model->text_encoder.encoder.window_size);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);

    sctx->get_ggml_node_data(output_tensor, output_data, output_size);
    x_nct.assign((size_t) model->hidden_channels * tokens, 0.0f);
    for (uint32_t c = 0; c < model->hidden_channels; ++c) {
        for (uint32_t t = 0; t < tokens; ++t) {
            x_nct[(size_t) c * tokens + t] = output_data[(size_t) t + (size_t) tokens * c];
        }
    }

    m_p_nct.assign((size_t) model->inter_channels * tokens, 0.0f);
    logs_p_nct.assign((size_t) model->inter_channels * tokens, 0.0f);
    for (uint32_t c = 0; c < model->inter_channels; ++c) {
        for (uint32_t t = 0; t < tokens; ++t) {
            const size_t m_channel = (size_t) model->hidden_channels + c;
            const size_t logs_channel = (size_t) model->hidden_channels + (size_t) model->inter_channels + c;
            m_p_nct[(size_t) c * tokens + t] = output_data[(size_t) t + (size_t) tokens * m_channel];
            logs_p_nct[(size_t) c * tokens + t] =
                output_data[(size_t) t + (size_t) tokens * logs_channel];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_text_encoder_ffn_graph(uint32_t layer_index, uint32_t tokens) {
    init_build();
    sctx->text_tokens = tokens;
    sctx->encoder_ffn_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, model->hidden_channels, 1);
    ggml_set_input(sctx->encoder_ffn_x);
    sctx->set_tensor_backend(sctx->encoder_ffn_x);
    ggml_set_name(sctx->encoder_ffn_x, "style_bert_vits2_encoder_ffn_x");

    sctx->encoder_ffn_x_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tokens, 1, 1);
    ggml_set_input(sctx->encoder_ffn_x_mask);
    sctx->set_tensor_backend(sctx->encoder_ffn_x_mask);
    ggml_set_name(sctx->encoder_ffn_x_mask, "style_bert_vits2_encoder_ffn_x_mask");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * output = build_text_encoder_ffn(ctx,
                                                  model->text_encoder.encoder.ffn_layers.at(layer_index),
                                                  sctx->encoder_ffn_x,
                                                  sctx->encoder_ffn_x_mask);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_text_encoder_ffn_inputs(const float * x_nct,
                                                          const float * x_mask_nct,
                                                          uint32_t tokens) {
    if (sctx->encoder_ffn_x && sctx->encoder_ffn_x->buffer) {
        std::vector<float> x((size_t) tokens * model->hidden_channels);
        for (uint32_t c = 0; c < model->hidden_channels; ++c) {
            for (uint32_t t = 0; t < tokens; ++t) {
                x[(size_t) t + (size_t) tokens * c] = x_nct[(size_t) c * tokens + t];
            }
        }
        ggml_backend_tensor_set(sctx->encoder_ffn_x, x.data(), 0, x.size() * sizeof(float));
    }

    if (sctx->encoder_ffn_x_mask && sctx->encoder_ffn_x_mask->buffer) {
        ggml_backend_tensor_set(sctx->encoder_ffn_x_mask, x_mask_nct, 0, (size_t) tokens * sizeof(float));
    }
}

void style_bert_vits2_runner::run_text_encoder_ffn(uint32_t layer_index,
                                                   const float * x_nct,
                                                   const float * x_mask_nct,
                                                   uint32_t tokens,
                                                   std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_text_encoder_ffn_graph(layer_index, tokens);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.text_encoder.ffn");
    set_text_encoder_ffn_inputs(x_nct, x_mask_nct, tokens);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);

    output.assign((size_t) model->hidden_channels * tokens, 0.0f);
    for (uint32_t c = 0; c < model->hidden_channels; ++c) {
        for (uint32_t t = 0; t < tokens; ++t) {
            output[(size_t) c * tokens + t] = output_data[(size_t) t + (size_t) tokens * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
}

style_bert_vits2_alignment_result style_bert_vits2_runner::expand_alignment(const float * logw_nct,
                                                                            const float * x_mask_nct,
                                                                            const float * m_p_nct,
                                                                            const float * logs_p_nct,
                                                                            uint32_t tokens,
                                                                            float length_scale) {
    style_bert_vits2_alignment_result result;
    result.w.assign(tokens, 0.0f);
    result.w_ceil.assign(tokens, 0.0f);

    uint32_t frames = 0;
    for (uint32_t t = 0; t < tokens; ++t) {
        const float mask = x_mask_nct[t];
        const float w = std::exp(logw_nct[t]) * mask * length_scale;
        const float w_ceil = std::ceil(w);
        result.w[t] = w;
        result.w_ceil[t] = w_ceil;
        if (w_ceil > 0.0f) {
            frames += (uint32_t) w_ceil;
        }
    }
    if (frames == 0) {
        frames = 1;
    }
    result.frames = frames;
    result.y_mask.assign(frames, 1.0f);
    result.attn.assign((size_t) frames * tokens, 0.0f);
    result.m_p_expanded.assign((size_t) model->inter_channels * frames, 0.0f);
    result.logs_p_expanded.assign((size_t) model->inter_channels * frames, 0.0f);

    uint32_t start = 0;
    for (uint32_t t = 0; t < tokens; ++t) {
        const uint32_t duration = result.w_ceil[t] > 0.0f ? (uint32_t) result.w_ceil[t] : 0;
        const uint32_t end = std::min(frames, start + duration);
        for (uint32_t y = start; y < end; ++y) {
            const float active = x_mask_nct[t];
            result.attn[(size_t) y * tokens + t] = active;
            if (active == 0.0f) {
                continue;
            }
            for (uint32_t c = 0; c < model->inter_channels; ++c) {
                result.m_p_expanded[(size_t) c * frames + y] = m_p_nct[(size_t) c * tokens + t];
                result.logs_p_expanded[(size_t) c * frames + y] = logs_p_nct[(size_t) c * tokens + t];
            }
        }
        start = end;
    }
    return result;
}

ggml_cgraph * style_bert_vits2_runner::build_prior_sample_graph(uint32_t frames) {
    init_build();
    sctx->prior_frames = frames;
    sctx->prior_m_p = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, model->inter_channels, 1);
    ggml_set_input(sctx->prior_m_p);
    sctx->set_tensor_backend(sctx->prior_m_p);
    ggml_set_name(sctx->prior_m_p, "style_bert_vits2_prior_m_p");

    sctx->prior_logs_p = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, model->inter_channels, 1);
    ggml_set_input(sctx->prior_logs_p);
    sctx->set_tensor_backend(sctx->prior_logs_p);
    ggml_set_name(sctx->prior_logs_p, "style_bert_vits2_prior_logs_p");

    sctx->prior_noise = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, model->inter_channels, 1);
    ggml_set_input(sctx->prior_noise);
    sctx->set_tensor_backend(sctx->prior_noise);
    ggml_set_name(sctx->prior_noise, "style_bert_vits2_prior_noise");

    sctx->prior_noise_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(sctx->prior_noise_scale);
    sctx->set_tensor_backend(sctx->prior_noise_scale);
    ggml_set_name(sctx->prior_noise_scale, "style_bert_vits2_prior_noise_scale");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * output = build_prior_sample(ctx, &*model, sctx);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_prior_sample_inputs(const float * m_p_nct,
                                                      const float * logs_p_nct,
                                                      const float * noise_nct,
                                                      uint32_t frames,
                                                      float noise_scale) {
    const size_t elements = (size_t) frames * model->inter_channels;
    std::vector<float> m_p(elements);
    std::vector<float> logs_p(elements);
    std::vector<float> noise(elements);
    for (uint32_t c = 0; c < model->inter_channels; ++c) {
        for (uint32_t t = 0; t < frames; ++t) {
            const size_t src = (size_t) c * frames + t;
            const size_t dst = (size_t) t + (size_t) frames * c;
            m_p[dst] = m_p_nct[src];
            logs_p[dst] = logs_p_nct[src];
            noise[dst] = noise_nct[src];
        }
    }
    ggml_backend_tensor_set(sctx->prior_m_p, m_p.data(), 0, m_p.size() * sizeof(float));
    ggml_backend_tensor_set(sctx->prior_logs_p, logs_p.data(), 0, logs_p.size() * sizeof(float));
    ggml_backend_tensor_set(sctx->prior_noise, noise.data(), 0, noise.size() * sizeof(float));
    ggml_backend_tensor_set(sctx->prior_noise_scale, &noise_scale, 0, sizeof(float));
}

void style_bert_vits2_runner::sample_prior(const float * m_p_nct,
                                           const float * logs_p_nct,
                                           const float * noise_nct,
                                           uint32_t frames,
                                           float noise_scale,
                                           std::vector<float> & output) {
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_prior_sample_graph(frames);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.prior_sample");
    set_prior_sample_inputs(m_p_nct, logs_p_nct, noise_nct, frames, noise_scale);
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);

    output.assign((size_t) model->inter_channels * frames, 0.0f);
    for (uint32_t c = 0; c < model->inter_channels; ++c) {
        for (uint32_t t = 0; t < frames; ++t) {
            output[(size_t) c * frames + t] = output_data[(size_t) t + (size_t) frames * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
}

ggml_cgraph * style_bert_vits2_runner::build_flow_reverse_step_graph(uint32_t flow_index, uint32_t frames) {
    init_build();
    sctx->prior_frames = frames;
    sctx->flow_z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, model->inter_channels, 1);
    ggml_set_input(sctx->flow_z);
    sctx->set_tensor_backend(sctx->flow_z);
    ggml_set_name(sctx->flow_z, "style_bert_vits2_flow_z_p");

    sctx->flow_y_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, 1, 1);
    ggml_set_input(sctx->flow_y_mask);
    sctx->set_tensor_backend(sctx->flow_y_mask);
    ggml_set_name(sctx->flow_y_mask, "style_bert_vits2_flow_y_mask");

    sctx->flow_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model->gin_channels);
    ggml_set_input(sctx->flow_g);
    sctx->set_tensor_backend(sctx->flow_g);
    ggml_set_name(sctx->flow_g, "style_bert_vits2_flow_g");

    ggml_tensor * rel_pos_ids = new_relative_position_ids_input(
        ctx, sctx, frames, "style_bert_vits2_flow_rel_pos_ids");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, flow_graph_node_capacity(frames, 1), false);
    ggml_tensor * output = flip_channels(ctx, sctx->flow_z);
    output = build_flow_layer_reverse(ctx,
                                      model->flow.layers.at(flow_index),
                                      output,
                                      sctx->flow_y_mask,
                                      sctx->flow_g,
                                      rel_pos_ids);
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

ggml_cgraph * style_bert_vits2_runner::build_flow_reverse_group_graph(uint32_t start_flow_index,
                                                                       uint32_t layer_count,
                                                                       uint32_t frames) {
    init_build();
    sctx->prior_frames = frames;
    sctx->flow_z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, model->inter_channels, 1);
    ggml_set_input(sctx->flow_z);
    sctx->set_tensor_backend(sctx->flow_z);
    ggml_set_name(sctx->flow_z, "style_bert_vits2_flow_z_p");

    sctx->flow_y_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, 1, 1);
    ggml_set_input(sctx->flow_y_mask);
    sctx->set_tensor_backend(sctx->flow_y_mask);
    ggml_set_name(sctx->flow_y_mask, "style_bert_vits2_flow_y_mask");

    sctx->flow_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model->gin_channels);
    ggml_set_input(sctx->flow_g);
    sctx->set_tensor_backend(sctx->flow_g);
    ggml_set_name(sctx->flow_g, "style_bert_vits2_flow_g");

    ggml_tensor * rel_pos_ids = new_relative_position_ids_input(
        ctx, sctx, frames, "style_bert_vits2_flow_rel_pos_ids");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, flow_graph_node_capacity(frames, layer_count), false);
    ggml_tensor * output = sctx->flow_z;
    uint32_t consumed = 0;
    for (int64_t i = (int64_t) start_flow_index; i >= 0 && consumed < layer_count; --i, ++consumed) {
        output = flip_channels(ctx, output);
        output = build_flow_layer_reverse(ctx,
                                          model->flow.layers.at((size_t) i),
                                          output,
                                          sctx->flow_y_mask,
                                          sctx->flow_g,
                                          rel_pos_ids);
    }
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

ggml_cgraph * style_bert_vits2_runner::build_flow_reverse_graph(uint32_t frames) {
    init_build();
    sctx->prior_frames = frames;
    sctx->flow_z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, model->inter_channels, 1);
    ggml_set_input(sctx->flow_z);
    sctx->set_tensor_backend(sctx->flow_z);
    ggml_set_name(sctx->flow_z, "style_bert_vits2_flow_z_p");

    sctx->flow_y_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, 1, 1);
    ggml_set_input(sctx->flow_y_mask);
    sctx->set_tensor_backend(sctx->flow_y_mask);
    ggml_set_name(sctx->flow_y_mask, "style_bert_vits2_flow_y_mask");

    sctx->flow_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model->gin_channels);
    ggml_set_input(sctx->flow_g);
    sctx->set_tensor_backend(sctx->flow_g);
    ggml_set_name(sctx->flow_g, "style_bert_vits2_flow_g");

    ggml_tensor * rel_pos_ids = new_relative_position_ids_input(
        ctx, sctx, frames, "style_bert_vits2_flow_rel_pos_ids");

    ggml_cgraph * gf = ggml_new_graph_custom(
        ctx,
        flow_graph_node_capacity(frames, (uint32_t) model->flow.layers.size()),
        false);
    ggml_tensor * output = sctx->flow_z;
    for (int64_t i = (int64_t) model->flow.layers.size() - 1; i >= 0; --i) {
        output = flip_channels(ctx, output);
        output = build_flow_layer_reverse(ctx,
                                          model->flow.layers.at((size_t) i),
                                          output,
                                          sctx->flow_y_mask,
                                          sctx->flow_g,
                                          rel_pos_ids);
    }
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_flow_reverse_inputs(const float * z_p_nct,
                                                      const float * y_mask_nct,
                                                      const float * g_nct,
                                                      uint32_t frames) {
    if (sctx->flow_z && sctx->flow_z->buffer) {
        std::vector<float> z((size_t) frames * model->inter_channels);
        for (uint32_t c = 0; c < model->inter_channels; ++c) {
            for (uint32_t t = 0; t < frames; ++t) {
                z[(size_t) t + (size_t) frames * c] = z_p_nct[(size_t) c * frames + t];
            }
        }
        ggml_backend_tensor_set(sctx->flow_z, z.data(), 0, z.size() * sizeof(float));
    }

    if (sctx->flow_y_mask && sctx->flow_y_mask->buffer) {
        ggml_backend_tensor_set(sctx->flow_y_mask, y_mask_nct, 0, (size_t) frames * sizeof(float));
    }

    if (sctx->flow_g && sctx->flow_g->buffer) {
        ggml_backend_tensor_set(sctx->flow_g, g_nct, 0, (size_t) model->gin_channels * sizeof(float));
    }

    set_relative_position_ids_input(
        sctx->encoder_rel_pos_ids,
        frames,
        model->flow.window_size);
}

void style_bert_vits2_runner::run_flow_reverse_step(uint32_t flow_index,
                                                    const float * z_p_nct,
                                                    const float * y_mask_nct,
                                                    const float * g_nct,
                                                    uint32_t frames,
                                                    std::vector<float> & output) {
    const bool timings = debug_timings_enabled();
    const auto t_start = std::chrono::steady_clock::now();
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_flow_reverse_step_graph(flow_index, frames);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    const auto t_built = std::chrono::steady_clock::now();
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.flow.reverse.step");
    const auto t_alloc = std::chrono::steady_clock::now();
    set_flow_reverse_inputs(z_p_nct, y_mask_nct, g_nct, frames);
    const auto t_inputs = std::chrono::steady_clock::now();
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    const auto t_compute = std::chrono::steady_clock::now();
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);
    const auto t_read = std::chrono::steady_clock::now();

    output.assign((size_t) model->inter_channels * frames, 0.0f);
    for (uint32_t c = 0; c < model->inter_channels; ++c) {
        for (uint32_t t = 0; t < frames; ++t) {
            output[(size_t) c * frames + t] = output_data[(size_t) t + (size_t) frames * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
    const auto t_done = std::chrono::steady_clock::now();
    if (timings) {
        const char * backend_name = sctx->backend ? ggml_backend_name(sctx->backend) : "CPU";
        std::cerr << "STYLE_BERT_VITS2_FLOW_STEP_TIMING"
                  << " index=" << flow_index
                  << " backend=" << backend_name
                  << " frames=" << frames
                  << " nodes=" << gf->n_nodes
                  << " build_ms=" << elapsed_ms(t_start, t_built)
                  << " alloc_ms=" << elapsed_ms(t_built, t_alloc)
                  << " input_ms=" << elapsed_ms(t_alloc, t_inputs)
                  << " compute_submit_ms=" << elapsed_ms(t_inputs, t_compute)
                  << " read_ms=" << elapsed_ms(t_compute, t_read)
                  << " copy_ms=" << elapsed_ms(t_read, t_done)
                  << " total_ms=" << elapsed_ms(t_start, t_done)
                  << std::endl;
    }
}

void style_bert_vits2_runner::run_flow_reverse_group(uint32_t start_flow_index,
                                                     uint32_t layer_count,
                                                     const float * z_p_nct,
                                                     const float * y_mask_nct,
                                                     const float * g_nct,
                                                     uint32_t frames,
                                                     std::vector<float> & output) {
    if (layer_count <= 1) {
        run_flow_reverse_step(start_flow_index, z_p_nct, y_mask_nct, g_nct, frames, output);
        return;
    }

    const bool timings = debug_timings_enabled();
    const auto t_start = std::chrono::steady_clock::now();
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_flow_reverse_group_graph(start_flow_index, layer_count, frames);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    const auto t_built = std::chrono::steady_clock::now();
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.flow.reverse.group");
    const auto t_alloc = std::chrono::steady_clock::now();
    set_flow_reverse_inputs(z_p_nct, y_mask_nct, g_nct, frames);
    const auto t_inputs = std::chrono::steady_clock::now();
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    const auto t_compute = std::chrono::steady_clock::now();
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);
    const auto t_read = std::chrono::steady_clock::now();

    output.assign((size_t) model->inter_channels * frames, 0.0f);
    for (uint32_t c = 0; c < model->inter_channels; ++c) {
        for (uint32_t t = 0; t < frames; ++t) {
            output[(size_t) c * frames + t] = output_data[(size_t) t + (size_t) frames * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
    const auto t_done = std::chrono::steady_clock::now();
    if (timings) {
        const char * backend_name = sctx->backend ? ggml_backend_name(sctx->backend) : "CPU";
        std::cerr << "STYLE_BERT_VITS2_FLOW_GROUP_TIMING"
                  << " start_index=" << start_flow_index
                  << " layers=" << layer_count
                  << " backend=" << backend_name
                  << " frames=" << frames
                  << " nodes=" << gf->n_nodes
                  << " build_ms=" << elapsed_ms(t_start, t_built)
                  << " alloc_ms=" << elapsed_ms(t_built, t_alloc)
                  << " input_ms=" << elapsed_ms(t_alloc, t_inputs)
                  << " compute_submit_ms=" << elapsed_ms(t_inputs, t_compute)
                  << " read_ms=" << elapsed_ms(t_compute, t_read)
                  << " copy_ms=" << elapsed_ms(t_read, t_done)
                  << " total_ms=" << elapsed_ms(t_start, t_done)
                  << std::endl;
    }
}

void style_bert_vits2_runner::run_flow_reverse_fused(const float * z_p_nct,
                                                     const float * y_mask_nct,
                                                     const float * g_nct,
                                                     uint32_t frames,
                                                     std::vector<float> & output) {
    const bool timings = debug_timings_enabled();
    const auto t_start = std::chrono::steady_clock::now();
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_flow_reverse_graph(frames);
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    const auto t_built = std::chrono::steady_clock::now();
    sctx->prep_output_buffer(output_size);
    float * output_data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.flow.reverse.fused");
    const auto t_alloc = std::chrono::steady_clock::now();
    set_flow_reverse_inputs(z_p_nct, y_mask_nct, g_nct, frames);
    const auto t_inputs = std::chrono::steady_clock::now();
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    const auto t_compute = std::chrono::steady_clock::now();
    sctx->get_ggml_node_data(output_tensor, output_data, output_size);
    const auto t_read = std::chrono::steady_clock::now();

    output.assign((size_t) model->inter_channels * frames, 0.0f);
    for (uint32_t c = 0; c < model->inter_channels; ++c) {
        for (uint32_t t = 0; t < frames; ++t) {
            output[(size_t) c * frames + t] = output_data[(size_t) t + (size_t) frames * c];
        }
    }
    ggml_backend_sched_reset(sctx->sched);
    const auto t_done = std::chrono::steady_clock::now();
    if (timings) {
        const char * backend_name = sctx->backend ? ggml_backend_name(sctx->backend) : "CPU";
        std::cerr << "STYLE_BERT_VITS2_FLOW_FUSED_TIMING"
                  << " backend=" << backend_name
                  << " frames=" << frames
                  << " layers=" << model->flow.layers.size()
                  << " nodes=" << gf->n_nodes
                  << " build_ms=" << elapsed_ms(t_start, t_built)
                  << " alloc_ms=" << elapsed_ms(t_built, t_alloc)
                  << " input_ms=" << elapsed_ms(t_alloc, t_inputs)
                  << " compute_submit_ms=" << elapsed_ms(t_inputs, t_compute)
                  << " read_ms=" << elapsed_ms(t_compute, t_read)
                  << " copy_ms=" << elapsed_ms(t_read, t_done)
                  << " total_ms=" << elapsed_ms(t_start, t_done)
                  << std::endl;
    }
}

void style_bert_vits2_runner::run_flow_reverse(const float * z_p_nct,
                                               const float * y_mask_nct,
                                               const float * g_nct,
                                               uint32_t frames,
                                               std::vector<float> & output) {
    if (flow_fused_enabled()) {
        run_flow_reverse_fused(z_p_nct, y_mask_nct, g_nct, frames, output);
        return;
    }
    std::vector<float> current(z_p_nct, z_p_nct + (size_t) model->inter_channels * frames);
    std::vector<float> next;
    const uint32_t requested_group_size = flow_group_size_env((uint32_t) model->flow.layers.size());
    for (int64_t i = (int64_t) model->flow.layers.size() - 1; i >= 0;) {
        uint32_t layer_count = std::min<uint32_t>(requested_group_size, (uint32_t) i + 1);
        run_flow_reverse_group((uint32_t) i, layer_count, current.data(), y_mask_nct, g_nct, frames, next);
        current.swap(next);
        i -= layer_count;
    }
    output = std::move(current);
}

style_bert_vits2_alignment_result style_bert_vits2_runner::build_decoder_latent(const float * logw_nct,
                                                                                 const float * x_mask_nct,
                                                                                 const float * m_p_nct,
                                                                                 const float * logs_p_nct,
                                                                                 const float * noise_nct,
                                                                                 const float * g_nct,
                                                                                 uint32_t tokens,
                                                                                 float length_scale,
                                                                                 float noise_scale,
                                                                                 std::vector<float> & decoder_z_nct) {
    const bool timings = debug_timings_enabled();
    const auto t_start = std::chrono::steady_clock::now();
    style_bert_vits2_alignment_result alignment =
        expand_alignment(logw_nct, x_mask_nct, m_p_nct, logs_p_nct, tokens, length_scale);
    const auto t_alignment = std::chrono::steady_clock::now();
    std::vector<float> z_p;
    sample_prior(alignment.m_p_expanded.data(),
                 alignment.logs_p_expanded.data(),
                 noise_nct,
                 alignment.frames,
                 noise_scale,
                 z_p);
    const auto t_prior = std::chrono::steady_clock::now();
    std::vector<float> z;
    run_flow_reverse(z_p.data(), alignment.y_mask.data(), g_nct, alignment.frames, z);
    const auto t_flow = std::chrono::steady_clock::now();

    decoder_z_nct.assign(z.size(), 0.0f);
    for (uint32_t c = 0; c < model->inter_channels; ++c) {
        for (uint32_t t = 0; t < alignment.frames; ++t) {
            decoder_z_nct[(size_t) c * alignment.frames + t] =
                z[(size_t) c * alignment.frames + t] * alignment.y_mask[t];
        }
    }
    const auto t_mask = std::chrono::steady_clock::now();
    if (timings) {
        std::cerr << "STYLE_BERT_VITS2_LATENT_TIMINGS"
                  << " tokens=" << tokens
                  << " frames=" << alignment.frames
                  << " alignment_ms=" << elapsed_ms(t_start, t_alignment)
                  << " prior_ms=" << elapsed_ms(t_alignment, t_prior)
                  << " flow_ms=" << elapsed_ms(t_prior, t_flow)
                  << " mask_ms=" << elapsed_ms(t_flow, t_mask)
                  << " total_ms=" << elapsed_ms(t_start, t_mask)
                  << std::endl;
    }
    return alignment;
}

ggml_cgraph * style_bert_vits2_runner::build_decoder_graph(uint32_t frames) {
    init_build();
    sctx->decoder_frames = frames;
    sctx->decoder_z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames, model->inter_channels, 1);
    ggml_set_input(sctx->decoder_z);
    sctx->set_tensor_backend(sctx->decoder_z);
    ggml_set_name(sctx->decoder_z, "style_bert_vits2_decoder_z");

    sctx->decoder_g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, model->gin_channels, 1);
    ggml_set_input(sctx->decoder_g);
    sctx->set_tensor_backend(sctx->decoder_g);
    ggml_set_name(sctx->decoder_g, "style_bert_vits2_decoder_g");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    const char * debug_stop_node = debug_output_node_name();
    bool debug_stop_found = false;
    ggml_tensor * output = build_decoder(ctx, &*model, sctx, debug_stop_node, &debug_stop_found);
    if (debug_stop_node && !debug_stop_found) {
        TTS_ABORT("STYLE_BERT_VITS2_DEBUG_OUTPUT_NODE='%s' was not found in decoder graph.\n", debug_stop_node);
    }
    ggml_build_forward_expand(gf, output);
    free_build();
    return gf;
}

void style_bert_vits2_runner::set_decoder_inputs(const float * decoder_z_nct, const float * decoder_g_nct, uint32_t frames) {
    if (sctx->decoder_z && sctx->decoder_z->buffer) {
        std::vector<float> z((size_t) frames * model->inter_channels);
        for (uint32_t c = 0; c < model->inter_channels; ++c) {
            for (uint32_t t = 0; t < frames; ++t) {
                z[(size_t) t + (size_t) frames * c] = decoder_z_nct[(size_t) c * frames + t];
            }
        }
        ggml_backend_tensor_set(sctx->decoder_z, z.data(), 0, z.size() * sizeof(float));
    }

    if (sctx->decoder_g && sctx->decoder_g->buffer) {
        std::vector<float> g(model->gin_channels);
        for (uint32_t c = 0; c < model->gin_channels; ++c) {
            g[c] = decoder_g_nct[c];
        }
        ggml_backend_tensor_set(sctx->decoder_g, g.data(), 0, g.size() * sizeof(float));
    }
}

void style_bert_vits2_runner::decode(const float * decoder_z_nct, const float * decoder_g_nct, uint32_t frames, tts_response & output) {
    const bool debug_timings = debug_timings_enabled();
    const auto t_start = std::chrono::steady_clock::now();
    ggml_backend_sched_reset(sctx->sched);
    ggml_cgraph * gf = build_decoder_graph(frames);
    const auto t_built = std::chrono::steady_clock::now();
    ggml_tensor * output_tensor = gf->nodes[gf->n_nodes - 1];
    const size_t output_size = (size_t) ggml_nelements(output_tensor) * sizeof(float);
    sctx->prep_output_buffer(output_size);
    output.data = (float *) ggml_backend_buffer_get_base(sctx->buf_output);
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    sctx->alloc_graph(gf, "style_bert_vits2.decoder");
    const auto t_alloc = std::chrono::steady_clock::now();
    set_decoder_inputs(decoder_z_nct, decoder_g_nct, frames);
    const auto t_inputs = std::chrono::steady_clock::now();
    ggml_backend_sched_graph_compute_async(sctx->sched, gf);
    const auto t_compute = std::chrono::steady_clock::now();
    sctx->get_ggml_node_data(output_tensor, output.data, output_size);
    const auto t_read = std::chrono::steady_clock::now();
    output.n_outputs = (size_t) ggml_nelements(output_tensor);
    output.hidden_size = 1;
    ggml_backend_sched_reset(sctx->sched);
    const auto t_done = std::chrono::steady_clock::now();

    if (debug_timings) {
        std::cerr << "STYLE_BERT_VITS2_TIMING phase=decoder"
                  << " backend=" << (sctx->backend ? ggml_backend_name(sctx->backend) : "CPU")
                  << " frames=" << frames
                  << " output_samples=" << output.n_outputs
                  << " nodes=" << gf->n_nodes
                  << " build_ms=" << elapsed_ms(t_start, t_built)
                  << " alloc_ms=" << elapsed_ms(t_built, t_alloc)
                  << " input_ms=" << elapsed_ms(t_alloc, t_inputs)
                  << " compute_submit_ms=" << elapsed_ms(t_inputs, t_compute)
                  << " read_ms=" << elapsed_ms(t_compute, t_read)
                  << " reset_ms=" << elapsed_ms(t_read, t_done)
                  << " total_ms=" << elapsed_ms(t_start, t_done)
                  << std::endl;
    }
}

void style_bert_vits2_runner::generate(const char *, tts_response &, const generation_configuration &) {
    TTS_ABORT("Style-Bert-VITS2 GGML currently exposes decoder inference only. Use the debug backend sidecar with explicit phones/tones.\n");
}

style_bert_vits2_context * build_new_style_bert_vits2_context(style_bert_vits2_model * model, int n_threads, bool use_cpu) {
    style_bert_vits2_context * sctx = new style_bert_vits2_context(model, n_threads);
    sctx->backend = tts_backend_init_accelerator(use_cpu);
    sctx->backend_cpu = ggml_backend_cpu_init();
    sctx->set_threads();
    const size_t max_nodes = style_bert_vits2_graph_max_nodes(model->max_nodes());
    sctx->build_schedule(max_nodes);
    sctx->buf_compute_meta.resize(ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false));
    return sctx;
}
