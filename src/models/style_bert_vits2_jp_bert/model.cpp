#include "model.h"

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string_view>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

namespace {
int required_key(gguf_context * meta, const char * key) {
    const int found = gguf_find_key(meta, key);
    if (found == -1) {
        TTS_ABORT("Missing Style-Bert-VITS2 JP BERT GGUF key '%s'\n", key);
    }
    return found;
}

uint32_t get_u32(gguf_context * meta, const char * key) {
    return gguf_get_val_u32(meta, required_key(meta, key));
}

int32_t get_i32(gguf_context * meta, const char * key) {
    return gguf_get_val_i32(meta, required_key(meta, key));
}

float get_f32(gguf_context * meta, const char * key) {
    return gguf_get_val_f32(meta, required_key(meta, key));
}

bool get_bool(gguf_context * meta, const char * key) {
    return gguf_get_val_bool(meta, required_key(meta, key));
}

std::string get_string(gguf_context * meta, const char * key) {
    return gguf_get_val_str(meta, required_key(meta, key));
}

std::string prefixed_key(const char * suffix) {
    return std::string(style_bert_vits2_jp_bert_model::arch) + "." + suffix;
}

int32_t make_log_bucket_position(int32_t relative_pos, uint32_t bucket_size, uint32_t max_position) {
    const int32_t sign = relative_pos < 0 ? -1 : 1;
    const int32_t mid = (int32_t) bucket_size / 2;
    const int32_t abs_rel = std::abs(relative_pos);
    const int32_t abs_pos = (relative_pos < mid && relative_pos > -mid) ? mid - 1 : abs_rel;
    const double numerator = std::log((double) abs_pos / (double) mid);
    const double denominator = std::log(((double) max_position - 1.0) / (double) mid);
    const int32_t log_pos = (int32_t) std::ceil(numerator / denominator * (double) (mid - 1)) + mid;
    return abs_pos <= mid ? relative_pos : log_pos * sign;
}

int32_t clamp_position_index(int32_t value, uint32_t span) {
    const int32_t low = 0;
    const int32_t high = (int32_t) span * 2 - 1;
    return std::max(low, std::min(value, high));
}

bool debug_shapes() {
    const char * env = std::getenv("STYLE_BERT_VITS2_JP_BERT_DEBUG_SHAPES");
    return env && env[0] && std::strcmp(env, "0") != 0;
}

void print_shape(const char * name, ggml_tensor * tensor) {
    if (!debug_shapes() || !tensor) {
        return;
    }
    std::fprintf(stderr,
                 "JP_BERT_SHAPE %s [%lld,%lld,%lld,%lld]\n",
                 name,
                 (long long) tensor->ne[0],
                 (long long) tensor->ne[1],
                 (long long) tensor->ne[2],
                 (long long) tensor->ne[3]);
}
}

void style_bert_vits2_jp_bert_build_relative_position_indices(uint32_t query_size, uint32_t key_size,
                                                              uint32_t bucket_size, uint32_t max_position,
                                                              std::vector<int32_t> & c2p_indices,
                                                              std::vector<int32_t> & p2c_indices) {
    const uint32_t span = bucket_size > 0 ? bucket_size : max_position;
    c2p_indices.resize((size_t) query_size * key_size);
    p2c_indices.resize((size_t) query_size * key_size);
    for (uint32_t q = 0; q < query_size; ++q) {
        for (uint32_t k = 0; k < key_size; ++k) {
            int32_t rel = (int32_t) q - (int32_t) k;
            if (bucket_size > 0 && max_position > 0) {
                rel = make_log_bucket_position(rel, bucket_size, max_position);
            }
            const size_t offset = (size_t) q * key_size + k;
            c2p_indices[offset] = clamp_position_index(rel + (int32_t) span, span);
            p2c_indices[offset] = clamp_position_index(-rel + (int32_t) span, span);
        }
    }
}

void style_bert_vits2_jp_bert_model::prep_constants(gguf_context * meta_ctx) {
    vocab_size = get_u32(meta_ctx, prefixed_key("vocab_size").c_str());
    hidden_size = get_u32(meta_ctx, prefixed_key("hidden_size").c_str());
    intermediate_size = get_u32(meta_ctx, prefixed_key("intermediate_size").c_str());
    n_layers = get_u32(meta_ctx, prefixed_key("layers").c_str());
    n_attn_heads = get_u32(meta_ctx, prefixed_key("attn_heads").c_str());
    head_size = get_u32(meta_ctx, prefixed_key("head_size").c_str());
    max_position_embeddings = get_u32(meta_ctx, prefixed_key("max_position_embeddings").c_str());
    position_buckets = get_u32(meta_ctx, prefixed_key("position_buckets").c_str());
    max_relative_positions = get_i32(meta_ctx, prefixed_key("max_relative_positions").c_str());
    type_vocab_size = get_u32(meta_ctx, prefixed_key("type_vocab_size").c_str());
    pad_token_id = get_u32(meta_ctx, prefixed_key("pad_token_id").c_str());
    layer_norm_eps = get_f32(meta_ctx, prefixed_key("layer_norm_eps").c_str());
    hidden_act = get_string(meta_ctx, prefixed_key("hidden_act").c_str());
    relative_attention = get_bool(meta_ctx, prefixed_key("relative_attention").c_str());
    position_biased_input = get_bool(meta_ctx, prefixed_key("position_biased_input").c_str());
    share_att_key = get_bool(meta_ctx, prefixed_key("share_att_key").c_str());
    feature_hidden_state_offset = get_u32(meta_ctx, prefixed_key("feature_hidden_state_offset").c_str());
}

void style_bert_vits2_jp_bert_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
    prep_constants(meta_ctx);
    tts_model::setup_from_file(meta_ctx, load_context, cpu_only, arch, 1.35f);
}

void style_bert_vits2_jp_bert_model::assign_weight(const char * name, ggml_tensor & tensor) {
    const std::string key{name};
    if (tensors.contains(key)) {
        TTS_ABORT("Duplicate Style-Bert-VITS2 JP BERT tensor '%s'\n", name);
    }
    ggml_tensor * copy = ggml_dup_tensor(ctx, &tensor);
    set_tensor(copy, &tensor);
    tensors.emplace(key, copy);
}

bool style_bert_vits2_jp_bert_model::has_tensor(const char * name) const {
    return tensors.find(name) != tensors.end();
}

ggml_tensor * style_bert_vits2_jp_bert_model::tensor(const char * name) const {
    const auto found = tensors.find(name);
    if (found == tensors.end()) {
        TTS_ABORT("Missing Style-Bert-VITS2 JP BERT tensor '%s'\n", name);
    }
    return found->second;
}

style_bert_vits2_jp_bert_context::style_bert_vits2_jp_bert_context(style_bert_vits2_jp_bert_model * model,
                                                                   int n_threads):
    runner_context(n_threads), model(model) {}

void style_bert_vits2_jp_bert_context::build_schedule() {
    runner_context::build_schedule(std::max<size_t>(model->max_nodes() * 64, 32768));
}

style_bert_vits2_jp_bert_runner::style_bert_vits2_jp_bert_runner(
    std::unique_ptr<style_bert_vits2_jp_bert_model> model,
    style_bert_vits2_jp_bert_context * context):
    tts_generation_runner{style_bert_vits2_jp_bert_loader}, model{std::move(model)}, bctx(context) {}

style_bert_vits2_jp_bert_runner::~style_bert_vits2_jp_bert_runner() {
    delete bctx;
    model->free();
}

void style_bert_vits2_jp_bert_runner::assign_weight(const char * name, ggml_tensor & tensor) {
    const std::string_view name_sv{name};
    constexpr std::string_view prefix{"style-bert-vits2-jp-bert."};
    if (!name_sv.starts_with(prefix)) {
        TTS_ABORT("Unexpected Style-Bert-VITS2 JP BERT tensor prefix '%s'\n", name);
    }
    const std::string trimmed{name_sv.substr(prefix.size())};
    model->assign_weight(trimmed.c_str(), tensor);
}

void style_bert_vits2_jp_bert_runner::prepare_post_load() {
    const char * required[] = {
        "emb.word.weight",
        "emb.norm.weight",
        "emb.norm.bias",
        "layers.0.attn.self.query.weight",
        "layers.0.attn.self.key.weight",
        "layers.0.attn.self.value.weight",
        "layers.0.attn.out.dense.weight",
        "layers.0.attn.out.norm.weight",
        "layers.0.intermediate.dense.weight",
        "layers.0.output.dense.weight",
        "layers.0.output.norm.weight",
        "enc.rel_embeddings.weight",
        "enc.norm.weight",
        "enc.norm.bias",
        "enc.conv.conv.weight",
        "enc.conv.conv.bias",
        "enc.conv.norm.weight",
        "enc.conv.norm.bias",
    };
    for (const char * name : required) {
        if (!model->has_tensor(name)) {
            TTS_ABORT("Style-Bert-VITS2 JP BERT GGUF did not load required tensor '%s'\n", name);
        }
    }
}

namespace {
ggml_tensor * jp_bert_layer_norm(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias,
                                 float eps) {
    ggml_tensor * cur = ggml_norm(ctx, input, eps);
    cur = ggml_mul(ctx, cur, weight);
    return ggml_add(ctx, cur, bias);
}

ggml_tensor * jp_bert_linear(ggml_context * ctx, style_bert_vits2_jp_bert_model * model, ggml_tensor * input,
                             const std::string & weight, const std::string & bias) {
    ggml_tensor * cur = ggml_mul_mat(ctx, model->tensor(weight.c_str()), input);
    return ggml_add(ctx, cur, model->tensor(bias.c_str()));
}

ggml_tensor * jp_bert_view_head_channels(ggml_context * ctx, ggml_tensor * channels_time, uint32_t head,
                                         uint32_t head_size) {
    return ggml_view_2d(ctx,
                        channels_time,
                        head_size,
                        channels_time->ne[1],
                        channels_time->nb[1],
                        (size_t) head * head_size * channels_time->nb[0]);
}

ggml_tensor * jp_bert_view_value_head(ggml_context * ctx, ggml_tensor * time_channels, uint32_t head,
                                      uint32_t head_size) {
    return ggml_view_2d(ctx,
                        time_channels,
                        time_channels->ne[0],
                        head_size,
                        time_channels->nb[1],
                        (size_t) head * head_size * time_channels->nb[1]);
}

ggml_tensor * jp_bert_gather_flat_scores(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * flat_indices,
                                         uint32_t rows, uint32_t cols) {
    ggml_tensor * flat = ggml_reshape_2d(ctx, ggml_cont(ctx, logits), 1, logits->ne[0] * logits->ne[1]);
    ggml_tensor * gathered = ggml_get_rows(ctx, flat, flat_indices);
    return ggml_reshape_2d(ctx, gathered, rows, cols);
}

ggml_tensor * jp_bert_embeddings(style_bert_vits2_jp_bert_runner * runner, uint32_t tokens) {
    ggml_context * ctx = runner->ctx;
    ggml_tensor * cur = ggml_get_rows(ctx, runner->model->tensor("emb.word.weight"), runner->bctx->input_ids);
    ggml_set_name(cur, "style_bert_vits2_jp_bert.emb.word");
    cur = jp_bert_layer_norm(ctx,
                             cur,
                             runner->model->tensor("emb.norm.weight"),
                             runner->model->tensor("emb.norm.bias"),
                             runner->model->layer_norm_eps);
    ggml_set_name(cur, "style_bert_vits2_jp_bert.embeddings");
    GGML_ASSERT(cur->ne[0] == runner->model->hidden_size);
    GGML_ASSERT(cur->ne[1] == tokens);
    return cur;
}

std::string layer_key(uint32_t layer, const char * suffix) {
    return "layers." + std::to_string(layer) + "." + suffix;
}

ggml_tensor * jp_bert_layer_attention(style_bert_vits2_jp_bert_runner * runner, ggml_tensor * input,
                                      uint32_t tokens, uint32_t layer) {
    ggml_context * ctx = runner->ctx;
    style_bert_vits2_jp_bert_model * model = runner->model.get();
    const uint32_t heads = model->n_attn_heads;
    const uint32_t head_size = model->head_size;
    const uint32_t span = model->position_buckets > 0
                              ? model->position_buckets
                              : (model->max_relative_positions < 1
                                     ? model->max_position_embeddings
                                     : (uint32_t) model->max_relative_positions);
    const float inv_scale = 1.0f / std::sqrt((float) head_size * 3.0f);

    ggml_tensor * q = jp_bert_linear(ctx, model, input,
                                     layer_key(layer, "attn.self.query.weight"),
                                     layer_key(layer, "attn.self.query.bias"));
    ggml_tensor * k = jp_bert_linear(ctx, model, input,
                                     layer_key(layer, "attn.self.key.weight"),
                                     layer_key(layer, "attn.self.key.bias"));
    ggml_tensor * v = jp_bert_linear(ctx, model, input,
                                     layer_key(layer, "attn.self.value.weight"),
                                     layer_key(layer, "attn.self.value.bias"));
    ggml_tensor * v_time_channels = ggml_cont(ctx, ggml_transpose(ctx, v));
    print_shape("input", input);
    print_shape("q", q);
    print_shape("v_time_channels", v_time_channels);

    ggml_tensor * rel = jp_bert_layer_norm(ctx,
                                           model->tensor("enc.rel_embeddings.weight"),
                                           model->tensor("enc.norm.weight"),
                                           model->tensor("enc.norm.bias"),
                                           model->layer_norm_eps);
    ggml_tensor * pos_query = jp_bert_linear(ctx, model, rel,
                                             layer_key(layer, "attn.self.query.weight"),
                                             layer_key(layer, "attn.self.query.bias"));
    ggml_tensor * pos_key = jp_bert_linear(ctx, model, rel,
                                           layer_key(layer, "attn.self.key.weight"),
                                           layer_key(layer, "attn.self.key.bias"));

    ggml_tensor * merged_heads = nullptr;
    for (uint32_t head = 0; head < heads; ++head) {
        ggml_tensor * q_head = jp_bert_view_head_channels(ctx, q, head, head_size);
        ggml_tensor * k_head = jp_bert_view_head_channels(ctx, k, head, head_size);
        ggml_tensor * v_head = jp_bert_view_value_head(ctx, v_time_channels, head, head_size);
        ggml_tensor * pos_query_head = jp_bert_view_head_channels(ctx, pos_query, head, head_size);
        ggml_tensor * pos_key_head = jp_bert_view_head_channels(ctx, pos_key, head, head_size);
        if (head == 0) {
            print_shape("q_head", q_head);
            print_shape("k_head", k_head);
            print_shape("v_head", v_head);
            print_shape("pos_query_head", pos_query_head);
        }

        ggml_tensor * scores = ggml_mul_mat(ctx, k_head, q_head);
        ggml_tensor * c2p_logits = ggml_mul_mat(ctx, pos_key_head, q_head);
        ggml_tensor * c2p = jp_bert_gather_flat_scores(ctx, c2p_logits, runner->bctx->c2p_flat_indices, tokens, tokens);
        scores = ggml_add(ctx, scores, c2p);

        ggml_tensor * p2c_logits = ggml_mul_mat(ctx, pos_query_head, k_head);
        ggml_tensor * p2c = jp_bert_gather_flat_scores(ctx, p2c_logits, runner->bctx->p2c_flat_indices, tokens, tokens);
        scores = ggml_add(ctx, scores, p2c);

        scores = ggml_scale(ctx, scores, inv_scale);
        ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);
        ggml_tensor * value = ggml_mul_mat(ctx, probs, v_head);
        value = ggml_cont(ctx, ggml_transpose(ctx, value));
        if (head == 0) {
            print_shape("scores", scores);
            print_shape("probs", probs);
            print_shape("value", value);
        }
        merged_heads = merged_heads ? ggml_concat(ctx, merged_heads, value, 0) : value;
    }

    ggml_tensor * context = merged_heads;
    print_shape("merged_heads", merged_heads);
    print_shape("context", context);
    ggml_tensor * projected = jp_bert_linear(ctx, model, context,
                                             layer_key(layer, "attn.out.dense.weight"),
                                             layer_key(layer, "attn.out.dense.bias"));
    ggml_tensor * residual = ggml_add(ctx, projected, input);
    return jp_bert_layer_norm(ctx,
                              residual,
                              model->tensor(layer_key(layer, "attn.out.norm.weight").c_str()),
                              model->tensor(layer_key(layer, "attn.out.norm.bias").c_str()),
                              model->layer_norm_eps);
}

ggml_tensor * jp_bert_layer_ffn(style_bert_vits2_jp_bert_runner * runner, ggml_tensor * input, uint32_t layer) {
    ggml_context * ctx = runner->ctx;
    style_bert_vits2_jp_bert_model * model = runner->model.get();
    ggml_tensor * cur = jp_bert_linear(ctx, model, input,
                                       layer_key(layer, "intermediate.dense.weight"),
                                       layer_key(layer, "intermediate.dense.bias"));
    cur = ggml_gelu_erf(ctx, cur);
    cur = jp_bert_linear(ctx, model, cur,
                         layer_key(layer, "output.dense.weight"),
                         layer_key(layer, "output.dense.bias"));
    cur = ggml_add(ctx, cur, input);
    return jp_bert_layer_norm(ctx,
                              cur,
                              model->tensor(layer_key(layer, "output.norm.weight").c_str()),
                              model->tensor(layer_key(layer, "output.norm.bias").c_str()),
                              model->layer_norm_eps);
}

ggml_tensor * jp_bert_layer0_conv(style_bert_vits2_jp_bert_runner * runner, ggml_tensor * embeddings,
                                  ggml_tensor * residual_states, uint32_t tokens) {
    ggml_context * ctx = runner->ctx;
    style_bert_vits2_jp_bert_model * model = runner->model.get();
    ggml_tensor * conv_input = ggml_cont(ctx, ggml_transpose(ctx, embeddings));
    conv_input = ggml_reshape_3d(ctx, conv_input, tokens, model->hidden_size, 1);
    ggml_tensor * out = ggml_conv_1d(ctx, model->tensor("enc.conv.conv.weight"), conv_input, 1, 1, 1);
    ggml_tensor * conv_bias = ggml_reshape_3d(ctx, model->tensor("enc.conv.conv.bias"), 1, model->hidden_size, 1);
    out = ggml_add(ctx, out, conv_bias);
    out = ggml_gelu_erf(ctx, out);
    out = ggml_reshape_2d(ctx, out, tokens, model->hidden_size);
    out = ggml_cont(ctx, ggml_transpose(ctx, out));
    out = ggml_add(ctx, residual_states, out);
    return jp_bert_layer_norm(ctx,
                              out,
                              model->tensor("enc.conv.norm.weight"),
                              model->tensor("enc.conv.norm.bias"),
                              model->layer_norm_eps);
}
}

ggml_cgraph * style_bert_vits2_jp_bert_runner::build_embedding_graph(uint32_t tokens) {
    init_build(&bctx->buf_compute_meta);

    bctx->input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(bctx->input_ids);
    bctx->set_tensor_backend(bctx->input_ids);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor * cur = jp_bert_embeddings(this, tokens);

    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void style_bert_vits2_jp_bert_runner::set_embedding_inputs(const int32_t * input_ids, uint32_t tokens) {
    ggml_backend_tensor_set(bctx->input_ids, input_ids, 0, (size_t) tokens * sizeof(int32_t));
}

void style_bert_vits2_jp_bert_runner::encode_embeddings(const int32_t * input_ids, uint32_t tokens,
                                                        std::vector<float> & output) {
    if (tokens == 0) {
        output.clear();
        return;
    }
    ggml_backend_sched_reset(bctx->sched);
    const size_t output_size = (size_t) tokens * model->hidden_size * sizeof(float);
    bctx->prep_output_buffer(output_size);

    ggml_cgraph * gf = build_embedding_graph(tokens);
    ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    bctx->alloc_graph(gf, "style_bert_vits2_jp_bert.embeddings");
    set_embedding_inputs(input_ids, tokens);
    ggml_backend_sched_graph_compute_async(bctx->sched, gf);

    output.resize((size_t) tokens * model->hidden_size);
    bctx->get_ggml_node_data(result, output.data(), output_size);
    ggml_backend_sched_reset(bctx->sched);
}

ggml_cgraph * style_bert_vits2_jp_bert_runner::build_layer0_graph(uint32_t tokens) {
    init_build(&bctx->buf_compute_meta);

    bctx->input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(bctx->input_ids);
    bctx->set_tensor_backend(bctx->input_ids);
    bctx->c2p_flat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) tokens * tokens);
    ggml_set_input(bctx->c2p_flat_indices);
    bctx->set_tensor_backend(bctx->c2p_flat_indices);
    bctx->p2c_flat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) tokens * tokens);
    ggml_set_input(bctx->p2c_flat_indices);
    bctx->set_tensor_backend(bctx->p2c_flat_indices);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_tensor * cur = jp_bert_embeddings(this, tokens);
    cur = jp_bert_layer_attention(this, cur, tokens, 0);
    cur = jp_bert_layer_ffn(this, cur, 0);
    ggml_set_name(cur, "style_bert_vits2_jp_bert.layer0");

    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void style_bert_vits2_jp_bert_runner::set_layer_inputs(const int32_t * input_ids, uint32_t tokens) {
    ggml_backend_tensor_set(bctx->input_ids, input_ids, 0, (size_t) tokens * sizeof(int32_t));
    std::vector<int32_t> c2p;
    std::vector<int32_t> p2c;
    const uint32_t max_position = model->max_relative_positions < 1
                                      ? model->max_position_embeddings
                                      : (uint32_t) model->max_relative_positions;
    const uint32_t span = model->position_buckets > 0 ? model->position_buckets : max_position;
    style_bert_vits2_jp_bert_build_relative_position_indices(tokens, tokens, model->position_buckets, max_position,
                                                             c2p, p2c);
    std::vector<int32_t> c2p_flat(c2p.size());
    std::vector<int32_t> p2c_flat(p2c.size());
    const int32_t rel_rows = (int32_t) span * 2;
    for (uint32_t q = 0; q < tokens; ++q) {
        for (uint32_t k = 0; k < tokens; ++k) {
            const size_t offset = (size_t) q * tokens + k;
            c2p_flat[offset] = c2p[offset] + rel_rows * (int32_t) q;
            p2c_flat[offset] = p2c[(size_t) k * tokens + q] + rel_rows * (int32_t) k;
        }
    }
    ggml_backend_tensor_set(bctx->c2p_flat_indices, c2p_flat.data(), 0, c2p_flat.size() * sizeof(int32_t));
    ggml_backend_tensor_set(bctx->p2c_flat_indices, p2c_flat.data(), 0, p2c_flat.size() * sizeof(int32_t));
}

void style_bert_vits2_jp_bert_runner::encode_layer0(const int32_t * input_ids, uint32_t tokens,
                                                    std::vector<float> & output) {
    if (tokens == 0) {
        output.clear();
        return;
    }
    ggml_backend_sched_reset(bctx->sched);
    const size_t output_size = (size_t) tokens * model->hidden_size * sizeof(float);
    bctx->prep_output_buffer(output_size);

    ggml_cgraph * gf = build_layer0_graph(tokens);
    ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    bctx->alloc_graph(gf, "style_bert_vits2_jp_bert.layer0");
    set_layer_inputs(input_ids, tokens);
    ggml_backend_sched_graph_compute_async(bctx->sched, gf);

    output.resize((size_t) tokens * model->hidden_size);
    bctx->get_ggml_node_data(result, output.data(), output_size);
    ggml_backend_sched_reset(bctx->sched);
}

ggml_cgraph * style_bert_vits2_jp_bert_runner::build_layer0_with_conv_graph(uint32_t tokens) {
    init_build(&bctx->buf_compute_meta);

    bctx->input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(bctx->input_ids);
    bctx->set_tensor_backend(bctx->input_ids);
    bctx->c2p_flat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) tokens * tokens);
    ggml_set_input(bctx->c2p_flat_indices);
    bctx->set_tensor_backend(bctx->c2p_flat_indices);
    bctx->p2c_flat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) tokens * tokens);
    ggml_set_input(bctx->p2c_flat_indices);
    bctx->set_tensor_backend(bctx->p2c_flat_indices);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_tensor * embeddings = jp_bert_embeddings(this, tokens);
    ggml_tensor * cur = jp_bert_layer_attention(this, embeddings, tokens, 0);
    cur = jp_bert_layer_ffn(this, cur, 0);
    cur = jp_bert_layer0_conv(this, embeddings, cur, tokens);
    ggml_set_name(cur, "style_bert_vits2_jp_bert.layer0_with_conv");

    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void style_bert_vits2_jp_bert_runner::encode_layer0_with_conv(const int32_t * input_ids, uint32_t tokens,
                                                              std::vector<float> & output) {
    if (tokens == 0) {
        output.clear();
        return;
    }
    ggml_backend_sched_reset(bctx->sched);
    const size_t output_size = (size_t) tokens * model->hidden_size * sizeof(float);
    bctx->prep_output_buffer(output_size);

    ggml_cgraph * gf = build_layer0_with_conv_graph(tokens);
    ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    bctx->alloc_graph(gf, "style_bert_vits2_jp_bert.layer0_with_conv");
    set_layer_inputs(input_ids, tokens);
    ggml_backend_sched_graph_compute_async(bctx->sched, gf);

    output.resize((size_t) tokens * model->hidden_size);
    bctx->get_ggml_node_data(result, output.data(), output_size);
    ggml_backend_sched_reset(bctx->sched);
}

ggml_cgraph * style_bert_vits2_jp_bert_runner::build_feature_graph(uint32_t tokens) {
    init_build(&bctx->buf_compute_meta);

    bctx->input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_set_input(bctx->input_ids);
    bctx->set_tensor_backend(bctx->input_ids);
    bctx->c2p_flat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) tokens * tokens);
    ggml_set_input(bctx->c2p_flat_indices);
    bctx->set_tensor_backend(bctx->c2p_flat_indices);
    bctx->p2c_flat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) tokens * tokens);
    ggml_set_input(bctx->p2c_flat_indices);
    bctx->set_tensor_backend(bctx->p2c_flat_indices);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, std::max<size_t>(32768, model->max_nodes() * 64), false);
    ggml_tensor * embeddings = jp_bert_embeddings(this, tokens);
    ggml_tensor * cur = embeddings;
    uint32_t layers_to_run = model->n_layers;
    if (model->feature_hidden_state_offset > 0 && model->feature_hidden_state_offset < model->n_layers) {
        layers_to_run = model->n_layers - model->feature_hidden_state_offset;
    }
    for (uint32_t layer = 0; layer < layers_to_run; ++layer) {
        cur = jp_bert_layer_attention(this, cur, tokens, layer);
        cur = jp_bert_layer_ffn(this, cur, layer);
        if (layer == 0) {
            cur = jp_bert_layer0_conv(this, embeddings, cur, tokens);
        }
    }
    ggml_set_name(cur, "style_bert_vits2_jp_bert.features");

    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void style_bert_vits2_jp_bert_runner::encode_features(const int32_t * input_ids, uint32_t tokens,
                                                      std::vector<float> & output) {
    if (tokens == 0) {
        output.clear();
        return;
    }
    ggml_backend_sched_reset(bctx->sched);
    const size_t output_size = (size_t) tokens * model->hidden_size * sizeof(float);
    bctx->prep_output_buffer(output_size);

    ggml_cgraph * gf = build_feature_graph(tokens);
    ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    bctx->alloc_graph(gf, "style_bert_vits2_jp_bert.features");
    set_layer_inputs(input_ids, tokens);
    ggml_backend_sched_graph_compute_async(bctx->sched, gf);

    output.resize((size_t) tokens * model->hidden_size);
    bctx->get_ggml_node_data(result, output.data(), output_size);
    ggml_backend_sched_reset(bctx->sched);
}

void style_bert_vits2_jp_bert_runner::generate(const char *, tts_response &, const generation_configuration &) {
    TTS_ABORT("Style-Bert-VITS2 JP BERT generation is not implemented yet; use the feature runner entrypoint once wired.\n");
}
