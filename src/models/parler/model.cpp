#include "model.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <map>

static constexpr uint32_t PARLER_MIN_DECODABLE_AUDIO_TOKENS = 12;

static bool parler_debug_tokens_enabled() {
    const char * env = std::getenv("PARLER_DEBUG_TOKENS");
    return env && env[0] && std::strcmp(env, "0") != 0;
}

static bool parler_debug_timings_enabled() {
    const char * env = std::getenv("PARLER_DEBUG_TIMINGS");
    return env && env[0] && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

static bool parler_env_truthy(const char * env) {
    return env && env[0] && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

static bool parler_flash_attn_mode_enabled(const char * needle) {
    const char * env = std::getenv("PARLER_USE_FLASH_ATTN");
    if (!parler_env_truthy(env)) {
        return false;
    }
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "all") == 0) {
        return true;
    }
    return std::strstr(env, needle) != nullptr;
}

static bool parler_fused_heads_enabled() {
    const char * env = std::getenv("PARLER_FUSE_HEADS");
    if (!env || !env[0]) {
        return true;
    }
    return parler_env_truthy(env);
}

static bool parler_fused_qkv_enabled() {
    return parler_env_truthy(std::getenv("PARLER_FUSE_QKV"));
}

static bool parler_fused_qkv_direct_views_enabled() {
    return parler_env_truthy(std::getenv("PARLER_FUSE_QKV_DIRECT_VIEWS"));
}

static bool parler_profile_nodes_enabled() {
    return parler_env_truthy(std::getenv("PARLER_PROFILE_NODES"));
}

static bool parler_prefill_last_logits_enabled() {
    const char * env = std::getenv("PARLER_PREFILL_LAST_LOGITS");
    if (!env || !env[0]) {
        return false;
    }
    return parler_env_truthy(env);
}

static bool parler_prefill_last_logits_only(const parler_ubatch & batch) {
    return parler_prefill_last_logits_enabled() &&
           batch.audio_generation &&
           batch.n_tokens > 0 &&
           batch.sequence_length > 1;
}

static bool parler_mlp_gelu_inplace_enabled() {
    const char * env = std::getenv("PARLER_MLP_GELU_INPLACE");
    if (!env || !env[0]) {
        return true;
    }
    return parler_env_truthy(env);
}

static bool parler_mlp_gelu_graph_output_enabled() {
    return parler_env_truthy(std::getenv("PARLER_MLP_GELU_GRAPH_OUTPUT"));
}

static bool parler_precompute_cross_v_flash_enabled() {
    const char * env = std::getenv("PARLER_PRECOMPUTE_CROSS_V_FLASH");
    if (!env || !env[0]) {
        return true;
    }
    return parler_env_truthy(env);
}

static bool parler_self_v_flash_cache_enabled() {
    const char * env = std::getenv("PARLER_SELF_V_FLASH_CACHE");
    if (!env || !env[0]) {
        return parler_flash_attn_mode_enabled("self");
    }
    return parler_env_truthy(env);
}

static bool parler_self_v_flash_only_enabled() {
    const char * env = std::getenv("PARLER_SELF_V_FLASH_ONLY");
    if (!env || !env[0]) {
        return parler_self_v_flash_cache_enabled();
    }
    return parler_env_truthy(env);
}

static bool parler_fp16_activations_enabled() {
    return parler_env_truthy(std::getenv("PARLER_FP16_ACTIVATIONS"));
}

static ggml_type parler_dtype_from_env(const char * env_name, ggml_type fallback) {
    const char * env = std::getenv(env_name);
    if (!env || !env[0] || std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0) {
        return fallback;
    }
    if (std::strcmp(env, "f32") == 0 || std::strcmp(env, "fp32") == 0 || std::strcmp(env, "float32") == 0) {
        return GGML_TYPE_F32;
    }
    if (std::strcmp(env, "f16") == 0 || std::strcmp(env, "fp16") == 0 || std::strcmp(env, "float16") == 0) {
        return GGML_TYPE_F16;
    }
    if (std::strcmp(env, "bf16") == 0 || std::strcmp(env, "bfloat16") == 0) {
        return GGML_TYPE_BF16;
    }
    fprintf(stderr, "Warning: ignoring unsupported %s=%s; expected f32, f16, or bf16.\n", env_name, env);
    return fallback;
}

static ggml_type parler_activation_dtype() {
    ggml_type fallback = parler_fp16_activations_enabled() ? GGML_TYPE_F16 : GGML_TYPE_F32;
    return parler_dtype_from_env("PARLER_ACTIVATION_DTYPE", fallback);
}

static bool parler_activation_scope_enabled(const char * scope) {
    const char * env = std::getenv("PARLER_ACTIVATION_DTYPE_SCOPE");
    if (!env || !env[0] || std::strcmp(env, "all") == 0 || std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0) {
        return true;
    }
    if (std::strcmp(env, "none") == 0 || std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0) {
        return false;
    }

    const std::string scopes(env);
    size_t start = 0;
    while (start <= scopes.size()) {
        size_t end = scopes.find(',', start);
        if (end == std::string::npos) {
            end = scopes.size();
        }
        std::string item = scopes.substr(start, end - start);
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }), item.end());
        if (item == scope) {
            return true;
        }
        if (end == scopes.size()) {
            break;
        }
        start = end + 1;
    }
    return false;
}

static ggml_type parler_kv_cache_dtype() {
    return parler_dtype_from_env("PARLER_KV_CACHE_TYPE", GGML_TYPE_F32);
}

static ggml_type parler_cross_kv_dtype() {
    return parler_dtype_from_env("PARLER_CROSS_KV_TYPE", GGML_TYPE_F32);
}

static ggml_type parler_fused_heads_dtype(ggml_type fallback) {
    return parler_dtype_from_env("PARLER_FUSED_HEADS_TYPE", fallback);
}

static ggml_type parler_sanitize_accelerator_dtype(ggml_type type, ggml_backend_t backend, const char * label) {
    if (type != GGML_TYPE_BF16 || !backend) {
        return type;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        static std::map<std::string, bool> warned_labels;
        if (!warned_labels[label]) {
            warned_labels[label] = true;
            fprintf(stderr,
                    "Warning: %s=bf16 is disabled on accelerator backend %s; falling back to f32. "
                    "Use f16 for accelerator-side reduced precision.\n",
                    label,
                    ggml_backend_name(backend));
        }
        return GGML_TYPE_F32;
    }
    return type;
}

static bool parler_gpu_argmax_enabled() {
    const char * env = std::getenv("PARLER_GPU_ARGMAX");
    if (!env || !env[0]) {
        return true;
    }
    return parler_env_truthy(env);
}

static uint32_t parler_requested_unroll_steps() {
    const char * env = std::getenv("PARLER_UNROLL_STEPS");
    if (!env || !env[0]) {
        return 1;
    }
    const int value = std::atoi(env);
    if (value <= 1) {
        return 1;
    }
    return (uint32_t) std::min(value, 32);
}

static size_t parler_decode_graph_max_nodes(parler_tts_model * model, uint32_t unroll_steps = 1) {
    return std::max<size_t>(model->max_nodes(), model->max_nodes() * std::max<uint32_t>(1, unroll_steps));
}

static uint32_t parler_eos_tail_step_for_head(const parler_tts_model * model, uint32_t head) {
    return model->max_generation_size > model->n_output_heads
        ? model->max_generation_size - model->n_output_heads + 2 + head
        : head + 1;
}

static ggml_tensor * parler_maybe_cast_activation(ggml_context * ctx, const parler_context * pctx, ggml_tensor * tensor, const char * scope = "all") {
    if (!parler_activation_scope_enabled(scope)) {
        return tensor;
    }
    const ggml_type target = parler_sanitize_accelerator_dtype(
        parler_activation_dtype(),
        pctx ? pctx->backend : nullptr,
        "PARLER_ACTIVATION_DTYPE");
    if (target == GGML_TYPE_F32 || tensor->type != GGML_TYPE_F32) {
        return tensor;
    }
    return ggml_cast(ctx, tensor, target);
}

static bool parler_self_attn_mask_required(const parler_context * pctx, const parler_ubatch & batch) {
    if (batch.sequence_length != 1) {
        return true;
    }
    if (!batch.audio_generation || !batch.positions) {
        return true;
    }

    // Incremental generation decodes the newest single token against the cache.
    // There are no future keys in the cache, so the causal mask is all zeros.
    return batch.positions[0] != pctx->current_position;
}

static uint32_t parler_profile_node_steps() {
    const char * env = std::getenv("PARLER_PROFILE_NODE_STEPS");
    if (!env || !env[0]) {
        return 1;
    }
    const int value = std::atoi(env);
    return value > 0 ? (uint32_t) value : 1;
}

static uint32_t parler_profile_node_skip() {
    const char * env = std::getenv("PARLER_PROFILE_NODE_SKIP");
    if (!env || !env[0]) {
        return 0;
    }
    const int value = std::atoi(env);
    return value > 0 ? (uint32_t) value : 0;
}

static size_t parler_pad_to_alignment(size_t offset, size_t alignment) {
    if (alignment == 0) {
        return offset;
    }
    return (offset + alignment - 1) & ~(alignment - 1);
}

static double parler_elapsed_ms(std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct parler_tts_runner::decode_timing {
    uint32_t steps = 0;
    uint32_t prompt_steps = 0;
    uint32_t audio_steps = 0;
    double reserve_ms = 0.0;
    double build_ms = 0.0;
    double alloc_ms = 0.0;
    double input_ms = 0.0;
    double submit_ms = 0.0;
    double read_ms = 0.0;
    double reset_ms = 0.0;
    double total_ms = 0.0;
};

struct parler_node_profile_entry {
    ggml_tensor * node = nullptr;
    double total_ms = 0.0;
    uint32_t samples = 0;
};

struct parler_node_profile_state {
    std::vector<parler_node_profile_entry> entries;
    ggml_tensor * active_node = nullptr;
    std::chrono::steady_clock::time_point active_start;
    uint32_t step = 0;
};

static int parler_profile_node_index(const parler_node_profile_state * state, ggml_tensor * node) {
    for (size_t i = 0; i < state->entries.size(); ++i) {
        if (state->entries[i].node == node) {
            return (int) i;
        }
    }
    return -1;
}

static bool parler_node_profile_callback(ggml_tensor * node, bool ask, void * user_data) {
    auto * state = static_cast<parler_node_profile_state *>(user_data);
    if (ask) {
        state->active_node = node;
        state->active_start = std::chrono::steady_clock::now();
        // Force node-by-node execution so the follow-up callback gives a useful timing boundary.
        return true;
    }

    const auto t_done = std::chrono::steady_clock::now();
    const int index = parler_profile_node_index(state, node);
    if (index >= 0) {
        state->entries[(size_t) index].total_ms += parler_elapsed_ms(state->active_start, t_done);
        state->entries[(size_t) index].samples += 1;
    }
    state->active_node = nullptr;
    return true;
}

static void parler_print_node_profile(const parler_node_profile_state & state) {
    std::vector<size_t> order;
    order.reserve(state.entries.size());
    double total_ms = 0.0;
    std::map<std::string, double> op_total_ms;
    std::map<std::string, uint32_t> op_samples;

    for (size_t i = 0; i < state.entries.size(); ++i) {
        const parler_node_profile_entry & entry = state.entries[i];
        if (entry.samples == 0) {
            continue;
        }
        order.push_back(i);
        total_ms += entry.total_ms;
        const char * op = ggml_op_name(entry.node->op);
        op_total_ms[op] += entry.total_ms;
        op_samples[op] += entry.samples;
    }

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return state.entries[a].total_ms > state.entries[b].total_ms;
    });

    std::vector<std::pair<std::string, double>> op_order;
    op_order.reserve(op_total_ms.size());
    for (const auto & item : op_total_ms) {
        op_order.push_back(item);
    }
    std::sort(op_order.begin(), op_order.end(), [](const auto & a, const auto & b) {
        return a.second > b.second;
    });

    fprintf(stderr, "PARLER_NODE_PROFILE step=%u nodes=%zu observed=%zu total_ms=%.3f by_op=",
            state.step, state.entries.size(), order.size(), total_ms);
    const size_t op_limit = std::min<size_t>(op_order.size(), 12);
    for (size_t i = 0; i < op_limit; ++i) {
        const std::string & op = op_order[i].first;
        if (i > 0) {
            fputc(',', stderr);
        }
        fprintf(stderr, "%s:%.3f/%u", op.c_str(), op_order[i].second, op_samples[op]);
    }
    fputc('\n', stderr);

    const size_t node_limit = std::min<size_t>(order.size(), 40);
    for (size_t rank = 0; rank < node_limit; ++rank) {
        const parler_node_profile_entry & entry = state.entries[order[rank]];
        const ggml_tensor * node = entry.node;
        fprintf(stderr,
                "PARLER_NODE_PROFILE_NODE step=%u rank=%zu ms=%.3f samples=%u op=%s name=%s type=%s "
                "ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                state.step,
                rank + 1,
                entry.total_ms,
                entry.samples,
                ggml_op_name(node->op),
                ggml_get_name(node),
                ggml_type_name(node->type),
                node->ne[0],
                node->ne[1],
                node->ne[2],
                node->ne[3]);
    }
}

static void parler_debug_print_tokens(const char * label, const std::vector<uint32_t> & tokens, uint32_t n_heads) {
    if (!parler_debug_tokens_enabled()) {
        return;
    }
    fprintf(stderr, "PARLER_DEBUG_%s count=%zu heads=%u tokens=", label, tokens.size(), n_heads);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            fputc(',', stderr);
        }
        fprintf(stderr, "%u", tokens[i]);
    }
    fputc('\n', stderr);
}

// For loading parler model from gguf file.
static const std::map<std::string, parler_tensor> PARLER_TENSOR_GGUF_LOOKUP = {
    {"layer_norm.weight", PARLER_NORM},
    {"layer_norm.bias", PARLER_NORM_BIAS},
    {"embed_prompts", PARLER_EMBD_PROMPTS},
    {"text_encoding", PARLER_TEXT_ENCODING},
    {"positional_embed", PARLER_POSITIONAL_EMBD},
    {".self_attn.q_proj.weight", PARLER_LAYER_SELF_ATTN_Q},
    {".self_attn.k_proj.weight", PARLER_LAYER_SELF_ATTN_K},
    {".self_attn.v_proj.weight", PARLER_LAYER_SELF_ATTN_V},
    {".self_attn.out_proj.weight", PARLER_LAYER_SELF_ATTN_O},
    {".self_attn_layer_norm.weight", PARLER_LAYER_SELF_ATTN_NORM},
    {".self_attn_layer_norm.bias", PARLER_LAYER_SELF_ATTN_NORM_BIAS},
    {".encoder_attn.q_proj.weight", PARLER_LAYER_ATTN_Q},
    {".encoder_attn.k_proj.weight", PARLER_LAYER_ATTN_K},
    {".encoder_attn.v_proj.weight", PARLER_LAYER_ATTN_V},
    {".encoder_attn.out_proj.weight", PARLER_LAYER_ATTN_O},
    {".encoder_attn_layer_norm.weight", PARLER_LAYER_ATTN_NORM},
    {".encoder_attn_layer_norm.bias", PARLER_LAYER_ATTN_NORM_BIAS},
    {".fc1.weight", PARLER_LAYER_FC1},
    {".fc2.weight", PARLER_LAYER_FC2},
    {".final_layer_norm.weight", PARLER_LAYER_OUT_NORM},
    {".final_layer_norm.bias", PARLER_LAYER_OUT_NORM_BIAS},
    {".weight", PARLER_EMBD},
    {".weight.head", PARLER_HEAD}
};

void parler_tts_model::assign_weight(std::string name, ggml_tensor * tensor) {
    assign_to_decoder(this, name, tensor);
}

void parler_tts_model::prep_layers(gguf_context * meta_ctx) {
    layers.reserve((size_t) n_layers);
    for (int i = 0; i < (int) n_layers; i++) {
        parler_layer * l = new parler_layer{};
        layers.push_back(l);
    }
    
    embds.reserve((size_t) n_output_heads);
    heads.reserve((size_t) n_output_heads);
    for (int i = 0; i < n_output_heads; i++) {
        struct ggml_tensor * h = nullptr;
        struct ggml_tensor * embd = nullptr;
        embds.push_back(embd);
        heads.push_back(h);
    }
}

void parler_tts_model::prep_constants(gguf_context * meta) {
    int encode_length_key = search_for_gguf_keys(meta, {"parler-tts.decoder.encode_length", "encode_length"});
    if (encode_length_key == -1) {
        TTS_ABORT("key 'parler-tts.decoder.encode_length' must be specified in gguf file.");
    }
    n_encode_length = gguf_get_val_u32(meta, encode_length_key);

    int hidden_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.hidden_size", "hidden_size"});
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int output_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.output_heads", "output_heads"});
    if (output_heads_key != -1) {
        n_output_heads = gguf_get_val_u32(meta, output_heads_key);
    }
    int ctx_length_key = search_for_gguf_keys(meta, {"parler-tts.decoder.context_length", "ctx_length"});
    if (ctx_length_key != -1) {
        max_ctx_length = gguf_get_val_u32(meta, ctx_length_key);
    }
    
    int attn_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.attention.head_count", "attn_heads"});
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
    }
    head_size = hidden_size / n_attn_heads;
    max_cross_nodes = n_attn_heads * 2;
    
    int output_vocab_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.out_vocab_size", "out_vocab_size"});
    if (output_vocab_size_key != -1) {
        output_vocab_size = gguf_get_val_u32(meta, output_vocab_size_key);
    }
    
    int audio_vocab_size_key = search_for_gguf_keys(meta, {"parler-tts.decoder.audio_vocab_size", "audio_vocab_size"});
    if (audio_vocab_size_key != -1) {
        audio_vocab_size = gguf_get_val_u32(meta, audio_vocab_size_key);
    }
    
    int max_gen_key = search_for_gguf_keys(meta, {"parler-tts.decoder.max_generation", "max_generation"});
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }
    
    int n_layers_key = search_for_gguf_keys(meta, {"parler-tts.decoder.num_hidden_layers", "num_hidden_layers"});
    if (n_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, n_layers_key);
    }

    int bos_token_id_key = search_for_gguf_keys(meta, {"audio.bos_token_id", "bos_token_id"});
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int eos_token_id_key = search_for_gguf_keys(meta, {"audio.eos_token_id", "eos_token_id"});
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }
}

void parler_tts_model::prep_cross_key_values(int n_threads, struct tts_response * conditional_prompt) {
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    // Let it create a disposable threadpool just this once
    ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
    std::vector<ggml_backend_buffer_type_t> bufs;
    std::vector<ggml_backend_t> backs;
    ggml_backend_dev_t model_dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (model_dev && ggml_backend_dev_type(model_dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        bufs = {tts_backend_get_buffer_type(backend), backend_cpu_buffer};
        backs = {backend, backend_cpu};
    } else {
        bufs = {backend_cpu_buffer};
        backs = {backend_cpu};
    }
    ggml_backend_sched_t sched = tts_backend_sched_new(backs.data(), bufs.data(), backs.size(), max_cross_nodes*n_layers, false, true);
    
    std::vector<uint8_t> buf_compute_meta;
    buf_compute_meta.resize(max_cross_nodes*n_layers*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_cross_nodes*n_layers, false));
        
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * cctx = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(cctx, 4096, false);
    const ggml_type cross_kv_type = parler_sanitize_accelerator_dtype(
        parler_cross_kv_dtype(),
        backend,
        "PARLER_CROSS_KV_TYPE");
    if (conditional_prompt) {
        // If we are updating the conditional prompt then we have to reset the tensor offsets into the ggml_context otherwise we could overflow the assigned buffer and lose our prompt.
        // These offsets are assigned by #set_tensor below.
        const uint32_t cross_value_slots = parler_precompute_cross_v_flash_enabled() ? 3 : 2;
        offset -= n_encode_length*hidden_size*sizeof(float)*n_layers*cross_value_slots;
        precomputed_input_emb = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, conditional_prompt->hidden_size, conditional_prompt->n_outputs);
        ggml_set_input(precomputed_input_emb);
        if (backend) {
            ggml_backend_sched_set_tensor_backend(sched, precomputed_input_emb, backend);
        }
        n_encode_length = conditional_prompt->n_outputs;
    }
    
    for (int i = 0; i < layers.size(); i++) {
        struct ggml_tensor * Kcur = ggml_mul_mat(cctx, layers[i]->attn_k_proj, precomputed_input_emb);
        struct ggml_tensor * Vcur = ggml_mul_mat(cctx, layers[i]->attn_v_proj, precomputed_input_emb);

        Kcur = ggml_reshape_3d(cctx, Kcur, head_size, n_attn_heads, n_encode_length);
        Vcur = ggml_transpose(cctx, Vcur);

        struct ggml_tensor * k = ggml_cont(cctx, ggml_permute(cctx, Kcur, 0, 2, 1, 3));
        if (cross_kv_type != GGML_TYPE_F32) {
            k = ggml_cast(cctx, k, cross_kv_type);
        }
        ggml_set_name(k, ("cross_key_" + std::to_string(i)).c_str());
        ggml_set_output(k);
        ggml_build_forward_expand(gf, k);

        struct ggml_tensor * v = ggml_cont_3d(cctx, Vcur, n_encode_length, head_size, n_attn_heads);
        if (cross_kv_type != GGML_TYPE_F32) {
            v = ggml_cast(cctx, v, cross_kv_type);
        }
        ggml_set_name(v, ("cross_value_" + std::to_string(i)).c_str());
        ggml_set_output(v);
        ggml_build_forward_expand(gf, v);

        if (parler_precompute_cross_v_flash_enabled()) {
            struct ggml_tensor * v_flash = ggml_cont(cctx, ggml_permute(cctx, v, 1, 0, 2, 3));
            ggml_set_name(v_flash, ("cross_value_flash_" + std::to_string(i)).c_str());
            ggml_set_output(v_flash);
            ggml_build_forward_expand(gf, v_flash);
        }
    }

    ggml_backend_sched_reserve(sched, gf);
    tts_backend_sched_alloc_graph_checked(sched, backend, gf, "parler.cross_attn_precompute");
    if (conditional_prompt) {
        ggml_backend_tensor_set(precomputed_input_emb, conditional_prompt->data, 0, conditional_prompt->n_outputs*conditional_prompt->hidden_size*ggml_element_size(precomputed_input_emb));
    }

    ggml_backend_sched_graph_compute_async(sched, gf);
    ggml_backend_sched_synchronize(sched);
    
    for (int i = 0; i < layers.size(); i++) {
        struct ggml_tensor * k = ggml_graph_get_tensor(gf, ("cross_key_" + std::to_string(i)).c_str());
        layers[i]->cross_k = ggml_dup_tensor(ctx, k);
        set_tensor_from_backend_tensor(layers[i]->cross_k, k);
        struct ggml_tensor * v = ggml_graph_get_tensor(gf, ("cross_value_" + std::to_string(i)).c_str());
        layers[i]->cross_v = ggml_dup_tensor(ctx, v);
        set_tensor_from_backend_tensor(layers[i]->cross_v, v);
        struct ggml_tensor * v_flash = ggml_graph_get_tensor(gf, ("cross_value_flash_" + std::to_string(i)).c_str());
        if (v_flash) {
            layers[i]->cross_v_flash = ggml_dup_tensor(ctx, v_flash);
            set_tensor_from_backend_tensor(layers[i]->cross_v_flash, v_flash);
        } else {
            layers[i]->cross_v_flash = nullptr;
        }
    }
    ggml_free(cctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_cpu);
}

void parler_tts_model::prep_fused_heads() {
    if (fused_heads || heads.empty()) {
        return;
    }

    const ggml_type source_head_type = heads[0]->type;
    const ggml_type head_type = parler_sanitize_accelerator_dtype(
        parler_fused_heads_dtype(source_head_type),
        backend,
        "PARLER_FUSED_HEADS_TYPE");
    fused_heads = ggml_new_tensor_2d(ctx, head_type, hidden_size, (int64_t) output_vocab_size * n_output_heads);
    ggml_set_name(fused_heads, "decoder.lm_heads.fused.weight");

    const size_t alignment = ggml_backend_buffer_get_alignment(buf);
    offset = parler_pad_to_alignment(offset, alignment);
    fused_heads->buffer = buf;
    fused_heads->data = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);

    const size_t fused_size = ggml_nbytes(fused_heads);
    if (offset + fused_size > ggml_backend_buffer_get_size(buf)) {
        TTS_ABORT("Model tensor buffer overflow while storing fused Parler lm heads: %zu + %zu > %zu\n",
                  offset, fused_size, ggml_backend_buffer_get_size(buf));
    }

    std::vector<uint8_t> fused_data(fused_size);
    size_t dst_offset = 0;
    for (uint32_t head = 0; head < n_output_heads; ++head) {
        if (heads[head]->type != source_head_type) {
            TTS_ABORT("Cannot fuse Parler lm head %u: mixed source types %s and %s.\n",
                      head, ggml_type_name(source_head_type), ggml_type_name(heads[head]->type));
        }
        const int64_t head_nelements = ggml_nelements(heads[head]);
        const size_t src_head_bytes = ggml_nbytes(heads[head]);
        const size_t dst_head_bytes = ggml_row_size(head_type, head_nelements);
        if (dst_offset + dst_head_bytes > fused_data.size()) {
            TTS_ABORT("Fused Parler lm head buffer overflow at head %u: %zu + %zu > %zu\n",
                      head, dst_offset, dst_head_bytes, fused_data.size());
        }

        if (head_type == source_head_type) {
            ggml_backend_tensor_get(heads[head], fused_data.data() + dst_offset, 0, src_head_bytes);
        } else if (source_head_type == GGML_TYPE_F32 && head_type == GGML_TYPE_F16) {
            std::vector<float> src(head_nelements);
            ggml_backend_tensor_get(heads[head], src.data(), 0, src_head_bytes);
            ggml_fp32_to_fp16_row(src.data(), reinterpret_cast<ggml_fp16_t *>(fused_data.data() + dst_offset), head_nelements);
        } else if (source_head_type == GGML_TYPE_F16 && head_type == GGML_TYPE_F32) {
            std::vector<ggml_fp16_t> src(head_nelements);
            ggml_backend_tensor_get(heads[head], src.data(), 0, src_head_bytes);
            ggml_fp16_to_fp32_row(src.data(), reinterpret_cast<float *>(fused_data.data() + dst_offset), head_nelements);
        } else {
            TTS_ABORT("Cannot fuse Parler lm heads from %s to %s.\n",
                      ggml_type_name(source_head_type), ggml_type_name(head_type));
        }
        dst_offset += dst_head_bytes;
    }
    ggml_backend_tensor_set(fused_heads, fused_data.data(), 0, fused_size);
    offset += fused_size;
}

void parler_tts_model::free_fused_self_attn_qkv() {
    if (fused_qkv_buf) {
        ggml_backend_buffer_free(fused_qkv_buf);
        fused_qkv_buf = nullptr;
    }
    if (fused_qkv_ctx) {
        ggml_free(fused_qkv_ctx);
        fused_qkv_ctx = nullptr;
    }
    for (parler_layer * layer : layers) {
        if (layer) {
            layer->self_attn_qkv_proj = nullptr;
        }
    }
}

void parler_tts_model::prep_fused_self_attn_qkv() {
    if (fused_qkv_buf || fused_qkv_ctx) {
        return;
    }
    if (layers.empty()) {
        return;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * layers.size() + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    fused_qkv_ctx = ggml_init(params);
    if (!fused_qkv_ctx) {
        TTS_ABORT("Failed to create Parler fused QKV context.\n");
    }

    for (uint32_t i = 0; i < layers.size(); ++i) {
        parler_layer * layer = layers[i];
        if (!layer->self_attn_q_proj || !layer->self_attn_k_proj || !layer->self_attn_v_proj) {
            TTS_ABORT("Cannot fuse Parler self-attention QKV for layer %u: missing projection tensor.\n", i);
        }
        if (layer->self_attn_q_proj->type != layer->self_attn_k_proj->type ||
            layer->self_attn_q_proj->type != layer->self_attn_v_proj->type ||
            layer->self_attn_q_proj->ne[0] != layer->self_attn_k_proj->ne[0] ||
            layer->self_attn_q_proj->ne[0] != layer->self_attn_v_proj->ne[0] ||
            layer->self_attn_q_proj->ne[1] != layer->self_attn_k_proj->ne[1] ||
            layer->self_attn_q_proj->ne[1] != layer->self_attn_v_proj->ne[1]) {
            TTS_ABORT("Cannot fuse Parler self-attention QKV for layer %u: incompatible tensor shapes.\n", i);
        }

        ggml_tensor * fused = ggml_new_tensor_2d(
            fused_qkv_ctx,
            layer->self_attn_q_proj->type,
            layer->self_attn_q_proj->ne[0],
            layer->self_attn_q_proj->ne[1] * 3);
        ggml_set_name(fused, ("decoder.layers." + std::to_string(i) + ".self_attn.qkv_proj.weight").c_str());
        layer->self_attn_qkv_proj = fused;
    }

    fused_qkv_buf = ggml_backend_alloc_ctx_tensors_from_buft(fused_qkv_ctx, buffer);
    if (!fused_qkv_buf) {
        TTS_ABORT("Failed to allocate Parler fused QKV buffer.\n");
    }

    for (uint32_t i = 0; i < layers.size(); ++i) {
        parler_layer * layer = layers[i];
        const size_t q_size = ggml_nbytes(layer->self_attn_q_proj);
        const size_t k_size = ggml_nbytes(layer->self_attn_k_proj);
        const size_t v_size = ggml_nbytes(layer->self_attn_v_proj);
        const size_t fused_size = ggml_nbytes(layer->self_attn_qkv_proj);
        if (q_size != k_size || q_size != v_size || fused_size != q_size * 3) {
            TTS_ABORT("Cannot fuse Parler self-attention QKV for layer %u: byte size mismatch.\n", i);
        }

        std::vector<uint8_t> fused_data(fused_size);
        ggml_backend_tensor_get(layer->self_attn_q_proj, fused_data.data(), 0, q_size);
        ggml_backend_tensor_get(layer->self_attn_k_proj, fused_data.data() + q_size, 0, k_size);
        ggml_backend_tensor_get(layer->self_attn_v_proj, fused_data.data() + q_size + k_size, 0, v_size);
        ggml_backend_tensor_set(layer->self_attn_qkv_proj, fused_data.data(), 0, fused_size);
    }

    fprintf(stderr, "PARLER_FUSE_QKV prepared layers=%zu bytes=%zu\n",
            layers.size(), ggml_backend_buffer_get_size(fused_qkv_buf));
}

void assign_parler_layer(parler_tts_model * model, parler_layer * layer, std::string name, ggml_tensor * tensor) {
    try {
        switch(PARLER_TENSOR_GGUF_LOOKUP.at(name)) {
            case PARLER_LAYER_SELF_ATTN_Q:
                layer->self_attn_q_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_q_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_K:
                layer->self_attn_k_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_k_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_V:
                layer->self_attn_v_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_v_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_O:
                layer->self_attn_o_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_o_proj, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_NORM:
                layer->self_attn_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_norm, tensor);
                break;
            case PARLER_LAYER_SELF_ATTN_NORM_BIAS:
                layer->self_attn_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->self_attn_norm_bias, tensor);
                break;
            case PARLER_LAYER_ATTN_Q:
                if (model->use_cross_attn) {
                    layer->attn_q_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_q_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_K:
                if (model->use_cross_attn) {
                    layer->attn_k_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_k_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_V:
                if (model->use_cross_attn) {
                    layer->attn_v_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_v_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_O:
                if (model->use_cross_attn) {
                    layer->attn_o_proj = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_o_proj, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_NORM:
                if (model->use_cross_attn) {
                    layer->attn_norm = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_norm, tensor);
                }
                break;
            case PARLER_LAYER_ATTN_NORM_BIAS:
                if (model->use_cross_attn) {
                    layer->attn_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(layer->attn_norm_bias, tensor);
                }
                break;
            case PARLER_LAYER_FC1:
                layer->fc1 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->fc1, tensor);
                break;
            case PARLER_LAYER_FC2:
                layer->fc2 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->fc2, tensor);
                break;
            case PARLER_LAYER_OUT_NORM:
                layer->final_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->final_norm, tensor);
                break;
            case PARLER_LAYER_OUT_NORM_BIAS:
                layer->final_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer->final_norm_bias, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } catch (const std::out_of_range& e) {
        TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
    }
}

void assign_to_decoder(parler_tts_model * model, const std::string name, ggml_tensor * tensor) {
    if (PARLER_TENSOR_GGUF_LOOKUP.find(name) != PARLER_TENSOR_GGUF_LOOKUP.end()) {
        try {
            switch (PARLER_TENSOR_GGUF_LOOKUP.at(name)) {
                case PARLER_NORM:
                    model->layer_norm = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->layer_norm, tensor);
                    break;
                case PARLER_NORM_BIAS:
                    model->layer_norm_bias = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->layer_norm_bias, tensor);
                    break;
                case PARLER_EMBD_PROMPTS:
                    model->prompt_embd = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->prompt_embd, tensor);
                    break;
                case PARLER_TEXT_ENCODING:
                    if (model->use_cross_attn) {
                        model->precomputed_input_emb = ggml_dup_tensor(model->ctx, tensor);
                        model->set_tensor(model->precomputed_input_emb, tensor);                        
                    }
                    break;
                case PARLER_POSITIONAL_EMBD:
                    model->precomputed_positional_embds = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(model->precomputed_positional_embds, tensor);
                    break;
                default:
                    fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                    break;
            }
        } catch (const std::out_of_range& e) {
            TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
        auto pair = parse_layer_count(name);
        int layer = pair.first;
        std::string lt_name = pair.second;
        if (name.find("embed_tokens") != std::string::npos) {
            model->embds[layer] = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(model->embds[layer], tensor);
        } else if (name.find("lm_heads") != std::string::npos) {
            model->heads[layer] = ggml_dup_tensor(model->ctx, tensor);
            model->set_tensor(model->heads[layer], tensor);
        } else {
            assign_parler_layer(model, model->layers[layer], lt_name, tensor);
        }
    }
}

void parler_context::reset(int32_t n_output_heads) {
    n_outputs = 0;
    prompt_end_position = 0;
    current_position = 0;
    first_codebook_unfinished = 0;
    profiled_decode_steps = 0;
    last_decode_frames = 0;
    last_decode_used_gpu_argmax = false;
    logits_mask = nullptr;
    output_size = 0;
    output_tokens.clear();
    eos_seen.clear();
    for (int i = 0; i < (int) n_output_heads; i++) {
        eos_seen.push_back(false);
    }
}

struct parler_context * build_new_parler_context(struct parler_tts_model * model, int n_threads, bool use_cpu) {
    parler_context * pctx = new parler_context(model, n_threads);
    pctx->backend = tts_backend_init_accelerator(use_cpu);
    pctx->eos_seen.reserve(model->n_output_heads);
    pctx->backend_cpu = ggml_backend_cpu_init();
    pctx->set_threads();
    const size_t max_nodes = parler_decode_graph_max_nodes(model, parler_requested_unroll_steps());
    pctx->build_schedule(max_nodes);
    pctx->buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));
    return pctx;
}

static bool parler_kv_cache_init(struct parler_kv_cache * cache, parler_tts_model * model, parler_context * pctx) {
    const int64_t n_layer = (int64_t) model->layers.size();

    ggml_backend_buffer_type_t buft = nullptr;
    if (pctx->backend != nullptr) {
        buft = tts_backend_get_buffer_type(pctx->backend);
    } else {
        buft = ggml_backend_cpu_buffer_type();
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ (3u*model->n_layers+1)*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    cache->ctx = ctx;
    const ggml_type kv_type = parler_sanitize_accelerator_dtype(
        parler_kv_cache_dtype(),
        pctx->backend,
        "PARLER_KV_CACHE_TYPE");
    cache->type_k = kv_type;
    cache->type_v = kv_type;
    

    cache->k_l.reserve(n_layer);
    cache->v_l.reserve(n_layer);
    if (parler_self_v_flash_cache_enabled()) {
        cache->v_flash_l.reserve(n_layer);
    }

    for (int i = 0; i < (int) n_layer; i++) {
        ggml_tensor * k = ggml_new_tensor_1d(cache->ctx, cache->type_k, model->hidden_size*model->max_ctx_length);
        ggml_tensor * v = ggml_new_tensor_1d(cache->ctx, cache->type_v, model->hidden_size*model->max_ctx_length);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache->k_l.push_back(k);
        cache->v_l.push_back(v);
        if (parler_self_v_flash_cache_enabled()) {
            ggml_tensor * v_flash = ggml_new_tensor_3d(cache->ctx, cache->type_v, model->head_size, model->max_ctx_length, model->n_attn_heads);
            ggml_format_name(v_flash, "cache_v_flash_l%d", i);
            cache->v_flash_l.push_back(v_flash);
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(cache->ctx, buft);
    if (!buf) {
        return false;
    }
    ggml_backend_buffer_clear(buf, 0);
    cache->buf = buf;

    return true;
}

struct ggml_tensor * parler_build_inp_embd(struct ggml_context * ctx, struct parler_context * pctx, parler_tts_model * model, parler_ubatch & batch) {
    // Parler has two embedding schemas one for the text input and one for generative audio tokens. These two schemas have effectively distinct shapes (i.e. [batch_size, sequence_length] and [batch_size, sequence_lenghth, num_codebooks] respectively).
    // This means that depending on where we are in generation we need to follow a distinct pattern
    struct ggml_tensor * input_embs;
    pctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length);
    ggml_set_input(pctx->positions);
    pctx->set_tensor_backend(pctx->positions);
    if (batch.audio_generation) {
        const int64_t audio_frames = (int64_t) batch.n_audio_tokens / model->n_output_heads;
        pctx->audio_inp_tokens_by_head.clear();
        pctx->audio_inp_tokens_by_head.reserve(model->n_output_heads);
        struct ggml_tensor * audio_embs = nullptr;
        for (int i = 0; i < model->n_output_heads; i++) {
            struct ggml_tensor * head_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, audio_frames);
            ggml_set_input(head_tokens);
            pctx->set_tensor_backend(head_tokens);
            pctx->audio_inp_tokens_by_head.push_back(head_tokens);
            struct ggml_tensor * head_embs = ggml_get_rows(ctx, model->embds[i], head_tokens);
            if (i == 0) {
                audio_embs = head_embs;
            } else {
                audio_embs = ggml_add(ctx, head_embs, audio_embs);
            }
        }
        input_embs = audio_embs;
        if (batch.n_tokens > 0) {
            pctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
            ggml_set_input(pctx->inp_tokens);
            pctx->set_tensor_backend(pctx->inp_tokens);
            struct ggml_tensor * prompt_embs = ggml_get_rows(ctx, model->prompt_embd, pctx->inp_tokens);
            input_embs = ggml_concat(ctx, prompt_embs, audio_embs, 1);
        }
    } else {
        pctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        ggml_set_input(pctx->inp_tokens);
        pctx->set_tensor_backend(pctx->inp_tokens);
        input_embs = ggml_get_rows(ctx, model->prompt_embd, pctx->inp_tokens);
    }
    return ggml_add(ctx, input_embs, ggml_get_rows(ctx, model->precomputed_positional_embds, pctx->positions));
}

struct ggml_tensor * parler_build_layer_norm(struct ggml_context * ctx, struct ggml_tensor * inputs, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // parler always uses default eps
    float eps = 0.00001;
    inputs = ggml_norm(ctx, inputs, eps);
    inputs = ggml_mul(ctx, inputs, weight);
    return ggml_add(ctx, inputs, bias);
}

void parler_build_kv_store(struct ggml_context * ctx, parler_kv_cache * kv, struct ggml_cgraph * graph, struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, int32_t n_tokens, int32_t kv_head, int32_t index, int32_t n_embd_gqa) {
    // this is the max context size;
    const int64_t n_ctx = 4096;
    struct ggml_tensor * v_cur_hidden_first = v_cur;

    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv->k_l[index], n_tokens*n_embd_gqa, ggml_row_size(kv->k_l[index]->type, n_embd_gqa)*kv_head);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd_gqa && v_cur->ne[1] == n_tokens);

    const bool store_legacy_v = kv->v_flash_l.empty() || !parler_self_v_flash_only_enabled() || n_tokens != 1;
    if (store_legacy_v) {
        struct ggml_tensor * v_cache_view = ggml_view_2d(ctx, kv->v_l[index], n_tokens, n_embd_gqa,
                (  n_ctx)*ggml_element_size(kv->v_l[index]),
                (kv_head)*ggml_element_size(kv->v_l[index]));

        struct ggml_tensor * v_legacy = ggml_cont(ctx, ggml_transpose(ctx, v_cur));

        ggml_build_forward_expand(graph, ggml_cpy(ctx, v_legacy, v_cache_view));
    }

    if (!kv->v_flash_l.empty()) {
        struct ggml_tensor * v_flash_cache = kv->v_flash_l[index];
        const int64_t head_size = v_flash_cache->ne[0];
        const int64_t n_heads = n_embd_gqa / head_size;
        struct ggml_tensor * v_flash_cache_view = ggml_view_3d(
            ctx,
            v_flash_cache,
            head_size,
            n_tokens,
            n_heads,
            v_flash_cache->nb[1],
            v_flash_cache->nb[2],
            kv_head * v_flash_cache->nb[1]);
        struct ggml_tensor * v_flash_cur = ggml_reshape_3d(ctx, v_cur_hidden_first, head_size, n_heads, n_tokens);
        v_flash_cur = ggml_cont(ctx, ggml_permute(ctx, v_flash_cur, 0, 2, 1, 3));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, v_flash_cur, v_flash_cache_view));
    }
}

struct ggml_tensor * parler_build_head_outputs(struct ggml_context * ctx, parler_tts_model * model, struct ggml_tensor * cur) {
    if (model->fused_heads && cur->ne[1] == 1) {
        struct ggml_tensor * out = ggml_mul_mat(ctx, model->fused_heads, cur);
        ggml_set_name(out, "final_out_fused");
        return ggml_reshape_3d(ctx, out, model->output_vocab_size, 1, model->n_output_heads);
    }

    // going to cat the heads together and then reshape them;
    // honestly ggml doesn't provide good support for stacking and discrete tensor access
    struct ggml_tensor * out;
    for (int i = 0; i < model->n_output_heads; i++) {
        if (i == 0) {
            out = ggml_mul_mat(ctx, model->heads[i], cur);
        } else {
            out = ggml_concat(ctx, out, ggml_mul_mat(ctx, model->heads[i], cur), 1);
        }
    }
    ggml_set_name(out, "final_out");
    //out = ggml_cont(ctx, ggml_transpose(ctx, out));

    int32_t sql_len = (int32_t) (ggml_nelements(out) / (model->output_vocab_size * model->n_output_heads));
    return ggml_cont_3d(ctx, out, model->output_vocab_size, sql_len, model->n_output_heads);
}

struct ggml_tensor * build_attn_mask(ggml_context * ctx, parler_context * pctx, parler_ubatch & batch) {
    pctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) pctx->current_position + batch.sequence_length, (int64_t) pctx->current_position + batch.sequence_length);
    ggml_set_input(pctx->attn_mask);
    pctx->set_tensor_backend(pctx->attn_mask);

    return pctx->attn_mask;
}

struct ggml_tensor * build_attn_mask_cross(ggml_context * ctx, parler_context * pctx, parler_tts_model * model, parler_ubatch & batch) {
    pctx->attn_mask_cross = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) model->n_encode_length, (int64_t) batch.sequence_length);
    ggml_set_input(pctx->attn_mask_cross);
    pctx->set_tensor_backend(pctx->attn_mask_cross);
    
    return pctx->attn_mask_cross;
}

static struct parler_ubatch batch_from_sentence(std::string sentence, parler_tts_model * model, unigram_tokenizer * tokenizer, bool include_audio_start = false) {
    struct parler_ubatch batch;
    batch.audio_generation = include_audio_start;
    std::vector<uint32_t>* token_ids = new std::vector<uint32_t>;
    tokenizer->tokenize(sentence, *token_ids);
    token_ids->push_back(tokenizer->eos_token);
    parler_debug_print_tokens("TEXT_TOKENS", *token_ids, 0);
    batch.current_step = include_audio_start ? 1 : 0;
    batch.n_tokens = token_ids->size();
    std::vector<uint32_t>* audio_token_ids = nullptr;
    if (include_audio_start) {
        audio_token_ids = new std::vector<uint32_t>;
        audio_token_ids->reserve(model->n_output_heads);
        for (uint32_t i = 0; i < model->n_output_heads; ++i) {
            audio_token_ids->push_back(model->bos_token_id);
        }
        batch.n_audio_tokens = audio_token_ids->size();
        batch.audio_tokens = audio_token_ids->data();
    } else {
        batch.n_audio_tokens = 0;
        batch.audio_tokens = nullptr;
    }
    batch.sequence_length = batch.n_tokens + (batch.n_audio_tokens / model->n_output_heads);
    std::vector<uint32_t>* position = new std::vector<uint32_t>;
    for (uint32_t i = 0; i < batch.sequence_length; i++) {
        position->push_back(i);
    }
    std::vector<uint32_t>* order = new std::vector<uint32_t>;
    for (int i = 0; i < batch.sequence_length; i++) {
        if (i >= batch.sequence_length - 1) {
            order->push_back(0);
        } else {
            order->push_back(i+1);
        }
    }
    batch.positions = position->data();
    batch.tokens = token_ids->data();
    return batch;
}

void parler_tts_runner::assign_weight(const char * name, ggml_tensor & tensor) {
    if (const string_view name_sv{ name }; name_sv.starts_with("audio_encoder.")) {
        dac_runner->model->assign_weight(string{ name_sv.substr(sizeof("audio_encoder.") - 1) }, &tensor);
    } else if (name_sv.starts_with("decoder.")) {
        model->assign_weight(string{ name_sv.substr(sizeof("decoder.") - 1) }, &tensor);
    } else {
        fprintf(stdout, "Warning: function %s encountered an unhandled tensor named '%s'.\n", __func__, name);
    }
}

void parler_tts_runner::update_conditional_prompt(const char * file_path, const char * prompt) {
    const int      n_threads{ pctx->n_threads };
    const bool     cpu_only{ pctx->backend == nullptr };
    t5_runner * text_encoder = text_encoder_from_file(file_path, n_threads, nullptr, cpu_only);
    tts_response response;
    text_encoder->generate(prompt, &response);
    model->prep_cross_key_values(n_threads, &response);
    delete text_encoder;
}

static ggml_tensor * parler_build_decoder_layers(
        ggml_context * ctx,
        ggml_cgraph * gf,
        parler_context * pctx,
        parler_tts_model * model,
        parler_kv_cache * kv_self,
        ggml_tensor * inpL,
        uint32_t sequence_length,
        uint32_t current_position,
        ggml_tensor * KQ_mask_dec) {
    const bool use_flash_self_attn = sequence_length == 1 && parler_flash_attn_mode_enabled("self");
    const bool use_flash_cross_attn = sequence_length == 1 && parler_flash_attn_mode_enabled("cross");
    const int32_t full_sequence_length = (int32_t) current_position + (int32_t) sequence_length;
    ggml_tensor * cur = inpL;

    for (int l = 0; l < model->n_layers; l++) {
        const std::string layer_name = "layer_" + std::to_string(l);
        struct ggml_tensor * residual = inpL;
        ggml_set_name(inpL, (layer_name + "_input").c_str());

        cur = parler_build_layer_norm(ctx, inpL, model->layers[l]->self_attn_norm, model->layers[l]->self_attn_norm_bias);
        struct ggml_tensor * self_attn_inp = parler_maybe_cast_activation(ctx, pctx, cur, "self_attn");

        struct ggml_tensor * attn_out;

        // self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;
            if (model->layers[l]->self_attn_qkv_proj) {
                struct ggml_tensor * qkv = ggml_mul_mat(ctx, model->layers[l]->self_attn_qkv_proj, self_attn_inp);
                ggml_set_name(qkv, (layer_name + "_self_attn_qkv").c_str());
                Qcur = ggml_view_2d(ctx, qkv, model->hidden_size, sequence_length, qkv->nb[1], 0);
                Kcur = ggml_view_2d(ctx, qkv, model->hidden_size, sequence_length, qkv->nb[1], model->hidden_size * qkv->nb[0]);
                Vcur = ggml_view_2d(ctx, qkv, model->hidden_size, sequence_length, qkv->nb[1], 2 * model->hidden_size * qkv->nb[0]);
                if (!parler_fused_qkv_direct_views_enabled() || sequence_length != 1) {
                    Qcur = ggml_cont(ctx, Qcur);
                    Kcur = ggml_cont(ctx, Kcur);
                    Vcur = ggml_cont(ctx, Vcur);
                }
            } else {
                Qcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_q_proj, self_attn_inp);
                ggml_set_name(Qcur, (layer_name + "_self_attn_q").c_str());
                Kcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_k_proj, self_attn_inp);
                ggml_set_name(Kcur, (layer_name + "_self_attn_k").c_str());
                Vcur = ggml_mul_mat(ctx, model->layers[l]->self_attn_v_proj, self_attn_inp);
                ggml_set_name(Vcur, (layer_name + "_self_attn_v").c_str());
            }

            parler_build_kv_store(ctx, kv_self, gf, Kcur, Vcur, (int32_t) sequence_length, current_position, l, model->hidden_size);
            struct ggml_tensor * k =
                ggml_view_3d(ctx, kv_self->k_l[l],
                        model->head_size, full_sequence_length, model->n_attn_heads,
                        ggml_row_size(kv_self->k_l[l]->type, model->hidden_size),
                        ggml_row_size(kv_self->k_l[l]->type, model->head_size),
                        0);

            struct ggml_tensor * v =
                ggml_view_3d(ctx, kv_self->v_l[l],
                        full_sequence_length, model->head_size, model->n_attn_heads,
                        ggml_element_size(kv_self->v_l[l])*model->max_ctx_length,
                        ggml_element_size(kv_self->v_l[l])*model->max_ctx_length*model->head_size,
                        0);

            Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, sequence_length);
            struct ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));

            if (use_flash_self_attn) {
                struct ggml_tensor * v_flash = nullptr;
                if (!kv_self->v_flash_l.empty()) {
                    ggml_tensor * v_flash_cache = kv_self->v_flash_l[l];
                    v_flash = ggml_view_3d(ctx,
                            v_flash_cache,
                            model->head_size, full_sequence_length, model->n_attn_heads,
                            v_flash_cache->nb[1],
                            v_flash_cache->nb[2],
                            0);
                } else {
                    v_flash = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
                }
                struct ggml_tensor * flash = ggml_flash_attn_ext(ctx, q, k, v_flash, nullptr, 1.0f/sqrtf(model->head_size), 0.0f, 0.0f);
                ggml_set_name(flash, (layer_name + "_self_attn_flash").c_str());
                attn_out = ggml_cont_2d(ctx, flash, model->hidden_size, sequence_length);
            } else {
                struct ggml_tensor * kq = ggml_mul_mat(ctx, ggml_cont(ctx, k), q);
                ggml_set_name(kq, (layer_name + "_self_attn_kq").c_str());
                kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, 1.0f/sqrtf(model->head_size), 0.0f);
                struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
                ggml_set_name(kqv, (layer_name + "_self_attn_kqv").c_str());
                struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
                attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, sequence_length);
            }
            attn_out = ggml_mul_mat(ctx, model->layers[l]->self_attn_o_proj, parler_maybe_cast_activation(ctx, pctx, attn_out, "self_attn_out"));
            ggml_set_name(attn_out, (layer_name + "_self_attn_out").c_str());
        }

        cur = ggml_add(ctx, attn_out, residual);

        if (model->use_cross_attn) {
            struct ggml_tensor * residuala = cur;

            // norm
            cur = parler_build_layer_norm(ctx, cur, model->layers[l]->attn_norm, model->layers[l]->attn_norm_bias);
            struct ggml_tensor * cross_attn_inp = parler_maybe_cast_activation(ctx, pctx, cur, "cross_attn");

            //cross-attention
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l]->attn_q_proj, cross_attn_inp);
            ggml_set_name(Qcur, (layer_name + "_cross_attn_q").c_str());
            Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, sequence_length);

            struct ggml_tensor * q = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));

            if (use_flash_cross_attn) {
                struct ggml_tensor * v_flash = model->layers[l]->cross_v_flash
                    ? model->layers[l]->cross_v_flash
                    : ggml_cont(ctx, ggml_permute(ctx, model->layers[l]->cross_v, 1, 0, 2, 3));
                struct ggml_tensor * flash = ggml_flash_attn_ext(ctx, q, model->layers[l]->cross_k, v_flash, nullptr, 1.0f/sqrtf(model->head_size), 0.0f, 0.0f);
                ggml_set_name(flash, (layer_name + "_cross_attn_flash").c_str());
                cur = ggml_cont_2d(ctx, flash, model->hidden_size, sequence_length);
            } else {
                struct ggml_tensor * kq = ggml_mul_mat(ctx, model->layers[l]->cross_k, q);
                ggml_set_name(kq, (layer_name + "_cross_attn_kq").c_str());
                kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f/sqrtf(model->head_size), 0.0f);

                struct ggml_tensor * kqv  = ggml_mul_mat(ctx, kq, model->layers[l]->cross_v);
                ggml_set_name(kqv, (layer_name + "_cross_attn_kqv").c_str());
                struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
                cur = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, sequence_length);
            }
            cur = ggml_mul_mat(ctx, model->layers[l]->attn_o_proj, parler_maybe_cast_activation(ctx, pctx, cur, "cross_attn_out"));
            ggml_set_name(cur, (layer_name + "_cross_attn_out").c_str());
            cur = ggml_add(ctx, cur, residuala);
        }

        struct ggml_tensor * residualffn = cur;

        cur = parler_build_layer_norm(ctx, cur, model->layers[l]->final_norm, model->layers[l]->final_norm_bias);
        cur = parler_maybe_cast_activation(ctx, pctx, cur, "mlp_fc1");
        cur = ggml_mul_mat(ctx, model->layers[l]->fc1, cur);
        ggml_set_name(cur, (layer_name + "_mlp_fc1").c_str());
#if TTS_GGML_HAS_GELU_ERF
        cur = parler_mlp_gelu_inplace_enabled() ? ggml_gelu_erf_inplace(ctx, cur) : ggml_gelu_erf(ctx, cur);
#else
        cur = parler_mlp_gelu_inplace_enabled() ? ggml_gelu_inplace(ctx, cur) : ggml_gelu(ctx, cur);
#endif
        if (parler_mlp_gelu_graph_output_enabled()) {
            ggml_set_output(cur);
        }
        cur = parler_maybe_cast_activation(ctx, pctx, cur, "mlp_fc2");
        cur = ggml_mul_mat(ctx, model->layers[l]->fc2, cur);
        ggml_set_name(cur, (layer_name + "_mlp_fc2").c_str());
        cur = ggml_add(ctx, cur, residualffn);
        inpL = cur;
    }

    return cur;
}

struct ggml_cgraph * parler_tts_runner::build_parler_graph(parler_ubatch & batch) {
    init_build();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    struct ggml_tensor * inpL;

    inpL = parler_build_inp_embd(ctx, pctx, model, batch);

    pctx->attn_mask = nullptr;
    pctx->attn_mask_cross = nullptr;
    pctx->logits_mask = nullptr;
    const bool need_self_attn_mask = parler_self_attn_mask_required(pctx, batch);
    const bool use_flash_self_attn = batch.sequence_length == 1 && parler_flash_attn_mode_enabled("self");
    struct ggml_tensor * KQ_mask_dec = (use_flash_self_attn || !need_self_attn_mask) ? nullptr : build_attn_mask(ctx, pctx, batch);
    // Cross-attention in this runner uses the full precomputed text encoding and the mask is always all zeros.
    struct ggml_tensor * KQ_mask_cross = nullptr;

    (void) KQ_mask_cross;
    ggml_tensor * cur = parler_build_decoder_layers(
            ctx, gf, pctx, model, kv_self, inpL,
            (uint32_t) batch.sequence_length, pctx->current_position, KQ_mask_dec);

    if (parler_prefill_last_logits_only(batch)) {
        cur = ggml_view_2d(
                ctx,
                cur,
                model->hidden_size,
                1,
                cur->nb[1],
                (batch.sequence_length - 1) * cur->nb[1]);
        cur = ggml_cont_2d(ctx, cur, model->hidden_size, 1);
        ggml_set_name(cur, "prefill_last_hidden");
    }

    cur = parler_build_layer_norm(ctx, cur, model->layer_norm, model->layer_norm_bias);
    cur = parler_maybe_cast_activation(ctx, pctx, cur, "head");
    cur = parler_build_head_outputs(ctx, model, cur);
    if (can_use_gpu_argmax(batch)) {
        cur = ggml_reshape_2d(ctx, cur, model->output_vocab_size, model->n_output_heads);
        pctx->logits_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model->output_vocab_size, model->n_output_heads);
        ggml_set_name(pctx->logits_mask, "final_out_logits_mask");
        ggml_set_input(pctx->logits_mask);
        pctx->set_tensor_backend(pctx->logits_mask);
        cur = ggml_add(ctx, cur, pctx->logits_mask);
        ggml_set_name(cur, "final_out_masked");
        cur = ggml_argmax(ctx, cur);
        ggml_set_name(cur, "final_out_argmax");
    }
    ggml_build_forward_expand(gf, cur);
    free_build();
    
    return gf;
}

void parler_tts_runner::set_inputs(parler_ubatch & batch) {
    if (batch.audio_generation) {
        if (batch.n_tokens > 0) {
            ggml_backend_tensor_set(pctx->inp_tokens, batch.tokens, 0, batch.n_tokens*ggml_element_size(pctx->inp_tokens));
        }
        const size_t frame_count = batch.n_audio_tokens / model->n_output_heads;
        if (pctx->audio_inp_token_scratch.size() < model->n_output_heads) {
            pctx->audio_inp_token_scratch.resize(model->n_output_heads);
        }
        for (uint32_t head = 0; head < model->n_output_heads; ++head) {
            std::vector<uint32_t> & head_tokens = pctx->audio_inp_token_scratch[head];
            head_tokens.resize(frame_count);
            for (size_t frame = 0; frame < frame_count; ++frame) {
                head_tokens[frame] = batch.audio_tokens[frame * model->n_output_heads + head];
            }
            ggml_backend_tensor_set(
                pctx->audio_inp_tokens_by_head[head],
                head_tokens.data(),
                0,
                head_tokens.size() * ggml_element_size(pctx->audio_inp_tokens_by_head[head]));
        }
    } else {
        ggml_backend_tensor_set(pctx->inp_tokens, batch.tokens, 0, batch.n_tokens*ggml_element_size(pctx->inp_tokens));
    }
    ggml_backend_tensor_set(pctx->positions, batch.positions, 0, batch.sequence_length*ggml_element_size(pctx->positions));
    uint32_t max_pos = pctx->current_position + batch.sequence_length;
    std::vector<float> attn_mask((size_t) batch.sequence_length * max_pos);
    for (int i = 0; i < batch.sequence_length; i++) {
        uint32_t pos = batch.positions[i];
        for (int ii = 0; ii < max_pos; ii++) {
            attn_mask[(size_t)i*max_pos + ii] = ii > pos ? -INFINITY : 0.0f;
        }
    }
    if (pctx->attn_mask) {
        ggml_backend_tensor_set(pctx->attn_mask, attn_mask.data(), 0, attn_mask.size() * sizeof(float));
    }
    
    if (model->use_cross_attn && pctx->attn_mask_cross) {
        std::vector<float> attn_mask_cross((size_t) model->n_encode_length * batch.sequence_length, 0.0f);
        ggml_backend_tensor_set(pctx->attn_mask_cross, attn_mask_cross.data(), 0, attn_mask_cross.size() * sizeof(float));
    }

    if (pctx->logits_mask) {
        const size_t mask_size = (size_t) model->output_vocab_size * model->n_output_heads;
        pctx->logits_mask_scratch.assign(mask_size, 0.0f);
        for (uint32_t head = pctx->first_codebook_unfinished + 1; head < model->n_output_heads; ++head) {
            pctx->logits_mask_scratch[(size_t) head * model->output_vocab_size + model->eos_token_id] = -INFINITY;
        }
        ggml_backend_tensor_set(
            pctx->logits_mask,
            pctx->logits_mask_scratch.data(),
            0,
            pctx->logits_mask_scratch.size() * sizeof(float));
    }

}
void parler_tts_runner::parler_graph_compute(ggml_cgraph * gf) {
    ggml_backend_sched_graph_compute_async(pctx->sched, gf);
}

bool parler_tts_runner::can_use_gpu_argmax(const parler_ubatch & batch) const {
    return parler_gpu_argmax_enabled() &&
           pctx->backend != nullptr &&
           batch.audio_generation &&
           batch.sequence_length == 1 &&
           sampler->can_use_argmax_fast_path();
}

uint32_t parler_tts_runner::decoder_unroll_steps(const parler_ubatch & batch) const {
    const uint32_t requested = parler_requested_unroll_steps();
    if (requested <= 1 || !can_use_gpu_argmax(batch)) {
        return 1;
    }

    // First implementation keeps the logits mask static inside the unrolled graph. If an earlier
    // codebook has already finished, the next valid mask depends on runtime EOS state, so use the
    // existing single-step path.
    if (pctx->first_codebook_unfinished != 0) {
        return 1;
    }

    const uint32_t generated_frames = (uint32_t) (pctx->output_tokens.size() / model->n_output_heads);
    if (generated_frames >= model->max_generation_size) {
        return 1;
    }
    uint32_t remaining = model->max_generation_size - generated_frames;
    if (remaining <= 1) {
        return 1;
    }

    // Avoid the delayed-pattern EOS tail in the unrolled graph. Tail steps intentionally update
    // first_codebook_unfinished before every decode; that branch stays on the exact single-step path.
    const uint32_t safe_tail_inclusive = parler_eos_tail_step_for_head(model, 0);
    if ((uint32_t) batch.current_step > safe_tail_inclusive) {
        return 1;
    }
    const uint32_t safe_before_tail = safe_tail_inclusive - (uint32_t) batch.current_step + 1;
    return std::max<uint32_t>(1, std::min({requested, remaining, safe_before_tail}));
}

static void parler_update_first_codebook_unfinished(parler_tts_model * model, parler_context * pctx, const parler_ubatch & batch) {
    if (!batch.audio_generation || batch.n_audio_tokens == 0 || pctx->first_codebook_unfinished + 1 >= model->n_output_heads) {
        return;
    }

    const uint32_t frame_count = batch.n_audio_tokens / model->n_output_heads;
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        const uint32_t token = batch.audio_tokens[frame * model->n_output_heads + pctx->first_codebook_unfinished];
        if (token == model->eos_token_id) {
            ++pctx->first_codebook_unfinished;
        }
    }
}

static ggml_tensor * parler_new_i32_input(
        ggml_context * ctx,
        parler_context * pctx,
        std::vector<ggml_tensor *> & tensors,
        std::vector<int32_t> & values,
        int32_t value,
        const char * name) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    pctx->set_tensor_backend(tensor);
    tensors.push_back(tensor);
    values.push_back(value);
    return tensor;
}

static ggml_tensor * parler_build_audio_embd_from_head_tokens(
        ggml_context * ctx,
        parler_context * pctx,
        parler_tts_model * model,
        const std::vector<ggml_tensor *> & head_tokens,
        ggml_tensor * position) {
    ggml_tensor * audio_embs = nullptr;
    for (uint32_t head = 0; head < model->n_output_heads; ++head) {
        ggml_tensor * head_embs = ggml_get_rows(ctx, model->embds[head], head_tokens[head]);
        if (head == 0) {
            audio_embs = head_embs;
        } else {
            audio_embs = ggml_add(ctx, head_embs, audio_embs);
        }
    }
    return ggml_add(ctx, audio_embs, ggml_get_rows(ctx, model->precomputed_positional_embds, position));
}

static ggml_tensor * parler_build_masked_argmax(
        ggml_context * ctx,
        parler_context * pctx,
        parler_tts_model * model,
        ggml_tensor * cur,
        ggml_tensor * logits_mask) {
    cur = parler_build_layer_norm(ctx, cur, model->layer_norm, model->layer_norm_bias);
    cur = parler_maybe_cast_activation(ctx, pctx, cur, "head");
    cur = parler_build_head_outputs(ctx, model, cur);
    cur = ggml_reshape_2d(ctx, cur, model->output_vocab_size, model->n_output_heads);
    cur = ggml_add(ctx, cur, logits_mask);
    cur = ggml_argmax(ctx, cur);
    return cur;
}

int parler_tts_runner::decode_unrolled(parler_ubatch & batch, uint32_t unroll_steps, decode_timing * timing) {
    const auto t_start = std::chrono::steady_clock::now();
    ggml_backend_sched_reset(pctx->sched);
    pctx->last_decode_used_gpu_argmax = false;
    pctx->last_decode_frames = 0;
    parler_update_first_codebook_unfinished(model, pctx, batch);

    pctx->output_tokens.reserve((size_t) model->max_generation_size * model->n_output_heads);
    const size_t output_bytes = (size_t) unroll_steps * model->n_output_heads * sizeof(int32_t);
    const size_t prev_size = pctx->buf_output ? ggml_backend_buffer_get_size(pctx->buf_output) : 0;
    if (!pctx->buf_output || prev_size < output_bytes) {
        if (pctx->buf_output) {
            ggml_backend_buffer_free(pctx->buf_output);
            pctx->buf_output = nullptr;
            pctx->logits = nullptr;
        }
        pctx->buf_output = ggml_backend_buft_alloc_buffer(pctx->backend_cpu_buffer, output_bytes);
    }
    const auto t_reserved = std::chrono::steady_clock::now();

    init_build();
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, parler_decode_graph_max_nodes(model, unroll_steps), false);
    pctx->unroll_token_inputs.clear();
    pctx->unroll_token_values.clear();
    pctx->unroll_position_inputs.clear();
    pctx->unroll_position_values.clear();

    ggml_tensor * logits_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model->output_vocab_size, model->n_output_heads);
    ggml_set_name(logits_mask, "unroll_logits_mask");
    ggml_set_input(logits_mask);
    pctx->set_tensor_backend(logits_mask);
    pctx->logits_mask = logits_mask;

    ggml_tensor * prev_argmax = nullptr;
    ggml_tensor * merged_argmax = nullptr;
    for (uint32_t step_index = 0; step_index < unroll_steps; ++step_index) {
        const uint32_t current_step = (uint32_t) batch.current_step + step_index;
        const uint32_t current_position = pctx->current_position + step_index;
        std::vector<ggml_tensor *> head_tokens;
        head_tokens.reserve(model->n_output_heads);

        for (uint32_t head = 0; head < model->n_output_heads; ++head) {
            ggml_tensor * token_tensor = nullptr;
            char name[64];
            if (step_index == 0) {
                snprintf(name, sizeof(name), "unroll_token_s%u_h%u", step_index, head);
                token_tensor = parler_new_i32_input(
                        ctx,
                        pctx,
                        pctx->unroll_token_inputs,
                        pctx->unroll_token_values,
                        (int32_t) batch.audio_tokens[head],
                        name);
            } else if (current_step <= head + 1) {
                snprintf(name, sizeof(name), "unroll_bos_s%u_h%u", step_index, head);
                token_tensor = parler_new_i32_input(
                        ctx,
                        pctx,
                        pctx->unroll_token_inputs,
                        pctx->unroll_token_values,
                        (int32_t) model->bos_token_id,
                        name);
            } else {
                token_tensor = ggml_view_1d(ctx, prev_argmax, 1, head * ggml_element_size(prev_argmax));
            }
            head_tokens.push_back(token_tensor);
        }

        char pos_name[64];
        snprintf(pos_name, sizeof(pos_name), "unroll_pos_s%u", step_index);
        ggml_tensor * position = parler_new_i32_input(
                ctx,
                pctx,
                pctx->unroll_position_inputs,
                pctx->unroll_position_values,
                (int32_t) current_position,
                pos_name);

        ggml_tensor * inpL = parler_build_audio_embd_from_head_tokens(ctx, pctx, model, head_tokens, position);
        ggml_tensor * cur = parler_build_decoder_layers(
                ctx, gf, pctx, model, kv_self, inpL, 1, current_position, nullptr);
        ggml_tensor * argmax = parler_build_masked_argmax(ctx, pctx, model, cur, logits_mask);
        ggml_set_name(argmax, ("unroll_argmax_s" + std::to_string(step_index)).c_str());
        ggml_tensor * argmax_2d = ggml_reshape_2d(ctx, argmax, model->n_output_heads, 1);
        merged_argmax = merged_argmax ? ggml_concat(ctx, merged_argmax, argmax_2d, 1) : argmax_2d;
        prev_argmax = argmax;
    }
    ggml_set_name(merged_argmax, "unroll_argmax_merged");
    ggml_build_forward_expand(gf, merged_argmax);
    free_build();
    const auto t_built = std::chrono::steady_clock::now();

    pctx->alloc_graph(gf, "parler.decode_unrolled");
    const auto t_alloc = std::chrono::steady_clock::now();

    for (size_t i = 0; i < pctx->unroll_token_inputs.size(); ++i) {
        ggml_backend_tensor_set(
                pctx->unroll_token_inputs[i],
                &pctx->unroll_token_values[i],
                0,
                sizeof(int32_t));
    }
    for (size_t i = 0; i < pctx->unroll_position_inputs.size(); ++i) {
        ggml_backend_tensor_set(
                pctx->unroll_position_inputs[i],
                &pctx->unroll_position_values[i],
                0,
                sizeof(int32_t));
    }
    const size_t mask_size = (size_t) model->output_vocab_size * model->n_output_heads;
    pctx->logits_mask_scratch.assign(mask_size, 0.0f);
    for (uint32_t head = pctx->first_codebook_unfinished + 1; head < model->n_output_heads; ++head) {
        pctx->logits_mask_scratch[(size_t) head * model->output_vocab_size + model->eos_token_id] = -INFINITY;
    }
    ggml_backend_tensor_set(
            logits_mask,
            pctx->logits_mask_scratch.data(),
            0,
            pctx->logits_mask_scratch.size() * sizeof(float));
    const auto t_inputs = std::chrono::steady_clock::now();

    parler_graph_compute(gf);
    const auto t_submit = std::chrono::steady_clock::now();

    pctx->argmax_tokens_scratch.resize((size_t) unroll_steps * model->n_output_heads);
    pctx->get_ggml_node_data_raw(
            merged_argmax,
            pctx->argmax_tokens_scratch.data(),
            output_bytes);
    const auto t_read = std::chrono::steady_clock::now();

    uint32_t kept_frames = 0;
    for (uint32_t step_index = 0; step_index < unroll_steps; ++step_index) {
        const size_t base = (size_t) step_index * model->n_output_heads;
        for (uint32_t head = 0; head < model->n_output_heads; ++head) {
            pctx->output_tokens.push_back((uint32_t) pctx->argmax_tokens_scratch[base + head]);
        }
        kept_frames += 1;

        // If the model itself emits EOS for the first unfinished codebook, the next step's mask
        // changes. We already computed later speculative steps with the old mask, so discard them
        // by not advancing current_position past this frame; the next loop will overwrite cache.
        if (pctx->argmax_tokens_scratch[base + pctx->first_codebook_unfinished] == (int32_t) model->eos_token_id) {
            break;
        }
    }

    pctx->last_decode_used_gpu_argmax = true;
    pctx->last_decode_frames = kept_frames;
    pctx->n_outputs += kept_frames;
    pctx->profiled_decode_steps += kept_frames;

    ggml_backend_sched_reset(pctx->sched);
    const auto t_done = std::chrono::steady_clock::now();

    if (timing) {
        timing->steps += kept_frames;
        timing->audio_steps += kept_frames;
        timing->reserve_ms += parler_elapsed_ms(t_start, t_reserved);
        timing->build_ms += parler_elapsed_ms(t_reserved, t_built);
        timing->alloc_ms += parler_elapsed_ms(t_built, t_alloc);
        timing->input_ms += parler_elapsed_ms(t_alloc, t_inputs);
        timing->submit_ms += parler_elapsed_ms(t_inputs, t_submit);
        timing->read_ms += parler_elapsed_ms(t_submit, t_read);
        timing->reset_ms += parler_elapsed_ms(t_read, t_done);
        timing->total_ms += parler_elapsed_ms(t_start, t_done);
    }

    return 0;
}

int parler_tts_runner::decode(parler_ubatch & batch, decode_timing * timing) {
    const auto t_start = std::chrono::steady_clock::now();
    ggml_backend_sched_reset(pctx->sched);
    pctx->last_decode_used_gpu_argmax = false;
    pctx->last_decode_frames = 0;
    const bool use_gpu_argmax = can_use_gpu_argmax(batch);
    if (use_gpu_argmax) {
        parler_update_first_codebook_unfinished(model, pctx, batch);
    }
    
    pctx->output_tokens.reserve((size_t) model->max_generation_size * model->n_output_heads);

    const size_t active_positions = (size_t) pctx->current_position + batch.sequence_length;
    const size_t reserved_positions = (batch.audio_generation ? (size_t) pctx->prompt_end_position : batch.sequence_length) +
                                      (size_t) model->max_generation_size;
    const size_t logits_positions = std::max(active_positions, reserved_positions);
    const size_t logits_size = (size_t) model->output_vocab_size * logits_positions * model->n_output_heads;
    const size_t prev_size = pctx->buf_output ? ggml_backend_buffer_get_size(pctx->buf_output) : 0;
    const size_t new_size  = logits_size * sizeof(float);
    
    if (!pctx->buf_output || prev_size < new_size) {
        if (pctx->buf_output) {
            ggml_backend_buffer_free(pctx->buf_output);
            pctx->buf_output = nullptr;
            pctx->logits = nullptr;
        }

        pctx->buf_output = ggml_backend_buft_alloc_buffer(pctx->backend_cpu_buffer, new_size);
    }
    
    pctx->logits = (float *) ggml_backend_buffer_get_base(pctx->buf_output);
    //ggml_backend_buffer_clear(pctx->buf_output, 0);
    const auto t_reserved = std::chrono::steady_clock::now();

    ggml_cgraph * gf = build_parler_graph(batch);
    const auto t_built = std::chrono::steady_clock::now();

    // the output is always the last tensor in the graph
    struct ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    pctx->alloc_graph(gf, "parler.decode");
    const auto t_alloc = std::chrono::steady_clock::now();
    
    // use the sequence_length variable here so that audio input tokens are handled correctly.
    size_t n_outputs_new = batch.sequence_length;
    const bool prefill_last_logits_only = parler_prefill_last_logits_only(batch);
    const size_t logits_outputs_new = prefill_last_logits_only ? 1 : n_outputs_new;
    const size_t logits_write_position = prefill_last_logits_only
        ? active_positions - 1
        : (size_t) pctx->n_outputs;

    set_inputs(batch);
    const auto t_inputs = std::chrono::steady_clock::now();

    parler_node_profile_state node_profile;
    const uint32_t profile_skip = parler_profile_node_skip();
    const uint32_t profile_steps = parler_profile_node_steps();
    const bool profile_nodes = batch.audio_generation &&
                               parler_profile_nodes_enabled() &&
                               pctx->profiled_decode_steps >= profile_skip &&
                               pctx->profiled_decode_steps < profile_skip + profile_steps;
    if (profile_nodes) {
        node_profile.step = pctx->profiled_decode_steps;
        node_profile.entries.reserve((size_t) gf->n_nodes);
        for (int i = 0; i < gf->n_nodes; ++i) {
            node_profile.entries.push_back(parler_node_profile_entry{gf->nodes[i], 0.0, 0});
        }
        ggml_backend_sched_set_eval_callback(pctx->sched, parler_node_profile_callback, &node_profile);
    }

    parler_graph_compute(gf);
    const auto t_submit = std::chrono::steady_clock::now();

    if (use_gpu_argmax) {
        pctx->argmax_tokens_scratch.resize(model->n_output_heads);
        pctx->get_ggml_node_data_raw(
            res,
            pctx->argmax_tokens_scratch.data(),
            pctx->argmax_tokens_scratch.size() * sizeof(int32_t));
        for (int32_t token : pctx->argmax_tokens_scratch) {
            pctx->output_tokens.push_back((uint32_t) token);
        }
        pctx->last_decode_used_gpu_argmax = true;
    } else {
        float * logits_out = pctx->logits + logits_write_position * model->output_vocab_size * model->n_output_heads;
        const size_t logits_count = logits_outputs_new * model->output_vocab_size * model->n_output_heads;
        if (logits_outputs_new == 1) {
            pctx->get_ggml_node_data(res, logits_out, logits_count * sizeof(float));
        } else {
            std::vector<float> logits_tmp(logits_count);
            pctx->get_ggml_node_data(res, logits_tmp.data(), logits_count * sizeof(float));
            for (size_t pos = 0; pos < logits_outputs_new; ++pos) {
                for (uint32_t head = 0; head < model->n_output_heads; ++head) {
                    const float * src = logits_tmp.data() + ((size_t) head * logits_outputs_new + pos) * model->output_vocab_size;
                    float * dst = logits_out + (pos * model->n_output_heads + head) * model->output_vocab_size;
                    memcpy(dst, src, model->output_vocab_size * sizeof(float));
                }
            }
        }
    }
    const auto t_read = std::chrono::steady_clock::now();

    if (profile_nodes) {
        ggml_backend_sched_set_eval_callback(pctx->sched, nullptr, nullptr);
        parler_print_node_profile(node_profile);
    }
    if (batch.audio_generation) {
        pctx->profiled_decode_steps += 1;
    }

    // set to total number of outputs in the batch*/
    pctx->n_outputs += n_outputs_new;
    pctx->last_decode_frames = (uint32_t) n_outputs_new;

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(pctx->sched);
    const auto t_done = std::chrono::steady_clock::now();

    if (timing) {
        timing->steps += 1;
        if (batch.audio_generation) {
            timing->audio_steps += 1;
        } else {
            timing->prompt_steps += 1;
        }
        timing->reserve_ms += parler_elapsed_ms(t_start, t_reserved);
        timing->build_ms += parler_elapsed_ms(t_reserved, t_built);
        timing->alloc_ms += parler_elapsed_ms(t_built, t_alloc);
        timing->input_ms += parler_elapsed_ms(t_alloc, t_inputs);
        timing->submit_ms += parler_elapsed_ms(t_inputs, t_submit);
        timing->read_ms += parler_elapsed_ms(t_submit, t_read);
        timing->reset_ms += parler_elapsed_ms(t_read, t_done);
        timing->total_ms += parler_elapsed_ms(t_start, t_done);
    }

    return 0;
}

parler_ubatch parler_tts_runner::build_worst_case_batch()  {
    struct parler_ubatch batch;
    batch.audio_generation = false;
    batch.n_tokens = model->max_ctx_length;
    batch.n_audio_tokens = 0;
    batch.sequence_length = model->max_ctx_length;
    return batch;
}

void parler_tts_runner::prepare_post_load() {
    if (model->use_cross_attn) {
        model->prep_cross_key_values(pctx->n_threads);
    }
    if (parler_fused_qkv_enabled()) {
        model->prep_fused_self_attn_qkv();
    }
    if (parler_fused_heads_enabled()) {
        model->prep_fused_heads();
    }
    dac_runner->prepare_post_load();
    parler_kv_cache_init(kv_self, model, pctx);
    auto batch = build_worst_case_batch();
    auto gf = build_parler_graph(batch);
    pctx->prep_schedule(gf);
}

bool parler_tts_runner::check_stopping() {
    if (model->max_generation_size > 0 &&
        pctx->output_tokens.size() / model->n_output_heads >= model->max_generation_size) {
        return true;
    }

    int32_t token_position = (int32_t) pctx->output_tokens.size() - (int32_t) model->n_output_heads;
    if (token_position < 0) {
        return false;
    }

    bool channels_complete = true;
    for (int i = 0; i < model->n_output_heads; i++) {
        pctx->eos_seen[i] = pctx->eos_seen[i] || pctx->output_tokens[token_position+i] == model->eos_token_id;
        if (channels_complete) {
            channels_complete = pctx->eos_seen[i];
        }
    }
    return channels_complete;
}

static void parler_apply_logits_processor(parler_tts_model * model, parler_context * pctx, parler_ubatch & batch, float * logits) {
    parler_update_first_codebook_unfinished(model, pctx, batch);
    for (uint32_t head = pctx->first_codebook_unfinished + 1; head < model->n_output_heads; ++head) {
        logits[(size_t) head * model->output_vocab_size + model->eos_token_id] = -INFINITY;
    }
}

static uint32_t parler_next_decoder_input_token(parler_tts_model * model, uint32_t next_step, uint32_t head, const uint32_t * last_outputs) {
    if (next_step <= head + 1) {
        return model->bos_token_id;
    }

    // Match HF Parler's delayed-pattern tail: near max_length, later decoder inputs are forced to pad/eos per codebook.
    const uint32_t eos_tail_step = parler_eos_tail_step_for_head(model, head);
    if (next_step > eos_tail_step) {
        return model->eos_token_id;
    }

    return last_outputs[head];
}

void parler_tts_runner::adjust_output_tokens(std::vector<uint32_t> & output_tokens, std::vector<uint32_t> & filtered) {
    // currently this is applying sliding window over the heads and filtering out bad tokens.
    // If we convert the DAC model's quantizer layers to support by row + column embeddings then we will need to transpose
    // the heads and the sequence here, but right now simplying using a strided view is more peformant.
    size_t size = output_tokens.size();
    filtered.reserve(size);
    for (int i = 0; i < size / model->n_output_heads; i++) {
        bool remove = false;
        for (int ii = 0; ii < model->n_output_heads; ii++) {
            int next_index = i*model->n_output_heads+ii*model->n_output_heads+ii;
            if (next_index > size || output_tokens[next_index] >= model->audio_vocab_size) {
                remove = true;
                break;
            }
        }
        if (!remove) {
            for (int ii = 0; ii < model->n_output_heads; ii++) {
                int next_index = i*model->n_output_heads+ii*model->n_output_heads+ii;
                if (next_index > size) {
                    filtered.push_back(model->eos_token_id);
                } else {
                    filtered.push_back(output_tokens[next_index]);
                }
            }
        }
    }
}

int parler_tts_runner::generate_from_batch(parler_ubatch & batch, tts_response & output) {
    const bool debug_timings = parler_debug_timings_enabled();
    decode_timing timing;
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<uint32_t> next_decoder_token_ids;
    next_decoder_token_ids.reserve(model->n_output_heads);

    while (!check_stopping()) {
        const uint32_t unroll_steps = decoder_unroll_steps(batch);
        int state = unroll_steps > 1
            ? decode_unrolled(batch, unroll_steps, debug_timings ? &timing : nullptr)
            : decode(batch, debug_timings ? &timing : nullptr);
        if (state != 0) {
            return state;
        }
        const uint32_t position_advance = std::max<uint32_t>(1, pctx->last_decode_frames);
        const uint32_t audio_step_advance = batch.audio_generation && batch.n_tokens > 0 ? 1 : position_advance;
        if (!batch.audio_generation) {
            pctx->prompt_end_position += batch.sequence_length;
        } else if (batch.n_tokens > 0) {
            pctx->prompt_end_position += batch.n_tokens;
        }
        if (batch.audio_generation && !pctx->last_decode_used_gpu_argmax) {
            const size_t sample_position = (size_t) pctx->current_position + batch.sequence_length - 1;
            float * logits = pctx->logits + sample_position * model->n_output_heads * model->output_vocab_size;
            parler_apply_logits_processor(model, pctx, batch, logits);
            sampler->sample(logits, pctx->output_tokens);
        }
        pctx->current_position += position_advance;
        next_decoder_token_ids.clear();
        uint32_t * last_outputs = (pctx->output_tokens.data() + (int) pctx->output_tokens.size() - model->n_output_heads);
        const uint32_t next_step = (uint32_t) batch.current_step + audio_step_advance;
        for (int i = 0; i < model->n_output_heads; i++) {
            next_decoder_token_ids.push_back(parler_next_decoder_input_token(model, next_step, (uint32_t) i, last_outputs));
        }
        batch = parler_ubatch{
            true, 0, model->n_output_heads, 1, nullptr, next_decoder_token_ids.data(), &pctx->current_position, nullptr, batch.current_step + (int) audio_step_advance
        };
    }

    const auto t_generated = std::chrono::steady_clock::now();
    std::vector<uint32_t> filtered_output_tokens;
    adjust_output_tokens(pctx->output_tokens, filtered_output_tokens);
    const auto t_adjusted = std::chrono::steady_clock::now();
    parler_debug_print_tokens("RAW_TOKENS", pctx->output_tokens, model->n_output_heads);
    parler_debug_print_tokens("FILTERED_TOKENS", filtered_output_tokens, model->n_output_heads);
    dac_runner->run(filtered_output_tokens.data(), (int32_t) filtered_output_tokens.size() / model->n_output_heads, &output);
    const auto t_done = std::chrono::steady_clock::now();
    if (debug_timings) {
        const uint32_t raw_frames = (uint32_t) (pctx->output_tokens.size() / model->n_output_heads);
        const uint32_t filtered_frames = (uint32_t) (filtered_output_tokens.size() / model->n_output_heads);
        fprintf(stderr,
                "PARLER_TIMING steps=%u prompt_steps=%u audio_steps=%u raw_frames=%u filtered_frames=%u "
                "decode_total_ms=%.3f decode_reserve_ms=%.3f decode_build_ms=%.3f decode_alloc_ms=%.3f "
                "decode_input_ms=%.3f decode_submit_ms=%.3f decode_read_ms=%.3f decode_reset_ms=%.3f "
                "generate_loop_ms=%.3f adjust_ms=%.3f dac_ms=%.3f total_ms=%.3f\n",
                timing.steps,
                timing.prompt_steps,
                timing.audio_steps,
                raw_frames,
                filtered_frames,
                timing.total_ms,
                timing.reserve_ms,
                timing.build_ms,
                timing.alloc_ms,
                timing.input_ms,
                timing.submit_ms,
                timing.read_ms,
                timing.reset_ms,
                parler_elapsed_ms(t_start, t_generated),
                parler_elapsed_ms(t_generated, t_adjusted),
                parler_elapsed_ms(t_adjusted, t_done),
                parler_elapsed_ms(t_start, t_done));
    }
    return 0;
}

int parler_tts_runner::generate_audio_tokens(std::string sentence) {
    parler_ubatch batch = batch_from_sentence(sentence, model, tokenizer, true);
    pctx->reset(model->n_output_heads);
    sampler->reset();
    int32_t seq_id = std::mt19937(std::random_device{}())();
    if (!kv_self) {
        kv_self = new parler_kv_cache;
        if (!parler_kv_cache_init(kv_self, model, pctx)) {
            return 1;
        }
    }

    std::vector<uint32_t> next_decoder_token_ids;
    next_decoder_token_ids.reserve(model->n_output_heads);

    while (!check_stopping()) {
        const uint32_t unroll_steps = decoder_unroll_steps(batch);
        int state = unroll_steps > 1 ? decode_unrolled(batch, unroll_steps) : decode(batch);
        if (state != 0) {
            return state;
        }
        const uint32_t position_advance = std::max<uint32_t>(1, pctx->last_decode_frames);
        const uint32_t audio_step_advance = batch.audio_generation && batch.n_tokens > 0 ? 1 : position_advance;
        if (!batch.audio_generation) {
            pctx->prompt_end_position += batch.sequence_length;
        } else if (batch.n_tokens > 0) {
            pctx->prompt_end_position += batch.n_tokens;
        }
        if (batch.audio_generation && !pctx->last_decode_used_gpu_argmax) {
            const size_t sample_position = (size_t) pctx->current_position + batch.sequence_length - 1;
            float * logits = pctx->logits + sample_position * model->n_output_heads * model->output_vocab_size;
            parler_apply_logits_processor(model, pctx, batch, logits);
            sampler->sample(logits, pctx->output_tokens);
        }
        pctx->current_position += position_advance;
        next_decoder_token_ids.clear();
        uint32_t * last_outputs = (pctx->output_tokens.data() + (int) pctx->output_tokens.size() - model->n_output_heads);
        const uint32_t next_step = (uint32_t) batch.current_step + audio_step_advance;
        for (int i = 0; i < model->n_output_heads; i++) {
            next_decoder_token_ids.push_back(parler_next_decoder_input_token(model, next_step, (uint32_t) i, last_outputs));
        }
        batch = parler_ubatch{
            true, 0, model->n_output_heads, 1, nullptr, next_decoder_token_ids.data(), &pctx->current_position, nullptr, batch.current_step + (int) audio_step_advance
        };
    }

    return 0;
}

void parler_tts_runner::just_audio_token_decode(uint32_t * tokens, int32_t sq_len, struct tts_response * outputs) {
    dac_runner->run(tokens, sq_len, outputs);
}

void parler_tts_runner::generate(const char * sentence, tts_response & output,
                                 const generation_configuration & config) {
    const uint32_t previous_max_generation_size = model->max_generation_size;
    sampler->temperature        = config.temperature;
    sampler->repetition_penalty = config.repetition_penalty;
    sampler->do_sample          = config.sample;
    sampler->top_k              = config.top_k;
    sampler->top_p              = config.top_p;
    model->use_cross_attn       = config.use_cross_attn;
    if (config.max_tokens > 0) {
        if ((uint32_t) config.max_tokens < PARLER_MIN_DECODABLE_AUDIO_TOKENS) {
            fprintf(stderr,
                    "Warning: Parler max_tokens=%d is below the DAC minimum of %u; using %u.\n",
                    config.max_tokens,
                    PARLER_MIN_DECODABLE_AUDIO_TOKENS,
                    PARLER_MIN_DECODABLE_AUDIO_TOKENS);
            model->max_generation_size = PARLER_MIN_DECODABLE_AUDIO_TOKENS;
        } else {
            model->max_generation_size = config.max_tokens;
        }
    }

    parler_ubatch batch = batch_from_sentence(sentence, model, tokenizer, true);
    pctx->reset(model->n_output_heads);
    sampler->reset();
    pctx->current_position = 0;
    if (!kv_self) {
        kv_self = new parler_kv_cache;
        if (!parler_kv_cache_init(kv_self, model, pctx)) {
            model->max_generation_size = previous_max_generation_size;
            return;
        }
    }
    generate_from_batch(batch, output);
    model->max_generation_size = previous_max_generation_size;
}
