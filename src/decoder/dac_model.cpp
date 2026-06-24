#include "dac_model.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// For loading DAC model from gguf file.
static const std::map<std::string, dac_tensor> DAC_TENSOR_GGUF_LOOKUP = {
    {"initial.bias", DAC_ENCODER_IN_BIAS},
    {"initial.weight", DAC_ENCODER_IN_KERNEL},
    {"final.bias", DAC_ENCODER_OUT_BIAS},
    {"final.weight", DAC_ENCODER_OUT_KERNEL},
    {"final.alpha", DAC_ENCODER_SNAKE_ALPHA},
};

static bool dac_env_truthy(const char * env) {
    return env && env[0] && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

static bool dac_debug_timings_enabled() {
    return dac_env_truthy(std::getenv("DAC_DEBUG_TIMINGS"));
}

static bool dac_profile_nodes_enabled() {
    return dac_env_truthy(std::getenv("DAC_PROFILE_NODES"));
}

static double dac_elapsed_ms(std::chrono::steady_clock::time_point start,
                             std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct dac_node_profile_entry {
    ggml_tensor * node = nullptr;
    double total_ms = 0.0;
    uint32_t samples = 0;
};

struct dac_node_profile_state {
    std::vector<dac_node_profile_entry> entries;
    ggml_tensor * active_node = nullptr;
    std::chrono::steady_clock::time_point active_start;
};

static int dac_profile_node_index(const dac_node_profile_state * state, ggml_tensor * node) {
    for (size_t i = 0; i < state->entries.size(); ++i) {
        if (state->entries[i].node == node) {
            return (int) i;
        }
    }
    return -1;
}

static bool dac_node_profile_callback(ggml_tensor * node, bool ask, void * user_data) {
    auto * state = static_cast<dac_node_profile_state *>(user_data);
    if (ask) {
        state->active_node = node;
        state->active_start = std::chrono::steady_clock::now();
        return true;
    }
    if (state->active_node) {
        const auto end = std::chrono::steady_clock::now();
        const int index = dac_profile_node_index(state, state->active_node);
        if (index >= 0) {
            state->entries[(size_t) index].total_ms += dac_elapsed_ms(state->active_start, end);
            state->entries[(size_t) index].samples += 1;
        }
    }
    state->active_node = nullptr;
    return true;
}

static void dac_print_node_profile(const dac_node_profile_state & state) {
    double total_ms = 0.0;
    std::map<std::string, double> op_total_ms;
    std::map<std::string, uint32_t> op_samples;
    std::vector<std::pair<size_t, double>> order;
    order.reserve(state.entries.size());
    for (size_t i = 0; i < state.entries.size(); ++i) {
        const auto & entry = state.entries[i];
        if (entry.samples == 0) {
            continue;
        }
        const double avg_ms = entry.total_ms / entry.samples;
        total_ms += entry.total_ms;
        const std::string op = ggml_op_name(entry.node->op);
        op_total_ms[op] += entry.total_ms;
        op_samples[op] += entry.samples;
        order.emplace_back(i, avg_ms);
    }
    std::sort(order.begin(), order.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.second > rhs.second;
    });

    std::vector<std::pair<std::string, double>> op_order(op_total_ms.begin(), op_total_ms.end());
    std::sort(op_order.begin(), op_order.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.second > rhs.second;
    });

    fprintf(stderr, "DAC_NODE_PROFILE nodes=%zu observed=%zu total_ms=%.3f by_op=",
            state.entries.size(), order.size(), total_ms);
    for (size_t i = 0; i < op_order.size(); ++i) {
        if (i > 0) {
            fprintf(stderr, ",");
        }
        const auto & item = op_order[i];
        fprintf(stderr, "%s:%.3f/%u", item.first.c_str(), item.second, op_samples[item.first]);
    }
    fprintf(stderr, "\n");

    const size_t max_nodes = std::min<size_t>(order.size(), 40);
    for (size_t rank = 0; rank < max_nodes; ++rank) {
        const auto & entry = state.entries[order[rank].first];
        const char * name = ggml_get_name(entry.node);
        fprintf(stderr,
                "DAC_NODE_PROFILE_NODE rank=%zu ms=%.3f samples=%u op=%s name=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                rank + 1,
                order[rank].second,
                entry.samples,
                ggml_op_name(entry.node->op),
                name && name[0] ? name : "<unnamed>",
                ggml_type_name(entry.node->type),
                entry.node->ne[0],
                entry.node->ne[1],
                entry.node->ne[2],
                entry.node->ne[3]);
    }
}

void dac_model::prep_constants(gguf_context * meta) {
    int output_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.output_heads", "output_heads", "dia.decoder.output_heads"});
    if (output_heads_key != -1) {
        n_heads = gguf_get_val_u32(meta, output_heads_key);
    }

    int sampling_factor_key = search_for_gguf_keys(meta, {"dac.up_sampling_factor", "up_sampling_factor"});
    if (sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, sampling_factor_key);
    }
    
    int max_gen_key = search_for_gguf_keys(meta, {"parler-tts.decoder.max_generation", "max_generation", "dia.decoder.max_generation"});
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }
}

void dac_model::prep_layers(gguf_context * meta) {
    for (int i = 0; i < n_heads; i++) {
        quantizer_layers.push_back(general_neural_audio_codec::residual_vector_quantize_layer{});
    }
    
    for (int i = 0; i < n_layers; i++) {
        std::string stride_key = "dac_layer_stride_" + std::to_string(i);
        std::string padding_key = "dac_layer_padding_" + std::to_string(i);
        int layer_stride_key = search_for_gguf_keys(meta, {"dac." + stride_key, stride_key});
        if (layer_stride_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file inorder to initialize the DAC audio decoder.", stride_key.c_str());
        }        
        int layer_padding_key = search_for_gguf_keys(meta, {"dac." + padding_key, padding_key});
        if (layer_padding_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file inorder to initialize the DAC audio decoder.", padding_key.c_str());
        }
        layers.push_back(
            general_neural_audio_codec::layer{
                gguf_get_val_u32(meta, layer_padding_key),
                gguf_get_val_u32(meta, layer_stride_key),
            }
        );
    }
}

static uint32_t dac_transpose_kernel_add_on_size(ggml_context * load_context) {
    if (!general_neural_audio_codec::use_col2im_transpose_1d()) {
        return 0;
    }

    size_t bytes = 0;
    for (ggml_tensor * cur = ggml_get_first_tensor(load_context); cur; cur = ggml_get_next_tensor(load_context, cur)) {
        const std::string name = cur->name;
        if (!has_prefix(name, "audio_encoder.decoder_block.")) {
            continue;
        }
        if (name.find(".ru.") != std::string::npos || name.find("quantizers") != std::string::npos) {
            continue;
        }
        if (!has_suffix(name, ".final.weight") && !has_suffix(name, ".weight")) {
            continue;
        }
        bytes += ggml_nbytes(cur);
    }

    constexpr size_t alignment_guard = 1024 * 1024;
    bytes += alignment_guard;
    if (bytes > std::numeric_limits<uint32_t>::max()) {
        TTS_ABORT("DAC col2im transpose kernel add-on is too large: %zu\n", bytes);
    }
    return (uint32_t) bytes;
}

void dac_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only) {
    prep_layers(meta_ctx);
    prep_constants(meta_ctx);
    tts_model::setup_from_file(
        meta_ctx,
        load_context,
        cpu_only,
        "audio_encoder",
        1.4f,
        dac_transpose_kernel_add_on_size(load_context));
}

void dac_model::assign_weight(std::string name, ggml_tensor * tensor) {
    assign_to_audio_encoder(this, name, tensor);
}

void assign_to_audio_encoder(dac_model * model, std::string name, ggml_tensor * tensor) {
    if (DAC_TENSOR_GGUF_LOOKUP.find(name) != DAC_TENSOR_GGUF_LOOKUP.end()) {
        switch(DAC_TENSOR_GGUF_LOOKUP.at(name)) {
            case DAC_ENCODER_IN_BIAS:
                model->in_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(model->in_conv_bias, tensor);
                break;
            case DAC_ENCODER_IN_KERNEL:
                model->in_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->in_conv_kernel, tensor);
                break;
            case DAC_ENCODER_OUT_BIAS:
                model->out_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(model->out_conv_bias, tensor);
                break;
            case DAC_ENCODER_OUT_KERNEL:
                model->out_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->out_conv_kernel, tensor);
                break;
            case DAC_ENCODER_SNAKE_ALPHA:
                model->snake_alpha = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->snake_alpha, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
        auto pair = parse_layer_count(name);
        int l = pair.first;
        std::string lt_name = pair.second;
        if (name.find("quantizers") != std::string::npos) {
            general_neural_audio_codec::assign_to_quantize_layer((tts_model *) model, model->quantizer_layers[l], lt_name, tensor);
        } else {
            general_neural_audio_codec::assign_to_layer((tts_model *) model, model->layers[l - 1], lt_name, tensor);
        }
    }
}

static struct ggml_tensor * dac_build_audio_inputs(struct ggml_context * ctx, struct dac_context * dctx, const dac_ubatch & batch, std::vector<general_neural_audio_codec::residual_vector_quantize_layer> layers) {
    struct ggml_tensor * embd;
    
    dctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length*dctx->model->n_heads);
    ggml_set_input(dctx->inp_tokens);
    dctx->set_tensor_backend(dctx->inp_tokens);

    for(int i = 0; i < dctx->model->n_heads; i++) {
        auto quantize_layer = dctx->model->quantizer_layers[i];
        struct ggml_tensor * code = ggml_view_1d(ctx, dctx->inp_tokens, batch.sequence_length, i*ggml_type_size(GGML_TYPE_I32));
        code->nb[0] = dctx->model->n_heads*ggml_type_size(GGML_TYPE_I32);
        code = general_neural_audio_codec::build_quantize_layer(ctx, code, quantize_layer);

        if (i == 0) {
            embd = code;
        } else {
            embd = ggml_add(ctx, embd, code);
        }
    }
    return embd;
}

struct dac_context * build_new_dac_context(struct dac_model * model, int n_threads, bool use_cpu) {
    dac_context * dctx = new dac_context(model, n_threads);
    dctx->backend = tts_backend_init_accelerator(use_cpu);
    dctx->backend_cpu = ggml_backend_cpu_init();
    dctx->set_threads();
    dctx->build_schedule();
    dctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return dctx;
}

void dac_runner::prepare_post_load() {
    dac_ubatch batch;
    batch.sequence_length = model->max_generation_size;
    ggml_cgraph * gf = build_dac_graph(batch);
    dctx->prep_schedule(gf);
}
    
struct ggml_cgraph * dac_runner::build_dac_graph(dac_ubatch & batch) {
    init_build();
    // splitting this out from the primary graph so that we can better manage streaming (i.e. sentence chunks are better performed this way)
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    
    struct ggml_tensor * cur;
    struct ggml_tensor * inputs;
    
    inputs = dac_build_audio_inputs(ctx, dctx, batch, model->quantizer_layers);
    ggml_set_name(inputs, "quanitzed_inputs");
    
    // everything besides the inputs is just a forward pass
    cur = ggml_conv_1d(ctx, model->in_conv_kernel, inputs, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->in_conv_bias);
    for (auto l : model->layers) {
        cur = general_neural_audio_codec::build_layer(ctx, cur, l);
    }
    cur = snake_1d(ctx, model->snake_alpha, cur);
    cur = ggml_conv_1d(ctx, model->out_conv_kernel, cur, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->out_conv_bias);
    cur = ggml_tanh(ctx, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void dac_runner::run(uint32_t * input_tokens, uint32_t sequence_length, struct tts_response * outputs) {
    const bool debug_timings = dac_debug_timings_enabled();
    const bool profile_nodes = dac_profile_nodes_enabled();
    const auto t_start = std::chrono::steady_clock::now();

    dac_ubatch batch;
    batch.input_tokens = input_tokens;
    batch.sequence_length = sequence_length;
    ggml_backend_sched_reset(dctx->sched);
    const auto t_reset_initial = std::chrono::steady_clock::now();
    
    const size_t prev_size = dctx->buf_output ? ggml_backend_buffer_get_size(dctx->buf_output) : 0;
    const size_t new_size = model->max_generation_size * model->up_sampling_factor * sizeof(float);
    
    if (!dctx->buf_output || prev_size < new_size) {
        if (dctx->buf_output) {
            ggml_backend_buffer_free(dctx->buf_output);
            dctx->buf_output = nullptr;
            dctx->logits = nullptr;
        }

        dctx->buf_output = ggml_backend_buft_alloc_buffer(dctx->backend_cpu_buffer, new_size);
    }
    
    outputs->data = (float *) ggml_backend_buffer_get_base(dctx->buf_output);
    ggml_backend_buffer_clear(dctx->buf_output, 0);
    const auto t_reserved = std::chrono::steady_clock::now();
    
    struct ggml_cgraph * gf = NULL;
    gf = build_dac_graph(batch);
    const auto t_built = std::chrono::steady_clock::now();
    
    // the output is always the last tensor in the graph
    struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    dctx->alloc_graph(gf, "dac.decode");
    const auto t_alloc = std::chrono::steady_clock::now();
    
    ggml_backend_tensor_set(dctx->inp_tokens, batch.input_tokens, 0, batch.sequence_length*model->n_heads*ggml_element_size(dctx->inp_tokens));
    const auto t_inputs = std::chrono::steady_clock::now();

    dac_node_profile_state node_profile;
    if (profile_nodes) {
        node_profile.entries.reserve((size_t) gf->n_nodes);
        for (int i = 0; i < gf->n_nodes; ++i) {
            node_profile.entries.push_back(dac_node_profile_entry{gf->nodes[i], 0.0, 0});
        }
        ggml_backend_sched_set_eval_callback(dctx->sched, dac_node_profile_callback, &node_profile);
    }

    ggml_backend_sched_graph_compute_async(dctx->sched, gf);
    const auto t_submit = std::chrono::steady_clock::now();

    dctx->get_ggml_node_data(result, outputs->data, batch.sequence_length*sizeof(float)*model->up_sampling_factor);
    const auto t_read = std::chrono::steady_clock::now();

    if (profile_nodes) {
        ggml_backend_sched_set_eval_callback(dctx->sched, nullptr, nullptr);
        dac_print_node_profile(node_profile);
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(dctx->sched);
    const auto t_done = std::chrono::steady_clock::now();
    outputs->n_outputs = sequence_length * model->up_sampling_factor;
    if (debug_timings) {
        fprintf(stderr,
                "DAC_TIMING backend=%s frames=%u samples=%zu nodes=%d "
                "reset_ms=%.3f reserve_ms=%.3f build_ms=%.3f alloc_ms=%.3f "
                "input_ms=%.3f submit_ms=%.3f read_ms=%.3f final_reset_ms=%.3f total_ms=%.3f\n",
                dctx->backend ? ggml_backend_name(dctx->backend) : "CPU",
                sequence_length,
                outputs->n_outputs,
                gf->n_nodes,
                dac_elapsed_ms(t_start, t_reset_initial),
                dac_elapsed_ms(t_reset_initial, t_reserved),
                dac_elapsed_ms(t_reserved, t_built),
                dac_elapsed_ms(t_built, t_alloc),
                dac_elapsed_ms(t_alloc, t_inputs),
                dac_elapsed_ms(t_inputs, t_submit),
                dac_elapsed_ms(t_submit, t_read),
                dac_elapsed_ms(t_read, t_done),
                dac_elapsed_ms(t_start, t_done));
    }
    return;
}
