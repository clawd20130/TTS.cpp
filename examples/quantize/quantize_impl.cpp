#include "quantize_impl.h"

#include <array>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common.h"
#include "ggml-cpp.h"
#include "util.h"

static bool kokoro_is_f16_compatible(std::string_view name) {
    return name.find("voice_tensors") == std::string::npos && name.find("bias") == std::string::npos &&
           name.find("gamma") == std::string::npos && name.find("beta") == std::string::npos &&
           name.find("alpha") == std::string::npos && !name.ends_with("embd") && !name.ends_with("norm");
}

static bool is_gguf_internal_tensor(std::string_view name) {
    return name == "GGUF tensor data binary blob";
}

static bool kokoro_is_quantizable(const std::string & name, const quantization_params & params) {
    // A list of all of the top level GGUF names under kokoro.duration_predictor that have quantization compatible tensors.
    static constexpr std::array<std::string_view, 5> DURATION_PREDICTOR_QUANTIZATION_COMPATIBLE_PARTS = {
        "duration_proj", "encode", "shared_lstm", "duration_lstm", "layers"
    };

    if (kokoro_is_f16_compatible(name)) {
        if (name.starts_with("kokoro.albert") || name.starts_with("kokoro.text_encoder.lstm")) {
            return true;
        }
        if (name.starts_with("kokoro.duration_predictor.")) {
            std::vector<std::string> parts = split(name, ".");
            for (const auto part : DURATION_PREDICTOR_QUANTIZATION_COMPATIBLE_PARTS) {
                if (part == parts[2]) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool dia_is_quantizable(std::string_view name, const quantization_params & params) {
    // The DAC audio encoder / decoder is not compatible with quantization and normalization tensors should not be quantized.
    bool quantizable = !name.starts_with("audio_encoder") && !name.ends_with("norm");
    if (!params.quantize_output_heads) {
        quantizable = quantizable && !name.starts_with("dia.decoder.heads");
    }
    return quantizable;
}

static bool parler_quantize_scope_is_default(std::string_view scope) {
    return scope.empty() || scope == "default" || scope == "legacy";
}

static bool parler_quantize_scope_contains(std::string_view scopes, std::string_view target) {
    if (scopes == "all") {
        return true;
    }
    size_t begin = 0;
    while (begin <= scopes.size()) {
        const size_t comma = scopes.find(',', begin);
        const size_t end   = comma == std::string_view::npos ? scopes.size() : comma;
        while (begin < end && scopes[begin] == ' ') {
            begin++;
        }
        size_t trimmed_end = end;
        while (trimmed_end > begin && scopes[trimmed_end - 1] == ' ') {
            trimmed_end--;
        }
        if (scopes.substr(begin, trimmed_end - begin) == target) {
            return true;
        }
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1;
    }
    return false;
}

static bool parler_is_base_quantizable(std::string_view name) {
    return !name.starts_with("audio_encoder") && !name.ends_with("norm.weight") &&
           !name.ends_with("text_encoding") && !name.ends_with("positional_embed") &&
           !name.ends_with("norm.bias");
}

static bool parler_is_scope_quantizable(std::string_view name, const quantization_params & params) {
    if (!parler_is_base_quantizable(name)) {
        return false;
    }

    const std::string_view scopes = params.parler_quantize_scope;
    if (parler_quantize_scope_contains(scopes, "all")) {
        return true;
    }
    if (parler_quantize_scope_contains(scopes, "mlp")) {
        if (name.ends_with(".fc1.weight") || name.ends_with(".fc2.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "attention")) {
        if (name.find(".self_attn.") != std::string_view::npos ||
            name.find(".encoder_attn.") != std::string_view::npos) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "self_attn")) {
        if (name.find(".self_attn.") != std::string_view::npos) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "self_attn_q")) {
        if (name.ends_with(".self_attn.q_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "self_attn_k")) {
        if (name.ends_with(".self_attn.k_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "self_attn_v")) {
        if (name.ends_with(".self_attn.v_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "self_attn_out")) {
        if (name.ends_with(".self_attn.out_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "encoder_attn")) {
        if (name.find(".encoder_attn.") != std::string_view::npos) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "encoder_attn_q")) {
        if (name.ends_with(".encoder_attn.q_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "encoder_attn_k")) {
        if (name.ends_with(".encoder_attn.k_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "encoder_attn_v")) {
        if (name.ends_with(".encoder_attn.v_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "encoder_attn_out")) {
        if (name.ends_with(".encoder_attn.out_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "encoder_attn_kv")) {
        if (name.ends_with(".encoder_attn.k_proj.weight") || name.ends_with(".encoder_attn.v_proj.weight")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "output_heads")) {
        if (name.ends_with("weight.head")) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scopes, "text_embeddings")) {
        if (name.ends_with("embed_prompts")) {
            return true;
        }
    }
    return false;
}

static bool parler_is_quanitizable(std::string_view name, const quantization_params & params) {
    // the DAC audio encoder / decoder is not compatible with quantization, normalization weight shouldn't be quantized, and the text encoding shouldn't be normalized.
    if (!parler_quantize_scope_is_default(params.parler_quantize_scope)) {
        return parler_is_scope_quantizable(name, params);
    }
    bool quantizable = parler_is_base_quantizable(name);
    if (!params.quantize_output_heads) {
        quantizable = quantizable && !name.ends_with("weight.head");
    }
    if (!params.quantize_text_embeddings) {
        quantizable = quantizable && !name.ends_with("embed_prompts");
    }
    if (!params.quantize_cross_attn_kv) {
        quantizable = quantizable && !name.ends_with("encoder_attn.k_proj.weight") &&
                      !name.ends_with("encoder_attn.v_proj.weight");
    }
    return quantizable;
}

static std::string_view jp_bert_tensor_body(std::string_view name) {
    constexpr std::string_view prefix{"style-bert-vits2-jp-bert."};
    if (name.starts_with(prefix)) {
        return name.substr(prefix.size());
    }
    return name;
}

static bool jp_bert_quantize_scope_is_default(std::string_view scope) {
    return scope.empty() || scope == "default" || scope == "linear";
}

static bool jp_bert_is_attention_q_weight(std::string_view body) {
    return body.starts_with("layers.") && body.ends_with(".attn.self.query.weight");
}

static bool jp_bert_is_attention_k_weight(std::string_view body) {
    return body.starts_with("layers.") && body.ends_with(".attn.self.key.weight");
}

static bool jp_bert_is_attention_v_weight(std::string_view body) {
    return body.starts_with("layers.") && body.ends_with(".attn.self.value.weight");
}

static bool jp_bert_is_attention_out_weight(std::string_view body) {
    return body.starts_with("layers.") && body.ends_with(".attn.out.dense.weight");
}

static bool jp_bert_is_attention_weight(std::string_view body) {
    return jp_bert_is_attention_q_weight(body) || jp_bert_is_attention_k_weight(body) ||
           jp_bert_is_attention_v_weight(body) || jp_bert_is_attention_out_weight(body);
}

static bool jp_bert_is_ffn_weight(std::string_view body) {
    return body.starts_with("layers.") &&
           (body.ends_with(".intermediate.dense.weight") || body.ends_with(".output.dense.weight"));
}

static bool jp_bert_is_embedding_weight(std::string_view body) {
    return body == "emb.word.weight" || body == "enc.rel_embeddings.weight";
}

static bool jp_bert_is_conv_weight(std::string_view body) {
    return body == "enc.conv.conv.weight";
}

static bool jp_bert_is_linear_weight(std::string_view body) {
    return jp_bert_is_attention_weight(body) || jp_bert_is_ffn_weight(body);
}

static bool jp_bert_is_non_norm_weight(std::string_view body) {
    return body.ends_with(".weight") && body.find(".norm.") == std::string_view::npos;
}

static bool style_bert_vits2_jp_bert_is_quantizable(std::string_view name, const quantization_params & params) {
    const std::string_view body  = jp_bert_tensor_body(name);
    const std::string_view scope = params.jp_bert_quantize_scope;
    if (jp_bert_quantize_scope_is_default(scope)) {
        return jp_bert_is_linear_weight(body);
    }
    if (parler_quantize_scope_contains(scope, "all_weights")) {
        return jp_bert_is_non_norm_weight(body);
    }
    if (parler_quantize_scope_contains(scope, "linear")) {
        if (jp_bert_is_linear_weight(body)) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scope, "ffn")) {
        if (jp_bert_is_ffn_weight(body)) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scope, "attention")) {
        if (jp_bert_is_attention_weight(body)) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scope, "attention_q")) {
        if (jp_bert_is_attention_q_weight(body)) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scope, "attention_k")) {
        if (jp_bert_is_attention_k_weight(body)) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scope, "attention_v")) {
        if (jp_bert_is_attention_v_weight(body)) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scope, "attention_out")) {
        if (jp_bert_is_attention_out_weight(body)) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scope, "embeddings")) {
        if (jp_bert_is_embedding_weight(body)) {
            return true;
        }
    }
    if (parler_quantize_scope_contains(scope, "conv")) {
        if (jp_bert_is_conv_weight(body)) {
            return true;
        }
    }
    return false;
}

static bool style_bert_vits2_is_quantizable(std::string_view name, const quantization_params & params) {
    if (params.quantize_type != GGML_TYPE_F16 && params.quantize_type != GGML_TYPE_BF16) {
        return false;
    }
    const std::string_view scope = params.style_bert_vits2_quantize_scope;
    if (scope.empty() || parler_quantize_scope_contains(scope, "all")) {
        return true;
    }
    if (parler_quantize_scope_contains(scope, "decoder_weights")) {
        return name.starts_with("style_bert_vits2.decoder.") && name.ends_with(".weight");
    }
    if (parler_quantize_scope_contains(scope, "weights_no_embed_norm")) {
        const bool is_weight = name.ends_with(".weight") || name.ends_with(".w");
        return is_weight && name.find("embedding") == std::string_view::npos &&
               name.find(".norm") == std::string_view::npos && name.find("norm_") == std::string_view::npos;
    }
    if (parler_quantize_scope_contains(scope, "weights_no_embed_norm_no_ups") ||
        parler_quantize_scope_contains(scope, "no_embed_norm_no_ups")) {
        const bool is_weight = name.ends_with(".weight") || name.ends_with(".w");
        return is_weight && name.find("embedding") == std::string_view::npos &&
               name.find(".norm") == std::string_view::npos && name.find("norm_") == std::string_view::npos &&
               !name.starts_with("style_bert_vits2.decoder.ups.");
    }
    return false;
}

static bool is_quantizable(tts_arch arch, const std::string & name, const quantization_params & params) {
    switch (arch) {
        case PARLER_TTS_ARCH:
            return parler_is_quanitizable(name, params);
        case DIA_ARCH:
            return dia_is_quantizable(name, params);
        case KOKORO_ARCH:
            return kokoro_is_quantizable(name, params);
        case STYLE_BERT_VITS2_ARCH:
            return style_bert_vits2_is_quantizable(name, params);
        case STYLE_BERT_VITS2_JP_BERT_ARCH:
            return style_bert_vits2_jp_bert_is_quantizable(name, params);
        default:
            GGML_ABORT("%s failed. The architecture '%d' is not supported.", __func__, arch);
    }
}

static size_t quantize_tensor(void * new_data, const ggml_tensor * tensor, const float * imatrix, ggml_type qtype,
                              uint32_t n_threads) {
    // much of this is form copied from llama.cpp
    int chunk_size_multiplier = 1;
    size_t                   out_size       = 0;
    const int32_t            d3_step        = tensor->ne[0] * tensor->ne[1];
    const int32_t            n_per_row      = tensor->ne[0];
    const int32_t            nrows          = tensor->ne[1];
    static constexpr int32_t min_chunk_size = 32 * 512;
    const int32_t            chunk_size =
        (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1) / n_per_row)) *
        chunk_size_multiplier;
    uint32_t thread_count =
        std::max(1, std::min(static_cast<int>(n_threads), (int) (d3_step + chunk_size - 1) / chunk_size));
    std::mutex mutex;

    for (int32_t d3_index = 0; d3_index < tensor->ne[2]; d3_index++) {
        const float * f32_data_d3 = static_cast<float *>(tensor->data) + d3_index * d3_step;
        void * new_data_d3 = static_cast<char *>(new_data) + ggml_row_size(qtype, tensor->ne[0]) * d3_index * nrows;
        // const float * imatrix_03  = imatrix ? imatrix + d3_index * tensor->ne[0] : nullptr;
        if (thread_count <= 1) {
            // not threaded
            out_size += ggml_quantize_chunk(qtype, f32_data_d3, new_data_d3, 0, nrows, n_per_row, imatrix);
        } else {
            std::vector<std::thread> threads;
            int64_t                  counter  = 0;
            size_t                   new_size = 0;
            bool                     valid    = true;
            for (uint32_t t = 0; t < thread_count; t++) {
                auto func = [&mutex, &counter, &new_size, &valid, qtype, f32_data_d3, new_data_d3, chunk_size, nrows,
                             n_per_row, imatrix]() {
                    const int64_t nrows_per_chunk = chunk_size / n_per_row;
                    size_t        local_size      = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        int64_t                      first_row = counter;
                        counter += nrows_per_chunk;
                        if (first_row >= nrows) {
                            if (local_size > 0) {
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
                        const size_t  this_size = ggml_quantize_chunk(
                            qtype, f32_data_d3, new_data_d3, first_row * n_per_row, this_nrow, n_per_row, imatrix);
                        local_size += this_size;

                        // validate the quantized data; I am not sure how this would occur, but there is always the safe fallback on doing this single threaded.
                        const size_t row_size  = ggml_row_size(qtype, n_per_row);
                        void *       this_data = static_cast<char *>(new_data_d3) + first_row * row_size;
                        if (!ggml_validate_row_data(qtype, this_data, this_size)) {
                            std::unique_lock<std::mutex> lock(mutex);
                            valid = false;
                            break;
                        }
                    }
                };
                threads.emplace_back(func);
            }
            for (auto & t : threads) {
                t.join();
            }

            if (!valid) {
                GGML_ABORT(
                    "Validation of quantized data failed. Please try again and/or switch to single thread "
                    "quantization.\n");
            }
            out_size += new_size;
        }
    }
    return out_size;
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

template <typename T> struct no_init {
    T value;

    no_init() { /* do nothing */ }
};

void quantize_gguf(const char * ifile, const char * ofile, const quantization_params & params) {
    ggml_context *   weight_ctx{};
    gguf_init_params gguf_params{
        .no_alloc{ false },
        .ctx{ &weight_ctx },
    };
    gguf_context * meta_ctx = gguf_init_from_file(ifile, gguf_params);
    std::string    arch     = "parler-tts";  // only parler-tts gguf files should lack an explicit architecture.

    if (int arch_key = gguf_find_key(meta_ctx, "general.architecture"); arch_key != -1) {
        arch = std::string(gguf_get_val_str(meta_ctx, arch_key));
    }
    tts_arch arch_type = SUPPORTED_ARCHITECTURES.at(arch);

    if (params.quantize_type != GGML_TYPE_Q5_0 && params.quantize_type != GGML_TYPE_Q8_0 &&
        params.quantize_type != GGML_TYPE_F16 && params.quantize_type != GGML_TYPE_BF16 &&
        params.quantize_type != GGML_TYPE_Q4_0 && params.quantize_type != GGML_TYPE_Q4_K &&
        params.quantize_type != GGML_TYPE_Q5_K && params.quantize_type != GGML_TYPE_Q6_K) {
        fprintf(stdout, "Warning, %s is untested for quantization type '%d'. Use at your own risk.\n", arch.c_str(),
                params.quantize_type);
    }

    gguf_context_ptr ctx_out{ gguf_init_empty() };

    // copy the KV pairs from the input file
    gguf_set_kv(ctx_out.get(), meta_ctx);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_type", params.quantize_type);
    for (ggml_tensor * tensor = ggml_get_first_tensor(weight_ctx); tensor;
         tensor               = ggml_get_next_tensor(weight_ctx, tensor)) {
        if (*ggml_get_name(tensor) && !is_gguf_internal_tensor(ggml_get_name(tensor))) {
            gguf_add_tensor(ctx_out.get(), tensor);
        }
    }

    std::vector<no_init<uint8_t>> work;

    std::ofstream fout;
    auto          close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_out.get()));
            gguf_get_meta_data(ctx_out.get(), data.data());
            fout.write(reinterpret_cast<const char *>(data.data()), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&]() {
        std::string fname = ofile;
        fout              = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit);  // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_out.get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };
    new_ofstream();
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        ggml_type              new_type;
        void *                 new_data;
        size_t                 new_size;
        const char * const     name = ggml_get_name(cur);
        const std::string_view name_sv{ name };

        if (!*name || is_gguf_internal_tensor(name)) {
            continue;
        }

        if (is_quantizable(arch_type, name, params) && cur->type == GGML_TYPE_F32) {
            new_type = params.quantize_type;
            if ((new_type >= GGML_TYPE_IQ2_XXS && new_type <= GGML_TYPE_IQ4_XS)) {
                GGML_ABORT("ERROR: Quantization type '%d' requires an importance matrix.\n", new_type);
            }
            const int64_t nelement_size = ggml_nelements(cur) * 4;
            if (work.size() < static_cast<size_t>(nelement_size)) {
                work.resize(nelement_size);  // upper bound on size
            }
            new_data = work.data();
            new_size = quantize_tensor(new_data, cur, nullptr, new_type, params.n_threads);
        } else if (((params.convert_non_quantizable_to_f16 && kokoro_is_f16_compatible(name)) ||
                    (params.convert_dac_to_f16 && name_sv.starts_with("audio_encoder") && !name_sv.ends_with("alpha"))) &&
                   cur->type == GGML_TYPE_F32) {
            new_type = GGML_TYPE_F16;
            const int64_t nelement_size = ggml_nelements(cur) * 4;
            if (work.size() < static_cast<size_t>(nelement_size)) {
                work.resize(nelement_size);  // upper bound on size
            }
            new_data = work.data();
            new_size = quantize_tensor(new_data, cur, nullptr, new_type, params.n_threads);
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }

        gguf_set_tensor_type(ctx_out.get(), name, new_type);
        gguf_set_tensor_data(ctx_out.get(), name, new_data);
        fprintf(stdout, "At tensor: '%s' with new size: %zu bytes\n", name, new_size);
        // write tensor data + padding
        fout.write(static_cast<const char *>(new_data), new_size);
        zeros(fout, GGML_PAD(new_size, GGUF_DEFAULT_ALIGNMENT) - new_size);
    }
    close_ofstream();
}
