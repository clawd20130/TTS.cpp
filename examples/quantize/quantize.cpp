#include <cstdio>
#include <map>
#include <thread>
#include <vector>

#include "../../src/models/loaders.h"
#include "args.h"
#include "ggml.h"
#include "quantize_impl.h"

const std::map<std::string, ggml_type> valid_quantization_types = {
    {"FP16", GGML_TYPE_F16},
    {"F16", GGML_TYPE_F16},
    {"BF16", GGML_TYPE_BF16},
    {"BFloat16", GGML_TYPE_BF16},
    {"BFLOAT16", GGML_TYPE_BF16},
    {"Q4_0", GGML_TYPE_Q4_0},
    {"Q4", GGML_TYPE_Q4_0},
    {"Q4_K", GGML_TYPE_Q4_K},
    {"Q4K", GGML_TYPE_Q4_K},
    {"Q5_0", GGML_TYPE_Q5_0},
    {"Q5", GGML_TYPE_Q5_0},
    {"Q5_K", GGML_TYPE_Q5_K},
    {"Q5K", GGML_TYPE_Q5_K},
    {"Q6_K", GGML_TYPE_Q6_K},
    {"Q6K", GGML_TYPE_Q6_K},
    {"Q8_0", GGML_TYPE_Q8_0},
    {"Q8", GGML_TYPE_Q8_0},
};

int main(int argc, const char ** argv) {
    int default_n_threads = std::max((int)std::thread::hardware_concurrency(), 1);
    arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini v1 to quantize.", "-mp", true));
    args.add_argument(string_arg("--quantized-model-path", "(REQUIRED) The path to save the model in a quantized format.", "-qp", true));
    args.add_argument(string_arg("--quantized-type", "(OPTIONAL) The ggml enum of the quantized type to convert compatible model tensors to. For more information see readme. Defaults to Q4_0 quantization (2).", "-qt", false, "Q4_0"));
    args.add_argument(int_arg("--n-threads", "(OPTIONAL) The number of cpu threads to run the quantization process with. Defaults to known hardware concurrency.", "-nt", false, &default_n_threads));
    args.add_argument(bool_arg("--convert-dac-to-f16", "(OPTIONAL) Whether to convert the DAC audio decoder model to a 16 bit float.", "-df"));
    args.add_argument(bool_arg("--quantize-output-heads", "(OPTIONAL) Whether to quantize the output heads. Defaults to false and is true when passed (does not accept a parameter).", "-qh"));
    args.add_argument(bool_arg("--quantize-text-embedding", "(OPTIONAL) Whether to quantize the input text embededings (only applicable for Parler TTS). Defaults to false and is true when passed (does not accept a parameter).", "-qe"));
    args.add_argument(bool_arg("--quantize-cross-attn-kv", "(OPTIONAL) Whether to quantize the cross attention keys and values (only applicable for Parler TTS). Defaults to false and is true when passed (does not accept a parameter).", "-qkv"));
    args.add_argument(bool_arg("--convert-non-quantized-to-f16", "(OPTIONAL) Whether or not to convert quantization incompatible tensors to 16 bit precision. Only currently applicable to Kokoro. defaults to false.", "-nqf"));
    args.add_argument(string_arg("--parler-quantize-scope", "(OPTIONAL) Parler-only comma separated tensor scope: default, all, mlp, attention, self_attn, self_attn_q, self_attn_k, self_attn_v, self_attn_out, encoder_attn, encoder_attn_q, encoder_attn_k, encoder_attn_v, encoder_attn_out, encoder_attn_kv, output_heads, text_embeddings. Defaults to default, which preserves the legacy Parler rules.", "-pqs", false, "default"));
    args.add_argument(string_arg("--jp-bert-quantize-scope", "(OPTIONAL) Style-Bert-VITS2 JP-BERT-only comma separated tensor scope: default, linear, ffn, attention, attention_q, attention_k, attention_v, attention_out, embeddings, conv, all_weights. Defaults to linear.", "-jbqs", false, "linear"));
    args.add_argument(string_arg("--style-bert-vits2-quantize-scope", "(OPTIONAL) Style-Bert-VITS2 voice-only comma separated tensor scope: all, decoder_weights, weights_no_embed_norm. Defaults to all.", "-sbv2qs", false, "all"));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();
    std::string qtype = args.get_string_param("--quantized-type");
    if (!valid_quantization_types.contains(qtype)) {
        fprintf(stderr, "ERROR: %s is not a valid quantization type.\n",
                qtype.c_str());
        exit(1);
    }
    quantization_params qp {
        .n_threads{ static_cast<uint32_t>(*args.get_int_param("--n-threads")) },
        .quantize_type{valid_quantization_types.at(qtype)},  // quantization type
        .quantize_output_heads{ args.get_bool_param("--quantize-output-heads")},
        .quantize_text_embeddings{args.get_bool_param("--quantize-text-embedding")},
        .quantize_cross_attn_kv{ args.get_bool_param("--quantize-cross-attn-kv")},
        .convert_dac_to_f16{ args.get_bool_param("--convert-dac-to-f16")},
        .convert_non_quantizable_to_f16{ args.get_bool_param("--convert-non-quantized-to-f16")},
        .parler_quantize_scope{ args.get_string_param("--parler-quantize-scope")},
        .jp_bert_quantize_scope{ args.get_string_param("--jp-bert-quantize-scope")},
        .style_bert_vits2_quantize_scope{ args.get_string_param("--style-bert-vits2-quantize-scope")},
    };
    quantize_gguf(args.get_string_param("--model-path").c_str(), args.get_string_param("--quantized-model-path").c_str(), qp);
    return 0;
}
