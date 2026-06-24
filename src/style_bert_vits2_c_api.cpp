#include "../include/tts_style_bert_vits2_c_api.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "models/loaders.h"
#include "models/style_bert_vits2/model.h"
#include "models/style_bert_vits2_jp_bert/model.h"

struct tts_style_bert_vits2_handle {
    std::unique_ptr<tts_generation_runner> base;
    style_bert_vits2_runner * runner = nullptr;
};

struct tts_style_bert_vits2_jp_bert_handle {
    std::unique_ptr<tts_generation_runner> base;
    style_bert_vits2_jp_bert_runner * runner = nullptr;
    std::vector<float> last_features;
};

namespace {
thread_local std::string last_error;

void set_error(const std::string & message) {
    last_error = message;
}

void set_exception_error(const char * operation) {
    try {
        throw;
    } catch (const std::exception & ex) {
        last_error = std::string(operation) + " failed: " + ex.what();
    } catch (...) {
        last_error = std::string(operation) + " failed with an unknown exception.";
    }
}

void clear_buffer(tts_style_bert_vits2_float_buffer * buffer) {
    if (!buffer) {
        return;
    }
    buffer->data = nullptr;
    buffer->length = 0;
    buffer->hidden_size = 0;
    buffer->sample_rate = 0.0f;
}
}

const char * tts_style_bert_vits2_last_error(void) {
    return last_error.c_str();
}

int tts_style_bert_vits2_load_model(const char * model_path,
                                    int n_threads,
                                    int cpu_only,
                                    tts_style_bert_vits2_handle ** out_handle) {
    if (!out_handle) {
        set_error("out_handle must not be null.");
        return 0;
    }
    *out_handle = nullptr;
    if (!model_path || !model_path[0]) {
        set_error("model_path must not be empty.");
        return 0;
    }

    try {
        generation_configuration config;
        std::unique_ptr<tts_generation_runner> base =
            runner_from_file(model_path, n_threads, config, cpu_only != 0);
        auto * runner = dynamic_cast<style_bert_vits2_runner *>(base.get());
        if (!runner) {
            set_error("Model is not a Style-Bert-VITS2 GGML model.");
            return 0;
        }

        auto handle = std::make_unique<tts_style_bert_vits2_handle>();
        handle->runner = runner;
        handle->base = std::move(base);
        *out_handle = handle.release();
        last_error.clear();
        return 1;
    } catch (...) {
        set_exception_error("tts_style_bert_vits2_load_model");
        return 0;
    }
}

void tts_style_bert_vits2_free_model(tts_style_bert_vits2_handle * handle) {
    delete handle;
}

int tts_style_bert_vits2_synthesize_front(tts_style_bert_vits2_handle * handle,
                                          const int32_t * phone_ids,
                                          const int32_t * tone_ids,
                                          const int32_t * language_ids,
                                          size_t tokens,
                                          const float * bert,
                                          size_t bert_length,
                                          int32_t speaker_id,
                                          int32_t style_id,
                                          float style_weight,
                                          float sdp_ratio,
                                          float length_scale,
                                          float noise_scale,
                                          float noise_w_scale,
                                          tts_style_bert_vits2_float_buffer * out_audio) {
    clear_buffer(out_audio);
    if (!handle || !handle->runner) {
        set_error("Style-Bert-VITS2 handle must not be null.");
        return 0;
    }
    if (!out_audio) {
        set_error("out_audio must not be null.");
        return 0;
    }
    if (!phone_ids || !tone_ids || !language_ids || !bert) {
        set_error("Style-Bert-VITS2 synthesis inputs must not be null.");
        return 0;
    }
    if (tokens == 0) {
        set_error("Style-Bert-VITS2 synthesis token count must not be zero.");
        return 0;
    }

    try {
        std::vector<int32_t> phone(phone_ids, phone_ids + tokens);
        std::vector<int32_t> tone(tone_ids, tone_ids + tokens);
        std::vector<int32_t> language(language_ids, language_ids + tokens);
        std::vector<float> bert_features(bert, bert + bert_length);
        tts_response response{};
        style_bert_vits2_alignment_result alignment;
        std::string error;
        if (!handle->runner->synthesize_front(phone,
                                              tone,
                                              language,
                                              bert_features,
                                              speaker_id,
                                              style_id,
                                              style_weight,
                                              sdp_ratio,
                                              length_scale,
                                              noise_scale,
                                              noise_w_scale,
                                              response,
                                              alignment,
                                              error)) {
            set_error(error.empty() ? "Style-Bert-VITS2 synthesis failed." : error);
            return 0;
        }
        out_audio->data = response.data;
        out_audio->length = response.n_outputs;
        out_audio->hidden_size = response.hidden_size;
        out_audio->sample_rate = handle->runner->sampling_rate;
        last_error.clear();
        return response.data && response.n_outputs > 0;
    } catch (...) {
        set_exception_error("tts_style_bert_vits2_synthesize_front");
        return 0;
    }
}

int tts_style_bert_vits2_jp_bert_load_model(const char * model_path,
                                            int n_threads,
                                            int cpu_only,
                                            tts_style_bert_vits2_jp_bert_handle ** out_handle) {
    if (!out_handle) {
        set_error("out_handle must not be null.");
        return 0;
    }
    *out_handle = nullptr;
    if (!model_path || !model_path[0]) {
        set_error("model_path must not be empty.");
        return 0;
    }

    try {
        generation_configuration config;
        std::unique_ptr<tts_generation_runner> base =
            runner_from_file(model_path, n_threads, config, cpu_only != 0);
        auto * runner = dynamic_cast<style_bert_vits2_jp_bert_runner *>(base.get());
        if (!runner) {
            set_error("Model is not a Style-Bert-VITS2 JP-BERT GGML model.");
            return 0;
        }

        auto handle = std::make_unique<tts_style_bert_vits2_jp_bert_handle>();
        handle->runner = runner;
        handle->base = std::move(base);
        *out_handle = handle.release();
        last_error.clear();
        return 1;
    } catch (...) {
        set_exception_error("tts_style_bert_vits2_jp_bert_load_model");
        return 0;
    }
}

void tts_style_bert_vits2_jp_bert_free_model(tts_style_bert_vits2_jp_bert_handle * handle) {
    delete handle;
}

int tts_style_bert_vits2_jp_bert_encode_features(tts_style_bert_vits2_jp_bert_handle * handle,
                                                 const int32_t * input_ids,
                                                 size_t tokens,
                                                 tts_style_bert_vits2_float_buffer * out_features) {
    clear_buffer(out_features);
    if (!handle || !handle->runner) {
        set_error("Style-Bert-VITS2 JP-BERT handle must not be null.");
        return 0;
    }
    if (!out_features) {
        set_error("out_features must not be null.");
        return 0;
    }
    if (!input_ids) {
        set_error("input_ids must not be null.");
        return 0;
    }
    if (tokens == 0) {
        set_error("Style-Bert-VITS2 JP-BERT input_ids must not be empty.");
        return 0;
    }
    if (tokens > handle->runner->model->max_position_embeddings) {
        set_error("Style-Bert-VITS2 JP-BERT input_ids exceed max_position_embeddings.");
        return 0;
    }
    for (size_t index = 0; index < tokens; ++index) {
        const int32_t token_id = input_ids[index];
        if (token_id < 0 || static_cast<uint32_t>(token_id) >= handle->runner->model->vocab_size) {
            set_error("Style-Bert-VITS2 JP-BERT input_ids contain an out-of-vocabulary token.");
            return 0;
        }
    }

    try {
        handle->runner->encode_features(input_ids,
                                        static_cast<uint32_t>(tokens),
                                        handle->last_features);
        const size_t expected = tokens * handle->runner->model->hidden_size;
        if (handle->last_features.size() != expected) {
            set_error("Style-Bert-VITS2 JP-BERT feature size mismatch.");
            return 0;
        }
        out_features->data = handle->last_features.data();
        out_features->length = handle->last_features.size();
        out_features->hidden_size = handle->runner->model->hidden_size;
        out_features->sample_rate = 0.0f;
        last_error.clear();
        return 1;
    } catch (...) {
        set_exception_error("tts_style_bert_vits2_jp_bert_encode_features");
        return 0;
    }
}
