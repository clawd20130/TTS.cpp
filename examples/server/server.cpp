#include "httplib.h"
#include "ggml.h"
#include "util.h"
#include <cstdio>
#include <format>
#include <string>
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_WAV "audio/wav"
#define MIMETYPE_AIFF "audio/aiff"
#define MIMETYPE_JSON "application/json; charset=utf-8"
#define MIMETYPE_HTML "text/html; charset=utf-8"

#include <signal.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "../../src/models/loaders.h"
#include "../../src/models/style_bert_vits2/model.h"
#include "../../src/models/style_bert_vits2_jp_bert/model.h"
#include "args.h"
#include "audio_file.h"
#include "common.h"
#include "index.html.hpp"
#include "tts_server_threading_osx.h"

enum server_state {
    LOADING,  // Server is starting up / model loading
    READY,    // Server is ready
};

// These are form copied from llama.cpp which copied them from openAI chat:
// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
// In testing, openAI TTS endpoints make use of the same behavior.
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION, // not currently supported as auth keys are not built in yet
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION, // not currently supported as auth keys are not built in yet
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

enum task_type {
    TTS,
    CONDITIONAL_PROMPT,
    VOICES,
    STYLE_BERT_VITS2_DECODE,
    STYLE_BERT_VITS2_SYNTHESIZE_LATENT,
    STYLE_BERT_VITS2_SYNTHESIZE_FRONT,
    STYLE_BERT_VITS2_JP_BERT_FEATURES,
};

using json = nlohmann::ordered_json;

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            fprintf(stderr, "Wrong type supplied for parameter '%s'. Expected '%s', using default value\n", key.c_str(), json(default_value).type_name());
            return default_value;
        }
    } else {
        return default_value;
    }
}

bool write_audio_data(float * data,
                      size_t length,
                      std::vector<uint8_t> & output,
                      AudioFileFormat format = AudioFileFormat::Wave,
                      float sample_rate = 44100.f,
                      float frequency = 440.f,
                      int channels = 1,
                      bool peak_normalize = false) {
    AudioFile<float> file;
    file.setBitDepth(16);
    file.setSampleRate(sample_rate);
    file.setNumChannels(channels);
    int samples = (int) (length / channels);
    file.setNumSamplesPerChannel(samples);
    float scale = 1.0f;
    if (peak_normalize) {
        float peak = 0.0f;
        for (size_t i = 0; i < length; ++i) {
            const float sample = data[i];
            if (std::isfinite(sample)) {
                peak = std::max(peak, std::fabs(sample));
            }
        }
        if (peak > 0.0f) {
            scale = 1.0f / peak;
        }
    }
    for (int channel = 0; channel < channels; channel++) {
        for (int i = 0; i < samples; i++) {
            const size_t index = (size_t) i * (size_t) channels + (size_t) channel;
            float sample = index < length ? data[index] : 0.0f;
            if (!std::isfinite(sample)) {
                sample = 0.0f;
            }
            file.samples[channel][i] = sample * scale;
        }
    }
    return file.writeData(output, format);
}

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    if (req.path == "/v1/health") {
        return;
    }

    fprintf(stdout, "request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);
}

static std::string env_string_or_default(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return value && value[0] ? std::string(value) : std::string(fallback);
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0;
}

static json style_bert_vits2_runtime_config() {
    const bool flow_fused = env_enabled("STYLE_BERT_VITS2_FLOW_FUSED");
    const std::string flow_group_size = env_string_or_default("STYLE_BERT_VITS2_FLOW_GROUP_SIZE", "1");
    std::string attention_mode = env_string_or_default("STYLE_BERT_VITS2_ATTENTION_MODE", "");
    if (attention_mode.empty()) {
        attention_mode = env_string_or_default("STYLE_BERT_VITS2_ATTENTION_EXPERIMENT", "full");
    }
    return {
        {"backend_env", env_string_or_default("TTS_BACKEND", "auto")},
        {"device_env", env_string_or_default("TTS_DEVICE", "")},
        {"attention_mode", attention_mode},
        {"debug_timings", env_enabled("STYLE_BERT_VITS2_DEBUG_TIMINGS")},
        {"flow_fused", flow_fused},
        {"flow_group_size", flow_group_size},
        {"flow_mode", flow_fused ? "fused" : (flow_group_size == "1" ? "step" : "group")},
    };
}

static std::atomic<int> next_server_task_id{1};

struct simple_server_task {
    simple_server_task(task_type task, std::string prompt = ""): task(task), prompt(prompt) {
        id = next_server_task_id.fetch_add(1, std::memory_order_relaxed);
        time = std::chrono::steady_clock::now();
    }

    task_type task;
    int id;
    std::string prompt;
    generation_configuration gen_config;
    void * response;
    size_t length;
    bool success = false;
    std::string message;
    std::chrono::time_point<std::chrono::steady_clock> time;
    float sample_rate = 44100.0f;
    std::string model;
    std::vector<float> style_bert_decoder_z;
    std::vector<float> style_bert_decoder_g;
    uint32_t style_bert_decoder_frames = 0;
    std::vector<float> style_bert_logw;
    std::vector<float> style_bert_x_mask;
    std::vector<float> style_bert_m_p;
    std::vector<float> style_bert_logs_p;
    std::vector<float> style_bert_noise;
    std::vector<float> style_bert_alignment_w;
    std::vector<float> style_bert_alignment_w_ceil;
    uint32_t style_bert_alignment_frames = 0;
    std::vector<float> kokoro_duration_lengths;
    std::vector<uint32_t> kokoro_token_ids;
    uint32_t kokoro_duration_frames = 0;
    uint32_t kokoro_duration_frame_samples = 0;
    std::vector<int32_t> style_bert_phone_ids;
    std::vector<int32_t> style_bert_tone_ids;
    std::vector<int32_t> style_bert_language_ids;
    std::vector<float> style_bert_bert;
    std::vector<int32_t> style_bert_jp_bert_input_ids;
    std::vector<float> style_bert_jp_bert_features;
    uint32_t style_bert_jp_bert_hidden_size = 0;
    uint32_t style_bert_tokens = 0;
    int32_t style_bert_speaker_id = 0;
    int32_t style_bert_style_id = 0;
    float style_bert_style_weight = 1.0f;
    float style_bert_sdp_ratio = 0.0f;
    float style_bert_length_scale = 1.0f;
    float style_bert_noise_scale = 0.6f;
    float style_bert_noise_w_scale = 0.8f;
    bool style_bert_return_alignment = false;
    bool return_alignment = false;

    bool timed_out(int t) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::ratio<1>> duration = now - time;
        return (int) duration.count() > t;
    }
};

static std::string generic_tts_unsupported_message(const tts_generation_runner & runner) {
    const char * arch = runner.loader.get().arch;
    if (std::strcmp(arch, "style-bert-vits2") == 0) {
        return "Style-Bert-VITS2 does not support /v1/audio/speech text generation yet. "
               "Use /v1/style-bert-vits2/synthesize-front or /v1/style-bert-vits2/synthesize-symbols "
               "with explicit frontend tensors.";
    }
    if (std::strcmp(arch, "style-bert-vits2-jp-bert") == 0) {
        return "Style-Bert-VITS2 JP-BERT does not support /v1/audio/speech text generation. "
               "Use /v1/style-bert-vits2/jp-bert/features for BERT feature extraction.";
    }
    return "";
}

struct simple_task_queue {
    std::mutex rw_mutex;
    std::condition_variable condition;
    std::deque<simple_server_task*> queue;
    bool running = true;

    struct simple_server_task * get_next() {
        struct simple_server_task * resp;
        std::unique_lock<std::mutex> lock(rw_mutex);
        condition.wait(lock, [&]{ 
            return !queue.empty() || !running; 
        });
        if (!running) {
            return nullptr;
        }
        resp = queue.front();
        queue.pop_front();
        lock.unlock();
        return resp;
    }

    void terminate() {
        std::lock_guard<std::mutex> lock(rw_mutex);
        running = false; 
        condition.notify_all();
    }

    void push(struct simple_server_task * task) {
        std::lock_guard<std::mutex> lock(rw_mutex);
        queue.push_back(task);
        condition.notify_one();
    }
};

struct simple_response_map {
    std::mutex rw_mutex;
    std::condition_variable updated;
    int cleanup_timeout = 300;
    std::atomic<bool> running = true;
    std::thread * cleanup_thread;

    std::map<int, simple_server_task*> completed;

    void cleanup_routine() {
        std::unique_lock<std::mutex> lock(rw_mutex);
        while(true) {
            updated.wait(lock, [&]{
                return completed.size() > 100 || !running;
            });
            if (!running) {
                return;
            }
            auto now = std::chrono::steady_clock::now();
            std::vector<int> deletable;
            for (auto const& [key, task] : completed) {
                if (task->timed_out(cleanup_timeout)) {
                    deletable.push_back(key);
                }
            }
            for (auto const id : deletable) {
                completed.erase(id);
            }
        }
    }

    void terminate() {
        std::lock_guard<std::mutex> lock(rw_mutex);
        running = false; 
        updated.notify_all();
    }

    void push(struct simple_server_task * task) {
        std::unique_lock<std::mutex> lock(rw_mutex);
        completed[task->id] = task;
        lock.unlock();
        updated.notify_all();
    }

    struct simple_server_task * get(int id) {
        std::unique_lock<std::mutex> lock(rw_mutex);
        updated.wait(lock, [&]{
            return completed.find(id) != completed.end() || !running;
        });
        if (!running) {
            return nullptr;
        }
        struct simple_server_task * resp = completed.at(id);
        completed.erase(id);
        return resp;
    }
};

void init_response_map(simple_response_map * rmap) {
    rmap->cleanup_routine();
}

struct worker {
    worker(struct simple_task_queue * task_queue, struct simple_response_map * response_map, std::string text_encoder_path = "", int task_timeout = 300): task_queue(task_queue), response_map(response_map), text_encoder_path(text_encoder_path), task_timeout(task_timeout) {};
    ~worker() {
        // runners.clear();
        for (auto & runner : views::values(runners)) {
            static_cast<void>(!runner.release()); // TODO the destructor doesn't work yet
        }
    }
    struct simple_task_queue * task_queue;
    struct simple_response_map * response_map;

    unordered_map<string, unique_ptr<tts_generation_runner>> runners{};
    std::string text_encoder_path;
    std::atomic<bool> running = true;
    tts_server_threading::native_thread * thread = nullptr;

    int task_timeout;

    void terminate() {
        running = false;
    }

    void loop() {
        while (running) {
            struct simple_server_task * task = task_queue->get_next();
            if (task) {
                process_task(task);
            }
        }
    }

    const void process_task(struct simple_server_task * task) {
        if (task->timed_out(task_timeout)) {
            return;
        }
        tts_response * data = nullptr;
        tts_generation_runner & runner{*runners[task->model]};
        switch(task->task) {
            case TTS: {
                const std::string unsupported_message = generic_tts_unsupported_message(runner);
                if (!unsupported_message.empty()) {
                    task->success = false;
                    task->message = unsupported_message;
                    response_map->push(task);
                    break;
                }
                data              = new tts_response;
                runner.generate(task->prompt.c_str(), *data, task->gen_config);
                task->response    = (void *) data->data;
                task->length      = data->n_outputs;
                task->sample_rate = runner.sampling_rate;
                task->kokoro_duration_lengths = data->kokoro_duration_lengths;
                task->kokoro_token_ids = data->kokoro_token_ids;
                task->kokoro_duration_frames = data->kokoro_duration_frames;
                task->kokoro_duration_frame_samples = data->kokoro_duration_frame_samples;
                task->success     = data->n_outputs != 0;
                if (!task->success) {
                    task->message = "Model returned an empty response.";
                }
                response_map->push(task);
                break;
            }
            case CONDITIONAL_PROMPT:
                if (text_encoder_path.size() == 0) {
                    task->message = "A text encoder path must be specified on server initialization in order to support conditional prompting.";
                    response_map->push(task);
                    break;
                }
                runner.update_conditional_prompt(text_encoder_path.c_str(), task->prompt.c_str());
                task->success = true;
                response_map->push(task);
                break;
            case VOICES: {
                // Maybe there is a better way to pass the voices rather than
                // needing a custom serialized message?
                // Getting all voices
                std::unordered_map<std::string, std::string> voice_map = {};
                for (const auto &[id, runner] : runners) {
                    if (!runner->supports_voices) {
                        continue;
                    }
                    std::string voices_string{};
                    for (const auto voice : runner->list_voices()) {
                        if (!voices_string.empty()) {
                            voices_string += ",";
                        }
                        voices_string += voice;
                    }
                    voice_map[id] = voices_string;
                }
                // Formatting final message
                for (const auto &[id, voices] : voice_map) {
                    if (!task->message.empty()) {
                        task->message += ";";
                    }
                    task->message += id;
                    task->message += "/";
                    task->message += voices;
                }
                task->success = true;
                response_map->push(task);
                break;
            }
            case STYLE_BERT_VITS2_DECODE: {
                auto * style_runner = dynamic_cast<style_bert_vits2_runner *>(&runner);
                if (!style_runner) {
                    task->message = "Selected model is not a Style-Bert-VITS2 GGML decoder.";
                    response_map->push(task);
                    break;
                }
                data = new tts_response;
                style_runner->decode(task->style_bert_decoder_z.data(),
                                     task->style_bert_decoder_g.data(),
                                     task->style_bert_decoder_frames,
                                     *data);
                task->response    = (void *) data->data;
                task->length      = data->n_outputs;
                task->sample_rate = style_runner->sampling_rate;
                task->success     = data->n_outputs != 0;
                response_map->push(task);
                break;
            }
            case STYLE_BERT_VITS2_SYNTHESIZE_LATENT: {
                auto * style_runner = dynamic_cast<style_bert_vits2_runner *>(&runner);
                if (!style_runner) {
                    task->message = "Selected model is not a Style-Bert-VITS2 GGML model.";
                    response_map->push(task);
                    break;
                }
                data = new tts_response;
                style_bert_vits2_alignment_result alignment;
                std::string error;
                if (!style_runner->synthesize_latent(task->style_bert_logw,
                                                     task->style_bert_x_mask,
                                                     task->style_bert_m_p,
                                                     task->style_bert_logs_p,
                                                     task->style_bert_noise,
                                                     task->style_bert_decoder_g,
                                                     task->style_bert_tokens,
                                                     task->style_bert_length_scale,
                                                     task->style_bert_noise_scale,
                                                     *data,
                                                     alignment,
                                                     error)) {
                    task->message = error.empty() ? "Style-Bert latent synthesis failed." : error;
                    delete data;
                    response_map->push(task);
                    break;
                }
                task->style_bert_alignment_w = alignment.w;
                task->style_bert_alignment_w_ceil = alignment.w_ceil;
                task->style_bert_alignment_frames = alignment.frames;
                task->response    = (void *) data->data;
                task->length      = data->n_outputs;
                task->sample_rate = style_runner->sampling_rate;
                task->success     = true;
                response_map->push(task);
                break;
            }
            case STYLE_BERT_VITS2_SYNTHESIZE_FRONT: {
                auto * style_runner = dynamic_cast<style_bert_vits2_runner *>(&runner);
                if (!style_runner) {
                    task->message = "Selected model is not a Style-Bert-VITS2 GGML model.";
                    response_map->push(task);
                    break;
                }
                data = new tts_response;
                style_bert_vits2_alignment_result alignment;
                std::string error;
                if (!style_runner->synthesize_front(task->style_bert_phone_ids,
                                                    task->style_bert_tone_ids,
                                                    task->style_bert_language_ids,
                                                    task->style_bert_bert,
                                                    task->style_bert_speaker_id,
                                                    task->style_bert_style_id,
                                                    task->style_bert_style_weight,
                                                    task->style_bert_sdp_ratio,
                                                    task->style_bert_length_scale,
                                                    task->style_bert_noise_scale,
                                                    task->style_bert_noise_w_scale,
                                                    *data,
                                                    alignment,
                                                    error)) {
                    task->message = error.empty() ? "Style-Bert front synthesis failed." : error;
                    delete data;
                    response_map->push(task);
                    break;
                }
                task->style_bert_alignment_w = alignment.w;
                task->style_bert_alignment_w_ceil = alignment.w_ceil;
                task->style_bert_alignment_frames = alignment.frames;
                task->response    = (void *) data->data;
                task->length      = data->n_outputs;
                task->sample_rate = style_runner->sampling_rate;
                task->success     = true;
                response_map->push(task);
                break;
            }
            case STYLE_BERT_VITS2_JP_BERT_FEATURES: {
                auto * jp_bert_runner = dynamic_cast<style_bert_vits2_jp_bert_runner *>(&runner);
                if (!jp_bert_runner) {
                    task->message = "Selected model is not a Style-Bert-VITS2 JP BERT GGML model.";
                    response_map->push(task);
                    break;
                }
                const uint32_t tokens = (uint32_t) task->style_bert_jp_bert_input_ids.size();
                if (tokens == 0) {
                    task->message = "Style-Bert JP BERT input_ids must not be empty.";
                    response_map->push(task);
                    break;
                }
                for (size_t i = 0; i < task->style_bert_jp_bert_input_ids.size(); ++i) {
                    const int32_t token_id = task->style_bert_jp_bert_input_ids[i];
                    if (token_id < 0 || (uint32_t) token_id >= jp_bert_runner->model->vocab_size) {
                        task->message = std::format(
                            "Style-Bert JP BERT input_ids[{}]={} is outside vocab_size {}.",
                            i,
                            token_id,
                            jp_bert_runner->model->vocab_size);
                        response_map->push(task);
                        return;
                    }
                }
                if (tokens > jp_bert_runner->model->max_position_embeddings) {
                    task->message = std::format(
                        "Style-Bert JP BERT input_ids length {} exceeds max_position_embeddings {}.",
                        tokens,
                        jp_bert_runner->model->max_position_embeddings);
                    response_map->push(task);
                    break;
                }
                jp_bert_runner->encode_features(task->style_bert_jp_bert_input_ids.data(),
                                                tokens,
                                                task->style_bert_jp_bert_features);
                task->style_bert_tokens = tokens;
                task->style_bert_jp_bert_hidden_size = jp_bert_runner->model->hidden_size;
                task->success = task->style_bert_jp_bert_features.size() ==
                                (size_t) tokens * jp_bert_runner->model->hidden_size;
                if (!task->success) {
                    task->message = std::format(
                        "Style-Bert JP BERT feature size mismatch: got {}, expected {}.",
                        task->style_bert_jp_bert_features.size(),
                        (size_t) tokens * jp_bert_runner->model->hidden_size);
                }
                response_map->push(task);
                break;
            }
        }
    }
};

void init_worker(std::unordered_map<std::string, std::string>* model_path, int n_threads, bool cpu_only, const generation_configuration & config, worker * w) {
    for (const auto &[id, path] : *model_path) {
        w->runners[id] = runner_from_file(path.c_str(), n_threads, config, cpu_only);
    }
    w->loop();
}

typedef std::vector<worker*> worker_pool;

void terminate(worker_pool * pool) {
    for (auto w : *pool) {
        w->terminate();
    }
    if (pool->size() > 0) {
        (*pool)[0]->task_queue->terminate();
        (*pool)[0]->response_map->terminate();
    }
}

void complete(worker_pool * pool) {
    for (auto w : *pool) {
        if (w->thread) {
            w->thread->join();
        }
        delete w;
    }
}

static std::string safe_json_to_str(json data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}

static bool base64_decode_bytes(const std::string & input, std::vector<uint8_t> & output, std::string & error) {
    static const std::array<int16_t, 256> table = [] {
        std::array<int16_t, 256> values{};
        values.fill(-1);
        const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < alphabet.size(); ++i) {
            values[(uint8_t) alphabet[i]] = (int16_t) i;
        }
        return values;
    }();

    output.clear();
    output.reserve((input.size() * 3) / 4);
    int value = 0;
    int bits = -8;
    bool padded = false;
    for (const unsigned char c : input) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            continue;
        }
        if (c == '=') {
            padded = true;
            continue;
        }
        if (padded) {
            error = "base64 data contains non-padding characters after '='.";
            return false;
        }
        const int decoded = table[c];
        if (decoded < 0) {
            error = "base64 data contains an invalid character.";
            return false;
        }
        value = (value << 6) | decoded;
        bits += 6;
        if (bits >= 0) {
            output.push_back((uint8_t) ((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    return true;
}

static std::string base64_encode_bytes(const uint8_t * input, size_t length) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve(((length + 2) / 3) * 4);
    for (size_t i = 0; i < length; i += 3) {
        const uint32_t b0 = input[i];
        const uint32_t b1 = i + 1 < length ? input[i + 1] : 0;
        const uint32_t b2 = i + 2 < length ? input[i + 2] : 0;
        const uint32_t value = (b0 << 16) | (b1 << 8) | b2;
        output.push_back(alphabet[(value >> 18) & 0x3f]);
        output.push_back(alphabet[(value >> 12) & 0x3f]);
        output.push_back(i + 1 < length ? alphabet[(value >> 6) & 0x3f] : '=');
        output.push_back(i + 2 < length ? alphabet[value & 0x3f] : '=');
    }
    return output;
}

template <typename T>
static bool json_binary_array(const json & data, const std::string & key, std::vector<T> & output, std::string & error) {
    const std::string binary_key = key + "_b64";
    if (!data.contains(binary_key)) {
        return false;
    }
    if (!data.at(binary_key).is_string()) {
        error = std::format("the '{}' field must be a base64 string.", binary_key);
        return true;
    }
    std::vector<uint8_t> bytes;
    if (!base64_decode_bytes(data.at(binary_key).get<std::string>(), bytes, error)) {
        error = std::format("the '{}' field is invalid: {}", binary_key, error);
        return true;
    }
    if (bytes.empty()) {
        error = std::format("the '{}' field must not be empty.", binary_key);
        return true;
    }
    if (bytes.size() % sizeof(T) != 0) {
        error = std::format("the '{}' decoded byte length must be a multiple of {}.", binary_key, sizeof(T));
        return true;
    }
    output.resize(bytes.size() / sizeof(T));
    std::memcpy(output.data(), bytes.data(), bytes.size());
    return true;
}

static bool json_float_array(const json & data, const std::string & key, std::vector<float> & output, std::string & error) {
    if (json_binary_array<float>(data, key, output, error)) {
        return error.empty();
    }
    if (!data.contains(key) || !data.at(key).is_array()) {
        error = std::format("the '{}' field is required and must be an array or provide '{}_b64'.", key, key);
        return false;
    }
    output.clear();
    output.reserve(data.at(key).size());
    for (const auto & item : data.at(key)) {
        if (!item.is_number()) {
            error = std::format("the '{}' field must contain only numbers.", key);
            return false;
        }
        output.push_back(item.get<float>());
    }
    if (output.empty()) {
        error = std::format("the '{}' field must not be empty.", key);
        return false;
    }
    return true;
}

static bool json_int_array(const json & data, const std::string & key, std::vector<int32_t> & output, std::string & error) {
    if (json_binary_array<int32_t>(data, key, output, error)) {
        return error.empty();
    }
    if (!data.contains(key) || !data.at(key).is_array()) {
        error = std::format("the '{}' field is required and must be an array or provide '{}_b64'.", key, key);
        return false;
    }
    output.clear();
    output.reserve(data.at(key).size());
    for (const auto & item : data.at(key)) {
        if (!item.is_number_integer()) {
            error = std::format("the '{}' field must contain only integers.", key);
            return false;
        }
        output.push_back(item.get<int32_t>());
    }
    if (output.empty()) {
        error = std::format("the '{}' field must not be empty.", key);
        return false;
    }
    return true;
}

static bool json_string_array(const json & data, const std::string & key, std::vector<std::string> & output, std::string & error) {
    if (!data.contains(key) || !data.at(key).is_array()) {
        error = std::format("the '{}' field is required and must be an array.", key);
        return false;
    }
    output.clear();
    output.reserve(data.at(key).size());
    for (const auto & item : data.at(key)) {
        if (!item.is_string()) {
            error = std::format("the '{}' field must contain only strings.", key);
            return false;
        }
        output.push_back(item.get<std::string>());
    }
    if (output.empty()) {
        error = std::format("the '{}' field must not be empty.", key);
        return false;
    }
    return true;
}

static const std::array<std::string_view, 112> STYLE_BERT_VITS2_SYMBOLS = {
    "_", "AA", "E", "EE", "En", "N", "OO", "V", "a", "a:", "aa", "ae", "ah", "ai", "an", "ang",
    "ao", "aw", "ay", "b", "by", "c", "ch", "d", "dh", "dy", "e", "e:", "eh", "ei", "en", "eng",
    "er", "ey", "f", "g", "gy", "h", "hh", "hy", "i", "i0", "i:", "ia", "ian", "iang", "iao", "ie",
    "ih", "in", "ing", "iong", "ir", "iu", "iy", "j", "jh", "k", "ky", "l", "m", "my", "n", "ng",
    "ny", "o", "o:", "ong", "ou", "ow", "oy", "p", "py", "q", "r", "ry", "s", "sh", "t", "th",
    "ts", "ty", "u", "u:", "ua", "uai", "uan", "uang", "uh", "ui", "un", "uo", "uw", "v", "van",
    "ve", "vn", "w", "x", "y", "z", "zh", "zy", "!", "?", "…", ",", ".", "'", "-", "SP", "UNK",
};

static const std::unordered_map<std::string, int32_t> & style_bert_vits2_symbol_to_id() {
    static const std::unordered_map<std::string, int32_t> map = [] {
        std::unordered_map<std::string, int32_t> values;
        values.reserve(STYLE_BERT_VITS2_SYMBOLS.size());
        for (size_t i = 0; i < STYLE_BERT_VITS2_SYMBOLS.size(); ++i) {
            values.emplace(std::string(STYLE_BERT_VITS2_SYMBOLS[i]), (int32_t) i);
        }
        return values;
    }();
    return map;
}

static bool style_bert_vits2_language_settings(const std::string & language,
                                               int32_t & language_id,
                                               int32_t & tone_start,
                                               int32_t & max_raw_tone,
                                               std::string & error) {
    std::string normalized = language;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return (char) std::toupper(c);
    });
    if (normalized == "ZH") {
        language_id = 0;
        tone_start = 0;
        max_raw_tone = 5;
        return true;
    }
    if (normalized == "JP") {
        language_id = 1;
        tone_start = 6;
        max_raw_tone = 1;
        return true;
    }
    if (normalized == "EN") {
        language_id = 2;
        tone_start = 8;
        max_raw_tone = 3;
        return true;
    }
    error = std::format("unsupported Style-Bert language '{}'; expected ZH, JP, or EN.", language);
    return false;
}

static bool style_bert_vits2_symbols_to_ids(const std::vector<std::string> & phones,
                                            const std::vector<int32_t> & raw_tones,
                                            const std::string & language,
                                            bool add_blank,
                                            std::vector<int32_t> & phone_ids,
                                            std::vector<int32_t> & tone_ids,
                                            std::vector<int32_t> & language_ids,
                                            std::string & error) {
    if (phones.size() != raw_tones.size()) {
        error = std::format("phones and tones must have the same length: phones={}, tones={}.", phones.size(), raw_tones.size());
        return false;
    }
    int32_t language_id = 0;
    int32_t tone_start = 0;
    int32_t max_raw_tone = 0;
    if (!style_bert_vits2_language_settings(language, language_id, tone_start, max_raw_tone, error)) {
        return false;
    }

    const auto & symbol_to_id = style_bert_vits2_symbol_to_id();
    const size_t output_size = add_blank ? phones.size() * 2 + 1 : phones.size();
    phone_ids.clear();
    tone_ids.clear();
    language_ids.clear();
    phone_ids.reserve(output_size);
    tone_ids.reserve(output_size);
    language_ids.reserve(output_size);

    const auto append_blank = [&]() {
        phone_ids.push_back(0);
        tone_ids.push_back(0);
        language_ids.push_back(0);
    };
    if (add_blank) {
        append_blank();
    }
    for (size_t i = 0; i < phones.size(); ++i) {
        const auto found = symbol_to_id.find(phones[i]);
        if (found == symbol_to_id.end()) {
            error = std::format("unknown Style-Bert phone '{}' at index {}.", phones[i], i);
            return false;
        }
        if (raw_tones[i] < 0 || raw_tones[i] > max_raw_tone) {
            error = std::format("Style-Bert {} tone at index {} must be between 0 and {}; got {}.",
                                language,
                                i,
                                max_raw_tone,
                                raw_tones[i]);
            return false;
        }
        phone_ids.push_back(found->second);
        tone_ids.push_back(raw_tones[i] + tone_start);
        language_ids.push_back(language_id);
        if (add_blank) {
            append_blank();
        }
    }
    return true;
}

// this function maybe used outside of server_task_result_error
static json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

static void set_backend_env(const std::string & backend) {
    if (backend.empty()) {
        return;
    }
#ifdef _WIN32
    _putenv_s("TTS_BACKEND", backend.c_str());
#else
    setenv("TTS_BACKEND", backend.c_str(), 1);
#endif
}

int main(int argc, const char ** argv) {
    int default_n_threads = std::max((int)std::thread::hardware_concurrency(), 1);
    int default_http_threads = std::max((int)std::thread::hardware_concurrency() - 1, 3);
    int default_n_parallel = 1;
    int default_port = 8080;
    int default_timeout = 300;
    std::string default_host = "127.0.0.1";
    float default_temperature = 1.0f;
    int default_top_k = 50;
    float default_repetition_penalty = 1.0f;
    float default_top_p = 1.0f;

    arg_list args;
    args.add_argument(float_arg("--temperature", "(OPTIONAL) The temperature to use when generating outputs. Defaults to 1.0.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--topk", "(OPTIONAL) when set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.", "-tk", false, &default_top_k));
    args.add_argument(float_arg("--repetition-penalty", "The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.0.", "-r", false, &default_repetition_penalty));
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file or a directory containing only gguf model files for Parler TTS mini or large v1, Dia, or Kokoro.", "-mp", true));
    args.add_argument(string_arg("--default-model", "(OPTIONAL) The default model to use when multiple models (a directory with multiple GGUF files) are provided. This can be set by giving the path to the model (./models/Kokoro_no_espeak.gguf), the filename (Kokoro_no_espeak.gguf), or the model ID itself (Kokoro_no_espeak).", "-dm", false));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to hardware concurrency.", "-nt", false, &default_n_threads));
    args.add_argument(string_arg("--backend", "(OPTIONAL) Runtime backend: auto, cpu, metal, or vulkan. Overrides TTS_BACKEND.", "-b", false));
    args.add_argument(bool_arg("--use-metal", "(OPTIONAL) Whether to use metal acceleration", "-m"));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.add_argument(string_arg("--text-encoder-path", "(OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.", "-tep", false));
    args.add_argument(string_arg("--ssl-file-cert", "(OPTIONAL) The local path to the PEM encoded ssl cert.", "-sfc", false));
    args.add_argument(string_arg("--ssl-file-key", "(OPTIONAL) The local path to the PEM encoded ssl private key.", "-sfk", false));
    args.add_argument(int_arg("--port", "(OPTIONAL) The port to use. Defaults to 8080.", "-p", false, &default_port));
    args.add_argument(string_arg("--host", "(OPTIONAL) the hostname of the server. Defaults to '127.0.0.1'.", "-h", false, default_host));
    args.add_argument(int_arg("--n-http-threads", "(OPTIONAL) The number of http threads to use. Defaults to hardware concurrency minus 1.", "-ht", false, &default_http_threads));
    args.add_argument(int_arg("--timeout", "(OPTIONAL) The server side timeout on http calls in seconds. Defaults to 300 seconds.", "-t", false, &default_timeout));
    args.add_argument(int_arg("--n-parallelism", "(OPTIONAL) the number of parallel models to run asynchronously. Deafults to 1.", "-np", false, &default_n_parallel));
    args.add_argument(string_arg("--voice", "(OPTIONAL) the default voice to use when generating audio. Only used with applicable models.", "-v", false, ""));
    args.add_argument(string_arg("--espeak-voice-id", "(OPTIONAL) The espeak voice id to use for phonemization. This should only be specified when the correct espeak voice cannot be inferred from the kokoro voice (see #MultiLanguage Configuration in the cli README for more info).", "-eid", false));
    args.add_argument(float_arg("--top-p", "(OPTIONAL) the default sum of probabilities to sample over. Must be a value between 0.0 and 1.0. Defaults to 1.0.", "-tp", false, &default_top_p));

    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();
    set_backend_env(args.get_string_param("--backend"));

    if (*args.get_float_param("--top-p") > 1.0f || *args.get_float_param("--top-p") <= 0.0f) {
        fprintf(stderr, "The '--top-p' value must be between 0.0 and 1.0. It was set to '%.6f'.\n", *args.get_float_param("--top-p"));
        exit(1);
    }

    const generation_configuration default_generation_config{
        args.get_string_param("--voice"),
        *args.get_int_param("--topk"),
        *args.get_float_param("--temperature"),
        *args.get_float_param("--repetition-penalty"),
        !args.get_bool_param("--no-cross-attn"),
        args.get_string_param("--espeak-voice-id"),
        0,
        *args.get_float_param("--top-p")};

    worker_pool * pool = nullptr;
    struct simple_task_queue * tqueue = new simple_task_queue;
    struct simple_response_map * rmap  = new simple_response_map;

    bool conditional_prompt_viable = args.get_string_param("--text-encoder-path").size() > 0 && *args.get_int_param("--n-parallelism") <= 1;

    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (args.get_string_param("--ssl-file-cert") != "" && args.get_string_param("--ssl-file-key") != "") {
        fprintf(stdout, "Running with SSL: key = %s, cert = %s\n", args.get_string_param("--ssl-file-key").c_str(), args.get_string_param("--ssl-file-cert").c_str());
        svr.reset(new httplib::SSLServer(args.get_string_param("--ssl-file-key").c_str(), args.get_string_param("--ssl-file-cert").c_str()));
    } else {
        fprintf(stdout, "Running without SSL\n");
        svr.reset(new httplib::Server());
    }
#else
    if (args.get_string_param("--ssl-file-cert") != "" && args.get_string_param("--ssl-file-key") != "") {
        fprintf(stderr, "Server is built without SSL support\n");
        return 1;
    }
    svr.reset(new httplib::Server());
#endif

    // Models Variables
    std::unordered_map<std::string, std::string> model_map = {};
    const std::string model_path = args.get_string_param("--model-path");
    if (std::filesystem::is_directory(model_path)) {
        for (auto const &entry : std::filesystem::directory_iterator(model_path)) {
            if (!entry.is_directory() && entry.path().extension() == ".gguf") {
                const std::string id = entry.path().stem().string();
                model_map[id] = entry.path().string();
            }
        }
        if (model_map.size() == 0) {
            fprintf(stderr, "No model found in directory %s", model_path.c_str());
            return 1;
        }
    } else {
        const std::filesystem::path path = model_path;
        model_map[path.stem().string()] = path.string();
    }

    auto model_creation = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();

    std::string default_model = "";
    if (args.get_string_param("--default-model") != "") {
        const std::string model = std::filesystem::path { args.get_string_param("--default-model") }.stem().string();
        if (model_map.contains(model)) {
            default_model = model;
        } else {
            fprintf(stderr, "Invalid Default Model Provided: %s", model.c_str());
            return 1;
        }
    } else {
        default_model = model_map.begin()->first;
    }

    std::vector<json> models = {};
    for (const auto &[id, _] : model_map) {
      json model = {{"id", ""},
                    {"object", "model"},
                    {"created", 0},
                    {"owned_by", "tts.cpp"}};
      model["id"] = id;
      model["created"] = model_creation;
      models.push_back(model);
    }
    const json models_json = {{"object", "list"}, {"data", models}};

    // Voices Variables
    json voices_json = nullptr;

    std::atomic<server_state> state{LOADING};

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, const json & error_data) {
        json final_response {{"error", error_data}};
        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
        res.status = json_value(error_data, "code", 500);
    };

    auto res_ok_html = [](httplib::Response & res, const char * const & data) {
        res.set_content(data, MIMETYPE_HTML);
        res.status = 200;
    };

    auto res_ok_json = [](httplib::Response & res, const json & data) {
        res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
        res.status = 200;
    };

    auto res_ok_audio = [](httplib::Response & res, const std::vector<uint8_t> & audio, std::string mime_type) {
        res.set_content((char*)audio.data(), audio.size(), mime_type);
        res.status = 200;
    };

    auto style_bert_alignment_json = [](const simple_server_task * rtask,
                                        const std::vector<uint8_t> & audio,
                                        const std::string & audio_format) {
        json alignment = {
            {"frames", rtask->style_bert_alignment_frames},
            {"w", rtask->style_bert_alignment_w},
            {"w_ceil", rtask->style_bert_alignment_w_ceil},
        };
        return json {
            {"status", "ok"},
            {"audio_format", audio_format},
            {"audio_wav_base64", base64_encode_bytes(audio.data(), audio.size())},
            {"sample_rate", rtask->sample_rate},
            {"audio_samples", rtask->length},
            {"alignment", alignment},
        };
    };

    auto kokoro_alignment_json = [](const simple_server_task * rtask,
                                    const std::vector<uint8_t> & audio,
                                    const std::string & audio_format) {
        const float frame_seconds = rtask->sample_rate > 0.0f ?
            (float)rtask->kokoro_duration_frame_samples / rtask->sample_rate : 0.0f;
        json alignment = {
            {"duration_lengths", rtask->kokoro_duration_lengths},
            {"token_ids", rtask->kokoro_token_ids},
            {"duration_frames", rtask->kokoro_duration_frames},
            {"frame_samples", rtask->kokoro_duration_frame_samples},
            {"frame_seconds", frame_seconds},
        };
        return json {
            {"status", "ok"},
            {"audio_format", audio_format},
            {"audio_wav_base64", base64_encode_bytes(audio.data(), audio.size())},
            {"sample_rate", rtask->sample_rate},
            {"audio_samples", rtask->length},
            {"alignment", alignment},
        };
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        fprintf(stderr, "got exception: %s\n", formatted_error.dump().c_str());
        res_error(res, formatted_error);
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout(*args.get_int_param("--timeout"));
    svr->set_write_timeout(*args.get_int_param("--timeout"));

    auto middleware_server_state = [&res_error, &state](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == LOADING) {
            res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_server_state](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods",     "GET, POST");
            res.set_header("Access-Control-Allow-Headers",     "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    const auto handle_index = [&](const httplib::Request &, httplib::Response & res) {
        res_ok_html(res, reinterpret_cast<const char*>(index_html));
    };

    const auto handle_health = [&](const httplib::Request &, httplib::Response & res) {
        json health = {
            {"status", "ok"},
            {"style_bert_vits2", style_bert_vits2_runtime_config()},
        };
        res_ok_json(res, health);
    };

    const auto handle_tts = [
        &tqueue,
        &rmap,
        &res_error,
        &res_ok_audio,
        &res_ok_json,
        &kokoro_alignment_json,
        &default_generation_config,
        &model_map,
        &default_model
    ](const httplib::Request &req, httplib::Response & res) {
        json data = json::parse(req.body);
        if (!data.contains("input") || !data.at("input").is_string()) {
            json formatted_error = format_error_response("the 'input' field is required for tts generation and must be passed as a string.", ERROR_TYPE_INVALID_REQUEST);
            res_error(res, formatted_error);
            return;
        }

        std::string mime_type = MIMETYPE_WAV;
        AudioFileFormat audio_type = AudioFileFormat::Wave;
        if (data.contains("response_format") && data.at("response_format").is_string()) {
            std::string format = data.at("response_format").get<std::string>();
            if (format != "wav" && format != "wave" && format != "aiff") {
                json formatted_error = format_error_response("Currently 'wav' and 'aiff' are the only supported formats for the 'response_format' field.", ERROR_TYPE_NOT_SUPPORTED);
                res_error(res, formatted_error);
                return;
            } else if (format == "aiff") {
                mime_type = MIMETYPE_AIFF;
                audio_type = AudioFileFormat::Aiff;
            }
        }

        std::string prompt = data.at("input").get<std::string>();
        if (prompt.empty()) {
            json formatted_error = format_error_response("the 'input' field must be a non empty string", ERROR_TYPE_INVALID_REQUEST);
            res_error(res, formatted_error);
            return;
        }
        struct simple_server_task * task = new simple_server_task(TTS, prompt);
        int id = task->id;
        generation_configuration conf{default_generation_config};
        float temp;
        float rep_pen;
        float top_p;
        int top_k;
        if (data.contains("temperature") && data.at("temperature").is_number()) {
            temp = data.at("temperature").get<float>();
            conf.temperature = temp;
        }

        if (data.contains("top_k") && data.at("top_k").is_number()) {
            top_k = data.at("top_k").get<int>();
            conf.top_k = top_k;
        }

        if (data.contains("top_p") && data.at("top_p").is_number()) {
            top_p = data.at("top_p").get<float>();
            conf.top_p = top_p;
        }

        if (data.contains("max_tokens") && data.at("max_tokens").is_number_integer()) {
            conf.max_tokens = data.at("max_tokens").get<int>();
        }

        if (data.contains("repetition_penalty") && data.at("repetition_penalty").is_number()) {
            rep_pen = data.at("repetition_penalty").get<float>();
            conf.repetition_penalty = rep_pen;
        }

        if (data.contains("voice") && data.at("voice").is_string()) {
            conf.voice = data.at("voice").get<std::string>();
        }

        if (data.contains("input_phonemes") && data.at("input_phonemes").is_boolean()) {
            conf.input_phonemes = data.at("input_phonemes").get<bool>();
        }
        if (data.contains("sample") && data.at("sample").is_boolean()) {
            conf.sample = data.at("sample").get<bool>();
        }
        task->return_alignment = json_value(data, "return_alignment", false);

        if (data.contains("model") && data.at("model").is_string()) {
            const std::string model = data.at("model");
            if (!model_map.contains(model)) {
                const std::string message = std::format("Invalid Model: {0}", model);
                json formatted_error = format_error_response(message, ERROR_TYPE_INVALID_REQUEST);
                res_error(res, formatted_error);
                return;
            }
            task->model = data.at("model").get<std::string>();
        } else {
            task->model = default_model;
        }

        task->gen_config = conf;
        tqueue->push(task);
        struct simple_server_task * rtask = rmap->get(id);
        if (!rtask->success) {
            json formatted_error = format_error_response(rtask->message, ERROR_TYPE_SERVER);
            res_error(res, formatted_error);
            return;
        }

        if (rtask->length == 0) {
            json formatted_error = format_error_response("Model returned an empty response.", ERROR_TYPE_SERVER);
            res_error(res, formatted_error);
            return;
        }

        std::vector<uint8_t> audio;
        bool success = write_audio_data((float *)rtask->response, rtask->length, audio, audio_type, rtask->sample_rate);
        if (!success) {
            json formatted_error = format_error_response("failed to write audio data", ERROR_TYPE_SERVER);
            res_error(res, formatted_error);
            return;
        }

        if (rtask->return_alignment) {
            res_ok_json(res, kokoro_alignment_json(rtask, audio, mime_type == MIMETYPE_AIFF ? "aiff" : "wav"));
            return;
        }

        res_ok_audio(res, audio, mime_type);
    };

    const auto handle_conditional = [
        &args,
        &tqueue,
        &rmap,
        &res_error,
        &res_ok_json,
        &model_map,
        &default_model
    ](const httplib::Request & req, httplib::Response & res) {
        if (args.get_string_param("--text-encoder-path").size() == 0) {
            json formatted_error = format_error_response("A '--text-encoder-path' must be specified for conditional generation.", ERROR_TYPE_NOT_SUPPORTED);
            res_error(res, formatted_error);
            return;
        }
        if (*args.get_int_param("--n-parallelism") > 1) {
            json formatted_error = format_error_response("Conditional prompting is not supported for parallelism greater than 1.", ERROR_TYPE_NOT_SUPPORTED);
            res_error(res, formatted_error);
            return;
        }
        json data = json::parse(req.body);
        if (!data.contains("input") || !data.at("input").is_string()) {
            json formatted_error = format_error_response("the 'input' field is required for conditional prompting.", ERROR_TYPE_INVALID_REQUEST);
            res_error(res, formatted_error);
            return;
        }
        std::string prompt = data.at("input").get<std::string>();
        struct simple_server_task * task = new simple_server_task(CONDITIONAL_PROMPT, prompt);

        if (data.contains("model") && data.at("model").is_string()) {
            const std::string model = data.at("model");
            if (!model_map.contains(model)) {
                const std::string message = std::format("Invalid Model: {0}", model);
                json formatted_error = format_error_response(message, ERROR_TYPE_INVALID_REQUEST);
                res_error(res, formatted_error);
                return;
            }
            task->model = data.at("model").get<std::string>();
        } else {
            task->model = default_model;
        }

        int id = task->id;
        tqueue->push(task);
        struct simple_server_task * rtask = rmap->get(id);
        if (!rtask->success) {
            json formatted_error = format_error_response(rtask->message, ERROR_TYPE_SERVER);
            res_error(res, formatted_error);
            return;
        }
        json health = {{"status", "ok"}};
        res_ok_json(res, health);
    };

    const auto handle_models = [
        &args,
        &res_error,
        &res_ok_json,
        &models_json
    ](const httplib::Request & _, httplib::Response & res) {
        res_ok_json(res, models_json);
    };

    const auto handle_voices = [
        &args,
        &tqueue,
        &rmap,
        &res_error,
        &res_ok_json,
        &voices_json,
        &default_model
    ](const httplib::Request & req, httplib::Response & res) {
        // Using Cached Values
        if (!voices_json.is_null()) {
            res_ok_json(res, voices_json);
            return;
        }

        struct simple_server_task * task = new simple_server_task(VOICES);
        // Setting the model to default model (as dummy value) so no new runner is created
        task->model = default_model;

        int id = task->id;
        tqueue->push(task);
        struct simple_server_task * rtask = rmap->get(id);
        if (!rtask->success) {
            json formatted_error;
            if (has_prefix(rtask->message, "Voices are not supported")) {
                formatted_error = format_error_response(rtask->message, ERROR_TYPE_NOT_SUPPORTED);
            } else {
                formatted_error = format_error_response(rtask->message, ERROR_TYPE_SERVER);
            }
            res_error(res, formatted_error);
            return;
        }
        voices_json = json::object();
        std::vector<std::string> model_voices = split(rtask->message, ";");
        for (const std::string entry : model_voices) {
            const std::vector<std::string> entry_split  = split(entry, "/");
            voices_json[entry_split[0]] = split(entry_split[1], ",");
        }
        res_ok_json(res, voices_json);
    };

    const auto handle_style_bert_vits2_decode = [
        &tqueue,
        &rmap,
        &res_error,
        &res_ok_audio,
        &model_map,
        &default_model
    ](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        std::vector<float> decoder_z;
        std::vector<float> decoder_g;
        std::string error;
        if (!json_float_array(data, "decoder_z", decoder_z, error) ||
            !json_float_array(data, "decoder_g", decoder_g, error)) {
            res_error(res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (!data.contains("frames") || !data.at("frames").is_number_integer()) {
            res_error(res, format_error_response("the 'frames' field is required and must be an integer.", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        const int frames = data.at("frames").get<int>();
        if (frames <= 0) {
            res_error(res, format_error_response("the 'frames' field must be greater than 0.", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::string mime_type = MIMETYPE_WAV;
        AudioFileFormat audio_type = AudioFileFormat::Wave;
        if (data.contains("response_format") && data.at("response_format").is_string()) {
            std::string format = data.at("response_format").get<std::string>();
            if (format != "wav" && format != "wave" && format != "aiff") {
                res_error(res, format_error_response("Currently 'wav' and 'aiff' are the only supported formats for the 'response_format' field.", ERROR_TYPE_NOT_SUPPORTED));
                return;
            } else if (format == "aiff") {
                mime_type = MIMETYPE_AIFF;
                audio_type = AudioFileFormat::Aiff;
            }
        }

        struct simple_server_task * task = new simple_server_task(STYLE_BERT_VITS2_DECODE);
        int id = task->id;
        if (data.contains("model") && data.at("model").is_string()) {
            const std::string model = data.at("model");
            if (!model_map.contains(model)) {
                const std::string message = std::format("Invalid Model: {0}", model);
                res_error(res, format_error_response(message, ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            task->model = model;
        } else {
            task->model = default_model;
        }
        task->style_bert_decoder_z = std::move(decoder_z);
        task->style_bert_decoder_g = std::move(decoder_g);
        task->style_bert_decoder_frames = (uint32_t) frames;

        tqueue->push(task);
        struct simple_server_task * rtask = rmap->get(id);
        if (!rtask->success) {
            res_error(res, format_error_response(rtask->message, ERROR_TYPE_SERVER));
            return;
        }

        std::vector<uint8_t> audio;
        bool success = write_audio_data((float *)rtask->response, rtask->length, audio, audio_type, rtask->sample_rate);
        if (!success) {
            res_error(res, format_error_response("failed to write audio data", ERROR_TYPE_SERVER));
            return;
        }
        res_ok_audio(res, audio, mime_type);
    };

    const auto handle_style_bert_vits2_synthesize_latent = [
        &tqueue,
        &rmap,
        &res_error,
        &res_ok_audio,
        &res_ok_json,
        &style_bert_alignment_json,
        &model_map,
        &default_model
    ](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        std::vector<float> logw;
        std::vector<float> x_mask;
        std::vector<float> m_p;
        std::vector<float> logs_p;
        std::vector<float> noise;
        std::vector<float> g;
        std::string error;
        if (!json_float_array(data, "logw", logw, error) ||
            !json_float_array(data, "x_mask", x_mask, error) ||
            !json_float_array(data, "m_p", m_p, error) ||
            !json_float_array(data, "logs_p", logs_p, error) ||
            !json_float_array(data, "noise", noise, error)) {
            res_error(res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (data.contains("g")) {
            if (!json_float_array(data, "g", g, error)) {
                res_error(res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        } else if (!json_float_array(data, "decoder_g", g, error)) {
            res_error(res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        int tokens = (int) logw.size();
        if (data.contains("tokens") && data.at("tokens").is_number_integer()) {
            tokens = data.at("tokens").get<int>();
        }
        if (tokens <= 0) {
            res_error(res, format_error_response("the 'tokens' field must be greater than 0.", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if ((size_t) tokens != logw.size() || (size_t) tokens != x_mask.size()) {
            res_error(res, format_error_response("the 'tokens' field must match logw and x_mask lengths.", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::string mime_type = MIMETYPE_WAV;
        AudioFileFormat audio_type = AudioFileFormat::Wave;
        if (data.contains("response_format") && data.at("response_format").is_string()) {
            std::string format = data.at("response_format").get<std::string>();
            if (format != "wav" && format != "wave" && format != "aiff") {
                res_error(res, format_error_response("Currently 'wav' and 'aiff' are the only supported formats for the 'response_format' field.", ERROR_TYPE_NOT_SUPPORTED));
                return;
            } else if (format == "aiff") {
                mime_type = MIMETYPE_AIFF;
                audio_type = AudioFileFormat::Aiff;
            }
        }

        struct simple_server_task * task = new simple_server_task(STYLE_BERT_VITS2_SYNTHESIZE_LATENT);
        int id = task->id;
        if (data.contains("model") && data.at("model").is_string()) {
            const std::string model = data.at("model");
            if (!model_map.contains(model)) {
                const std::string message = std::format("Invalid Model: {0}", model);
                res_error(res, format_error_response(message, ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            task->model = model;
        } else {
            task->model = default_model;
        }
        task->style_bert_logw = std::move(logw);
        task->style_bert_x_mask = std::move(x_mask);
        task->style_bert_m_p = std::move(m_p);
        task->style_bert_logs_p = std::move(logs_p);
        task->style_bert_noise = std::move(noise);
        task->style_bert_decoder_g = std::move(g);
        task->style_bert_tokens = (uint32_t) tokens;
        task->style_bert_length_scale = json_value(data, "length_scale", 1.0f);
        task->style_bert_noise_scale = json_value(data, "noise_scale", 0.6f);
        task->style_bert_return_alignment = json_value(data, "return_alignment", false);

        tqueue->push(task);
        struct simple_server_task * rtask = rmap->get(id);
        if (!rtask->success) {
            res_error(res, format_error_response(rtask->message, ERROR_TYPE_SERVER));
            return;
        }

        std::vector<uint8_t> audio;
        bool success = write_audio_data((float *)rtask->response, rtask->length, audio, audio_type, rtask->sample_rate);
        if (!success) {
            res_error(res, format_error_response("failed to write audio data", ERROR_TYPE_SERVER));
            return;
        }
        if (task->style_bert_return_alignment) {
            res_ok_json(res, style_bert_alignment_json(rtask, audio, mime_type == MIMETYPE_AIFF ? "aiff" : "wav"));
            return;
        }
        res_ok_audio(res, audio, mime_type);
    };

    const auto handle_style_bert_vits2_synthesize_front = [
        &tqueue,
        &rmap,
        &res_error,
        &res_ok_audio,
        &res_ok_json,
        &style_bert_alignment_json,
        &model_map,
        &default_model
    ](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        std::vector<int32_t> phone_ids;
        std::vector<int32_t> tone_ids;
        std::vector<int32_t> language_ids;
        std::vector<float> bert;
        std::string error;
        if (!json_int_array(data, "phone_ids", phone_ids, error) ||
            !json_int_array(data, "tone_ids", tone_ids, error) ||
            !json_int_array(data, "language_ids", language_ids, error) ||
            !json_float_array(data, "bert", bert, error)) {
            res_error(res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (phone_ids.size() != tone_ids.size() || phone_ids.size() != language_ids.size()) {
            res_error(res, format_error_response("phone_ids, tone_ids, and language_ids must have the same length.",
                                                 ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (bert.size() != phone_ids.size() * 1024) {
            res_error(res, format_error_response(std::format("bert must contain tokens*1024 floats: got {}, expected {}.",
                                                             bert.size(),
                                                             phone_ids.size() * 1024),
                                                 ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        const float sdp_ratio = json_value(data, "sdp_ratio", 0.0f);

        std::string mime_type = MIMETYPE_WAV;
        AudioFileFormat audio_type = AudioFileFormat::Wave;
        if (data.contains("response_format") && data.at("response_format").is_string()) {
            std::string format = data.at("response_format").get<std::string>();
            if (format != "wav" && format != "wave" && format != "aiff") {
                res_error(res, format_error_response("Currently 'wav' and 'aiff' are the only supported formats for the 'response_format' field.", ERROR_TYPE_NOT_SUPPORTED));
                return;
            } else if (format == "aiff") {
                mime_type = MIMETYPE_AIFF;
                audio_type = AudioFileFormat::Aiff;
            }
        }

        struct simple_server_task * task = new simple_server_task(STYLE_BERT_VITS2_SYNTHESIZE_FRONT);
        int id = task->id;
        if (data.contains("model") && data.at("model").is_string()) {
            const std::string model = data.at("model");
            if (!model_map.contains(model)) {
                const std::string message = std::format("Invalid Model: {0}", model);
                res_error(res, format_error_response(message, ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            task->model = model;
        } else {
            task->model = default_model;
        }
        task->style_bert_tokens = (uint32_t) phone_ids.size();
        task->style_bert_phone_ids = std::move(phone_ids);
        task->style_bert_tone_ids = std::move(tone_ids);
        task->style_bert_language_ids = std::move(language_ids);
        task->style_bert_bert = std::move(bert);
        task->style_bert_speaker_id = json_value(data, "speaker_id", 0);
        task->style_bert_style_id = json_value(data, "style_id", 0);
        task->style_bert_style_weight = json_value(data, "style_weight", 1.0f);
        task->style_bert_sdp_ratio = sdp_ratio;
        task->style_bert_length_scale = json_value(data, "length_scale", 1.0f);
        task->style_bert_noise_scale = json_value(data, "noise_scale", 0.6f);
        task->style_bert_noise_w_scale = json_value(
            data,
            "sdp_noise_scale",
            json_value(data, "noise_w_scale", json_value(data, "noise_w", 0.8f)));
        task->style_bert_return_alignment = json_value(data, "return_alignment", false);

        tqueue->push(task);
        struct simple_server_task * rtask = rmap->get(id);
        if (!rtask->success) {
            res_error(res, format_error_response(rtask->message, ERROR_TYPE_SERVER));
            return;
        }

        std::vector<uint8_t> audio;
        bool success = write_audio_data((float *)rtask->response, rtask->length, audio, audio_type, rtask->sample_rate);
        if (!success) {
            res_error(res, format_error_response("failed to write audio data", ERROR_TYPE_SERVER));
            return;
        }
        if (task->style_bert_return_alignment) {
            res_ok_json(res, style_bert_alignment_json(rtask, audio, mime_type == MIMETYPE_AIFF ? "aiff" : "wav"));
            return;
        }
        res_ok_audio(res, audio, mime_type);
    };

    const auto handle_style_bert_vits2_synthesize_symbols = [
        &tqueue,
        &rmap,
        &res_error,
        &res_ok_audio,
        &res_ok_json,
        &style_bert_alignment_json,
        &model_map,
        &default_model
    ](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        std::vector<std::string> phones;
        std::vector<int32_t> raw_tones;
        std::vector<float> bert;
        std::string error;
        if (!json_string_array(data, "phones", phones, error) ||
            !json_int_array(data, "tones", raw_tones, error) ||
            !json_float_array(data, "bert", bert, error)) {
            res_error(res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        const std::string language = json_value<std::string>(data, "language", "JP");
        const bool add_blank = json_value(data, "add_blank", false);
        std::vector<int32_t> phone_ids;
        std::vector<int32_t> tone_ids;
        std::vector<int32_t> language_ids;
        if (!style_bert_vits2_symbols_to_ids(phones,
                                             raw_tones,
                                             language,
                                             add_blank,
                                             phone_ids,
                                             tone_ids,
                                             language_ids,
                                             error)) {
            res_error(res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        if (bert.size() != phone_ids.size() * 1024) {
            res_error(res, format_error_response(std::format("bert must contain converted_tokens*1024 floats: got {}, expected {}.",
                                                             bert.size(),
                                                             phone_ids.size() * 1024),
                                                 ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        const float sdp_ratio = json_value(data, "sdp_ratio", 0.0f);

        std::string mime_type = MIMETYPE_WAV;
        AudioFileFormat audio_type = AudioFileFormat::Wave;
        if (data.contains("response_format") && data.at("response_format").is_string()) {
            std::string format = data.at("response_format").get<std::string>();
            if (format != "wav" && format != "wave" && format != "aiff") {
                res_error(res, format_error_response("Currently 'wav' and 'aiff' are the only supported formats for the 'response_format' field.", ERROR_TYPE_NOT_SUPPORTED));
                return;
            } else if (format == "aiff") {
                mime_type = MIMETYPE_AIFF;
                audio_type = AudioFileFormat::Aiff;
            }
        }

        struct simple_server_task * task = new simple_server_task(STYLE_BERT_VITS2_SYNTHESIZE_FRONT);
        int id = task->id;
        if (data.contains("model") && data.at("model").is_string()) {
            const std::string model = data.at("model");
            if (!model_map.contains(model)) {
                const std::string message = std::format("Invalid Model: {0}", model);
                res_error(res, format_error_response(message, ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            task->model = model;
        } else {
            task->model = default_model;
        }
        task->style_bert_tokens = (uint32_t) phone_ids.size();
        task->style_bert_phone_ids = std::move(phone_ids);
        task->style_bert_tone_ids = std::move(tone_ids);
        task->style_bert_language_ids = std::move(language_ids);
        task->style_bert_bert = std::move(bert);
        task->style_bert_speaker_id = json_value(data, "speaker_id", 0);
        task->style_bert_style_id = json_value(data, "style_id", 0);
        task->style_bert_style_weight = json_value(data, "style_weight", 1.0f);
        task->style_bert_sdp_ratio = sdp_ratio;
        task->style_bert_length_scale = json_value(data, "length_scale", 1.0f);
        task->style_bert_noise_scale = json_value(data, "noise_scale", 0.6f);
        task->style_bert_noise_w_scale = json_value(
            data,
            "sdp_noise_scale",
            json_value(data, "noise_w_scale", json_value(data, "noise_w", 0.8f)));
        task->style_bert_return_alignment = json_value(data, "return_alignment", false);

        tqueue->push(task);
        struct simple_server_task * rtask = rmap->get(id);
        if (!rtask->success) {
            res_error(res, format_error_response(rtask->message, ERROR_TYPE_SERVER));
            return;
        }

        std::vector<uint8_t> audio;
        bool success = write_audio_data((float *)rtask->response, rtask->length, audio, audio_type, rtask->sample_rate);
        if (!success) {
            res_error(res, format_error_response("failed to write audio data", ERROR_TYPE_SERVER));
            return;
        }
        if (task->style_bert_return_alignment) {
            res_ok_json(res, style_bert_alignment_json(rtask, audio, mime_type == MIMETYPE_AIFF ? "aiff" : "wav"));
            return;
        }
        res_ok_audio(res, audio, mime_type);
    };

    const auto handle_style_bert_vits2_jp_bert_features = [
        &tqueue,
        &rmap,
        &res_error,
        &res_ok_json,
        &model_map,
        &default_model
    ](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);

        std::vector<int32_t> input_ids;
        std::string error;
        if (!json_int_array(data, "input_ids", input_ids, error)) {
            res_error(res, format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        struct simple_server_task * task = new simple_server_task(STYLE_BERT_VITS2_JP_BERT_FEATURES);
        int id = task->id;
        if (data.contains("model") && data.at("model").is_string()) {
            const std::string model = data.at("model");
            if (!model_map.contains(model)) {
                const std::string message = std::format("Invalid Model: {0}", model);
                res_error(res, format_error_response(message, ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            task->model = model;
        } else {
            task->model = default_model;
        }
        task->style_bert_jp_bert_input_ids = std::move(input_ids);

        tqueue->push(task);
        struct simple_server_task * rtask = rmap->get(id);
        if (!rtask->success) {
            res_error(res, format_error_response(rtask->message, ERROR_TYPE_SERVER));
            return;
        }

        const auto * bytes = reinterpret_cast<const uint8_t *>(rtask->style_bert_jp_bert_features.data());
        const size_t byte_count = rtask->style_bert_jp_bert_features.size() * sizeof(float);
        json response = {
            {"status", "ok"},
            {"model", rtask->model},
            {"tokens", rtask->style_bert_tokens},
            {"hidden_size", rtask->style_bert_jp_bert_hidden_size},
            {"dtype", "float32"},
            {"layout", "hidden_major"},
            {"features_b64", base64_encode_bytes(bytes, byte_count)},
        };
        res_ok_json(res, response);
    };

    // register API routes
    svr->Get("/", handle_index);
    svr->Get("/health", handle_health);
    svr->Post("/v1/audio/speech", handle_tts);
    svr->Post("/v1/style-bert-vits2/decode", handle_style_bert_vits2_decode);
    svr->Post("/v1/style-bert-vits2/synthesize-latent", handle_style_bert_vits2_synthesize_latent);
    svr->Post("/v1/style-bert-vits2/synthesize-front", handle_style_bert_vits2_synthesize_front);
    svr->Post("/v1/style-bert-vits2/synthesize-symbols", handle_style_bert_vits2_synthesize_symbols);
    svr->Post("/v1/style-bert-vits2/jp-bert/features", handle_style_bert_vits2_jp_bert_features);
    svr->Post("/v1/audio/conditional-prompt", handle_conditional);
    svr->Get("/v1/models", handle_models);
    svr->Get("/v1/audio/voices", handle_voices);

    // Start the server
    svr->new_task_queue = [&args] { 
        return new httplib::ThreadPool(*args.get_int_param("--n-http-threads")); 
    };

    // clean up function, to be called before exit
    auto clean_up = [&svr]() {
        svr->stop();
    };

    // bind HTTP listen port
    bool bound = svr->bind_to_port(args.get_string_param("--host"), *args.get_int_param("--port"));

    if (!bound) {
        fprintf(stderr, "%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n", __func__, args.get_string_param("--host").c_str(), *args.get_int_param("--port"));
        clean_up();
        return 1;
    }

    rmap->cleanup_timeout = *args.get_int_param("--timeout");
    rmap->cleanup_thread = new std::thread(init_response_map, rmap);

    // run the HTTP server in a thread
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();
    fprintf(stdout, "%s: HTTP server is listening, hostname: %s, port: %d, http threads: %d\n", __func__, args.get_string_param("--host").c_str(), *args.get_int_param("--port"), *args.get_int_param("--n-http-threads"));


    pool = new worker_pool;
    shutdown_handler = [&](int) {
        // this should unblock the primary thread;
        terminate(pool);
        return;
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    fprintf(stdout, "%s: loading model and initializing main loop\n", __func__);
    // It might make sense in the long run to have the primary thread run clean up on the response map and keep the model workers parallel.    
    for (int i = *args.get_int_param("--n-parallelism"); i > 0; i--) {
        if (i == 1) {
            fprintf(stdout, "%s: server is listening on http://%s:%d\n", __func__, args.get_string_param("--host").c_str(), *args.get_int_param("--port"));
            worker * w = new worker(tqueue, rmap, args.get_string_param("--text-encoder-path"), *args.get_int_param("--timeout"));
            state.store(READY);
            pool->push_back(w);
            init_worker(&model_map, *args.get_int_param("--n-threads"), !args.get_bool_param("--use-metal"), default_generation_config, w);
        } else {
            worker * w = new worker(tqueue, rmap, args.get_string_param("--text-encoder-path"), *args.get_int_param("--timeout"));
            w->thread = new tts_server_threading::native_thread(init_worker, &model_map, *args.get_int_param("--n-threads"), !args.get_bool_param("--use-metal"), default_generation_config, w);
            pool->push_back(w);
        }
    }
    fprintf(stdout, "%s: HTTP server listening on hostname: %s and port: %d, is shutting down.\n", __func__, args.get_string_param("--host").c_str(), *args.get_int_param("--port"));
    svr->stop();
    t.join();
    complete(pool);
    rmap->cleanup_thread->join();

    return 0;
}
