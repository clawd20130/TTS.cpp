#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#    if defined(TTS_BUILD)
#        define TTS_API __declspec(dllexport)
#    else
#        define TTS_API __declspec(dllimport)
#    endif
#else
#    define TTS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tts_style_bert_vits2_handle tts_style_bert_vits2_handle;
typedef struct tts_style_bert_vits2_jp_bert_handle tts_style_bert_vits2_jp_bert_handle;

typedef struct tts_style_bert_vits2_float_buffer {
    const float * data;
    size_t length;
    uint32_t hidden_size;
    float sample_rate;
} tts_style_bert_vits2_float_buffer;

TTS_API const char * tts_style_bert_vits2_last_error(void);

TTS_API int tts_style_bert_vits2_load_model(const char * model_path,
                                            int n_threads,
                                            int cpu_only,
                                            tts_style_bert_vits2_handle ** out_handle);
TTS_API void tts_style_bert_vits2_free_model(tts_style_bert_vits2_handle * handle);
TTS_API int tts_style_bert_vits2_synthesize_front(tts_style_bert_vits2_handle * handle,
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
                                                  tts_style_bert_vits2_float_buffer * out_audio);

TTS_API int tts_style_bert_vits2_jp_bert_load_model(const char * model_path,
                                                    int n_threads,
                                                    int cpu_only,
                                                    tts_style_bert_vits2_jp_bert_handle ** out_handle);
TTS_API void tts_style_bert_vits2_jp_bert_free_model(tts_style_bert_vits2_jp_bert_handle * handle);
TTS_API int tts_style_bert_vits2_jp_bert_encode_features(tts_style_bert_vits2_jp_bert_handle * handle,
                                                         const int32_t * input_ids,
                                                         size_t tokens,
                                                         tts_style_bert_vits2_float_buffer * out_features);

// Returned buffer pointers are owned by the handle and remain valid until the
// next call on the same handle or until the handle is freed. Callers that need
// longer lifetimes must copy the data before making another call.

#ifdef __cplusplus
}
#endif
