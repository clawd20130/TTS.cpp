#include <cmath>
#include <cstdint>
#include <cstdio>

#include "../src/tts_model.h"

namespace {
size_t tensor_offset(ggml_backend_buffer_t buffer, const ggml_tensor * tensor) {
    auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    auto data = reinterpret_cast<uintptr_t>(tensor->data);
    return data - base;
}
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);

    ggml_init_params src_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2 + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * src_ctx = ggml_init(src_params);
    ggml_tensor * src_f16 = ggml_new_tensor_1d(src_ctx, GGML_TYPE_F16, 1);
    ggml_tensor * src_f32 = ggml_new_tensor_1d(src_ctx, GGML_TYPE_F32, 2);
    ggml_set_name(src_f16, "test.f16");
    ggml_set_name(src_f32, "test.f32");
    static_cast<ggml_fp16_t *>(src_f16->data)[0] = GGML_FP32_TO_FP16(0.5f);
    static_cast<float *>(src_f32->data)[0] = 1.25f;
    static_cast<float *>(src_f32->data)[1] = -2.5f;

    ggml_init_params dst_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 3 + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * dst_ctx = ggml_init(dst_params);
    ggml_tensor * dst_f16 = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_F16, 1);
    ggml_tensor * dst_f32 = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_F32, 2);

    tts_model model{};
    model.backend = backend;
    model.buffer = buft;
    model.buf = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(src_f16) + ggml_nbytes(src_f32) + 3 * sizeof(float) + alignment * 3);
    model.set_tensor(dst_f16, src_f16);
    model.set_tensor(dst_f32, src_f32);

    const size_t off_f16 = tensor_offset(model.buf, dst_f16);
    const size_t off_f32 = tensor_offset(model.buf, dst_f32);
    if (off_f16 % alignment != 0 || off_f32 % alignment != 0) {
        std::fprintf(stderr, "unaligned tensor offsets: f16=%zu f32=%zu alignment=%zu\n",
                     off_f16, off_f32, alignment);
        return 1;
    }
    if (off_f32 <= off_f16) {
        std::fprintf(stderr, "unexpected tensor order: f16=%zu f32=%zu\n", off_f16, off_f32);
        return 1;
    }

    float roundtrip[2] = {};
    ggml_backend_tensor_get(dst_f32, roundtrip, 0, sizeof(roundtrip));
    if (std::fabs(roundtrip[0] - 1.25f) > 1e-6f || std::fabs(roundtrip[1] + 2.5f) > 1e-6f) {
        std::fprintf(stderr, "roundtrip mismatch: %f %f\n", roundtrip[0], roundtrip[1]);
        return 1;
    }

    ggml_init_params backend_src_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * backend_src_ctx = ggml_init(backend_src_params);
    ggml_tensor * backend_src = ggml_new_tensor_1d(backend_src_ctx, GGML_TYPE_F32, 3);
    ggml_set_name(backend_src, "test.backend-src");
    ggml_backend_buffer_t backend_src_buffer = ggml_backend_alloc_ctx_tensors_from_buft(backend_src_ctx, buft);
    const float backend_values[3] = {3.5f, -4.25f, 8.0f};
    ggml_backend_tensor_set(backend_src, backend_values, 0, sizeof(backend_values));

    ggml_tensor * dst_backend_copy = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_F32, 3);
    model.set_tensor_from_backend_tensor(dst_backend_copy, backend_src);
    const size_t off_backend_copy = tensor_offset(model.buf, dst_backend_copy);
    if (off_backend_copy % alignment != 0 || off_backend_copy <= off_f32) {
        std::fprintf(stderr, "unexpected backend-copy tensor offset: f32=%zu copy=%zu alignment=%zu\n",
                     off_f32, off_backend_copy, alignment);
        return 1;
    }

    float backend_roundtrip[3] = {};
    ggml_backend_tensor_get(dst_backend_copy, backend_roundtrip, 0, sizeof(backend_roundtrip));
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(backend_roundtrip[i] - backend_values[i]) > 1e-6f) {
            std::fprintf(stderr, "backend tensor roundtrip mismatch at %d: %f != %f\n",
                         i, backend_roundtrip[i], backend_values[i]);
            return 1;
        }
    }

    model.buf = nullptr;
    model.backend = nullptr;
    ggml_backend_buffer_free(backend_src_buffer);
    ggml_free(backend_src_ctx);
    ggml_backend_buffer_free(dst_f16->buffer);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    ggml_backend_free(backend);
    return 0;
}
