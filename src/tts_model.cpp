#include "tts_model.h"
#include "llama-mmap.h"

#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include "models/loaders.h"

namespace {
bool tts_iequals(const char * a, const char * b) {
    if (!a || !b) {
        return false;
    }
    while (*a && *b) {
        if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

bool tts_parse_non_negative_int(const char * s, int & out) {
    if (!s || s[0] == '\0') {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 0 || v > INT_MAX) {
        return false;
    }

    out = (int) v;
    return true;
}

const char * tts_backend_env() {
    const char * env = std::getenv("TTS_BACKEND");
    return env && env[0] ? env : nullptr;
}

int tts_backend_device_index() {
    const char * env = std::getenv("TTS_DEVICE");
    if (!env || env[0] == '\0') {
        return -1;
    }

    int parsed = -1;
    if (!tts_parse_non_negative_int(env, parsed)) {
        fprintf(stderr, "  [backend] Invalid TTS_DEVICE=%s, using default device\n", env);
        return -1;
    }
    return parsed;
}

bool tts_backend_strict() {
    const char * env = std::getenv("TTS_BACKEND_STRICT");
    return env && env[0] && !tts_iequals(env, "0") && !tts_iequals(env, "false");
}

ggml_backend_t tts_init_named_backend(const char * target_backend_name) {
    const int target_idx = tts_backend_device_index();
    int matched_devices = 0;

    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        if (!reg_name || !tts_iequals(reg_name, target_backend_name)) {
            continue;
        }

        if (target_idx >= 0 && matched_devices != target_idx) {
            ++matched_devices;
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend) {
            return backend;
        }

        if (target_idx >= 0) {
            break;
        }
        ++matched_devices;
    }

    if (target_idx >= 0) {
        fprintf(stderr, "  [backend] Requested %s device index %d not available\n",
                target_backend_name, target_idx);
    }
    return nullptr;
}

ggml_backend_t tts_init_requested_accelerator() {
    const char * env = tts_backend_env();
    ggml_backend_t backend = nullptr;

    if (!env || tts_iequals(env, "auto") || tts_iequals(env, "gpu")) {
        backend = ggml_backend_init_best();
    } else if (tts_iequals(env, "vulkan") || tts_iequals(env, "vk")) {
        backend = tts_init_named_backend("Vulkan");
    } else if (tts_iequals(env, "metal")) {
#ifdef GGML_USE_METAL
        const int target_idx = tts_backend_device_index();
        if (target_idx <= 0) {
            backend = ggml_backend_metal_init();
        } else {
            fprintf(stderr, "  [backend] Requested Metal device index %d not available\n", target_idx);
        }
#else
        backend = tts_init_named_backend("Metal");
#endif
    } else if (tts_iequals(env, "cpu")) {
        return nullptr;
    } else {
        fprintf(stderr, "  [backend] Unknown TTS_BACKEND=%s, using auto\n", env);
        backend = ggml_backend_init_best();
    }

    if (!backend) {
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_free(backend);
        return nullptr;
    }

    return backend;
}

bool tts_should_try_accelerator(bool cpu_only) {
    const char * env = tts_backend_env();
    if (!env) {
        return !cpu_only;
    }
    return !tts_iequals(env, "cpu");
}

size_t tts_pad_to_alignment(size_t offset, size_t alignment) {
    if (alignment == 0) {
        return offset;
    }
    TTS_ASSERT((alignment & (alignment - 1)) == 0);
    return (offset + alignment - 1) & ~(alignment - 1);
}
}

ggml_backend_t tts_backend_init_accelerator(bool cpu_only) {
    if (!tts_should_try_accelerator(cpu_only)) {
        return nullptr;
    }

    ggml_backend_t backend = tts_init_requested_accelerator();
    if (!backend && tts_backend_strict()) {
        TTS_ABORT("Failed to initialize requested TTS accelerator backend. Set TTS_BACKEND=cpu to force CPU.\n");
    }
    return backend;
}

ggml_backend_t tts_backend_init_model(bool cpu_only) {
    ggml_backend_t backend = tts_backend_init_accelerator(cpu_only);
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    return backend;
}

ggml_backend_buffer_type_t tts_backend_get_buffer_type(ggml_backend_t backend) {
    return backend ? ggml_backend_get_default_buffer_type(backend) : ggml_backend_cpu_buffer_type();
}

static bool tts_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW ||
        op == GGML_OP_RESHAPE ||
        op == GGML_OP_PERMUTE ||
        op == GGML_OP_TRANSPOSE;
}

static bool tts_backend_is_cpu(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

void tts_backend_sched_alloc_graph_checked(ggml_backend_sched_t sched, ggml_backend_t required_backend,
                                           ggml_cgraph * gf, const char * graph_name) {
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        TTS_ABORT("Failed to allocate ggml graph '%s'.\n", graph_name ? graph_name : "<unnamed>");
    }

    if (!required_backend || !tts_backend_strict() || tts_backend_is_cpu(required_backend)) {
        return;
    }

    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (!node || node->op == GGML_OP_NONE || tts_is_view_op(node->op)) {
            continue;
        }

        ggml_backend_t actual_backend = ggml_backend_sched_get_tensor_backend(sched, node);
        if (actual_backend == required_backend) {
            continue;
        }

        TTS_ABORT("Strict backend graph '%s' placed node #%d op=%s name='%s' on %s instead of %s.\n",
                  graph_name ? graph_name : "<unnamed>",
                  i,
                  ggml_op_name(node->op),
                  ggml_get_name(node),
                  actual_backend ? ggml_backend_name(actual_backend) : "<unassigned>",
                  ggml_backend_name(required_backend));
    }
}

void append_to_response(tts_response & response, tts_response & to_append) {
    float * new_data = (float *) malloc((response.n_outputs + to_append.n_outputs) * sizeof(float));
    if (response.n_outputs > 0) {
        std::memcpy(new_data, response.data, response.n_outputs*sizeof(float));
    }
    if (to_append.n_outputs > 0) {
        float * next_loc = new_data + response.n_outputs;
        std::memcpy(next_loc, to_append.data, to_append.n_outputs*sizeof(float));
    }
    response.data = new_data;
    response.n_outputs += to_append.n_outputs;
    response.kokoro_duration_lengths.insert(
        response.kokoro_duration_lengths.end(),
        to_append.kokoro_duration_lengths.begin(),
        to_append.kokoro_duration_lengths.end());
    response.kokoro_token_ids.insert(
        response.kokoro_token_ids.end(),
        to_append.kokoro_token_ids.begin(),
        to_append.kokoro_token_ids.end());
    response.kokoro_duration_frames += to_append.kokoro_duration_frames;
    if (response.kokoro_duration_frame_samples == 0) {
        response.kokoro_duration_frame_samples = to_append.kokoro_duration_frame_samples;
    }
}

/*
 * Pulls output_size to prepped buffer 'output' from 'output_node' tensor. If no buffer is passed will default to the existing output buffer present
 * on runner_context.
 */
void runner_context::get_ggml_node_data_raw(struct ggml_tensor * output_node, void * output, size_t output_size, ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        buffer = buf_output;
    }
    if (ggml_backend_buffer_get_size(buffer) < output_size) {
        TTS_ABORT("Output buffer overflow of %zu / %zu for output node '%s'\n", output_size, ggml_backend_buffer_get_size(buffer), ggml_get_name(output_node));
    } else if (ggml_nbytes(output_node) < output_size) {
        TTS_ABORT("Output node, '%s', with %zu bytes is too small for #ggml_backend_tensor_get_async with size of %zu.\n", ggml_get_name(output_node), ggml_nbytes(output_node), output_size);
    }
    ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched, output_node);
    ggml_backend_tensor_get_async(backend_res, output_node, output, 0, output_size);
    ggml_backend_synchronize(backend_res);
}

void runner_context::get_ggml_node_data(struct ggml_tensor * output_node, float * output, size_t output_size, ggml_backend_buffer_t buffer) {
    get_ggml_node_data_raw(output_node, output, output_size, buffer);
}

void runner_context::set_threads() {
    if (backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new(&ttp);
        ggml_backend_cpu_set_threadpool(backend_cpu, threadpool);
    }
}

void runner_context::build_schedule(size_t max_nodes) {
    backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    if (backend != nullptr && !tts_backend_is_cpu(backend)) {
        backend_buffer = tts_backend_get_buffer_type(backend);
        std::vector<ggml_backend_buffer_type_t> bufs = {backend_buffer, backend_cpu_buffer};
        std::vector<ggml_backend_t> backs = {backend, backend_cpu};
        sched = tts_backend_sched_new(backs.data(), bufs.data(), 2, max_nodes, false, true);
    } else {
        std::vector<ggml_backend_buffer_type_t> bufs = {backend_cpu_buffer};
        std::vector<ggml_backend_t> backs = {backend_cpu};
        sched = tts_backend_sched_new(backs.data(), bufs.data(), 1, max_nodes, false, true);
    }
}

void runner_context::set_tensor_backend(ggml_tensor * tensor) {
    if (backend && !tts_backend_is_cpu(backend) && tensor) {
        ggml_backend_sched_set_tensor_backend(sched, tensor, backend);
    }
}

bool runner_context::prep_schedule(struct ggml_cgraph * gf) {
    if (backend && !tts_backend_is_cpu(backend)) {
        return true;
    }
    return ggml_backend_sched_reserve(sched, gf);
}

void runner_context::alloc_graph(ggml_cgraph * gf, const char * graph_name) {
    tts_backend_sched_alloc_graph_checked(sched, backend, gf, graph_name);
}

void runner_context::prep_output_buffer(size_t new_size) {
    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output) : 0;
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
            ggml_backend_buffer_free(buf_output);
            buf_output = nullptr;
            logits = nullptr;
        }
        buf_output = ggml_backend_buft_alloc_buffer(backend_cpu_buffer, new_size);
    }
    logits = (float *) ggml_backend_buffer_get_base(buf_output);
}

void tts_runner::init_build(std::vector<uint8_t>* buf_compute_meta) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta->size(),
        /*.mem_buffer =*/ buf_compute_meta->data(),
        /*.no_alloc   =*/ true,
    };

    ctx = ggml_init(params);
}

void tts_runner::free_build() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

tts_generation_runner::tts_generation_runner(const tts_model_loader & loader) : loader{ ref(loader) } {}

tts_generation_runner::~tts_generation_runner() {}

std::vector<std::string_view> tts_generation_runner::list_voices() {
    GGML_ABORT("The architecture '%s' does not support #list_voices.", loader.get().arch);
}

void tts_generation_runner::update_conditional_prompt(const char * file_path, const char * prompt) {
    GGML_ABORT("The architecture '%s' does not support update_conditional_prompt.", loader.get().arch);
}

test_tts_generation_runner::test_tts_generation_runner(const tts_model_loader & loader) :
    tts_generation_runner{ loader } {
    GGML_ASSERT(loader.is_test);
}

void test_tts_generation_runner::assign_weight(const char *, ggml_tensor &) {
    GGML_ABORT("Assumed loader.is_test");
}

void test_tts_generation_runner::prepare_post_load() {
    GGML_ABORT("Assumed loader.is_test");
}

void tts_model::prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size) {
    backend = tts_backend_init_model(cpu_only);
    buffer = tts_backend_get_buffer_type(backend);
    if (!backend || !buffer) {
        TTS_ABORT("Failed to initialize model backend. Set TTS_BACKEND=cpu to force CPU.\n");
    }
    const size_t alignment = ggml_backend_buft_get_alignment(buffer);
    const size_t max_alignment_padding = alignment > 0 ? (alignment - 1) * tensor_meta.n_tensors : 0;
    size_t ctx_size = ggml_tensor_overhead() * (tensor_meta.n_tensors * size_offset);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    buf = ggml_backend_buft_alloc_buffer(buffer, tensor_meta.n_bytes + max_alignment_padding + dedicated_add_on_size);
}

void tts_model::assign_weight(std::string name, ggml_tensor * tensor) {
	TTS_ABORT("%s received name, %s, tensor without being defined. %s must be defined for all implementations of tts_model. \n", __func__, name.c_str(), __func__);
}

void tts_model::set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
    const size_t alignment = ggml_backend_buffer_get_alignment(buf);
    offset = tts_pad_to_alignment(offset, alignment);
    tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size = ggml_nbytes(target);
    if (offset + size > ggml_backend_buffer_get_size(buf)) {
        TTS_ABORT("Model tensor buffer overflow while loading '%s': %zu + %zu > %zu\n",
                  target->name, offset, size, ggml_backend_buffer_get_size(buf));
    }
    ggml_backend_tensor_set(tensor, target->data, 0, size);
    ggml_set_name(tensor, target->name);
    offset += size;
}

void tts_model::set_tensor_from_data(struct ggml_tensor * tensor, const void * data, size_t size, const char * name) {
    const size_t alignment = ggml_backend_buffer_get_alignment(buf);
    offset = tts_pad_to_alignment(offset, alignment);
    tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    const size_t expected_size = ggml_nbytes(tensor);
    if (size != expected_size) {
        TTS_ABORT("Model tensor raw data size mismatch while loading '%s': %zu != %zu\n",
                  name ? name : "<unnamed>", size, expected_size);
    }
    if (offset + size > ggml_backend_buffer_get_size(buf)) {
        TTS_ABORT("Model tensor buffer overflow while loading '%s': %zu + %zu > %zu\n",
                  name ? name : "<unnamed>", offset, size, ggml_backend_buffer_get_size(buf));
    }
    ggml_backend_tensor_set(tensor, data, 0, size);
    if (name) {
        ggml_set_name(tensor, name);
    }
    offset += size;
}

void tts_model::set_tensor_from_backend_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
    const size_t alignment = ggml_backend_buffer_get_alignment(buf);
    offset = tts_pad_to_alignment(offset, alignment);
    tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size = ggml_nbytes(target);
    if (offset + size > ggml_backend_buffer_get_size(buf)) {
        TTS_ABORT("Model tensor buffer overflow while storing '%s': %zu + %zu > %zu\n",
                  target->name, offset, size, ggml_backend_buffer_get_size(buf));
    }
    std::vector<uint8_t> data(size);
    ggml_backend_tensor_get(target, data.data(), 0, size);
    ggml_backend_tensor_set(tensor, data.data(), 0, size);
    ggml_set_name(tensor, target->name);
    offset += size;
}

void tts_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only, std::string model_prefix, float size_offset, uint32_t dedicated_add_on_size) {
    tensor_meta = compute_tensor_meta(model_prefix, load_context, compute_tensor_meta_cb);
    prep_buffers_and_context(cpu_only, size_offset, dedicated_add_on_size);
}

size_t tts_model::max_nodes() {
    return std::max<size_t>(8192, tensor_meta.n_tensors*5);
}

void tts_model::free() {
    if (ctx) {
        ggml_free(ctx);
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
}
