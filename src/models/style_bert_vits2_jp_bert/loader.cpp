#include "../loaders.h"
#include "model.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

void style_bert_vits2_jp_bert_register() {}

style_bert_vits2_jp_bert_model_loader::style_bert_vits2_jp_bert_model_loader():
    tts_model_loader{style_bert_vits2_jp_bert_model::arch} {}

namespace {
bool env_truthy(const char * value) {
    return value && value[0] && std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0;
}

void setenv_if_empty(const char * name, const char * value) {
    const char * current = std::getenv(name);
    if (!current || !current[0]) {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
}

void configure_jp_bert_vulkan_precision() {
    const char * mode = std::getenv("STYLE_BERT_VITS2_JP_BERT_VULKAN_PRECISION");
    if (mode && (std::strcmp(mode, "fast") == 0 || std::strcmp(mode, "0") == 0)) {
        return;
    }
    setenv_if_empty("GGML_VK_DISABLE_F16", "1");
    setenv_if_empty("GGML_VK_DISABLE_COOPMAT", "1");
    setenv_if_empty("GGML_VK_DISABLE_COOPMAT2", "1");
    if (env_truthy(std::getenv("STYLE_BERT_VITS2_JP_BERT_DEBUG_LOAD"))) {
        std::cerr << "STYLE_BERT_VITS2_JP_BERT_LOAD vulkan_precision=accurate"
                  << " GGML_VK_DISABLE_F16=" << std::getenv("GGML_VK_DISABLE_F16")
                  << " GGML_VK_DISABLE_COOPMAT=" << std::getenv("GGML_VK_DISABLE_COOPMAT")
                  << " GGML_VK_DISABLE_COOPMAT2=" << std::getenv("GGML_VK_DISABLE_COOPMAT2")
                  << std::endl;
    }
}
}

unique_ptr<tts_generation_runner> style_bert_vits2_jp_bert_model_loader::from_file(
    gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, bool cpu_only,
    const generation_configuration &) const {
    configure_jp_bert_vulkan_precision();
    unique_ptr<style_bert_vits2_jp_bert_model> model = make_unique<style_bert_vits2_jp_bert_model>();
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    style_bert_vits2_jp_bert_context * context = new style_bert_vits2_jp_bert_context(&*model, n_threads);
    context->backend = model->backend;
    context->owns_backend = false;
    context->backend_cpu = ggml_backend_cpu_init();
    context->set_threads();
    context->build_schedule();
    const size_t max_nodes = std::max<size_t>(model->max_nodes() * 64, 32768);
    context->buf_compute_meta.resize(ggml_tensor_overhead() * max_nodes +
                                     ggml_graph_overhead_custom(max_nodes, false));
    return make_unique<style_bert_vits2_jp_bert_runner>(move(model), context);
}

const style_bert_vits2_jp_bert_model_loader style_bert_vits2_jp_bert_loader{};
