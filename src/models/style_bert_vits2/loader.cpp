#include "../loaders.h"
#include "model.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

void style_bert_vits2_register() {}

style_bert_vits2_model_loader::style_bert_vits2_model_loader() : tts_model_loader{ "style-bert-vits2" } {}

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

void configure_style_bert_vulkan_precision() {
    const char * mode = std::getenv("STYLE_BERT_VITS2_VULKAN_PRECISION");
    if (mode && (std::strcmp(mode, "fast") == 0 || std::strcmp(mode, "0") == 0)) {
        return;
    }
    setenv_if_empty("GGML_VK_DISABLE_F16", "1");
    setenv_if_empty("GGML_VK_DISABLE_COOPMAT", "1");
    setenv_if_empty("GGML_VK_DISABLE_COOPMAT2", "1");
    if (env_truthy(std::getenv("STYLE_BERT_VITS2_DEBUG_LOAD"))) {
        std::cerr << "STYLE_BERT_VITS2_LOAD vulkan_precision=accurate"
                  << " GGML_VK_DISABLE_F16=" << std::getenv("GGML_VK_DISABLE_F16")
                  << " GGML_VK_DISABLE_COOPMAT=" << std::getenv("GGML_VK_DISABLE_COOPMAT")
                  << " GGML_VK_DISABLE_COOPMAT2=" << std::getenv("GGML_VK_DISABLE_COOPMAT2")
                  << std::endl;
    }
}
}

unique_ptr<tts_generation_runner> style_bert_vits2_model_loader::from_file(
    gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, bool cpu_only,
    const generation_configuration &) const {
    configure_style_bert_vulkan_precision();
    unique_ptr<style_bert_vits2_model> model = make_unique<style_bert_vits2_model>();
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    style_bert_vits2_context * sctx = build_new_style_bert_vits2_context(&*model, n_threads, cpu_only);
    return make_unique<style_bert_vits2_runner>(move(model), sctx);
}

const style_bert_vits2_model_loader style_bert_vits2_loader{};
