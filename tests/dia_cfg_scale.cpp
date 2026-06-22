#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/util.h"

namespace {
ggml_tensor * build_cfg_scale_expr(ggml_context * ctx, ggml_tensor * cond, ggml_tensor * uncond, float scale) {
    return ggml_add(ctx, cond, ggml_scale(ctx, ggml_add(ctx, cond, ggml_scale(ctx, uncond, -1.0f)), scale));
}
}

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int vocab = 5;
    constexpr int heads = 3;
    constexpr float scale = 3.0f;
    ggml_tensor * cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab, heads);
    ggml_tensor * uncond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab, heads);

    auto * cond_data = static_cast<float *>(cond->data);
    auto * uncond_data = static_cast<float *>(uncond->data);
    for (int i = 0; i < vocab * heads; ++i) {
        cond_data[i] = 0.25f * static_cast<float>(i + 1);
        uncond_data[i] = -0.125f * static_cast<float>(i + 2);
    }

    ggml_tensor * out = build_cfg_scale_expr(ctx, cond, uncond, scale);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->op == GGML_OP_MAP_CUSTOM1 ||
            gf->nodes[i]->op == GGML_OP_MAP_CUSTOM2 ||
            gf->nodes[i]->op == GGML_OP_MAP_CUSTOM3 ||
            gf->nodes[i]->op == GGML_OP_SUB) {
            std::fprintf(stderr, "unexpected CFG op in graph: %s\n", ggml_op_name(gf->nodes[i]->op));
            ggml_free(ctx);
            return 1;
        }
    }

    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "CFG graph compute failed\n");
        ggml_free(ctx);
        return 1;
    }

    const auto * actual = static_cast<float *>(out->data);
    for (int i = 0; i < vocab * heads; ++i) {
        const float expected = cond_data[i] + scale * (cond_data[i] - uncond_data[i]);
        if (std::fabs(actual[i] - expected) > 1e-6f) {
            std::fprintf(stderr, "CFG mismatch at %d: actual=%f expected=%f\n", i, actual[i], expected);
            ggml_free(ctx);
            return 1;
        }
    }

    ggml_free(ctx);
    return 0;
}
