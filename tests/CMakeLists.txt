if (LINUX)
    add_subdirectory(aPaleBlueDot)
endif ()

add_executable(model-buffer-alignment-test
        model_buffer_alignment.cpp
)
target_link_libraries(model-buffer-alignment-test tts)

add_executable(dia-cfg-scale-test
        dia_cfg_scale.cpp
)
target_link_libraries(dia-cfg-scale-test tts)

add_executable(vulkan-conv-transpose-1d-test
        vulkan_conv_transpose_1d.cpp
)
target_link_libraries(vulkan-conv-transpose-1d-test tts)

add_executable(vulkan-im2col-test
        vulkan_im2col.cpp
)
target_link_libraries(vulkan-im2col-test tts)

add_executable(vulkan-conv-1d-test
        vulkan_conv_1d.cpp
)
target_link_libraries(vulkan-conv-1d-test tts)

add_executable(vulkan-transpose-cont-2d-test
        vulkan_transpose_cont_2d.cpp
)
target_link_libraries(vulkan-transpose-cont-2d-test tts)

add_executable(kokoro-conv-1d-test
        kokoro_conv_1d.cpp
)
target_link_libraries(kokoro-conv-1d-test tts)

add_executable(kokoro-conv-1d-bench
        kokoro_conv_1d_bench.cpp
)
target_link_libraries(kokoro-conv-1d-bench tts)

add_executable(kokoro-snake-1d-test
        kokoro_snake_1d.cpp
)
target_link_libraries(kokoro-snake-1d-test tts)

add_executable(kokoro-adain-snake-1d-test
        kokoro_adain_snake_1d.cpp
)
target_link_libraries(kokoro-adain-snake-1d-test tts)

add_executable(kokoro-lstm-scan-test
        kokoro_lstm_scan.cpp
)
target_link_libraries(kokoro-lstm-scan-test tts)

add_executable(kokoro-lstm-step-test
        kokoro_lstm_step.cpp
)
target_link_libraries(kokoro-lstm-step-test tts)

add_executable(style-bert-vits2-decoder-fixture-test
        style_bert_vits2_decoder_fixture.cpp
)
target_link_libraries(style-bert-vits2-decoder-fixture-test tts)

add_executable(style-bert-vits2-decoder-backend-compare
        style_bert_vits2_decoder_backend_compare.cpp
)
target_link_libraries(style-bert-vits2-decoder-backend-compare tts)

add_executable(style-bert-vits2-jp-bert-loader-fixture-test
        style_bert_vits2_jp_bert_loader_fixture.cpp
)
target_link_libraries(style-bert-vits2-jp-bert-loader-fixture-test tts)

add_executable(style-bert-vits2-jp-bert-feature-fixture-test
        style_bert_vits2_jp_bert_feature_fixture.cpp
)
target_link_libraries(style-bert-vits2-jp-bert-feature-fixture-test tts)

add_executable(style-bert-vits2-text-encoder-input-fixture-test
        style_bert_vits2_text_encoder_input_fixture.cpp
)
target_link_libraries(style-bert-vits2-text-encoder-input-fixture-test tts)

add_executable(style-bert-vits2-speaker-embedding-fixture-test
        style_bert_vits2_speaker_embedding_fixture.cpp
)
target_link_libraries(style-bert-vits2-speaker-embedding-fixture-test tts)

add_executable(style-bert-vits2-style-vector-fixture-test
        style_bert_vits2_style_vector_fixture.cpp
)
target_link_libraries(style-bert-vits2-style-vector-fixture-test tts)

add_executable(style-bert-vits2-text-encoder-projection-fixture-test
        style_bert_vits2_text_encoder_projection_fixture.cpp
)
target_link_libraries(style-bert-vits2-text-encoder-projection-fixture-test tts)

add_executable(style-bert-vits2-text-encoder-ffn-fixture-test
        style_bert_vits2_text_encoder_ffn_fixture.cpp
)
target_link_libraries(style-bert-vits2-text-encoder-ffn-fixture-test tts)

add_executable(style-bert-vits2-text-encoder-attention-fixture-test
        style_bert_vits2_text_encoder_attention_fixture.cpp
)
target_link_libraries(style-bert-vits2-text-encoder-attention-fixture-test tts)

add_executable(style-bert-vits2-text-encoder-layer-fixture-test
        style_bert_vits2_text_encoder_layer_fixture.cpp
)
target_link_libraries(style-bert-vits2-text-encoder-layer-fixture-test tts)

add_executable(style-bert-vits2-text-encoder-stack-fixture-test
        style_bert_vits2_text_encoder_stack_fixture.cpp
)
target_link_libraries(style-bert-vits2-text-encoder-stack-fixture-test tts)

add_executable(style-bert-vits2-text-encoder-fixture-test
        style_bert_vits2_text_encoder_fixture.cpp
)
target_link_libraries(style-bert-vits2-text-encoder-fixture-test tts)

add_executable(style-bert-vits2-duration-predictor-fixture-test
        style_bert_vits2_duration_predictor_fixture.cpp
)
target_link_libraries(style-bert-vits2-duration-predictor-fixture-test tts)

add_executable(style-bert-vits2-sdp-condition-fixture-test
        style_bert_vits2_sdp_condition_fixture.cpp
)
target_link_libraries(style-bert-vits2-sdp-condition-fixture-test tts)

add_executable(style-bert-vits2-sdp-reverse-fixture-test
        style_bert_vits2_sdp_reverse_fixture.cpp
)
target_link_libraries(style-bert-vits2-sdp-reverse-fixture-test tts)

add_executable(style-bert-vits2-alignment-fixture-test
        style_bert_vits2_alignment_fixture.cpp
)
target_link_libraries(style-bert-vits2-alignment-fixture-test tts)

add_executable(style-bert-vits2-prior-sample-fixture-test
        style_bert_vits2_prior_sample_fixture.cpp
)
target_link_libraries(style-bert-vits2-prior-sample-fixture-test tts)

add_executable(style-bert-vits2-flow-fixture-test
        style_bert_vits2_flow_fixture.cpp
)
target_link_libraries(style-bert-vits2-flow-fixture-test tts)

add_executable(style-bert-vits2-decoder-latent-fixture-test
        style_bert_vits2_decoder_latent_fixture.cpp
)
target_link_libraries(style-bert-vits2-decoder-latent-fixture-test tts)
