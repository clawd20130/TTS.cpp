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
