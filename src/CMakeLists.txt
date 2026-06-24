# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif ()
endif ()

# TTS

add_library(tts
        ../include/args.h
        tokenizer.cpp
        sampler.cpp
        sampler.h
        util.cpp
        util.h
        args.cpp
        style_bert_vits2_c_api.cpp
        tts_model.cpp
        tts_model.h
)

target_compile_definitions(tts PRIVATE TTS_BUILD)

get_target_property(TTS_GGML_ACTIVE_SOURCE_DIR ggml SOURCE_DIR)
target_include_directories(tts PUBLIC . ../include "${TTS_GGML_ACTIVE_SOURCE_DIR}")

target_link_libraries(tts PUBLIC ggml)

set(TTS_GGML_HEADER_PATH "")
if (TTS_GGML_ACTIVE_SOURCE_DIR AND EXISTS "${TTS_GGML_ACTIVE_SOURCE_DIR}/include/ggml.h")
    set(TTS_GGML_HEADER_PATH "${TTS_GGML_ACTIVE_SOURCE_DIR}/include/ggml.h")
elseif (TTS_GGML_ACTIVE_SOURCE_DIR AND EXISTS "${TTS_GGML_ACTIVE_SOURCE_DIR}/../include/ggml.h")
    get_filename_component(TTS_GGML_ACTIVE_PARENT_SOURCE_DIR "${TTS_GGML_ACTIVE_SOURCE_DIR}/.." ABSOLUTE)
    set(TTS_GGML_HEADER_PATH "${TTS_GGML_ACTIVE_PARENT_SOURCE_DIR}/include/ggml.h")
endif()

set(TTS_GGML_HAS_GELU_ERF OFF)
if (TTS_GGML_HEADER_PATH)
    file(READ "${TTS_GGML_HEADER_PATH}" TTS_GGML_HEADER_FOR_FEATURES)
    if (TTS_GGML_HEADER_FOR_FEATURES MATCHES "ggml_gelu_erf")
        set(TTS_GGML_HAS_GELU_ERF ON)
    endif()
endif()
target_compile_definitions(tts PRIVATE TTS_GGML_HAS_GELU_ERF=$<BOOL:${TTS_GGML_HAS_GELU_ERF}>)

if (BUILD_SHARED_LIBS)
    set_target_properties(tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(tts PRIVATE LLAMA_BUILD)
    target_compile_definitions(tts PUBLIC LLAMA_SHARED)
endif ()

add_subdirectory(decoder)
add_subdirectory(models)

if (ESPEAK_INCLUDE_DIRS)
    set_source_files_properties(models/kokoro/phonemizer.cpp PROPERTIES COMPILE_FLAGS "${ESPEAK_CFLAGS_OTHER}")
    set_source_files_properties(models/kokoro/phonemizer.cpp PROPERTIES INCLUDE_DIRECTORIES "${ESPEAK_INCLUDE_DIRS}")
    target_link_libraries(tts PUBLIC ${ESPEAK_LIBRARIES})
endif ()
