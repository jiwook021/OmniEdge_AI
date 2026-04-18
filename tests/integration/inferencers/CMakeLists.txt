# =========================================================================
# tests/integration/inferencers — Real GPU inferencer tests
#
# All GPU-gated, built only when dependencies are available.
# GTest and CUDAToolkit are resolved at root — no find_package() needed.
#
# Accepted deviation: raw add_executable used throughout because these tests
# compile cross-module sources from modules/core/*/src/ and need manual
# target configuration not supported by oe_add_test().
# =========================================================================

# =========================================================================
# TRT Whisper inferencer (requires TensorRT, resolved by oe_find_tensorrt())
# =========================================================================
if(OE_TENSORRT_FOUND)
    add_executable(test_trt_whisper_inferencer
        test_trt_whisper_inferencer.cpp
        ${CMAKE_SOURCE_DIR}/modules/core/stt/src/trt_whisper_inferencer.cpp
    )
    target_compile_features(test_trt_whisper_inferencer PRIVATE cxx_std_20)
    oe_set_target_warnings_no_error(test_trt_whisper_inferencer)
    target_include_directories(test_trt_whisper_inferencer PRIVATE
        ${CMAKE_SOURCE_DIR} ${OE_TENSORRT_INCLUDE_DIR})
    target_link_libraries(test_trt_whisper_inferencer PRIVATE
        GTest::gtest_main omniedge_stt_common CUDA::cudart)
    if(TARGET nvinfer)
        target_link_libraries(test_trt_whisper_inferencer PRIVATE nvinfer)
    endif()
    add_test(NAME TrtWhisperInferencerTests COMMAND test_trt_whisper_inferencer)

    add_executable(test_real_speech_transcription
        test_real_speech_transcription.cpp
        ${CMAKE_SOURCE_DIR}/modules/core/stt/src/trt_whisper_inferencer.cpp
    )
    target_compile_features(test_real_speech_transcription PRIVATE cxx_std_20)
    oe_set_target_warnings_no_error(test_real_speech_transcription)
    target_include_directories(test_real_speech_transcription PRIVATE
        ${CMAKE_SOURCE_DIR} ${OE_TENSORRT_INCLUDE_DIR})
    target_link_libraries(test_real_speech_transcription PRIVATE
        GTest::gtest_main omniedge_stt_common CUDA::cudart)
    if(TARGET nvinfer)
        target_link_libraries(test_real_speech_transcription PRIVATE nvinfer)
    endif()
    add_test(NAME RealSpeechTranscriptionTests
        COMMAND test_real_speech_transcription WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(RealSpeechTranscriptionTests PROPERTIES LABELS "gpu")
else()
    message(STATUS "[tests/integration/inferencers] TensorRT not found — skipping TRT tests")
endif()

# =========================================================================
# Legacy TRT-LLM inferencer test (disabled)
# =========================================================================
# TODO: Create conversation inferencer GPU tests for Gemma-4 E2B and E4B.
# The old TensorRTLLMInferencer path has been replaced by
# the generic ConversationInferencer interface with GemmaInferencer as sole impl.

# =========================================================================
# ONNX Kokoro TTS inferencer (requires ONNX Runtime + espeak-ng)
# =========================================================================
add_executable(test_onnx_kokoro_inferencer EXCLUDE_FROM_ALL
    test_onnx_kokoro_inferencer.cpp
    ${CMAKE_SOURCE_DIR}/modules/core/tts/src/onnx_kokoro_inferencer.cpp
)
target_compile_features(test_onnx_kokoro_inferencer PRIVATE cxx_std_20)
oe_set_target_warnings_no_error(test_onnx_kokoro_inferencer)
target_include_directories(test_onnx_kokoro_inferencer PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_onnx_kokoro_inferencer PRIVATE
    GTest::gtest_main omniedge_core_tts CUDA::cudart)

if(OE_ONNXRUNTIME_FOUND)
    target_link_libraries(test_onnx_kokoro_inferencer PRIVATE ${OE_ONNXRUNTIME_LIB})
    target_include_directories(test_onnx_kokoro_inferencer PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
endif()
if(OE_ESPEAK_FOUND)
    target_link_libraries(test_onnx_kokoro_inferencer PRIVATE ${OE_ESPEAK_LIB})
    target_include_directories(test_onnx_kokoro_inferencer PRIVATE ${OE_ESPEAK_INCLUDE_DIR})
endif()
