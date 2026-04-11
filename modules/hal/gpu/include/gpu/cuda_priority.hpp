#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// OmniEdge_AI — CUDA Stream Priority Constants
//
// Blackwell priority range: [0, -5] (lower number = higher priority).
// All module CUDA streams use cudaStreamNonBlocking with these priorities.
// ---------------------------------------------------------------------------


inline constexpr int kCudaPriorityLlm    = -5;  ///< LLMNode (highest — blocks TTS pipeline)
inline constexpr int kCudaPriorityStt    = -5;  ///< STTNode (same tier as LLM)
inline constexpr int kCudaPriorityTts    = -3;  ///< TTSNode
inline constexpr int kCudaPriorityCv     =  0;  ///< BackgroundBlurNode + FaceRecognitionNode

// Enhancement modules (on-demand)
inline constexpr int kCudaPriorityVideoDenoise = -1;  ///< BasicVSR++ (optional, low priority)
inline constexpr int kCudaPriorityAudioDenoise =  0;  ///< DTLN (lightweight, same tier as CV)

