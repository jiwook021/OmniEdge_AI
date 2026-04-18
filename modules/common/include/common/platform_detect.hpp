#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Platform Detection
//
// Macros are injected by the CMake build system:
//   OE_PLATFORM_WSL2            — set when /proc/version contains "Microsoft"
//   OE_HAS_BLACKWELL_FEATURES   — set when CMAKE_CUDA_ARCHITECTURES includes "100a"
// ---------------------------------------------------------------------------

#if defined(OE_PLATFORM_WSL2)
#	define OE_HOST_REGISTER_SUPPORTED 0
#else
#	define OE_HOST_REGISTER_SUPPORTED 1
#endif
