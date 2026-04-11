#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — ZMQ Port Assignments
//
// Canonical port table for all modules.
// Must match conventions.md exactly.
//
// Port Layout:
//   5555-5556  Data-plane (ingest)
//   5561-5565  Inference (conversation models, legacy LLM/STT/TTS)
//   5566-5567  CV (always-on: face recognition, background blur)
//   5568-5569  Enhancement (on-demand: BasicVSR++, DTLN audio denoise)
//   5570-5571  Control plane (WebSocket bridge, daemon)
//   5572-5574  Four-mode architecture (conversation model, super res, image transform)
// ---------------------------------------------------------------------------


// Data-plane publishers (always-on)
inline constexpr int kVideoIngest = 5555;  ///< VideoIngestNode PUB
inline constexpr int kAudioIngest = 5556;  ///< AudioIngestNode PUB

// Legacy inference publishers (separate STT/LLM/TTS modules)
inline constexpr int kLlm         = 5561;  ///< LLMNode PUB (standalone Qwen TRT-LLM)
inline constexpr int kStt         = 5563;  ///< STTNode PUB (standalone Whisper)
inline constexpr int kTts         = 5565;  ///< TTSNode PUB (Kokoro — also used as companion for Gemma 4 E4B)

// CV publishers (always-on across all modes)
inline constexpr int kFaceRecog   = 5566;  ///< FaceRecognitionNode PUB
inline constexpr int kBgBlur      = 5567;  ///< BackgroundBlurNode PUB

// Enhancement publishers (on-demand spawn)
inline constexpr int kVideoDenoise = 5568;  ///< VideoDenoiseNode PUB (BasicVSR++ — legacy alias for super_resolution)
inline constexpr int kAudioDenoise = 5569;  ///< AudioDenoiseNode PUB (DTLN)

// Control plane publishers
inline constexpr int kWsBridge    = 5570;  ///< WebSocketBridge PUB
inline constexpr int kDaemon      = 5571;  ///< OmniEdgeDaemon PUB

// ── Four-mode architecture publishers ──────────────────────────────────────
// Unified conversation model (Qwen2.5-Omni or Gemma 4 E4B) — handles
// STT+LLM+TTS natively in a single process.  Publishes transcription,
// llm_response, and audio_output on a single port.
inline constexpr int kConversationModel = 5572;  ///< Unified conversation model PUB

// Super Resolution mode — BasicVSR++ temporal video enhancement.
// Shares the same binary as video_denoise but runs in dedicated SR mode
// with the full video ingest SHM feed.
inline constexpr int kSuperResolution   = 5573;  ///< SuperResolutionNode PUB (BasicVSR++)

// Image Transform mode — Stable Diffusion + ControlNet/IP-Adapter
// for structure-preserving style transfer (cat, anime, cartoon, etc.).
inline constexpr int kImageTransform    = 5574;  ///< ImageTransformNode PUB (Stable Diffusion)

// ── CV extension publishers ────────────────────────────────────────────────
// Face filter AR — FaceMesh V2 landmark detection + texture warping.
inline constexpr int kFaceFilter        = 5575;  ///< FaceFilterNode PUB

// SAM2 — Segment Anything Model 2 interactive segmentation.
inline constexpr int kSam2              = 5576;  ///< Sam2Node PUB

// ── HTTP / WebSocket server ───────────────────────────────────────────────
inline constexpr int kWsHttp            = 9001;  ///< uWebSockets HTTP + WS listen port

