#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — ZMQ Port Assignments
//
// Canonical port table for all modules.
// Must match conventions.md exactly.
//
// Port Layout:
//   5555-5556  Data-plane (ingest)
//   5563-5565  Inference (STT, TTS)
//   5566-5567  CV (always-on: face recognition, background blur)
//   5568-5569  Enhancement (on-demand: BasicVSR++, DTLN audio denoise)
//   5570-5571  Control plane (WebSocket bridge, daemon)
//   5572       Unified conversation model (Gemma-4)
//   5575-5579  CV extensions (face filter, sam2, screen ingest, security, beauty)
// ---------------------------------------------------------------------------


// Data-plane publishers (always-on)
inline constexpr int kVideoIngest = 5555;  ///< VideoIngestNode PUB
inline constexpr int kAudioIngest = 5556;  ///< AudioIngestNode PUB

// Inference publishers (separate STT/TTS modules)
inline constexpr int kStt         = 5563;  ///< STTNode PUB (standalone Whisper)
inline constexpr int kTts         = 5565;  ///< TTSNode PUB (Kokoro — also used as companion for Gemma 4 E4B)

// CV publishers (always-on across all modes)
inline constexpr int kFaceRecog   = 5566;  ///< FaceRecognitionNode PUB
inline constexpr int kBgBlur      = 5567;  ///< BackgroundBlurNode PUB

// Enhancement publishers (on-demand spawn)
inline constexpr int kVideoDenoise = 5568;  ///< VideoDenoiseNode PUB (BasicVSR++)
inline constexpr int kAudioDenoise = 5569;  ///< AudioDenoiseNode PUB (DTLN)

// Control plane publishers
inline constexpr int kWsBridge    = 5570;  ///< WebSocketBridge PUB
inline constexpr int kDaemon      = 5571;  ///< OmniEdgeDaemon PUB

// ── Unified conversation model publisher ──────────────────────────────────
// Gemma-4 E2B / E4B — handles STT + LLM in a single process, with text-only
// output streamed to KokoroTTSNode for audio. Publishes transcription and
// llm_response on this port; audio_output comes from the TTS sidecar.
inline constexpr int kConversationModel = 5572;  ///< Unified conversation model PUB

// ── CV extension publishers ────────────────────────────────────────────────
// Face filter AR — FaceMesh V2 landmark detection + texture warping.
inline constexpr int kFaceFilter        = 5575;  ///< FaceFilterNode PUB

// SAM2 — Segment Anything Model 2 interactive segmentation.
inline constexpr int kSam2              = 5576;  ///< Sam2Node PUB

// ── Screen capture (always-on, CPU-only) ──────────────────────────────────
// Screen ingest — receives DXGI desktop frames from Windows over TCP,
// decodes JPEG, writes BGR24 to /oe.screen.ingest SHM.
inline constexpr int kScreenIngest      = 5577;  ///< ScreenIngestNode PUB

// ── Security Camera mode ──────────────────────────────────────────────────
// YOLO detection + NVENC MP4 recording + event logging.
// Publishes: security_detection, security_event, security_recording_status.
inline constexpr int kSecurityCamera    = 5578;  ///< SecurityCameraNode PUB

// ── Beauty mode (real-time face beautification) ──────────────────────────
// FaceMesh V2 landmark detection + skin smoothing + TPS warp + ISP.
// Publishes: beauty_frame.
inline constexpr int kBeauty            = 5579;  ///< BeautyNode PUB

// ── HTTP / WebSocket server ───────────────────────────────────────────────
inline constexpr int kWsHttp            = 9001;  ///< uWebSockets HTTP + WS listen port

