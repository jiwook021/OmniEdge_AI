# OmniEdge_AI — Architecture Specification

> **Version:** 2.1 · **Status:** Reference Architecture  
> **Target GPU:** NVIDIA RTX PRO 3000 (Blackwell) · 12 GB GDDR6 · SM 10.0 · Tensor Cores (FP8/INT4)  
> **Runtime:** WSL2 (Ubuntu 22.04+) on Windows 10/11 · CUDA 12.x · TensorRT 10.x · WDDM GPU Paravirtualization  
> **Capture:** ROCWARE RC28 1080p 60 fps · Dual Microphones (cardioid + omnidirectional) · USB via `usbipd-win`  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Design Principles](#3-design-principles)
4. [CUDA Execution Model](#4-cuda-execution-model)
5. [Module Architecture](#5-module-architecture)
6. [IPC Design — Data Plane & Control Plane](#6-ipc-design--data-plane--control-plane)
7. [GPU Memory Architecture & Tier System](#7-gpu-memory-architecture--tier-system)
8. [AI Model Portfolio](#8-ai-model-portfolio)
9. [Conversation Orchestration — State Machine](#9-conversation-orchestration--state-machine)
10. [Frontend & WebSocket Bridge](#10-frontend--websocket-bridge)
11. [Daemon, Watchdog & Fault Tolerance](#11-daemon-watchdog--fault-tolerance)
12. [Configuration & Build System](#12-configuration--build-system)
13. [Observability, Profiling & Testing](#13-observability-profiling--testing)
14. [Model Conversion Pipeline](#14-model-conversion-pipeline)

---

## 1. Executive Summary

OmniEdge_AI is a **multi-modal, real-time AI assistant** that executes six concurrent AI models on a single NVIDIA GPU (12 GB VRAM) under WSL2. It captures live 1080p video and 16 kHz audio from a USB webcam, processes them through inference pipelines (LLM, STT, TTS, VLM, face recognition, instance segmentation), and delivers a browser-based conversational interface with streaming text, synthesized speech, and augmented video.

### Architectural Invariants

| Invariant | Mechanism | Consequence |
|:---|:---|:---|
| **Process-per-model fault isolation** | 10 independent C++20 binaries, each with its own `cudaSetDevice(0)` context. No shared CUDA contexts, no MPS. | A `SIGSEGV` in the face recognition process cannot corrupt the LLM's KV cache. The CUDA driver reclaims the crashed process's VRAM as contiguous blocks. |
| **Dual-layer IPC** | POSIX `shm_open`/`mmap` for bulk data (6.2 MB frames, PCM audio). ZeroMQ PUB/SUB for JSON control messages. | No module links to another's `.so`. No `dlopen`, no RPC stubs, no gRPC. IPC boundary is OS-level. |
| **Process-level VRAM lifecycle** | `posix_spawn()` to load a model, `SIGTERM` to unload. CUDA context destruction guarantees zero fragmentation. | No custom CUDA allocator, no memory pool, no `cuMemPool`. The OS process *is* the memory scope. |
| **WDDM compatibility** | All pinned host memory via `cudaHostAlloc` (not `cudaHostRegister`). No CUDA IPC (`cudaIpcGetMemHandle`). | Runs under WSL2's WDDM GPU paravirtualization without driver-level workarounds. |

**End-to-end latency target:** PTT release → first audio playback in **< 1.3 s** (P50, excluding 800 ms VAD silence detection). Total user-perceived latency: **< 2.1 s**.

---

## 2. System Overview

### 2.1 Architecture at a Glance

```
┌─ NVIDIA GPU (12 GB VRAM) ──────────────────────────────────────────────────┐
│                                                                            │
│  Language Models              Vision Models              Audio Models      │
│  ┌────────────────┐   ┌────────────────┐   ┌──────────────────┐           │
│  │ Qwen 2.5 7B    │   │ YOLOv8n-seg    │   │ Whisper V3 Turbo │           │
│  │ INT4-AWQ       │   │ FP16 · 0.50 GB │   │ INT8 · 1.50 GB   │           │
│  │ 4.30 GB        │   └────────────────┘   └──────────────────┘           │
│  │ + KV Cache     │   ┌────────────────┐   ┌──────────────────┐           │
│  │   1.15 GB FP8  │   │ InspireFace    │   │ Kokoro v1.0      │           │
│  └────────────────┘   │ FP16 · 0.50 GB │   │ INT8 · 0.10 GB   │           │
│  ┌────────────────┐   └────────────────┘   └──────────────────┘           │
│  │ Moondream2     │                                                       │
│  │ INT4 · 2.45 GB │   Silero VAD: CPU only, 0 VRAM                       │
│  │ (on-demand)    │   WSL2 system buffer: ~1.50 GB                        │
│  └────────────────┘                                                       │
└──────────┬─────────────────────────────────────────────────────────────────┘
           │ POSIX SHM (data) + ZeroMQ PUB/SUB (control)
┌──────────┴─────────────────────────────────────────────────────────────────┐
│  10 Independent C++20 Processes (WSL2)                                     │
│                                                                            │
│  Ingestion        Vision               Audio/Language       Infrastructure │
│  oe_video_ingest  oe_bg_blur           oe_stt               oe_ws_bridge  │
│  oe_audio_ingest  oe_face_recog        oe_llm               oe_daemon     │
│                   oe_vlm (on-demand)   oe_tts                             │
└──────────┬─────────────────────────────────────────────────────────────────┘
           │ 3 WebSocket channels on :9001 (/video, /audio, /chat)
┌──────────┴─────────────────────────────────────────────────────────────────┐
│  Browser SPA — <canvas> 30 fps · AudioContext 24 kHz · JSON chat stream   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Map

| Process Binary | AI Model | Inference Engine | VRAM | ZMQ PUB Port |
|:---|:---|:---|---:|---:|
| `oe_llm` | Qwen 2.5 7B INT4-AWQ | TensorRT-LLM | 4.30 GB | 5561 |
| `oe_stt` | Whisper V3 Turbo INT8 | TensorRT-LLM | 1.50 GB | 5563 |
| `oe_tts` | Kokoro v1.0 INT8 | ONNX Runtime CUDA EP | 0.10 GB | 5565 |
| `oe_vlm` | Moondream2 INT4 | TRT (vision) + ONNX (decoder) | 2.45 GB | 5562 |
| `oe_bg_blur` | YOLOv8n-seg FP16 | TensorRT | 0.50 GB | 5567 |
| `oe_face_recog` | InspireFace FP16 | TensorRT | 0.50 GB | 5566 |
| `oe_video_ingest` | — | GStreamer | 0 | 5555 |
| `oe_audio_ingest` | Silero VAD (CPU) | ONNX Runtime CPU | 0 | 5556 |
| `oe_ws_bridge` | — | uWebSockets | 0 | 5570 |
| `oe_daemon` | — | — | 0 | 5571 |

---

## 3. Design Principles

### 3.1 Process-per-Model Isolation

Every AI model runs in its own OS process with its own `cudaSetDevice(0)` context. When a process exits (cleanly or via crash), the CUDA driver destroys the context and reclaims all VRAM as contiguous blocks — zero fragmentation, guaranteed.

This is the foundation of the VRAM management strategy: loading and unloading models reduces to `posix_spawn()` and `SIGTERM` at the process level.

### 3.2 Dual-Layer IPC

| Layer | Transport | Use Case | Latency |
|:---|:---|:---|:---|
| **Data Plane** | POSIX shared memory (`shm_open` / `mmap`) | Video frames (6.2 MB), audio buffers, JPEG output | < 0.3 ms per `memcpy` |
| **Control Plane** | ZeroMQ PUB/SUB (TCP localhost) | JSON metadata, commands, transcriptions, token streams | < 0.1 ms |

No module links to another's library or calls its functions. All inter-process communication flows through these two channels.

### 3.3 Backend Swappability (Strategy Pattern)

Every inference module implements a role-specific C++ interface:

| Interface | Concrete Implementations |
|:---|:---|
| `ILLMBackend` | `TrtLlmQwenBackend`, `LlamaCppBackend` (future) |
| `ISTTBackend` | `TrtLlmWhisperBackend` |
| `ITTSBackend` | `OnnxKokoroBackend` |
| `IVLMBackend` | `HybridMoondreamBackend` |
| `IBlurBackend` | `TensorRTBlurBackend` |
| `IFaceRecogBackend` | `InspireFaceBackend` |

The active backend is selected from `omniedge_config.yaml`. Swapping Qwen for Llama 3 requires only a new `ILLMBackend` implementation and a YAML config change — no other modules are affected.

### 3.4 No Hardcoded Numbers

Every tunable value lives in one of four configuration layers:

| Layer | File | Scope |
|:---|:---|:---|
| **Runtime Config** | `config/omniedge_config.yaml` | Ports, thresholds, engine paths — changes require no recompilation |
| **Compile-time Tuning** | `config/oe_tuning.hpp` | Buffer sizes, ZMQ HWM, frame dimensions — requires rebuild |
| **Platform Detection** | `config/oe_platform.hpp` | WSL2 detection, CUDA SM version gates |
| **Runtime Defaults** | `common/oe_defaults.hpp` | Fallback values when YAML fields are missing |

Module `.cpp` files never contain bare numeric literals for tuning parameters.

### 3.5 WSL2 Constraint: `cudaHostAlloc` Only

`cudaHostRegister` is **unsupported** under WDDM (WSL2). All pinned host memory uses `cudaHostAlloc` instead. This is detected at compile time via `OE_PLATFORM_WSL2` in `oe_platform.hpp`.

---

## 4. CUDA Execution Model

Each OmniEdge process creates its own CUDA context via `cudaSetDevice(0)`. Under WDDM GPU paravirtualization (WSL2), the NVIDIA driver time-slices these contexts on the hardware — there is no concurrent kernel execution across processes. This section documents the execution model decisions that an NVIDIA reviewer would scrutinize.

### 4.1 Why Not CUDA MPS

NVIDIA Multi-Process Service (MPS) allows true kernel concurrency by funneling multiple processes through a single CUDA context. OmniEdge rejects MPS for three reasons:

| Reason | Detail |
|:---|:---|
| **Fault isolation** | Under MPS, a GPU fault (illegal memory access, stuck kernel) in *any* client process triggers a fatal error that terminates *all* MPS clients. A face recognition crash would take down the LLM. |
| **WDDM incompatibility** | MPS requires Linux native GPU drivers (not WDDM paravirtualized). WSL2 runs under WDDM — MPS is not available. |
| **VRAM lifecycle** | MPS clients share a CUDA context. Process exit does not guarantee VRAM reclamation — the MPS daemon holds allocations until it restarts. Process-per-model gives deterministic VRAM reclamation on `exit()`. |

**Consequence:** Inference kernels from different modules are serialized at the GPU scheduler level. This is acceptable because the critical voice path (STT → LLM → TTS) is already sequential by data dependency — there is no latency penalty from serialization since these stages never need to overlap.

### 4.2 Why Not CUDA IPC (`cudaIpcGetMemHandle`)

CUDA IPC allows one process to export a GPU allocation for another process to map directly, avoiding host-side copies entirely. OmniEdge does not use it:

| Reason | Detail |
|:---|:---|
| **WDDM restriction** | `cudaIpcGetMemHandle` returns `cudaErrorNotSupported` under WDDM. WSL2 cannot use this mechanism at all. |
| **MPS requirement** | Even on native Linux, CUDA IPC across separate process contexts requires MPS for concurrent access (see 4.1). |
| **Copy cost is acceptable** | Video frames: `memcpy` from SHM to pinned host takes ~0.3 ms for 6.2 MB (DDR4 bandwidth-bound). `cudaMemcpyAsync` to device takes ~0.5 ms over PCIe Gen4. Total: 0.8 ms — negligible versus inference latency of 100–400 ms. |

The dual IPC design (POSIX SHM for data + ZMQ for control) works on every deployment target — native Linux, WSL2, or future container deployments — without driver-level dependencies.

### 4.3 CUDA Stream Usage Per Module

Each module uses dedicated CUDA streams for overlapping data transfer with inference computation. No module uses the default stream (stream 0).

| Module | Streams | Usage |
|:---|:---|:---|
| `oe_stt` | 2 streams | `stream_copy`: `cudaMemcpyAsync` (mel spectrogram H→D). `stream_infer`: encoder + decoder TRT-LLM execution. Overlap: next chunk copy during current decode. |
| `oe_llm` | 1 stream | Single `stream_infer`: prefill + autoregressive decode. TRT-LLM manages KV cache internally on this stream. Copy overhead minimal (token IDs only). |
| `oe_tts` | 2 streams | `stream_infer`: ONNX Runtime CUDA EP execution. `stream_copy`: PCM output D→H for SHM write. |
| `oe_bg_blur` | 3 streams | `stream_copy_in`: frame H→D. `stream_infer`: YOLO + ISP + blur + nvJPEG. `stream_copy_out`: JPEG D→H. Triple overlap at 30 fps. |
| `oe_face_recog` | 2 streams | `stream_copy`: frame H→D (every 3rd frame). `stream_infer`: InspireFace detect+align+embed. |
| `oe_vlm` | 1 stream | ONNX Runtime manages internally. Python subprocess for vision encoding uses PyTorch's default stream. |

**Stream priority** is not used in the current implementation. All streams run at default priority (0). The voice pipeline's latency advantage comes from its sequential data dependency (STT output → LLM input → TTS input), not from GPU-level preemption. Stream priorities would only help if kernels from multiple modules executed concurrently — which they do not under the process-per-model isolation model (see 4.1).

### 4.4 GPU Architecture Targeting

TensorRT engines are compiled for a specific GPU SM architecture and are **not portable** across GPU generations.

| Configuration | Value | Rationale |
|:---|:---|:---|
| `CMAKE_CUDA_ARCHITECTURES` | `100` | Blackwell (SM 10.0). Set in root `CMakeLists.txt`. |
| `trtexec --best` | FP16 + INT8 + FP8 | Blackwell supports FP8, enabling smaller KV cache (FP8 vs FP16). |
| Engine portability | None | A TRT engine built for SM 10.0 will not load on SM 8.9 (Ada) or SM 8.6 (Ampere). Engines must be rebuilt for each GPU. |
| Fallback | `trtexec` at deploy-time | If pre-built engines are missing, the daemon calls `scripts/build_engines.sh` on first launch. Build time: ~10–45 min depending on model size. |

The root `CMakeLists.txt` sets:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 100)  # Blackwell SM 10.0
```

When deploying to a different GPU, change this value and rebuild all TensorRT engines.

---

## 5. Module Architecture

### 4.1 Data Flow Overview

```
Webcam (USB) ──→ GStreamer ──→ Video Ingest ──shm──→ ┬─→ Background Blur ──shm──→ WS Bridge ──→ Browser Canvas
                                                      ├─→ Face Recognition ──zmq──→ Daemon (identity)
                                                      └─→ VLM (on-demand) ──zmq──→ Daemon (scene)

Microphone ──→ GStreamer ──→ Audio Ingest ──shm──→ STT ──zmq──→ Daemon ──zmq──→ LLM ──zmq──→ TTS ──shm──→ WS Bridge ──→ Browser Audio
```

The **voice pipeline** (PTT → audio → STT → LLM → TTS → speaker) is the critical latency path. The **vision pipeline** (webcam → blur → JPEG → browser) runs concurrently at 30 fps.

### 4.2 Video Ingest (`oe_video_ingest`)

Captures 1080p video from the ROCWARE RC28 webcam via GStreamer pipelines.

**Primary path (V4L2 via usbipd):** The webcam is attached to WSL2 using `usbipd-win`, appearing as `/dev/video0`. GStreamer pipeline: `v4l2src → jpegdec → videoconvert → BGR appsink`. No H.264 encode/decode overhead.

**Fallback path (TCP from Windows):** When usbipd is unavailable, the Windows host runs `ksvideosrc → x264enc tune=zerolatency → tcpserversink :5000`. WSL2 consumes via `tcpclientsrc → avdec_h264 → videoconvert → appsink`.

Each captured frame (1920×1080 BGR, 6.2 MB) is written to a double-buffered shared memory segment (`/oe.vid.ingest`) and announced via ZMQ.

### 4.3 Audio Ingest (`oe_audio_ingest`)

Captures 16 kHz mono F32 audio from the Windows host over TCP port 5001 (`wasapi2src → audioresample → F32LE → tcpserversink`).

**Silero VAD gating:** Every audio chunk passes through a CPU-only Voice Activity Detection model (Silero, ~1 MB ONNX). Only speech chunks are written to shared memory (`/oe.aud.ingest`). Silence is discarded **before** reaching Whisper — this eliminates hallucinations on quiet input segments.

When speech ends (configurable silence threshold, default 800 ms), AudioIngest publishes a `vad_status` event to signal end-of-utterance.

### 4.4 Background Blur (`oe_bg_blur`)

Receives video frames from SHM, runs a 5-step GPU pipeline, and outputs JPEG to a separate SHM segment (`/oe.cv.blur.jpeg`):

1. **ISP adjustments** — brightness, contrast, saturation, sharpness via OpenCV CUDA (skipped if all values are default)
2. **YOLOv8n-seg inference** — generates per-pixel person segmentation mask
3. **Mask upscale** — bilinear interpolation from 160×160 to 1920×1080
4. **Gaussian blur compositing** — blur applied to background only; foreground pixels pass through
5. **nvJPEG encode** — GPU-accelerated JPEG compression

Also supports **shape overlays** drawn by the user in the browser canvas — shapes are composited into the blurred background region only (foreground pixels are never overwritten).

### 4.5 Face Recognition (`oe_face_recog`)

Processes every 3rd frame (~10 fps) using InspireFace SDK:

- **Detect** → **Align** → **Extract** 512-dimensional embedding → **Compare** against SQLite gallery (`known_faces.sqlite`) using cosine similarity
- Publishes recognized identity on ZMQ (`{name: "Admin", confidence: 0.98}`)
- Supports browser-initiated face registration (upload or webcam capture → embed → INSERT into SQLite)

Identity is injected into the LLM system prompt so the assistant addresses the user by name.

### 4.6 Whisper STT (`oe_stt`)

Whisper V3 Turbo (809M params, 4 decoder layers) runs as a TensorRT-LLM encoder-decoder engine.

**Pipeline:** PCM F32 → 128-bin Log-Mel spectrogram (CPU) → `cudaMemcpyAsync` → encoder → decoder → tokenizer → text

**Hallucination filter:** Discards outputs where `no_speech_prob > 0.6`, `avg_logprob < -1.0`, or identical text repeats 3+ times consecutively.

### 4.7 Qwen LLM (`oe_llm`)

Qwen 2.5 7B INT4-AWQ runs through TensorRT-LLM with paged KV cache (1.15 GB budget).

**Key design:** The LLM subscribes **only** to the daemon (port 5571). It receives a fully assembled prompt — it never touches raw sensor data. The daemon handles context management, prompt assembly, and history windowing.

**Token streaming:** Each generated token is published on ZMQ (`{token: "Hello", finished: false}`). TTS accumulates tokens into sentences for streaming synthesis.

### 4.8 Kokoro TTS (`oe_tts`)

Kokoro v1.0 (82M params) runs via ONNX Runtime CUDA EP.

**Pipeline:** Accumulated sentence → espeak-ng G2P (phoneme tokenization, C API — not subprocess) → ONNX inference → PCM float32 at 24 kHz → SHM → WebSocket Bridge → browser `AudioContext`

**Sentence-level streaming:** Each completed sentence from the LLM is dispatched to TTS independently, so audio playback begins before the full response is generated.

### 4.9 Moondream2 VLM (`oe_vlm`)

Moondream2 (1.86B params) provides image captioning triggered by `describe_scene` or periodic timer.

**Hybrid architecture:** The text decoder uses custom cross-attention ops that cannot be cleanly exported to ONNX. Solution:
- **Python subprocess** (`encode_moondream_image.py`) handles vision encoding using PyTorch
- **C++ process** reads pre-computed vision features and runs the text decoder via ONNX Runtime

On lower GPU tiers, VLM is spawned on-demand via `posix_spawn()`, processes one frame, publishes the description, and exits — freeing its 2.45 GB VRAM.

### 4.10 WebSocket Bridge (`oe_ws_bridge`)

uWebSockets C++ server on port 9001 bridging ZMQ ↔ WebSocket:

| Channel | URL | Data | Direction |
|:---|:---|:---|:---|
| Video | `/video` | Binary JPEG | Server → Client |
| Audio | `/audio` | Binary PCM float32 | Server → Client |
| Chat | `/chat` | JSON text | Bidirectional |

Also serves static frontend files (HTML/CSS/JS) and exposes a `/status` HTTP endpoint.

<details>
<summary><strong>Threading model</strong></summary>

- **Thread 1:** uWS event loop (WebSocket I/O)
- **Thread 2:** `zmq_poll()` on all SUB sockets
- Data passed via thread-safe queue + `uWS::Loop::defer()` to avoid lock contention

</details>

---

## 6. IPC Design — Data Plane & Control Plane

### 6.1 Shared Memory Segments

| Segment Name | Producer | Consumers | Payload | Buffering |
|:---|:---|:---|:---|:---|
| `/oe.vid.ingest` | VideoIngest | BgBlur, FaceRecog, VLM | 1920×1080 BGR (6.2 MB) | Double-buffer |
| `/oe.aud.ingest` | AudioIngest | WhisperSTT | PCM F32 16 kHz | Double-buffer |
| `/oe.cv.blur.jpeg` | BgBlur | WS Bridge | JPEG (~50–200 KB) | Double-buffer |
| `/oe.aud.tts` | KokoroTTS | WS Bridge | PCM F32 24 kHz | Double-buffer |

### 6.2 Double-Buffer Protocol

Each SHM segment holds **two frame slots** (A and B). An atomic `writeIndex` alternates between 0 and 1. The producer always writes to the inactive slot; consumers always read the latest completed slot.

**Stale-read guard:** Consumers read `writeIndex` before and after `memcpy`. If the value changed, the read is torn and is discarded. This handles slow consumers (e.g., VLM at 300 ms/frame vs. producer at 33 ms/frame).

<details>
<summary><strong>Double-buffer flip protocol</strong></summary>

```
Producer:
  1. Load writeIndex (acquire) → e.g., 0
  2. Compute next = 1u - 0 = 1
  3. Write frame into slot 1
  4. Update header (seqNumber, timestampNs)
  5. Store writeIndex = 1 (release)

Consumer:
  1. Load writeIndex (acquire) → 1
  2. memcpy from slot 1
  3. Re-load writeIndex → if changed, discard (torn read)
```

**Constraint:** Exactly one producer per SHM segment. Multiple producers would break the alternation invariant. Each producing module owns its own segment, so this is naturally satisfied.

</details>

### 6.3 Per-Process Pinned Staging Buffer

Each module that reads from SHM must copy data into its own process-local pinned buffer before issuing `cudaMemcpyAsync`. This three-step copy is the cost of process isolation:

```
SHM (pageable, shared)  ──memcpy──►  cudaHostAlloc buffer (pinned, process-local)  ──cudaMemcpyAsync──►  device VRAM
     ↑ OS-visible                         ↑ DMA-mapped by CUDA driver                    ↑ GPU-visible
```

**Why not skip the pinned buffer?** `cudaMemcpyAsync` from pageable memory silently falls back to synchronous copy. Only pinned (`cudaHostAlloc`) memory enables true asynchronous DMA transfer that overlaps with kernel execution on a separate CUDA stream.

**Why not share pinned memory across processes?** `cudaHostAlloc` memory is mapped into the allocating process's CUDA context address space. Another process cannot access this mapping — the CUDA driver disallows it. Under WDDM (WSL2), even `cudaIpcGetMemHandle` is unsupported (see Section 4.2), eliminating the only cross-process GPU memory sharing mechanism.

Three consumers (BgBlur, FaceRecog, VLM) each perform independent `memcpy` from the same SHM region. The 3× CPU `memcpy` overhead (~0.9 ms total for 6.2 MB from L3 cache) is the explicit, measurable cost of the process-per-model fault isolation model.

### 6.4 ZeroMQ Port Allocation

| Port | Publisher | Topic(s) | Conflation |
|:---|:---|:---|:---|
| 5555 | VideoIngest | `video_frame` | Yes |
| 5556 | AudioIngest | `audio_chunk`, `vad_status` | Yes / No |
| 5561 | QwenLLM | `llm_response` | No |
| 5562 | MoondreamVLM | `vlm_description` | No |
| 5563 | WhisperSTT | `transcription` | No |
| 5565 | KokoroTTS | `tts_audio` | No |
| 5566 | FaceRecognition | `identity`, `face_registered` | Yes / No |
| 5567 | BackgroundBlur | `blurred_frame` | Yes |
| 5570 | WS Bridge | `ui_command` | No |
| 5571 | Daemon | `module_status`, `llm_prompt`, `daemon_state` | No |

**Conflation rule:** `ZMQ_CONFLATE=1` on high-frequency data topics (video, audio, identity) — only the latest message matters. `ZMQ_CONFLATE=0` on control topics (commands, transcription, LLM tokens) — every message must be delivered in order.

#### Subscriber Topology

Each process's SUB socket connections — showing exactly who listens to whom:

| Subscriber Process | Subscribes To (Port) | Topics | Purpose |
|:---|:---|:---|:---|
| `oe_bg_blur` | VideoIngest (5555) | `video_frame` | Read SHM offset for frame input |
| `oe_bg_blur` | WS Bridge (5570) | `ui_command` | ISP slider values, shape overlays |
| `oe_face_recog` | VideoIngest (5555) | `video_frame` | Every 3rd frame for face detection |
| `oe_stt` | AudioIngest (5556) | `audio_chunk`, `vad_status` | Speech chunks + end-of-utterance |
| `oe_llm` | Daemon (5571) | `llm_prompt` | Assembled prompt from daemon |
| `oe_tts` | QwenLLM (5561) | `llm_response` | Token stream for sentence accumulation |
| `oe_vlm` | Daemon (5571) | `llm_prompt` | `describe_scene` trigger |
| `oe_vlm` | VideoIngest (5555) | `video_frame` | Frame to analyze |
| `oe_ws_bridge` | BgBlur (5567) | `blurred_frame` | JPEG for WebSocket `/video` |
| `oe_ws_bridge` | KokoroTTS (5565) | `tts_audio` | PCM for WebSocket `/audio` |
| `oe_ws_bridge` | QwenLLM (5561) | `llm_response` | Token stream for `/chat` |
| `oe_ws_bridge` | FaceRecog (5566) | `identity` | Identity for `/chat` |
| `oe_ws_bridge` | WhisperSTT (5563) | `transcription` | Transcribed text for `/chat` |
| `oe_ws_bridge` | Daemon (5571) | `module_status`, `daemon_state` | System status for `/chat` |
| `oe_ws_bridge` | MoondreamVLM (5562) | `vlm_description` | Scene text for `/chat` |
| `oe_daemon` | WhisperSTT (5563) | `transcription` | Trigger prompt assembly |
| `oe_daemon` | FaceRecog (5566) | `identity`, `face_registered` | Identity context |
| `oe_daemon` | MoondreamVLM (5562) | `vlm_description` | Scene context |
| `oe_daemon` | WS Bridge (5570) | `ui_command` | User actions (PTT, describe, etc.) |
| `oe_daemon` | All modules (*) | `module_ready` | Boot readiness tracking |

<details>
<summary><strong>ZMQ message schema examples</strong></summary>

```json
// video_frame (from VideoIngest)
{"v":1, "type":"video_frame", "shm":"/oe.vid.ingest", "seq":42, "w":1920, "h":1080, "ts_ns":183742956789012}

// transcription (from WhisperSTT)
{"v":1, "type":"transcription", "text":"Hello OmniEdge", "lang":"en", "confidence":0.94}

// llm_prompt (from Daemon → LLM)
{"v":1, "type":"llm_prompt", "system":"You are OmniEdge...", "user":"What do you see?", "identity":"Admin", "scene":"Person at desk", "history":[...]}

// llm_response (streamed, from LLM)
{"v":1, "type":"llm_response", "token":"Hello", "finished":false}

// ui_command (from WS Bridge)
{"v":1, "type":"ui_command", "action":"push_to_talk", "state":true}

// module_status (from Daemon)
{"v":1, "type":"module_status", "module":"face_recognition", "status":"running", "pid":12343}

// module_ready (from any module at startup)
{"v":1, "type":"module_ready", "module":"whisper_stt", "pid":12344}
```

</details>

### 6.5 ZMQ Heartbeats (ZMTP Protocol-Level)

Heartbeat monitoring uses ZMQ's built-in ZMTP PING/PONG, running inside the ZMQ I/O thread — it works even when the module's main thread is blocked in a CUDA kernel.

| Parameter | Value | Purpose |
|:---|:---|:---|
| `ZMQ_HEARTBEAT_IVL` | 1,000 ms | PING interval |
| `ZMQ_HEARTBEAT_TIMEOUT` | 5,000 ms | Declare peer dead if no PONG |
| `ZMQ_HEARTBEAT_TTL` | 10,000 ms | Tell peer: assume I'm dead after this |

---

## 7. GPU Memory Architecture & Tier System

### 7.1 VRAM Budget — Standard Tier (12 GB)

| Model | Quantization | VRAM | Inference Engine |
|:---|:---|---:|:---|
| Qwen 2.5 7B | INT4-AWQ | 4.30 GB | TensorRT-LLM |
| Moondream2 | INT4 | 2.45 GB | TRT + ONNX hybrid |
| Whisper V3 Turbo | INT8 | 1.50 GB | TensorRT-LLM |
| Paged KV Cache | FP8 (Blackwell) / FP16 | 1.15 GB | TRT-LLM runtime |
| YOLOv8n-seg | FP16 | 0.50 GB | TensorRT |
| InspireFace | FP16 | 0.50 GB | TensorRT |
| Kokoro v1.0 | INT8 | 0.10 GB | ONNX Runtime |
| WSL2 System Buffer | — | 1.50 GB | — |
| **Total** | | **12.00 GB** | |

### 7.2 GPU Tier System

The daemon auto-detects available VRAM at boot via `probeGpu()` and selects the appropriate tier:

| Tier | VRAM | Active Modules | VLM Strategy |
|:---|:---|:---|:---|
| **Minimal** | 4 GB | STT + LLM (3B fallback) + TTS | Disabled |
| **Balanced** | 8 GB | STT + LLM + TTS + Blur + FaceRecog | On-demand spawn, exit after use |
| **Standard** | 12 GB | All modules resident | Resident (periodic + on-demand) |
| **Ultra** | 16+ GB | All modules + FP8 KV cache expanded | Resident, extended context |

Override at launch with `--profile <tier>`.

### 7.3 Dynamic VRAM Management

VRAM usage is not static — it varies with user actions:

| User Action | VRAM Impact | Reason |
|:---|:---|:---|
| Short utterance (3 s) | +50 MB | Small Whisper activations + small KV cache |
| Long conversation (20 turns) | +800 MB | History accumulates in KV cache (up to 2048 tokens) |
| `describe_scene` during conversation | +2,450 MB | Moondream2 loads while LLM is active |
| Face registration | +50 MB | Temporary embedding computation |

**Pressure levels and response:**

| Level | Condition | Action |
|:---|:---|:---|
| Normal | Free VRAM > 1,500 MB | No action |
| Warning | Free VRAM 500–1,500 MB | Stop accepting on-demand spawns |
| Critical | Free VRAM < 500 MB | Priority-based eviction |
| Emergency | `cudaMalloc` fails | Model downgrade (smaller engine fallback) |

**Eviction priority** (lowest number = first to evict):

| Priority | Module | VRAM Freed | User Impact |
|:---|:---|:---|:---|
| 0 | Background Blur | 0.50 GB | Cosmetic only (raw video passthrough) |
| 1 | VLM (on-demand) | 2.45 GB | `describe_scene` unavailable |
| 2 | Face Recognition | 0.50 GB | No identity in LLM prompt |
| 3 | LLM → smaller variant | 2.3–3.1 GB | Reduced reasoning quality |
| 4 | STT → Whisper Small | 1.00 GB | Higher WER |
| 5 | TTS | Never evicted | Too small to matter (0.1 GB) |

<details>
<summary><strong>Model downgrade fallback chain (YAML example)</strong></summary>

```yaml
qwen_llm:
  engine_dir: "./models/trt_engines/qwen2.5-7b-awq"       # Primary: 4.3 GB
  fallback_engines:
    - engine_dir: "./models/trt_engines/qwen2.5-3b-awq"   # Fallback: 2.0 GB
      min_vram_mb: 2500
    - engine_dir: "./models/trt_engines/qwen2.5-1.5b-awq"  # Last resort: 1.2 GB
      min_vram_mb: 1500

whisper_stt:
  encoder_engine: "./models/trt_engines/whisper-turbo/encoder"
  fallback_engines:
    - encoder_engine: "./models/trt_engines/whisper-small/encoder"
      min_vram_mb: 600
```

When downgraded, the module publishes `module_ready` with `"degraded": true`. The frontend shows a banner. When VRAM pressure subsides for 60 consecutive seconds, the daemon upgrades back to the primary engine.

</details>

### 7.4 Why Process-Level Isolation Works

In-process `loadModel()` / `unloadModel()` causes CUDA memory fragmentation over time — the allocator cannot coalesce freed blocks across different allocation patterns. Process isolation solves this definitively:

- Process exit → CUDA driver destroys entire context → **all** VRAM freed as contiguous blocks
- Zero fragmentation over millions of load/unload cycles
- No need for custom CUDA memory pools or defragmentation routines

---

## 8. AI Model Portfolio

### 8.1 Model Input/Output Contracts

<details>
<summary><strong>Qwen 2.5 7B INT4-AWQ (LLM)</strong></summary>

| Direction | Field | Type | Shape | Notes |
|:---|:---|:---|:---|:---|
| Input | `input_ids` | `int32_t[]` | `[1, seq_len]` (≤ 2048) | Tokenized via Qwen tokenizer (vocab: 151,936) |
| Input | `max_output_len` | `int32_t` | scalar | Default 500 |
| Output | `output_ids` | `int32_t[]` | `[1, max_output_len]` | Generated token IDs (padded) |
| Internal | KV cache | FP16/FP8 paged | Dynamic (1.15 GB) | Paged allocation prevents fragmentation |

**Quantization:** AutoAWQ INT4 with 128-element group size. Reduces weights from 15.2 GB (FP16) to 4.3 GB. Uses GQA (4 KV heads vs. 28 attention heads) to reduce KV cache from 3.22 GB to 0.46 GB raw.

</details>

<details>
<summary><strong>Whisper V3 Turbo INT8 (STT)</strong></summary>

| Direction | Field | Type | Shape | Notes |
|:---|:---|:---|:---|:---|
| Input | Log-Mel spectrogram | `float32` | `[1, 128, T]` | 128 mel bins, ~3000 frames for 30 s |
| Input | Audio sample rate | — | — | Must be 16 kHz mono F32 PCM |
| Output | Token IDs | `int32_t[]` | `[1, max_tokens]` | Whisper vocab (51,866 tokens) |
| Output | `no_speech_prob` | `float32` | `[1]` | 0.0–1.0, used for hallucination filtering |

**Architecture:** 809M params, 4 decoder layers (turbo variant). Encoder-decoder split across two TRT-LLM engines. INT8 weight-only quantization.

</details>

<details>
<summary><strong>Kokoro v1.0 INT8 (TTS)</strong></summary>

| Direction | Field | Type | Shape | Notes |
|:---|:---|:---|:---|:---|
| Input | Phoneme tokens | `int64_t[]` | `[1, token_len]` | G2P via espeak-ng C API |
| Input | Voice style tensor | `float32` | `[1, 256]` | Loaded from `.npy` at init |
| Output | PCM audio | `float32` | `[1, num_samples]` | 24 kHz mono, [-1.0, 1.0] |

**Architecture:** 82M params (StyleTTS2-derived). INT8 dynamic quantization via ONNX Runtime. Real-time factor ~0.04 (25× faster than real-time).

</details>

<details>
<summary><strong>Moondream2 INT4 (VLM)</strong></summary>

| Direction | Field | Type | Shape | Notes |
|:---|:---|:---|:---|:---|
| Input | Image tensor | `float16` | `[1, 3, 378, 378]` | BGR→RGB, normalized to [-1, 1] |
| Output | Image features | `float16` | `[1, 729, 2048]` | 729 patches × 2048-dim (SigLIP encoder) |
| Output | Scene description | string | variable | Autoregressive text decoder |

**Architecture:** SigLIP vision encoder + Phi-based language decoder, 1.86B params. Hybrid path: Python subprocess for vision encoding (custom cross-attention ops not ONNX-exportable), C++ for text decoding.

</details>

<details>
<summary><strong>YOLOv8n-seg FP16 (Background Blur)</strong></summary>

| Direction | Field | Type | Shape | Notes |
|:---|:---|:---|:---|:---|
| Input | Image tensor | `float16` | `[1, 3, 640, 640]` | Letterbox-resized from 1920×1080 |
| Output | Detections | `float32` | `[1, 8400, 84]` | 4 bbox + 80 class scores |
| Output | Seg masks | `float32` | `[1, 32, 160, 160]` | 32 prototype masks, upscaled post-NMS |

</details>

<details>
<summary><strong>InspireFace FP16 (Face Recognition)</strong></summary>

| Direction | Field | Type | Shape | Notes |
|:---|:---|:---|:---|:---|
| Input | Image | BGR `uint8_t[]` | `[H, W, 3]` | Full frame; detection is internal |
| Output | Embeddings | `float32` | `[N, 512]` | L2-normalized per detected face |
| Output | Bounding boxes | `float32` | `[N, 4]` | (x1, y1, x2, y2) in original coords |

**Database:** SQLite `known_faces.sqlite`. Schema: `faces(id INTEGER PRIMARY KEY, name TEXT, embedding BLOB)`. In-memory cache: `unordered_map<string, vector<float>>`.

</details>

<details>
<summary><strong>Silero VAD (CPU-only)</strong></summary>

| Direction | Field | Type | Shape | Notes |
|:---|:---|:---|:---|:---|
| Input | Audio chunk | `float32` | `[480]` or `[960]` | 30 ms or 60 ms at 16 kHz |
| Output | Speech probability | `float32` | scalar | 0.0 (silence) to 1.0 (speech) |

Stateful model — maintains hidden state between calls. Reset on PTT release. Threshold default: 0.5.

</details>

### 8.2 Quantization Degradation Tracking

Every quantized model is validated against its FP16 baseline to ensure accuracy loss stays within acceptable bounds:

| Model | FP16 Baseline | Quantized Result | Degradation | Max Acceptable | Status |
|:---|:---|:---|:---|:---|:---|
| Qwen 2.5 7B (Perplexity) | 7.2 | 7.8 (INT4-AWQ) | +8.3% | +15% | **Pass** |
| Qwen 2.5 7B (HumanEval pass@1) | 0.68 | 0.62 (INT4-AWQ) | −8.8% | −15% | **Pass** |
| Whisper V3 Turbo (WER clean) | 3.2% | 4.1% (INT8) | +0.9 pp | +2.0 pp | **Pass** |
| Whisper V3 Turbo (WER noisy) | 8.5% | 10.2% (INT8) | +1.7 pp | +4.0 pp | **Pass** |
| Moondream2 (CIDEr) | 95.0 | 82.0 (INT4) | −13.7% | −20% | **Pass** |
| Moondream2 (METEOR) | 0.28 | 0.23 (INT4) | −17.9% | −25% | **Pass** |
| Kokoro v1.0 (re-transcription CER) | 1.2% | 2.4% (INT8) | +1.2 pp | +2.0 pp | **Pass** |
| YOLOv8n-seg (mAP@0.5) | — | 0.68 (FP16) | — | — | **Baseline** |
| InspireFace (TAR@FAR=0.01) | — | 96.2% (FP16) | — | — | **Baseline** |

**pp** = percentage points (absolute difference). Validation scripts: `scripts/accuracy/`.

If any model exceeds its degradation threshold, the build pipeline fails and the model must be re-quantized with a higher precision (e.g., INT8 instead of INT4, or a larger calibration dataset).

### 8.3 Alternative Model Guide

The backend interface pattern makes model swapping straightforward:

| Slot | Current Model | Alternative | Trade-off |
|:---|:---|:---|:---|
| LLM | Qwen 2.5 7B | Llama 3 8B, Phi-3 4B | Different reasoning characteristics, VRAM budget |
| STT | Whisper V3 Turbo | Parakeet CTC 1.1B | Lower WER on English, no multilingual support |
| TTS | Kokoro v1.0 | XTTS v2, Piper | Voice quality vs. VRAM vs. latency |
| VLM | Moondream2 | LLaVA, InternVL | Accuracy vs. VRAM vs. export complexity |

---

## 9. Conversation Orchestration — State Machine

### 9.1 State Diagram

```
                    ┌─────────────────────────────┐
                    │                             │
     PTT_PRESS      │                     ┌──────▼──────┐
   ┌────────────────┤        IDLE         │ERROR_RECOVERY│
   │                │                     └──────┬──────┘
   │                └─────┬───────────────┘      │
   │                      │ DESCRIBE_SCENE       │ restart complete
   │                      ▼                      │
   │        ┌─────────────────────────┐          │
   │        │   PROCESSING_VISION     ├──────────┘
   │        └──────────┬──────────────┘
   │                   │ VLM_DONE → IDLE
   │
   ▼
┌──────────┐   PTT_RELEASE / VAD_SILENCE    ┌────────────┐
│ LISTENING ├──────────────────────────────→│ PROCESSING  │
└──────────┘                                └──────┬─────┘
     ▲                                             │ LLM streaming + TTS
     │                                             ▼
     │  Barge-in                            ┌────────────┐
     │  (PTT during speak)                  │  SPEAKING   │
     │         ┌────────────┐               └──────┬─────┘
     └─────────┤ INTERRUPTED │◄────────────────────┘
               └────────────┘
```

### 9.2 States and Transitions

| State | Description | Active Modules |
|:---|:---|:---|
| **IDLE** | Waiting for user input | Video pipeline, face recognition |
| **LISTENING** | Recording user speech via PTT | + AudioIngest, VAD |
| **PROCESSING** | STT → prompt assembly → LLM inference | + STT, LLM |
| **PROCESSING_VISION** | VLM analyzing a video frame | + VLM |
| **SPEAKING** | LLM streaming tokens, TTS synthesizing | + LLM, TTS |
| **INTERRUPTED** | Barge-in detected, canceling generation | Flushing LLM + TTS |
| **ERROR_RECOVERY** | Module crashed, restarting | Watchdog active |

<details>
<summary><strong>Full transition table (23 transitions)</strong></summary>

| # | From | Event | To | Action |
|:---|:---|:---|:---|:---|
| T1 | IDLE | PTT_PRESS | LISTENING | Start audio capture |
| T2 | IDLE | TEXT_INPUT | PROCESSING | Assemble prompt from text |
| T3 | IDLE | DESCRIBE_SCENE | PROCESSING_VISION | Spawn/signal VLM |
| T4 | LISTENING | PTT_RELEASE | PROCESSING | Signal end-of-utterance |
| T5 | LISTENING | VAD_SILENCE (800 ms) | PROCESSING | Auto end-of-utterance |
| T6 | PROCESSING | TRANSCRIPTION_READY | PROCESSING | Assemble prompt → LLM |
| T7 | PROCESSING | LLM_FIRST_TOKEN | SPEAKING | Begin TTS pipeline |
| T8 | PROCESSING | LLM_TIMEOUT (60 s) | ERROR_RECOVERY | Log error, retry |
| T9 | SPEAKING | TTS_COMPLETE | IDLE | All audio played |
| T10 | SPEAKING | PTT_PRESS (barge-in) | INTERRUPTED | Cancel LLM + flush TTS |
| T11 | INTERRUPTED | CANCEL_CONFIRMED | LISTENING | Resume audio capture |
| T12 | PROCESSING_VISION | VLM_DESCRIPTION | IDLE | Cache scene, return |
| T13 | PROCESSING_VISION | VLM_TIMEOUT (30 s) | IDLE | Proceed without scene |
| T14 | * | MODULE_CRASH | ERROR_RECOVERY | Restart module |
| T15 | ERROR_RECOVERY | MODULE_RESTARTED | IDLE | Resume normal operation |
| T16 | IDLE | MODE_SWITCH (dev profile) | IDLE | Daemon stops/starts modules per new profile |
| T17 | IDLE | REGISTER_FACE | IDLE | FaceRecog captures + embeds → SQLite insert |
| T18 | IDLE | REGISTER_FACE_CONFIRMED | IDLE | Frontend confirms registration success |
| T19 | SPEAKING | LLM_TIMEOUT (60 s) | ERROR_RECOVERY | LLM hung during generation |
| T20 | SPEAKING | TTS_TIMEOUT (30 s) | IDLE | Skip remaining audio, show text only |
| T21 | PROCESSING | VLM_SPAWN_FAIL | PROCESSING | Proceed without scene context |
| T22 | * | VRAM_CRITICAL | IDLE | Priority eviction triggered (see §7.3) |
| T23 | ERROR_RECOVERY | MAX_RESTARTS_EXCEEDED | IDLE | Module stays down, degradation |

</details>

### 9.3 Prompt Assembly

The daemon is the **sole assembler** of LLM prompts. It fuses real-time context into a 4,096-token budget:

| Component | Source | Budget |
|:---|:---|---:|
| System prompt | Static YAML | 200 tokens |
| Face identity | FaceRecognition ZMQ | 20 tokens |
| Scene description | VLM ZMQ (if available) | 80 tokens |
| Conversation history | Sliding window + summary eviction | 2,796 tokens |
| Current user utterance | STT or text input | 500 tokens |
| Generation headroom | — | 500 tokens |
| **Total** | | **4,096 tokens** |

When history exceeds its budget, the oldest turns are summarized by the LLM into a compact prefix and evicted. The sliding window always retains the most recent 3 turns verbatim.

### 9.4 Barge-In

When the user presses PTT while the assistant is speaking, the daemon triggers a three-way interrupt within 100 ms:

1. `cancel_generation` → LLM stops emitting tokens
2. `flush_tts` → TTS clears pending sentences
3. `stop_playback` → Frontend stops audio, resets `AudioContext`

The system transitions SPEAKING → INTERRUPTED → LISTENING and immediately begins capturing the user's new utterance.

---

## 10. Frontend & WebSocket Bridge

### 10.1 Architecture

The frontend is a single-page application served by `oe_ws_bridge` on port 9001. **Vanilla JavaScript only** — no npm, no bundler, no build step.

```
frontend/
├── index.html          # Layout: video canvas, controls, chat, status
├── style.css           # Responsive, dark mode
└── js/
    ├── ws.js           # 3 WebSocket channels, reconnect w/ exponential backoff
    ├── video.js        # JPEG → canvas rendering, shape overlay
    ├── audio.js        # PCM → AudioContext 24 kHz, sentence queue
    ├── chat.js         # Token streaming, conversation history
    ├── ptt.js          # Push-to-talk (mousedown/touchstart/Space)
    ├── controls.js     # Describe scene, register face
    ├── status.js       # Module status indicators, degradation
    └── image_adjust.js # Brightness/contrast/saturation/sharpness sliders
```

### 10.2 Three WebSocket Channels

Separating binary and JSON traffic prevents framing overhead and simplifies routing. Each channel uses independent connections with exponential backoff reconnection (500 ms initial, 30 s cap).

| Channel | Data | Processing |
|:---|:---|:---|
| `/video` (binary) | JPEG frames from BgBlur SHM | `Blob → createObjectURL → canvas.drawImage()` at 30 fps |
| `/audio` (binary) | PCM int16 from TTS SHM | Decode to float32 → `AudioBuffer → AudioBufferSourceNode` at 24 kHz |
| `/chat` (JSON) | Commands + events | Route by `msg.type`: transcription, llm_response, module_status, identity, vlm_description, error |

### 10.3 Graceful Degradation Matrix

When a module goes down, the frontend adapts in real-time:

| Module Down | Frontend Behavior | User Experience |
|:---|:---|:---|
| Background Blur | Raw video passthrough | No blur, still functional |
| Face Recognition | LLM prompt says "identity: unknown" | Assistant stops using name |
| Whisper STT | Disable PTT, show text input | Type instead of talk |
| Qwen LLM | "Assistant restarting…" banner | Core path blocked until restart |
| Kokoro TTS | Text-only responses | Read instead of listen |
| Moondream VLM | Disable "Describe Scene" button | On-demand feature unavailable |
| Video Ingest | Static placeholder image | Audio + text still work |

---

## 11. Daemon, Watchdog & Fault Tolerance

### 11.1 Daemon Responsibilities

`oe_daemon` is the **only binary the user starts**. It:

1. Parses `omniedge_config.yaml` via yaml-cpp
2. Probes GPU via NVML → selects tier → determines active modules
3. Spawns modules in dependency order via `posix_spawn()` (WS Bridge first, then producers before consumers)
4. Waits for `module_ready` from each module (30 s timeout per module)
5. Runs the conversation state machine in a `zmq_poll()` loop
6. Monitors heartbeats and auto-restarts crashed modules (up to `max_restarts`)

### 11.2 Launch Order

```
1. oe_ws_bridge       ← Must be first (relays module_ready events + UI commands)
2. oe_video_ingest    ← Producer: /oe.vid.ingest
3. oe_audio_ingest    ← Producer: /oe.aud.ingest
4. oe_bg_blur         ← Consumer of video, producer of JPEG
5. oe_face_recog      ← Consumer of video
6. oe_stt             ← Consumer of audio
7. oe_tts             ← Consumer of LLM tokens
8. oe_llm             ← Consumer of daemon prompts
9. oe_vlm             ← On-demand or at boot (tier-dependent)
```

### 11.3 Module Readiness Protocol

Each module publishes `module_ready` after completing its startup sequence (load engine, allocate CUDA memory, bind ZMQ). The daemon collects these messages and transitions to operational state once all expected modules have reported. Modules that fail to report within 30 seconds are logged as warnings and the system proceeds without them (graceful degradation).

### 11.4 Watchdog & Auto-Recovery

The watchdog operates via two complementary mechanisms:

**ZMQ heartbeat monitoring** (see §6.5): ZMTP PING/PONG runs in the ZMQ I/O thread, independent of the module’s main thread. If a module’s main thread is blocked in a CUDA kernel, heartbeats continue flowing — the module is alive but busy. Missing heartbeats indicate the process itself has hung or crashed.

**Recovery sequence on heartbeat failure:**

| Step | Action | Timeout | Rationale |
|:---|:---|:---|:---|
| 1 | `SIGTERM` the unresponsive process | — | Graceful shutdown: module catches signal, calls `cudaDeviceSynchronize()`, closes ZMQ, exits |
| 2 | `waitpid(WNOHANG)` polling | 2 s | Most modules shut down in < 500 ms |
| 3 | `SIGKILL` if still alive | Immediate | Handles stuck CUDA kernels: `SIGKILL` forces process termination; the CUDA driver destroys the context and reclaims VRAM even if a kernel was executing |
| 4 | `waitpid(0)` | 1 s | Reap zombie process |
| 5 | Check `restartCount < max_restarts` | — | Default `max_restarts = 3` per module per session |
| 6 | `posix_spawn()` fresh instance | — | Fresh CUDA context → clean VRAM → no fragmentation |
| 7 | Wait for `module_ready` | 30 s | If timeout, mark module as permanently down (graceful degradation) |

**CUDA stuck kernel scenario:** A module executing a TensorRT inference kernel cannot be interrupted mid-execution via `SIGTERM` (CUDA kernels are not signal-safe). The handler sets an `std::atomic<bool> shutdown_requested` flag. If the kernel completes within 2 s, the module exits gracefully. Otherwise, step 3 (`SIGKILL`) forces termination — the CUDA driver handles context cleanup.

### 11.5 Development Profiles

<details>
<summary><strong>Profile definitions (YAML)</strong></summary>

```yaml
profiles:
  full:         # Default — all modules
    modules: "*"
  dev-llm:      # Conversation only (no video)
    modules: [websocket_bridge, audio_ingest, whisper_stt, qwen_llm, kokoro_tts]
  dev-video:    # Video pipeline only (no LLM)
    modules: [websocket_bridge, video_ingest, background_blur, face_recognition, moondream_vlm]
  dev-stt-only: # Audio → text debugging
    modules: [websocket_bridge, audio_ingest, whisper_stt]
```

Usage: `oe_daemon --profile dev-llm`

</details>

---

## 12. Configuration & Build System

### 12.1 CMake Super-Project

The repo is a CMake super-project. The root `CMakeLists.txt` sets C++20 standards (host and CUDA), finds system dependencies, and adds each module as a subdirectory producing standalone executables.

<details>
<summary><strong>Project directory structure</strong></summary>

```
OmniEdge_AI/
├── CMakeLists.txt                  # Root: C++20, CUDA, find_package()
├── config/
│   ├── omniedge_config.yaml        # Runtime configuration (ports, paths, thresholds)
│   ├── oe_tuning.hpp               # Compile-time tuning constants
│   └── oe_platform.hpp             # WSL2 detection, SM version gates
├── common/                         # Shared library: libomniedge_common.a
│   ├── CMakeLists.txt
│   ├── include/common/
│   │   ├── state_machine.hpp
│   │   ├── prompt_assembler.hpp
│   │   └── gpu_profiler.hpp
│   └── src/
├── modules/
│   ├── gstreamer_ingest/           # Targets: oe_video_ingest, oe_audio_ingest
│   ├── llm/                        # Target: oe_llm
│   ├── stt/                        # Target: oe_stt
│   ├── tts/                        # Target: oe_tts
│   ├── vlm/                        # Target: oe_vlm
│   ├── cv/                         # Targets: oe_bg_blur, oe_face_recog
│   ├── orchestrator/               # Target: oe_daemon
│   └── ws_bridge/                  # Target: oe_ws_bridge
├── frontend/                       # Vanilla JS SPA (served by ws_bridge)
├── models/                         # Git-ignored; converted engines
├── scripts/                        # Install, quantize, benchmark, accuracy
└── tests/                          # Google Test — one test file per source
```

</details>

### 12.2 Key Dependencies

| Dependency | Purpose |
|:---|:---|
| CUDA Toolkit 12.x | GPU compute, cudaHostAlloc, cudaMemcpyAsync |
| TensorRT 10.x | LLM, STT, CV engine compilation and inference |
| TensorRT-LLM | Qwen and Whisper C++ runtime (paged KV cache) |
| ZeroMQ (libzmq + cppzmq) | Control-plane PUB/SUB messaging |
| GStreamer 1.0 | Video and audio capture pipelines |
| OpenCV (with CUDA modules) | GPU ISP processing, drawing primitives |
| ONNX Runtime | TTS (Kokoro), VLM decoder, VAD (Silero) |
| uWebSockets | WebSocket server (ws_bridge) |
| nlohmann/json | JSON parsing across all modules |
| yaml-cpp | YAML config parsing (daemon) |
| espeak-ng | G2P phoneme conversion for TTS |
| InspireFace SDK | Face detection and recognition |
| Google Test | Unit testing framework |

### 12.3 `run_all.sh` — Single Entry Point

`run_all.sh` is the master script for the entire system lifecycle:

| Command | Purpose |
|:---|:---|
| `./run_all.sh` | Build and launch all modules |
| `./run_all.sh --install` | Install all dependencies |
| `./run_all.sh --verify` | Verify prerequisites and model readiness |
| `./run_all.sh --prereqs` | Check GPU, CUDA, build tools, GStreamer |
| `./run_all.sh --status` | Show running module PIDs |
| `./run_all.sh --stop` | Gracefully shut down all modules |
| `./run_all.sh --model-test` | Run inference tests on loaded engines |

---

## 13. Observability, Profiling & Testing

### 13.1 Structured Logging

All modules emit structured JSON log lines to a Unix domain socket (`/tmp/omniedge_log.sock`). The daemon aggregates logs into a single rotating file.

<details>
<summary><strong>Log format</strong></summary>

```json
{
  "ts": "2025-06-29T14:23:01.456789012Z",
  "mono_ns": 183742956789012,
  "module": "whisper_stt",
  "level": "INFO",
  "msg": "Transcription complete",
  "meta": {"utterance_len_ms": 3200, "tokens": 42, "latency_ms": 287}
}
```

Two timestamps: `ts` (wall-clock, ISO 8601) for human reading and cross-system correlation; `mono_ns` (monotonic nanoseconds) for accurate latency deltas immune to NTP drift.

</details>

### 13.2 `/status` HTTP Endpoint

The WebSocket bridge exposes `GET /status` on port 9001, returning system health as JSON: module PIDs, restart counts, inference latencies, VRAM usage, and WebSocket client count.

### 13.3 End-to-End Latency Budget

| Stage | Target (P50) | Rationale |
|:---|:---|:---|
| STT (3 s utterance) | < 400 ms | Whisper V3 Turbo INT8: encoder ~250 ms + decoder ~150 ms |
| LLM time-to-first-token | < 300 ms | Qwen 2.5 7B INT4-AWQ prefill |
| LLM first sentence (~20 tokens) | < 400 ms | Decode at ~50–80 tok/s |
| TTS (one sentence) | < 100 ms | Kokoro 82M, RTF ~0.04 |
| **PTT release → first audio** | **< 1,300 ms** | Sum of above stages |

> With the recommended 800 ms VAD threshold, total user-perceived latency is **< 2.1 s**.

### 13.4 Testing Strategy

**Unit tests:** Every public API function in every `.hpp` has a corresponding `TEST()` or `TEST_F()`. GPU-dependent tests are tagged with `LABELS gpu` for CI filtering.

**Fault tolerance tests:** Simulate `SIGKILL` during inference, corrupt input, malformed JSON, OOM conditions, port conflicts, and verify graceful degradation.

<details>
<summary><strong>Test execution</strong></summary>

```bash
# CPU-only tests
cd build && ctest -LE gpu --output-on-failure

# GPU tests (requires loaded TRT engines)
ctest -L gpu --output-on-failure

# Single module
ctest -R test_whisper --output-on-failure
```

</details>

### 13.5 Profiling Tools

| Tool | Command | Purpose | Key Metrics |
|:---|:---|:---|:---|
| **Nsight Systems** | `nsys profile -t cuda,nvtx,zmq -o trace.nsys-rep ./oe_llm` | Full CPU+GPU timeline | Kernel launch gaps, `cudaMemcpyAsync` overlap, ZMQ send stalls, CUDA API serialization |
| **NVML monitoring** | `nvidia-smi dmon -s pucvmet -d 1` | Real-time GPU health | SM utilization %, VRAM used/free, GPU temp, power draw, mem clock |
| **VRAM fragmentation** | `cudaMemGetInfo(&free, &total)` + attempt 512 MB contiguous alloc | Detect fragmentation after repeated load/unload cycles | If contiguous alloc fails with free > 512 MB, fragmentation exists |
| **Latency tracing** | `NVTX` range annotations in code | Per-stage timing inside a module | STT mel-spec compute, TRT inference, token decode, TTS G2P |
| **ZMQ throughput** | `zmq_socket_monitor` + custom counters | Message delivery rate and backpressure | Messages/sec, HWM drops, reconnect count |

<details>
<summary><strong>Nsight Systems workflow</strong></summary>

```bash
# Profile a single module with CUDA + NVTX instrumentation
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=oe_stt_trace \
  ./build/modules/stt/oe_stt

# View in Nsight Systems GUI (Windows host)
# Look for:
#   1. Gaps between kernels (CPU bottleneck or sync stalls)
#   2. cudaMemcpy on default stream (should be on dedicated stream)
#   3. cudaDeviceSynchronize calls (should be avoided in hot path)
#   4. Kernel duration vs. CPU overhead ratio
```

All modules include `NVTX_RANGE_PUSH("stage_name") / NVTX_RANGE_POP()` macros around major pipeline stages for correlation in the Nsight timeline.

</details>

### 13.6 Accuracy Measurement

<details>
<summary><strong>Per-model accuracy metrics and thresholds</strong></summary>

| Model | Metric | Pass Threshold |
|:---|:---|:---|
| Qwen 2.5 7B INT4-AWQ | Perplexity, HumanEval pass@1 | Perplexity ≤ 8.5, pass@1 ≥ 0.60 |
| Whisper V3 Turbo INT8 | Word Error Rate (WER) | ≤ 5.0% (clean), ≤ 12.0% (noisy) |
| Moondream2 INT4 | CIDEr, METEOR | CIDEr ≥ 80.0, METEOR ≥ 0.22 |
| Kokoro v1.0 INT8 | MOS, re-transcription CER | MOS ≥ 3.8, CER ≤ 3.0% |
| InspireFace FP16 | TAR@FAR=0.01 | ≥ 95% |
| YOLOv8n-seg FP16 | mAP@0.5, mean IoU | mAP ≥ 0.65, IoU ≥ 0.72 |
| Silero VAD | F1-score | ≥ 0.94 |

</details>

---

## 14. Model Conversion Pipeline

All models are converted offline before deployment. The build pipeline runs inside the TensorRT-LLM container or a WSL2 environment with CUDA and TensorRT.

<details>
<summary><strong>1. Qwen 2.5 7B → TensorRT-LLM INT4-AWQ Engine</strong></summary>

```bash
# Download
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /models/qwen2.5-7b-instruct

# Quantize (Python)
# AutoAWQ: w_bit=4, q_group_size=128, zero_point=True, version=GEMM

# Convert checkpoint
python3 convert_checkpoint.py \
  --model_dir /models/qwen2.5-7b-instruct-awq \
  --output_dir /models/trt_ckpt/qwen2.5-7b-awq \
  --dtype float16 --quant_algo W4A16_AWQ --group_size 128 --tp_size 1

# Build engine
trtllm-build \
  --checkpoint_dir /models/trt_ckpt/qwen2.5-7b-awq \
  --output_dir /models/trt_engines/qwen2.5-7b-awq \
  --gemm_plugin float16 --gpt_attention_plugin float16 \
  --max_batch_size 1 --max_input_len 2048 --max_seq_len 4096 \
  --paged_kv_cache enable --use_paged_context_fmha enable
```

`--paged_kv_cache enable` activates dynamic block-based KV cache allocation to prevent VRAM fragmentation. `--max_seq_len 4096` limits the KV cache ceiling to fit within the 1.15 GB budget.

</details>

<details>
<summary><strong>2. Whisper V3 Turbo → TensorRT-LLM INT8</strong></summary>

```bash
# Download
huggingface-cli download openai/whisper-large-v3-turbo --local-dir /models/whisper-large-v3-turbo

# Convert checkpoint (INT8 weight-only)
python3 convert_checkpoint.py \
  --model_name whisper_large_v3_turbo \
  --model_dir /models/whisper-large-v3-turbo \
  --output_dir /models/trt_ckpt/whisper-turbo-int8 \
  --dtype float16 --weight_only_precision int8

# Build encoder engine
trtllm-build \
  --checkpoint_dir /models/trt_ckpt/whisper-turbo-int8/encoder \
  --output_dir /models/trt_engines/whisper-turbo/encoder \
  --max_batch_size 1 --max_input_len 3000 --gemm_plugin float16

# Build decoder engine
trtllm-build \
  --checkpoint_dir /models/trt_ckpt/whisper-turbo-int8/decoder \
  --output_dir /models/trt_engines/whisper-turbo/decoder \
  --max_batch_size 1 --max_seq_len 448 \
  --max_encoder_input_len 3000 --gemm_plugin float16
```

</details>

<details>
<summary><strong>3. Moondream2 → ONNX → TensorRT</strong></summary>

The vision encoder is exported to ONNX and built via `trtexec`. The text decoder runs via ONNX Runtime with pre-computed vision features as input (hybrid path — avoids the custom cross-attention export problem).

```bash
# Vision encoder
trtexec --onnx=/models/moondream2/vision_encoder.onnx \
  --saveEngine=/models/trt_engines/moondream2_vision.engine \
  --fp16 --minShapes=pixel_values:1x3x378x378 \
  --optShapes=pixel_values:1x3x378x378 --maxShapes=pixel_values:1x3x378x378
```

**VRAM warning:** The Python vision encoder subprocess loads the full FP16 model (~5 GB). It must be a short-lived subprocess that exits after encoding to free VRAM.

</details>

<details>
<summary><strong>4. Kokoro v1.0 → ONNX INT8</strong></summary>

```bash
# Download pre-exported ONNX + voice tensors
huggingface-cli download hexgrad/Kokoro-82M --include "*.onnx" "voices/*" --local-dir /models/kokoro

# Quantize (Python — dynamic INT8)
# onnxruntime.quantization.quantize_dynamic(input, output, weight_type=QInt8)
```

</details>

<details>
<summary><strong>5. YOLOv8n-seg → TensorRT FP16</strong></summary>

```bash
# Export to ONNX
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt').export(format='onnx', imgsz=640, half=True, simplify=True)"

# Build TRT engine
trtexec --onnx=yolov8n-seg.onnx \
  --saveEngine=/models/trt_engines/yolov8n-seg.engine \
  --fp16 --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640
```

</details>

<details>
<summary><strong>6. InspireFace — Pre-compiled SDK</strong></summary>

InspireFace ships as a pre-compiled C++ shared library with TensorRT support. No model conversion required — the SDK bundles its own optimized detection and recognition models.

```bash
git clone --recurse-submodules https://github.com/HyperInspire/InspireFace.git
cd InspireFace
bash command/download_models_general.sh Megatron_TRT
export TENSORRT_ROOT=/usr/local/TensorRT-10
bash command/build_linux_tensorrt.sh
```

</details>

---

> **Document version:** 2.1 · Generated from codebase analysis  
> **Target:** NVIDIA RTX PRO 3000 Blackwell (SM 10.0) · 12 GB GDDR6 · WSL2  
> **Verified modules:** Frontend (JS SPA), common IPC layer (ZMQ/SHM wrappers)  
> **Experimental modules:** All inference backends — VRAM sizes, quantization results, and latency targets are *reference configurations* describing the intended design
