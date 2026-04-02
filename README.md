# OmniEdge_AI -- Design Architecture

> **Version:** 5.0 -- **Status:** Reference Architecture
> **Target GPU:** NVIDIA RTX PRO 3000 (Blackwell) -- 12 GB GDDR6 -- SM 10.0+ -- Tensor Cores (FP8/INT4). Scales to 4-16+ GB via GPU tier system.
> **Runtime:** WSL2 (Ubuntu 22.04+) on Windows 10/11 -- CUDA 12.x -- TensorRT 10.x -- WDDM GPU Paravirtualization
> **Capture:** USB webcam (reference: ROCWARE RC28, 1080p 60 fps, dual microphones) via `usbipd-win` or TCP fallback


---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Design Principles](#3-design-principles)
4. [CUDA Execution Model](#4-cuda-execution-model)
5. [Module Architecture](#5-module-architecture)
6. [IPC Design -- Data Plane and Control Plane](#6-ipc-design----data-plane-and-control-plane)
7. [GPU Memory Architecture and Tier System](#7-gpu-memory-architecture-and-tier-system)
8. [Conversation Orchestration -- State Machine](#8-conversation-orchestration----state-machine)
9. [Frontend and WebSocket Bridge](#9-frontend-and-websocket-bridge)
10. [Daemon, Watchdog and Fault Tolerance](#10-daemon-watchdog-and-fault-tolerance)
11. [Configuration and Build System](#11-configuration-and-build-system)
12. [Observability and Testing](#12-observability-and-testing)

---


## 1. Executive Summary

OmniEdge_AI is a multi-modal, real-time AI assistant running on a single NVIDIA GPU (12 GB VRAM) under WSL2. It captures live video and audio from a USB webcam and operates in **three mutually exclusive GPU inference modes** -- Conversation, Super Resolution, Image Transform -- managed by the daemon's `ModeOrchestrator`. Video ingest, audio ingest, ISP adjustments, and background blur run continuously across all modes.

On every mode switch the daemon kills previous-mode processes (`SIGTERM` / `waitpid`), verifies VRAM reclamation via `GpuProfiler`, spawns new processes (`posix_spawn`), and waits for `module_ready`.

Four invariants define the architecture.

### Architectural Invariants

| Invariant | Mechanism | Consequence |
|:---|:---|:---|
| **Process-per-model fault isolation** | Independent C++20 binaries (always-on + mode-dependent), each with its own `cudaSetDevice(0)` context. No shared CUDA contexts, no MPS. | A segfault in one process cannot corrupt another's model state. CUDA driver reclaims VRAM as contiguous blocks. |
| **Dual-layer IPC** | POSIX `shm_open`/`mmap` for bulk data (BGR frames, PCM audio). ZeroMQ PUB/SUB for JSON control messages. | No module links to another's library. The IPC boundary is the OS process boundary. |
| **Process-level VRAM lifecycle** | `posix_spawn()` to load a model, `SIGTERM` to unload. CUDA context destruction guarantees zero fragmentation. | No custom CUDA allocator, no memory pool, no `cuMemPool`. The OS process is the memory scope. |
| **WDDM compatibility** | All pinned host memory via `cudaHostAlloc` (not `cudaHostRegister`). No CUDA IPC (`cudaIpcGetMemHandle`). | Runs under WSL2's WDDM GPU paravirtualization without driver-level workarounds. |

**End-to-end latency target:** PTT release to first audio < 1.3 s (P50, excluding 800 ms VAD). Total user-perceived: < 2.1 s.

---


## 2. System Overview

### 2.1 Architecture at a Glance

The system is organized into three layers: GPU inference, independent C++20 processes, and the browser frontend. All three communicate through POSIX shared memory for bulk data and ZeroMQ PUB/SUB for control messages.

**GPU layer (NVIDIA RTX PRO 3000 Blackwell, 12 GB GDDR6).** Two workloads are always resident: the YOLO-based instance segmentation model with ISP adjustments and background blur compositing (FP16, approximately 0.50 GB), and the video/audio ingest pipelines (zero VRAM). WSL2 reserves approximately 1.50 GB as a system buffer, leaving the remaining VRAM for one of the three mutually exclusive inference modes. Mode 1 (Conversation, default and highest priority) loads one of three model options -- Qwen2.5-Omni-7B with 4-bit GPTQ/AWQ quantization for unified STT, LLM, and TTS; Qwen2.5-Omni-3B with the same unified architecture at reduced parameter count; or Gemma 4 E4B for audio and vision input with a separate user-selectable TTS sidecar. Conversation mode also enables audio denoise as an optional toggle and face recognition for identity injection into the LLM system prompt. Mode 2 (Super Resolution) unloads the conversation model and loads BasicVSR++ (FP16, approximately 1.50 GB) for temporal video enhancement. Mode 3 (Image Transform) unloads the conversation model and loads the Stable Diffusion pipeline with ControlNet, IP-Adapter, InstantStyle, and IP-Adapter FaceID for style transfer and image generation. Only one mode occupies the GPU at a time; switching modes evicts the current model before loading the next.

**Process layer (independent C++20 binaries under WSL2).** Always-on processes include `oe_video_ingest`, `oe_audio_ingest`, and `oe_bg_blur`. Mode-dependent processes are spawned and terminated by the daemon as the user switches modes: `oe_conversation` and optionally `oe_tts` (Gemma only), `oe_face_recog`, and `oe_audio_denoise` in Mode 1; `oe_super_res` in Mode 2; `oe_img_transform` in Mode 3. Infrastructure processes `oe_ws_bridge` and `oe_daemon` run continuously and manage WebSocket relay and process lifecycle respectively.

**Frontend layer (browser SPA).** The browser connects to `oe_ws_bridge` on port 9001 over three WebSocket channels: `/video` for binary JPEG frames rendered to an HTML `<canvas>` at 30 fps, `/audio` for binary PCM audio played through an AudioContext at 24 kHz, and `/chat` for JSON control messages and streamed LLM tokens. The UI exposes a model selector dropdown, a mode selector, and a TTS selector visible only when Gemma 4 E4B is active.

### 2.2 Process Map

Each process is a standalone C++20 binary with its own CUDA context. The daemon spawns and manages all others.

**Always-On Processes:**

| Process Binary | Role | Inference Engine | ZMQ PUB Port |
|:---|:---|:---|---:|
| `oe_video_ingest` | Webcam capture via GStreamer | -- | 5555 |
| `oe_audio_ingest` | Microphone capture, VAD gating | ONNX Runtime CPU | 5556 |
| `oe_bg_blur` | Segmentation, ISP, blur compositing | TensorRT | 5567 |
| `oe_ws_bridge` | WebSocket server, ZMQ-to-browser relay | uWebSockets | 5570 |
| `oe_daemon` | Process lifecycle, state machine, mode orchestration | -- | 5571 |

**Mode 1 -- Conversation (default):**

| Process Binary | Role | Inference Engine | ZMQ PUB Port |
|:---|:---|:---|---:|
| `oe_conversation` | Unified STT+LLM+TTS (Qwen2.5-Omni) or STT+LLM (Gemma 4 E4B) | TensorRT-LLM / ONNX | 5561 |
| `oe_tts` | Text-to-speech (spawned only when Gemma 4 E4B is selected) | ONNX Runtime CUDA EP | 5565 |
| `oe_face_recog` | Face detection and identification | TensorRT | 5566 |
| `oe_audio_denoise` | Real-time audio denoising (optional toggle) | ONNX Runtime CUDA EP | 5569 |

The UI exposes a model selector dropdown with three conversation model options:
1. **Qwen2.5-Omni-7B** (4-bit GPTQ/AWQ) -- default. Handles speech-to-text, language modeling, and text-to-speech natively in a single process. No separate STT or TTS modules are spawned.
2. **Qwen2.5-Omni-3B** -- smaller model for constrained platforms. Same unified STT+LLM+TTS behavior.
3. **Gemma 4 E4B** -- accepts audio buffers and image frames as native input (no separate STT). Produces no audio output, so a user-selectable TTS process is spawned alongside it and the UI reveals an additional TTS model selector.

Switching between conversation models follows the same kill-spawn cycle: the daemon terminates the previous model process and spawns the newly selected one.

**Mode 2 -- Super Resolution:**

| Process Binary | Role | Inference Engine | ZMQ PUB Port |
|:---|:---|:---|---:|
| `oe_super_res` | Temporal video enhancement (BasicVSR++) | ONNX Runtime CUDA EP | 5568 |

Uses the always-on video ingest SHM feed. All conversation models and image transform processes are killed on entry. The UI presents frame and clip selection controls.

**Mode 3 -- Image Transform:**

| Process Binary | Role | Inference Engine | ZMQ PUB Port |
|:---|:---|:---|---:|
| `oe_img_transform` | Stable Diffusion + ControlNet / IP-Adapter | TensorRT | 5572 |

Spawns a Stable Diffusion pipeline with ControlNet or IP-Adapter for structure-preserving style transfer. All conversation and super resolution processes are killed on entry. The UI provides a style picker and an image upload area.

**On-Demand CV Extensions:**

| Process Binary | Role | Inference Engine | ZMQ PUB Port | VRAM |
|:---|:---|:---|---:|---:|
| `oe_face_filter` | FaceMesh V2 AR filters | ONNX Runtime (TRT EP) | 5575 | ~100 MiB |
| `oe_sam2` | SAM2 interactive segmentation | ONNX Runtime (TRT EP) | 5576 | ~800 MiB |

**SAM2 (Segment Anything Model 2)** provides interactive, pixel-perfect segmentation. The user clicks a point or draws a bounding box on the live video feed; SAM2 produces a binary mask at the original resolution. The two-stage ONNX architecture (image encoder + mask decoder) allows the image to be encoded once and then re-used for multiple prompts at ~20ms per prompt. SAM2 is spawned on-demand when the user activates it from the sidebar. See `omniedge-cv/references/sam2.md` for the full model definition and pipeline details.

Model names, parameter counts, quantization methods, and VRAM footprints for each module are defined in the corresponding per-module skill's Model Definition section. The per-module skill is authoritative for model specifics; the VRAM Budget table in Section 7 is derived from those skills.

---


## 3. Design Principles

### 3.1 Process-per-Model Isolation

Loading a model = `posix_spawn()`. Unloading = `SIGTERM` + `waitpid()`. On process exit the CUDA driver destroys the context and reclaims VRAM as contiguous blocks -- zero fragmentation across millions of load/unload cycles.

The in-process alternative (`loadModel` / `unloadModel` in a long-lived process) fragments CUDA memory over time. After many cycles, `cudaMalloc` fails even with sufficient total free VRAM because no contiguous block is large enough. TensorRT engines and paged KV caches require large contiguous allocations, making fragmentation especially damaging. Process isolation eliminates this entirely via low-level virtual memory unmapping on context destruction.

### 3.2 Dual-Layer IPC

| Layer | Transport | Use Case | Latency |
|:---|:---|:---|:---|
| **Data Plane** | POSIX shared memory (`shm_open` / `mmap`) | BGR frames, PCM audio, JPEG output | < 0.3 ms |
| **Control Plane** | ZeroMQ PUB/SUB (TCP localhost) | JSON commands, transcriptions, token streams | < 0.1 ms |

Each module consolidates all ZMQ networking into a single `MessageRouter` instance (Section 6.4).

### 3.3 Layered Module Design

| Layer | Directory | Contains |
|:---|:---|:---|
| **Interfaces** | `modules/interfaces/include/` | Pure virtual C++ contracts (`ConversationInferencer`, `TTSInferencer`, `BlurInferencer`, etc.) |
| **Core** | `modules/core/<module>/` | Concrete inferencers, model loading, pure algorithms (no transport) |
| **Node** | `modules/nodes/<module>/` | `*Node` wiring classes + `main.cpp` producing standalone binaries |

**Dependency direction (strict, no cycles):**

```text
nodes/ → core/ + interfaces/ + transport/ + hal/ + common/
core/ → interfaces/ + hal/ + common/
transport/ → common/
hal/ → common/
interfaces/ → common/
```

This enables isolated testing: core tests link only `core/` + `interfaces/`, node tests inject mocks via interfaces, integration tests use real ZMQ + GPU inferencers. Models are swappable at `core/` without touching node wiring or transport.

### 3.4 Inferencer Swappability (Strategy Pattern)

Every inference module implements a role-specific C++ interface selected from `config/omniedge_config.yaml`. Swapping models requires only a new inferencer implementation and a YAML change.

| Interface | Current Inferencer |
|:---|:---|
| `ConversationInferencer` | `QwenOmniInferencer` / `GemmaE4BInferencer` |
| `TTSInferencer` | TTS sidecar (Gemma only) |
| `BlurInferencer` | YoloSegEngine + ISP + nvJPEG |
| `FaceRecogInferencer` | `InspireFaceWrapper` (C API) |
| `SuperResInferencer` | `OnnxBasicVsrppInferencer` |
| `ImgTransformInferencer` | `SDControlNetInferencer` |
| `AudioDenoiseInferencer` | `OnnxDtlnInferencer` |

`ConversationInferencer` unifies STT+LLM+TTS. Implementations exposing native audio output (Qwen2.5-Omni) vs. text-only (Gemma 4 E4B) share the same interface; `supportsNativeAudio()` tells the daemon whether to spawn a TTS sidecar.

### 3.5 No Hardcoded Numbers

| Layer | Location | Scope |
|:---|:---|:---|
| **Runtime Config** | `config/omniedge_config.yaml` | Ports, thresholds, engine paths |
| **Compile-time Tuning** | `*_constants.hpp` in `modules/common/include/common/constants/` | Buffer sizes, ZMQ HWM, frame dimensions |
| **Platform Detection** | `modules/common/include/common/platform_detect.hpp` | `OE_PLATFORM_WSL2`, `OE_HAS_BLACKWELL_FEATURES` |
| **Runtime Defaults** | `modules/common/include/common/runtime_defaults.hpp` | Fallback values when YAML fields are missing |

No bare numeric literals in `.cpp` files. Architectural constants live as `constexpr` in headers.

### 3.6 WSL2 Constraints

**`cudaHostAlloc` only.** `cudaHostRegister` is unsupported under WDDM. Detected at compile time via `OE_PLATFORM_WSL2`.

**No CUDA IPC.** `cudaIpcGetMemHandle` returns `cudaErrorNotSupported` under WDDM. The POSIX SHM data-plane design works on native Linux, WSL2, and containers without driver-level workarounds.

---


## 4. CUDA Execution Model

Each process creates its own CUDA context via `cudaSetDevice(0)`. Under WDDM, the driver time-slices contexts -- no concurrent kernel execution across processes. MPS is rejected (WDDM-incompatible, breaks fault isolation, loses deterministic VRAM reclamation). CUDA IPC is unavailable under WDDM; the host-side copy cost (~0.8 ms for a 6.2 MB BGR frame via SHM + pinned buffer + PCIe Gen4) is negligible vs. 100-400 ms inference latency.

### 4.1 CUDA Stream Usage Per Module

Each module uses dedicated CUDA streams for overlapping data transfer with inference computation. No module uses the default stream (stream 0).

| Module | Streams | Overlap Strategy |
|:---|:---|:---|
| `oe_conversation` | Model-dependent | Qwen2.5-Omni: internal stream management for unified STT+LLM+TTS. Gemma 4 E4B: similar, audio+vision input streams |
| `oe_tts` (Gemma sidecar) | 2 | `stream_infer` (ONNX execution) overlaps with `stream_copy` (PCM output D-to-H) |
| `oe_bg_blur` | 3 | `stream_copy_in` (frame H-to-D), `stream_infer` (segmentation + ISP + blur + nvJPEG), `stream_copy_out` (JPEG D-to-H). Triple overlap at 30 fps |
| `oe_face_recog` | 2 | `stream_copy` (frame H-to-D, every 3rd frame) overlaps with `stream_infer` (detect + align + embed) |
| `oe_super_res` | 2 | `stream_copy` (frame H-to-D) overlaps with `stream_infer` (temporal enhancement + nvJPEG encode) |
| `oe_img_transform` | Model-dependent | Stable Diffusion: internal stream management for diffusion steps + ControlNet/IP-Adapter |
| `oe_audio_denoise` | 1 | ONNX Runtime CUDA EP manages internally. Audio chunks are small (~2 KB), copy overhead negligible |

**Stream priorities** (`modules/hal/gpu/include/gpu/cuda_priority.hpp`) affect only intra-process scheduling. Under process-per-model isolation, cross-process scheduling is managed by the WDDM time-slicing scheduler.

| Priority | Modules | Rationale |
|---:|:---|:---|
| -5 (highest) | Conversation Model | Unified STT+LLM+TTS latency is directly user-visible |
| -3 | TTS Sidecar (Gemma only) | Audio synthesis; slight delay masked by text bubble display |
| -1 | Super Resolution, Image Transform | Mode-specific inference, not in voice latency path |
| 0 (lowest) | Background Blur, Face Recognition, Audio Denoise | Background/passive tasks |

### 4.2 GPU Architecture Targeting

TensorRT engines are SM-specific and not portable across GPU generations.

| Configuration | Value | Rationale |
|:---|:---|:---|
| `CMAKE_CUDA_ARCHITECTURES` | `75;80;86;89;90;100;120` | Turing through Blackwell |
| `trtexec --best` | FP16 + INT8 + FP8 | Blackwell supports FP8 for smaller KV cache |
| Engine portability | None | SM 10.0 engine does not load on SM 8.9 or 8.6 |
| Fallback | `trtexec` at deploy-time | `scripts/build_engines.sh` builds on first launch |

### 4.3 CUDA Optimization Strategy

Every inferencer class that processes per-frame or per-request data on the GPU must apply CUDA optimization patterns to eliminate CPU bottlenecks. The full pattern catalog is in `.claude/skills/omniedge-codegen/references/cuda-optimization.md`.

#### 4.3.1 GPU-Accelerated Preprocessing

Image inferencers must perform preprocessing (resize, color conversion, normalization) on the GPU via fused CUDA kernels, not CPU OpenCV. CPU preprocessing (`cv::resize` + `cv::cvtColor` + `cv::split`) wastes 2-5 ms per frame — operations the GPU executes in < 0.1 ms.

**Canonical pattern:** Write a single fused CUDA kernel in `<module>_preprocess.cu` that performs bilinear resize + BGR-to-RGB + uint8-to-float normalize in one pass. One thread per output pixel. Reference implementation: `resizeNormFP16Kernel` in `modules/core/cv/src/yolo_seg_engine.cu`.

| Inferencer | Preprocessing | Required Kernel |
|:---|:---|:---|
| `YoloSegEngine` | GPU (already has `resizeNormFP16Kernel`) | Done |
| `OnnxBasicvsrppInferencer` | CPU (`packFrameToChw` with OpenCV) | Fused resize+BGR2RGB+normalize |
| `OnnxFaceFilterInferencer` | CPU (OpenCV) | Fused resize+normalize for FaceMesh |
| `OnnxSam2Inferencer` | CPU (placeholder) | GPU kernel when real model added |
| `GpuCompositor` | GPU (already has upscale+composite+ISP kernels) | Consider kernel fusion |
| `OnnxKokoroInferencer` | N/A (text input) | Not applicable |
| `OnnxDtlnInferencer` | CPU buffer copy (audio, small) | Pinned memory only |

#### 4.3.2 Pinned Memory and Async Transfers

All host-device transfer buffers must use `PinnedStagingBuffer` (from `gpu/pinned_buffer.hpp`) for page-locked DMA. Standard `std::vector` memory requires a driver-side copy that roughly halves bandwidth. All copies use `cudaMemcpyAsync` — synchronous `cudaMemcpy` is banned in hot paths. `cudaStreamSynchronize` is called at most once per frame, only before reading host output.

#### 4.3.3 Stream Overlap

Modules use 2-3 CUDA streams to pipeline H2D copy, compute, and D2H copy. Cross-stream dependencies use `cudaStreamWaitEvent` (GPU-side, non-blocking) — never `cudaStreamSynchronize` for this purpose. See Section 4.1 for per-module stream counts.

#### 4.3.4 GPU-Accelerated JPEG Encoding (nvJPEG)

Modules producing JPEG output (BackgroundBlur, VideoDenoise, FaceFilter) should use NVIDIA's nvJPEG library for GPU-side encoding. This eliminates the D2H copy of the raw frame (~6 MB at 1080p) — only the compressed JPEG (typically 50-150 KB) is transferred to host. Saves 3-8 ms per frame vs. CPU `cv::imencode`.

#### 4.3.5 ONNX Runtime CUDA EP Configuration

All ONNX-based inferencers configure the CUDA Execution Provider with:
- `do_copy_in_default_stream = 0` — enables internal copy/compute overlap
- `gpu_mem_limit` — set to module's VRAM budget from `vram_thresholds.hpp`
- `enable_cuda_graph = 1` — for fixed-shape models only (DTLN, BasicVSR++, FaceMesh)
- `cudnn_conv_algo_search = EXHAUSTIVE` — selects optimal cuDNN algorithm at first inference

#### 4.3.6 CUDA Graphs

For fixed-shape, fixed-sequence inference pipelines (e.g., GPU compositor: upscale + composite + ISP), the kernel launch sequence can be captured into a CUDA graph and replayed with near-zero launch overhead. Not applicable to variable-length operations (LLM generation, variable audio).

---


## 5. Module Architecture

### 5.1 Data Flow Overview

```text


Always-On Pipeline (all modes):
Webcam (USB) --> GStreamer --> Video Ingest --shm--> Background Blur (ISP + YOLO seg) --shm--> WS Bridge --> Browser Canvas
Microphone  --> GStreamer --> Audio Ingest --shm--> (available to active mode)

Conversation Mode (default):
Audio Ingest --shm--> Conversation Model (Qwen2.5-Omni or Gemma 4 E4B)
Video Ingest --shm--> Conversation Model (multimodal input)
                      Conversation Model --zmq--> WS Bridge --> Browser (text + audio)
                      Conversation Model --zmq--> TTS (Gemma only) --shm--> WS Bridge --> Browser
Face Recognition --zmq--> Daemon (identity injection into conversation prompt)
Audio Denoise (optional) --shm--> Conversation Model (cleaned audio input)

Super Resolution Mode:
Video Ingest --shm--> BasicVSR++ --shm--> WS Bridge --> Browser

Image Transform Mode:
User upload --zmq--> Stable Diffusion + ControlNet/IP-Adapter --zmq--> WS Bridge --> Browser
```

The **vision pipeline** (webcam → blur → JPEG → browser) runs continuously at 30 fps. In Conversation mode, the **voice pipeline** (audio → conversation model → speech) is the critical latency path.

### 5.2 Video Ingest (`oe_video_ingest`, port 5555)

**Primary path (V4L2 via usbipd):** Webcam attached to WSL2 via `usbipd attach --wsl` as `/dev/video0`. Pipeline: `v4l2src` → `jpegdec` → `videoconvert BGR` → `appsink drop=true max-buffers=2`. Eliminates H.264 overhead and TCP buffering.

### 5.3 Audio Ingest (`oe_audio_ingest`, port 5556)

Captures 16 kHz mono F32 audio from the Windows host over TCP port 5001. Host pipeline: `wasapi2src` → `audioresample` 16 kHz mono F32LE → `tcpserversink` port 5001. Audio uses TCP because USB audio passthrough via usbipd is unreliable.

**VAD gating:** Silero VAD (CPU-only ONNX) gates every audio chunk. Only speech is written to SHM (`/oe.aud.ingest`); silence is discarded to prevent hallucinations. End-of-utterance published as `vad_status` after configurable silence threshold (default 800 ms).

### 5.4 Background Blur (`oe_bg_blur`, port 5567)

Five-step GPU pipeline reading from video SHM, writing JPEG to `/oe.cv.blur.jpeg`:

1. **ISP adjustments** -- brightness, contrast, saturation, sharpness via OpenCV CUDA (skipped at defaults)
2. **Instance segmentation** -- per-pixel person mask
3. **Mask upscale** -- bilinear to 1920x1080
4. **Gaussian blur compositing** -- background only; foreground passes through
5. **nvJPEG encode** -- GPU-accelerated JPEG compression

**Shape overlays** from browser canvas are composited into the blurred region only (< 0.5 ms for 50 shapes).

### 5.5 Face Recognition (`oe_face_recog`, port 5566)

Every 3rd frame (~10 fps): detect → align → extract 512-dim embedding → compare against SQLite gallery (`known_faces.sqlite`) via cosine similarity. Publishes identity on ZMQ for injection into the conversation prompt.

### 5.6 Conversation Model (`oe_conversation`, port 5561)

Three model options via UI dropdown. Switching models follows the standard kill-spawn cycle.

| Model | Capabilities | TTS Sidecar? |
|:---|:---|:---|
| **Qwen2.5-Omni-7B** (default) | Unified STT+LLM+TTS, 4-bit GPTQ/AWQ | No |
| **Qwen2.5-Omni-3B** | Same unified behavior, smaller footprint | No |
| **Gemma 4 E4B** | Audio+vision input, text-only output | Yes -- user-selectable TTS spawned alongside |

The model subscribes to the daemon for assembled prompts, receives audio from AudioIngest SHM and optionally video from VideoIngest SHM. Tokens stream on ZMQ (`{token: "Hello", finished: false}`). For Gemma, the TTS sidecar accumulates tokens into sentences for streaming synthesis.

**Audio denoise toggle:** optional cleanup stage before conversation model audio input, available with all models.

### 5.7 Text-to-Speech Sidecar (`oe_tts`, port 5565)

Spawned only when Gemma 4 E4B is selected. Qwen2.5-Omni handles TTS natively.

**Pipeline:** Sentence accumulation → espeak-ng G2P (C API) → ONNX inference → PCM F32 24 kHz → SHM (`/oe.aud.tts`) → WebSocket → browser. Each sentence dispatched independently for streaming playback.

### 5.8 Super Resolution Mode (`oe_super_res`, port 5568)

Mutually exclusive with Conversation and Image Transform. BasicVSR++ reads from `/oe.vid.ingest` SHM, applies temporal enhancement via bidirectional propagation with flow-guided deformable alignment, writes to `/oe.vid.superres` SHM. UI presents frame/clip selection controls.

### 5.9 Image Transform Mode (`oe_img_transform`, port 5572)

Mutually exclusive with Conversation and Super Resolution. Stable Diffusion + ControlNet/IP-Adapter for structure-preserving style transfer.

**Adapters:** InstantStyle (artistic style, preserve pose), IP-Adapter FaceID (character transformation, preserve facial structure), ControlNet (structural constraints: edges, depth, pose).

**Workflow:** User uploads image + selects style → ZMQ → FSM enters `kImageTransform` → diffusion inference → result published to UI.

### 5.10 Audio Denoise (`oe_audio_denoise`, port 5569)

Optional toggle in Conversation mode. Two-stage ONNX denoising (STFT magnitude separation → complex spectrum reconstruction) on PCM from `/oe.aud.ingest`, writes cleaned audio to `/oe.aud.denoise`. Stateful LSTM hidden state across chunks.

### 5.11 Face Recognition (`oe_face_recog`, port 5566)

Active in Conversation mode only. Provides identity context for prompt assembly. Killed on mode switches to Super Resolution or Image Transform.

### 5.12 WebSocket Bridge (`oe_ws_bridge`, port 5570)

uWebSockets C++ server on port 9001. Serves static frontend files and `/status` HTTP endpoint.

| Channel | Data | Direction |
|:---|:---|:---|
| `/video` | Binary JPEG from BgBlur SHM (always-on, 30 fps) | Server → Client |
| `/audio` | Binary PCM F32 from conversation model or TTS sidecar | Server → Client |
| `/chat` | JSON commands/events + binary (image upload, mode results) | Bidirectional |

**Threading:** uWS event loop (thread 1) + `zmq_poll()` (thread 2), bridged via thread-safe queue + `uWS::Loop::defer()`.

---


## 6. IPC Design -- Data Plane and Control Plane

### 6.1 Shared Memory Segments

**Always-On:**

| Segment Name | Producer | Consumers | Payload | Buffering |
|:---|:---|:---|:---|:---|
| `/oe.vid.ingest` | VideoIngest | BgBlur, ConversationModel, FaceRecog, SuperRes | 1920x1080 BGR (6,220,800 bytes) | 4-slot circular buffer (`ShmCircularBuffer`) |
| `/oe.aud.ingest` | AudioIngest | ConversationModel, AudioDenoise | PCM F32 16 kHz (30 ms chunks) | 4-slot circular buffer |
| `/oe.cv.blur.jpeg` | BgBlur | WS Bridge | JPEG (variable, max 1 MiB/slot) | `ShmCircularBuffer` (lock-free atomics) |

**Conversation Mode:**

| Segment Name | Producer | Consumers | Payload | Buffering |
|:---|:---|:---|:---|:---|
| `/oe.aud.conversation` | ConversationModel (Qwen Omni) | WS Bridge | PCM F32 (synthesized speech, native TTS) | `ShmCircularBuffer` |
| `/oe.aud.tts` | TTS Sidecar (Gemma only) | WS Bridge | PCM F32 24 kHz (max 10 s = 240,000 samples) | `ShmCircularBuffer` |
| `/oe.aud.denoise` | AudioDenoise | ConversationModel | PCM F32 16 kHz (cleaned audio, optional) | `ShmCircularBuffer` |
| `/oe.vid.register` | WS Bridge (temp) | FaceRecognition | Raw image bytes for face registration | Variable |

**Super Resolution Mode:**

| Segment Name | Producer | Consumers | Payload | Buffering |
|:---|:---|:---|:---|:---|
| `/oe.vid.superres` | SuperRes | WS Bridge | Enhanced JPEG (variable, max 1 MiB/slot) | `ShmCircularBuffer` |

SHM naming follows the pattern `/oe.<layer>.<module>[.<qualifier>]`. Layer values: `vid`, `aud`, `cv`, `nlp`.

### 6.2 Circular Buffer Layout

All SHM segments use `ShmCircularBuffer<Header>` (defined in `modules/transport/shm/include/shm/shm_circular_buffer.hpp`). The memory layout within each segment:

```text
Offset 0:          [Header]            -- POD header (e.g., ShmVideoHeader), padded to 64 bytes
Offset 64:         [ShmCircularControl] -- 64 bytes: uint64_t writePos, readPos (atomics, cache-line aligned)
Offset 128:        [Slot 0]            -- payload bytes
Offset 128+S:      [Slot 1]
  ...
Offset 128+(N-1)*S:[Slot N-1]
```

All segments use 4 slots. Producer: `acquireWriteSlot()` / `commitWrite()`. Consumer: `acquireReadSlot()` / `advanceRead()` or `readLatestSlot()` (skip to freshest). `ShmCircularReader` provides a lightweight consumer.

**Atomic ordering:** `writePos`/`readPos` use `std::atomic<uint64_t>` with acquire/release semantics -- guarantees frame-complete reads without locks. Monotonic counters never overflow in practice.

**Constraint:** Exactly one producer per segment.

### 6.3 Per-Process Pinned Staging Buffer

```text
SHM (pageable, shared)  --memcpy-->  cudaHostAlloc (pinned, process-local)  --cudaMemcpyAsync-->  GPU VRAM
```

`cudaMemcpyAsync` from pageable memory silently falls back to synchronous copy. Only pinned memory enables true async DMA overlapping with kernel execution. Pinned memory cannot be shared across processes (CUDA IPC unsupported under WDDM). The 3x `memcpy` overhead in Conversation mode (~0.9 ms total) is the measurable cost of process isolation.

### 6.4 MessageRouter

Single `MessageRouter` per process (`modules/transport/zmq/include/zmq/message_router.hpp`): ZMQ context, PUB socket, SUB sockets, interrupt PAIR for signal-safe `stop()`, blocking poll loop.

```cpp
MessageRouter router(Config{.moduleName="conversation_model", .pubPort=5561});
router.subscribe(5571, "conversation_prompt", false, [](auto& msg) { ... });
router.publishModuleReady();
router.run();  // blocks until stop()
```

### 6.5 ZeroMQ Port Allocation

**Always-On:**

| Port | Publisher | Topics | Conflation |
|:---|:---|:---|:---|
| 5555 | VideoIngest | `video_frame` | Yes |
| 5556 | AudioIngest | `audio_chunk`, `vad_status` | Yes / No |
| 5567 | BackgroundBlur | `blurred_frame` | Yes |
| 5570 | WS Bridge | `ui_command` | No |
| 5571 | Daemon | `module_status`, `conversation_prompt`, `image_transform_prompt`, `daemon_state` | No |

**Conversation Mode:**

| Port | Publisher | Topics | Conflation |
|:---|:---|:---|:---|
| 5561 | ConversationModel | `conversation_response`, `conversation_audio` | No |
| 5565 | TTS Sidecar | `tts_audio` (Gemma 4 E4B only) | No |
| 5566 | FaceRecognition | `identity`, `face_registered` | Yes / No |
| 5569 | AudioDenoise | `denoised_audio` (optional toggle) | Yes |

**Super Resolution Mode:**

| Port | Publisher | Topics | Conflation |
|:---|:---|:---|:---|
| 5568 | SuperRes | `enhanced_frame` | Yes |

**Image Transform Mode:**

| Port | Publisher | Topics | Conflation |
|:---|:---|:---|:---|
| 5572 | ImgTransform | `transformed_image` | No |

Ports 5557-5560, 5562-5564, 5573-5579 reserved.

**Conflation:** `ZMQ_CONFLATE=1` on data topics (video, audio) -- latest-only. `ZMQ_CONFLATE=0` on control topics -- ordered delivery. Separate SUB sockets for conflated vs. non-conflated topics.

**Constants** (`zmq_constants.hpp`): `kPublisherDataHighWaterMark=2`, `kPublisherControlHighWaterMark=100`, `kSubscriberControlHighWaterMark=1000`, `kPollTimeoutMs=50`.

### 6.6 Subscriber Topology

**Always-On:**

| Subscriber Process | Subscribes To (Port) | Topics | Purpose |
|:---|:---|:---|:---|
| `oe_bg_blur` | VideoIngest (5555) | `video_frame` | Read SHM offset for frame input |
| `oe_bg_blur` | WS Bridge (5570) | `ui_command` | ISP slider values, shape overlays |
| `oe_ws_bridge` | BgBlur (5567) | `blurred_frame` | JPEG for WebSocket `/video` |
| `oe_ws_bridge` | VideoIngest (5555) | `video_frame` | Raw JPEG fallback when blur is unavailable |
| `oe_ws_bridge` | Daemon (5571) | `module_status` | System status for `/chat` |
| `oe_daemon` | WS Bridge (5570) | `ui_command` | User actions (PTT, mode switch, model select, etc.) |
| `oe_daemon` | AudioIngest (5556) | `vad_status` | End-of-utterance detection |
| `oe_daemon` | All modules | `module_ready` | Boot readiness tracking |

**Conversation Mode Subscriptions:**

| Subscriber Process | Subscribes To (Port) | Topics | Purpose |
|:---|:---|:---|:---|
| `oe_conversation` | Daemon (5571) | `conversation_prompt` | Assembled prompt from daemon |
| `oe_conversation` | AudioIngest (5556) | `audio_chunk` | Audio input (native STT) |
| `oe_conversation` | VideoIngest (5555) | `video_frame` | Video input (multimodal understanding) |
| `oe_conversation` | AudioDenoise (5569) | `denoised_audio` | Cleaned audio input (when denoise enabled) |
| `oe_tts` | ConversationModel (5561) | `conversation_response` | Token stream for sentence accumulation (Gemma only) |
| `oe_face_recog` | VideoIngest (5555) | `video_frame` | Every 3rd frame for face detection |
| `oe_audio_denoise` | AudioIngest (5556) | `audio_chunk` | Audio input for denoising |
| `oe_ws_bridge` | ConversationModel (5561) | `conversation_response`, `conversation_audio` | Tokens + audio for `/chat` and `/audio` |
| `oe_ws_bridge` | TTS Sidecar (5565) | `tts_audio` | PCM for `/audio` (Gemma only) |
| `oe_ws_bridge` | FaceRecog (5566) | `face_registered` | Face registration confirmations |
| `oe_daemon` | ConversationModel (5561) | `conversation_response` | Token streaming, state transitions |
| `oe_daemon` | FaceRecog (5566) | `identity`, `face_registered` | Identity context |

**Super Resolution Mode Subscriptions:**

| Subscriber Process | Subscribes To (Port) | Topics | Purpose |
|:---|:---|:---|:---|
| `oe_super_res` | VideoIngest (5555) | `video_frame` | Frame input for temporal enhancement |
| `oe_ws_bridge` | SuperRes (5568) | `enhanced_frame` | Enhanced frames for delivery |

**Image Transform Mode Subscriptions:**

| Subscriber Process | Subscribes To (Port) | Topics | Purpose |
|:---|:---|:---|:---|
| `oe_img_transform` | Daemon (5571) | `image_transform_prompt` | Image + style selection from daemon |
| `oe_ws_bridge` | ImgTransform (5572) | `transformed_image` | Generated image for `/chat` |

### 6.7 ZMQ Message Protocol

All messages carry JSON with `"v": 1`. Consumers ignore unknown fields. Breaking changes bump version; additive changes do not.

```json
{"v":1, "type":"video_frame", "shm":"/oe.vid.ingest", "seq":42, "w":1920, "h":1080, "ts_ns":183742956789012}
{"v":1, "type":"conversation_prompt", "system":"You are OmniEdge...", "user":"What do you see?", "identity":"Admin", "history":[]}
{"v":1, "type":"conversation_response", "token":"Hello", "finished":false}
{"v":1, "type":"ui_command", "action":"push_to_talk", "state":true}
{"v":1, "type":"ui_command", "action":"switch_mode", "mode":"super_resolution"}
{"v":1, "type":"ui_command", "action":"select_model", "model":"qwen2.5-omni-3b"}
{"v":1, "type":"ui_command", "action":"image_upload", "style":"anime", "image_path":"/tmp/oe_upload_001.jpg"}
{"v":1, "type":"module_status", "module":"conversation_model", "status":"running", "pid":12343, "model":"qwen2.5-omni-7b"}
{"v":1, "type":"module_ready", "module":"conversation_model", "pid":12344}
```

### 6.8 ZMQ Heartbeats

ZMTP PING/PONG in ZMQ I/O thread -- works even when the main thread is blocked in a CUDA kernel.

| Parameter | Value |
|:---|:---|
| `ZMQ_HEARTBEAT_IVL` | 1,000 ms |
| `ZMQ_HEARTBEAT_TIMEOUT` | 5,000 ms |
| `ZMQ_HEARTBEAT_TTL` | 10,000 ms |

---


## 7. GPU Memory Architecture and Tier System

### 7.1 VRAM Budget -- Mode-Based Allocation (12 GB)

Only one heavy inference mode active at a time, sharing the GPU with always-on modules.

**Always-On (resident across all modes):**

| Component | VRAM | Notes |
|:---|---:|:---|
| Background Blur (YOLO seg + ISP) | ~0.50 GB | YOLOv8n-seg FP16 TRT + OpenCV CUDA ISP + nvJPEG |
| Video Ingest | 0.00 GB | GStreamer, no GPU allocation |
| Audio Ingest + VAD | 0.00 GB | Silero VAD, CPU-only ONNX Runtime |
| WSL2 System Buffer (WDDM overhead) | ~1.50 GB | Empirically measured on RTX PRO 3000 |
| **Always-on total** | **~2.00 GB** | Leaves ~10.00 GB available for the active mode |

**Mode 1 -- Conversation (default):**

| Component | VRAM | Notes |
|:---|---:|:---|
| Qwen2.5-Omni-7B (4-bit GPTQ/AWQ) | TBD | Default. Unified STT+LLM+TTS. 4-bit quantization cuts VRAM by >50% vs. FP16. Designed to fit within 12 GB budget with on-demand weight loading and CPU offloading to manage peak allocation |
| *or* Qwen2.5-Omni-3B | TBD | Smaller variant. Fits comfortably at higher precision within 12 GB |
| *or* Gemma 4 E4B + TTS sidecar | TBD | Audio+vision input, text output. TTS sidecar spawned alongside |
| Face Recognition | ~0.50 GB | Identity injection into conversation prompt |
| Audio Denoise (optional) | ~0.05 GB | Toggle-based; cleanup before conversation model audio input |

**Mode 2 -- Super Resolution:**

| Component | VRAM | Notes |
|:---|---:|:---|
| BasicVSR++ | ~1.50 GB | Temporal video enhancement using video ingest SHM feed |

**Mode 3 -- Image Transform:**

| Component | VRAM | Notes |
|:---|---:|:---|
| Stable Diffusion + ControlNet/IP-Adapter | TBD | Structure-preserving style transfer. InstantStyle, IP-Adapter FaceID |

> **TBD values.** VRAM entries marked "TBD" are design targets not yet validated on hardware. The per-module skill's Model Definition section becomes authoritative once profiled on the RTX PRO 3000.

### 7.2 GPU Tier System

Auto-detected via `GpuProfiler::probe()` (NVML). Override: `--profile <tier>`. Tier definitions in `gpu_tier.hpp`.

| Tier | VRAM | Conversation Models | Modes |
|:---|:---|:---|:---|
| **Minimal** | 4 GB | 3B only | Conversation only |
| **Balanced** | 8 GB | 3B, Gemma 4 E4B | Conversation, Super Resolution |
| **Standard** | 12 GB | All three (7B default) | All three |
| **Ultra** | 16+ GB | All three + expanded context | All three + concurrent features |

### 7.3 Dynamic VRAM Management

Intra-mode VRAM pressure monitoring:

| Level | Condition | Action |
|:---|:---|:---|
| Normal | Free VRAM > 1,500 MB | No action |
| Warning | Free VRAM 500-1,500 MB | Stop accepting optional module spawns (audio denoise) |
| Critical | Free VRAM < 500 MB | Evict optional modules within current mode |
| Emergency | `cudaMalloc` fails | Conversation model downgrade (7B to 3B fallback) |

**Mode switch lifecycle:** `SIGTERM` → `waitpid` → verify VRAM freed (`GpuProfiler`) → `posix_spawn` → wait `module_ready`. Guarantees full reclamation before new allocation.

**Conversation model fallback:**

```yaml
conversation:
  default_model: "qwen2.5-omni-7b-awq"
  fallback_models:
    - model: "qwen2.5-omni-3b"
      min_vram_mb: 4000
```

On downgrade, module publishes `module_ready` with `"degraded": true`. Frontend shows warning banner. Upgrades back after 60 s of sustained free VRAM.

### 7.4 Mode Switch Lifecycle

1. User selects mode (e.g., `switch_mode: "super_resolution"`)
2. `ModeOrchestrator` identifies processes to kill
3. `SIGTERM` each process, `waitpid` (5 s grace, escalate to `SIGKILL`)
4. Verify VRAM freed via `GpuProfiler::probe()`
5. `posix_spawn` new mode's processes
6. Wait `module_ready` (30 s timeout per module)
7. Transition FSM to operational state

### 7.5 VRAM Pre-Flight Guard

`ensureVramAvailableMiB()` (`cuda_guard.hpp`) checks free VRAM before heavy inference. Skips gracefully on insufficient VRAM instead of hitting CUDA OOM.

Per-module headroom (`vram_thresholds.hpp`):

| Constant | MiB | What It Covers |
|:---|---:|:---|
| `kConversationInferenceHeadroomMiB` | 512 | KV cache growth + TRT workspace (unified model) |
| `kTtsSidecarInferenceHeadroomMiB` | 64 | ONNX Runtime intermediate tensors (Gemma sidecar) |
| `kBgBlurInferenceHeadroomMiB` | 128 | Segmentation TRT workspace |
| `kFaceRecogInferenceHeadroomMiB` | 128 | Detection workspace |
| `kSuperResInferenceHeadroomMiB` | 256 | Temporal-window activations (BasicVSR++) |
| `kImgTransformInferenceHeadroomMiB` | 512 | Diffusion workspace (SD + ControlNet) |

**TOCTOU caveat:** Not a hard guarantee. Inferencers must still handle `cudaErrorMemoryAllocation` gracefully.

---


## 8. Conversation Orchestration -- State Machine

### 8.1 State Diagram

```text
                    +-----------------------------+
                    |                             |
     PTT_PRESS      |                     +------v------+
   +----------------+        IDLE         |ERROR_RECOVERY|
   |                |                     +------+------+
   |                +-----+-----+---------+      |
   |                      |     |                | restart complete
   |            MODE_SWITCH|     |IMAGE_UPLOAD
   |                      v     v
   |        +-----------+ +------------------+
   |        |MODE_SWITCH| |  IMAGE_TRANSFORM |
   |        +-----------+ +------------------+
   |
   v
+----------+   PTT_RELEASE / VAD_SILENCE    +------------+
| LISTENING +------------------------------->| PROCESSING  |
+----------+                                +------+-----+
     ^                                             | Conversation model streaming
     |                                             v
     |  Barge-in                            +------------+
     |  (PTT during speak)                  |  SPEAKING   |
     |         +------------+               +------+-----+
     +---------+ INTERRUPTED |<--------------------+
               +------------+
```

### 8.2 States and Transitions

Implemented in `state_machine.hpp`. Up to 32 registered transitions, each with optional guard and action functions.

**8 States** (`enum class State : uint8_t`):

| State | Description | Active Modules |
|:---|:---|:---|
| **kIdle** | Waiting for user input | Always-on pipeline + current mode modules |
| **kListening** | Recording user speech via PTT | + AudioIngest, VAD |
| **kProcessing** | Conversation model inference | + Conversation model |
| **kSpeaking** | Conversation model streaming tokens + audio | + Conversation model (+ TTS sidecar for Gemma) |
| **kInterrupted** | Barge-in detected, canceling generation | Flushing conversation model + TTS |
| **kModeSwitching** | Killing previous mode, spawning new mode | SIGTERM/waitpid/posix_spawn cycle |
| **kImageTransform** | Stable Diffusion inference in progress | + Image Transform pipeline |
| **kErrorRecovery** | Module crashed, restarting | Watchdog active |

**18 Events** (`enum class Event : uint8_t`): `kPttPress`, `kPttRelease`, `kPttCancel`, `kTextInput`, `kVadSilence`, `kLlmFirstSentence`, `kLlmComplete`, `kLlmTimeout`, `kTtsComplete`, `kModuleCrash`, `kModuleRecovered`, `kRecoveryTimeout`, `kInterruptDone`, `kSwitchMode`, `kModelSelected`, `kImageUpload`, `kImageTransformComplete`, `kModeSwitchComplete`.

**19 UI Actions** (`enum class UiAction`): parsed via `constexpr parseUiAction()`, dispatched by `switch`. Raw string comparison is banned.

**Key transitions:**

| From | Event | To | Action |
|:---|:---|:---|:---|
| IDLE | PTT_PRESS | LISTENING | Start audio capture |
| IDLE | TEXT_INPUT | PROCESSING | Assemble prompt from text |
| IDLE | SWITCH_MODE | MODE_SWITCHING | Begin mode switch (kill-spawn cycle) |
| IDLE | IMAGE_UPLOAD (Image Transform mode) | IMAGE_TRANSFORM | Send image + style to SD pipeline |
| LISTENING | PTT_RELEASE or VAD_SILENCE | PROCESSING | Signal end-of-utterance |
| PROCESSING | LLM_FIRST_TOKEN | SPEAKING | Begin audio output (native or TTS sidecar) |
| PROCESSING | LLM_TIMEOUT (60 s) | ERROR_RECOVERY | Log error, retry |
| SPEAKING | TTS_COMPLETE | IDLE | All audio played |
| SPEAKING | PTT_PRESS (barge-in) | INTERRUPTED | Cancel conversation model + flush TTS |
| INTERRUPTED | CANCEL_CONFIRMED | LISTENING | Resume audio capture |
| MODE_SWITCHING | MODE_SWITCH_COMPLETE | IDLE | New mode operational |
| IMAGE_TRANSFORM | IMAGE_TRANSFORM_COMPLETE | IDLE | Result published to UI |
| Any | MODEL_SELECTED | MODE_SWITCHING | Kill current model, spawn selected model |
| Any | MODULE_CRASH | ERROR_RECOVERY | Restart module via watchdog |
| ERROR_RECOVERY | MODULE_RESTARTED | IDLE | Resume normal operation |

### 8.3 Prompt Assembly

`PromptAssembler` fuses real-time context into a 4,096-token budget:

| Component | Source | Budget |
|:---|:---|---:|
| System prompt | Static YAML | 200 tokens |
| Face identity | FaceRecognition ZMQ | 20 tokens |
| Visual context | Conversation model (multimodal, if video input active) | 80 tokens |
| Conversation history | Sliding window + summary eviction | 2,796 tokens |
| Current user utterance | STT or text input | 500 tokens |
| Generation headroom | -- | 500 tokens |
| **Total** | | **4,096 tokens** |

Budget set by `--max_seq_len 4096`. Oldest turns summarized and evicted; most recent 3 retained verbatim.

### 8.4 Barge-In

PTT during assistant speech triggers three-way interrupt (< 100 ms): `cancel_generation` → `flush_tts` → `stop_playback`. Transitions SPEAKING → INTERRUPTED → LISTENING.

---


## 9. Frontend and WebSocket Bridge

### 9.1 Architecture

Vanilla JS SPA served by `oe_ws_bridge` on port 9001. No npm, no bundler, no build step.

```text
frontend/
+-- index.html          # Layout: video canvas, controls, chat, status, mode selector
+-- style.css           # Responsive, dark mode
+-- js/
    +-- ws.js           # WebSocket channels, reconnect w/ exponential backoff
    +-- video.js        # JPEG -> canvas rendering, shape overlay
    +-- audio.js        # PCM -> AudioContext, sentence queue
    +-- chat.js         # Token streaming, conversation history
    +-- ptt.js          # Push-to-talk (mousedown/touchstart/Space)
    +-- controls.js     # Register face, audio denoise toggle
    +-- mode.js         # Mode selector (Conversation/Super Resolution/Image Transform)
    +-- model_select.js # Conversation model dropdown (Qwen2.5-Omni-7B/3B, Gemma 4 E4B)
    +-- tts_select.js   # TTS model selector (visible only when Gemma 4 E4B selected)
    +-- upload.js       # Image upload for Image Transform mode (style transfer)
    +-- super_res.js    # Super Resolution UI: frame/clip selection
    +-- status.js       # Module status indicators, degradation
    +-- image_adjust.js # Brightness/contrast/saturation/sharpness sliders
```

### 9.2 WebSocket Channels

Independent connections with exponential backoff reconnection (500 ms → 30 s cap).

| Channel | Data | Processing |
|:---|:---|:---|
| `/video` | Binary JPEG | `Blob` → `createImageBitmap` → `canvas.drawImage` at 30 fps (always-on) |
| `/audio` | Binary PCM F32 | `Float32Array` → `AudioBuffer` → `AudioBufferSourceNode` (Conversation mode) |
| `/chat` | JSON + binary | Commands, events, image upload, mode results; routed by `msg.type` |

### 9.3 Graceful Degradation

`module_status` events drive real-time UI state. Unavailable controls are disabled; lower tiers hide unsupported features entirely.

| Module Down | Frontend Behavior | User Experience |
|:---|:---|:---|
| Background Blur | Raw video passthrough | No blur, still functional |
| Face Recognition | Conversation prompt says "identity: unknown" | Assistant stops using name |
| Conversation Model | "Assistant restarting..." banner | Core path blocked until restart or model switch |
| TTS Sidecar (Gemma only) | Text-only responses | Read instead of listen; switch to Qwen Omni for native audio |
| Audio Denoise | Disable audio denoise toggle | Enhancement unavailable |
| Super Resolution (BasicVSR++) | Disable Super Resolution mode | Enhancement unavailable |
| Image Transform (SD) | Disable Image Transform mode | Style transfer unavailable |
| Video Ingest | Static placeholder image | Audio + text still work |

---


## 10. Daemon, Watchdog and Fault Tolerance

### 10.1 Daemon Responsibilities

`oe_daemon` is the sole user-started binary (`modules/nodes/orchestrator/`). Init sequence:

1. Parse `config/omniedge_config.yaml`
2. GPU probe → `selectTier()`
3. Register modules with `VramTracker`
4. Initialize `VramGate` with eviction callback
5. Create `PromptAssembler`, `ModeOrchestrator`, `SessionPersistence`
6. Create `MessageRouter` + ZMQ subscriptions
7. Spawn modules in dependency order (`posix_spawn`, WS Bridge first)
8. Wait `module_ready` (30 s timeout, graceful degradation)
9. Register FSM transitions (up to 32)
10. Enter blocking `zmq_poll()` loop

Key components:

- **`ModuleLauncher`**: `posix_spawn` / `SIGTERM` (5 s grace → `SIGKILL`) / `restartModule`. Tracks pid, restartCount, maxRestarts, evictPriority, vramBudgetMb.
- **`ModeOrchestrator`**: Three mutually exclusive modes with VRAM-aware kill-spawn cycles. Handles intra-mode model switching.
- **`SessionPersistence`**: Atomic JSON save/load of conversation state. 30 s periodic save. Rejects stale sessions > 60 s.

### 10.2 Launch Order

**Always-on modules (launched at boot, every session):**

```text
1. oe_ws_bridge       -- Must be first (relays module_ready events + UI commands)
2. oe_video_ingest    -- Producer: /oe.vid.ingest
3. oe_audio_ingest    -- Producer: /oe.aud.ingest
4. oe_bg_blur         -- Consumer of video, producer of JPEG (YOLO seg + ISP)
```

**Conversation mode (default, launched after always-on modules):**

```text
5. oe_conversation    -- Qwen2.5-Omni-7B (default) or user-selected model
6. oe_face_recog      -- Identity injection into conversation prompt
7. oe_audio_denoise   -- Optional (user toggle)
8. oe_tts             -- Only if Gemma 4 E4B selected
```

**Super Resolution mode (on mode switch):**

```text
5. oe_super_res       -- BasicVSR++ temporal video enhancement
```

**Image Transform mode (on mode switch):**

```text
5. oe_img_transform   -- Stable Diffusion + ControlNet / IP-Adapter
```

Each module publishes `module_ready` after startup. Modules failing to report within 30 s are logged as warnings (graceful degradation).

### 10.3 Watchdog and Auto-Recovery

ZMTP heartbeats (Section 6.8) detect hung/crashed processes. Recovery sequence:

| Step | Action | Timeout |
|:---|:---|:---|
| 1 | `SIGTERM` | -- |
| 2 | `waitpid(WNOHANG)` polling | 2 s |
| 3 | `SIGKILL` if still alive | immediate |
| 4 | `waitpid(0)` reap zombie | 1 s |
| 5 | Check `restartCount < max_restarts` | -- |
| 6 | `posix_spawn` fresh instance | -- |
| 7 | Wait `module_ready` | 30 s |

CUDA kernels are not signal-safe. `SIGTERM` sets `atomic<bool> shutdown_requested`; if the kernel completes within 2 s the module exits gracefully, otherwise `SIGKILL` forces termination and the CUDA driver handles cleanup.

### 10.4 Development Profiles

**GPU Inference Modes (production):**

```yaml
modes:
  conversation:       # Default -- unified conversation model + always-on pipeline
    always_on: [websocket_bridge, video_ingest, audio_ingest, background_blur]
    mode_modules: [conversation_model, face_recognition]
    optional: [audio_denoise, tts]  # tts only for Gemma 4 E4B
    models:
      - qwen2.5-omni-7b-awq    # Default
      - qwen2.5-omni-3b
      - gemma-4-e4b
  super_resolution:   # Temporal video enhancement
    always_on: [websocket_bridge, video_ingest, audio_ingest, background_blur]
    mode_modules: [super_res]
  image_transform:    # Style transfer via Stable Diffusion
    always_on: [websocket_bridge, video_ingest, audio_ingest, background_blur]
    mode_modules: [img_transform]
```

**Development profiles (debugging):**

```yaml
profiles:
  dev-conversation:  # Conversation mode only (no video pipeline)
    modules: [websocket_bridge, audio_ingest, conversation_model]
  dev-video:         # Video pipeline only (no conversation)
    modules: [websocket_bridge, video_ingest, background_blur]
```

Usage: `oe_daemon --mode conversation` or `oe_daemon --profile dev-conversation`

---


## 11. Configuration and Build System

### 11.1 CMake Super-Project

Root `CMakeLists.txt`: C++20 (host + CUDA), system dependencies, each module as a subdirectory producing standalone executables.

```text
OmniEdge_AI/
+-- CMakeLists.txt                  # Root: C++20, CUDA 12.x, find_package()
+-- config/
|   +-- omniedge_config.yaml        # Runtime configuration (ports, paths, thresholds)
+-- modules/
|   +-- common/                     # Logger + shared utility headers (omniedge_logger)
|   |   +-- cmake/                  # OmniEdgeHelpers.cmake
|   |   +-- include/common/         # expected_result, oe_logger, platform_detect, runtime_defaults, constants/
|   |   +-- src/                    # oe_logger.cpp
|   +-- interfaces/                 # Pure virtual inferencer headers (omniedge_interfaces)
|   |   +-- include/                # conversation_inferencer.hpp, tts_inferencer.hpp, blur_inferencer.hpp, super_res_inferencer.hpp, img_transform_inferencer.hpp, ...
|   +-- core/                       # Pure algorithm implementations -- no ZMQ, no node wiring
|   |   +-- conversation/           # QwenOmniInferencer, GemmaE4BInferencer
|   |   +-- tts/                    # OnnxKokoroInferencer, SentenceSplitter (TTS sidecar for Gemma)
|   |   +-- cv/                     # YoloSegEngine, InspireFace, FaceGallery, BasicVSR++
|   |   +-- img_transform/          # SDControlNetInferencer, IP-Adapter
|   |   +-- audio_denoise/          # OnnxDtlnInferencer
|   |   +-- ingest/                 # GStreamerPipeline, SileroVad
|   |   +-- statemachine/           # StateMachine, PromptAssembler, SessionPersistence
|   +-- nodes/                      # Wiring: core + interfaces + transport -> process binaries
|   |   +-- conversation/           # omniedge_conversation (ConversationNode + main)
|   |   +-- tts/                    # omniedge_tts (TTS sidecar for Gemma 4 E4B)
|   |   +-- cv/                     # omniedge_bg_blur, omniedge_face_recog
|   |   +-- super_res/              # omniedge_super_res (BasicVSR++ temporal enhancement)
|   |   +-- img_transform/          # omniedge_img_transform (Stable Diffusion + ControlNet)
|   |   +-- audio_denoise/          # omniedge_audio_denoise
|   |   +-- ingest/                 # omniedge_video_ingest, omniedge_audio_ingest
|   |   +-- ws_bridge/              # omniedge_ws_bridge (WebSocketBridgeNode)
|   |   +-- orchestrator/           # omniedge_daemon (OmniEdgeDaemon, ModuleLauncher, ModeOrchestrator)
|   +-- transport/                  # ZMQ, shared memory, WebSocket server
|   |   +-- zmq/                    # MessageRouter, ZmqPublisher, ZmqSubscriber
|   |   +-- shm/                    # ShmCircularBuffer, ShmMapping, ShmFrameReader
|   |   +-- ws/                     # WsServer (uWebSockets)
|   +-- hal/                        # CUDA, GPU, VRAM
|       +-- gpu/                    # CudaStream, GpuTier, GpuProfiler, GpuCompositor, PinnedBuffer
|       +-- vram/                   # VramTracker, VramGate, VramThresholds
+-- tests/                          # Google Test -- layered test pyramid
|   +-- core/                       # Pure algorithm unit tests (link: core + interfaces only)
|   +-- interfaces/                 # Inferencer contract tests (link: mocks + interfaces)
|   +-- nodes/                      # Node wiring tests (link: mocks + node _common libs)
|   +-- mocks/                      # Shared mock inferencer headers
|   +-- integration/                # Real ZMQ, real GPU, real inferencer tests
|   |   +-- transport/              # SHM + ZMQ integration
|   |   +-- hal/                    # GPU, VRAM, pinned memory, compositor
|   |   +-- inferencers/               # Real TRT/ONNX inferencer tests (GPU-gated)
|   |   +-- pipeline/              # Cross-module pipeline tests
|   +-- fixtures/                   # Shared test data (PCM, BGR, images)
|   +-- e2e/                        # End-to-end Python tests
+-- frontend/                       # Vanilla JS SPA (served by ws_bridge on :9001)
+-- models/                         # Git-ignored; converted TRT engines and ONNX models
+-- logs/                           # Runtime log directory (omniedge.log)
+-- run_all.sh                      # Entry point: build, install, verify, status, stop, model-test
+-- session_state.json              # Persisted conversation state (atomic save/load)
```

**Layered dependency graph:**

```text
omniedge_common (INTERFACE target)
  +-- omniedge_logger        (spdlog, tl::expected)
  +-- omniedge_interfaces    (header-only inferencer contracts)
  +-- omniedge_hal_gpu       (CUDA::cudart)
  +-- omniedge_hal_vram      (hal/gpu, tl::expected)
  +-- omniedge_transport_zmq (nlohmann_json, cppzmq, rt)
  +-- omniedge_transport_shm (rt)
  +-- omniedge_core_statemachine (nlohmann_json, transport/zmq)

core libraries (link: omniedge_common + inference SDKs)
  +-- omniedge_core_conversation, omniedge_core_tts, omniedge_core_cv, omniedge_core_img_transform, ...

node binaries (link: core + omniedge_common)
  +-- omniedge_conversation, omniedge_tts, omniedge_bg_blur, omniedge_super_res, omniedge_img_transform, ...
```

`omniedge_common` INTERFACE target aggregates all infrastructure: logger, interfaces, GPU HAL, VRAM, ZMQ, SHM, state machine.

### 11.2 Key Dependencies

| Dependency | Purpose |
|:---|:---|
| CUDA Toolkit 12.x | GPU compute, `cudaHostAlloc`, `cudaMemcpyAsync` |
| TensorRT 10.x | Conversation model, CV, Image Transform engine compilation and inference |
| TensorRT-LLM | Conversation model C++ runtime (paged KV cache) |
| ZeroMQ (libzmq + cppzmq) | Control-plane PUB/SUB messaging |
| GStreamer 1.0 | Video and audio capture pipelines |
| OpenCV (with CUDA modules) | GPU ISP processing, drawing primitives |
| ONNX Runtime (CUDA EP) | TTS sidecar (Gemma), VAD, Audio Denoise, Super Resolution (BasicVSR++) |
| uWebSockets | WebSocket server, three-channel binary/JSON relay |
| nlohmann/json | JSON parsing across all modules |
| yaml-cpp | YAML config parsing (daemon) |
| spdlog | Structured logging |
| espeak-ng | G2P phoneme conversion for TTS |
| InspireFace SDK | Face detection and recognition |
| tl::expected | C++23 `std::expected` polyfill (via FetchContent) |
| Google Test | Unit and integration testing |
| Tracy (optional) | Performance profiling, enabled via `cmake -DOE_ENABLE_TRACY=ON` |

### 11.3 `run_all.sh` -- Single Entry Point

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


## 12. Observability and Testing

### 12.1 Structured Logging

`OeLogger` singleton → `logs/omniedge.log`. Format: `YYYY-MM-DD HH:MM:SS.mmm [LEVEL] [module] [function] message`. Macros: `OE_LOG_DEBUG/INFO/WARN/ERROR` with `std::format`.

### 12.2 Tracy Profiler

Optional: `cmake -DOE_ENABLE_TRACY=ON`. Zero-overhead no-ops when disabled. Key macros: `OE_ZONE_SCOPED` (every hot-path function), `OE_FRAME_MARK` (poll loop boundaries), `OE_PLOT` (live metrics). Banned inside per-sample/per-pixel loops (~5 ns overhead compounds). Tracy GUI connects via TCP port 8086.

### 12.3 `/status` HTTP Endpoint

`GET /status` on port 9001: module PIDs, restart counts, latencies, VRAM usage, WebSocket client count.

### 12.4 End-to-End Latency Budget (Reference Targets)

| Stage | Qwen Omni | Gemma + TTS |
|:---|:---|:---|
| Audio encoding + prefill | < 500 ms | < 500 ms |
| First output (audio / text) | < 300 ms | < 400 ms |
| First playable audio | < 400 ms | < 100 ms (sidecar) |
| **PTT release → first audio** | **< 1,300 ms** | **< 1,100 ms** |

With 800 ms VAD threshold, total user-perceived latency: **< 2.1 s**.

### 12.5 Testing Strategy

| Layer | Directory | Links | Purpose |
|:---|:---|:---|:---|
| **Core** | `tests/core/` | core + interfaces | Pure algorithms, no transport/GPU |
| **Interface** | `tests/interfaces/` | mocks + interfaces | Contract tests |
| **Node** | `tests/nodes/` | mocks + node libs | Wiring verification |
| **Integration** | `tests/integration/` | everything real | Real ZMQ, GPU, inference inferencers |

GPU tests tagged `LABELS gpu`. Run: `ctest -LE gpu` (CPU-safe) or `ctest -L gpu` (GPU).

### 12.6 Profiling Tools

| Tool | Command | Purpose |
|:---|:---|:---|
| **Nsight Systems** | `nsys profile -t cuda,nvtx ./oe_conversation` | Full CPU+GPU timeline: kernel gaps, memcpy overlap, ZMQ stalls |
| **NVML monitoring** | `nvidia-smi dmon -s pucvmet -d 1` | Real-time GPU health: SM utilization, VRAM, temperature |
| **VRAM fragmentation** | `cudaMemGetInfo` + 512 MB contiguous alloc test | Detect fragmentation after repeated load/unload cycles |
| **ZMQ throughput** | `zmq_socket_monitor` + counters | Message delivery rate, HWM drops, reconnect count |

---

> **Document version:** 5.0
> **Target:** NVIDIA RTX PRO 3000 Blackwell (SM 10.0+) -- 12 GB GDDR6 -- WSL2
> **Architecture:** Three mutually exclusive GPU inference modes + always-on video/audio/blur pipeline
> **Verified:** Frontend (JS SPA), IPC layer (ZMQ/SHM)
> **Experimental:** All inference inferencers -- VRAM sizes and latency targets are reference configurations, not proven benchmarks
