# ONNX Export & Quantization Scripts — Engineering Rationale

> **Note:** Export scripts live in `scripts/export/`, quantization scripts
> in `scripts/quantize/`. These were consolidated from a separate `onnx/`
> directory to reduce confusion.

> **Audience:** OmniEdge_AI developers who need to understand *why* these
> scripts exist and *why* specific engineering decisions were made.

---

## 1. Why ONNX at All?

OmniEdge_AI is a C++ daemon.  AI models (e.g., Kokoro TTS) originate as
PyTorch/HuggingFace checkpoints — they cannot be loaded directly into a
C++ process.  We need a **serialized, framework-agnostic intermediate
representation** that C++ runtimes can consume.

Two viable options exist on NVIDIA hardware:

| Format      | Runtime              | Strength                         | Weakness                              |
|-------------|----------------------|----------------------------------|---------------------------------------|
| **ONNX**    | ONNX Runtime (ORT)   | Portable, easy export, fast dev  | Not as optimized as TRT for pure GPU  |
| **TensorRT**| TensorRT / trtexec   | Maximum GPU throughput           | GPU-specific engine, longer build     |

**Our strategy is a two-tier pipeline:**

```
PyTorch model  ──►  ONNX (.onnx)  ──►  TensorRT engine (.trt)
                         │                      │
                    Used by ORT            Used by TRT API
                   (fallback / dev)       (production / perf)
```

- **ONNX is the canonical interchange format.** Every model must export to
  ONNX first, even if the production path uses TensorRT.
- **TensorRT engines are built *from* the ONNX** via `trtexec --onnx=...
  --saveEngine=... --fp16`.  TRT engines are GPU-architecture-specific
  (an engine built on Ada Lovelace won't load on Blackwell) so they are
  never checked in — they are built on-device at install time.
- **ONNX Runtime serves as the fallback** when a TRT engine hasn't been
  built yet, or for modules where TRT gains are marginal (e.g., Kokoro TTS
  at 82 M params).

---

## 2. File-by-File Breakdown

### 2.1 `quantize_kokoro_onnx.py` — Kokoro TTS INT8 Quantization

**What it does:** Takes the Kokoro v1.0 TTS model (`model.onnx`, 82 M params)
and applies **dynamic INT8 quantization** to produce `model_int8.onnx`.

**Why dynamic quantization (not static)?**

| Type | Calibration needed? | Quality | Latency |
|------|---------------------|---------|---------|
| Dynamic INT8 | No | Good | Slightly slower (scales computed at runtime) |
| Static INT8 | Yes (needs representative dataset) | Better | Faster |

Dynamic quantization requires **zero calibration data** — just point it at the
ONNX file and it produces INT8 weights.  For Kokoro TTS where latency is
dominated by vocoder waveform generation (not matrix multiplies), the marginal
speed difference doesn't matter.  Static quantization would require recording
representative mel-spectrogram inputs, adding pipeline complexity for minimal
benefit.

**Key decisions:**

| Decision | Why |
|----------|-----|
| `QuantType.QInt8` (not QUInt8) | Signed INT8 is the standard for weights.  QUInt8 is typically used for activations in static quantization. |
| Post-export validation with ORT session | Catches export corruption early — a broken quantized model would silently produce garbage audio at runtime. |
| `CUDAExecutionProvider` first in provider list | ORT tries providers in order.  CUDA EP first ensures GPU inference; falls back to CPU if no GPU. |
| Skip if `model_int8.onnx` exists | Idempotency — safe to re-run without overwriting existing outputs. |

---

## 3. How the C++ Side Consumes These

```
                    ┌─────────────────────────────────────────────┐
                    │  OmniEdgeDaemon (C++ process)               │
                    │                                             │
  Text ───────────► │  model_int8.onnx     (ONNX Runtime CUDA EP) │
                    │         │                                   │
                    │         ▼  audio waveform                   │
                    └─────────────────────────────────────────────┘
```

- `onnx_kokoro_inferencer.cpp` loads `model_int8.onnx` via ONNX Runtime.
- Uses CUDA Execution Provider for GPU inference within the C++ daemon.

---

## 4. Running the Scripts

```bash
# Quantize Kokoro TTS → INT8
python3 scripts/quantize/quantize_kokoro_onnx.py
```

All scripts respect `OE_MODELS_DIR` (default: `~/omniedge_models`) and are
idempotent — safe to re-run without overwriting existing outputs.
