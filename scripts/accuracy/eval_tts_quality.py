#!/usr/bin/env python3
"""OmniEdge_AI -- TTS Quality Benchmark (Kokoro ONNX)

Synthesises speech from text using Kokoro ONNX, then re-transcribes
with Whisper and measures Character Error Rate (CER) as a proxy for
intelligibility.

Metrics:
  - Re-transcription CER: Whisper transcription of TTS output vs original text
    (target: CER <= 40% -- Kokoro + espeak-ng G2P has imperfect phonemisation)
  - Audio quality: RMS energy > 0 (not silence), no NaN/Inf, duration reasonable

Requirements:
  - Kokoro ONNX model in OE_MODELS_DIR/kokoro-onnx/onnx/
  - Kokoro voices in OE_MODELS_DIR/kokoro/voices/
  - onnxruntime, numpy
  - espeak-ng (for G2P)
  - Whisper model (for re-transcription CER)

Exit codes: 0 = all thresholds met, 1 = below threshold, 77 = skipped
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Callable, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    check_espeak_ng,
    compute_cer,
    load_whisper_transcriber,
    resolve_model_path,
    setup_logging,
    write_wav,
)

logger = setup_logging(__name__)

# -- Thresholds -------------------------------------------------------------
CER_THRESHOLD: float = 0.40
MIN_RMS_ENERGY: float = 0.001
MIN_DURATION_S: float = 0.1
MAX_DURATION_S: float = 30.0
SAMPLE_RATE: int = 24000
QUALITY_PASS_RATE_THRESHOLD: float = 0.80

# -- Test data (shared with benchmark_all.py) --------------------------------
TEST_SENTENCES: List[str] = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "One two three four five.",
    "Good morning, how are you today?",
    "The weather is nice and sunny.",
]


def _locate_kokoro_model() -> Optional[str]:
    """Return the path to the best available Kokoro ONNX model, or None."""
    candidates = [
        resolve_model_path("kokoro-onnx/onnx/model_int8.onnx"),
        resolve_model_path("kokoro/onnx/model_quantized.onnx"),
        resolve_model_path("kokoro/kokoro-v1_0-int8.onnx"),
        resolve_model_path("kokoro-onnx/onnx/kokoro-v1_0.onnx"),
        resolve_model_path("kokoro-onnx/onnx/model.onnx"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _locate_voice_dir():
    """Return (npy_dir, pt_dir) -- at least one non-None if voices found."""
    voice_dirs = [
        resolve_model_path("kokoro/voices"),
        resolve_model_path("kokoro-npy/voices"),
        resolve_model_path("kokoro-onnx/voices"),
    ]
    npy_dir = pt_dir = None
    for vd in voice_dirs:
        if os.path.isdir(vd):
            if any(f.endswith(".npy") for f in os.listdir(vd)):
                npy_dir = vd
                break
            if any(f.endswith(".pt") or f.endswith(".bin") for f in os.listdir(vd)):
                pt_dir = vd
    return npy_dir, pt_dir


def _load_voice_style(npy_dir, pt_dir):
    """Load a voice style array, return numpy array or None."""
    import numpy as np

    if npy_dir:
        vp = os.path.join(npy_dir, "af_heart.npy")
        if not os.path.isfile(vp):
            npy_files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]
            vp = os.path.join(npy_dir, npy_files[0]) if npy_files else None
        if vp and os.path.isfile(vp):
            return np.load(vp)

    if pt_dir:
        try:
            import torch
            pt_files = sorted(f for f in os.listdir(pt_dir) if f.endswith(".pt"))
            preferred = [f for f in pt_files if "af_heart" in f]
            pt_file = preferred[0] if preferred else (pt_files[0] if pt_files else None)
            if pt_file:
                tensor = torch.load(
                    os.path.join(pt_dir, pt_file),
                    map_location="cpu", weights_only=True,
                )
                if hasattr(tensor, "numpy"):
                    return tensor.numpy().astype(np.float32)
                return np.array(tensor, dtype=np.float32)
        except Exception as e:
            logger.warning("Could not load .pt voice: %s", e)

    return None


def _build_g2p_tokenizer(model_path: str):
    """Return a text_to_phoneme_ids function."""
    import json as _json

    phoneme_vocab: dict = {}
    for tok_path in [
        resolve_model_path("kokoro-onnx/tokenizer.json"),
        resolve_model_path("kokoro/tokenizer.json"),
        os.path.join(os.path.dirname(model_path), "..", "tokenizer.json"),
    ]:
        if os.path.isfile(tok_path):
            with open(tok_path) as f:
                phoneme_vocab = _json.load(f).get("model", {}).get("vocab", {})
            if phoneme_vocab:
                break

    misaki_g2p = None
    try:
        os.environ.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")
        from misaki.en import G2P as _G2P
        misaki_g2p = _G2P()
    except ImportError:
        pass

    def text_to_ids(text: str) -> List[int]:
        ps = ""
        if misaki_g2p:
            try:
                ps, _ = misaki_g2p(text)
            except Exception:
                ps = ""
        if not ps:
            r = subprocess.run(
                ["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                capture_output=True, text=True, timeout=10,
            )
            ps = r.stdout.strip()
        ids = [0]
        if phoneme_vocab:
            ids.extend(phoneme_vocab[ch] for ch in ps if ch in phoneme_vocab)
        else:
            ids.extend(min(ord(ch), 177) for ch in ps)
        ids.append(0)
        return ids

    return text_to_ids


def benchmark(warmup: int = 0):
    """Run the TTS quality benchmark and return a results dict."""
    result = {
        "status": "pending",
        "metrics": {},
        "latencies_ms": [],
        "detail_lines": [],
        "error": "",
        "thresholds": {"cer": CER_THRESHOLD, "quality_rate": QUALITY_PASS_RATE_THRESHOLD},
        "model_path": None,
    }

    model_path = _locate_kokoro_model()
    if not model_path:
        result["status"] = "skip"
        result["error"] = "No Kokoro ONNX model found"
        return result
    result["model_path"] = model_path

    npy_dir, pt_dir = _locate_voice_dir()
    if npy_dir is None and pt_dir is None:
        result["status"] = "skip"
        result["error"] = "No voice directory found"
        return result

    if not check_espeak_ng():
        result["status"] = "skip"
        result["error"] = "espeak-ng not available"
        return result

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        result["status"] = "skip"
        result["error"] = f"Missing dependency: {e}"
        return result

    # Load model
    eps = ort.get_available_providers()
    providers: List[str] = []
    if "CUDAExecutionProvider" in eps:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        result["status"] = "fail"
        result["error"] = f"Model load failed: {e}"
        return result

    # Load voice
    voice_style = _load_voice_style(npy_dir, pt_dir)
    if voice_style is None:
        result["status"] = "skip"
        result["error"] = "No usable voice files found"
        return result

    # Build G2P + synthesizer
    text_to_ids = _build_g2p_tokenizer(model_path)
    input_names = [i.name for i in session.get_inputs()]

    def synthesize(text: str) -> List[float]:
        tokens = text_to_ids(text)
        input_ids = np.array([tokens], dtype=np.int64)
        style = voice_style.reshape(1, -1).astype(np.float32)
        speed = np.array([1.0], dtype=np.float32)
        feeds: dict = {}
        for name in input_names:
            nl = name.lower()
            if "token" in nl or "input" in nl:
                feeds[name] = input_ids
            elif "style" in nl or "voice" in nl:
                feeds[name] = style
            elif "speed" in nl:
                feeds[name] = speed
        if len(feeds) != len(input_names) and len(input_names) >= 3:
            feeds = {input_names[0]: input_ids, input_names[1]: style,
                     input_names[2]: speed}
        return session.run(None, feeds)[0].flatten().tolist()

    # Warmup
    for _ in range(warmup):
        try:
            synthesize(TEST_SENTENCES[0])
        except Exception:
            pass

    # Load Whisper for CER
    whisper_result = load_whisper_transcriber()
    whisper_transcribe: Optional[Callable[[str], str]] = whisper_result[1] if whisper_result else None

    # Run benchmark
    pass_count = 0
    cer_scores: List[float] = []

    for idx, sentence in enumerate(TEST_SENTENCES):
        t0 = time.monotonic()
        try:
            audio = synthesize(sentence)
        except Exception as e:
            result["detail_lines"].append(f"  S{idx+1}: synthesis error: {e}")
            continue
        elapsed = (time.monotonic() - t0) * 1000
        result["latencies_ms"].append(elapsed)

        n = len(audio)
        dur = n / SAMPLE_RATE
        has_nan = any(math.isnan(s) for s in audio)
        has_inf = any(math.isinf(s) for s in audio)
        rms = math.sqrt(sum(s * s for s in audio) / max(1, n))

        quality_ok = (not has_nan and not has_inf
                      and rms > MIN_RMS_ENERGY
                      and MIN_DURATION_S < dur < MAX_DURATION_S)

        cer_val: Optional[float] = None
        if whisper_transcribe and quality_ok:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wp = tmp.name
            try:
                write_wav(wp, audio, SAMPLE_RATE)
                retrans = whisper_transcribe(wp)
                cer_val = compute_cer(sentence, retrans)
                cer_scores.append(cer_val)
            except Exception:
                pass
            finally:
                os.unlink(wp)

        if quality_ok:
            pass_count += 1

        cer_str = f"CER={cer_val:.1%}" if cer_val is not None else "CER=N/A"
        tag = "OK" if quality_ok else "QFAIL"
        result["detail_lines"].append(
            f"  S{idx+1}: [{tag}] {cer_str}  dur={dur:.2f}s  rms={rms:.4f}  ({elapsed:.0f}ms)"
        )

    total = len(TEST_SENTENCES)
    quality_rate = pass_count / total if total > 0 else 0
    mean_cer = sum(cer_scores) / len(cer_scores) if cer_scores else float("nan")

    result["metrics"] = {
        "quality_pass_rate": round(quality_rate, 4),
        "mean_cer": round(mean_cer, 4) if not math.isnan(mean_cer) else None,
        "cer_samples": len(cer_scores),
        "pass_count": pass_count,
        "total_count": total,
    }

    passed = quality_rate >= QUALITY_PASS_RATE_THRESHOLD
    if cer_scores and mean_cer > CER_THRESHOLD:
        passed = False
    result["status"] = "pass" if passed else "fail"
    return result


def main() -> int:
    """Run the TTS quality benchmark and return a shell exit code."""
    print("\033[1mTTS Quality Benchmark (Kokoro ONNX)\033[0m\n")

    r = benchmark()

    if r["status"] == "skip":
        print(f"SKIP: {r['error']}")
        return 77

    if r["model_path"]:
        print(f"  Model: {r['model_path']}\n")

    for line in r["detail_lines"]:
        is_ok = "[OK]" in line
        status = "\033[32mPASS\033[0m" if is_ok else "\033[31mFAIL\033[0m"
        print(f"  {status} {line}")

    m = r["metrics"]
    mean_cer = m.get("mean_cer")
    mean_latency = sum(r["latencies_ms"]) / len(r["latencies_ms"]) if r["latencies_ms"] else 0

    print(f"\n{'=' * 60}")
    print(f"  Quality pass rate:   {m['pass_count']}/{m['total_count']} = {m['quality_pass_rate']:.1%}")
    if mean_cer is not None:
        print(f"  Mean CER:            {mean_cer:.1%}  (on {m['cer_samples']} sentences)")
        print(f"  CER threshold:       {CER_THRESHOLD:.0%}")
    else:
        print("  Mean CER:            N/A (Whisper not available)")
    print(f"  Mean latency:        {mean_latency:.0f} ms/sentence")
    print(f"{'=' * 60}")

    if r["status"] == "pass":
        qr = m["quality_pass_rate"]
        print(f"  \033[32mPASS\033[0m  Quality pass rate {qr:.1%} >= {QUALITY_PASS_RATE_THRESHOLD:.0%}")
        if mean_cer is not None:
            print(f"  \033[32mPASS\033[0m  Mean CER {mean_cer:.1%} <= {CER_THRESHOLD:.0%}")
        return 0
    else:
        print(f"  \033[31mFAIL\033[0m  Below thresholds")
        return 1


if __name__ == "__main__":
    sys.exit(main())
