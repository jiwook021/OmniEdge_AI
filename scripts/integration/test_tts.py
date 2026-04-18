#!/usr/bin/env python3
"""OmniEdge_AI -- TTS Integration Test

Tests Kokoro ONNX inferencer TTS with real model inference.
Generates speech from text, verifies PCM output is valid audio.

Requirements:
  - Kokoro ONNX model in OE_MODELS_DIR/kokoro-onnx/onnx/
  - Kokoro voices in OE_MODELS_DIR/kokoro/voices/
  - onnxruntime (with CUDA EP preferred)
  - espeak-ng (for G2P phonemisation)
  - numpy

Exit codes: 0 = passed, 1 = failed, 77 = skipped
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import tempfile
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import resolve_model_path, setup_logging, TestReporter

logger = setup_logging(__name__)

# -- Constants --------------------------------------------------------------
SAMPLE_RATE: int = 24000   # Kokoro outputs at 24 kHz
SUBPROCESS_TIMEOUT_S: int = 30


def rms(samples: List[float]) -> float:
    """Root mean square of float samples.

    Args:
        samples: Float audio samples.

    Returns:
        RMS value.
    """
    if len(samples) == 0:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def write_wav(path: str, samples: List[float], rate: int = SAMPLE_RATE) -> None:
    """Write float32 samples as 16-bit PCM WAV.

    Args:
        path: Output file path.
        samples: Audio samples in [-1.0, 1.0].
        rate: Sample rate in Hz.
    """
    data = b""
    for s in samples:
        val = int(max(-32768, min(32767, s * 32767)))
        data += struct.pack("<h", val)

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def main() -> int:
    """Run the TTS integration test and return a shell exit code."""
    print("\033[1mTTS Integration Test (Kokoro ONNX)\033[0m\n")

    reporter = TestReporter("TTS")

    # --- Locate model ---
    onnx_dir: str = resolve_model_path("kokoro-onnx/onnx")
    voice_dir: str = resolve_model_path("kokoro/voices")

    # Fallback: kokoro-npy/voices/ contains .npy files when kokoro/voices/ has
    # only .bin/.pt voice packs (the ONNX inferencer needs .npy format).
    if os.path.isdir(voice_dir):
        has_npy = any(f.endswith(".npy") for f in os.listdir(voice_dir))
        if not has_npy:
            alt_dir = resolve_model_path("kokoro-npy/voices")
            if os.path.isdir(alt_dir) and any(
                f.endswith(".npy") for f in os.listdir(alt_dir)
            ):
                voice_dir = alt_dir
    elif not os.path.isdir(voice_dir):
        alt_dir = resolve_model_path("kokoro-npy/voices")
        if os.path.isdir(alt_dir):
            voice_dir = alt_dir

    model_path: Optional[str] = None
    for name in ["model_int8.onnx", "kokoro-v1_0.onnx", "model.onnx"]:
        candidate = os.path.join(onnx_dir, name)
        if os.path.isfile(candidate):
            model_path = candidate
            break

    if not model_path:
        print("SKIP: No Kokoro ONNX model found in", onnx_dir)
        return 77

    if not os.path.isdir(voice_dir):
        print("SKIP: Voice directory not found:", voice_dir)
        return 77

    # --- Import deps ---
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"SKIP: Missing dependency -- {e}")
        return 77

    # --- Check espeak-ng ---
    try:
        subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True,
            check=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        has_espeak: bool = True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        has_espeak = False

    if not has_espeak:
        print("SKIP: espeak-ng not available (required for G2P)")
        return 77

    # --- Load ONNX model ---
    logger.info("Loading model: %s", model_path)
    print(f"  Loading model: {model_path}")
    try:
        providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        active_ep: str = session.get_providers()[0]
        reporter.pass_test("Model loaded", f"EP: {active_ep}")
    except Exception as e:
        reporter.fail_test("Model load", str(e))
        reporter.print_summary()
        return 1

    # --- Load voice style ---
    # Kokoro voices may be stored as .npy (NumPy), .bin (raw float32), or
    # .pt (PyTorch) files depending on the download source.
    voice_name: str = "af_heart"
    voice_style: Optional[np.ndarray] = None
    VOICE_DIM: int = 256  # Kokoro style embedding dimension

    for ext in (".npy", ".bin", ".pt"):
        candidate = os.path.join(voice_dir, f"{voice_name}{ext}")
        if os.path.isfile(candidate):
            voice_path = candidate
            break
    else:
        # Fallback: pick the first available voice in any supported format
        for ext in (".npy", ".bin", ".pt"):
            matches = [f for f in os.listdir(voice_dir) if f.endswith(ext)]
            if matches:
                voice_name = matches[0].replace(ext, "")
                voice_path = os.path.join(voice_dir, matches[0])
                break
        else:
            print("SKIP: No voice files (.npy/.bin/.pt) found in", voice_dir)
            return 77

    try:
        if voice_path.endswith(".npy"):
            voice_style = np.load(voice_path)
        elif voice_path.endswith(".bin"):
            raw = np.fromfile(voice_path, dtype=np.float32)
            voice_style = raw.reshape(-1, VOICE_DIM)
        elif voice_path.endswith(".pt"):
            import torch
            data = torch.load(voice_path, map_location="cpu", weights_only=True)
            if isinstance(data, torch.Tensor):
                voice_style = data.numpy()
            elif isinstance(data, dict):
                # Some .pt files store the tensor under a key
                voice_style = next(iter(data.values())).numpy()
            else:
                voice_style = np.array(data, dtype=np.float32)
        reporter.pass_test(
            "Voice style loaded", f"{voice_name} shape={voice_style.shape}"
        )
    except Exception as e:
        reporter.fail_test("Voice style load", str(e))
        reporter.print_summary()
        return 1

    # --- G2P function ---
    def text_to_phoneme_ids(text: str) -> List[int]:
        """Simple phoneme ID generation using espeak-ng IPA output."""
        result = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        ipa: str = result.stdout.strip()
        tokens: List[int] = [0]  # BOS
        for ch in ipa:
            tokens.append(ord(ch) % 177 + 1)
        tokens.append(0)  # EOS
        return tokens

    # --- Synthesis helper ---
    def synthesize(text: str, speed: float = 1.0) -> List[float]:
        """Run ONNX inference and return float32 PCM samples."""
        tokens = text_to_phoneme_ids(text)

        input_ids = np.array([tokens], dtype=np.int64)
        # Kokoro style packs are shape (N_lengths, 256) — one 256-dim embedding
        # per possible phoneme-sequence length. Select the row matching the
        # current token count (clamped to the pack size) and present it as
        # (1, 256) to the model. Flat (256,) and pre-batched (1, 256) files
        # are passed through unchanged.
        if voice_style.ndim == 2 and voice_style.shape[-1] == VOICE_DIM \
                and voice_style.shape[0] > 1:
            idx = min(max(0, len(tokens) - 1), voice_style.shape[0] - 1)
            style = voice_style[idx:idx + 1].astype(np.float32)
        else:
            style = voice_style.reshape(1, -1).astype(np.float32)
        speed_arr = np.array([speed], dtype=np.float32)

        input_names: List[str] = [inp.name for inp in session.get_inputs()]
        feeds: dict = {}
        for name in input_names:
            if "token" in name.lower() or "input" in name.lower():
                feeds[name] = input_ids
            elif "style" in name.lower() or "voice" in name.lower():
                feeds[name] = style
            elif "speed" in name.lower():
                feeds[name] = speed_arr

        if len(feeds) != len(input_names):
            feeds = {}
            if len(input_names) >= 3:
                feeds[input_names[0]] = input_ids
                feeds[input_names[1]] = style
                feeds[input_names[2]] = speed_arr

        outputs = session.run(None, feeds)
        audio: List[float] = outputs[0].flatten().tolist()
        return audio

    # --- Test 1: Basic synthesis produces non-empty PCM ---
    audio: Optional[List[float]] = None
    try:
        audio = synthesize("Hello world.")
        if len(audio) > 0:
            reporter.pass_test("Non-empty PCM output", f"Samples: {len(audio)}")
        else:
            reporter.fail_test("Non-empty PCM output", "Got 0 samples")
    except Exception as e:
        reporter.fail_test("Non-empty PCM output", str(e))
        audio = None

    # --- Test 2: PCM is not all silence ---
    if audio:
        audio_rms: float = rms(audio)
        if audio_rms > 0.001:
            reporter.pass_test("PCM not silence", f"RMS: {audio_rms:.4f}")
        else:
            reporter.fail_test("PCM not silence", f"RMS too low: {audio_rms:.6f}")

    # --- Test 3: Duration is reasonable ---
    if audio:
        duration_s: float = len(audio) / SAMPLE_RATE
        if 0.1 < duration_s < 30.0:
            reporter.pass_test(
                "Duration reasonable", f"{duration_s:.2f}s for 'Hello world.'"
            )
        else:
            reporter.fail_test(
                "Duration", f"Unexpected duration: {duration_s:.2f}s"
            )

    # --- Test 4: No NaN or Inf in output ---
    if audio:
        has_nan: bool = any(math.isnan(s) for s in audio)
        has_inf: bool = any(math.isinf(s) for s in audio)
        if not has_nan and not has_inf:
            reporter.pass_test("No NaN/Inf in PCM")
        else:
            reporter.fail_test(
                "PCM contains NaN/Inf", f"NaN: {has_nan}, Inf: {has_inf}"
            )

    # --- Test 5: Longer sentence produces more samples ---
    try:
        short_audio = synthesize("Hi.")
        long_audio = synthesize(
            "This is a longer sentence with more words to speak."
        )

        if len(long_audio) > len(short_audio):
            reporter.pass_test(
                "Longer text -> more samples",
                f"Short: {len(short_audio)}, Long: {len(long_audio)}",
            )
        else:
            reporter.fail_test(
                "Longer text should produce more samples",
                f"Short: {len(short_audio)}, Long: {len(long_audio)}",
            )
    except Exception as e:
        reporter.fail_test("Length comparison", str(e))

    # --- Test 6: Write WAV and verify file ---
    if audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            write_wav(wav_path, audio)
            file_size: int = os.path.getsize(wav_path)
            if file_size > 100:
                reporter.pass_test(
                    "WAV file written", f"{file_size} bytes at {wav_path}"
                )
            else:
                reporter.fail_test("WAV file too small", f"{file_size} bytes")
        except Exception as e:
            reporter.fail_test("WAV file write", str(e))
        finally:
            os.unlink(wav_path)

    # --- Summary ---
    reporter.print_summary()
    return reporter.exit_code()


if __name__ == "__main__":
    sys.exit(main())
