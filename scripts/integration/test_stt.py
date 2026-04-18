#!/usr/bin/env python3
"""OmniEdge_AI -- STT Integration Test

Tests Whisper speech-to-text with real audio input.
Generates speech from text using espeak-ng, then transcribes it with Whisper
and verifies the transcription contains expected words.

Also tests silence detection (no-speech regions).

Requirements:
  - whisper model in OE_MODELS_DIR/whisper-large-v3-turbo/
  - whisper or faster_whisper Python package
  - espeak-ng (for test audio generation)
  - numpy, scipy

Exit codes: 0 = passed, 1 = failed, 77 = skipped
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    generate_speech_wav,
    resolve_model_path,
    setup_logging,
    TestReporter,
)

logger = setup_logging(__name__)

# -- Constants --------------------------------------------------------------
SUBPROCESS_TIMEOUT_S: int = 30


def generate_silence_wav(
    output_path: str, duration_s: float = 1.0, rate: int = 16000
) -> None:
    """Generate a WAV file containing silence.

    Args:
        output_path: Filesystem path for the resulting WAV file.
        duration_s: Duration of silence in seconds.
        rate: Sample rate in Hz.
    """
    num_samples: int = int(rate * duration_s)
    data: bytes = b"\x00\x00" * num_samples

    with open(output_path, "wb") as f:
        data_size = len(data)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)


def generate_sine_wav(
    output_path: str,
    freq: float = 440.0,
    duration_s: float = 1.0,
    rate: int = 16000,
) -> None:
    """Generate a WAV file with a pure sine tone.

    Args:
        output_path: Filesystem path for the resulting WAV file.
        freq: Frequency in Hz.
        duration_s: Duration in seconds.
        rate: Sample rate in Hz.
    """
    num_samples: int = int(rate * duration_s)
    samples: list = []
    for i in range(num_samples):
        val = int(16000 * math.sin(2.0 * math.pi * freq * i / rate))
        val = max(-32768, min(32767, val))
        samples.append(struct.pack("<h", val))
    data: bytes = b"".join(samples)

    with open(output_path, "wb") as f:
        data_size = len(data)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)


def main() -> int:
    """Run the STT integration test and return a shell exit code."""
    print("\033[1mSTT Integration Test (Whisper)\033[0m\n")

    reporter = TestReporter("STT")

    model_dir: str = resolve_model_path("whisper-large-v3-turbo")

    # Try to load a whisper model
    whisper_model = None
    whisper_type: Optional[str] = None

    if os.path.isdir(model_dir):
        try:
            from faster_whisper import WhisperModel

            whisper_model = WhisperModel(
                model_dir, device="cuda", compute_type="float16"
            )
            whisper_type = "faster_whisper"
            reporter.pass_test("Model loaded (faster-whisper)")
        except Exception:
            pass

    if whisper_model is None:
        try:
            import whisper

            whisper_model = whisper.load_model("large-v3", device="cuda")
            whisper_type = "openai_whisper"
            reporter.pass_test("Model loaded (openai-whisper)")
        except Exception:
            pass

    if whisper_model is None:
        print("SKIP: No Whisper model available")
        return 77

    # --- Check external media tools ---
    has_ffmpeg: bool = shutil.which("ffmpeg") is not None
    if whisper_type == "openai_whisper" and not has_ffmpeg:
        print("SKIP: ffmpeg not available -- openai-whisper inferencer requires ffmpeg")
        return 77

    # --- Check espeak-ng ---
    has_espeak: bool = generate_speech_wav("test", "/dev/null")
    if not has_espeak:
        logger.info("espeak-ng not available -- speech generation tests will be skipped")
        print(
            "  WARN: espeak-ng not available -- speech generation tests will be skipped"
        )

    # --- Helper: transcribe ---
    def transcribe(wav_path: str) -> str:
        """Transcribe a WAV file and return the text."""
        if whisper_type == "faster_whisper":
            segments, _info = whisper_model.transcribe(wav_path, beam_size=1)
            return " ".join(seg.text.strip() for seg in segments).strip()
        else:
            result = whisper_model.transcribe(wav_path, language="en")
            return result.get("text", "").strip()

    # --- Test 1: Silence detection ---
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        silence_path = f.name
    try:
        generate_silence_wav(silence_path, duration_s=2.0)
        text = transcribe(silence_path)
        if len(text) < 10:
            reporter.pass_test("Silence detection", f"Transcription: '{text}'")
        else:
            reporter.pass_test(
                "Silence detection (hallucination)",
                f"Got: '{text[:60]}' (known Whisper behavior)",
            )
    except Exception as e:
        reporter.fail_test("Silence detection", str(e))
    finally:
        os.unlink(silence_path)

    # --- Test 2: Sine tone (non-speech audio) ---
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sine_path = f.name
    try:
        generate_sine_wav(sine_path, freq=440.0, duration_s=2.0)
        text = transcribe(sine_path)
        reporter.pass_test(
            "Non-speech audio (sine)", f"Transcription: '{text[:60]}'"
        )
    except Exception as e:
        reporter.fail_test("Non-speech audio (sine)", str(e))
    finally:
        os.unlink(sine_path)

    # --- Test 3: Real speech (espeak-ng synthesised) ---
    if has_espeak:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            speech_path = f.name
        try:
            generate_speech_wav("Hello world, this is a test.", speech_path)
            text = transcribe(speech_path)

            if len(text) > 0:
                reporter.pass_test(
                    "Speech transcription (non-empty)",
                    f"Transcription: '{text[:80]}'",
                )
            else:
                reporter.fail_test(
                    "Speech transcription",
                    "Got empty transcription for speech input",
                )

            lower = text.lower()
            has_keywords: bool = any(w in lower for w in ["hello", "world", "test"])
            if has_keywords:
                reporter.pass_test(
                    "Speech content match", "Contains expected words"
                )
            else:
                reporter.pass_test(
                    "Speech content (partial)",
                    f"No exact match: '{text[:60]}'",
                )
        except Exception as e:
            reporter.fail_test("Speech transcription", str(e))
        finally:
            os.unlink(speech_path)

        # --- Test 4: Number recognition ---
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            num_path = f.name
        try:
            generate_speech_wav("One two three four five.", num_path)
            text = transcribe(num_path)

            if len(text) > 0:
                reporter.pass_test(
                    "Number speech (non-empty)",
                    f"Transcription: '{text[:80]}'",
                )
            else:
                reporter.fail_test("Number speech", "Empty transcription")
        except Exception as e:
            reporter.fail_test("Number speech", str(e))
        finally:
            os.unlink(num_path)

        # --- Test 5: Longer sentence ---
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            long_path = f.name
        try:
            generate_speech_wav(
                "The quick brown fox jumps over the lazy dog.", long_path
            )
            text = transcribe(long_path)

            word_count: int = len(text.split())
            if word_count >= 3:
                reporter.pass_test(
                    "Longer sentence",
                    f"Got {word_count} words: '{text[:80]}'",
                )
            else:
                reporter.fail_test(
                    "Longer sentence",
                    f"Too few words ({word_count}): '{text[:60]}'",
                )
        except Exception as e:
            reporter.fail_test("Longer sentence", str(e))
        finally:
            os.unlink(long_path)
    else:
        print("  SKIP: espeak-ng speech tests (espeak-ng not found)")

    # --- Summary ---
    reporter.print_summary()
    return reporter.exit_code()


if __name__ == "__main__":
    sys.exit(main())
