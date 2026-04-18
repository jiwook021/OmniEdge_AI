#!/usr/bin/env python3
"""OmniEdge_AI -- Shared Python utilities for scripts.

Consolidates duplicated helpers across accuracy and integration test scripts:
path resolution, error-rate metrics, audio/image generation, logging, and
test result tracking.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from PIL import Image as _PILImage

__all__ = [
    "PROJECT_ROOT",
    "BUILD_DIR",
    "resolve_env",
    "resolve_model_path",
    "resolve_engine_path",
    "compute_wer",
    "compute_cer",
    "generate_speech_wav",
    "generate_silence_pcm",
    "generate_speech_pcm",
    "write_wav",
    "generate_silence_wav",
    "check_espeak_ng",
    "load_whisper_transcriber",
    "run_gtest_binary",
    "make_synthetic_image",
    "setup_logging",
    "TestReporter",
    "OnnxModelProfile",
    "profile_onnx_model",
    "compute_latency_percentiles",
]

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR: str = os.path.join(PROJECT_ROOT, "build")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_env(name: str, default: str = "") -> str:
    """Resolve an environment variable with a fallback default.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset or empty.

    Returns:
        The environment variable value, or *default*.
    """
    return os.environ.get(name, default) or default


def resolve_model_path(relative: str) -> str:
    """Resolve *relative* underneath ``OE_MODELS_DIR``.

    The base directory defaults to ``$HOME/omniedge_models`` when the
    environment variable is not set.

    Args:
        relative: Sub-path (e.g. ``"whisper-large-v3-turbo"``).

    Returns:
        Absolute path to the requested model resource.
    """
    base = resolve_env("OE_MODELS_DIR", os.path.expanduser("~/omniedge_models"))
    return os.path.join(base, relative)


def resolve_engine_path(relative: str) -> str:
    """Resolve *relative* underneath ``OE_ENGINES_DIR``.

    Defaults to ``$OE_MODELS_DIR/trt_engines`` when the variable is unset.

    Args:
        relative: Sub-path (e.g. ``"qwen2.5-7b-nvfp4"``).

    Returns:
        Absolute path to the requested engine resource.
    """
    models_base = resolve_env("OE_MODELS_DIR", os.path.expanduser("~/omniedge_models"))
    base = resolve_env("OE_ENGINES_DIR", os.path.join(models_base, "trt_engines"))
    return os.path.join(base, relative)


# ---------------------------------------------------------------------------
# Error-rate metrics (Levenshtein-based)
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate via dynamic-programming edit distance.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model-produced transcription.

    Returns:
        WER as a float in ``[0.0, ...]`` (can exceed 1.0 when insertions
        outnumber reference words).
    """
    ref_words: List[str] = reference.lower().split()
    hyp_words: List[str] = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    d: List[List[int]] = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,          # deletion
                d[i][j - 1] + 1,          # insertion
                d[i - 1][j - 1] + cost,   # substitution
            )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate via dynamic-programming edit distance.

    Args:
        reference: Ground-truth string.
        hypothesis: Model-produced string.

    Returns:
        CER as a float in ``[0.0, ...]``.
    """
    ref_chars: List[str] = list(reference.lower().strip())
    hyp_chars: List[str] = list(hypothesis.lower().strip())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    d: List[List[int]] = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------

def generate_speech_wav(
    text: str,
    output_path: str,
    sample_rate: int = 16000,
) -> bool:
    """Generate a WAV file from *text* using ``espeak-ng``.

    Args:
        text: Input text to synthesize.
        output_path: Filesystem path for the resulting WAV file.
        sample_rate: Desired sample rate (espeak-ng natively outputs 22050 Hz;
            the file is written at espeak-ng's native rate -- callers should
            resample if needed).

    Returns:
        ``True`` on success, ``False`` when ``espeak-ng`` is missing or fails.
    """
    try:
        subprocess.run(
            ["espeak-ng", "-w", output_path, "-v", "en", "-s", "140", text],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_espeak_ng() -> bool:
    """Return ``True`` if ``espeak-ng`` is installed and runnable."""
    try:
        subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def generate_silence_pcm(duration_seconds: float, sample_rate: int = 16000) -> List[float]:
    """Generate silence as float32 samples.

    Args:
        duration_seconds: Duration of silence.
        sample_rate: Sample rate in Hz.

    Returns:
        List of zero-valued float samples.
    """
    return [0.0] * int(sample_rate * duration_seconds)


def _parse_wav_samples(data: bytes) -> Tuple[int, List[float]]:
    """Parse a WAV file's raw bytes into (sample_rate, float32_samples).

    Returns ``(0, [])`` if the data is not a valid WAV.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0, []
    fmt_off = data.find(b"fmt ")
    if fmt_off < 0:
        return 0, []
    sample_rate: int = struct.unpack_from("<I", data, fmt_off + 12)[0]
    data_off = data.find(b"data")
    if data_off < 0:
        return 0, []
    data_size: int = struct.unpack_from("<I", data, data_off + 4)[0]
    raw = data[data_off + 8 : data_off + 8 + data_size]
    num_samples = len(raw) // 2
    samples = [
        struct.unpack_from("<h", raw, i * 2)[0] / 32768.0
        for i in range(num_samples)
    ]
    return sample_rate, samples


def _linear_resample(
    samples: List[float], src_rate: int, dst_rate: int
) -> List[float]:
    """Resample via linear interpolation (fallback when no CLI tool is available)."""
    ratio = dst_rate / src_rate
    out_n = int(len(samples) * ratio)
    resampled: List[float] = []
    for i in range(out_n):
        src_pos = i / ratio
        lo = int(src_pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = src_pos - lo
        resampled.append(samples[lo] * (1.0 - frac) + samples[hi] * frac)
    return resampled


def generate_speech_pcm(
    text: str,
    sample_rate: int = 16000,
    subprocess_timeout: int = 30,
) -> List[float]:
    """Generate speech using espeak-ng, resample to *sample_rate*, return float32 PCM.

    Args:
        text: Input text to synthesise.
        sample_rate: Desired output sample rate in Hz.
        subprocess_timeout: Timeout in seconds for subprocesses.

    Returns:
        List of float32 samples at *sample_rate*, or empty list on failure.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    resampled_path = wav_path + ".16k.wav"

    try:
        subprocess.run(
            ["espeak-ng", "-w", wav_path, "-v", "en", "-s", "150", text],
            check=True,
            capture_output=True,
            timeout=subprocess_timeout,
        )

        # Try resampling with a CLI tool (ffmpeg, then sox)
        cli_resampled = False
        for cmd in [
            [
                "ffmpeg", "-y", "-i", wav_path, "-ar", str(sample_rate),
                "-ac", "1", "-acodec", "pcm_s16le", resampled_path,
            ],
            ["sox", wav_path, "-r", str(sample_rate), "-c", "1", resampled_path],
        ]:
            try:
                subprocess.run(
                    cmd, check=True, capture_output=True,
                    timeout=subprocess_timeout,
                )
                cli_resampled = True
                break
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                continue

        source = resampled_path if cli_resampled else wav_path
        with open(source, "rb") as f:
            data = f.read()
        src_rate, samples = _parse_wav_samples(data)
        if not samples:
            return []

        if not cli_resampled and src_rate != sample_rate:
            return _linear_resample(samples, src_rate, sample_rate)
        return samples

    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return []
    finally:
        for p in [wav_path, resampled_path]:
            if os.path.exists(p):
                os.unlink(p)


def write_wav(
    path: str,
    float_samples: List[float],
    sample_rate: int = 24000,
) -> None:
    """Write float32 samples as 16-bit PCM WAV.

    Args:
        path: Output WAV file path.
        float_samples: Audio samples in [-1.0, 1.0].
        sample_rate: Sample rate in Hz.
    """
    pcm_data = b""
    for sample in float_samples:
        clamped = int(max(-32768, min(32767, sample * 32767)))
        pcm_data += struct.pack("<h", clamped)

    with open(path, "wb") as wav_file:
        data_size = len(pcm_data)
        wav_file.write(b"RIFF")
        wav_file.write(struct.pack("<I", 36 + data_size))
        wav_file.write(b"WAVE")
        wav_file.write(b"fmt ")
        wav_file.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                                   sample_rate * 2, 2, 16))
        wav_file.write(b"data")
        wav_file.write(struct.pack("<I", data_size))
        wav_file.write(pcm_data)


def generate_silence_wav(
    output_path: str,
    duration_seconds: float = 3.0,
    sample_rate: int = 16000,
) -> None:
    """Generate a WAV file containing silence.

    Args:
        output_path: Filesystem path for the resulting WAV file.
        duration_seconds: Duration of silence in seconds.
        sample_rate: Sample rate in Hz.
    """
    sample_count = int(sample_rate * duration_seconds)
    pcm_data = b"\x00\x00" * sample_count

    with open(output_path, "wb") as wav_file:
        data_size = len(pcm_data)
        wav_file.write(b"RIFF")
        wav_file.write(struct.pack("<I", 36 + data_size))
        wav_file.write(b"WAVE")
        wav_file.write(b"fmt ")
        wav_file.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                                   sample_rate * 2, 2, 16))
        wav_file.write(b"data")
        wav_file.write(struct.pack("<I", data_size))
        wav_file.write(pcm_data)


# ---------------------------------------------------------------------------
# Whisper model loader
# ---------------------------------------------------------------------------

def load_whisper_transcriber() -> Optional[Tuple[str, object]]:
    """Load a Whisper model for transcription, trying faster_whisper first.

    Returns:
        ``(backend_name, transcribe_fn)`` where *transcribe_fn* accepts a WAV
        path and returns the transcribed text, or ``None`` if unavailable.
        *backend_name* is ``"faster_whisper"`` or ``"openai_whisper"``.
    """
    model_dir = resolve_model_path("whisper-large-v3-turbo")

    if os.path.isdir(model_dir):
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_dir, device="cuda", compute_type="float16")

            def _transcribe(wav_path: str) -> str:
                segments, _ = model.transcribe(wav_path, beam_size=1, language="en")
                return " ".join(seg.text.strip() for seg in segments).strip()

            return ("faster_whisper", _transcribe)
        except Exception:
            pass

    try:
        import whisper

        model = whisper.load_model("large-v3", device="cuda")

        def _transcribe(wav_path: str) -> str:
            result = model.transcribe(wav_path, language="en")
            return result.get("text", "").strip()

        return ("openai_whisper", _transcribe)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# GTest binary runner
# ---------------------------------------------------------------------------

def run_gtest_binary(
    binary_path: str,
    timeout_s: int = 120,
    env: Optional[Dict[str, str]] = None,
    filter_warnings: bool = True,
) -> Tuple[int, str, str]:
    """Run a GTest binary and return ``(returncode, stdout, stderr)``.

    Args:
        binary_path: Absolute path to the test executable.
        timeout_s: Subprocess timeout in seconds.
        env: Optional environment dict (merged with ``os.environ``).
        filter_warnings: If ``True``, strip lines containing ``"warning"``
            from stderr before returning.

    Returns:
        ``(returncode, stdout, filtered_stderr)``.  Returns ``(-1, "", msg)``
        on timeout or other exceptions.
    """
    run_env = {**os.environ, **(env or {})}
    try:
        result = subprocess.run(
            [binary_path, "--gtest_color=yes"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=run_env,
        )
        stderr = result.stderr
        if filter_warnings and stderr:
            stderr = "\n".join(
                line for line in stderr.splitlines()
                if "warning" not in line.lower()
            )
        return result.returncode, result.stdout, stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {timeout_s}s"
    except Exception as e:
        return -1, "", str(e)


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------

def make_synthetic_image(
    description: str,
    width: int = 224,
    height: int = 224,
) -> "_PILImage.Image":
    """Create a simple synthetic test image based on *description*.

    Supported descriptions (case-insensitive substring match):
        ``"red"``, ``"blue"``, ``"green"``, ``"white"``, ``"black"``,
        ``"gray"``/``"grey"`` -- solid colour fills.
        ``"gradient"`` -- horizontal grayscale gradient.
        ``"circle"`` -- red circle on white background.
        ``"checker"`` -- black-and-white checkerboard.
        ``"stripe"`` -- horizontal red/blue stripes.

    Falls back to a mid-gray fill for unrecognised descriptions.

    Args:
        description: Human-readable description driving image content.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A PIL ``Image`` in RGB mode.
    """
    import numpy as np
    from PIL import Image

    desc = description.lower()

    if "gradient" in desc:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            value = int(255 * x / max(1, width - 1))
            arr[:, x] = [value, value, value]
        return Image.fromarray(arr, "RGB")

    if "circle" in desc:
        arr = np.full((height, width, 3), 255, dtype=np.uint8)
        cx, cy, r = width // 2, height // 2, min(width, height) // 4
        for y in range(height):
            for x in range(width):
                if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                    arr[y, x] = [255, 0, 0]
        return Image.fromarray(arr, "RGB")

    if "checker" in desc:
        block = 28
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if ((x // block) + (y // block)) % 2 == 0:
                    arr[y, x] = [255, 255, 255]
        return Image.fromarray(arr, "RGB")

    if "stripe" in desc:
        stripe_h = 32
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            if (y // stripe_h) % 2 == 0:
                arr[y, :] = [255, 0, 0]
            else:
                arr[y, :] = [0, 0, 255]
        return Image.fromarray(arr, "RGB")

    # Solid colour fills
    colour_map = {
        "red":   (255, 0,   0),
        "blue":  (0,   0, 255),
        "green": (0, 255,   0),
        "white": (255, 255, 255),
        "black": (0,   0,   0),
        "gray":  (128, 128, 128),
        "grey":  (128, 128, 128),
    }
    for key, rgb in colour_map.items():
        if key in desc:
            arr = np.full((height, width, 3), rgb, dtype=np.uint8)
            return Image.fromarray(arr, "RGB")

    # Fallback: mid-gray
    arr = np.full((height, width, 3), [128, 128, 128], dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(module_name: str) -> logging.Logger:
    """Configure a Python logger that writes to ``logs/omniedgepy.log`` and stdout.

    The log format mirrors the C++ spdlog convention::

        YYYY-MM-DD HH:MM:SS.mmm [LEVEL] [module] [function] message

    Calling this function multiple times with the same *module_name* is safe;
    handlers are only added once.

    Args:
        module_name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(module_name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    # Determine project root (two levels up from scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "omniedgepy.log")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

class TestReporter:
    """Track PASS / FAIL / SKIP results and print a summary.

    Replaces the global mutable ``PASS`` / ``FAIL`` counter anti-pattern
    that was duplicated across many scripts.

    Usage::

        reporter = TestReporter("STT Integration")
        reporter.pass_test("Silence detection", "got empty transcription")
        reporter.fail_test("Speech content", "expected 'hello'")
        sys.exit(reporter.exit_code())
    """

    def __init__(self, suite_name: str) -> None:
        self.suite_name: str = suite_name
        self._pass: int = 0
        self._fail: int = 0
        self._skip: int = 0

    # -- recording results ---------------------------------------------------

    def pass_test(self, name: str, detail: str = "") -> None:
        """Record a passing test and print a status line."""
        self._pass += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  \033[32mPASS\033[0m  {name}{suffix}")

    def fail_test(self, name: str, detail: str = "") -> None:
        """Record a failing test and print a status line."""
        self._fail += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  \033[31mFAIL\033[0m  {name}{suffix}")

    def skip_test(self, name: str, detail: str = "") -> None:
        """Record a skipped test and print a status line."""
        self._skip += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  \033[33mSKIP\033[0m  {name}{suffix}")

    # -- summary -------------------------------------------------------------

    @property
    def passed(self) -> int:
        """Number of passed tests."""
        return self._pass

    @property
    def failed(self) -> int:
        """Number of failed tests."""
        return self._fail

    @property
    def skipped(self) -> int:
        """Number of skipped tests."""
        return self._skip

    @property
    def total(self) -> int:
        """Total number of recorded test results."""
        return self._pass + self._fail + self._skip

    def print_summary(self) -> None:
        """Print a one-line summary to stdout."""
        print(f"\n  {self.suite_name}: {self._pass} passed, "
              f"{self._fail} failed, {self._skip} skipped")

    def exit_code(self) -> int:
        """Return a shell-compatible exit code.

        Returns:
            ``0`` when all tests passed, ``77`` when everything was skipped,
            ``1`` when any test failed.
        """
        if self._fail > 0:
            return 1
        if self._pass == 0 and self._skip > 0:
            return 77
        return 0


# ---------------------------------------------------------------------------
# ONNX model profiling
# ---------------------------------------------------------------------------

@dataclass
class OnnxModelProfile:
    """Metadata extracted from an ONNX model file."""

    path: str = ""
    file_size_mb: float = 0.0
    opset_version: int = 0
    ir_version: int = 0
    producer: str = ""
    quantization: str = "unknown"
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    num_nodes: int = 0
    execution_provider: str = ""

    def summary_line(self) -> str:
        """One-line summary for table display."""
        quant_tag = self.quantization if self.quantization != "unknown" else "fp32"
        return (
            f"{os.path.basename(self.path)}  "
            f"{self.file_size_mb:.1f}MB  opset={self.opset_version}  "
            f"quant={quant_tag}  nodes={self.num_nodes}  EP={self.execution_provider}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "path": self.path,
            "file_size_mb": round(self.file_size_mb, 2),
            "opset_version": self.opset_version,
            "ir_version": self.ir_version,
            "producer": self.producer,
            "quantization": self.quantization,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "num_nodes": self.num_nodes,
            "execution_provider": self.execution_provider,
        }


def _detect_quantization(model_path: str) -> str:
    """Heuristic quantization detection from filename and ONNX graph.

    Checks the filename first (``int8``, ``fp16``, ``awq``, ``gptq``),
    then inspects ONNX node op types for quantised operators.
    """
    basename = os.path.basename(model_path).lower()
    for tag in ["int4", "int8", "uint8", "fp16", "float16", "awq", "gptq", "nvfp4"]:
        if tag in basename:
            return tag

    try:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
        op_types = {node.op_type for node in model.graph.node}
        if op_types & {"QLinearConv", "QLinearMatMul", "QuantizeLinear",
                       "DequantizeLinear", "MatMulInteger", "QAttention",
                       "QGemm", "DynamicQuantizeLinear"}:
            if "MatMulInteger" in op_types or "QGemm" in op_types:
                return "int8(dynamic)"
            return "int8"
        if "Cast" in op_types:
            # Check if there are fp16 casts — indicates mixed precision
            for node in model.graph.node:
                if node.op_type == "Cast":
                    for attr in node.attribute:
                        if attr.name == "to" and attr.i == 10:  # FLOAT16
                            return "fp16(mixed)"
    except Exception:
        pass
    return "fp32"


def profile_onnx_model(
    model_path: str,
    execution_provider: str = "",
) -> OnnxModelProfile:
    """Extract metadata from an ONNX model file.

    Args:
        model_path: Absolute path to the ``.onnx`` file.
        execution_provider: The EP that will be used at inference time
            (informational — not validated here).

    Returns:
        Populated :class:`OnnxModelProfile`.
    """
    profile = OnnxModelProfile(path=model_path, execution_provider=execution_provider)

    if not os.path.isfile(model_path):
        return profile

    profile.file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    profile.quantization = _detect_quantization(model_path)

    try:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
        profile.opset_version = (
            model.opset_import[0].version if model.opset_import else 0
        )
        profile.ir_version = model.ir_version
        profile.producer = model.producer_name or ""
        profile.num_nodes = len(model.graph.node)

        for inp in model.graph.input:
            shape = []
            if inp.type.tensor_type.HasField("shape"):
                for dim in inp.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value > 0 else "dynamic")
            dtype_num = inp.type.tensor_type.elem_type
            profile.inputs.append({
                "name": inp.name,
                "shape": shape,
                "dtype": dtype_num,
            })

        for out in model.graph.output:
            shape = []
            if out.type.tensor_type.HasField("shape"):
                for dim in out.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value > 0 else "dynamic")
            dtype_num = out.type.tensor_type.elem_type
            profile.outputs.append({
                "name": out.name,
                "shape": shape,
                "dtype": dtype_num,
            })
    except ImportError:
        pass
    except Exception:
        pass

    return profile


def compute_latency_percentiles(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """Compute p50, p90, p95, p99, mean, min, max from a list of latencies.

    Args:
        latencies_ms: Per-sample latency measurements in milliseconds.

    Returns:
        Dict with keys ``p50``, ``p90``, ``p95``, ``p99``, ``mean``,
        ``min``, ``max``, ``std``.  All values in milliseconds.
    """
    if not latencies_ms:
        return {k: 0.0 for k in ["p50", "p90", "p95", "p99", "mean", "min", "max", "std"]}

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    def _percentile(p: float) -> float:
        idx = p / 100.0 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_lat[lo] * (1 - frac) + sorted_lat[hi] * frac

    mean_val = sum(sorted_lat) / n
    variance = sum((x - mean_val) ** 2 for x in sorted_lat) / max(1, n)
    std_val = math.sqrt(variance)

    return {
        "p50": round(_percentile(50), 2),
        "p90": round(_percentile(90), 2),
        "p95": round(_percentile(95), 2),
        "p99": round(_percentile(99), 2),
        "mean": round(mean_val, 2),
        "min": round(sorted_lat[0], 2),
        "max": round(sorted_lat[-1], 2),
        "std": round(std_val, 2),
    }
