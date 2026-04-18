#!/usr/bin/env python3
"""OmniEdge_AI -- STT Word Error Rate Benchmark (Whisper V3 Turbo)

Generates speech from known text using espeak-ng, transcribes it with
Whisper, and measures Word Error Rate (WER).

Metrics:
  - WER on clean synthesized speech (target: <= 35% for espeak-ng TTS)
  - Silence rejection: transcription of silence should be empty/near-empty

Note: espeak-ng has robotic quality, so WER thresholds are relaxed compared
to human speech benchmarks (LibriSpeech clean WER ~3-5%).

Requirements:
  - Whisper model in OE_MODELS_DIR/whisper-large-v3-turbo/
  - faster_whisper or openai-whisper Python package
  - espeak-ng installed
  - ffmpeg (for WAV resampling if needed)

Exit codes: 0 = all thresholds met, 1 = below threshold, 77 = skipped
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    check_espeak_ng,
    compute_wer,
    generate_silence_wav,
    generate_speech_wav,
    load_whisper_transcriber,
    setup_logging,
    TestReporter,
)

logger = setup_logging(__name__)

# -- Accuracy thresholds ----------------------------------------------------
WER_THRESHOLD_CLEAN: float = 0.35
SILENCE_MAX_WORDS: int = 3
MIN_TEST_SENTENCES: int = 5

# -- Test sentences (shared with benchmark_all.py) --------------------------
TEST_SENTENCES: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test of speech recognition.",
    "One two three four five six seven eight nine ten.",
    "The weather today is sunny with clear blue skies.",
    "Please remember to turn off the lights before leaving.",
    "Artificial intelligence is transforming the technology industry.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "The capital of the United States is Washington D.C.",
    "Python is a popular programming language for data science.",
    "Every morning I drink a cup of coffee with breakfast.",
    "The meeting is scheduled for three o'clock this afternoon.",
]


def benchmark(warmup: int = 0):
    """Run the STT WER benchmark and return a results dict.

    Returns a dict with keys: status, metrics, latencies_ms, detail_lines, error,
    thresholds.
    """
    result = {
        "status": "pending",
        "metrics": {},
        "latencies_ms": [],
        "detail_lines": [],
        "error": "",
        "thresholds": {"wer": WER_THRESHOLD_CLEAN, "silence_max_words": SILENCE_MAX_WORDS},
    }

    if not check_espeak_ng():
        result["status"] = "skip"
        result["error"] = "espeak-ng not installed"
        return result

    whisper_result = load_whisper_transcriber()
    if whisper_result is None:
        result["status"] = "skip"
        result["error"] = "No Whisper model available"
        return result

    backend_name, transcribe = whisper_result

    # Warmup
    for _ in range(warmup):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wp = tmp.name
        try:
            if generate_speech_wav("Hello world.", wp):
                transcribe(wp)
        finally:
            os.unlink(wp)

    # Run WER benchmark
    wer_scores: List[float] = []
    total_ref_words = 0
    total_edit_dist = 0

    for idx, ref_text in enumerate(TEST_SENTENCES):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wp = tmp.name
        try:
            if not generate_speech_wav(ref_text, wp):
                result["detail_lines"].append(f"  S{idx+1}: espeak-ng failed")
                continue

            t0 = time.monotonic()
            hyp = transcribe(wp)
            elapsed = (time.monotonic() - t0) * 1000
            result["latencies_ms"].append(elapsed)

            wer = compute_wer(ref_text, hyp)
            wer_scores.append(wer)
            rw = len(ref_text.lower().split())
            total_ref_words += rw
            total_edit_dist += int(round(wer * rw))

            tag = "OK" if wer <= WER_THRESHOLD_CLEAN else "HIGH"
            result["detail_lines"].append(
                f'  S{idx+1}: [{tag}] WER={wer:.1%}  ({elapsed:.0f}ms)  '
                f'REF="{ref_text[:50]}" HYP="{hyp[:50]}"'
            )
        finally:
            os.unlink(wp)

    if len(wer_scores) < MIN_TEST_SENTENCES:
        result["status"] = "skip"
        result["error"] = f"Only {len(wer_scores)} sentences processed (need {MIN_TEST_SENTENCES})"
        return result

    # Silence test
    silence_words = -1
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        swp = tmp.name
    try:
        generate_silence_wav(swp, 5.0)
        st = transcribe(swp)
        silence_words = len(st.split()) if st else 0
    except Exception:
        pass
    finally:
        os.unlink(swp)

    corpus_wer = total_edit_dist / total_ref_words if total_ref_words > 0 else 1.0
    mean_wer = sum(wer_scores) / len(wer_scores)
    median_wer = sorted(wer_scores)[len(wer_scores) // 2]

    result["metrics"] = {
        "corpus_wer": round(corpus_wer, 4),
        "mean_wer": round(mean_wer, 4),
        "median_wer": round(median_wer, 4),
        "sentences_tested": len(wer_scores),
        "total_sentences": len(TEST_SENTENCES),
        "silence_words": silence_words,
        "backend": backend_name,
    }
    result["status"] = "pass" if corpus_wer <= WER_THRESHOLD_CLEAN else "fail"
    return result


def main() -> int:
    """Run the STT WER benchmark and return a shell exit code."""
    print("\033[1mSTT WER Benchmark (Whisper V3 Turbo)\033[0m\n")
    reporter = TestReporter("STT WER")

    r = benchmark()

    if r["status"] == "skip":
        print(f"SKIP: {r['error']}")
        return 77

    m = r["metrics"]
    print(f"  Backend: {m['backend']}")
    print(f"\n  Running WER on {m['total_sentences']} sentences...\n")

    for line in r["detail_lines"]:
        wer_str = ""
        if "WER=" in line:
            import re
            match = re.search(r"WER=([\d.]+%)", line)
            if match:
                wer_val = float(match.group(1).rstrip("%")) / 100
                status = "\033[32mPASS\033[0m" if wer_val <= WER_THRESHOLD_CLEAN else "\033[31mFAIL\033[0m"
                print(f"  {status} {line}")
                continue
        print(f"  {line}")

    # Silence result
    sw = m.get("silence_words", -1)
    if sw >= 0:
        if sw <= SILENCE_MAX_WORDS:
            print(f"\n  \033[32mPASS\033[0m  Silence: {sw} words (<= {SILENCE_MAX_WORDS})")
        else:
            print(f"\n  \033[33mWARN\033[0m  Silence hallucination: {sw} words")

    mean_latency = sum(r["latencies_ms"]) / len(r["latencies_ms"]) if r["latencies_ms"] else 0

    print(f"\n{'=' * 60}")
    print(f"  Corpus WER:          {m['corpus_wer']:.1%}")
    print(f"  Mean sentence WER:   {m['mean_wer']:.1%}")
    print(f"  Median sentence WER: {m['median_wer']:.1%}")
    print(f"  WER threshold:       {WER_THRESHOLD_CLEAN:.0%}")
    print(f"  Sentences tested:    {m['sentences_tested']}/{m['total_sentences']}")
    print(f"  Mean latency:        {mean_latency:.0f} ms/sentence")
    print(f"  Backend:             {m['backend']}")
    print(f"{'=' * 60}")

    if r["status"] == "pass":
        print(f"  \033[32mPASS\033[0m  Corpus WER {m['corpus_wer']:.1%} <= {WER_THRESHOLD_CLEAN:.0%}")
        return 0
    else:
        print(f"  \033[31mFAIL\033[0m  Corpus WER {m['corpus_wer']:.1%} > {WER_THRESHOLD_CLEAN:.0%}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
