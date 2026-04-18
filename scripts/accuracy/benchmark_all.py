#!/usr/bin/env python3
"""OmniEdge_AI -- Unified Model Accuracy & Performance Benchmark

Imports and runs the per-module benchmark() functions from each eval_*.py
script, collects model metadata (file size, opset, quantization, execution
provider), computes detailed latency percentiles, and prints a cross-model
comparison table.

Results are exported to ``logs/benchmark_results.json`` for historical
tracking.

Usage:
    python scripts/accuracy/benchmark_all.py                # run all
    python scripts/accuracy/benchmark_all.py --module tts    # single module
    python scripts/accuracy/benchmark_all.py --module vad,stt
    python scripts/accuracy/benchmark_all.py --json-only     # suppress table, JSON only
    python scripts/accuracy/benchmark_all.py --warmup 3      # warmup iterations

Exit codes: 0 = all passed, 1 = any failure, 77 = all skipped
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    PROJECT_ROOT,
    compute_latency_percentiles,
    profile_onnx_model,
    setup_logging,
)

# Import each eval module's benchmark() function
from scripts.accuracy import (
    eval_face_recognition,
    eval_multimodal_accuracy,
    eval_sam2_segmentation,
    eval_stt_wer,
    eval_tts_quality,
    eval_vad_f1,
    eval_vlm_captioning,
)

logger = setup_logging(__name__)


# ============================================================================
# Data structures
# ============================================================================

class BenchmarkResult:
    """Stores results for a single model benchmark run."""

    def __init__(self, module: str, model_name: str) -> None:
        self.module: str = module
        self.model_name: str = model_name
        self.status: str = "pending"  # pending | pass | fail | skip
        self.profile: Optional[Any] = None
        self.metrics: Dict[str, Any] = {}
        self.thresholds: Dict[str, Any] = {}
        self.latencies_ms: List[float] = []
        self.latency_stats: Dict[str, float] = {}
        self.throughput: Dict[str, float] = {}
        self.detail_lines: List[str] = []
        self.error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON export."""
        return {
            "module": self.module,
            "model_name": self.model_name,
            "status": self.status,
            "profile": self.profile.to_dict() if self.profile and hasattr(self.profile, "to_dict") else None,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "latency_stats": self.latency_stats,
            "throughput": self.throughput,
            "error": self.error,
        }


# ============================================================================
# Benchmark wrappers -- import from eval modules, enrich with profiling
# ============================================================================

def _wrap_eval(
    module_name: str,
    model_name: str,
    eval_benchmark_fn: Callable,
    warmup: int,
) -> BenchmarkResult:
    """Call an eval module's benchmark() and wrap results into BenchmarkResult."""
    result = BenchmarkResult(module_name, model_name)

    raw = eval_benchmark_fn(warmup=warmup)

    result.status = raw.get("status", "fail")
    result.metrics = raw.get("metrics", {})
    result.thresholds = raw.get("thresholds", {})
    result.latencies_ms = raw.get("latencies_ms", [])
    result.detail_lines = raw.get("detail_lines", [])
    result.error = raw.get("error", "")

    # Compute latency percentiles
    if result.latencies_ms:
        result.latency_stats = compute_latency_percentiles(result.latencies_ms)

    # Profile ONNX model if path is available
    mp = raw.get("model_path")
    if mp and os.path.isfile(mp) and mp.endswith(".onnx"):
        try:
            result.profile = profile_onnx_model(mp)
        except Exception:
            pass

    return result


def bench_vad(warmup: int = 1) -> BenchmarkResult:
    """Benchmark Silero VAD via eval_vad_f1."""
    r = _wrap_eval("vad", "silero_vad", eval_vad_f1.benchmark, warmup)

    # Throughput: audio realtime factor
    if r.latencies_ms and r.status != "skip":
        total_wall_s = sum(r.latencies_ms) / 1000.0
        n_cases = len(r.latencies_ms)
        est_audio_s = n_cases * 2.5
        if total_wall_s > 0:
            r.throughput = {
                "audio_realtime_factor": round(est_audio_s / total_wall_s, 2),
                "samples_per_sec": round(n_cases / total_wall_s, 2),
                "unit": "x realtime",
            }

    return r


def bench_tts(warmup: int = 1) -> BenchmarkResult:
    """Benchmark Kokoro TTS via eval_tts_quality."""
    r = _wrap_eval("tts", "kokoro_onnx", eval_tts_quality.benchmark, warmup)

    if r.latencies_ms and r.status != "skip":
        total_chars = sum(len(s) for s in eval_tts_quality.TEST_SENTENCES[:len(r.latencies_ms)])
        total_wall_s = sum(r.latencies_ms) / 1000.0
        if total_wall_s > 0 and total_chars > 0:
            r.throughput = {
                "chars_per_sec": round(total_chars / total_wall_s, 1),
                "sentences_per_sec": round(len(r.latencies_ms) / total_wall_s, 2),
                "unit": "chars/s",
            }

    return r


def bench_stt(warmup: int = 1) -> BenchmarkResult:
    """Benchmark Whisper STT via eval_stt_wer."""
    r = _wrap_eval("stt", "whisper_v3_turbo", eval_stt_wer.benchmark, warmup)

    backend = r.metrics.get("backend")
    if backend:
        r.model_name = f"whisper_v3_turbo ({backend})"

    if r.latencies_ms and r.status != "skip":
        total_ref_words = sum(
            len(s.split()) for s in eval_stt_wer.TEST_SENTENCES[:r.metrics.get("sentences_tested", 0)]
        )
        total_wall_s = sum(r.latencies_ms) / 1000.0
        if total_wall_s > 0:
            r.throughput = {
                "words_per_sec": round(total_ref_words / total_wall_s, 1),
                "sentences_per_sec": round(len(r.latencies_ms) / total_wall_s, 2),
                "unit": "words/s",
            }

    return r


def bench_vlm(warmup: int = 1) -> BenchmarkResult:
    """Benchmark Moondream2 VLM via eval_vlm_captioning."""
    r = _wrap_eval("vlm", "moondream2", eval_vlm_captioning.benchmark, warmup)

    if r.latencies_ms and r.status != "skip":
        total_wall_s = sum(r.latencies_ms) / 1000.0
        if total_wall_s > 0:
            r.throughput = {
                "images_per_sec": round(len(r.latencies_ms) / total_wall_s, 2),
                "unit": "img/s",
            }

    return r


def bench_multimodal(warmup: int = 1) -> BenchmarkResult:
    """Benchmark Gemma-4 E4B/E2B multimodal via eval_multimodal_accuracy."""
    r = _wrap_eval("multimodal", "conversation_model",
                    eval_multimodal_accuracy.benchmark, warmup)

    inferencer_name = r.metrics.get("inferencer", "")
    if inferencer_name:
        r.model_name = f"conversation ({inferencer_name})"

    if r.latencies_ms and r.status != "skip":
        total_wall_s = sum(r.latencies_ms) / 1000.0
        if total_wall_s > 0:
            r.throughput = {
                "queries_per_sec": round(len(r.latencies_ms) / total_wall_s, 2),
                "unit": "query/s",
            }

    return r


def bench_sam2(warmup: int = 1) -> BenchmarkResult:
    """Benchmark SAM2 via eval_sam2_segmentation."""
    r = _wrap_eval("cv", "SAM2 Hiera Tiny", eval_sam2_segmentation.benchmark, warmup)

    r.thresholds = {
        "point_iou": ">= 0.70",
        "box_iou": ">= 0.80",
        "stability_score": ">= 0.80",
        "encode_latency_ms": "< 100",
        "decode_latency_ms": "< 50",
    }

    return r


def bench_face(warmup: int = 1) -> BenchmarkResult:
    """Benchmark face recognition via eval_face_recognition."""
    return _wrap_eval("face", "scrfd_auraface", eval_face_recognition.benchmark, warmup)


# ============================================================================
# Presentation
# ============================================================================

_STATUS_COLORS = {
    "pass": "\033[32m",
    "fail": "\033[31m",
    "skip": "\033[33m",
    "pending": "\033[90m",
}
_RESET = "\033[0m"


def _colored(status: str) -> str:
    c = _STATUS_COLORS.get(status, "")
    return f"{c}{status.upper()}{_RESET}"


def print_summary_table(results: List[BenchmarkResult]) -> None:
    """Print a cross-model comparison table."""
    print(f"\n{'=' * 80}")
    print("  OMNIEDGE_AI  --  UNIFIED MODEL BENCHMARK SUMMARY")
    print(f"{'=' * 80}\n")

    for r in results:
        status_str = _colored(r.status)
        print(f"  [{status_str}]  {r.module.upper():5s}  {r.model_name}")

        if r.error:
            print(f"         Error: {r.error}")
            print()
            continue

        if r.profile and hasattr(r.profile, "summary_line"):
            print(f"         Model: {r.profile.summary_line()}")

        _print_key_metric(r)

        if r.latency_stats:
            ls = r.latency_stats
            print(f"         Latency: p50={ls['p50']:.0f}ms  p95={ls['p95']:.0f}ms  "
                  f"p99={ls['p99']:.0f}ms  mean={ls['mean']:.0f}ms  std={ls['std']:.0f}ms")

        if r.throughput:
            unit = r.throughput.get("unit", "")
            parts = [f"{k}={v}" for k, v in r.throughput.items() if k != "unit"]
            print(f"         Throughput: {', '.join(parts)}  ({unit})")

        if r.detail_lines:
            print("         ---")
            for line in r.detail_lines:
                print(f"        {line}")

        print()

    # Compact comparison table
    print(f"{'=' * 80}")
    print(f"  {'Module':<8s} {'Status':<8s} {'Key Metric':<25s} "
          f"{'p50(ms)':<10s} {'p95(ms)':<10s} {'Model Size':<12s} {'Quant':<10s}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 10}")

    for r in results:
        km = _key_metric_str(r)
        p50 = f"{r.latency_stats.get('p50', 0):.0f}" if r.latency_stats else "-"
        p95 = f"{r.latency_stats.get('p95', 0):.0f}" if r.latency_stats else "-"
        size = (f"{r.profile.file_size_mb:.1f}MB"
                if r.profile and hasattr(r.profile, "file_size_mb") and r.profile.file_size_mb > 0
                else "-")
        quant = r.profile.quantization if r.profile and hasattr(r.profile, "quantization") else "-"

        status_raw = r.status.upper()
        status_fmt = f"{_STATUS_COLORS.get(r.status, '')}{status_raw}{_RESET}"
        pad = 8 + len(status_fmt) - len(status_raw)
        print(f"  {r.module:<8s} {status_fmt:<{pad}s} {km:<25s} "
              f"{p50:<10s} {p95:<10s} {size:<12s} {quant:<10s}")

    print(f"{'=' * 80}")

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped "
          f"out of {len(results)} modules\n")


def _print_key_metric(r: BenchmarkResult) -> None:
    """Print the primary metric line for a given benchmark result."""
    m = r.metrics
    t = r.thresholds
    if r.module == "vad":
        print(f"         F1={m.get('f1', 0):.1%}  Precision={m.get('precision', 0):.1%}  "
              f"Recall={m.get('recall', 0):.1%}  (threshold: F1>={t.get('f1', 0):.0%})")
    elif r.module == "tts":
        cer = m.get("mean_cer")
        cer_s = f"{cer:.1%}" if cer is not None else "N/A"
        print(f"         QualityRate={m.get('quality_pass_rate', 0):.1%}  MeanCER={cer_s}  "
              f"(thresholds: quality>={t.get('quality_rate', 0):.0%}, CER<={t.get('cer', 0):.0%})")
    elif r.module == "stt":
        print(f"         CorpusWER={m.get('corpus_wer', 0):.1%}  MeanWER={m.get('mean_wer', 0):.1%}  "
              f"MedianWER={m.get('median_wer', 0):.1%}  (threshold: <={t.get('wer', 0):.0%})")
    elif r.module == "vlm":
        print(f"         KeywordHit={m.get('keyword_hit_rate', 0):.1%}  "
              f"NonEmpty={m.get('non_empty_rate', 0):.1%}  "
              f"(thresholds: hit>={t.get('keyword_hit_rate', 0):.0%}, "
              f"nonempty>={t.get('non_empty_rate', 0):.0%})")
    elif r.module == "llm":
        print(f"         Accuracy={m.get('accuracy', 0):.1%}  "
              f"MeanRespLen={m.get('mean_response_len', 0):.0f}  "
              f"(threshold: >={t.get('accuracy', 0):.0%})")
    elif r.module == "multimodal":
        print(f"         TextAcc={m.get('text_accuracy', 0):.1%}  "
              f"AudioKw={m.get('audio_keyword_rate', 0):.1%}  "
              f"VisionKw={m.get('vision_keyword_rate', 0):.1%}  "
              f"(thresholds: text>={t.get('text_accuracy', 0):.0%}, "
              f"audio>={t.get('audio_keyword_rate', 0):.0%}, "
              f"vision>={t.get('vision_keyword_rate', 0):.0%})")


def _key_metric_str(r: BenchmarkResult) -> str:
    """Return a short key-metric string for the compact table."""
    m = r.metrics
    if r.module == "vad":
        return f"F1={m.get('f1', 0):.1%}"
    if r.module == "tts":
        cer = m.get("mean_cer")
        return f"CER={cer:.1%}" if cer is not None else "CER=N/A"
    if r.module == "stt":
        return f"WER={m.get('corpus_wer', 0):.1%}"
    if r.module == "vlm":
        return f"KwHit={m.get('keyword_hit_rate', 0):.1%}"
    if r.module == "llm":
        return f"Acc={m.get('accuracy', 0):.1%}"
    if r.module == "multimodal":
        return f"Txt={m.get('text_accuracy', 0):.1%} Aud={m.get('audio_keyword_rate', 0):.1%} Vis={m.get('vision_keyword_rate', 0):.1%}"
    return "N/A"


def export_json(results: List[BenchmarkResult]) -> str:
    """Export results to JSON and return the file path."""
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "benchmark_results.json")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": [r.to_dict() for r in results],
        "summary": {
            "passed": sum(1 for r in results if r.status == "pass"),
            "failed": sum(1 for r in results if r.status == "fail"),
            "skipped": sum(1 for r in results if r.status == "skip"),
            "total": len(results),
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Benchmark results written to %s", out_path)
    return out_path


# ============================================================================
# Main
# ============================================================================

BENCH_REGISTRY: Dict[str, Callable[..., BenchmarkResult]] = {
    "vad": bench_vad,
    "tts": bench_tts,
    "stt": bench_stt,
    "vlm": bench_vlm,
    "multimodal": bench_multimodal,
    "sam2": bench_sam2,
    "face": bench_face,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OmniEdge_AI unified model accuracy benchmark",
    )
    parser.add_argument(
        "--module", "-m",
        help="Comma-separated list of modules to benchmark "
             f"(choices: {','.join(BENCH_REGISTRY)}). Default: all.",
        default="",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Suppress table output, write JSON only.",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup iterations before measurement (default: 1).",
    )
    args = parser.parse_args()

    modules = (
        [m.strip() for m in args.module.split(",") if m.strip()]
        if args.module else list(BENCH_REGISTRY)
    )

    invalid = [m for m in modules if m not in BENCH_REGISTRY]
    if invalid:
        print(f"ERROR: Unknown module(s): {', '.join(invalid)}")
        print(f"  Available: {', '.join(BENCH_REGISTRY)}")
        return 1

    print("\033[1mOmniEdge_AI -- Unified Model Benchmark\033[0m")
    print(f"  Modules: {', '.join(modules)}")
    print(f"  Warmup:  {args.warmup} iteration(s)")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results: List[BenchmarkResult] = []
    for mod in modules:
        print(f"--- Benchmarking {mod.upper()} ---")
        t0 = time.monotonic()
        try:
            r = BENCH_REGISTRY[mod](warmup=args.warmup)
        except Exception as e:
            r = BenchmarkResult(mod, "unknown")
            r.status = "fail"
            r.error = f"Unhandled exception: {e}"
            logger.exception("Benchmark %s crashed", mod)
        wall_s = time.monotonic() - t0
        print(f"    {_colored(r.status)}  ({wall_s:.1f}s)\n")
        results.append(r)

    if not args.json_only:
        print_summary_table(results)

    json_path = export_json(results)
    print(f"  Results exported to: {json_path}\n")

    if any(r.status == "fail" for r in results):
        return 1
    if all(r.status == "skip" for r in results):
        return 77
    return 0


if __name__ == "__main__":
    sys.exit(main())
