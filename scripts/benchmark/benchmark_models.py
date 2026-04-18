#!/usr/bin/env python3
"""OmniEdge_AI -- Model Benchmark with Quantization Sweep & VRAM-Fit Ranking

Extends the existing accuracy benchmarks with:
  - Per-model quantization variant discovery (finds all available quant variants)
  - VRAM measurement via nvidia-smi before/after model load
  - VRAM-budget ranking: given a target budget (e.g., 12 GB), recommends the
    best combination of model variants that fits
  - Configurable quantization method selection for comparative testing

Usage:
    # Run default benchmarks (same as scripts/accuracy/benchmark_all.py)
    python scripts/benchmark/benchmark_models.py

    # Sweep all available quantization variants per module
    python scripts/benchmark/benchmark_models.py --quant-sweep

    # Rank models that fit within a VRAM budget
    python scripts/benchmark/benchmark_models.py --vram-budget 12288

    # Test a specific quant variant for LLM
    python scripts/benchmark/benchmark_models.py --module llm --quant awq

    # Test specific variant per module
    python scripts/benchmark/benchmark_models.py --quant-map llm=nvfp4,tts=int8

    # JSON report only
    python scripts/benchmark/benchmark_models.py --json-only

Exit codes: 0 = all passed, 1 = any failure, 77 = all skipped
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Insert project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    compute_latency_percentiles,
    resolve_engine_path,
    resolve_model_path,
    setup_logging,
)

logger = setup_logging(__name__)

PROJECT_ROOT: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

# ============================================================================
# Model variant registry — all known quantization variants per module
# ============================================================================

@dataclass
class ModelVariant:
    """A specific quantization variant of a model."""
    module: str
    name: str
    quant: str          # e.g., "awq", "nvfp4", "int8", "fp16", "fp32"
    engine_path: str    # relative to OE_ENGINES_DIR or OE_MODELS_DIR
    is_engine: bool     # True = TRT engine, False = ONNX/HF model
    vram_estimate_mb: int
    description: str = ""

    def resolve_path(self) -> str:
        if self.is_engine:
            return resolve_engine_path(self.engine_path)
        return resolve_model_path(self.engine_path)

    def exists(self) -> bool:
        p = self.resolve_path()
        if not os.path.exists(p):
            return False
        # Engine directories must contain at least one file (e.g. rank0.engine)
        if self.is_engine and os.path.isdir(p):
            return any(os.scandir(p))
        return True


# Registry of all known model variants — add new ones here.
# Includes both quantization variants (same model, different precision)
# and cross-model variants (different architectures/sizes).
VARIANT_REGISTRY: List[ModelVariant] = [
    # ── LLM variants (cross-model: alternative architectures) ────────────
    ModelVariant("llm", "Llama 3.1 8B AWQ", "awq",
                 "llama-3.1-8b-awq", True, 4800,
                 "INT4-AWQ Llama 3.1 (alternative architecture)"),
    # HF-only fallback (no TRT engine needed — uses transformers inferencer)
    ModelVariant("llm", "Llama 3.1 8B HF", "fp16",
                 "llama-3.1-8b-instruct-awq", False, 4800,
                 "Llama 3.1 via HuggingFace transformers"),

    # ── Conversation variants (Gemma-4 is the canonical path) ───────────
    ModelVariant("conversation", "Gemma-4 E4B", "fp16",
                 "gemma-4-e4b", False, 3000,
                 "Audio+vision input, text-only output (default, pairs with Kokoro TTS sidecar)"),
    ModelVariant("conversation", "Gemma-4 E2B", "fp16",
                 "gemma-4-e2b", False, 2500,
                 "Lightweight variant, audio+vision input (pairs with Kokoro TTS sidecar)"),

    # ── TTS variants (quantization) ──────────────────────────────────────
    ModelVariant("tts", "Kokoro INT8", "int8",
                 "kokoro-onnx/onnx/model_int8.onnx", False, 100,
                 "Dynamic INT8 ONNX"),
    ModelVariant("tts", "Kokoro FP32", "fp32",
                 "kokoro-onnx/onnx/model.onnx", False, 200,
                 "Original FP32 ONNX (baseline)"),
    # ── TTS variants (cross-model: different architectures) ──────────────
    ModelVariant("tts", "VITS2 ONNX", "fp32",
                 "vits2/model.onnx", False, 150,
                 "VITS2 end-to-end TTS (alternative architecture)"),
    ModelVariant("tts", "Piper ONNX", "fp32",
                 "piper/en_US-lessac-medium.onnx", False, 80,
                 "Piper TTS (lightweight, CPU-friendly)"),

    # ── VAD (no variants — always ONNX FP32 on CPU) ─────────────────────
    ModelVariant("vad", "Silero VAD", "fp32",
                 "silero_vad.onnx", False, 0,
                 "ONNX Runtime CPU — no VRAM"),

    # ── CV variants ──────────────────────────────────────────────────────
    # (Background-blur segmentation now runs MediaPipe Selfie Seg ONNX
    # directly inside omniedge_bg_blur; there is no standalone TRT-engine
    # benchmarking step for it.)
]


def discover_variants(module_filter: str = "",
                      quant_filter: str = "",
                      quant_map: Optional[Dict[str, str]] = None,
                      sweep: bool = False) -> List[ModelVariant]:
    """Return the list of variants to benchmark."""
    candidates = VARIANT_REGISTRY

    if module_filter:
        modules = {m.strip() for m in module_filter.split(",")}
        candidates = [v for v in candidates if v.module in modules]

    if quant_map:
        # Per-module quant selection: llm=nvfp4,tts=int8
        result = []
        for v in candidates:
            if v.module in quant_map:
                if v.quant == quant_map[v.module]:
                    result.append(v)
            else:
                # Include default (first available) for unspecified modules
                result.append(v)
        candidates = result
    elif quant_filter and not sweep:
        candidates = [v for v in candidates if v.quant == quant_filter]
    elif not sweep:
        # Default: test only the first (preferred) variant per module
        seen_modules: set = set()
        default_candidates = []
        for v in candidates:
            if v.module not in seen_modules:
                default_candidates.append(v)
                seen_modules.add(v.module)
        candidates = default_candidates

    return [v for v in candidates if v.exists()]


# ============================================================================
# VRAM measurement
# ============================================================================

def measure_gpu_vram_mb() -> Optional[int]:
    """Query current GPU VRAM usage via nvidia-smi (MiB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            timeout=10, text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def measure_gpu_vram_free_mb() -> Optional[int]:
    """Query free GPU VRAM via nvidia-smi (MiB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "-i", "0"],
            timeout=10, text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def measure_gpu_vram_total_mb() -> Optional[int]:
    """Query total GPU VRAM via nvidia-smi (MiB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits", "-i", "0"],
            timeout=10, text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


# ============================================================================
# Benchmark execution — delegates to existing benchmark_all.py functions
# ============================================================================

@dataclass
class VariantResult:
    """Result of benchmarking a single model variant."""
    variant: ModelVariant
    status: str = "pending"      # pass | fail | skip
    accuracy_metric: float = 0.0
    accuracy_label: str = ""     # e.g., "WER", "QA Accuracy", "F1"
    threshold: float = 0.0
    vram_measured_mb: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    throughput_value: float = 0.0
    throughput_unit: str = ""     # e.g., "tok/s", "chars/s", "img/s"
    error: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.variant.module,
            "name": self.variant.name,
            "quant": self.variant.quant,
            "path": self.variant.engine_path,
            "status": self.status,
            "accuracy_metric": round(self.accuracy_metric, 4),
            "accuracy_label": self.accuracy_label,
            "threshold": self.threshold,
            "vram_measured_mb": self.vram_measured_mb,
            "vram_estimate_mb": self.variant.vram_estimate_mb,
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "latency_mean_ms": round(self.latency_mean_ms, 1),
            "throughput_value": round(self.throughput_value, 1),
            "throughput_unit": self.throughput_unit,
            "error": self.error,
            "detail": self.detail,
        }


def run_variant_benchmark(variant: ModelVariant,
                          warmup: int = 1) -> VariantResult:
    """Run the accuracy benchmark for a specific model variant.

    Delegates to the existing benchmark functions in benchmark_all.py,
    adding VRAM measurement around the load/inference.
    """
    result = VariantResult(variant=variant)

    # Import the actual benchmark functions from benchmark_all.py
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "accuracy"))
        from benchmark_all import (
            bench_vad, bench_tts, bench_llm, bench_multimodal,
            BenchmarkResult,
        )
        bench_conversation = bench_multimodal
    except ImportError as e:
        result.status = "skip"
        result.error = f"Cannot import benchmark_all: {e}"
        return result

    # Measure VRAM before
    vram_before = measure_gpu_vram_mb()

    # Run the appropriate benchmark
    bench_fn_map = {
        "vad": bench_vad,
        "tts": bench_tts,
        "llm": bench_llm,
        "conversation": bench_conversation,
    }

    fn = bench_fn_map.get(variant.module)
    if fn is None:
        # For seg/face, run the standalone accuracy scripts
        result = _run_standalone_eval(variant)
        return result

    t0 = time.monotonic()
    try:
        bench_result: BenchmarkResult = fn(warmup=warmup)
    except Exception as e:
        result.status = "fail"
        result.error = f"Benchmark crashed: {e}"
        logger.exception("Variant %s benchmark failed", variant.name)
        return result
    wall_s = time.monotonic() - t0

    # Measure VRAM after
    vram_after = measure_gpu_vram_mb()
    if vram_before is not None and vram_after is not None:
        result.vram_measured_mb = max(0, vram_after - vram_before)
    else:
        result.vram_measured_mb = variant.vram_estimate_mb

    # Map benchmark_all result to our VariantResult
    result.status = bench_result.status
    result.error = bench_result.error
    result.detail = bench_result.metrics

    # Extract key metric
    if variant.module == "vad":
        result.accuracy_metric = bench_result.metrics.get("f1", 0)
        result.accuracy_label = "F1"
        result.threshold = bench_result.thresholds.get("f1", 0.75)
    elif variant.module == "tts":
        cer = bench_result.metrics.get("mean_cer")
        result.accuracy_metric = 1.0 - (cer if cer is not None else 1.0)
        result.accuracy_label = "1-CER"
        result.threshold = 1.0 - bench_result.thresholds.get("cer", 0.40)
    elif variant.module == "llm":
        result.accuracy_metric = bench_result.metrics.get("accuracy", 0)
        result.accuracy_label = "QA Acc"
        result.threshold = bench_result.thresholds.get("accuracy", 0.75)
    elif variant.module == "conversation":
        result.accuracy_metric = bench_result.metrics.get("text_accuracy", 0)
        result.accuracy_label = "Txt Acc"
        result.threshold = bench_result.thresholds.get("text_accuracy", 0.70)

    # Latency stats
    if bench_result.latency_stats:
        result.latency_p50_ms = bench_result.latency_stats.get("p50", 0)
        result.latency_p95_ms = bench_result.latency_stats.get("p95", 0)
        result.latency_mean_ms = bench_result.latency_stats.get("mean", 0)

    # Throughput
    if bench_result.throughput:
        result.throughput_unit = bench_result.throughput.get("unit", "")
        # Pick the primary throughput value (first numeric key that isn't "unit")
        for k, v in bench_result.throughput.items():
            if k != "unit" and isinstance(v, (int, float)):
                result.throughput_value = float(v)
                break

    return result


def _run_standalone_eval(variant: ModelVariant) -> VariantResult:
    """Run standalone eval scripts for seg/face modules."""
    result = VariantResult(variant=variant)

    script_map = {
        "face": os.path.join(PROJECT_ROOT, "scripts", "accuracy", "eval_face_recognition.py"),
    }
    script = script_map.get(variant.module)
    if not script or not os.path.isfile(script):
        result.status = "skip"
        result.error = f"Eval script not found for {variant.module}"
        return result

    try:
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=300,
        )
        result.status = "pass" if proc.returncode == 0 else (
            "skip" if proc.returncode == 77 else "fail")
        result.vram_measured_mb = variant.vram_estimate_mb
    except Exception as e:
        result.status = "fail"
        result.error = str(e)

    return result


# ============================================================================
# VRAM-fit ranking — find best model combination for a given budget
# ============================================================================

def rank_vram_fit(results: List[VariantResult],
                  budget_mb: int) -> List[Dict[str, Any]]:
    """Given benchmark results, find the best variant per module that fits budget.

    Strategy: for each module, pick the variant with highest accuracy that
    passed the quality threshold, preferring lower VRAM. Then check if the
    total combination fits within budget_mb.
    """
    # Group by module
    by_module: Dict[str, List[VariantResult]] = {}
    for r in results:
        by_module.setdefault(r.variant.module, []).append(r)

    # For each module, rank variants by: passing > accuracy > lower VRAM
    best_per_module: Dict[str, VariantResult] = {}
    for module, variants in by_module.items():
        passing = [v for v in variants if v.status == "pass"]
        if not passing:
            # Fall back to lowest VRAM estimate even if it didn't pass
            candidates = sorted(variants, key=lambda v: v.variant.vram_estimate_mb)
            if candidates:
                best_per_module[module] = candidates[0]
            continue

        # Sort by accuracy desc, then VRAM asc (prefer better quality, lower VRAM)
        passing.sort(key=lambda v: (-v.accuracy_metric, v.vram_measured_mb or v.variant.vram_estimate_mb))
        best_per_module[module] = passing[0]

    # Calculate total VRAM for the best combination
    total_vram = sum(
        (r.vram_measured_mb or r.variant.vram_estimate_mb)
        for r in best_per_module.values()
    )

    fits = total_vram <= budget_mb
    headroom = budget_mb - total_vram

    # If doesn't fit, try trimming — remove lowest-priority modules
    # Priority order (highest to lowest): conversation > llm > tts > face > seg > vad
    priority = ["conversation", "llm", "tts", "face", "seg", "vad"]
    trimmed = dict(best_per_module)
    if not fits:
        # Try replacing with lower-VRAM variants first
        for module in reversed(priority):
            if module not in by_module or total_vram <= budget_mb:
                break
            module_variants = sorted(
                by_module[module],
                key=lambda v: v.vram_measured_mb or v.variant.vram_estimate_mb
            )
            if module_variants and module in trimmed:
                old_vram = trimmed[module].vram_measured_mb or trimmed[module].variant.vram_estimate_mb
                new = module_variants[0]
                new_vram = new.vram_measured_mb or new.variant.vram_estimate_mb
                if new_vram < old_vram:
                    trimmed[module] = new
                    total_vram -= (old_vram - new_vram)

    recommendation = []
    for module in priority:
        if module in trimmed:
            r = trimmed[module]
            vram = r.vram_measured_mb or r.variant.vram_estimate_mb
            recommendation.append({
                "module": module,
                "variant": r.variant.name,
                "quant": r.variant.quant,
                "vram_mb": vram,
                "accuracy": round(r.accuracy_metric, 4),
                "accuracy_label": r.accuracy_label,
                "status": r.status,
            })

    return [{
        "budget_mb": budget_mb,
        "total_vram_mb": sum(r["vram_mb"] for r in recommendation),
        "fits": sum(r["vram_mb"] for r in recommendation) <= budget_mb,
        "headroom_mb": budget_mb - sum(r["vram_mb"] for r in recommendation),
        "models": recommendation,
    }]


# ============================================================================
# Presentation
# ============================================================================

_STATUS_COLORS = {"pass": "\033[32m", "fail": "\033[31m", "skip": "\033[33m"}
_RESET = "\033[0m"


def print_results_table(results: List[VariantResult],
                        vram_ranking: Optional[List[Dict]] = None) -> None:
    """Print cross-variant comparison table with optional VRAM ranking."""
    print(f"\n{'=' * 100}")
    print("  OMNIEDGE_AI  --  MODEL BENCHMARK (Quantization-Aware)")
    print(f"{'=' * 100}\n")

    # Header
    print(f"  {'Module':<6s} {'Variant':<28s} {'Quant':<8s} {'Status':<6s} "
          f"{'Metric':<12s} {'Throughput':<14s} {'VRAM(MB)':<10s} {'p50(ms)':<9s} {'p95(ms)':<9s}")
    print(f"  {'-'*6} {'-'*28} {'-'*8} {'-'*6} {'-'*12} {'-'*14} {'-'*10} {'-'*9} {'-'*9}")

    for r in results:
        color = _STATUS_COLORS.get(r.status, "")
        status = f"{color}{r.status.upper()}{_RESET}"

        metric_str = (f"{r.accuracy_label}={r.accuracy_metric:.1%}"
                      if r.accuracy_label else "N/A")
        vram_str = str(r.vram_measured_mb or r.variant.vram_estimate_mb)
        p50 = f"{r.latency_p50_ms:.0f}" if r.latency_p50_ms else "-"
        p95 = f"{r.latency_p95_ms:.0f}" if r.latency_p95_ms else "-"
        tput = (f"{r.throughput_value:.1f} {r.throughput_unit}"
                if r.throughput_value else "-")

        # Pad for ANSI escape code width
        pad = 6 + len(status) - len(r.status.upper())
        print(f"  {r.variant.module:<6s} {r.variant.name:<28s} {r.variant.quant:<8s} "
              f"{status:<{pad}s} {metric_str:<12s} {tput:<14s} {vram_str:<10s} {p50:<9s} {p95:<9s}")

    print(f"{'=' * 100}")

    # Totals
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")

    # VRAM ranking
    if vram_ranking:
        for rank in vram_ranking:
            print(f"\n{'=' * 100}")
            fits_str = f"\033[32mFITS\033[0m" if rank["fits"] else f"\033[31mEXCEEDS\033[0m"
            print(f"  VRAM BUDGET RECOMMENDATION ({rank['budget_mb']} MB)  [{fits_str}]")
            print(f"  Total: {rank['total_vram_mb']} MB  |  "
                  f"Headroom: {rank['headroom_mb']} MB")
            print(f"  {'-'*6} {'-'*28} {'-'*8} {'-'*10} {'-'*12} {'-'*6}")
            for m in rank["models"]:
                color = _STATUS_COLORS.get(m["status"], "")
                print(f"  {m['module']:<6s} {m['variant']:<28s} {m['quant']:<8s} "
                      f"{m['vram_mb']:<10d} "
                      f"{m['accuracy_label']}={m['accuracy']:.1%}"
                      f"  {color}{m['status'].upper()}{_RESET}")
            print(f"{'=' * 100}")


def export_json(results: List[VariantResult],
                vram_ranking: Optional[List[Dict]] = None) -> str:
    """Export results to JSON."""
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "benchmark_models.json")

    gpu_total = measure_gpu_vram_total_mb()

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_vram_total_mb": gpu_total,
        "variants": [r.to_dict() for r in results],
        "summary": {
            "passed": sum(1 for r in results if r.status == "pass"),
            "failed": sum(1 for r in results if r.status == "fail"),
            "skipped": sum(1 for r in results if r.status == "skip"),
            "total": len(results),
        },
    }
    if vram_ranking:
        payload["vram_ranking"] = vram_ranking

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return out_path


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="OmniEdge_AI model benchmark with quantization sweep",
    )
    parser.add_argument(
        "--module", "-m", default="",
        help="Comma-separated module filter (conversation,llm,tts,vad,seg,face)",
    )
    parser.add_argument(
        "--quant", "-q", default="",
        help="Filter to a single quantization type (awq,nvfp4,int8,fp16,fp32)",
    )
    parser.add_argument(
        "--quant-map", default="",
        help="Per-module quant selection: llm=nvfp4,tts=int8",
    )
    parser.add_argument(
        "--quant-sweep", action="store_true",
        help="Test ALL available quantization variants per module",
    )
    parser.add_argument(
        "--vram-budget", type=int, default=0,
        help="Target VRAM budget in MB; ranks best model combo that fits "
             "(e.g., 12288 for 12 GB)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Warmup iterations before measurement (default: 1)",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Suppress table output, JSON only",
    )
    args = parser.parse_args()

    # Parse quant-map: "llm=nvfp4,tts=int8" -> {"llm": "nvfp4", "tts": "int8"}
    quant_map: Optional[Dict[str, str]] = None
    if args.quant_map:
        quant_map = {}
        for pair in args.quant_map.split(","):
            k, _, v = pair.partition("=")
            quant_map[k.strip()] = v.strip()

    # Discover which variants to test
    variants = discover_variants(
        module_filter=args.module,
        quant_filter=args.quant,
        quant_map=quant_map,
        sweep=args.quant_sweep,
    )

    if not variants:
        print("No model variants found. Check $OE_MODELS_DIR and $OE_ENGINES_DIR.")
        return 77

    gpu_total = measure_gpu_vram_total_mb()
    budget = args.vram_budget or (gpu_total or 12288)

    print(f"\033[1mOmniEdge_AI -- Model Benchmark\033[0m")
    print(f"  Variants:   {len(variants)}")
    print(f"  Sweep:      {'yes' if args.quant_sweep else 'no'}")
    print(f"  VRAM total: {gpu_total or '?'} MB")
    print(f"  Budget:     {budget} MB")
    print(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results: List[VariantResult] = []
    for v in variants:
        print(f"--- {v.module.upper()}: {v.name} ({v.quant}) ---")
        t0 = time.monotonic()
        r = run_variant_benchmark(v, warmup=args.warmup)
        wall_s = time.monotonic() - t0
        color = _STATUS_COLORS.get(r.status, "")
        print(f"    {color}{r.status.upper()}{_RESET}  ({wall_s:.1f}s)\n")
        results.append(r)

        # Free VRAM between variants — TRT-LLM and ONNX Runtime hold GPU
        # memory until the Python objects are GC'd + CUDA cache is flushed.
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass

    # VRAM ranking
    vram_ranking = None
    if args.vram_budget or args.quant_sweep:
        vram_ranking = rank_vram_fit(results, budget)

    # Output
    if not args.json_only:
        print_results_table(results, vram_ranking)

    json_path = export_json(results, vram_ranking)
    print(f"\n  Results: {json_path}\n")

    if any(r.status == "fail" for r in results):
        return 1
    if all(r.status == "skip" for r in results):
        return 77
    return 0


if __name__ == "__main__":
    sys.exit(main())
