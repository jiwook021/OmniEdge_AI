#!/usr/bin/env python3
"""OmniEdge_AI -- VLM Captioning Quality Benchmark (Moondream2)

Generates synthetic images with known content, feeds them to Moondream2,
and checks whether the description matches expected keywords.

Metrics:
  - Keyword hit rate: fraction of test images where the description
    contains at least one expected keyword (target >= 0.60)
  - Non-empty rate: fraction of descriptions that are non-empty (target: 90%)

Requirements:
  - Moondream2 model in OE_MODELS_DIR/moondream2/
  - transformers, torch, Pillow, numpy

Exit codes: 0 = all thresholds met, 1 = below threshold, 77 = skipped
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    make_synthetic_image,
    resolve_model_path,
    setup_logging,
)

logger = setup_logging(__name__)

# -- Accuracy thresholds ----------------------------------------------------
KEYWORD_HIT_THRESHOLD: float = 0.60
NON_EMPTY_THRESHOLD: float = 0.90

# -- Test dataset (shared with benchmark_all.py) ----------------------------
# (image_description, prompt, list of acceptable keywords in lowercase)
TEST_CASES: List[Tuple[str, str, List[str]]] = [
    ("red", "What color is this image? Answer in one word.", ["red"]),
    ("blue", "What color is this image? Answer in one word.", ["blue"]),
    ("green", "What color is this image? Answer in one word.", ["green"]),
    ("white", "Describe this image briefly.",
     ["white", "blank", "plain", "empty", "solid"]),
    ("black", "Describe this image briefly.",
     ["black", "dark", "empty", "solid"]),
    ("gradient", "Describe this image briefly.",
     ["gradient", "gray", "grey", "fade", "transition", "pattern",
      "light", "dark", "black", "white"]),
    ("red circle", "Describe this image briefly.",
     ["circle", "dot", "round", "red", "shape", "ball", "spot"]),
    ("blue circle", "Describe this image briefly.",
     ["circle", "dot", "round", "blue", "shape", "ball", "spot"]),
    ("checker", "Describe this image briefly.",
     ["checker", "pattern", "grid", "square", "black", "white",
      "board", "tile"]),
    ("stripe", "Describe this image briefly.",
     ["stripe", "line", "red", "blue", "pattern", "horizontal",
      "color", "band"]),
]


def benchmark(warmup: int = 0):
    """Run the VLM captioning benchmark and return a results dict."""
    result = {
        "status": "pending",
        "metrics": {},
        "latencies_ms": [],
        "detail_lines": [],
        "error": "",
        "thresholds": {
            "keyword_hit_rate": KEYWORD_HIT_THRESHOLD,
            "non_empty_rate": NON_EMPTY_THRESHOLD,
        },
    }

    model_dir = resolve_model_path("moondream2")
    if not os.path.isdir(model_dir):
        result["status"] = "skip"
        result["error"] = f"Moondream2 not found at {model_dir}"
        return result

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        result["status"] = "skip"
        result["error"] = f"Missing dependency: {e}"
        return result

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            dtype=torch.float16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        result["status"] = "fail"
        result["error"] = f"Model load: {e}"
        return result

    # Build image test cases
    image_cases = [
        (make_synthetic_image(desc), prompt, keywords)
        for desc, prompt, keywords in TEST_CASES
    ]

    # Warmup
    for _ in range(warmup):
        try:
            enc = model.encode_image(image_cases[0][0])
            model.answer_question(enc, "What is this?", tokenizer)
        except Exception:
            pass

    kw_hits = 0
    non_empty = 0
    total = len(image_cases)

    for idx, (img, prompt, keywords) in enumerate(image_cases):
        t0 = time.monotonic()
        try:
            enc = model.encode_image(img)
            desc = model.answer_question(enc, prompt, tokenizer).strip()
        except Exception as e:
            result["detail_lines"].append(f"  T{idx+1}: exception: {e}")
            continue
        elapsed = (time.monotonic() - t0) * 1000
        result["latencies_ms"].append(elapsed)

        if desc:
            non_empty += 1
        hit = any(kw in desc.lower() for kw in keywords)
        if hit:
            kw_hits += 1

        tag = "HIT" if hit else "MISS"
        result["detail_lines"].append(
            f'  T{idx+1}: [{tag}] "{desc[:60]}"  ({elapsed:.0f}ms)'
        )

    kw_rate = kw_hits / total if total > 0 else 0
    ne_rate = non_empty / total if total > 0 else 0

    result["metrics"] = {
        "keyword_hit_rate": round(kw_rate, 4),
        "non_empty_rate": round(ne_rate, 4),
        "keyword_hits": kw_hits,
        "non_empty": non_empty,
        "total": total,
    }

    passed = kw_rate >= KEYWORD_HIT_THRESHOLD and ne_rate >= NON_EMPTY_THRESHOLD
    result["status"] = "pass" if passed else "fail"
    return result


def main() -> int:
    """Run the VLM captioning benchmark and return a shell exit code."""
    print("\033[1mVLM Captioning Benchmark (Moondream2)\033[0m\n")

    r = benchmark()

    if r["status"] == "skip":
        print(f"SKIP: {r['error']}")
        return 77

    if r["status"] == "fail" and r["error"]:
        print(f"  \033[31mFAIL\033[0m  {r['error']}")
        return 1

    for line in r["detail_lines"]:
        hit = "[HIT]" in line
        status = "\033[32mPASS\033[0m" if hit else "\033[31mFAIL\033[0m"
        print(f"  {status} {line}")
        if not hit and "exception" not in line:
            m_idx = re.match(r"\s*T(\d+):", line)
            if m_idx:
                idx = int(m_idx.group(1)) - 1
                if 0 <= idx < len(TEST_CASES):
                    print(f"         Expected one of: {TEST_CASES[idx][2]}")

    m = r["metrics"]
    mean_latency = sum(r["latencies_ms"]) / len(r["latencies_ms"]) if r["latencies_ms"] else 0

    print(f"\n{'=' * 60}")
    print(f"  Keyword hit rate:    {m['keyword_hits']}/{m['total']} = {m['keyword_hit_rate']:.1%}")
    print(f"  Hit threshold:       {KEYWORD_HIT_THRESHOLD:.0%}")
    print(f"  Non-empty rate:      {m['non_empty']}/{m['total']} = {m['non_empty_rate']:.1%}")
    print(f"  Non-empty threshold: {NON_EMPTY_THRESHOLD:.0%}")
    print(f"  Mean latency:        {mean_latency:.0f} ms/image")
    print(f"{'=' * 60}")

    if r["status"] == "pass":
        print(f"  \033[32mPASS\033[0m  Keyword hit rate {m['keyword_hit_rate']:.1%} >= {KEYWORD_HIT_THRESHOLD:.0%}")
        print(f"  \033[32mPASS\033[0m  Non-empty rate {m['non_empty_rate']:.1%} >= {NON_EMPTY_THRESHOLD:.0%}")
        return 0
    else:
        print(f"  \033[31mFAIL\033[0m  Below thresholds")
        return 1


if __name__ == "__main__":
    sys.exit(main())
