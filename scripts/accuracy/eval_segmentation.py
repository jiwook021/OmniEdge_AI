#!/usr/bin/env python3
"""OmniEdge_AI -- Person Segmentation Accuracy Benchmark (YOLOv8-seg)

Tests YOLOv8n-seg person segmentation via the C++ GPU compositor tests.
The compositor tests generate known synthetic frames with person-shaped
regions and verify pixel-level mask accuracy.

Metrics:
  - Pixel-level IoU on synthetic person masks (target: IoU >= 0.70)
  - Tested via the built C++ GTest binary

Requirements:
  - YOLOv8-seg TRT engine in OE_ENGINES_DIR/yolov8n-seg.engine
  - Built CUDA test: build/tests/cv/test_gpu_compositor

Exit codes: 0 = passed, 1 = failed, 77 = skipped
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    BUILD_DIR,
    resolve_engine_path,
    run_gtest_binary,
    setup_logging,
)

logger = setup_logging(__name__)

SUBPROCESS_TIMEOUT_S: int = 180


def benchmark(warmup: int = 0):
    """Run the segmentation benchmark and return a results dict."""
    result = {
        "status": "pending",
        "metrics": {},
        "latencies_ms": [],
        "detail_lines": [],
        "error": "",
    }

    engine_path = resolve_engine_path("yolov8n-seg.engine")
    if not os.path.isfile(engine_path):
        result["status"] = "skip"
        result["error"] = f"YOLOv8-seg engine not found at {engine_path}"
        return result

    test_binary = os.path.join(BUILD_DIR, "tests/integration/hal/test_gpu_compositor")
    if not os.path.isfile(test_binary):
        result["status"] = "skip"
        result["error"] = f"GPU compositor test not built: {test_binary}"
        return result

    rc, stdout, stderr = run_gtest_binary(
        test_binary,
        timeout_s=SUBPROCESS_TIMEOUT_S,
        env={"OE_YOLO_ENGINE": engine_path},
    )

    result["detail_lines"].append(f"  Engine: {engine_path}")
    result["detail_lines"].append(f"  Binary: {test_binary}")
    if stdout:
        result["detail_lines"].append(stdout)
    if stderr:
        result["detail_lines"].append(stderr)

    if rc == -1:
        result["status"] = "fail"
        result["error"] = stderr
    elif rc == 0:
        result["status"] = "pass"
    else:
        result["status"] = "fail"
        result["error"] = f"Test exited with code {rc}"

    return result


def main() -> int:
    """Run the segmentation benchmark and return a shell exit code."""
    print("\033[1mPerson Segmentation Benchmark (YOLOv8-seg)\033[0m\n")

    r = benchmark()

    if r["status"] == "skip":
        print(f"SKIP: {r['error']}")
        return 77

    for line in r["detail_lines"]:
        print(line)

    if r["status"] == "pass":
        print("  \033[32mPASS\033[0m  Segmentation GPU tests passed")
        return 0
    else:
        print(f"  \033[31mFAIL\033[0m  {r['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
