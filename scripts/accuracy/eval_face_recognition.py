#!/usr/bin/env python3
"""OmniEdge_AI -- Face Recognition Accuracy Benchmark (SCRFD + AuraFace v1)

Tests face detection and embedding similarity using the C++ test binary.
Since we cannot redistribute real face datasets (LFW licensing), this
benchmark delegates to the GTest binary built from C++ sources.

Metrics:
  - Same-identity pairs: cosine similarity > threshold (target: >= 0.80 accuracy)
  - Cross-identity pairs: cosine similarity < threshold
  - Detection rate: faces found in test images (target: >= 0.90)

Requirements:
  - SCRFD + AuraFace v1 ONNX models in OE_MODELS_DIR/face_models/
  - Built test binary: build/tests/cv/test_face_recog_node

Exit codes: 0 = passed, 1 = failed, 77 = skipped
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    BUILD_DIR,
    resolve_model_path,
    run_gtest_binary,
    setup_logging,
)

logger = setup_logging(__name__)

SUBPROCESS_TIMEOUT_S: int = 120


def benchmark(warmup: int = 0):
    """Run the face recognition benchmark and return a results dict."""
    result = {
        "status": "pending",
        "metrics": {},
        "latencies_ms": [],
        "detail_lines": [],
        "error": "",
    }

    face_model_dir = resolve_model_path("face_models/scrfd_auraface")
    if not os.path.isdir(face_model_dir):
        result["status"] = "skip"
        result["error"] = f"Face recognition models not found at {face_model_dir}"
        return result

    test_binaries: List[str] = [
        os.path.join(BUILD_DIR, "tests/nodes/cv/test_face_recog_integration"),
        os.path.join(BUILD_DIR, "tests/interfaces/cv/test_face_detection_api"),
        os.path.join(BUILD_DIR, "tests/interfaces/cv/test_face_matching"),
    ]

    test_binary: Optional[str] = None
    for candidate in test_binaries:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            test_binary = candidate
            break

    if not test_binary:
        result["status"] = "skip"
        result["error"] = "No face recognition test binary found"
        result["detail_lines"] = [f"  Looked for: {p}" for p in test_binaries]
        return result

    rc, stdout, stderr = run_gtest_binary(test_binary, timeout_s=SUBPROCESS_TIMEOUT_S)

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
    """Run the face recognition benchmark and return a shell exit code."""
    print("\033[1mFace Recognition Benchmark (SCRFD + AuraFace v1)\033[0m\n")

    r = benchmark()

    if r["status"] == "skip":
        print(f"SKIP: {r['error']}")
        for line in r["detail_lines"]:
            print(line)
        return 77

    for line in r["detail_lines"]:
        print(line)

    if r["status"] == "pass":
        print("  \033[32mPASS\033[0m  Face recognition tests passed")
        return 0
    else:
        print(f"  \033[31mFAIL\033[0m  {r['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
