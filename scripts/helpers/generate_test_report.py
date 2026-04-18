#!/usr/bin/env python3
"""Parse a test log file and generate a JSON report.

Extracts latencies, pass/fail counts, and produces a structured JSON output.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a test log and generate a JSON report."
    )
    parser.add_argument("--log-file", required=True, help="Path to the test log file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument(
        "--exit-code",
        type=int,
        default=0,
        help="Exit code to include in the report (default: 0)",
    )
    return parser.parse_args()


# Regex patterns
LATENCY_RE = re.compile(r"\[LATENCY\]\s+(.+?):\s+([\d.]+)\s+ms")
GTEST_OK_RE = re.compile(r"\[\s+OK\s+\]")
GTEST_FAILED_RE = re.compile(r"\[\s+FAILED\s+\]")
CTEST_PASSED_RE = re.compile(r"Test\s+#\d+.*Passed")
CTEST_FAILED_RE = re.compile(r"Test\s+#\d+.*\*+Failed")


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.log_file):
        print(f"[ERROR] Log file not found: {args.log_file}", file=sys.stderr)
        return 1

    with open(args.log_file, "r", errors="replace") as f:
        lines = f.readlines()

    # --- Extract latencies -----------------------------------------------
    latencies: dict[str, list[float]] = {}
    for line in lines:
        m = LATENCY_RE.search(line)
        if m:
            name = m.group(1).strip()
            value = float(m.group(2))
            latencies.setdefault(name, []).append(value)

    latency_summary = {}
    for name, values in latencies.items():
        latency_summary[name] = {
            "count": len(values),
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
            "avg_ms": round(sum(values) / len(values), 3),
        }

    # --- Count pass / fail -----------------------------------------------
    passed = 0
    failed = 0
    for line in lines:
        if GTEST_OK_RE.search(line):
            passed += 1
        if GTEST_FAILED_RE.search(line):
            failed += 1
        if CTEST_PASSED_RE.search(line):
            passed += 1
        if CTEST_FAILED_RE.search(line):
            failed += 1

    # --- Build report ----------------------------------------------------
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "log_file": os.path.abspath(args.log_file),
        "exit_code": args.exit_code,
        "total_lines": len(lines),
        "tests": {
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
        },
        "latencies": latency_summary,
    }

    # --- Write output ----------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[generate_test_report] Report written to: {args.output}")
    print(f"[generate_test_report] Tests passed={passed}, failed={failed}")
    if latency_summary:
        print(f"[generate_test_report] Latency metrics: {len(latency_summary)} entries")

    return 0


if __name__ == "__main__":
    sys.exit(main())
