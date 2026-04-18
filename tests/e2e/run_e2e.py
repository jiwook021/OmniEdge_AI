#!/usr/bin/env python3
"""Hybrid browser + websocket E2E suite with VRAM/load-time gates."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from browser_flows import browser_available, run_browser_flow
from metrics import NvidiaSmiSampler, VramSnapshot
from ws_fuzz import WsSuiteResult, run_mode_ws_suite


MODE_CONFIG: Dict[str, Dict[str, object]] = {
    "conversation": {"profile": "conversation", "service": "omniedge", "port": 9001},
    "security": {"profile": "security", "service": "omniedge-security", "port": 9002},
    "beauty": {"profile": "beauty", "service": "omniedge-beauty", "port": 9003},
}

DEFAULT_LOAD_LIMITS = {
    "conversation": 90.0,
    "security": 60.0,
    "beauty": 40.0,
}


@dataclass
class ModeOutcome:
    mode: str
    passed: bool
    errors: List[str]
    warnings: List[str]
    load_seconds: Optional[float]
    load_limit_seconds: float
    first_frame_seconds: Optional[float]
    stream_counts: Dict[str, int]
    edge_failures: List[str]
    browser_executed: bool
    browser_passed: bool
    browser_actions_sent: List[str]
    browser_warning: Optional[str]
    browser_error: Optional[str]
    vram_peak_mib: Optional[int]
    vram_baseline_mib: Optional[int]
    vram_limit_mib: int
    vram_available: bool
    notes: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniEdge E2E runner (browser + websocket + VRAM/load checks)."
    )
    parser.add_argument(
        "--lane",
        choices=["smoke", "full", "nightly"],
        default="smoke",
        help="Execution lane. smoke is lightweight; full/nightly are strict gates.",
    )
    parser.add_argument(
        "--modes",
        default="conversation,security,beauty",
        help="Comma-separated modes from: conversation,security,beauty",
    )
    parser.add_argument(
        "--report",
        default="logs/e2e_report.json",
        help="Path for JSON report output.",
    )
    parser.add_argument(
        "--no-compose",
        action="store_true",
        help="Do not start/stop Docker services; assume services are already running.",
    )
    parser.add_argument(
        "--build-images",
        action="store_true",
        help="Build docker image for each mode before running tests.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Force strict gating regardless of lane (enabled by default for full/nightly).",
    )
    parser.add_argument(
        "--vram-limit-mib",
        type=int,
        default=11264,
        help="Fail if peak used VRAM exceeds this limit.",
    )
    parser.add_argument(
        "--conversation-load-limit",
        type=float,
        default=DEFAULT_LOAD_LIMITS["conversation"],
        help="Max cold load seconds for conversation mode.",
    )
    parser.add_argument(
        "--security-load-limit",
        type=float,
        default=DEFAULT_LOAD_LIMITS["security"],
        help="Max cold load seconds for security mode.",
    )
    parser.add_argument(
        "--beauty-load-limit",
        type=float,
        default=DEFAULT_LOAD_LIMITS["beauty"],
        help="Max cold load seconds for beauty mode.",
    )
    return parser.parse_args()


def run_cmd(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def wait_http_ready(url: str, timeout_s: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if 200 <= resp.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1.0)
    return False


def compose_down_all() -> None:
    run_cmd(
        [
            "docker",
            "compose",
            "--profile",
            "conversation",
            "--profile",
            "security",
            "--profile",
            "beauty",
            "down",
            "--remove-orphans",
        ],
        check=False,
    )


def compose_build(profile: str, service: str) -> None:
    print(f"[build] {profile}:{service}")
    result = run_cmd(["docker", "compose", "--profile", profile, "build", service], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"docker compose build failed for {service}: {result.stderr.strip()}")


def compose_up(profile: str, service: str) -> None:
    result = run_cmd(
        ["docker", "compose", "--profile", profile, "up", "-d", service],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker compose up failed for {service}: {result.stderr.strip()}")


def load_limit_for_mode(mode: str, args: argparse.Namespace) -> float:
    if mode == "conversation":
        return args.conversation_load_limit
    if mode == "security":
        return args.security_load_limit
    if mode == "beauty":
        return args.beauty_load_limit
    raise ValueError(f"unsupported mode: {mode}")


def parse_modes(raw: str) -> List[str]:
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    for mode in modes:
        if mode not in MODE_CONFIG:
            raise ValueError(f"unsupported mode: {mode}")
    return modes


async def run_browser(mode: str, base_url: str):
    return await run_browser_flow(mode=mode, base_url=base_url)


async def run_ws(mode: str, host: str, port: int, lane: str, t0: float) -> WsSuiteResult:
    return await run_mode_ws_suite(
        mode=mode,
        host=host,
        port=port,
        lane=lane,
        suite_start_monotonic=t0,
    )


def evaluate_mode(
    mode: str,
    ws_result: WsSuiteResult,
    browser_result,
    vram: VramSnapshot,
    load_limit_s: float,
    vram_limit_mib: int,
    strict: bool,
) -> ModeOutcome:
    errors: List[str] = []
    warnings: List[str] = []

    if not ws_result.passed:
        errors.append("websocket functional/edge suite failed")

    if ws_result.load_seconds is None:
        msg = "mode load milestone was not reached"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
    elif ws_result.load_seconds > load_limit_s:
        msg = f"cold load {ws_result.load_seconds:.2f}s exceeded limit {load_limit_s:.2f}s"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    if not browser_result.executed:
        msg = browser_result.warning or "browser flow not executed"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
    elif not browser_result.passed:
        errors.append(browser_result.error or "browser flow failed")

    if vram.available:
        if vram.peak_mib is not None and vram.peak_mib > vram_limit_mib:
            msg = f"peak VRAM {vram.peak_mib} MiB exceeded limit {vram_limit_mib} MiB"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
    else:
        msg = vram.warning or "VRAM metrics unavailable"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    warnings.extend(ws_result.notes)
    passed = not errors

    return ModeOutcome(
        mode=mode,
        passed=passed,
        errors=errors,
        warnings=warnings,
        load_seconds=ws_result.load_seconds,
        load_limit_seconds=load_limit_s,
        first_frame_seconds=ws_result.first_frame_seconds,
        stream_counts=ws_result.stream_counts,
        edge_failures=ws_result.edge_failures,
        browser_executed=browser_result.executed,
        browser_passed=browser_result.passed,
        browser_actions_sent=browser_result.actions_sent,
        browser_warning=browser_result.warning,
        browser_error=browser_result.error,
        vram_peak_mib=vram.peak_mib,
        vram_baseline_mib=vram.baseline_mib,
        vram_limit_mib=vram_limit_mib,
        vram_available=vram.available,
        notes=ws_result.notes,
    )


def print_mode_summary(outcome: ModeOutcome) -> None:
    status = "PASS" if outcome.passed else "FAIL"
    print(f"\n[{status}] {outcome.mode}")
    print(
        f"  load={_fmt_seconds(outcome.load_seconds)} "
        f"(limit={outcome.load_limit_seconds:.2f}s), "
        f"peak_vram={_fmt_mib(outcome.vram_peak_mib)}"
    )
    print(f"  streams={json.dumps(outcome.stream_counts, sort_keys=True)}")
    if outcome.errors:
        for err in outcome.errors:
            print(f"  error: {err}")
    if outcome.warnings:
        for warn in outcome.warnings:
            print(f"  warn: {warn}")


def _fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def _fmt_mib(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return f"{value} MiB"


def ensure_docker_available(required: bool) -> None:
    if not required:
        return
    result = run_cmd(["docker", "compose", "version"], check=False)
    if result.returncode != 0:
        raise RuntimeError("docker compose is required for E2E when compose management is enabled")


def run_mode(args: argparse.Namespace, mode: str, strict: bool) -> ModeOutcome:
    cfg = MODE_CONFIG[mode]
    profile = str(cfg["profile"])
    service = str(cfg["service"])
    port = int(cfg["port"])
    base_url = f"http://localhost:{port}"
    host = "localhost"
    load_limit_s = load_limit_for_mode(mode, args)

    sampler = NvidiaSmiSampler(poll_interval_s=1.0)
    sampler.start()

    try:
        if not args.no_compose:
            compose_down_all()
            if args.build_images:
                compose_build(profile, service)
            # Load-time SLA should measure runtime startup, not image build time.
            t0 = time.monotonic()
            compose_up(profile, service)
        else:
            t0 = time.monotonic()

        http_ok = wait_http_ready(base_url, timeout_s=60.0)
        if not http_ok:
            vram_snapshot = sampler.stop()
            return ModeOutcome(
                mode=mode,
                passed=False,
                errors=[f"HTTP endpoint not ready: {base_url}"],
                warnings=[],
                load_seconds=None,
                load_limit_seconds=load_limit_s,
                first_frame_seconds=None,
                stream_counts={},
                edge_failures=[],
                browser_executed=False,
                browser_passed=False,
                browser_actions_sent=[],
                browser_warning=None,
                browser_error=None,
                vram_peak_mib=vram_snapshot.peak_mib,
                vram_baseline_mib=vram_snapshot.baseline_mib,
                vram_limit_mib=args.vram_limit_mib,
                vram_available=vram_snapshot.available,
                notes=[],
            )

        ws_result = asyncio.run(run_ws(mode=mode, host=host, port=port, lane=args.lane, t0=t0))
        browser_result = asyncio.run(run_browser(mode=mode, base_url=base_url))
        vram_snapshot = sampler.stop()

        outcome = evaluate_mode(
            mode=mode,
            ws_result=ws_result,
            browser_result=browser_result,
            vram=vram_snapshot,
            load_limit_s=load_limit_s,
            vram_limit_mib=args.vram_limit_mib,
            strict=strict,
        )
        return outcome
    finally:
        sampler.stop()
        if not args.no_compose:
            compose_down_all()


def write_report(report_path: Path, payload: Dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    strict = args.strict or args.lane in {"full", "nightly"}
    modes = parse_modes(args.modes)

    ensure_docker_available(required=not args.no_compose)

    print("OmniEdge E2E runner")
    print(f"  lane={args.lane} strict={strict} modes={','.join(modes)}")
    print(f"  browser_available={browser_available()}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outcomes: List[ModeOutcome] = []
    for mode in modes:
        print(f"\n=== Running mode: {mode} ===")
        outcome = run_mode(args=args, mode=mode, strict=strict)
        outcomes.append(outcome)
        print_mode_summary(outcome)

    passed = all(o.passed for o in outcomes)
    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload: Dict[str, object] = {
        "runner": "omniedge_e2e",
        "lane": args.lane,
        "strict": strict,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "modes": [asdict(o) for o in outcomes],
        "passed": passed,
    }

    report_path = Path(args.report)
    write_report(report_path=report_path, payload=payload)
    print(f"\nReport written: {report_path}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
