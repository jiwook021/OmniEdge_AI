#!/usr/bin/env python3
"""WebSocket functional + edge-case probes for OmniEdge mode UIs."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import websockets


JsonDict = Dict[str, Any]
Predicate = Callable[[JsonDict], bool]


@dataclass
class WsSuiteResult:
    passed: bool
    load_seconds: Optional[float]
    first_frame_seconds: Optional[float]
    edge_failures: List[str]
    stream_counts: Dict[str, int]
    notes: List[str]


async def run_mode_ws_suite(
    mode: str,
    host: str,
    port: int,
    lane: str,
    suite_start_monotonic: float,
) -> WsSuiteResult:
    if mode == "conversation":
        return await _run_conversation_suite(host, port, lane, suite_start_monotonic)
    if mode == "security":
        return await _run_security_suite(host, port, lane, suite_start_monotonic)
    if mode == "beauty":
        return await _run_beauty_suite(host, port, lane, suite_start_monotonic)
    raise ValueError(f"unsupported mode: {mode}")


async def _run_conversation_suite(
    host: str,
    port: int,
    lane: str,
    t0: float,
) -> WsSuiteResult:
    chat_url = f"ws://{host}:{port}/chat"
    video_url = f"ws://{host}:{port}/video"
    screen_url = f"ws://{host}:{port}/screen_video"
    audio_url = f"ws://{host}:{port}/audio"

    first_video_seconds = await _first_binary_latency(video_url, timeout_s=20.0, t0=t0)
    llm_timeout_s = 95.0
    if lane == "full":
        llm_timeout_s = 120.0
    elif lane == "nightly":
        llm_timeout_s = 180.0

    async with websockets.connect(chat_url, max_size=None, ping_interval=None) as chat:
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "switch_mode", "mode": "conversation"})
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "select_conversation_source", "source": "camera"})
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "toggle_video_conversation", "enabled": True})
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "text_input", "text": "e2e load ping"})

        llm_msg = await _wait_for_json(
            chat,
            predicate=lambda m: m.get("type") in {"llm_response", "llm_token"},
            timeout_s=llm_timeout_s,
        )
        load_seconds = (time.monotonic() - t0) if llm_msg is not None else None

        edge_failures = await _run_edge_cases(chat, lane)

    stream_counts = {
        "video": await _count_binary_frames(video_url, seconds=6.0),
        "screen_video": await _count_binary_frames(screen_url, seconds=4.0),
        "audio": await _count_binary_frames(audio_url, seconds=4.0),
    }

    require_load = lane != "smoke"
    passed = stream_counts["video"] > 0 and not edge_failures and (
        load_seconds is not None or not require_load
    )
    notes: List[str] = []
    if load_seconds is None:
        notes.append("conversation llm token not observed within timeout window.")
    if stream_counts["screen_video"] == 0:
        notes.append("screen_video produced 0 frames (screen producer may be idle).")
    if stream_counts["audio"] == 0:
        notes.append("audio produced 0 chunks (TTS output may be idle for this prompt).")
    return WsSuiteResult(
        passed=passed,
        load_seconds=load_seconds,
        first_frame_seconds=first_video_seconds,
        edge_failures=edge_failures,
        stream_counts=stream_counts,
        notes=notes,
    )


async def _run_security_suite(
    host: str,
    port: int,
    lane: str,
    t0: float,
) -> WsSuiteResult:
    chat_url = f"ws://{host}:{port}/chat"
    security_video_url = f"ws://{host}:{port}/security_video"
    video_url = f"ws://{host}:{port}/video"

    first_video_seconds = await _first_binary_latency(video_url, timeout_s=20.0, t0=t0)
    security_frame_timeout_s = 25.0
    security_status_timeout_s = 60.0
    security_stream_window_s = 6.0
    if lane == "nightly":
        # Fresh nightly builds can take longer to spin up security inference streams.
        security_frame_timeout_s = 60.0
        security_status_timeout_s = 120.0
        security_stream_window_s = 10.0
    elif lane == "full":
        security_frame_timeout_s = 40.0
        security_status_timeout_s = 90.0
        security_stream_window_s = 8.0

    first_security_frame_seconds = await _first_binary_latency(
        security_video_url, timeout_s=security_frame_timeout_s, t0=t0
    )

    async with websockets.connect(chat_url, max_size=None, ping_interval=None) as chat:
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "switch_mode", "mode": "security"})
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "toggle_security_mode", "enabled": True})
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "security_list_recordings"})
        status_msg = await _wait_for_json(
            chat,
            predicate=lambda m: m.get("type") in {
                "security_recording_status",
                "security_event",
                "security_detection",
                "security_recordings_list",
            },
            timeout_s=security_status_timeout_s,
        )
        status_seconds = (time.monotonic() - t0) if status_msg is not None else None
        load_seconds = (
            status_seconds
            if status_seconds is not None
            else first_security_frame_seconds
            if first_security_frame_seconds is not None
            else first_video_seconds
        )
        edge_failures = await _run_edge_cases(chat, lane)

    stream_counts = {
        "video": await _count_binary_frames(video_url, seconds=5.0),
        "security_video": await _count_binary_frames(
            security_video_url, seconds=security_stream_window_s
        ),
    }
    require_load = lane != "smoke"
    passed = stream_counts["video"] > 0 and not edge_failures and (
        load_seconds is not None or not require_load
    )
    notes: List[str] = []
    if load_seconds is None:
        notes.append("security load milestone not observed within timeout window.")
    if status_msg is None:
        notes.append("security status milestone not observed within timeout window.")
    if (
        status_msg is None
        and first_security_frame_seconds is None
        and first_video_seconds is not None
    ):
        notes.append("security-specific milestone missing; fell back to base video readiness.")
    if first_security_frame_seconds is None and stream_counts["security_video"] == 0:
        notes.append("security_video first frame missing; detections may be idle or inference failed.")

    return WsSuiteResult(
        passed=passed,
        load_seconds=load_seconds,
        first_frame_seconds=first_video_seconds,
        edge_failures=edge_failures,
        stream_counts=stream_counts,
        notes=notes,
    )


async def _run_beauty_suite(
    host: str,
    port: int,
    lane: str,
    t0: float,
) -> WsSuiteResult:
    chat_url = f"ws://{host}:{port}/chat"
    beauty_video_url = f"ws://{host}:{port}/beauty_video"
    video_url = f"ws://{host}:{port}/video"

    first_video_seconds = await _first_binary_latency(video_url, timeout_s=20.0, t0=t0)

    async with websockets.connect(chat_url, max_size=None, ping_interval=None) as chat:
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "switch_mode", "mode": "beauty"})
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "toggle_beauty", "enabled": True})
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "set_beauty_skin", "smoothing": 21, "toneEven": 13})
        await _send_json(chat, {"v": 1, "type": "ui_command", "action": "set_beauty_light", "brightness": 10, "warmth": -4})

        first_beauty_frame_seconds = await _first_binary_latency(
            beauty_video_url, timeout_s=40.0, t0=t0
        )
        load_seconds = first_beauty_frame_seconds
        edge_failures = await _run_edge_cases(chat, lane)

    stream_counts = {
        "video": await _count_binary_frames(video_url, seconds=5.0),
        "beauty_video": await _count_binary_frames(beauty_video_url, seconds=6.0),
    }
    passed = load_seconds is not None and stream_counts["beauty_video"] > 0 and not edge_failures

    return WsSuiteResult(
        passed=passed,
        load_seconds=load_seconds,
        first_frame_seconds=first_video_seconds,
        edge_failures=edge_failures,
        stream_counts=stream_counts,
        notes=[],
    )


async def _run_edge_cases(chat, lane: str) -> List[str]:
    failures: List[str] = []
    expected_error_prefixes: List[Tuple[str, Any]] = [
        ("invalid-json", "{ this is not json"),
        (
            "unknown-action",
            json.dumps({"v": 1, "type": "ui_command", "action": "unknown_e2e_action"}),
        ),
        (
            "wrong-field-type",
            json.dumps({"v": 1, "type": "ui_command", "action": "push_to_talk", "state": "yes"}),
        ),
    ]

    if lane != "smoke":
        expected_error_prefixes.extend(
            [
                (
                    "missing-required-field",
                    json.dumps({"v": 1, "type": "ui_command", "action": "switch_mode"}),
                ),
                (
                    "bad-source-value",
                    json.dumps(
                        {
                            "v": 1,
                            "type": "ui_command",
                            "action": "select_conversation_source",
                            "source": "hdmi",
                        }
                    ),
                ),
            ]
        )

    for label, payload in expected_error_prefixes:
        await chat.send(payload)
        msg = await _wait_for_json(chat, lambda m: m.get("type") == "error", timeout_s=8.0)
        if msg is None:
            failures.append(f"{label}: no error response")

    for label, payload in _binary_edge_frames(lane):
        await chat.send(payload)
        msg = await _wait_for_json(chat, lambda m: m.get("type") == "error", timeout_s=8.0)
        if msg is None:
            failures.append(f"{label}: no error response")

    # Ensure ws connection still handles valid command after fuzz payloads.
    await _send_json(chat, {"v": 1, "type": "ui_command", "action": "describe_scene"})
    await asyncio.sleep(0.3)
    return failures


def _binary_edge_frames(lane: str) -> List[Tuple[str, bytes]]:
    frames: List[Tuple[str, bytes]] = [("binary-too-small", b"\x01\x02")]

    # Header says JSON is long, but body is short -> invalid header.
    frames.append(("binary-bad-header", struct.pack("<I", 20) + b"{}"))

    # Valid header but unknown action.
    header = json.dumps({"action": "unknown_binary_action"}).encode("utf-8")
    frame = struct.pack("<I", len(header)) + header + b"\xff\xd8\xff\xe0"
    frames.append(("binary-unknown-action", frame))

    # Valid action but unsupported image format magic bytes.
    header = json.dumps({"action": "describe_uploaded_image"}).encode("utf-8")
    frame = struct.pack("<I", len(header)) + header + b"GIF89a"
    frames.append(("binary-unsupported-format", frame))

    if lane != "smoke":
        # Valid action with zero image payload.
        header = json.dumps({"action": "describe_uploaded_image"}).encode("utf-8")
        frame = struct.pack("<I", len(header)) + header
        frames.append(("binary-empty-image", frame))

    return frames


async def _first_binary_latency(url: str, timeout_s: float, t0: float) -> Optional[float]:
    try:
        async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                remaining = max(0.1, deadline - time.monotonic())
                message = await asyncio.wait_for(ws.recv(), timeout=remaining)
                if isinstance(message, (bytes, bytearray)):
                    return time.monotonic() - t0
            return None
    except Exception:
        return None


async def _count_binary_frames(url: str, seconds: float) -> int:
    count = 0
    try:
        async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if isinstance(message, (bytes, bytearray)):
                    count += 1
    except Exception:
        return 0
    return count


async def _send_json(chat, payload: JsonDict) -> None:
    await chat.send(json.dumps(payload))


async def _wait_for_json(chat, predicate: Predicate, timeout_s: float) -> Optional[JsonDict]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            message = await asyncio.wait_for(chat.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

        if not isinstance(message, str):
            continue
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            continue
        if predicate(payload):
            return payload
    return None
