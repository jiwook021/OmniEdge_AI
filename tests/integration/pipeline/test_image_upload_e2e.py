#!/usr/bin/env python3
"""
End-to-end test: Upload a real JPEG/PNG image via WebSocket to OmniEdge_AI
and verify the LLM produces an image description.

Usage:
  # With the full system running (daemon + ws_bridge + llm):
  python3 tests/integration/pipeline/test_image_upload_e2e.py --ws-url ws://localhost:9001/chat

  # Generate a test image and just verify the binary protocol:
  python3 tests/integration/pipeline/test_image_upload_e2e.py --ws-url ws://localhost:9001/chat --timeout 30

Prerequisites:
  pip install websocket-client Pillow

What this test verifies:
  1. Frontend binary frame protocol: [4B json_len LE][JSON header][image bytes]
  2. Bridge parses the binary frame, writes temp file, publishes ui_command
  3. LLM processes the image (native vision capability), publishes llm_response
  4. Bridge relays the response back to the /chat WebSocket as JSON
  5. Client receives an llm_response with non-empty text
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import threading
import time

import websocket


def create_test_jpeg() -> bytes:
    """Create a minimal valid JPEG image in memory."""
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 200, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        print("ERROR: Pillow not installed. Run: pip install Pillow", file=sys.stderr)
        sys.exit(1)


def create_test_png() -> bytes:
    """Create a minimal valid PNG image in memory."""
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(50, 100, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        print("ERROR: Pillow not installed. Run: pip install Pillow", file=sys.stderr)
        sys.exit(1)


def build_upload_frame(image_bytes: bytes, filename: str = "test.jpg") -> bytes:
    """Build the binary WebSocket frame for image upload.

    Format: [4B json_len LE][JSON header][image bytes]
    """
    json_header = json.dumps({
        "action": "describe_uploaded_image",
        "filename": filename,
    }).encode("utf-8")

    json_len = len(json_header)
    frame = struct.pack("<I", json_len) + json_header + image_bytes
    return frame


def run_test(ws_url: str, timeout: int, image_format: str) -> bool:
    """Send an image upload and wait for llm_response with description."""
    if image_format == "jpeg":
        image_bytes = create_test_jpeg()
        filename = "test_upload.jpg"
    else:
        image_bytes = create_test_png()
        filename = "test_upload.png"

    print(f"[test] Created {image_format.upper()} image: {len(image_bytes)} bytes")

    frame = build_upload_frame(image_bytes, filename)
    print(f"[test] Built binary frame: {len(frame)} bytes "
          f"(4B header + {len(json.dumps({'action':'describe_uploaded_image','filename':filename}))}B JSON + "
          f"{len(image_bytes)}B image)")

    result = {"received": False, "text": "", "error": ""}
    event = threading.Event()

    def on_message(ws, message):
        try:
            msg = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return

        msg_type = msg.get("type", "")
        if msg_type == "llm_response":
            text = msg.get("text", "") or msg.get("token", "")
            if msg.get("finished", False) and text:
                result["received"] = True
                result["text"] = text
                print(f"[test] Received llm_response: \"{result['text'][:200]}\"")
                event.set()
        elif msg_type == "error":
            result["error"] = msg.get("message", "unknown error")
            print(f"[test] Received error: {result['error']}")
            event.set()

    def on_open(ws):
        print(f"[test] Connected to {ws_url}")
        print(f"[test] Sending {image_format.upper()} upload ({len(frame)} bytes)...")
        ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
        print("[test] Upload sent, waiting for llm_response...")

    def on_error(ws, error):
        result["error"] = str(error)
        print(f"[test] WebSocket error: {error}")
        event.set()

    def on_close(ws, close_status_code, close_msg):
        print(f"[test] WebSocket closed: {close_status_code} {close_msg}")
        event.set()

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

    print(f"[test] Waiting up to {timeout}s for response...")
    event.wait(timeout=timeout)

    ws.close()
    ws_thread.join(timeout=2)

    if result["error"]:
        print(f"\n[FAIL] Error received: {result['error']}")
        return False

    if not result["received"]:
        print(f"\n[FAIL] No llm_response received within {timeout}s timeout")
        return False

    if not result["text"] or len(result["text"]) < 3:
        print(f"\n[FAIL] llm_response text is empty or too short: \"{result['text']}\"")
        return False

    print(f"\n[PASS] {image_format.upper()} upload -> LLM description received successfully!")
    print(f"       Description ({len(result['text'])} chars): \"{result['text'][:300]}\"")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E2E test: upload image via WebSocket, verify LLM vision response"
    )
    parser.add_argument(
        "--ws-url", default="ws://localhost:9001/chat",
        help="WebSocket /chat URL (default: ws://localhost:9001/chat)"
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Timeout in seconds to wait for LLM response (default: 60)"
    )
    parser.add_argument(
        "--format", choices=["jpeg", "png", "both"], default="both",
        help="Image format to test (default: both)"
    )
    args = parser.parse_args()

    formats = ["jpeg", "png"] if args.format == "both" else [args.format]
    all_passed = True

    for fmt in formats:
        print(f"\n{'='*60}")
        print(f"  Testing {fmt.upper()} image upload")
        print(f"{'='*60}\n")
        if not run_test(args.ws_url, args.timeout, fmt):
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
