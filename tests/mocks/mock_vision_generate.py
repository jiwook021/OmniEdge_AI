#!/usr/bin/env python3
"""Deterministic mock replacement for vision_generate.py used by integration tests.

Same JSON-lines protocol as the real worker but loads no model — returns a
fixed structured `result` object plus a counter that tests can assert on.
Accepts the same <model_dir> positional arg for drop-in substitution; the
value is ignored.
"""

import json
import os
import sys
import threading


def main() -> None:
    # Mirror the fd-1-to-stderr trick so any inherited native log writer
    # does not corrupt the test JSON stream.
    _json_fd = os.dup(1)
    os.dup2(2, 1)

    def emit(payload):
        os.write(_json_fd, (json.dumps(payload) + "\n").encode("utf-8"))

    emit({"status": "ready"})

    shutdown = threading.Event()
    call_count = 0

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "command" in msg:
                cmd = msg["command"]
                if cmd == "unload":
                    shutdown.set()
                    break
                if cmd == "ping":
                    emit({"status": "pong"})
                    continue

            event_id = msg.get("event_id", "")
            if msg.get("mode") != "analyze":
                emit({"event_id": event_id, "error": "expected analyze"})
                continue

            images = msg.get("images_b64") or []
            prompt = msg.get("prompt", "")
            if not images or not prompt:
                emit({"event_id": event_id,
                      "error": "missing images_b64 or prompt"})
                continue

            call_count += 1
            emit({
                "event_id": event_id,
                "result": {
                    "behavior": "mock: subject handling a package near exit",
                    "suspicion_level": 2,
                    "reasoning": "deterministic mock reply",
                    "objects_observed": ["person", "backpack"],
                    "mock_call_number": call_count,
                    "image_count": len(images),
                },
            })
    finally:
        shutdown.set()


if __name__ == "__main__":
    main()
