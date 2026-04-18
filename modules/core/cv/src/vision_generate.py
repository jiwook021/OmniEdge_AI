#!/usr/bin/env python3
"""
OmniEdge_AI — Vision-Language Model Worker (for LP security_vlm)

Long-running Python subprocess that analyzes short image sequences with a
vision-capable Gemma model and returns a single structured JSON result.

Kept separate from model_generate.py so the conversation streaming path is
not contaminated with multimodal input handling.

Protocol (JSON lines, stdin/stdout):
  -> stdin:  {"mode": "analyze",
              "images_b64": ["<jpeg-base64>", ...],
              "prompt": "...",
              "max_tokens": 512,
              "event_id": "uuid-optional"}
  <- stdout: {"status": "ready"}                   (after model load)
  <- stdout: {"event_id": "...", "result": {...}}  (one per request)
  <- stdout: {"event_id": "...", "error": "..."}   (on per-request failure)

Special commands:
  -> stdin:  {"command": "ping"}     <- stdout: {"status": "pong"}
  -> stdin:  {"command": "unload"}   — unload and exit

The C++ caller writes one request line and reads one response line.
Requests are handled serially.
"""

import base64
import io
import json
import os
import queue
import signal
import sys
import threading
from typing import Any


def stdin_reader(input_queue, shutdown_event):
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
                    shutdown_event.set()
                    return
                if cmd == "ping":
                    input_queue.put({"__pong": True})
                    continue
            input_queue.put(msg)
    except Exception:
        shutdown_event.set()


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: vision_generate.py <model_dir>"}),
              flush=True)
        sys.exit(1)

    model_dir = sys.argv[1]

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # Redirect C-level stdout to stderr BEFORE importing model libraries
    # to prevent native log output from corrupting the JSON protocol.
    _json_fd = os.dup(1)
    os.dup2(2, 1)
    _json_pipe = io.TextIOWrapper(
        io.FileIO(_json_fd, mode="w", closefd=True),
        encoding="utf-8", line_buffering=True)

    import builtins
    _builtin_print = builtins.print

    def print_json(payload):
        _builtin_print(json.dumps(payload), file=_json_pipe, flush=True)

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor
    except ImportError as e:
        print_json({"error": f"vision imports failed: {e}"})
        sys.exit(1)

    # Pick the right vision head in order of preference.
    model = None
    processor = None
    model_class_name = ""

    try:
        _builtin_print(f"Loading vision model from {model_dir}...",
                       file=sys.stderr, flush=True)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

        from transformers import AutoModelForImageTextToText
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
            model_class_name = "AutoModelForImageTextToText"
        except (ValueError, KeyError, OSError):
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
            model_class_name = "AutoModelForVision2Seq"
        model.eval()
        _builtin_print(f"Vision model loaded: {model_class_name}",
                       file=sys.stderr, flush=True)
    except Exception as e:
        print_json({"error": f"Vision model load failed: {e}"})
        sys.exit(1)

    print_json({"status": "ready"})

    shutdown_event = threading.Event()
    input_queue: queue.Queue = queue.Queue()

    def handle_sigterm(_signum, _frame):
        shutdown_event.set()

    signal.signal(signal.SIGTERM, handle_sigterm)

    reader_thread = threading.Thread(
        target=stdin_reader,
        args=(input_queue, shutdown_event),
        daemon=True,
    )
    reader_thread.start()

    while not shutdown_event.is_set():
        try:
            request = input_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if "__pong" in request:
            print_json({"status": "pong"})
            continue

        event_id = request.get("event_id", "")

        if request.get("mode") != "analyze":
            print_json({"event_id": event_id,
                        "error": "Unknown mode — expected 'analyze'"})
            continue

        images_b64 = request.get("images_b64") or []
        prompt = request.get("prompt", "")
        max_tokens = int(request.get("max_tokens", 512))

        if not images_b64 or not prompt:
            print_json({"event_id": event_id,
                        "error": "analyze requires non-empty images_b64 and prompt"})
            continue

        try:
            images = []
            for b64 in images_b64:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                images.append(img)
        except Exception as e:
            print_json({"event_id": event_id,
                        "error": f"Image decode failed: {e}"})
            continue

        try:
            # Chat-template style input: one user turn with interleaved images.
            messages = [{
                "role": "user",
                "content": [{"type": "image"} for _ in images]
                           + [{"type": "text", "text": prompt}],
            }]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images=images,
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )

            # Strip the prompt tokens so we only decode the completion.
            input_len = inputs["input_ids"].shape[1]
            completion_ids = out_ids[0, input_len:]
            completion = processor.batch_decode(
                [completion_ids], skip_special_tokens=True)[0].strip()
        except Exception as e:
            print_json({"event_id": event_id,
                        "error": f"Vision generation failed: {e}"})
            continue

        # Try to parse strict JSON out of the completion. If it is not valid
        # JSON, wrap the raw text so the caller can still surface it.
        parsed: Any
        try:
            parsed = json.loads(completion)
        except json.JSONDecodeError:
            # Pull the first {...} block if the model wrapped its JSON in prose.
            start = completion.find("{")
            end = completion.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(completion[start:end + 1])
                except json.JSONDecodeError:
                    parsed = {"raw_text": completion}
            else:
                parsed = {"raw_text": completion}

        print_json({"event_id": event_id, "result": parsed})

    try:
        del model
        del processor
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
