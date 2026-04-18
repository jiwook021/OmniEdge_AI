#!/usr/bin/env python3
"""
OmniEdge_AI — HuggingFace Transformers Streaming Generation Worker

Long-running Python subprocess for model inference via HuggingFace transformers.
The C++ inferencer spawns this process once, then sends prompts via stdin and
reads streaming tokens from stdout.

Supports any model loadable by AutoModelForCausalLM / AutoModel, including:
  - Gemma-4 E4B (default, multimodal MoE)
  - Gemma-4 E2B (lightweight multimodal MoE)
  - Any standard CausalLM

Protocol (JSON lines):
  -> stdin:  {"prompt": "...", "max_tokens": 512, "temperature": 0.7, "top_p": 0.9}
  <- stdout: {"token": "Hello", "done": false}
  <- stdout: {"token": " world", "done": false}
  <- stdout: {"token": "", "done": true}

Special commands:
  -> stdin:  {"command": "cancel"}     -- abort current generation
  -> stdin:  {"command": "unload"}     -- unload model and exit
  -> stdin:  {"command": "ping"}       -- health check
  <- stdout: {"status": "ready"}       -- sent after model is loaded
  <- stdout: {"status": "pong"}        -- reply to ping
  <- stdout: {"error": "..."}          -- on failure
"""

import json
import queue
import sys
import os
import signal
import threading
from typing import Optional


def stdin_reader(input_queue, cancel_event, shutdown_event):
    """Background thread that reads JSON lines from stdin and dispatches them."""
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
                if cmd == "cancel":
                    cancel_event.set()
                    continue
                elif cmd == "unload":
                    shutdown_event.set()
                    cancel_event.set()
                    return
                elif cmd == "ping":
                    input_queue.put({"__pong": True})
                    continue

            input_queue.put(msg)
    except Exception:
        shutdown_event.set()


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: model_generate.py <model_dir>"}),
              flush=True)
        sys.exit(1)

    model_dir = sys.argv[1]

    # Suppress noisy warnings during import
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # Redirect C-level stdout to stderr BEFORE importing model libraries.
    # Some C++ backends write log messages to fd 1 which corrupt our JSON
    # protocol.  We save fd 1 for our JSON output, then point C's fd 1
    # at stderr so library logs go to the container log.
    import io
    _json_fd = os.dup(1)          # save real stdout for our JSON
    os.dup2(2, 1)                 # C-level fd 1 now points to stderr
    _json_pipe = io.TextIOWrapper(
        io.FileIO(_json_fd, mode="w", closefd=True),
        encoding="utf-8", line_buffering=True)

    # Override print to write JSON to the saved fd.
    # If the parent dies mid-stream the JSON pipe breaks; catch that cleanly
    # and exit instead of surfacing a traceback on every SIGTERM-on-switch.
    import builtins
    _builtin_print = builtins.print
    def print(*args, **kwargs):
        kwargs.setdefault("file", _json_pipe)
        try:
            _builtin_print(*args, **kwargs)
        except (BrokenPipeError, OSError):
            os._exit(0)
    builtins.print = print

    # Import transformers
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    except ImportError as e:
        print(json.dumps({"error": f"transformers import failed: {e}"}), flush=True)
        sys.exit(1)

    # Vision: lazy-load AutoProcessor + PIL on first image prompt.
    # The same Gemma4ForConditionalGeneration model handles both text-only and
    # multimodal input — no reload required when switching modes.
    processor = None
    pil_image_cls = None

    def load_processor_lazy():
        nonlocal processor, pil_image_cls
        if processor is not None:
            return True
        try:
            from transformers import AutoProcessor
            from PIL import Image as _PILImage
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            pil_image_cls = _PILImage
            _builtin_print(f"Processor loaded: {type(processor).__name__}",
                           file=sys.stderr, flush=True)
            return True
        except Exception as ex:
            _builtin_print(f"Processor load failed: {ex}", file=sys.stderr, flush=True)
            return False

    # Load model and tokenizer
    try:
        _builtin_print(f"Loading model from {model_dir}...", file=sys.stderr, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        # Try CausalLM first (standard text models), fall back to AutoModel
        # for multimodal architectures like Gemma-4.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
        except (ValueError, KeyError):
            from transformers import AutoModel
            _builtin_print("AutoModelForCausalLM unsupported, using AutoModel",
                           file=sys.stderr, flush=True)
            model = AutoModel.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
        model.eval()
        _builtin_print(f"Model loaded: {type(model).__name__}", file=sys.stderr, flush=True)
    except Exception as e:
        print(json.dumps({"error": f"Model load failed: {e}"}), flush=True)
        sys.exit(1)

    eos_token_id = tokenizer.eos_token_id
    # Gemma-4 uses <turn|> (id 106) as the chat turn terminator. Without this,
    # generate() runs past the turn boundary and emits the next user/model
    # template stanza verbatim. Skip tokens that resolve to unk_token_id — those
    # aren't real added tokens in this tokenizer.
    unk_id = getattr(tokenizer, "unk_token_id", None)
    stop_token_ids = [eos_token_id] if eos_token_id is not None else []
    for candidate in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(candidate)
            if isinstance(tid, int) and tid >= 0 and tid != unk_id and tid not in stop_token_ids:
                stop_token_ids.append(tid)
        except Exception:
            pass

    # Signal readiness
    print(json.dumps({"status": "ready"}), flush=True)

    # Shared state
    cancel_event = threading.Event()
    shutdown_event = threading.Event()
    input_queue: queue.Queue = queue.Queue()
    generation_thread: Optional[threading.Thread] = None

    def handle_sigterm(signum, frame):
        shutdown_event.set()
        cancel_event.set()

    signal.signal(signal.SIGTERM, handle_sigterm)

    reader_thread = threading.Thread(
        target=stdin_reader,
        args=(input_queue, cancel_event, shutdown_event),
        daemon=True,
    )
    reader_thread.start()

    while not shutdown_event.is_set():
        try:
            request = input_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if "__pong" in request:
            print(json.dumps({"status": "pong"}), flush=True)
            continue

        prompt = request.get("prompt", "")
        if not prompt:
            print(json.dumps({"error": "Empty prompt"}), flush=True)
            continue

        cancel_event.clear()

        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        image_path = request.get("image_path", "")

        # Build inputs: multimodal (image + text) when image_path is provided
        # and processor loads successfully; otherwise fall back to text-only.
        use_vision = False
        if image_path:
            if not os.path.exists(image_path):
                _builtin_print(f"image_path not found: {image_path}",
                               file=sys.stderr, flush=True)
            elif load_processor_lazy():
                try:
                    pil_img = pil_image_cls.open(image_path).convert("RGB")
                    # Strip Gemma ChatML wrapper — processor.apply_chat_template
                    # injects turn tokens itself and needs plain text content.
                    user_text = prompt
                    if "<|turn>user\n" in prompt:
                        start = prompt.rfind("<|turn>user\n") + len("<|turn>user\n")
                        end   = prompt.find("<turn|>", start)
                        if end >= 0:
                            user_text = prompt[start:end].strip()
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text",  "text": user_text or "Describe this image."},
                        ],
                    }]
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    input_len = inputs["input_ids"].shape[1]
                    use_vision = True
                    _builtin_print(f"Vision prompt: image={image_path}, "
                                   f"text_len={len(user_text)}, tokens={input_len}",
                                   file=sys.stderr, flush=True)
                except Exception as ex:
                    _builtin_print(f"Vision tokenize failed: {ex} — falling back to text",
                                   file=sys.stderr, flush=True)

        if not use_vision:
            # Tokenize prompt
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[1]
            except Exception as e:
                print(json.dumps({"error": f"Tokenize failed: {e}", "done": True}),
                      flush=True)
                continue

        # Set up streaming
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),  # avoid division by zero
            top_p=top_p,
            do_sample=temperature > 0.01,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids if stop_token_ids else tokenizer.eos_token_id,
        )

        # Run generation in a background thread (streamer blocks until tokens ready)
        def generate_fn():
            try:
                with torch.no_grad():
                    model.generate(**generation_kwargs)
            except Exception as e:
                _builtin_print(f"Generation error: {e}", file=sys.stderr, flush=True)

        gen_thread = threading.Thread(target=generate_fn, daemon=True)
        gen_thread.start()

        # Stream tokens from the streamer
        try:
            cancelled = False
            for text_chunk in streamer:
                if cancel_event.is_set():
                    # Can't easily cancel model.generate() mid-way, but we stop emitting
                    print(json.dumps({"token": "", "done": True, "cancelled": True}),
                          flush=True)
                    cancelled = True
                    break

                if text_chunk:
                    print(json.dumps({"token": text_chunk, "done": False}),
                          flush=True)

            if not cancelled:
                print(json.dumps({"token": "", "done": True}), flush=True)

        except Exception as e:
            print(json.dumps({
                "error": f"Generation failed: {e}", "done": True,
            }), flush=True)

        # Wait for generation thread to finish
        gen_thread.join(timeout=5.0)

    # Cleanup
    try:
        del model
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
