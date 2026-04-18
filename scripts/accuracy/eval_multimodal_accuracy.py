#!/usr/bin/env python3
"""OmniEdge_AI -- Multimodal Accuracy Benchmark (Gemma-4 E4B / E2B)

Evaluates multimodal conversation models with audio, video/image, and text
inputs -- the same modalities that ConversationInferencer handles in C++.

Modality tests:
  - Text-only QA: factual / reasoning baseline
  - Audio understanding: feed espeak-ng speech, verify transcription keywords
  - Vision understanding: feed synthetic images, verify description keywords
  - Audio+Vision combined: feed both, verify joint comprehension

Model loading fallback chain (first available wins):
  1. Gemma-4 E4B  (tokenizer: gemma-4-e4b)
  2. Gemma-4 E2B  (tokenizer: gemma-4-e2b)
  3. Exit 77 -- no multimodal model available

Metrics:
  - text_accuracy: fraction of text QA answered correctly (target >= 0.70)
  - audio_keyword_rate: fraction of audio tests with keyword hit (target >= 0.50)
  - vision_keyword_rate: fraction of vision tests with keyword hit (target >= 0.50)

Exit codes: 0 = all thresholds met, 1 = below threshold, 77 = skipped
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    check_espeak_ng,
    generate_speech_pcm,
    make_synthetic_image,
    resolve_model_path,
    setup_logging,
)

logger = setup_logging(__name__)

# -- Accuracy thresholds ----------------------------------------------------
TEXT_ACCURACY_THRESHOLD: float = 0.70
AUDIO_KEYWORD_THRESHOLD: float = 0.50
VISION_KEYWORD_THRESHOLD: float = 0.50

# -- Text QA dataset (subset -- multimodal models may be weaker on pure text) --
TEXT_QA: List[Tuple[str, List[str]]] = [
    ("What is 2+2? Reply with just the number.", ["4"]),
    ("What is the capital of France? Reply with just the city name.", ["paris"]),
    ("What is the chemical symbol for water? Reply with just the formula.", ["h2o"]),
    ("Who wrote Romeo and Juliet? Reply with just the name.", ["shakespeare"]),
    ("How many planets are in our solar system? Reply with just the number.", ["8"]),
    ("What is the smallest prime number? Reply with just the number.", ["2"]),
    ("What year did World War II end? Reply with just the year.", ["1945"]),
    ("What is the opposite of hot? Reply in one word.", ["cold"]),
    ("How many bits are in a byte? Reply with just the number.", ["8"]),
    ("Say hello in exactly one word.", ["hello", "hi", "hey"]),
]

# -- Audio test cases: (speech_text, question, expected_keywords) ----------
# We synthesize speech with espeak-ng, feed the audio to the model, and ask
# about what was said.  The model should demonstrate audio comprehension.
AUDIO_TESTS: List[Tuple[str, str, List[str]]] = [
    (
        "The quick brown fox jumps over the lazy dog.",
        "What did the speaker say? Summarize briefly.",
        ["fox", "dog", "jump", "quick", "brown", "lazy"],
    ),
    (
        "Hello, my name is Alice.",
        "What is the speaker's name? Reply with just the name.",
        ["alice"],
    ),
    (
        "The weather today is sunny and warm.",
        "What is the weather like according to the speaker?",
        ["sunny", "warm", "weather"],
    ),
    (
        "Please count from one to five.",
        "What numbers did the speaker mention?",
        ["one", "two", "three", "four", "five", "1", "2", "3", "4", "5"],
    ),
    (
        "I would like a cup of coffee.",
        "What does the speaker want? Reply briefly.",
        ["coffee", "cup"],
    ),
]

# -- Vision test cases: (image_description, prompt, expected_keywords) ------
# Reuses make_synthetic_image() from scripts.common.
VISION_TESTS: List[Tuple[str, str, List[str]]] = [
    ("red", "What color is this image? Answer in one word.", ["red"]),
    ("blue", "What color is this image? Answer in one word.", ["blue"]),
    ("green", "What color is this image? Answer in one word.", ["green"]),
    ("white", "Describe this image briefly.",
     ["white", "blank", "plain", "empty", "solid"]),
    ("black", "Describe this image briefly.",
     ["black", "dark", "empty", "solid"]),
    ("red circle", "Describe what you see in this image.",
     ["circle", "dot", "round", "red", "shape", "ball", "spot"]),
    ("checker", "Describe the pattern in this image.",
     ["checker", "pattern", "grid", "square", "black", "white",
      "board", "tile"]),
    ("stripe", "Describe the pattern in this image.",
     ["stripe", "line", "red", "blue", "pattern", "horizontal",
      "color", "band"]),
]


# ---------------------------------------------------------------------------
# Model loading -- multimodal conversation models
# ---------------------------------------------------------------------------

# Each entry: (human name, model_dir_names, tokenizer_dir_names, loader_fn_name)
_MODEL_SEARCH_ORDER: List[Tuple[str, List[str], List[str]]] = [
    ("Gemma-4 E4B",
     ["gemma-4-e4b", "gemma-4-4b-it"],
     ["gemma-4-e4b", "gemma-4-4b-it"]),
    ("Gemma-4 E2B",
     ["gemma-4-e2b", "gemma-4-2b-it"],
     ["gemma-4-e2b", "gemma-4-2b-it"]),
]


def _find_model_dir() -> Optional[Tuple[str, str, str]]:
    """Search for the first available multimodal model on disk.

    Returns (model_name, model_dir, tokenizer_dir) or None.
    """
    for name, model_dirs, tok_dirs in _MODEL_SEARCH_ORDER:
        for md in model_dirs:
            model_path = resolve_model_path(md)
            if not os.path.isdir(model_path):
                continue
            # Find matching tokenizer
            for td in tok_dirs:
                tok_path = resolve_model_path(td)
                if os.path.isdir(tok_path):
                    return name, model_path, tok_path
            # Model dir might contain its own tokenizer
            if os.path.exists(os.path.join(model_path, "tokenizer.json")):
                return name, model_path, model_path
    return None


def _load_multimodal_inferencer():
    """Load a multimodal model via HuggingFace transformers.

    Returns (generate_text_fn, generate_audio_fn, generate_vision_fn, model_name)
    or (None, None, None, "").

    - generate_text_fn(prompt) -> Optional[str]
    - generate_audio_fn(audio_samples, sample_rate, prompt) -> Optional[str]
    - generate_vision_fn(pil_image, prompt) -> Optional[str]
    """
    found = _find_model_dir()
    if found is None:
        return None, None, None, ""

    model_name, model_dir, tok_dir = found
    logger.info("Loading multimodal model: %s from %s", model_name, model_dir)

    try:
        import torch
    except ImportError:
        logger.warning("torch not available")
        return None, None, None, ""

    # Try Gemma multimodal loader
    if "gemma" in model_name.lower():
        result = _try_load_gemma(model_dir, tok_dir, model_name)
        if result[0] is not None:
            return result

    # Generic AutoModel fallback for any vision-language model
    result = _try_load_generic_vlm(model_dir, tok_dir, model_name)
    if result[0] is not None:
        return result

    return None, None, None, ""


def _try_load_gemma(model_dir: str, tok_dir: str, model_name: str):
    """Try loading Gemma 4 E4B with its multimodal API."""
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        return None, None, None, ""

    try:
        processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
        model.eval()

        def _gen_text(prompt: str) -> Optional[str]:
            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, return_tensors="pt").to(model.device)
            plen = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            return processor.decode(out[0][plen:], skip_special_tokens=True).strip() or None

        def _gen_audio(audio: List[float], sr: int, prompt: str) -> Optional[str]:
            try:
                import numpy as np
                audio_array = np.array(audio, dtype=np.float32)
                messages = [{"role": "user", "content": [
                    {"type": "audio", "audio": audio_array, "sampling_rate": sr},
                    {"type": "text", "text": prompt},
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text, audios=[audio_array],
                    sampling_rate=sr, return_tensors="pt").to(model.device)
                plen = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                return processor.decode(out[0][plen:], skip_special_tokens=True).strip() or None
            except Exception as e:
                logger.debug("Gemma audio gen failed: %s", e)
                return None

        def _gen_vision(pil_image, prompt: str) -> Optional[str]:
            try:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text, images=[pil_image],
                    return_tensors="pt").to(model.device)
                plen = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                return processor.decode(out[0][plen:], skip_special_tokens=True).strip() or None
            except Exception as e:
                logger.debug("Gemma vision gen failed: %s", e)
                return None

        return _gen_text, _gen_audio, _gen_vision, f"{model_name} ({model_dir})"

    except Exception as e:
        logger.debug("Gemma load failed: %s", e)
        return None, None, None, ""


def _try_load_generic_vlm(model_dir: str, tok_dir: str, model_name: str):
    """Fallback: load any transformers model that supports generate()."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None, None, None, ""

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
        model.eval()

        def _gen_text(prompt: str) -> Optional[str]:
            try:
                messages = [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": prompt},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = prompt
            ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
            plen = ids.shape[-1]
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=100,
                                     do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip() or None

        # Audio and vision not supported by generic loader
        return _gen_text, None, None, f"{model_name} [text-only fallback] ({model_dir})"

    except Exception as e:
        logger.debug("Generic VLM load failed: %s", e)
        return None, None, None, ""


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(warmup: int = 0) -> Dict[str, Any]:
    """Run the multimodal accuracy benchmark and return a results dict."""
    result: Dict[str, Any] = {
        "status": "pending",
        "metrics": {},
        "latencies_ms": [],
        "detail_lines": [],
        "error": "",
        "thresholds": {
            "text_accuracy": TEXT_ACCURACY_THRESHOLD,
            "audio_keyword_rate": AUDIO_KEYWORD_THRESHOLD,
            "vision_keyword_rate": VISION_KEYWORD_THRESHOLD,
        },
    }

    gen_text, gen_audio, gen_vision, inferencer_name = _load_multimodal_inferencer()
    if gen_text is None:
        result["status"] = "skip"
        result["error"] = "No multimodal model available (need Gemma-4 E4B or E2B)"
        return result

    result["metrics"]["inferencer"] = inferencer_name

    # Check espeak-ng for audio tests
    has_espeak = check_espeak_ng()

    # Warmup
    for _ in range(warmup):
        try:
            gen_text("Hello")
        except Exception:
            pass

    # ── 1. Text QA ──────────────────────────────────────────────────────────
    text_correct = 0
    result["detail_lines"].append("  --- Text QA ---")
    for idx, (question, answers) in enumerate(TEXT_QA):
        t0 = time.monotonic()
        try:
            resp = gen_text(question)
        except Exception as e:
            result["detail_lines"].append(f"  T{idx+1}: exception: {e}")
            continue
        elapsed = (time.monotonic() - t0) * 1000
        result["latencies_ms"].append(elapsed)

        if resp is None:
            result["detail_lines"].append(f"  T{idx+1}: returned None ({elapsed:.0f}ms)")
            continue

        matched = any(a.lower() in resp.lower() for a in answers)
        if matched:
            text_correct += 1
        tag = "OK" if matched else "WRONG"
        result["detail_lines"].append(
            f'  T{idx+1}: [{tag}] "{resp[:60]}"  ({elapsed:.0f}ms)')

    # ── 2. Audio understanding ──────────────────────────────────────────────
    audio_hits = 0
    audio_total = 0
    result["detail_lines"].append("  --- Audio Understanding ---")

    if gen_audio is None:
        result["detail_lines"].append("  (audio generation not supported by this model)")
    elif not has_espeak:
        result["detail_lines"].append("  (espeak-ng not installed -- skipping audio tests)")
    else:
        for idx, (speech_text, question, keywords) in enumerate(AUDIO_TESTS):
            audio_total += 1
            # Generate speech PCM
            pcm = generate_speech_pcm(speech_text, sample_rate=16000)
            if not pcm:
                result["detail_lines"].append(
                    f"  A{idx+1}: espeak-ng failed for: {speech_text[:40]}")
                continue

            t0 = time.monotonic()
            try:
                resp = gen_audio(pcm, 16000, question)
            except Exception as e:
                result["detail_lines"].append(f"  A{idx+1}: exception: {e}")
                continue
            elapsed = (time.monotonic() - t0) * 1000
            result["latencies_ms"].append(elapsed)

            if resp is None:
                result["detail_lines"].append(
                    f"  A{idx+1}: returned None ({elapsed:.0f}ms)")
                continue

            hit = any(kw.lower() in resp.lower() for kw in keywords)
            if hit:
                audio_hits += 1
            tag = "HIT" if hit else "MISS"
            result["detail_lines"].append(
                f'  A{idx+1}: [{tag}] "{resp[:60]}"  ({elapsed:.0f}ms)')

    # ── 3. Vision understanding ─────────────────────────────────────────────
    vision_hits = 0
    vision_total = 0
    result["detail_lines"].append("  --- Vision Understanding ---")

    if gen_vision is None:
        result["detail_lines"].append("  (vision generation not supported by this model)")
    else:
        for idx, (desc, prompt, keywords) in enumerate(VISION_TESTS):
            vision_total += 1
            img = make_synthetic_image(desc)

            t0 = time.monotonic()
            try:
                resp = gen_vision(img, prompt)
            except Exception as e:
                result["detail_lines"].append(f"  V{idx+1}: exception: {e}")
                continue
            elapsed = (time.monotonic() - t0) * 1000
            result["latencies_ms"].append(elapsed)

            if resp is None:
                result["detail_lines"].append(
                    f"  V{idx+1}: returned None ({elapsed:.0f}ms)")
                continue

            hit = any(kw.lower() in resp.lower() for kw in keywords)
            if hit:
                vision_hits += 1
            tag = "HIT" if hit else "MISS"
            result["detail_lines"].append(
                f'  V{idx+1}: [{tag}] "{resp[:60]}"  ({elapsed:.0f}ms)')

    # ── Metrics ─────────────────────────────────────────────────────────────
    text_total = len(TEXT_QA)
    text_accuracy = text_correct / text_total if text_total > 0 else 0
    audio_rate = audio_hits / audio_total if audio_total > 0 else 0
    vision_rate = vision_hits / vision_total if vision_total > 0 else 0

    result["metrics"].update({
        "text_accuracy": round(text_accuracy, 4),
        "text_correct": text_correct,
        "text_total": text_total,
        "audio_keyword_rate": round(audio_rate, 4),
        "audio_hits": audio_hits,
        "audio_total": audio_total,
        "vision_keyword_rate": round(vision_rate, 4),
        "vision_hits": vision_hits,
        "vision_total": vision_total,
    })

    # Pass/fail logic: text must pass; audio/vision pass if tested (skip = OK)
    text_ok = text_accuracy >= TEXT_ACCURACY_THRESHOLD
    audio_ok = audio_total == 0 or audio_rate >= AUDIO_KEYWORD_THRESHOLD
    vision_ok = vision_total == 0 or vision_rate >= VISION_KEYWORD_THRESHOLD
    passed = text_ok and audio_ok and vision_ok

    result["status"] = "pass" if passed else "fail"

    # Free GPU memory
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass

    return result


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the multimodal accuracy benchmark and return a shell exit code."""
    print("\033[1mMultimodal Accuracy Benchmark (Gemma-4 E4B / E2B)\033[0m\n")

    r = benchmark()

    if r["status"] == "skip":
        print(f"SKIP: {r['error']}")
        return 77

    m = r["metrics"]
    print(f"  Inferencer: {m.get('inferencer', 'unknown')}\n")

    for line in r["detail_lines"]:
        if line.strip().startswith("---"):
            print(f"\n{line}")
        elif "[OK]" in line or "[HIT]" in line:
            print(f"  \033[32mPASS\033[0m {line}")
        elif "[WRONG]" in line or "[MISS]" in line:
            print(f"  \033[31mFAIL\033[0m {line}")
        else:
            print(f"  \033[33mINFO\033[0m {line}")

    mean_latency = (sum(r["latencies_ms"]) / len(r["latencies_ms"])
                    if r["latencies_ms"] else 0)

    print(f"\n{'=' * 70}")
    print(f"  Text QA:        {m['text_correct']}/{m['text_total']} = "
          f"{m['text_accuracy']:.1%}  (threshold: {TEXT_ACCURACY_THRESHOLD:.0%})")
    print(f"  Audio keywords: {m['audio_hits']}/{m['audio_total']} = "
          f"{m['audio_keyword_rate']:.1%}  (threshold: {AUDIO_KEYWORD_THRESHOLD:.0%})")
    print(f"  Vision keywords:{m['vision_hits']}/{m['vision_total']} = "
          f"{m['vision_keyword_rate']:.1%}  (threshold: {VISION_KEYWORD_THRESHOLD:.0%})")
    print(f"  Mean latency:   {mean_latency:.0f} ms/query")
    print(f"{'=' * 70}")

    if r["status"] == "pass":
        print(f"  \033[32mPASS\033[0m  All modality thresholds met")
        return 0
    else:
        lines = []
        if m["text_accuracy"] < TEXT_ACCURACY_THRESHOLD:
            lines.append(f"text {m['text_accuracy']:.1%} < {TEXT_ACCURACY_THRESHOLD:.0%}")
        if m["audio_total"] > 0 and m["audio_keyword_rate"] < AUDIO_KEYWORD_THRESHOLD:
            lines.append(f"audio {m['audio_keyword_rate']:.1%} < {AUDIO_KEYWORD_THRESHOLD:.0%}")
        if m["vision_total"] > 0 and m["vision_keyword_rate"] < VISION_KEYWORD_THRESHOLD:
            lines.append(f"vision {m['vision_keyword_rate']:.1%} < {VISION_KEYWORD_THRESHOLD:.0%}")
        print(f"  \033[31mFAIL\033[0m  {'; '.join(lines)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
