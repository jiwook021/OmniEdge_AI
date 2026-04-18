#!/usr/bin/env python3
"""OmniEdge_AI -- LLM Accuracy Benchmark (Qwen 2.5 7B)

Evaluates the quantised TRT-LLM engine against a curated question set.
Tests factual recall, arithmetic, reasoning, and instruction following.

Metrics:
  - QA accuracy: fraction of questions answered correctly (target >= 0.75)
  - Mean response length: sanity check for degenerate (empty/looping) outputs

Model loading fallback chain:
  1. TRT-LLM Python API (LLM() with engine)
  2. HuggingFace transformers AutoModelForCausalLM (float16, device_map="auto")
  3. Exit 77 if neither inferencer is available

Exit codes: 0 = all thresholds met, 1 = below threshold, 77 = skipped
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    resolve_engine_path,
    resolve_model_path,
    setup_logging,
    TestReporter,
)

logger = setup_logging(__name__)

# -- Accuracy thresholds ----------------------------------------------------
QA_ACCURACY_THRESHOLD: float = 0.75
MIN_MEAN_RESPONSE_LEN: int = 1
MAX_MEAN_RESPONSE_LEN: int = 2000

# -- Curated QA dataset (shared with benchmark_all.py) ----------------------
# Each entry: (question, list of acceptable substrings in lowercase answer)
QA_DATASET: List[Tuple[str, List[str]]] = [
    # Arithmetic
    ("What is 2+2? Reply with just the number.", ["4"]),
    ("What is 15 multiplied by 3? Reply with just the number.", ["45"]),
    ("What is 100 divided by 4? Reply with just the number.", ["25"]),
    # Factual recall -- geography
    ("What is the capital of France? Reply with just the city name.", ["paris"]),
    ("What is the capital of Japan? Reply with just the city name.", ["tokyo"]),
    ("What is the largest ocean on Earth? Reply in one word.", ["pacific"]),
    # Factual recall -- science
    ("What is the chemical symbol for water? Reply with just the formula.", ["h2o"]),
    ("How many planets are in our solar system? Reply with just the number.", ["8"]),
    (
        "What is the speed of light in km/s approximately? Reply with just the number.",
        ["300000", "299792", "3\u00d710", "3x10", "300,000"],
    ),
    # Factual recall -- general knowledge
    ("Who wrote Romeo and Juliet? Reply with just the name.", ["shakespeare"]),
    ("What year did World War II end? Reply with just the year.", ["1945"]),
    ("What is the smallest prime number? Reply with just the number.", ["2"]),
    # Reasoning
    (
        "If a shirt costs $25 and is 20% off, what is the sale price? "
        "Reply with just the dollar amount.",
        ["20", "$20"],
    ),
    (
        "What comes next in the sequence: 2, 4, 8, 16, ...? "
        "Reply with just the number.",
        ["32"],
    ),
    # Instruction following
    ("List three primary colors separated by commas.", ["red", "blue", "yellow"]),
    ("Say hello in exactly one word.", ["hello", "hi", "hey"]),
    # Language understanding
    (
        "Is the following statement true or false: The sun rises in the west. "
        "Reply with just true or false.",
        ["false"],
    ),
    ("What is the opposite of hot? Reply in one word.", ["cold"]),
    # Code/logic
    (
        "What does 'print(3>2)' output in Python? Reply with just the output.",
        ["true"],
    ),
    ("How many bits are in a byte? Reply with just the number.", ["8"]),
]


def _load_inferencer():
    """Try TRT-LLM first, then fall back to transformers.

    Returns (generate_fn, inferencer_name) or (None, "").
    """
    hf_dir: Optional[str] = None
    for name in ["qwen2.5-7b-instruct", "qwen2.5-7b-instruct-awq"]:
        c = resolve_model_path(name)
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "tokenizer.json")):
            hf_dir = c
            break

    engine_dir: Optional[str] = None
    for name in ["qwen2.5-7b-nvfp4", "qwen2.5-7b-awq"]:
        c = resolve_engine_path(name)
        if os.path.isdir(c):
            engine_dir = c
            break

    # 1) TRT-LLM
    if engine_dir and hf_dir:
        try:
            from tensorrt_llm import LLM as TrtLLM, SamplingParams
            try:
                llm_inst = TrtLLM(model=engine_dir, tokenizer=hf_dir)
            except TypeError:
                llm_inst = TrtLLM(model=engine_dir, tokenizer=hf_dir,
                                   skip_tokenizer_init=False)

            def _gen(prompt: str) -> Optional[str]:
                params = SamplingParams(max_tokens=50, temperature=0.0)
                outs = llm_inst.generate([prompt], sampling_params=params)
                if not outs:
                    return None
                o = outs[0]
                if hasattr(o, "outputs") and o.outputs:
                    inner = o.outputs[0]
                    if inner and hasattr(inner, "text") and inner.text:
                        return inner.text.strip()
                if hasattr(o, "text") and o.text:
                    return o.text.strip()
                return None

            return _gen, f"TRT-LLM ({engine_dir})"
        except (ImportError, Exception):
            pass

    # 2) HuggingFace transformers
    if hf_dir:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                hf_dir, dtype=torch.float16, device_map="auto",
                trust_remote_code=True,
            )
            mdl.eval()

            def _gen(prompt: str) -> Optional[str]:
                inp = tok(prompt, return_tensors="pt")
                ids = inp["input_ids"].to(mdl.device)
                plen = ids.shape[-1]
                with torch.no_grad():
                    out = mdl.generate(ids, max_new_tokens=50, temperature=0.0,
                                       do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0][plen:], skip_special_tokens=True)
                return text.strip() if text else None

            return _gen, f"transformers ({hf_dir})"
        except (ImportError, Exception):
            pass

    return None, ""


def benchmark(warmup: int = 0):
    """Run the LLM accuracy benchmark and return a results dict."""
    result = {
        "status": "pending",
        "metrics": {},
        "latencies_ms": [],
        "detail_lines": [],
        "error": "",
        "thresholds": {
            "accuracy": QA_ACCURACY_THRESHOLD,
            "resp_len_min": MIN_MEAN_RESPONSE_LEN,
            "resp_len_max": MAX_MEAN_RESPONSE_LEN,
        },
    }

    generate_fn, inferencer_name = _load_inferencer()
    if generate_fn is None:
        result["status"] = "skip"
        result["error"] = "No LLM inferencer available"
        return result

    result["metrics"]["inferencer"] = inferencer_name

    # Warmup
    for _ in range(warmup):
        try:
            generate_fn("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        except Exception:
            pass

    correct = 0
    resp_lens: List[int] = []

    for idx, (question, answers) in enumerate(QA_DATASET):
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant. Answer concisely and precisely.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        t0 = time.monotonic()
        try:
            resp = generate_fn(prompt)
        except Exception as e:
            result["detail_lines"].append(f"  Q{idx+1}: exception: {e}")
            resp_lens.append(0)
            continue
        elapsed = (time.monotonic() - t0) * 1000
        result["latencies_ms"].append(elapsed)

        if resp is None:
            result["detail_lines"].append(f"  Q{idx+1}: returned None")
            resp_lens.append(0)
            continue

        resp_lens.append(len(resp))
        matched = any(a.lower() in resp.lower() for a in answers)
        if matched:
            correct += 1
        tag = "OK" if matched else "WRONG"
        result["detail_lines"].append(
            f'  Q{idx+1}: [{tag}] "{resp[:50]}"  ({elapsed:.0f}ms)'
        )

    total = len(QA_DATASET)
    accuracy = correct / total if total > 0 else 0
    mean_len = sum(resp_lens) / len(resp_lens) if resp_lens else 0

    result["metrics"].update({
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "mean_response_len": round(mean_len, 1),
    })

    passed = (accuracy >= QA_ACCURACY_THRESHOLD
              and MIN_MEAN_RESPONSE_LEN <= mean_len <= MAX_MEAN_RESPONSE_LEN)
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


def main() -> int:
    """Run the LLM accuracy benchmark and return a shell exit code."""
    print("\033[1mLLM Accuracy Benchmark (Qwen 2.5 7B)\033[0m\n")

    r = benchmark()

    if r["status"] == "skip":
        print(f"SKIP: {r['error']}")
        return 77

    m = r["metrics"]
    print(f"  Inferencer: {m.get('inferencer', 'unknown')}\n")

    for line in r["detail_lines"]:
        is_ok = "[OK]" in line
        status = "\033[32mPASS\033[0m" if is_ok else "\033[31mFAIL\033[0m"
        print(f"  {status} {line}")

    mean_latency = sum(r["latencies_ms"]) / len(r["latencies_ms"]) if r["latencies_ms"] else 0

    print(f"\n{'=' * 60}")
    print(f"  QA Accuracy:         {m['correct']}/{m['total']} = {m['accuracy']:.1%}")
    print(f"  Accuracy threshold:  {QA_ACCURACY_THRESHOLD:.0%}")
    print(f"  Mean response len:   {m['mean_response_len']:.0f} chars")
    print(f"  Mean latency:        {mean_latency:.0f} ms/question")
    print(f"{'=' * 60}")

    if r["status"] == "pass":
        print(f"  \033[32mPASS\033[0m  QA accuracy {m['accuracy']:.1%} >= {QA_ACCURACY_THRESHOLD:.0%}")
        return 0
    else:
        print(f"  \033[31mFAIL\033[0m  QA accuracy {m['accuracy']:.1%} < {QA_ACCURACY_THRESHOLD:.0%}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
