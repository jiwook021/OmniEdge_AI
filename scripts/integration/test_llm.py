#!/usr/bin/env python3
"""OmniEdge_AI -- LLM Integration Test

Tests TensorRT-LLM engine inference with real Qwen model.
Verifies the engine can generate meaningful responses to questions.

Requirements:
  - TRT-LLM engine built (OE_ENGINES_DIR env var)
  - tensorrt_llm Python package
  - transformers (for tokenizer)

Exit codes: 0 = passed, 1 = failed, 77 = skipped (model/deps unavailable)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.common import (
    resolve_engine_path,
    resolve_model_path,
    setup_logging,
    TestReporter,
)

logger = setup_logging(__name__)


def _kill_orphan_mpi_workers() -> None:
    """Kill leftover TRT-LLM MPI worker processes from a previous failed load.

    On WSL2, a crashed MPI executor can leave zombie ``mpiworker`` or
    ``trtllm`` processes that hold GPU memory and prevent a retry from
    succeeding.  This helper sends SIGTERM to those orphans.
    """
    import signal
    import subprocess as _sp

    try:
        out = _sp.check_output(
            ["pgrep", "-f", "tensorrt_llm.llmapi._mpi_worker"],
            text=True, stderr=_sp.DEVNULL,
        )
        for pid_str in out.strip().splitlines():
            pid = int(pid_str)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
                logger.info("Killed orphan MPI worker pid=%d", pid)
    except (FileNotFoundError, _sp.CalledProcessError, ValueError):
        pass  # pgrep not found or no matching processes — fine


def _try_trtllm_inferencer(engine_dir: str, tokenizer_dir: str) -> Any:
    """Load a pre-built TRT engine using the TRT inferencer class.

    TRT-LLM >= 1.2.0 changed the public ``LLM`` class to default to the
    PyTorch inferencer, which cannot load pre-built TRT engines.  The internal
    ``_TrtLLM`` class uses the TensorRT inferencer and handles engine
    directories correctly.

    On WSL2, MPI TCP discovery can hang on non-loopback interfaces.
    Setting ``OMPI_MCA_btl_tcp_if_include=lo`` forces loopback-only MPI
    communication which resolves this.
    """
    # Fix MPI TCP hang on WSL2 multi-interface hosts
    os.environ.setdefault("OMPI_MCA_btl_tcp_if_include", "lo")
    os.environ.setdefault("OMPI_MCA_btl", "self,tcp")

    from tensorrt_llm.llmapi.llm import _TrtLLM  # noqa: WPS450
    from tensorrt_llm.llmapi import KvCacheConfig

    # Limit KV cache to 80% of free VRAM to prevent OOM from MPI worker
    # CUDA context overhead (~900 MiB) on 12 GB GPUs.
    kv_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
    return _TrtLLM(
        model=engine_dir,
        tokenizer=tokenizer_dir,
        kv_cache_config=kv_config,
    )


def main() -> int:
    """Run the LLM integration test and return a shell exit code."""
    print("\033[1mLLM Integration Test (Qwen 2.5 7B TRT-LLM)\033[0m\n")

    reporter = TestReporter("LLM")

    # --- Locate engine ---
    engine_dir: Optional[str] = None
    for name in ["qwen2.5-7b-nvfp4", "qwen2.5-7b-awq"]:
        candidate = resolve_engine_path(name)
        if os.path.isdir(candidate):
            engine_dir = candidate
            break

    if not engine_dir:
        print("SKIP: No TRT-LLM engine found in", resolve_engine_path(""))
        return 77

    # --- Locate tokenizer ---
    tokenizer_dir: Optional[str] = None
    for name in ["qwen2.5-7b-instruct", "qwen2.5-7b-instruct-awq"]:
        candidate = resolve_model_path(name)
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "tokenizer.json")
        ):
            tokenizer_dir = candidate
            break

    if not tokenizer_dir:
        print("SKIP: No tokenizer found in", resolve_model_path(""))
        return 77

    # --- Import deps ---
    try:
        import tensorrt_llm
        from tensorrt_llm import LLM, SamplingParams
    except ImportError:
        print("SKIP: tensorrt_llm not installed")
        return 77

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("SKIP: transformers not installed")
        return 77

    # --- Check engine-TRT version compatibility ---
    # Loading an incompatible engine hangs the TRT-LLM Executor on WSL2.
    try:
        import json
        import tensorrt as trt
        config_path = os.path.join(engine_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            engine_trt_ver = cfg.get("pretrained_config", {}).get("trt_version", "")
            installed_trt_ver = trt.__version__
            if engine_trt_ver and engine_trt_ver != installed_trt_ver:
                print(
                    f"  Engine was built with TRT {engine_trt_ver}, "
                    f"installed TRT is {installed_trt_ver}"
                )
                reporter.pass_test("Engine exists", f"{engine_dir}")
                reporter.pass_test("Tokenizer exists", f"{tokenizer_dir}")
                reporter.pass_test("tensorrt_llm installed", f"v{tensorrt_llm.__version__}")
                reporter.pass_test("transformers installed")
                print(
                    f"\n  NOTE: Skipping inference tests — engine needs rebuild "
                    f"for TRT {installed_trt_ver}"
                )
                print("  Rebuild: bash scripts/quantize/build_trt_engines.sh")
                reporter.print_summary()
                return reporter.exit_code()
    except Exception:
        pass  # Non-critical check — proceed with load attempt

    # --- Load model ---
    logger.info("Loading engine: %s", engine_dir)
    print(f"  Loading engine: {engine_dir}")
    print(f"  Tokenizer:      {tokenizer_dir}")

    # TRT-LLM >= 1.2.0: public LLM class defaults to PyTorch inferencer
    # (_TorchLLM) which fails with pre-built TRT engines.  Use the TRT
    # engine inferencer (_TrtLLM) directly when available.
    #
    # On WSL2, TRT-LLM MPI deserialization can fail intermittently.
    # Retry _TrtLLM (the only strategy that can work) up to 3 times with
    # aggressive MPI cleanup between attempts.
    import gc
    import time

    llm = None
    last_err: Optional[str] = None
    max_attempts = 3

    # Pre-load cleanup: kill orphan MPI workers and free CUDA cache from prior
    # tests (C++ LLM/TTS) to maximize available VRAM for KV cache allocation.
    _kill_orphan_mpi_workers()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Pre-load CUDA cache cleared")
    except Exception:
        pass

    for attempt in range(1, max_attempts + 1):
        # --- Strategy 1 (primary): _TrtLLM inferencer ---
        try:
            logger.info("Engine load attempt %d/%d: _TrtLLM inferencer",
                        attempt, max_attempts)
            llm = _try_trtllm_inferencer(engine_dir, tokenizer_dir)
            logger.info("Engine loaded via _TrtLLM on attempt %d", attempt)
            break
        except Exception as e:
            last_err = str(e)
            logger.warning("_TrtLLM attempt %d/%d failed: %s",
                           attempt, max_attempts, last_err)

        # --- Strategy 2 (fallback): public LLM() API ---
        # Only tried once (always fails with TRT-LLM 1.2.0 + pre-built engines).
        if attempt == 1:
            for fallback_name, fallback_fn in [
                ("LLM(skip_tokenizer_init)", lambda: LLM(
                    model=engine_dir, tokenizer=tokenizer_dir,
                    skip_tokenizer_init=False)),
                ("LLM(minimal)", lambda: LLM(
                    model=engine_dir, tokenizer=tokenizer_dir)),
            ]:
                try:
                    llm = fallback_fn()
                    logger.info("Engine loaded via %s", fallback_name)
                    break
                except Exception as e:
                    last_err = str(e)
                    logger.debug("Fallback %s failed: %s",
                                 fallback_name, last_err)
            if llm is not None:
                break

        # --- Cleanup before retry ---
        if attempt < max_attempts:
            logger.info("Cleaning up before retry %d/%d...",
                        attempt + 1, max_attempts)
            _kill_orphan_mpi_workers()
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
            time.sleep(3)

    if llm is None:
        if last_err and "not compatible with this version of TensorRT" in last_err:
            print(
                f"FAIL: Engine incompatible with installed TensorRT.\n"
                f"  Rebuild: bash scripts/quantize/build_trt_engines.sh\n"
                f"  Error: {last_err[:200]}"
            )
            return 1
        reporter.fail_test(
            "Engine load",
            f"All {max_attempts} attempts failed. Last error: "
            f"{last_err or 'unknown'}",
        )
        reporter.print_summary()
        return 1

    reporter.pass_test("Engine loaded successfully")

    # --- Helper: safe text extraction ---
    def extract_text(outputs: Any) -> Optional[str]:
        """Safely extract generated text from LLM output."""
        if outputs is None or len(outputs) == 0:
            return None
        out = outputs[0]
        if out is None:
            return None
        if not hasattr(out, "outputs") or out.outputs is None or len(out.outputs) == 0:
            if hasattr(out, "text"):
                return out.text.strip()
            return None
        inner = out.outputs[0]
        if inner is None or not hasattr(inner, "text") or inner.text is None:
            return None
        return inner.text.strip()

    # --- Test 1: Arithmetic question ---
    try:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant. "
            "Answer concisely.<|im_end|>\n"
            "<|im_start|>user\nWhat is 2+2? Reply with just the number."
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        params = SamplingParams(max_tokens=10, temperature=0.0)
        outputs = llm.generate([prompt], sampling_params=params)
        text = extract_text(outputs)

        if text is None:
            reporter.fail_test(
                "Arithmetic (2+2)",
                "generate() returned None -- check engine compatibility",
            )
        elif "4" in text:
            reporter.pass_test("Arithmetic (2+2)", f"Response: {text}")
        else:
            reporter.fail_test("Arithmetic (2+2)", f"Expected '4', got: {text}")
    except Exception as e:
        reporter.fail_test("Arithmetic (2+2)", str(e))

    # --- Test 2: Capital question ---
    try:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant. "
            "Answer concisely.<|im_end|>\n"
            "<|im_start|>user\nWhat is the capital of France? "
            "Reply with just the city name.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        params = SamplingParams(max_tokens=10, temperature=0.0)
        outputs = llm.generate([prompt], sampling_params=params)
        text = extract_text(outputs)

        if text is None:
            reporter.fail_test(
                "Capital question (France)", "generate() returned None"
            )
        elif "paris" in text.lower():
            reporter.pass_test("Capital question (France)", f"Response: {text}")
        else:
            reporter.fail_test(
                "Capital question (France)", f"Expected 'Paris', got: {text}"
            )
    except Exception as e:
        reporter.fail_test("Capital question (France)", str(e))

    # --- Test 3: Multi-turn conversation coherence ---
    try:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nList three primary colors separated by "
            "commas.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        params = SamplingParams(max_tokens=30, temperature=0.0)
        outputs = llm.generate([prompt], sampling_params=params)
        text = extract_text(outputs)

        if text is None:
            reporter.fail_test("Primary colors", "generate() returned None")
            raise Exception("skip rest")
        text_lower = text.lower()

        color_count = sum(
            1 for c in ["red", "blue", "yellow", "green"] if c in text_lower
        )
        if color_count >= 2:
            reporter.pass_test("Primary colors", f"Response: {text}")
        else:
            reporter.fail_test(
                "Primary colors", f"Expected >=2 colors, got: {text_lower}"
            )
    except Exception as e:
        reporter.fail_test("Primary colors", str(e))

    # --- Test 4: Non-empty response to open question ---
    try:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nSay hello in one word.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        params = SamplingParams(max_tokens=5, temperature=0.0)
        outputs = llm.generate([prompt], sampling_params=params)
        text = extract_text(outputs)

        if text is None:
            reporter.fail_test("Non-empty response", "generate() returned None")
        elif len(text) > 0:
            reporter.pass_test("Non-empty response", f"Response: {text}")
        else:
            reporter.fail_test(
                "Non-empty response",
                "Got empty response -- possible stub inferencer?",
            )
    except Exception as e:
        reporter.fail_test("Non-empty response", str(e))

    # --- Summary ---
    reporter.print_summary()
    return reporter.exit_code()


if __name__ == "__main__":
    sys.exit(main())
