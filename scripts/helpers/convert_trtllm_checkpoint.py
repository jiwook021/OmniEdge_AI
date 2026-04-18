#!/usr/bin/env python3
"""Convert a HuggingFace model to a TensorRT-LLM checkpoint.

Replaces inline Python blocks that import tensorrt_llm.models and call
from_hugging_face / save_checkpoint.
"""

import argparse
import importlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace model to a TRT-LLM checkpoint."
    )
    parser.add_argument(
        "--model-class",
        required=True,
        help="Model class name, e.g. LLaMAForCausalLM, Phi3ForCausalLM, GemmaForCausalLM",
    )
    parser.add_argument(
        "--hf-dir",
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the TRT-LLM checkpoint will be saved",
    )
    parser.add_argument(
        "--quant-algo",
        default=None,
        help="Quantization algorithm, e.g. W4A16_AWQ, W8A8_SQ_PER_CHANNEL",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Data type for the model (default: float16)",
    )
    parser.add_argument(
        "--use-autoawq",
        action="store_true",
        help="Pass use_autoawq=True to from_hugging_face",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- Import TRT-LLM -------------------------------------------------
    print(f"[convert_trtllm_checkpoint] Importing tensorrt_llm.models ...")
    try:
        models_module = importlib.import_module("tensorrt_llm.models")
    except ImportError as exc:
        print(f"[ERROR] Cannot import tensorrt_llm.models: {exc}", file=sys.stderr)
        return 1

    # --- Resolve model class --------------------------------------------
    model_cls = getattr(models_module, args.model_class, None)
    if model_cls is None:
        print(
            f"[ERROR] Model class '{args.model_class}' not found in tensorrt_llm.models",
            file=sys.stderr,
        )
        return 1
    print(f"[convert_trtllm_checkpoint] Using model class: {args.model_class}")

    # --- Build QuantConfig if requested ---------------------------------
    quant_config = None
    if args.quant_algo:
        print(f"[convert_trtllm_checkpoint] Quantization algo: {args.quant_algo}")
        try:
            from tensorrt_llm.models.modeling_utils import QuantConfig
        except ImportError:
            # Newer TRT-LLM versions moved QuantConfig
            try:
                from tensorrt_llm.quantization import QuantConfig
            except ImportError as exc:
                print(f"[ERROR] Cannot import QuantConfig: {exc}", file=sys.stderr)
                return 1
        quant_config = QuantConfig(quant_algo=args.quant_algo)

    # --- Load from HuggingFace ------------------------------------------
    print(f"[convert_trtllm_checkpoint] Loading from HF dir: {args.hf_dir}")
    hf_kwargs: dict = {"dtype": args.dtype}
    if quant_config is not None:
        hf_kwargs["quant_config"] = quant_config
    if args.use_autoawq:
        hf_kwargs["use_autoawq"] = True
        print("[convert_trtllm_checkpoint] use_autoawq=True")

    try:
        model = model_cls.from_hugging_face(args.hf_dir, **hf_kwargs)
    except Exception as exc:
        print(f"[ERROR] from_hugging_face failed: {exc}", file=sys.stderr)
        return 1

    # --- Save checkpoint ------------------------------------------------
    print(f"[convert_trtllm_checkpoint] Saving checkpoint to: {args.output_dir}")
    try:
        model.save_checkpoint(args.output_dir)
    except Exception as exc:
        print(f"[ERROR] save_checkpoint failed: {exc}", file=sys.stderr)
        return 1

    print("[convert_trtllm_checkpoint] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
