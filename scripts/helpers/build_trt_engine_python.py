#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX model using the Python API.

Fallback for when ``trtexec`` is not available on the system.
"""

import argparse
import os
import sys
from typing import Tuple


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    """Parse a shape string like '1x3x640x640' into a tuple of ints."""
    try:
        return tuple(int(d) for d in shape_str.split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{shape_str}'. Expected format: NxCxHxW (e.g. 1x3x640x640)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from an ONNX model."
    )
    parser.add_argument("--onnx", required=True, help="Path to the ONNX model file")
    parser.add_argument("--output", required=True, help="Output engine file path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument(
        "--input-name", default="images", help="Name of the input tensor (default: images)"
    )
    parser.add_argument(
        "--min-shape",
        required=True,
        type=parse_shape,
        help="Minimum input shape, e.g. 1x3x640x640",
    )
    parser.add_argument(
        "--opt-shape",
        required=True,
        type=parse_shape,
        help="Optimal input shape, e.g. 1x3x640x640",
    )
    parser.add_argument(
        "--max-shape",
        required=True,
        type=parse_shape,
        help="Maximum input shape, e.g. 1x3x640x640",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.onnx):
        print(f"[ERROR] ONNX file not found: {args.onnx}", file=sys.stderr)
        return 1

    # --- Import TensorRT ------------------------------------------------
    print("[build_trt_engine] Importing tensorrt ...")
    try:
        import tensorrt as trt
    except ImportError as exc:
        print(f"[ERROR] Cannot import tensorrt: {exc}", file=sys.stderr)
        return 1

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # --- Build network from ONNX ----------------------------------------
    print(f"[build_trt_engine] Parsing ONNX: {args.onnx}")
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[ONNX PARSE ERROR] {parser.get_error(i)}", file=sys.stderr)
            return 1

    print(f"[build_trt_engine] Network inputs: {network.num_inputs}, outputs: {network.num_outputs}")

    # --- Configure builder -----------------------------------------------
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    if args.fp16:
        if not builder.platform_has_fast_fp16:
            print("[WARNING] Platform does not have fast FP16; enabling anyway.", file=sys.stderr)
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build_trt_engine] FP16 enabled")

    # --- Optimization profile (dynamic shapes) ---------------------------
    profile = builder.create_optimization_profile()
    profile.set_shape(
        args.input_name,
        min=args.min_shape,
        opt=args.opt_shape,
        max=args.max_shape,
    )
    config.add_optimization_profile(profile)
    print(
        f"[build_trt_engine] Profile '{args.input_name}': "
        f"min={args.min_shape} opt={args.opt_shape} max={args.max_shape}"
    )

    # --- Build engine ----------------------------------------------------
    print("[build_trt_engine] Building engine (this may take a while) ...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("[ERROR] Engine build failed.", file=sys.stderr)
        return 1

    # --- Write to disk ---------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(serialized_engine)

    size_mb = len(serialized_engine) / (1024 * 1024)
    print(f"[build_trt_engine] Engine saved to: {args.output} ({size_mb:.1f} MiB)")
    print("[build_trt_engine] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
