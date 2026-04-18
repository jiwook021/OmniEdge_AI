#!/usr/bin/env python3
"""Export BasicVSR++ to ONNX using mmagic.

Creates an ONNX model with dynamic axes for frames, height, and width
(opset 17).
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BasicVSR++ model to ONNX via mmagic."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX file path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- Import dependencies --------------------------------------------
    print("[export_basicvsrpp] Importing mmagic and torch ...")
    try:
        import torch
    except ImportError as exc:
        print(f"[ERROR] Cannot import torch: {exc}", file=sys.stderr)
        return 1

    try:
        from mmagic.apis import MMagicInferencer
    except ImportError as exc:
        print(f"[ERROR] Cannot import mmagic: {exc}", file=sys.stderr)
        return 1

    # --- Create inferencer and extract model -----------------------------
    print("[export_basicvsrpp] Creating MMagicInferencer('basicvsr_plusplus') ...")
    try:
        inferencer = MMagicInferencer("basicvsr_plusplus")
    except Exception as exc:
        print(f"[ERROR] Failed to create inferencer: {exc}", file=sys.stderr)
        return 1

    model = inferencer.model
    model.eval()

    # --- Prepare dummy input ---------------------------------------------
    # BasicVSR++ expects input of shape (N, T, C, H, W)
    # Use a small dummy for tracing: batch=1, frames=2, 3 channels, 64x64
    dummy_input = torch.randn(1, 2, 3, 64, 64, device="cpu")
    print("[export_basicvsrpp] Dummy input shape: (1, 2, 3, 64, 64)")

    # Move model to CPU for export
    model = model.cpu()

    # --- Export to ONNX --------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dynamic_axes = {
        "input": {
            0: "batch_size",
            1: "num_frames",
            3: "height",
            4: "width",
        },
        "output": {
            0: "batch_size",
            1: "num_frames",
            3: "height",
            4: "width",
        },
    }

    print(f"[export_basicvsrpp] Exporting to ONNX (opset 17): {args.output}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    except Exception as exc:
        print(f"[ERROR] ONNX export failed: {exc}", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[export_basicvsrpp] Exported: {args.output} ({size_mb:.1f} MiB)")
    print("[export_basicvsrpp] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
