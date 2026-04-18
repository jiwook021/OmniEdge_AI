#!/usr/bin/env python3
"""Export BasicVSR to ONNX for OmniEdge_AI video denoising.

Usage:
    python3 scripts/export/export_basicvsrpp.py [--output DIR]

Output:
    <output>/basicvsrpp_denoise.onnx   — fixed-shape model (1, 5, 3, 270, 480)

Requires:
    pip install basicsr torch

The exported model accepts [1, 5, 3, 270, 480] float32 RGB input (normalized
to [0,1]) and produces [1, 5, 3, 1080, 1920] float32 RGB output (4x super-res).
The C++ inferencer (onnx_basicvsrpp_inferencer.cpp) extracts the center frame from
the temporal window.
"""

import argparse
import os
import sys
import types


def _patch_torchvision_compat() -> None:
    """Patch basicsr's broken import of removed torchvision internal module."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        import torchvision.transforms.functional as F
        fake = types.ModuleType("torchvision.transforms.functional_tensor")
        fake.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = fake


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BasicVSR to ONNX")
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/omniedge_models/basicvsrpp"),
        help="Output directory (default: ~/omniedge_models/basicvsrpp)",
    )
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "basicvsrpp_denoise.onnx")

    if os.path.isfile(output_path):
        print(f"Already exists: {output_path}")
        return

    import torch

    _patch_torchvision_compat()

    from basicsr.archs.basicvsr_arch import BasicVSR

    print("Creating BasicVSR model (num_feat=64, num_block=30)...")
    model = BasicVSR(num_feat=64, num_block=30).eval().cuda()

    # Fixed shapes: 5 frames at quarter-HD (270x480), matching temporal_window=5
    # in omniedge_config.yaml. The model internally handles SpyNet optical flow
    # which uses hardcoded resize steps, making dynamic export unreliable.
    dummy = torch.randn(1, 5, 3, 270, 480).cuda()

    print(f"Exporting to ONNX: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )

    if os.path.isfile(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"OK: BasicVSR ONNX exported ({size_mb:.1f} MB): {output_path}")
    else:
        print("ERROR: Export completed but output file not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
