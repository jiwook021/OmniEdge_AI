#!/usr/bin/env python3
"""quantize_kokoro_onnx.py — Quantize Kokoro v1.0 ONNX model to INT8.

Applies dynamic INT8 quantization to the Kokoro TTS ONNX model for
reduced VRAM usage (~0.10 GB) via ONNX Runtime CUDA Execution Provider.

Environment variables:
    OE_MODELS_DIR  — Root model directory (default: $HOME/omniedge_models)

Requires: onnxruntime, onnxruntime-gpu
    pip install onnxruntime-gpu
"""

import os
import sys

def main():
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install onnxruntime-gpu")
        sys.exit(1)

    models_dir = os.environ.get("OE_MODELS_DIR", os.path.expanduser("~/omniedge_models"))
    kokoro_dir = os.path.join(models_dir, "kokoro-onnx", "onnx")
    input_model = os.path.join(kokoro_dir, "model.onnx")
    output_model = os.path.join(kokoro_dir, "model_int8.onnx")

    if not os.path.isfile(input_model):
        print(f"ERROR: ONNX model not found at {input_model}")
        print("Download first: huggingface-cli download hexgrad/Kokoro-82M "
              f"--include '*.onnx' --local-dir {kokoro_dir}")
        sys.exit(1)

    if os.path.isfile(output_model):
        print(f"INT8 model already exists at {output_model}, skipping quantization.")
        return

    print(f"Quantizing {input_model} to INT8 ...")
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QInt8,
    )

    print(f"INT8 quantization complete: {output_model}")

    # Validate the quantized model loads
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            output_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        inputs = [i.name for i in sess.get_inputs()]
        outputs = [o.name for o in sess.get_outputs()]
        print(f"Validation OK — Inputs: {inputs}, Outputs: {outputs}")
    except Exception as e:
        print(f"WARNING: Validation failed: {e}")
        print("The quantized model was saved but may not load correctly.")

if __name__ == "__main__":
    main()
