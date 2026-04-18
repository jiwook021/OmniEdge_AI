#!/usr/bin/env python3
"""Download face-detection (SCRFD-10G) and face-recognition (AuraFace v1) ONNX models.

Runtime expects:
    {output_dir}/detector.onnx   -- SCRFD-10G (from InsightFace buffalo_l release)
    {output_dir}/recognizer.onnx -- AuraFace v1 (glintr100.onnx, fal/AuraFace-v1, Apache 2.0)

This is an install-time convenience. The C++ inferencer also auto-fetches
the AuraFace recognizer on first loadModel() via fetchHfModel(), so running
this script is optional when network is available at runtime.
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile
import zipfile
from urllib.request import urlretrieve


BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)

AURAFACE_REPO = "fal/AuraFace-v1"
AURAFACE_FILE = "glintr100.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SCRFD-10G detector and AuraFace v1 recognizer ONNX models."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory, e.g. $OE_MODELS_DIR/face_models/scrfd_auraface",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SCRFD detector (from InsightFace buffalo_l release)
# ---------------------------------------------------------------------------

def download_scrfd(output_dir: str, tmpdir: str) -> bool:
    zip_path = os.path.join(tmpdir, "buffalo_l.zip")
    print("[download_face_models] Downloading buffalo_l.zip for SCRFD detector ...")
    urlretrieve(BUFFALO_L_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)

    det_files = glob.glob(os.path.join(tmpdir, "**", "*det*.onnx"), recursive=True)
    if not det_files:
        print("[ERROR] No *det*.onnx found in buffalo_l.zip", file=sys.stderr)
        return False
    dst = os.path.join(output_dir, "detector.onnx")
    shutil.copy2(det_files[0], dst)
    print(f"[download_face_models] SCRFD detector saved: {dst}")
    return True


# ---------------------------------------------------------------------------
# AuraFace v1 recognizer (from fal/AuraFace-v1 on Hugging Face)
# ---------------------------------------------------------------------------

def download_auraface(output_dir: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub",
              file=sys.stderr)
        return False

    try:
        print(f"[download_face_models] Downloading {AURAFACE_FILE} from {AURAFACE_REPO} ...")
        local_path = hf_hub_download(
            repo_id=AURAFACE_REPO,
            filename=AURAFACE_FILE,
        )
    except Exception as exc:
        print(f"[ERROR] AuraFace download failed: {exc}", file=sys.stderr)
        return False

    dst = os.path.join(output_dir, "recognizer.onnx")
    shutil.copy2(local_path, dst)
    print(f"[download_face_models] AuraFace recognizer saved: {dst}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        if not download_scrfd(output_dir, tmpdir):
            print("[ERROR] Failed to download SCRFD detector.", file=sys.stderr)
            ok = False

    if not download_auraface(output_dir):
        print("[ERROR] Failed to download AuraFace recognizer.", file=sys.stderr)
        ok = False

    if ok:
        print(f"[download_face_models] All models saved to: {output_dir}")
        return 0

    print("[download_face_models] Some downloads failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
