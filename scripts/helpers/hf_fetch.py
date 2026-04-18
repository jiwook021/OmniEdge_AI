#!/usr/bin/env python3
"""Fetch a single file from Hugging Face Hub into a cache directory.

Idempotent: if the file already exists under cache_dir, prints its path and
exits 0 without contacting the network. On success prints the absolute local
path to stdout; on failure prints a diagnostic to stderr and exits non-zero.

Used by modules/common/src/hf_model_fetcher.cpp to provide runtime model
auto-download for AuraFace, Gemma-4, Whisper, and MediaPipe Selfie Seg.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a single file from HF Hub")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. fal/AuraFace-v1")
    ap.add_argument("--file", required=True, help="Filename inside the repo")
    ap.add_argument("--cache-dir", required=True, help="Destination directory")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    target = cache_dir / args.file
    if target.is_file():
        print(str(target.resolve()))
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(f"huggingface_hub not installed: {exc}", file=sys.stderr)
        return 2

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_path = hf_hub_download(
            repo_id=args.repo,
            filename=args.file,
            local_dir=str(cache_dir),
        )
    except Exception as exc:
        print(f"hf_hub_download('{args.repo}', '{args.file}') failed: {exc}",
              file=sys.stderr)
        return 3

    print(str(Path(local_path).resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
