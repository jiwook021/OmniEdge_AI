#!/usr/bin/env python3
"""Check if a TRT-LLM engine was built with the currently installed TensorRT version.

Reads config.json from the engine directory, extracts the TRT version used at
build time, and compares it with the installed tensorrt package version.

Output (stdout):
    "compatible"                                -- versions match
    "incompatible:ENGINE_VER:INSTALLED_VER"     -- version mismatch

Exit codes:
    0 -- compatible
    1 -- incompatible
    2 -- cannot determine (missing files, missing keys, import error)
"""

import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check TRT-LLM engine compatibility with installed TensorRT."
    )
    parser.add_argument(
        "--engine-dir",
        required=True,
        help="Directory containing the engine's config.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = os.path.join(args.engine_dir, "config.json")
    if not os.path.isfile(config_path):
        print(
            f"[ERROR] config.json not found in {args.engine_dir}",
            file=sys.stderr,
        )
        return 2

    # --- Read engine config ----------------------------------------------
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Cannot read config.json: {exc}", file=sys.stderr)
        return 2

    # Navigate to the TRT version field
    engine_trt_version = None
    if "pretrained_config" in config:
        engine_trt_version = config["pretrained_config"].get("trt_version")
    # Some config formats put it at the top level
    if engine_trt_version is None:
        engine_trt_version = config.get("trt_version")
    # Also check build_config
    if engine_trt_version is None and "build_config" in config:
        engine_trt_version = config["build_config"].get("trt_version")

    if engine_trt_version is None:
        print(
            "[ERROR] Cannot find trt_version in config.json "
            "(checked pretrained_config.trt_version, trt_version, build_config.trt_version)",
            file=sys.stderr,
        )
        return 2

    engine_trt_version = str(engine_trt_version)

    # --- Get installed TensorRT version ----------------------------------
    try:
        import tensorrt
        installed_version = tensorrt.__version__
    except ImportError:
        # Try the tensorrt_llm path as an alternative
        try:
            import tensorrt_llm
            installed_version = tensorrt_llm.trt_version()
        except Exception:
            print("[ERROR] Cannot import tensorrt or determine installed version.", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected error importing tensorrt: {exc}", file=sys.stderr)
        return 2

    installed_version = str(installed_version)

    # --- Compare ---------------------------------------------------------
    if engine_trt_version == installed_version:
        print("compatible")
        return 0
    else:
        print(f"incompatible:{engine_trt_version}:{installed_version}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
