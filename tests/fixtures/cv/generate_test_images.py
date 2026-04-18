#!/usr/bin/env python3
"""Generate test fixture images for OmniEdge_AI CV unit tests.

Creates both JPEG (.jpg) and raw BGR24 (.bgr24) files for use in C++ tests.
The BGR24 files can be loaded directly into uint8_t buffers without any
image decoding library.

Face fixtures use real photographs from the Labeled Faces in the Wild (LFW)
dataset via scikit-learn, ensuring that neural face detectors (SCRFD) can
actually detect them.  Scene fixtures (person, empty room) use synthetic
generation since they only need to exercise blur/segmentation pipelines.

Regenerate:  python3 tests/fixtures/cv/generate_test_images.py

Dependencies: pip install scikit-learn opencv-python numpy Pillow
"""

from pathlib import Path

import cv2
import numpy as np

FIXTURE_DIR = Path(__file__).parent

# All test images use 640x480 to keep fixture sizes small (~900 KB BGR24)
WIDTH, HEIGHT = 640, 480
BGR_BYTES = WIDTH * HEIGHT * 3


# ---------------------------------------------------------------------------
# Synthetic scene fixtures (blur / segmentation tests)
# ---------------------------------------------------------------------------

def generate_person_scene() -> np.ndarray:
    """A scene with a 'person' (skin-tone rectangle) on a blue background.

    This simulates what the blur inferencer expects: a person in a room.
    The person region is centered and occupies ~30% of the frame.
    """
    bgr = np.full((HEIGHT, WIDTH, 3), fill_value=0, dtype=np.uint8)
    # Blue-ish room background
    bgr[:, :, 0] = 180  # B
    bgr[:, :, 1] = 130  # G
    bgr[:, :, 2] = 100  # R

    # Person region: skin-tone rectangle in center
    person_top, person_bottom = 80, 400
    person_left, person_right = 220, 420
    bgr[person_top:person_bottom, person_left:person_right, 0] = 140  # B
    bgr[person_top:person_bottom, person_left:person_right, 1] = 170  # G
    bgr[person_top:person_bottom, person_left:person_right, 2] = 210  # R

    return bgr


def generate_empty_room() -> np.ndarray:
    """A plain room scene with no person — gradient background.

    Tests that the blur inferencer handles frames with no person detected.
    """
    bgr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    # Horizontal gradient: darker on left, lighter on right
    for x in range(WIDTH):
        intensity = int(80 + (x / WIDTH) * 120)
        bgr[:, x, 0] = min(intensity + 20, 255)  # B
        bgr[:, x, 1] = intensity                   # G
        bgr[:, x, 2] = max(intensity - 10, 0)      # R
    return bgr


# ---------------------------------------------------------------------------
# Real face fixtures (face detection / recognition tests)
# ---------------------------------------------------------------------------

def _lfw_to_canvas(face_rgb_float: np.ndarray, height_fraction: float = 0.65,
                   bg_value: int = 140) -> np.ndarray:
    """Place a LFW color portrait (125x94, float [0,1]) on a 640x480 canvas."""
    face_rgb = np.clip(face_rgb_float * 255, 0, 255).astype(np.uint8)
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    fh, fw = face_bgr.shape[:2]

    scale = (HEIGHT * height_fraction) / fh
    new_w = int(fw * scale)
    new_h = int(fh * scale)
    face_resized = cv2.resize(face_bgr, (new_w, new_h),
                              interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((HEIGHT, WIDTH, 3), bg_value, dtype=np.uint8)
    y_off = (HEIGHT - new_h) // 2
    x_off = (WIDTH - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = face_resized
    return canvas


def generate_face_fixtures() -> dict:
    """Generate face fixtures from LFW public-domain portraits.

    Returns a dict of {filename_stem: bgr_array}.
    """
    from sklearn.datasets import fetch_lfw_people

    lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)
    names = lfw.target_names.tolist()

    # Pick three distinct, well-known people
    person_a = names.index("Colin Powell")
    person_b = names.index("Jennifer Aniston")
    person_c = names.index("Serena Williams")

    idx_a = int(np.where(lfw.target == person_a)[0][0])
    idx_b = int(np.where(lfw.target == person_b)[0][0])
    idx_c = int(np.where(lfw.target == person_c)[0][0])

    print(f"  Alice: {names[person_a]}")
    print(f"  Bob:   {names[person_b]}")
    print(f"  Group: {names[person_a]}, {names[person_b]}, {names[person_c]}")

    # Single-face portraits
    alice_bgr = _lfw_to_canvas(lfw.images[idx_a])
    bob_bgr = _lfw_to_canvas(lfw.images[idx_b])

    # Group: 3 faces side by side
    group_bgr = np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)
    for cx, idx in zip([160, 320, 480], [idx_a, idx_b, idx_c]):
        face_rgb = np.clip(lfw.images[idx] * 255, 0, 255).astype(np.uint8)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        fh, fw = face_bgr.shape[:2]
        scale = (HEIGHT * 0.45) / fh
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(face_bgr, (nw, nh),
                             interpolation=cv2.INTER_LANCZOS4)
        y1 = (HEIGHT - nh) // 2
        x1 = cx - nw // 2
        y2, x2 = y1 + nh, x1 + nw
        cy1, cy2 = max(0, y1), min(HEIGHT, y2)
        cx1, cx2 = max(0, x1), min(WIDTH, x2)
        group_bgr[cy1:cy2, cx1:cx2] = resized[cy1 - y1:cy1 - y1 + cy2 - cy1,
                                               cx1 - x1:cx1 - x1 + cx2 - cx1]

    return {
        "face_alice_640x480": alice_bgr,
        "face_bob_640x480": bob_bgr,
        "group_three_faces_640x480": group_bgr,
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_bgr24(path: Path, bgr: np.ndarray) -> None:
    """Save a HxWx3 uint8 BGR array as flat binary."""
    assert bgr.shape == (HEIGHT, WIDTH, 3) and bgr.dtype == np.uint8
    bgr.tofile(path)


def save_jpg(path: Path, bgr: np.ndarray, quality: int = 85) -> None:
    """Save BGR array as JPEG via OpenCV."""
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Synthetic scene fixtures
    synthetic = {
        "person_scene_640x480": generate_person_scene,
        "empty_room_640x480": generate_empty_room,
    }

    for name, gen_fn in synthetic.items():
        bgr = gen_fn()
        bgr_path = FIXTURE_DIR / f"{name}.bgr24"
        jpg_path = FIXTURE_DIR / f"{name}.jpg"
        save_bgr24(bgr_path, bgr)
        save_jpg(jpg_path, bgr)
        print(f"  {bgr_path.name:40s} {bgr_path.stat().st_size:>10,} bytes")
        print(f"  {jpg_path.name:40s} {jpg_path.stat().st_size:>10,} bytes")

    # Real face fixtures from LFW
    print("Downloading LFW faces (first run may take a minute)...")
    face_fixtures = generate_face_fixtures()
    for name, bgr in face_fixtures.items():
        bgr_path = FIXTURE_DIR / f"{name}.bgr24"
        jpg_path = FIXTURE_DIR / f"{name}.jpg"
        save_bgr24(bgr_path, bgr)
        save_jpg(jpg_path, bgr)
        print(f"  {bgr_path.name:40s} {bgr_path.stat().st_size:>10,} bytes")
        print(f"  {jpg_path.name:40s} {jpg_path.stat().st_size:>10,} bytes")

    # Metadata
    meta_path = FIXTURE_DIR / "fixture_meta.txt"
    meta_path.write_text(
        f"width={WIDTH}\n"
        f"height={HEIGHT}\n"
        f"bgr_bytes={BGR_BYTES}\n"
    )
    print(f"  {meta_path.name}")


if __name__ == "__main__":
    main()
