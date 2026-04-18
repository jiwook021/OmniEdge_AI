#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — CV installation (MediaPipe Selfie Seg + Face Recognition)
# ═══════════════════════════════════════════════════════════════════════════════
[[ -n "${_OE_INSTALL_CV_SOURCED:-}" ]] && return 0
readonly _OE_INSTALL_CV_SOURCED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_memory.sh"

# ═══════════════════════════════════════════════════════════════════════════════
#  5. CV — MediaPipe Selfie Seg + Face Recognition
# ═══════════════════════════════════════════════════════════════════════════════
install_cv() {
    section "CV — MediaPipe Selfie Seg + Face Recognition"

    # MediaPipe Selfie Segmentation ONNX (~462 KB, Apache 2.0) — used by bg_blur.
    # Runtime path also auto-downloads from HuggingFace via fetchHfModel; this
    # step is a warm-cache optimization.
    local SELFIE_SEG_DIR="$OE_MODELS_DIR/bg_blur"
    local SELFIE_SEG_ONNX="$SELFIE_SEG_DIR/mediapipe_selfie_seg.onnx"
    if [ -f "$SELFIE_SEG_ONNX" ]; then
        skip "MediaPipe Selfie Seg ONNX"
    else
        echo "  Downloading MediaPipe Selfie Segmentation ONNX..."
        mkdir -p "$SELFIE_SEG_DIR"
        curl -fsSL -o "$SELFIE_SEG_ONNX" \
            "https://huggingface.co/onnx-community/mediapipe_selfie_segmentation/resolve/main/onnx/model.onnx" \
            && ok "MediaPipe Selfie Seg ONNX" \
            || warn "MediaPipe Selfie Seg download failed — runtime fetchHfModel() will retry"
    fi

    # FaceMesh V2 ONNX model (for face filter AR + beauty — ~2.4 MB FP16)
    # Path must match ini_config.hpp BeautyConfig::modelPath and
    # FaceFilterConfig::modelPath: "facemesh/face_landmarks_detector.onnx".
    local FACEMESH_DIR="$OE_MODELS_DIR/facemesh"
    local FACEMESH_ONNX="$FACEMESH_DIR/face_landmarks_detector.onnx"
    if [ -f "$FACEMESH_ONNX" ]; then
        skip "FaceMesh V2 ONNX model"
    else
        echo "  Downloading MediaPipe FaceMesh V2 ONNX model..."
        mkdir -p "$FACEMESH_DIR"

        # MediaPipe distributes FaceMesh as a .tflite — we convert via tf2onnx.
        # If the user has a pre-exported ONNX, skip the conversion.
        local FACEMESH_TFLITE="$FACEMESH_DIR/face_landmark.tflite"
        if [ ! -f "$FACEMESH_TFLITE" ]; then
            curl -fsSL -o "$FACEMESH_TFLITE" \
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
                || warn "FaceMesh download failed — face filter will use stub"
        fi

        if [ -f "$FACEMESH_TFLITE" ]; then
            if python3 -c "import tf2onnx" 2>/dev/null; then
                python3 -m tf2onnx.convert \
                    --tflite "$FACEMESH_TFLITE" \
                    --output "$FACEMESH_ONNX" \
                    --opset 17 2>/dev/null \
                    && ok "FaceMesh V2 ONNX model" \
                    || warn "FaceMesh ONNX conversion failed — face filter will use stub"
            else
                warn "tf2onnx not installed — skipping FaceMesh ONNX conversion (pip install tf2onnx)"
            fi
        fi
    fi

    # Face filter assets (manifest + textures)
    local FILTER_ASSETS_DIR="$OE_MODELS_DIR/face_filters"
    if [ -f "$FILTER_ASSETS_DIR/manifest.json" ]; then
        skip "Face filter assets"
    else
        echo "  Creating stub face filter manifest..."
        mkdir -p "$FILTER_ASSETS_DIR"
        cat > "$FILTER_ASSETS_DIR/manifest.json" <<'MANIFEST_EOF'
{
    "version": 1,
    "filters": [
        {"id": "dog",  "name": "Dog Ears",     "texture": "dog/texture.png",  "uv_map": "dog/uv_map.json"},
        {"id": "cat",  "name": "Cat Whiskers", "texture": "cat/texture.png",  "uv_map": "cat/uv_map.json"},
        {"id": "none", "name": "No Filter",    "texture": "",                 "uv_map": ""}
    ]
}
MANIFEST_EOF
        ok "Face filter manifest (stub)"
    fi

    # Face recognition uses ONNX Runtime (SCRFD detector + AuraFace v1 recognizer).
    # AuraFace is also auto-downloaded at runtime by the inferencer's
    # fetchHfModel() on first loadModel(); this pre-install path is an
    # optional warm-cache step.

    # ── Face Recognition ONNX models (SCRFD-10G + AuraFace v1 / glintr100) ──
    # Runtime expects: $OE_MODELS_DIR/face_models/scrfd_auraface/{detector,recognizer}.onnx
    local FACE_RECOG_DIR="$OE_MODELS_DIR/face_models/scrfd_auraface"
    local SCRFD_ONNX="$FACE_RECOG_DIR/detector.onnx"
    local AURAFACE_ONNX="$FACE_RECOG_DIR/recognizer.onnx"
    if [ -f "$SCRFD_ONNX" ] && [ -f "$AURAFACE_ONNX" ]; then
        skip "SCRFD + AuraFace face recognition models"
    else
        mkdir -p "$FACE_RECOG_DIR"
        echo "  Downloading SCRFD-10G detector + AuraFace v1 (glintr100) recognizer..."
        python3 -c "
import os, urllib.request, zipfile, shutil, glob, tempfile

face_dir = '$FACE_RECOG_DIR'
scrfd_out     = os.path.join(face_dir, 'detector.onnx')
auraface_out  = os.path.join(face_dir, 'recognizer.onnx')

# SCRFD detector comes from buffalo_l.zip (InsightFace GitHub release).
scrfd_url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'

if not os.path.exists(scrfd_out):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'buffalo_l.zip')
        print('Downloading buffalo_l.zip for SCRFD detector...')
        urllib.request.urlretrieve(scrfd_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        det_files = glob.glob(os.path.join(tmpdir, '**', '*det*.onnx'), recursive=True)
        if det_files:
            shutil.copy2(det_files[0], scrfd_out)
            print(f'SCRFD detector extracted: {scrfd_out}')
        else:
            raise FileNotFoundError('No *det*.onnx in buffalo_l.zip')

# AuraFace v1 (glintr100) from Hugging Face (Apache 2.0).
if not os.path.exists(auraface_out):
    try:
        from huggingface_hub import hf_hub_download
        print('Downloading AuraFace v1 (glintr100.onnx) from fal/AuraFace-v1 ...')
        src = hf_hub_download(repo_id='fal/AuraFace-v1',
                              filename='glintr100.onnx',
                              cache_dir=os.environ.get('HF_HOME'))
        shutil.copy2(src, auraface_out)
        print(f'AuraFace recognizer saved: {auraface_out}')
    except Exception as e:
        print(f'WARNING: AuraFace pre-download failed ({e}); runtime fetchHfModel() will retry')
" || record_error "Face recognition ONNX model download failed"

        if [ -f "$SCRFD_ONNX" ] && [ -f "$AURAFACE_ONNX" ]; then
            ok "SCRFD + AuraFace face recognition models"
        else
            warn "Face recognition models incomplete at install time — runtime will attempt HF fetch on first use"
        fi
    fi

    # (Background-blur segmentation now runs MediaPipe Selfie Seg ONNX
    # end-to-end inside omniedge_bg_blur — no standalone TRT engine build
    # is needed here.)
}
