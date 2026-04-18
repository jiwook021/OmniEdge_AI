#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — CV installation (YOLOv8-seg + Face Recognition)
# ═══════════════════════════════════════════════════════════════════════════════
[[ -n "${_OE_INSTALL_CV_SOURCED:-}" ]] && return 0
readonly _OE_INSTALL_CV_SOURCED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_memory.sh"

# ═══════════════════════════════════════════════════════════════════════════════
#  5. CV — YOLOv8-seg + Face Recognition
# ═══════════════════════════════════════════════════════════════════════════════
install_cv() {
    section "CV — YOLOv8-seg + Face Recognition"

    # --no-deps prevents ultralytics from upgrading numpy to 2.x
    # (tensorrt_llm requires numpy<2, opencv-python wants numpy>=2)
    pip install --break-system-packages --no-deps ultralytics 2>/dev/null || true
    pip install --break-system-packages 'numpy<2' 2>/dev/null || true

    # YOLOv8-seg TRT engine
    local YOLO_ENGINE="$OE_ENGINES_DIR/yolov8n-seg.engine"
    if [ -f "$YOLO_ENGINE" ]; then
        skip "YOLOv8-seg TRT engine"
    else
        echo "  Exporting YOLOv8-seg to ONNX + TensorRT..."
        local YOLO_WORK="$OE_MODELS_DIR/yolo_work"
        mkdir -p "$YOLO_WORK"
        pushd "$YOLO_WORK" > /dev/null
        python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
model.export(format='onnx', imgsz=640, half=True, simplify=True)
" || { record_error "YOLOv8 ONNX export failed"; popd > /dev/null; return; }

        if command -v trtexec &>/dev/null; then
            trtexec \
                --onnx=yolov8n-seg.onnx \
                --saveEngine="$YOLO_ENGINE" \
                --fp16 \
                --minShapes=images:1x3x640x640 \
                --optShapes=images:1x3x640x640 \
                --maxShapes=images:1x3x640x640 || record_error "YOLOv8 TRT engine build failed"
        else
            echo "  trtexec not found — using Python TensorRT API..."
            python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open('yolov8n-seg.onnx', 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError('ONNX parse failed')
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
profile = builder.create_optimization_profile()
shape = (1, 3, 640, 640)
profile.set_shape('images', shape, shape, shape)
config.add_optimization_profile(profile)
engine = builder.build_serialized_network(network, config)
with open('$YOLO_ENGINE', 'wb') as f:
    f.write(engine)
print('YOLOv8 engine saved')
" || record_error "YOLOv8 TRT engine build failed (Python API)"
        fi
        popd > /dev/null
    fi

    # FaceMesh V2 ONNX model (for face filter AR — ~2.4 MB FP16)
    local FACEMESH_DIR="$OE_MODELS_DIR/face_mesh"
    local FACEMESH_ONNX="$FACEMESH_DIR/face_landmark_detector.onnx"
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

    # ── YOLOv8s-seg TRT engine (larger variant for benchmark) ──
    local YOLO_S_ENGINE="$OE_ENGINES_DIR/yolov8s-seg.engine"
    if [ -f "$YOLO_S_ENGINE" ]; then
        skip "YOLOv8s-seg TRT engine"
    else
        echo "  Exporting YOLOv8s-seg to ONNX + TensorRT..."
        local YOLO_S_WORK="$OE_MODELS_DIR/yolo_s_work"
        mkdir -p "$YOLO_S_WORK"
        pushd "$YOLO_S_WORK" > /dev/null
        python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8s-seg.pt')
model.export(format='onnx', imgsz=640, half=True, simplify=True)
" || { record_error "YOLOv8s ONNX export failed"; popd > /dev/null; }

        if [ -f "yolov8s-seg.onnx" ]; then
            if command -v trtexec &>/dev/null; then
                trtexec \
                    --onnx=yolov8s-seg.onnx \
                    --saveEngine="$YOLO_S_ENGINE" \
                    --fp16 \
                    --minShapes=images:1x3x640x640 \
                    --optShapes=images:1x3x640x640 \
                    --maxShapes=images:1x3x640x640 || record_error "YOLOv8s TRT engine build failed"
            else
                python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open('yolov8s-seg.onnx', 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError('ONNX parse failed')
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
profile = builder.create_optimization_profile()
shape = (1, 3, 640, 640)
profile.set_shape('images', shape, shape, shape)
config.add_optimization_profile(profile)
engine = builder.build_serialized_network(network, config)
with open('$YOLO_S_ENGINE', 'wb') as f:
    f.write(engine)
print('YOLOv8s engine saved')
" || record_error "YOLOv8s TRT engine build failed (Python API)"
            fi
        fi
        popd > /dev/null
    fi
}
