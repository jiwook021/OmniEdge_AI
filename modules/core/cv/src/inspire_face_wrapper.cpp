#include "cv/inspire_face_wrapper.hpp"

#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>

#include <NvInferVersion.h>

#include "common/file.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


namespace {

void checkHF(HResult result, const char* what)
{
    if (result != HSUCCEED) {
        throw std::runtime_error(
            std::format("[InspireFaceWrapper] {} failed — HResult={:#x}",
                        what, static_cast<unsigned>(result)));
    }
}

/// Pre-flight check: read a TRT engine file header from the model pack and
/// validate the build version against the installed TensorRT runtime.
/// TRT serialized engines are NOT portable across minor versions — loading
/// a mismatched engine will SIGSEGV inside nvinfer::IRuntime::deserializeCudaEngine.
void validateTrtEngineCompatibility(const std::string& packPath)
{
    // Megatron_TRT pack stores engine files without extensions.
    // Look for the scrfd detector engine as a representative sample.
    std::filesystem::path packDir(packPath);
    if (!isDirectory(packDir)) {
        throw std::runtime_error(std::format(
            "[InspireFaceWrapper] Model pack not found: {}", packPath));
    }

    // Find any file starting with underscore (TRT engines in the pack)
    std::filesystem::path engineFile;
    for (const auto& entry : std::filesystem::directory_iterator(packDir)) {
        const auto name = entry.path().filename().string();
        if (name.starts_with("_") && entry.is_regular_file() && entry.file_size() > 32) {
            engineFile = entry.path();
            break;
        }
    }

    if (engineFile.empty()) {
        OE_LOG_WARN("inspire_face_no_trt_engines: pack={} — cannot validate TRT version",
                   packPath);
        return;
    }

    // Read first 32 bytes — TRT engine header contains version at offset 0x18
    std::ifstream fin(engineFile, std::ios::binary);
    if (!fin) return;

    uint8_t header[32]{};
    fin.read(reinterpret_cast<char*>(header), 32);
    if (fin.gcount() < 32) return;

    // TRT engine header: bytes [0x18..0x1b] = major.minor.patch.build (4 bytes)
    const int engineMajor = header[0x18];
    const int engineMinor = header[0x19];

    // Installed TRT runtime version
    const int installedMajor = NV_TENSORRT_MAJOR;
    const int installedMinor = NV_TENSORRT_MINOR;

    OE_LOG_INFO("inspire_face_trt_version_check: engine={}.{}, installed={}.{}, file={}",
              engineMajor, engineMinor, installedMajor, installedMinor,
              engineFile.filename().string());

    if (engineMajor != installedMajor || engineMinor != installedMinor) {
        throw std::runtime_error(std::format(
            "[InspireFaceWrapper] TRT engine version mismatch — model pack engines "
            "were built with TensorRT {}.{} but installed runtime is {}.{}. "
            "Rebuild the InspireFace model pack with the current TRT version. "
            "See: InspireFace/command/build_linux_tensorrt.sh",
            engineMajor, engineMinor, installedMajor, installedMinor));
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor — launch SDK + create session
// ---------------------------------------------------------------------------

InspireFaceWrapper::InspireFaceWrapper(const std::string& modelPackPath)
{
    // Pre-flight: validate TRT engine version BEFORE loading the pack.
    // Loading a version-mismatched engine causes SIGSEGV inside TRT runtime.
    validateTrtEngineCompatibility(modelPackPath);

    // Set CUDA device FIRST — TensorRT model packs (e.g., Megatron_TRT) need
    // a valid CUDA context before HFLaunchInspireFace loads TRT engines.
    // Without this ordering, the SDK segfaults during engine deserialization.
    const HResult cudaResult = HFSetCudaDeviceId(0);
    checkHF(cudaResult, "HFSetCudaDeviceId");

    // Load model pack (idempotent: SDK ref-counts pack loads)
    const HResult launchResult =
        HFLaunchInspireFace(modelPackPath.c_str());
    checkHF(launchResult, "HFLaunchInspireFace");

    // Create a session configured for:
    //   - Detect + recognition mode (HF_DETECT_MODE_IMAGE for single-frame)
    //   - Max 1 face simultaneously (typical desk assistant scenario)
    //   - BGR pixel format (matches SHM layout)
    //   - No pyramid scaling (single-scale, fast path)
    HFSessionCustomParameter param{};
    param.enable_recognition          = 1;
    param.enable_liveness             = 0;
    param.enable_ir_liveness          = 0;
    param.enable_mask_detect          = 0;
    param.enable_face_quality         = 0;
    param.enable_face_attribute       = 0;
    param.enable_interaction_liveness = 0;
    param.enable_detect_mode_landmark = 0;
    param.enable_face_pose            = 0;
    param.enable_face_emotion         = 0;

    const HResult sessionResult = HFCreateInspireFaceSession(
        param,
        HF_DETECT_MODE_ALWAYS_DETECT,  // full detect on every call (no tracking state)
        1,                      // maxDetectFaceNum
        -1,                     // detectPixelLevel (-1 = auto)
        0,                      // trackByDetectModeFPS (ignored in IMAGE mode)
        &session_);
    checkHF(sessionResult, "HFCreateInspireFaceSession");

    OE_LOG_INFO("inspire_face_session_created: pack={}", modelPackPath);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

InspireFaceWrapper::~InspireFaceWrapper()
{
    if (session_) {
        HFReleaseInspireFaceSession(session_);
        session_ = {};
    }
    // HFTerminateInspireFace is ref-counted — call even if session release fails
    HFTerminateInspireFace();
}

// ---------------------------------------------------------------------------
// detect()
// ---------------------------------------------------------------------------

std::vector<DetectedFace> InspireFaceWrapper::detect(
    const uint8_t* bgrData, int width, int height)
{
    HFImageData imageData{};
    imageData.data       = const_cast<uint8_t*>(bgrData);
    imageData.width      = width;
    imageData.height     = height;
    imageData.format     = HF_STREAM_BGR;
    imageData.rotation   = HF_CAMERA_ROTATION_0;

    HFImageStream imageStream{};
    const HResult createResult = HFCreateImageStream(&imageData, &imageStream);
    checkHF(createResult, "HFCreateImageStream");

    HFMultipleFaceData faceData{};
    const HResult trackResult =
        HFExecuteFaceTrack(session_, imageStream, &faceData);

    // Always release the image stream before checking the result
    HFReleaseImageStream(imageStream);
    checkHF(trackResult, "HFExecuteFaceTrack");

    std::vector<DetectedFace> results;
    results.reserve(static_cast<std::size_t>(faceData.detectedNum));

    for (HInt32 i = 0; i < faceData.detectedNum; ++i) {
        DetectedFace df;
        df.token      = faceData.tokens[i];
        df.trackScore = faceData.detConfidence[i];

        // bounding box is HFaceRect: x, y, width, height (pixel units)
        const HFaceRect& r = faceData.rects[i];
        df.x1 = static_cast<float>(r.x)              / static_cast<float>(width);
        df.y1 = static_cast<float>(r.y)              / static_cast<float>(height);
        df.x2 = static_cast<float>(r.x + r.width)    / static_cast<float>(width);
        df.y2 = static_cast<float>(r.y + r.height)   / static_cast<float>(height);

        results.push_back(df);
    }

    return results;
}

// ---------------------------------------------------------------------------
// extractEmbedding()
// ---------------------------------------------------------------------------

bool InspireFaceWrapper::extractEmbedding(
    const uint8_t*   bgrData,
    int              width,
    int              height,
    HFFaceBasicToken token,
    float*           embedding)
{
    HFImageData imageData{};
    imageData.data       = const_cast<uint8_t*>(bgrData);
    imageData.width      = width;
    imageData.height     = height;
    imageData.format     = HF_STREAM_BGR;
    imageData.rotation   = HF_CAMERA_ROTATION_0;

    HFImageStream imageStream{};
    const HResult createResult = HFCreateImageStream(&imageData, &imageStream);
    checkHF(createResult, "HFCreateImageStream (extract)");

    HFFaceFeature feature{};
    const HResult extractResult =
        HFFaceFeatureExtract(session_, imageStream, token, &feature);

    HFReleaseImageStream(imageStream);

    if (extractResult != HSUCCEED) {
        OE_LOG_WARN("inspire_face_extract_failed: result={}",
                static_cast<int>(extractResult));
        return false;
    }

    if (feature.size != kFaceEmbeddingDim) {
        OE_LOG_WARN("inspire_face_unexpected_dim: expected={}, got={}",
                kFaceEmbeddingDim, feature.size);
        return false;
    }

    std::memcpy(embedding, feature.data,
                kFaceEmbeddingDim * sizeof(float));
    return true;
}

