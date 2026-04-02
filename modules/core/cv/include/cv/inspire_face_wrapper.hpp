#pragma once

#include <cstdint>
#include <string>
#include <vector>

// InspireFace C API headers
#include <inspireface.h>

// ---------------------------------------------------------------------------
// OmniEdge_AI — InspireFace RAII Wrapper
//
// Wraps the InspireFace C SDK in C++ RAII semantics.
//
// InspireFace pipeline:
//   1. HFLaunchInspireFace(packPath) — load model pack (once per process)
//   2. HFCreateInspireFaceSessionOptional(...) — create per-thread session
//   3. HFCreateImageStream() — wrap raw BGR frame for SDK consumption
//   4. HFExecuteFaceTrack(session, imageStream, &faceData) — detect + track
//   5. HFFaceFeatureExtract(session, imageStream, token, &feature) — 512-dim
//   6. HFReleaseImageStream() — free SDK image handle
//   7. HFReleaseInspireFaceSession(session) — release session (destructor)
//   8. HFTerminateInspireFace() — unload pack (process teardown)
//
// ⚠️  Commercial license warning (from arch spec):
//     InspireFace's default Megatron_TRT model pack uses InsightFace-trained
//     weights restricted to non-commercial/academic use only.  For commercial
//     deployment, substitute permissively-licensed models or obtain a license.
//
// Thread safety: none. One wrapper per module instance.
// ---------------------------------------------------------------------------


/// Dimensionality of the face embedding vector produced by InspireFace.
inline constexpr int kFaceEmbeddingDim = 512;

// ---------------------------------------------------------------------------
// DetectedFace — lightweight POD result from one detection
// ---------------------------------------------------------------------------

struct DetectedFace {
    HFFaceBasicToken  token;         ///< Opaque SDK token for subsequent extraction
    float             x1, y1;        ///< Bounding box top-left (normalised 0–1)
    float             x2, y2;        ///< Bounding box bottom-right (normalised 0–1)
    float             trackScore;    ///< Detection confidence
};

// ---------------------------------------------------------------------------
// InspireFaceWrapper
// ---------------------------------------------------------------------------

class InspireFaceWrapper {
public:
    /**
     * @brief Load the InspireFace model pack and create a session.
     *
     * @param modelPackPath  Path to the Megatron_TRT pack directory.
     * @throws std::runtime_error on SDK or session init failure.
     */
    explicit InspireFaceWrapper(const std::string& modelPackPath);
    ~InspireFaceWrapper();

    InspireFaceWrapper(const InspireFaceWrapper&)            = delete;
    InspireFaceWrapper& operator=(const InspireFaceWrapper&) = delete;

    /**
     * @brief Detect faces in a BGR24 frame.
     *
     * @param bgrData  Pointer to BGR24 pixel data (no header, row-major).
     * @param width    Frame width in pixels.
     * @param height   Frame height in pixels.
     * @return         Vector of detected faces (may be empty).
     */
    [[nodiscard]] std::vector<DetectedFace> detect(const uint8_t* bgrData,
                                                    int            width,
                                                    int            height);

    /**
     * @brief Extract a 512-dim float embedding for one detected face.
     *
     * Must be called on the same frame as detect() — the SDK image stream
     * must still be valid.
     *
     * @param bgrData   Same frame used for detect().
     * @param width     Frame width.
     * @param height    Frame height.
     * @param token     DetectedFace::token from detect().
     * @param embedding Output: 512 floats written here.  Caller must provide
     *                  storage for kFaceEmbeddingDim floats.
     * @return          true on success.
     */
    [[nodiscard]] bool extractEmbedding(const uint8_t*    bgrData,
                          int               width,
                          int               height,
                          HFFaceBasicToken  token,
                          float*            embedding);

private:
    HFSession session_{};
};

