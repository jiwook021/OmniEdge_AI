#include "cv/face_recog_node.hpp"
#include "cv/onnx_face_recog_inferencer.hpp"

#include "common/cv_config_loader.hpp"
#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_face_recog entry point
//
// Single supported variant: scrfd_auraface_v1
//   SCRFD-10G detector + AuraFace v1 (glintr100.onnx) recognizer.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	return oeNodeMain<FaceRecognitionNode>(argc, argv,
		"omniedge_face_recog", "face_recognition",

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> FaceRecognitionNode::Config {
			FaceRecognitionNode::Config cfg;

			const auto common = loadCvCommonFromIni(ini, cfg.videoSubPort, cfg.wsBridgeSubPort);
			cfg.videoSubPort    = common.videoSubPort;
			cfg.wsBridgeSubPort = common.wsBridgeSubPort;

			const auto& fr = ini.faceRecognition();
			cfg.modelPackPath        = common.resolve(fr.modelPackPath);
			cfg.pubPort              = ini.port("face_recognition", cfg.pubPort);
			cfg.recognitionThreshold = fr.recognitionThreshold;
			cfg.knownFacesDb         = fr.facesDb;
			cfg.frameSubsample       = static_cast<uint32_t>(fr.frameSubsample);
			cfg.inputShmName         = fr.shmInput;
			cfg.moduleName           = "face_recognition";
			return cfg;
		},

		[](FaceRecognitionNode& node) {
			const auto variant = FaceRecogVariant::kScrfdAuraFaceV1;
			spdlog::info("face_recog_variant: {} (ONNX Runtime)",
			             faceRecogVariantName(variant));
			node.setInferencer(createOnnxFaceRecogInferencer(variant));
		}
	);
}
