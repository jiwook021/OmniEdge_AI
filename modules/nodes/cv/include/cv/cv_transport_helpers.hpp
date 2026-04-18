#pragma once

// ---------------------------------------------------------------------------
// cv_transport_helpers.hpp — Shared CV node transport setup utilities
//
// SHM open/create and BGR frame publish logic shared across CV nodes.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include <tl/expected.hpp>

#include "common/constants/video_constants.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_shm_helpers.hpp"
#include "common/pipeline_types.hpp"
#include "common/zmq_messages.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_mapping.hpp"
#include "zmq/jpeg_constants.hpp"
#include "zmq/message_router.hpp"

#include <nlohmann/json.hpp>

namespace oe::cv {

// ---------------------------------------------------------------------------
// SHM output result — holds whichever output path was created
// ---------------------------------------------------------------------------
struct CvShmOutput {
	std::unique_ptr<ShmMapping> jpeg;                                 ///< JPEG mode
	std::unique_ptr<ShmCircularBuffer<ShmVideoHeader>> bgr;           ///< BGR24 mode
};

// ---------------------------------------------------------------------------
// openInputShm — open an existing BGR24 circular-buffer SHM segment as a
// consumer, retrying until the producer creates it.
// ---------------------------------------------------------------------------
[[nodiscard]] inline tl::expected<std::unique_ptr<ShmMapping>, std::string>
openInputShm(std::string_view shmName, std::string_view logPrefix)
{
	const std::size_t size = ShmCircularBuffer<ShmVideoHeader>::segmentSize(
		kCircularBufferSlotCount, kMaxBgr24FrameBytes);
	return oe::shm::openConsumerWithRetry(shmName, size, logPrefix);
}

// ---------------------------------------------------------------------------
// createOutputShm — create the appropriate output SHM segment based on
// OutputFormat.  BGR24 mode creates a ShmCircularBuffer for downstream
// chaining; JPEG mode creates a flat double-buffer for WebSocketBridge.
// ---------------------------------------------------------------------------
[[nodiscard]] inline CvShmOutput
createOutputShm(OutputFormat format,
                std::string_view bgrShmName,
                uint32_t         bgrSlotCount,
                std::string_view jpegShmName,
                std::string_view logPrefix)
{
	CvShmOutput out;
	if (format == OutputFormat::kBgr24) {
		out.bgr = std::make_unique<ShmCircularBuffer<ShmVideoHeader>>(
			bgrShmName, bgrSlotCount, kMaxBgr24FrameBytes, /*create=*/true);
		OE_LOG_INFO("{}_output_bgr24: shm={}, slots={}",
			logPrefix, bgrShmName, bgrSlotCount);
	} else {
		out.jpeg = oe::shm::createProducer(jpegShmName, kJpegShmSegmentByteSize);
	}
	return out;
}

// ---------------------------------------------------------------------------
// CvTransportResult + configureCvTransport — one-call SHM setup for CV nodes.
//
// Steps:
//   1. Set logger module name
//   2. Open input SHM (consumer with retry)
//   3. Create output SHM (JPEG or BGR24 mode)
// ---------------------------------------------------------------------------
struct CvTransportResult {
	std::unique_ptr<ShmMapping>                          shmIn;
	std::unique_ptr<ShmMapping>                          shmOutJpeg; ///< nullptr in BGR24 mode
	std::unique_ptr<ShmCircularBuffer<ShmVideoHeader>>   shmOutBgr;  ///< nullptr in JPEG mode
};

[[nodiscard]] inline tl::expected<CvTransportResult, std::string>
configureCvTransport(std::string_view moduleName,
                     std::string_view inputShmName,
                     OutputFormat     outputFormat,
                     std::string_view outputBgrShmName,
                     uint32_t         outputSlotCount,
                     std::string_view outputJpegShmName)
{
	OeLogger::instance().setModule(moduleName);

	auto shmResult = openInputShm(inputShmName, moduleName);
	if (!shmResult) return tl::unexpected(shmResult.error());

	auto shmOutput = createOutputShm(outputFormat, outputBgrShmName,
		outputSlotCount, outputJpegShmName, moduleName);

	CvTransportResult result;
	result.shmIn      = std::move(*shmResult);
	result.shmOutBgr  = std::move(shmOutput.bgr);
	result.shmOutJpeg = std::move(shmOutput.jpeg);
	return result;
}

// ---------------------------------------------------------------------------
// publishBgrFrame — notify downstream consumers that a new BGR24 frame is
// available in the named SHM segment.
// ---------------------------------------------------------------------------
inline void publishBgrFrame(MessageRouter& router,
                            std::string_view topic,
                            std::string_view shmName,
                            uint64_t         seq,
                            uint32_t         width,
                            uint32_t         height)
{
	static thread_local BgrFrameMsg payload;
	payload.type = std::string{topic};
	payload.shm  = std::string{shmName};
	payload.seq  = seq;
	payload.w    = width;
	payload.h    = height;
	router.publish(topic, nlohmann::json(payload));
}

} // namespace oe::cv
