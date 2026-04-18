#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "tts_inferencer.hpp"


/// Deterministic mock TTS inferencer for node-level tests (no GPU required).
/// synthesize() returns silence at 24 kHz and tracks call count.
class MinimalMockInferencer : public TTSInferencer {
public:
	[[nodiscard]] tl::expected<void, std::string> loadModel(
		const std::string& /*onnxModelPath*/,
		const std::string& /*voiceDir*/) override
	{
		return {};
	}

	[[nodiscard]] tl::expected<std::vector<float>, std::string> synthesize(
		std::string_view text,
		std::string_view /*voiceName*/,
		float            /*speed*/) override
	{
		if (text.empty()) {
			return tl::unexpected(std::string("empty text"));
		}
		++synthesizeCallCount;
		lastSynthesizedText = std::string(text);
		// 1 second of silence at 24 kHz.
		return std::vector<float>(24'000, 0.0f);
	}

	void unloadModel() noexcept override {}

	[[nodiscard]] std::size_t currentVramUsageBytes() const override { return 0; }

	[[nodiscard]] std::string name() const override { return "minimal-mock"; }

	int         synthesizeCallCount{0};
	std::string lastSynthesizedText;
};

