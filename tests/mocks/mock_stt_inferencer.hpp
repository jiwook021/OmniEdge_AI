#pragma once

#include <span>
#include <string>

#include "stt_inferencer.hpp"


/// Deterministic mock STT inferencer for node-level tests (no GPU required).
/// Returns a configurable TranscribeResult from transcribe().
class MockSTTInferencer : public STTInferencer {
public:
	/// The transcription text that transcribe() will return.
	std::string cannedTranscription = "Hello world";
	/// Language code to return.
	std::string cannedLanguage = "en";
	/// Simulated no-speech probability (0 = definitely speech).
	float cannedNoSpeechProb = 0.05f;
	/// Simulated average log probability (higher = more confident).
	float cannedAvgLogprob = -0.3f;
	/// When true, transcribe() returns an error instead of a result.
	bool simulateError{false};
	/// Error message to return when simulateError is true.
	std::string errorMessage = "mock transcription error";

	void loadModel(const std::string& /*encoderDir*/,
	               const std::string& /*decoderDir*/,
	               const std::string& /*tokenizerDir*/) override
	{
		modelLoaded_ = true;
	}

	[[nodiscard]] tl::expected<TranscribeResult, std::string>
	transcribe(std::span<const float> melSpectrogram,
	           uint32_t /*numFrames*/) override
	{
		if (!modelLoaded_) {
			return tl::unexpected(std::string("model not loaded"));
		}
		if (simulateError) {
			return tl::unexpected(errorMessage);
		}
		if (melSpectrogram.empty()) {
			return tl::unexpected(std::string("empty mel spectrogram"));
		}
		TranscribeResult result;
		result.text         = cannedTranscription;
		result.language     = cannedLanguage;
		result.noSpeechProb = cannedNoSpeechProb;
		result.avgLogprob   = cannedAvgLogprob;
		return result;
	}

	void unloadModel() noexcept override { modelLoaded_ = false; }

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override
	{
		return 0;
	}

private:
	bool modelLoaded_{false};
};

