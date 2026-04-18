#include "stt/mel_spectrogram.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "common/oe_logger.hpp"
#include "common/constants/whisper_constants.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// MelSpectrogram implementation
//
// FFT approach: zero-pad each 400-sample Hann window to 512 samples, then
// run a radix-2 FFT.  We keep only the first 201 bins (0..200), which
// correspond to the frequencies a 400-point DFT would produce (0..8 kHz).
// Bins 201..256 are discarded — they alias from above Nyquist of the
// original window size and are not part of the Whisper mel filterbank.
// ---------------------------------------------------------------------------


namespace {

/// Upper frequency limit of the Whisper mel filterbank.
inline constexpr float kFMin = kMelFrequencyMinHz;
inline constexpr float kFMax = kMelFrequencyMaxHz;

/// Log10(1e-10) — floor used before log10 to avoid -inf.
inline constexpr float kLogFloor = 1e-10f;

/** @brief Return the smallest power-of-2 that is >= n. */
int nextPow2(int n) noexcept
{
	int p = 1;
	while (p < n) p <<= 1;
	return p;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MelSpectrogram::MelSpectrogram(int nMels, int nFft, int hopLen, int sampleRate)
	: nMels_(nMels)
	, nFft_(nFft)
	, nFftPadded_(nextPow2(nFft))
	, nFreqBins_(nFft / 2 + 1)  // 201 for nFft=400
	, hopLen_(hopLen)
	, sampleRate_(sampleRate)
{
	if (nMels <= 0 || nFft <= 0 || hopLen <= 0 || sampleRate <= 0) {
		throw std::invalid_argument(
			std::format("[MelSpectrogram] invalid params: nMels={} nFft={} "
			            "hopLen={} sampleRate={}", nMels, nFft, hopLen, sampleRate));
	}

	// Precompute Hann window coefficients
	hannWindow_.resize(static_cast<std::size_t>(nFft_));
	for (int i = 0; i < nFft_; ++i) {
		// Symmetric Hann: w[i] = 0.5 * (1 - cos(2π·i/(N-1)))
		// Whisper uses periodic Hann (N points in [0, 2π) denominator = N)
		const float angle = 2.0f * std::numbers::pi_v<float>
		                    * static_cast<float>(i) / static_cast<float>(nFft_);
		hannWindow_[i] = 0.5f * (1.0f - std::cos(angle));
	}

	// Precompute mel filterbank
	melFilterBank_ = buildMelFilterBank(nMels_, nFreqBins_, sampleRate_,
	                                    kFMin, kFMax);

	// Precompute sparse bin ranges per mel filter for O(active_bins) inner loop
	melBinStart_.resize(static_cast<std::size_t>(nMels_));
	melBinEnd_.resize(static_cast<std::size_t>(nMels_));
	for (int melBin = 0; melBin < nMels_; ++melBin) {
		const float* row = melFilterBank_.data()
		                   + static_cast<std::size_t>(melBin * nFreqBins_);
		int start = nFreqBins_;
		int end   = 0;
		for (int freqBin = 0; freqBin < nFreqBins_; ++freqBin) {
			if (row[freqBin] > 0.0f) {
				if (freqBin < start) start = freqBin;
				end = freqBin + 1;
			}
		}
		if (start >= end) { start = 0; end = 0; }  // empty filter
		melBinStart_[static_cast<std::size_t>(melBin)] = start;
		melBinEnd_[static_cast<std::size_t>(melBin)]   = end;
	}
}

// ---------------------------------------------------------------------------
// numFrames()
// ---------------------------------------------------------------------------

int MelSpectrogram::numFrames(std::size_t nSamples) const noexcept
{
	if (nSamples == 0) return 0;
	// Centre-mode: pad nFft/2 on each side → effective length = nSamples + nFft
	// n_frames = ceil(nSamples / hopLen_)
	return static_cast<int>((nSamples + static_cast<std::size_t>(hopLen_) - 1)
	                        / static_cast<std::size_t>(hopLen_));
}

// ---------------------------------------------------------------------------
// compute()
// ---------------------------------------------------------------------------

std::vector<float> MelSpectrogram::compute(const float* pcm,
                                            std::size_t  nSamples) const
{
	if (!pcm || nSamples == 0) {
		return {};
	}

	const int    nFrames   = numFrames(nSamples);
	OE_LOG_DEBUG("mel_compute: samples={}, n_frames={}, n_mels={}, hop_len={}",
	           nSamples, nFrames, nMels_, hopLen_);
	const int    padLen    = nFft_ / 2;                  // 200 samples
	const int    totalLen  = static_cast<int>(nSamples) + 2 * padLen;

	// Build reflected-padded signal: [pad | pcm | pad]
	std::vector<float> padded(static_cast<std::size_t>(totalLen));

	// Left reflect pad: pcm[padLen-1], pcm[padLen-2], ..., pcm[0]
	// Clamp index to [0, nSamples-1] to handle short inputs (nSamples < padLen).
	for (int i = 0; i < padLen; ++i) {
		const int srcIdx = std::min(padLen - 1 - i, static_cast<int>(nSamples) - 1);
		padded[static_cast<std::size_t>(i)] = pcm[std::max(0, srcIdx)];
	}
	// Centre: copy original signal
	std::copy(pcm, pcm + nSamples,
	          padded.begin() + padLen);
	// Right reflect pad: pcm[N-2], pcm[N-3], ..., pcm[N-padLen-1]
	const int lastIdx = static_cast<int>(nSamples) - 1;
	for (int i = 0; i < padLen; ++i) {
		const int srcIdx = std::max(0, lastIdx - 1 - i);
		padded[static_cast<std::size_t>(padLen + static_cast<int>(nSamples) + i)]
			= pcm[srcIdx];
	}

	// Temporary buffers for FFT (reused across frames)
	std::vector<float> fftRe(static_cast<std::size_t>(nFftPadded_));
	std::vector<float> fftIm(static_cast<std::size_t>(nFftPadded_));

	// Power spectrum accumulator: [nFreqBins_] per frame
	// Output: [nMels_ × nFrames] row-major
	std::vector<float> melOut(static_cast<std::size_t>(nMels_ * nFrames), 0.0f);

	for (int frame = 0; frame < nFrames; ++frame) {
		const int frameStart = frame * hopLen_;

		// Zero the full FFT buffers (FFT writes to all nFftPadded_ entries in-place)
		std::fill(fftRe.begin(), fftRe.end(), 0.0f);
		std::fill(fftIm.begin(), fftIm.end(), 0.0f);

		for (int s = 0; s < nFft_; ++s) {
			const int srcPos = frameStart + s;
			const float sample = (srcPos < totalLen) ? padded[static_cast<std::size_t>(srcPos)] : 0.0f;
			fftRe[static_cast<std::size_t>(s)] = hannWindow_[static_cast<std::size_t>(s)] * sample;
		}

		radix2Fft(fftRe, fftIm);

		// Compute power spectrum and accumulate into mel bands using sparse ranges.
		// Each mel filter is non-zero only in [melBinStart_[m], melBinEnd_[m]),
		// typically 3-8 bins wide, vs. all 201 bins in the dense version.
		for (int mel = 0; mel < nMels_; ++mel) {
			const int kStart = melBinStart_[static_cast<std::size_t>(mel)];
			const int kEnd   = melBinEnd_[static_cast<std::size_t>(mel)];
			const float* filterRow =
				melFilterBank_.data() + static_cast<std::size_t>(mel * nFreqBins_);
			float energy = 0.0f;
			for (int k = kStart; k < kEnd; ++k) {
				const float re = fftRe[static_cast<std::size_t>(k)];
				const float im = fftIm[static_cast<std::size_t>(k)];
				energy += filterRow[k] * (re * re + im * im);
			}
			melOut[static_cast<std::size_t>(mel * nFrames + frame)] = energy;
		}
	}

	// log10 + clamp + normalize — fused into two passes (need global max first).
	// Constants from OpenAI Whisper reference (whisper/audio.py) — must not change.
	constexpr float kLogClampRange = 8.0f;  // max dynamic range in log10 space
	constexpr float kLogNormShift  = 4.0f;  // shift+scale to [-1, +1] range

	// Pass 1: log10(max(x, 1e-10)) and find global max
	float globalMax = -std::numeric_limits<float>::infinity();
	for (float& v : melOut) {
		v = std::log10(std::max(v, kLogFloor));
		if (v > globalMax) globalMax = v;
	}

	// Pass 2: clamp + normalize (fused)
	const float floor = globalMax - kLogClampRange;
	for (float& v : melOut) {
		v = (std::max(v, floor) + kLogNormShift) / kLogNormShift;
	}

	return melOut;
}

// ---------------------------------------------------------------------------
// radix2Fft() — Cooley-Tukey in-place DIT FFT
// ---------------------------------------------------------------------------

void MelSpectrogram::radix2Fft(std::vector<float>& re, std::vector<float>& im)
{
	const int n = static_cast<int>(re.size());
	assert(re.size() == im.size());

	// Bit-reversal permutation
	for (int i = 1, j = 0; i < n; ++i) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) { j ^= bit; }
		j ^= bit;
		if (i < j) {
			std::swap(re[static_cast<std::size_t>(i)],
			          re[static_cast<std::size_t>(j)]);
			std::swap(im[static_cast<std::size_t>(i)],
			          im[static_cast<std::size_t>(j)]);
		}
	}

	// Butterfly stages
	for (int len = 2; len <= n; len <<= 1) {
		const float ang  = -2.0f * std::numbers::pi_v<float>
		                   / static_cast<float>(len);
		const float cosW = std::cos(ang);
		const float sinW = std::sin(ang);

		for (int i = 0; i < n; i += len) {
			float wr = 1.0f, wi = 0.0f;  // twiddle factor w = e^(j·ang·k)
			for (int k = 0; k < len / 2; ++k) {
				const int u = i + k;
				const int v = i + k + len / 2;
				const float ur = re[static_cast<std::size_t>(u)];
				const float ui = im[static_cast<std::size_t>(u)];
				const float vr = re[static_cast<std::size_t>(v)] * wr
				                 - im[static_cast<std::size_t>(v)] * wi;
				const float vi = re[static_cast<std::size_t>(v)] * wi
				                 + im[static_cast<std::size_t>(v)] * wr;
				re[static_cast<std::size_t>(u)] = ur + vr;
				im[static_cast<std::size_t>(u)] = ui + vi;
				re[static_cast<std::size_t>(v)] = ur - vr;
				im[static_cast<std::size_t>(v)] = ui - vi;
				// Rotate twiddle factor: (wr + j·wi) × (cosW + j·sinW)
				const float wrnew = wr * cosW - wi * sinW;
				wi                = wr * sinW + wi * cosW;
				wr                = wrnew;
			}
		}
	}
}

// ---------------------------------------------------------------------------
// buildMelFilterBank()
// ---------------------------------------------------------------------------

std::vector<float> MelSpectrogram::buildMelFilterBank(
	int nMels, int nFreqBins, int sampleRate, float fMin, float fMax)
{
	// nMels+2 evenly spaced points in mel space, converted back to Hz,
	// then to bin indices in the DFT output.
	const float melMin = hzToMel(fMin);
	const float melMax = hzToMel(fMax);

	// nMels + 2 mel centre points
	std::vector<float> melPts(static_cast<std::size_t>(nMels + 2));
	for (int i = 0; i <= nMels + 1; ++i) {
		melPts[static_cast<std::size_t>(i)] =
			melMin + static_cast<float>(i) * (melMax - melMin)
			         / static_cast<float>(nMels + 1);
	}

	// Convert mel points to Hz, then to DFT bin indices (float)
	// DFT frequency resolution: sampleRate / (nFftPadded)
	// But our nFreqBins = nFft/2+1 corresponds to nFft (not nFftPadded).
	// We build the filter against the nFft/2+1 bins at frequencies:
	//   freq[k] = k * sampleRate / nFft   for k = 0..nFreqBins-1
	const float freqPerBin = static_cast<float>(sampleRate)
	                         / static_cast<float>((nFreqBins - 1) * 2);  // = sampleRate/nFft

	std::vector<float> hzPts(static_cast<std::size_t>(nMels + 2));
	std::vector<float> binPts(static_cast<std::size_t>(nMels + 2));
	for (int i = 0; i <= nMels + 1; ++i) {
		hzPts[static_cast<std::size_t>(i)]  = melToHz(melPts[static_cast<std::size_t>(i)]);
		binPts[static_cast<std::size_t>(i)] = hzPts[static_cast<std::size_t>(i)] / freqPerBin;
	}

	// Build triangular filters: [nMels × nFreqBins] row-major
	std::vector<float> bank(static_cast<std::size_t>(nMels * nFreqBins), 0.0f);

	for (int m = 0; m < nMels; ++m) {
		const float f0 = binPts[static_cast<std::size_t>(m)];      // left edge
		const float fc = binPts[static_cast<std::size_t>(m + 1)];  // centre
		const float f1 = binPts[static_cast<std::size_t>(m + 2)];  // right edge

		float* row = bank.data() + static_cast<std::size_t>(m * nFreqBins);

		for (int k = 0; k < nFreqBins; ++k) {
			const float fk = static_cast<float>(k);
			if (fk >= f0 && fk < fc) {
				row[k] = (fk - f0) / (fc - f0);
			} else if (fk >= fc && fk <= f1) {
				row[k] = (f1 - fk) / (f1 - fc);
			}
			// else stays 0.0f
		}
	}

	return bank;
}

// ---------------------------------------------------------------------------
// Hz <-> Mel conversion (HTK formula used by librosa / OpenAI Whisper)
// ---------------------------------------------------------------------------
namespace {
inline constexpr float kHtkMelScale      = 2595.0f;
inline constexpr float kHtkMelBreakHz    = 700.0f;
} // namespace

float MelSpectrogram::hzToMel(float hz) noexcept
{
	return kHtkMelScale * std::log10(1.0f + hz / kHtkMelBreakHz);
}

float MelSpectrogram::melToHz(float mel) noexcept
{
	return kHtkMelBreakHz * (std::pow(10.0f, mel / kHtkMelScale) - 1.0f);
}

