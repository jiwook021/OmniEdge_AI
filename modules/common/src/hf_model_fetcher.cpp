#include "common/hf_model_fetcher.hpp"

#include <cstdlib>
#include <format>
#include <system_error>

#include "common/oe_logger.hpp"
#include "common/subprocess_manager.hpp"


namespace {

/// Resolve the path to scripts/helpers/hf_fetch.py.
///
/// Preference order:
///   1. OE_HF_FETCH_SCRIPT environment variable (absolute path).
///   2. /opt/omniedge/share/omniedge/hf_fetch.py   (install layout)
///   3. ${OE_SOURCE_ROOT}/scripts/helpers/hf_fetch.py (dev layout)
///
/// The script itself is <80 lines of Python; we only locate it.
[[nodiscard]] std::filesystem::path resolveFetchScript()
{
	if (const char* override_ = std::getenv("OE_HF_FETCH_SCRIPT"); override_ && *override_) {
		return std::filesystem::path(override_);
	}

	const std::filesystem::path installPath =
		"/opt/omniedge/share/omniedge/hf_fetch.py";
	std::error_code ec;
	if (std::filesystem::exists(installPath, ec)) {
		return installPath;
	}

	if (const char* sourceRoot = std::getenv("OE_SOURCE_ROOT"); sourceRoot && *sourceRoot) {
		return std::filesystem::path(sourceRoot) / "scripts" / "helpers" / "hf_fetch.py";
	}

	// Fall back to a hard-coded dev path — the caller will see a clear error
	// from hf_hub_download if this does not exist.
	return "/home/jiwook/jarvisAI/scripts/helpers/hf_fetch.py";
}

} // namespace


tl::expected<std::filesystem::path, std::string>
fetchHfModel(const std::string&             repo,
             const std::string&             file,
             const std::filesystem::path&   cacheDir,
             std::chrono::seconds           timeout)
{
	const std::filesystem::path target = cacheDir / file;

	// Fast path: the file is already on disk.
	std::error_code ec;
	if (std::filesystem::is_regular_file(target, ec)) {
		return target;
	}

	const std::filesystem::path script = resolveFetchScript();
	if (!std::filesystem::exists(script, ec)) {
		return tl::unexpected(
			std::format("hf_fetch.py not found at '{}'", script.string()));
	}

	const std::vector<std::string> args{
		script.string(),
		"--repo", repo,
		"--file", file,
		"--cache-dir", cacheDir.string(),
	};

	OE_LOG_INFO("hf_fetch_start: repo={}, file={}, cache={}",
	            repo, file, cacheDir.string());

	auto result = SubprocessManager::runOnce("python3", args, timeout);
	if (!result) {
		return tl::unexpected(std::format(
			"hf_fetch.py failed (repo={}, file={}): {}",
			repo, file, result.error()));
	}

	if (!std::filesystem::is_regular_file(target, ec)) {
		return tl::unexpected(std::format(
			"hf_fetch.py reported success but '{}' does not exist",
			target.string()));
	}

	OE_LOG_INFO("hf_fetch_done: file={}", target.string());
	return target;
}
