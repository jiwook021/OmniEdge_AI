#pragma once

#include <functional>
#include <string>
#include <string_view>

// ---------------------------------------------------------------------------
// OmniEdge_AI — Transparent string hash for heterogeneous lookup
//
// Enables unordered_map<std::string, V, StringHash, std::equal_to<>>
// to accept std::string_view keys without constructing a temporary
// std::string on every lookup.
// ---------------------------------------------------------------------------


struct StringHash {
	using is_transparent = void;

	[[nodiscard]] std::size_t operator()(std::string_view sv) const noexcept {
		return std::hash<std::string_view>{}(sv);
	}

	[[nodiscard]] std::size_t operator()(const std::string& s) const noexcept {
		return std::hash<std::string_view>{}(s);
	}
};

