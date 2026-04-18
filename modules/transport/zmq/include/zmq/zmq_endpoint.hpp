#pragma once

#include <filesystem>
#include <format>
#include <string>

/// Build the canonical ZMQ IPC endpoint for a given port identifier.
/// All inter-module ZMQ communication uses Unix domain sockets (IPC)
/// to avoid TCP/IP stack overhead on localhost.
[[nodiscard]] inline std::string zmqEndpoint(int port)
{
    return std::format("ipc:///tmp/omniedge_{}", port);
}

/// Remove a stale IPC socket file left by a crashed process.
/// Must be called before bind() to prevent EADDRINUSE errors.
/// Safe to call even if the file does not exist.
inline void cleanStaleEndpoint(int port)
{
    std::filesystem::remove(std::format("/tmp/omniedge_{}", port));
}
