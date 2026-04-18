#pragma once

#include <chrono>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>
#include <zmq.hpp>


/// Send a topic-prefixed ZMQ message on a PUB socket.
inline void publishTestMessage(zmq::socket_t& publisherSocket,
                               std::string_view topic,
                               const nlohmann::json& payload)
{
	const std::string envelope = std::string(topic) + " " + payload.dump();
	publisherSocket.send(zmq::buffer(envelope), zmq::send_flags::none);
}

/// Receive the next ZMQ frame from a SUB socket, parse JSON after stripping topic prefix.
/// Returns empty JSON if no message arrives within timeoutMs.
[[nodiscard]] inline nlohmann::json receiveTestMessage(
	zmq::socket_t& subscriberSocket,
	int timeoutMs = 500)
{
	zmq::pollitem_t pollItem{static_cast<void*>(subscriberSocket), 0, ZMQ_POLLIN, 0};
	zmq::poll(&pollItem, 1, std::chrono::milliseconds{timeoutMs});
	if (!(pollItem.revents & ZMQ_POLLIN)) {
		return {};
	}
	zmq::message_t receivedFrame;
	(void)subscriberSocket.recv(receivedFrame);
	std::string rawMessage(static_cast<const char*>(receivedFrame.data()),
	                       receivedFrame.size());
	const auto separatorPos = rawMessage.find(' ');
	if (separatorPos != std::string::npos) {
		rawMessage = rawMessage.substr(separatorPos + 1);
	}
	return nlohmann::json::parse(rawMessage, nullptr, false);
}

