#pragma once

#include <zmq.hpp>

/// Role of a socket in the OmniEdge ZMQ topology.
/// Determines whether send- or receive-side HWM is applied.
enum class ZmqSocketRole { Publisher, Subscriber };

/// Apply the project's standard socket options: HWM (role-appropriate)
/// and linger=0. Centralised so every socket drops pending frames at
/// shutdown the same way and no caller forgets one of the two flags.
///
/// This is intentionally header-only — trivially small and called from
/// ctors where we already depend on <zmq.hpp>.
inline void configureZmqSocket(zmq::socket_t& socket,
                                ZmqSocketRole  role,
                                int            hwm)
{
    if (role == ZmqSocketRole::Publisher) {
        socket.set(zmq::sockopt::sndhwm, hwm);
    } else {
        socket.set(zmq::sockopt::rcvhwm, hwm);
    }
    socket.set(zmq::sockopt::linger, 0);
}
