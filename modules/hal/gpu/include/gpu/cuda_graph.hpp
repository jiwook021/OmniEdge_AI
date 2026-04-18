#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — CUDA Graph RAII Wrapper
//
// Captures a sequence of GPU operations (kernel launches, memcpys, etc.)
// into a graph, then replays the graph with a single driver call.  This
// eliminates per-kernel launch overhead for repetitive fixed-shape workloads.
//
// Usage:
//   CudaGraphInstance graph;
//
//   // First frame: capture the kernel sequence
//   if (!graph.captured()) {
//       graph.beginCapture(stream);
//       // ... launch kernels, memcpys, etc. on `stream` ...
//       graph.endCapture(stream);
//   }
//
//   // Subsequent frames: update input data, then replay
//   graph.replay(stream);
//
// Thread safety: none.  One instance per inference pipeline per thread.
//
// Invalidation: call reset() when the pipeline changes (model reload,
// shape change, etc.) to force re-capture on the next frame.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "gpu/oe_cuda_check.hpp"


class CudaGraphInstance {
public:
	CudaGraphInstance() = default;

	~CudaGraphInstance() { reset(); }

	CudaGraphInstance(const CudaGraphInstance&)            = delete;
	CudaGraphInstance& operator=(const CudaGraphInstance&) = delete;

	CudaGraphInstance(CudaGraphInstance&& o) noexcept
		: graph_(o.graph_), graphExec_(o.graphExec_), captured_(o.captured_)
	{
		o.graph_     = nullptr;
		o.graphExec_ = nullptr;
		o.captured_  = false;
	}

	CudaGraphInstance& operator=(CudaGraphInstance&& o) noexcept
	{
		if (this != &o) {
			reset();
			graph_     = o.graph_;
			graphExec_ = o.graphExec_;
			captured_  = o.captured_;
			o.graph_     = nullptr;
			o.graphExec_ = nullptr;
			o.captured_  = false;
		}
		return *this;
	}

	/// True after a successful endCapture().
	[[nodiscard]] bool captured() const noexcept { return captured_; }

	/// Begin capturing GPU operations on the given stream.
	/// All subsequent GPU calls on this stream will be recorded into the graph.
	void beginCapture(cudaStream_t stream)
	{
		reset();
		OE_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
	}

	/// End capture and instantiate the graph for replay.
	void endCapture(cudaStream_t stream)
	{
		OE_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
		OE_CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, 0));
		captured_ = true;
	}

	/// Replay the captured graph on the given stream.
	/// Input data must already be in the same device buffers used during capture.
	void replay(cudaStream_t stream)
	{
		OE_CUDA_CHECK(cudaGraphLaunch(graphExec_, stream));
	}

	/// Release graph resources and reset to uncaptured state.
	/// Call this when the pipeline changes (model reload, shape change).
	void reset() noexcept
	{
		if (graphExec_) {
			cudaGraphExecDestroy(graphExec_);
			graphExec_ = nullptr;
		}
		if (graph_) {
			cudaGraphDestroy(graph_);
			graph_ = nullptr;
		}
		captured_ = false;
	}

private:
	cudaGraph_t     graph_     = nullptr;
	cudaGraphExec_t graphExec_ = nullptr;
	bool            captured_  = false;
};
