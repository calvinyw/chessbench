#pragma once

#include "chess.hpp"

#include <tbb/concurrent_priority_queue.h>

#include <cstddef>
#include <utility>

namespace engine_parallel {

struct PositionTask {
    chess::Board board;
    float depth;
};

struct PositionDepthCompare {
    bool operator()(const PositionTask &lhs, const PositionTask &rhs) const noexcept {
        // Return true if lhs has lower priority than rhs
        // Using depth as the priority (max-depth first)
        return lhs.depth < rhs.depth;
    }
};

class ConcurrentPositionQueue {
public:
    using QueueType = tbb::concurrent_priority_queue<PositionTask, PositionDepthCompare>;

    ConcurrentPositionQueue() = default;
    explicit ConcurrentPositionQueue(std::size_t initial_capacity)
        : queue_(initial_capacity) {}

    void push(chess::Board board, float depth) {
        queue_.push(PositionTask{std::move(board), depth});
    }

    void push(PositionTask task) {
        queue_.push(std::move(task));
    }

    template <typename... Args>
    void emplace(Args&&... args) {
        queue_.emplace(std::forward<Args>(args)...);
    }

    bool try_pop(PositionTask &out) {
        return queue_.try_pop(out);
    }

    bool empty() const { return queue_.empty(); }
    std::size_t size() const { return queue_.size(); }

    // Not thread-safe w.r.t. other operations, per TBB docs
    void clear() { queue_.clear(); }

private:
    QueueType queue_;
};

} // namespace engine_parallel


