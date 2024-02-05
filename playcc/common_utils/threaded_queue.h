//
// Created by qzz on 2023/12/29.
//

#ifndef THREADED_QUEUE_H
#define THREADED_QUEUE_H
#include <queue>

#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"

// A threadsafe-queue.
template<class T>
class ThreadedQueue {
  public:
  explicit ThreadedQueue(int max_size) : max_size_(max_size) {}

  // Add an element to the queue.
  bool Push(const T& value) { return Push(value, absl::InfiniteDuration()); }
  bool Push(const T& value, absl::Duration wait) { return Push(value, absl::Now() + wait); }
  bool Push(const T& value, absl::Time deadline) {
    absl::MutexLock lock(&m_);
    if (block_new_values_) {
      return false;
    }
    while (q_.size() >= max_size_) {
      if (absl::Now() > deadline || block_new_values_) {
        return false;
      }
      cv_.WaitWithDeadline(&m_, deadline);
    }
    q_.push(value);
    cv_.Signal();
    return true;
  }

  absl::optional<T> Pop() { return Pop(absl::InfiniteDuration()); }
  absl::optional<T> Pop(absl::Duration wait) { return Pop(absl::Now() + wait); }
  absl::optional<T> Pop(absl::Time deadline) {
    absl::MutexLock lock(&m_);
    while (q_.empty()) {
      if (absl::Now() > deadline || block_new_values_) {
        return absl::nullopt;
      }
      cv_.WaitWithDeadline(&m_, deadline);
    }
    T val = q_.front();
    q_.pop();
    cv_.Signal();
    return val;
  }

  bool Empty() {
    absl::MutexLock lock(&m_);
    return q_.empty();
  }

  void Clear() {
    absl::MutexLock lock(&m_);
    while (!q_.empty()) {
      q_.pop();
    }
  }

  int Size() {
    absl::MutexLock lock(&m_);
    return static_cast<int>(q_.size());
  }

  // Causes pushing new values to fail. Useful for shutting down the queue.
  void BlockNewValues() {
    absl::MutexLock lock(&m_);
    block_new_values_ = true;
    cv_.SignalAll();
  }

  private:
  bool block_new_values_ = false;
  int max_size_;
  std::queue<T> q_;
  absl::Mutex m_;
  absl::CondVar cv_;
};

#endif // THREADED_QUEUE_H
