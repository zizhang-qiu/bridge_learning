//
// Created by qzz on 2023/9/19.
//

#ifndef BRIDGE_LEARNING_RELA_THREAD_LOOP_H_
#define BRIDGE_LEARNING_RELA_THREAD_LOOP_H_
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace rela{
class ThreadLoop {
 public:
  ThreadLoop() = default;

  ThreadLoop(const ThreadLoop&) = delete;
  ThreadLoop& operator=(const ThreadLoop&) = delete;

  virtual ~ThreadLoop() {
  }

  virtual void terminate() {
    terminated_ = true;
    if (paused()) {
      resume();
    }
  }

  virtual void pause() {
    std::lock_guard<std::mutex> lk(mPaused_);
    paused_ = true;
  }

  virtual void resume() {
    {
      std::lock_guard<std::mutex> lk(mPaused_);
      paused_ = false;
    }
    cvPaused_.notify_one();
  }

  virtual void waitUntilResume() {
    std::unique_lock<std::mutex> lk(mPaused_);
    cvPaused_.wait(lk, [this] { return !paused_; });
  }

  virtual bool terminated() {
    return terminated_;
  }

  virtual bool paused() {
    return paused_;
  }

  virtual void mainLoop() = 0;

 private:
  std::atomic_bool terminated_{false};

  std::mutex mPaused_;
  bool paused_ = false;
  std::condition_variable cvPaused_;
};
}
#endif //BRIDGE_LEARNING_RELA_THREAD_LOOP_H_
