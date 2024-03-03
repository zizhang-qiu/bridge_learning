#ifndef PLAYCC_COMMON_UTILS_STOPWATCH_H
#define PLAYCC_COMMON_UTILS_STOPWATCH_H
#include <chrono>
class Stopwatch {
 public:
  void Start() {
    start_time_ = std::chrono::steady_clock::now();
    running_ = true;
  }

  void Stop() {
    if (running_) {
      end_time_ = std::chrono::steady_clock::now();
      running_ = false;
    }
  }

  void Reset() {
    start_time_ = end_time_ = std::chrono::steady_clock::now();
    running_ = false;
  }

  std::chrono::milliseconds Elapsed() const {
    auto end = running_ ? std::chrono::steady_clock::now() : end_time_;
    return std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                 start_time_);
  }

  bool IsRunning() const { return running_; }

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  std::chrono::time_point<std::chrono::steady_clock> end_time_;
  bool running_ = false;
};

#endif /* PLAYCC_COMMON_UTILS_STOPWATCH_H */
