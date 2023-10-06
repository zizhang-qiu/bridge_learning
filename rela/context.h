//
// Created by qzz on 2023/9/19.
//

#ifndef BRIDGE_LEARNING_RELA_CONTEXT_H_
#define BRIDGE_LEARNING_RELA_CONTEXT_H_
#include <atomic>
#include <cassert>
#include <memory>
#include <thread>
#include <vector>
#include "thread_loop.h"
namespace rela {

class Context {
public:
  Context()
      : started_(false)
      , numTerminatedThread_(0) {
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ~Context();

  int pushThreadLoop(std::shared_ptr<ThreadLoop> env);

  void start();

  void pause();

  void resume();

  void join();

  bool terminated();

 private:
  bool started_;
  std::atomic<int> numTerminatedThread_;
  std::vector<std::shared_ptr<ThreadLoop>> loops_;
  std::vector<std::thread> threads_;
};

} // rela

#endif //BRIDGE_LEARNING_RELA_CONTEXT_H_
